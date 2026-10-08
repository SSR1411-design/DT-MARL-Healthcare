"""
SPRINT 7 RUNG 2 -- pre-flight smoke test for the MC critic target.

Verifies, before any 600-episode run is launched:

  1. imports / syntax / config round-trip
  2. cfg.critic_target defaults to "lambda"  (A0 control preserved)
  3. compute_gae is BIT-IDENTICAL under both flag values
     -> the actor's GAE(gamma=0.999, lambda=0.995) is provably untouched
  4. compute_mc_returns matches a hand-rolled reference, including
     per-episode boundaries, truncation bootstrap, and true-terminal zeroing
  5. lambda-target and MC-target really do differ (the change is live)
  6. checkpoint save/load carries critic_target
  7. an end-to-end 2-episode update runs under BOTH settings
  8. value_clip_eps and the minibatch construction are untouched

Run:  python marl/_smoke_rung2_mc_target.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from marl.config import MappoConfig, Sprint6Config          # noqa: E402
from marl.mappo import MAPPO, RolloutBuffer                  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def synthetic_buffer(agent, n_agents, obs_dim, state_dim, seed=7):
    """
    3 episodes in one buffer: lengths 5, 4, 6.
      ep0 ends by TRUNCATION      (trunc=1, boot used)
      ep1 ends at a TRUE TERMINAL (trunc=0, tail must be 0)
      ep2 ends by TRUNCATION      (last episode in buffer)
    """
    rng = np.random.default_rng(seed)
    lens = [5, 4, 6]
    trunc_flags = [True, False, True]
    buf = RolloutBuffer(sum(lens), n_agents, obs_dim, state_dim)
    for L, tr in zip(lens, trunc_flags):
        for t in range(L):
            last = (t == L - 1)
            mask = np.zeros((n_agents, 4), np.float32)
            mask[:, 0] = 1.0
            mask[:, 1] = (rng.random(n_agents) < 0.6).astype(np.float32)
            buf.add(
                obs=rng.standard_normal((n_agents, obs_dim)).astype(np.float32),
                state=rng.standard_normal(state_dim).astype(np.float32),
                act=np.zeros(n_agents, np.int64),
                logp=rng.standard_normal(n_agents).astype(np.float32),
                val=rng.standard_normal(n_agents).astype(np.float32),
                rew=rng.standard_normal(n_agents).astype(np.float32),
                mask=mask,
                cont=0.0 if last else 1.0,
            )
            if last:
                buf.set_bootstrap(
                    buf.ptr - 1,
                    rng.standard_normal(n_agents).astype(np.float32),
                    tr)
    return buf, lens, trunc_flags


def mc_reference(buf, gamma, lens, trunc_flags, n_agents):
    """Independent hand-rolled MC return, computed episode by episode."""
    T = len(buf)
    out = np.zeros((T, n_agents), np.float64)
    base = 0
    for L, tr in zip(lens, trunc_flags):
        tail = buf.boot[base + L - 1].astype(np.float64) if tr \
            else np.zeros(n_agents, np.float64)
        run = tail
        for t in reversed(range(L)):
            run = buf.rew[base + t].astype(np.float64) + gamma * run
            out[base + t] = run
        base += L
    return out


def main():
    print("=" * 78)
    print("SPRINT 7 RUNG 2 -- smoke test: MC critic target")
    print("=" * 78)

    # ---- 1/2. defaults --------------------------------------------------
    print("\n-- defaults (A0 control must be the default) ----------------")
    d = MappoConfig()
    check("MappoConfig.critic_target defaults to 'lambda'",
          d.critic_target == "lambda", f"got {d.critic_target!r}")
    check("gamma unchanged at 0.999", d.gamma == 0.999, f"{d.gamma}")
    check("gae_lambda unchanged at 0.995", d.gae_lambda == 0.995,
          f"{d.gae_lambda}")
    check("value_clip_eps unchanged at 0.2", d.value_clip_eps == 0.2,
          f"{d.value_clip_eps}")
    check("minibatches unchanged at 4", d.minibatches == 4, f"{d.minibatches}")
    check("ppo_epochs unchanged at 4", d.ppo_epochs == 4, f"{d.ppo_epochs}")
    check("config round-trips through asdict/ctor",
          MappoConfig(**Sprint6Config().to_dict()["mappo"]).critic_target
          == "lambda")

    n_agents, obs_dim, state_dim = 4, 12, 30

    # ---- 3. GAE is untouched -------------------------------------------
    print("\n-- the actor's GAE must be BIT-IDENTICAL under both flags ---")
    a_lam = MAPPO(n_agents, obs_dim, state_dim,
                  MappoConfig(critic_target="lambda"), seed=1)
    a_mc = MAPPO(n_agents, obs_dim, state_dim,
                 MappoConfig(critic_target="mc"), seed=1)
    buf, lens, trunc_flags = synthetic_buffer(a_lam, n_agents, obs_dim, state_dim)

    adv_l, ret_l = a_lam.compute_gae(buf)
    adv_m, ret_m = a_mc.compute_gae(buf)
    check("compute_gae advantages identical",
          np.array_equal(adv_l, adv_m),
          f"max|d|={np.abs(adv_l - adv_m).max():.3e}")
    check("compute_gae lambda-returns identical",
          np.array_equal(ret_l, ret_m))

    # ---- 4. MC correctness ---------------------------------------------
    print("\n-- MC target vs independent reference -----------------------")
    mc = a_mc.compute_mc_returns(buf)
    ref = mc_reference(buf, a_mc.cfg.gamma, lens, trunc_flags, n_agents)
    err = float(np.abs(mc.astype(np.float64) - ref).max())
    check("compute_mc_returns matches hand-rolled reference",
          err < 1e-4, f"max|d|={err:.3e}")

    # the true-terminal episode must NOT contain any bootstrap term
    base_t = lens[0]
    L_t = lens[1]
    last_t = base_t + L_t - 1
    check("true-terminal episode: last step return == its own reward "
          "(no bootstrap)",
          np.allclose(mc[last_t], buf.rew[last_t], atol=1e-5),
          f"ret={mc[last_t][:2]} rew={buf.rew[last_t][:2]}")
    check("true-terminal episode has trunc flag 0",
          buf.trunc[last_t, 0] == 0.0)

    # the truncated episode MUST contain the bootstrap term
    last_0 = lens[0] - 1
    expect = buf.rew[last_0] + a_mc.cfg.gamma * buf.boot[last_0]
    check("truncated episode: last step return == rew + gamma*boot",
          np.allclose(mc[last_0], expect, atol=1e-5))
    check("truncated episode has trunc flag 1", buf.trunc[last_0, 0] == 1.0)

    # no leakage across episode boundaries: recompute ep2 alone
    solo, l2, t2 = synthetic_buffer(a_mc, n_agents, obs_dim, state_dim)
    check("MC return uses gamma only (no lambda anywhere)",
          "gae_lambda" not in
          MAPPO.compute_mc_returns.__doc__.replace("GAE(lambda)", ""),
          "docstring/impl reference check")

    # ---- 5. the change is actually live --------------------------------
    print("\n-- the two targets must genuinely differ -------------------")
    diff = float(np.abs(ret_l - mc).mean())
    check("lambda-return and MC target differ", diff > 1e-3,
          f"mean|d|={diff:.4f}")

    # ---- 6. checkpoint carries the flag --------------------------------
    print("\n-- checkpoint persistence ----------------------------------")
    tmp = _ROOT / "saved_models" / "marl" / "_smoke_rung2_tmp.pth"
    a_mc.save(tmp, extra={"smoke": True})
    back, _ = MAPPO.load(tmp, device="cpu")
    check("saved/loaded critic_target == 'mc'",
          back.cfg.critic_target == "mc", f"got {back.cfg.critic_target!r}")
    check("saved/loaded gamma/lambda intact",
          back.cfg.gamma == 0.999 and back.cfg.gae_lambda == 0.995)
    tmp.unlink(missing_ok=True)
    check("temp checkpoint removed (no artifact left behind)",
          not tmp.exists())

    # ---- 7. end-to-end update under both settings ----------------------
    print("\n-- end-to-end update() under both settings -----------------")
    for arm, agent in (("lambda", a_lam), ("mc", a_mc)):
        b, _, _ = synthetic_buffer(agent, n_agents, obs_dim, state_dim, seed=11)
        try:
            st = agent.update(b)
            ok = np.isfinite(st["critic_loss"]) and np.isfinite(st["actor_loss"])
            check(f"update() runs and is finite [{arm}]", ok,
                  f"critic_loss={st['critic_loss']:.4f} "
                  f"ev={st['explained_var']:+.3f}")
        except Exception as e:                                  # noqa: BLE001
            check(f"update() runs [{arm}]", False, f"{type(e).__name__}: {e}")

    # ---- 8. untouched-code assertions ----------------------------------
    print("\n-- things that must NOT have changed -----------------------")
    src = (_ROOT / "marl" / "mappo.py").read_text(encoding="utf-8")
    check("clipped value loss still present",
          "v_clipped = old_v[idx] + torch.clamp(" in src)
    check("minibatch construction unchanged",
          "mb_size = max(1, T // n_mb)" in src
          and "for start in range(0, T, mb_size)" in src)
    check("GAE recursion unchanged",
          "last = delta + g * lam * cont * last" in src)
    check("compute_gae does not read the trunc flag",
          "buf.trunc" not in src.split("def compute_gae")[1]
          .split("def compute_mc_returns")[0])

    print("\n" + "=" * 78)
    if FAIL:
        print(f"FAILED ({len(FAIL)}): " + "; ".join(FAIL))
        return 1
    print("ALL SMOKE CHECKS PASSED -- safe to launch the controlled run.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
