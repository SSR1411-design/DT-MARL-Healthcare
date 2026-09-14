package com.dtmarl.failure;

import com.dtmarl.ai.digitaltwin.DigitalTwinManager;
import com.dtmarl.ai.digitaltwin.EdgeNode;
import com.dtmarl.ai.digitaltwin.NetworkLink;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

/**
 * Progressive host degradation: the piece that was missing from the failure
 * simulation.
 *
 * ------------------------------------------------------------------------
 * WHY THIS EXISTS
 * ------------------------------------------------------------------------
 * The original {@link FailureManager} killed a host either at a
 * pre-scheduled instant or on a memoryless Bernoulli draw. Both are
 * *constant-hazard* processes: nothing in the observable state evolves
 * towards the event, so P(fail soon | telemetry) never changes and the
 * "willFailSoon" label is information-theoretically unpredictable. The
 * exported dataset confirmed it - CPU/RAM/bandwidth/tasks were literally
 * constant for the whole run and the only usable column was the post-mortem
 * `linkUp` flag.
 *
 * This manager replaces the constant hazard with a WEAR PROCESS.
 *
 * ------------------------------------------------------------------------
 * MODEL
 * ------------------------------------------------------------------------
 * Each host carries a latent scalar wear in [0,1] plus a permanent
 * susceptibility drawn once at construction (the hardware lottery).
 *
 *   1. ONSET. While no mechanism is active, each tick draws against
 *      faultOnsetProbability * susceptibility. On onset the host picks a
 *      {@link HostFaultMode} and a severity; a small configurable minority
 *      of episodes are flagged "abrupt".
 *
 *   2. ACCUMULATION. Wear grows every tick by
 *        rate * susceptibility * (0.5 + load) * lognormalNoise
 *      where rate is the base rate when healthy and the (much larger)
 *      episode rate once a mechanism is active. Load couples wear to the
 *      actual CloudSim utilisation, so busy hosts really do age faster.
 *
 *   3. SYMPTOMS. Once wear passes the degrading threshold, an observable
 *      overlay is written onto the Digital Twin (see applyOverlay). The
 *      overlay magnitude is a function of the CURRENT wear only.
 *
 *   4. FAILURE. Each tick the host dies with probability
 *        hazardScale * wear^hazardShape
 *      (or unconditionally once wear reaches 1.0).
 *
 *   5. REPAIR. After a drawn number of ticks the host is reactivated with
 *      part of its wear retained, so failure/recovery cycles repeat and
 *      some hosts become repeat offenders.
 *
 * ------------------------------------------------------------------------
 * CAUSALITY (the whole point)
 * ------------------------------------------------------------------------
 * The failure time is NEVER scheduled and never computed in advance. It
 * emerges from a hazard draw against the wear that has already accumulated.
 * At tick t this class only ever reads: the host's current CloudSim state,
 * its own wear/mode/severity/elapsed-episode counters (all of which are
 * functions of ticks <= t), and fresh random noise. There is no future
 * timestamp in existence to leak, and no code path in this class can see
 * the label - labelling happens afterwards and elsewhere, in
 * HistoryCollector.exportLabeledCsv, from the event log.
 */
public class HostDegradationManager {

    /** Angular frequency of the power-supply oscillation (rad per tick). */
    private static final double OSCILLATION_OMEGA = 0.45;

    /** Absolute noise floor on packet loss, so a healthy link is not a constant. */
    private static final double PACKET_LOSS_NOISE_FLOOR = 0.12;

    /** Latency reported for a link whose host is down (existing convention). */
    private static final double DEAD_LINK_LATENCY_MS = 9999.0;

    private final DigitalTwinManager digitalTwin;

    private final FailureManager failureManager;

    private final HostDegradationConfig config;

    private final Random random;

    private final List<HostHealth> healthByNode = new ArrayList<>();

    /** Per-host latent state. Never exported as a predictive feature. */
    private static class HostHealth {

        double wear;

        HostFaultMode mode = HostFaultMode.NONE;

        double severity = 1.0;

        double susceptibility = 1.0;

        boolean abrupt;

        int episodeTicks;

        /** Ticks the symptoms have been observable (wear > degrading threshold). */
        int observableTicks;

        /** Simulation time at which the current episode's symptoms first showed. */
        double degradationStartTime = -1;

        int repairTicksRemaining;

        /** Phase offset so oscillating hosts are not synchronised. */
        double phase;

        int failureCount;

        EdgeNode.HealthState state = EdgeNode.HealthState.NORMAL;
    }

    public HostDegradationManager(DigitalTwinManager digitalTwin,
                                  FailureManager failureManager,
                                  HostDegradationConfig config,
                                  long seed) {

        this.digitalTwin = digitalTwin;
        this.failureManager = failureManager;
        this.config = config;
        this.random = new Random(seed);

        for (EdgeNode node : digitalTwin.getNodes()) {
            HostHealth h = new HostHealth();
            h.susceptibility = uniform(config.getSusceptibilityMin(),
                                       config.getSusceptibilityMax());
            h.phase = random.nextDouble() * Math.PI * 2;
            healthByNode.add(h);

            System.out.printf(
                    "Host %d degradation profile: susceptibility=%.2f%n",
                    node.getId(), h.susceptibility);
        }

        System.out.println("HostDegradationManager ready for "
                + healthByNode.size() + " host(s) (seed=" + seed + ")");
    }

    // ----------------------------------------------------------------------
    // Per-tick entry point. MUST run AFTER DigitalTwinManager.syncWithHosts()
    // so the overlay is applied on top of a fresh CloudSim baseline (the sync
    // overwrites the twin unconditionally and would otherwise erase it).
    // ----------------------------------------------------------------------
    public void checkAndTrigger(double currentTime) {

        for (EdgeNode node : digitalTwin.getNodes()) {

            int id = node.getId();

            if (id >= healthByNode.size()) {
                continue;
            }

            HostHealth h = healthByNode.get(id);
            NetworkLink link = digitalTwin.getLink(id);

            // ---- already down: hold the flat-line, count down the repair --
            if (failureManager.hasFailed(id)) {
                applyFailedState(node, link);
                tickRepair(h, node, link, currentTime);
                continue;
            }

            maybeStartEpisode(h, node, currentTime);
            accumulateWear(h, node);
            updateHealthState(h, node, currentTime);

            if (rollFailure(h)) {
                onWearFailure(h, node, link, currentTime);
                continue;
            }

            applyOverlay(h, node, link, currentTime);
        }
    }

    // ----------------------------------------------------------------------
    // 1. Fault onset
    // ----------------------------------------------------------------------
    private void maybeStartEpisode(HostHealth h, EdgeNode node, double currentTime) {

        if (h.mode != HostFaultMode.NONE) {
            return;
        }

        double p = config.getFaultOnsetProbabilityPerTick() * h.susceptibility;

        if (random.nextDouble() >= p) {
            return;
        }

        HostFaultMode[] mechanisms = HostFaultMode.mechanisms();

        h.mode = mechanisms[random.nextInt(mechanisms.length)];
        h.severity = uniform(config.getSeverityMin(), config.getSeverityMax());
        h.abrupt = random.nextDouble() < config.getAbruptFailureProbability();
        h.episodeTicks = 0;
        h.observableTicks = 0;
        h.degradationStartTime = -1;

        failureManager.logEvent(new FailureEvent(
                node.getId(), currentTime,
                FailureEvent.Type.HOST_FAULT_ONSET,
                FailureEvent.Cause.WEAR,
                node.getCpuUsage()));

        System.out.printf(
                "  [t=%.2f] Host %d developed %s (severity=%.2f%s)%n",
                currentTime, node.getId(), h.mode, h.severity,
                h.abrupt ? ", ABRUPT" : "");
    }

    // ----------------------------------------------------------------------
    // 2. Wear accumulation
    // ----------------------------------------------------------------------
    private void accumulateWear(HostHealth h, EdgeNode node) {

        // Baseline load straight from CloudSim (the overlay for this tick has
        // not been applied yet), so wear is driven by REAL utilisation.
        double load = clamp(node.getCpuUsage() / 100.0, 0.0, 1.0);

        double rate = (h.mode == HostFaultMode.NONE)
                ? config.getBaseWearPerTick()
                : config.getEpisodeWearPerTick() * h.severity;

        if (h.abrupt) {
            rate *= config.getAbruptWearMultiplier();
        }

        h.wear = Math.min(1.0,
                h.wear + rate * h.susceptibility * (0.5 + load) * lognormalNoise());

        if (h.mode != HostFaultMode.NONE) {
            h.episodeTicks++;
        }
    }

    // ----------------------------------------------------------------------
    // 3. Health-state lifecycle: NORMAL -> DEGRADING -> CRITICAL
    // ----------------------------------------------------------------------
    private void updateHealthState(HostHealth h, EdgeNode node, double currentTime) {

        EdgeNode.HealthState next;

        if (h.mode == HostFaultMode.NONE || h.wear < config.getDegradingWearThreshold()) {
            // A mechanism may be active but has not yet produced any
            // measurable effect - honestly still NORMAL from an operator's
            // point of view.
            next = EdgeNode.HealthState.NORMAL;
        } else if (h.wear < config.getCriticalWearThreshold()) {
            next = EdgeNode.HealthState.DEGRADING;
        } else {
            next = EdgeNode.HealthState.CRITICAL;
        }

        if (next != EdgeNode.HealthState.NORMAL) {
            h.observableTicks++;
        }

        if (next != h.state) {

            // First departure from NORMAL, whatever the target. An abrupt
            // episode can cross both thresholds inside a single tick and jump
            // NORMAL -> CRITICAL; without this it would report no onset time
            // at all and its lead time would be indistinguishable from
            // "never degraded".
            if (h.state == EdgeNode.HealthState.NORMAL
                    && next != EdgeNode.HealthState.NORMAL) {

                h.degradationStartTime = currentTime;

                failureManager.logEvent(new FailureEvent(
                        node.getId(), currentTime,
                        FailureEvent.Type.HOST_DEGRADATION_START,
                        FailureEvent.Cause.WEAR,
                        node.getCpuUsage()));
            }

            if (next == EdgeNode.HealthState.CRITICAL) {
                failureManager.logEvent(new FailureEvent(
                        node.getId(), currentTime,
                        FailureEvent.Type.HOST_CRITICAL,
                        FailureEvent.Cause.WEAR,
                        node.getCpuUsage()));
            }

            h.state = next;
        }

        node.setHealthState(next);

        // NOTE: `degraded` is deliberately NOT written here. It stays owned by
        // FailureManager.checkOverload, i.e. it remains an OBSERVABLE
        // CPU-threshold rule (what a real monitoring agent computes) rather
        // than a discretised view of the latent wear. Exporting a perfect
        // NORMAL/DEGRADING oracle as a model feature would be circular.
    }

    // ----------------------------------------------------------------------
    // 4. Failure hazard
    // ----------------------------------------------------------------------
    private boolean rollFailure(HostHealth h) {

        if (h.mode == HostFaultMode.NONE) {
            return false;
        }

        if (h.wear >= 1.0) {
            return true;
        }

        double hazard = config.getHazardScale()
                * Math.pow(h.wear, config.getHazardShape());

        return random.nextDouble() < hazard;
    }

    private void onWearFailure(HostHealth h, EdgeNode node, NetworkLink link,
                               double currentTime) {

        double leadTime = h.degradationStartTime < 0
                ? 0.0
                : currentTime - h.degradationStartTime;

        failureManager.triggerWearFailure(
                node.getId(), currentTime, h.mode, h.wear, h.severity, leadTime);

        h.failureCount++;
        h.state = EdgeNode.HealthState.FAILED;
        node.setHealthState(EdgeNode.HealthState.FAILED);

        h.repairTicksRemaining = config.isRecoveryEnabled()
                ? repairTicks()
                : Integer.MAX_VALUE;

        applyFailedState(node, link);
    }

    // ----------------------------------------------------------------------
    // 5. Repair
    // ----------------------------------------------------------------------
    private void tickRepair(HostHealth h, EdgeNode node, NetworkLink link,
                            double currentTime) {

        if (!config.isRecoveryEnabled()) {
            return;
        }

        if (h.repairTicksRemaining == Integer.MAX_VALUE) {
            return;
        }

        h.repairTicksRemaining--;

        if (h.repairTicksRemaining > 0) {
            h.state = EdgeNode.HealthState.RECOVERING;
            node.setHealthState(EdgeNode.HealthState.RECOVERING);
            return;
        }

        // Imperfect repair: some wear survives, so a repaired host is more
        // fragile than a fresh one.
        h.wear *= config.getImperfectRepairRetention();
        h.mode = HostFaultMode.NONE;
        h.severity = 1.0;
        h.abrupt = false;
        h.episodeTicks = 0;
        h.observableTicks = 0;
        h.degradationStartTime = -1;
        h.state = EdgeNode.HealthState.NORMAL;

        node.setHealthState(EdgeNode.HealthState.NORMAL);
        node.setDegraded(false);

        failureManager.recoverHost(node.getId(), currentTime, h.wear);

        if (link != null) {
            link.resetToNormal();
        }
    }

    // ----------------------------------------------------------------------
    // Observable overlay
    //
    // Recomputed from the freshly-synced CloudSim baseline every tick, so it
    // is idempotent: running twice in one tick cannot compound. Everything
    // here is a function of the CURRENT wear plus fresh noise.
    // ----------------------------------------------------------------------
    private void applyOverlay(HostHealth h, EdgeNode node, NetworkLink link,
                              double currentTime) {

        double response = response(h);
        double amplitude = amplitude(h);
        HostFaultMode mode = h.mode;

        double noise = config.getTelemetryNoisePercent();

        // ---- compute side ------------------------------------------------
        double baseEnergy = node.getEnergyConsumption();

        double cpu = node.getCpuUsage()
                + config.getCpuRisePercent() * mode.getCpuWeight() * response * amplitude;

        double ram = node.getRamUsage()
                + config.getRamRisePercent() * mode.getRamWeight() * response * amplitude;

        double bwUtil = node.getBandwidthUsage()
                + config.getBandwidthUtilRisePercent() * mode.getBandwidthWeight()
                  * response * amplitude;

        double energy = baseEnergy * (1.0
                + config.getEnergyRiseFraction() * mode.getEnergyWeight()
                  * response * amplitude);

        // Non-monotonic power signature for PSU faults: a rising ramp would
        // be indistinguishable from thermal creep, an oscillation is not.
        if (mode.getEnergyOscillation() > 0 && response > 0) {
            energy += baseEnergy * mode.getEnergyOscillation() * response
                    * Math.sin((currentTime + h.phase) * OSCILLATION_OMEGA);
        }

        int taskPressure = (int) Math.round(
                config.getMaxTaskPressure() * mode.getTaskWeight() * response);

        node.setCpuUsage(clamp(jitter(cpu, noise), 0.0, 100.0));
        node.setRamUsage(clamp(jitter(ram, noise), 0.0, 100.0));
        node.setBandwidthUsage(clamp(jitter(bwUtil, noise), 0.0, 100.0));
        node.setEnergyConsumption(Math.max(0.0, jitter(energy, noise)));
        node.setRunningTasks(Math.max(0, node.getRunningTasks() + taskPressure));

        // ---- network side ------------------------------------------------
        // Skipped while the link is under cyber-attack so the Sprint 3.5
        // attack signature written by NetworkFailureManager survives intact.
        if (link != null && link.isUp() && !link.isUnderAttack()) {

            double latency = NetworkLink.NORMAL_LATENCY_MS * (1.0
                    + (config.getLatencyRiseFactor() - 1.0) * mode.getLatencyWeight()
                      * response * amplitude);

            double loss = config.getMaxPacketLossPercent() * mode.getPacketLossWeight()
                    * Math.pow(response, 1.4) * amplitude;

            double linkBw = NetworkLink.NORMAL_BANDWIDTH_MBPS * (1.0
                    - config.getBandwidthFadeFraction() * mode.getBandwidthFadeWeight()
                      * response * amplitude);

            link.setLatencyMs(Math.max(0.5, jitter(latency, noise)));
            link.setPacketLossPercent(clamp(
                    jitter(loss, noise) + Math.abs(random.nextGaussian()) * PACKET_LOSS_NOISE_FLOOR,
                    0.0, 100.0));
            link.setBandwidthMbps(clamp(jitter(linkBw, noise),
                    0.0, NetworkLink.NORMAL_BANDWIDTH_MBPS));
        }
    }

    /** Flat-line telemetry for a host that is down. */
    private void applyFailedState(EdgeNode node, NetworkLink link) {

        node.setCpuUsage(0);
        node.setRamUsage(0);
        node.setBandwidthUsage(0);
        node.setEnergyConsumption(0);
        node.setRunningTasks(0);
        node.setActive(false);
        node.setDegraded(false);

        if (link != null) {
            link.setUp(false);
            link.setBandwidthMbps(0);
            link.setLatencyMs(DEAD_LINK_LATENCY_MS);
            link.setPacketLossPercent(100);
            link.setUnderAttack(false);
        }
    }

    // ----------------------------------------------------------------------
    // Shape helpers
    // ----------------------------------------------------------------------

    /**
     * Normalised degradation progress in [0,1], curved by the mechanism's
     * shape exponent. Zero below the degrading threshold, which gives every
     * episode a silent incubation period - "a fault has started" is NOT the
     * same instant as "the fault is visible".
     */
    private double response(HostHealth h) {

        if (h.mode == HostFaultMode.NONE) {
            return 0.0;
        }

        double onset = config.getDegradingWearThreshold();

        if (h.wear <= onset) {
            return 0.0;
        }

        double progress = (h.wear - onset) / (1.0 - onset);

        return Math.pow(Math.min(progress, 1.0), h.mode.getShapeExponent());
    }

    /** Severity-scaled symptom amplitude. */
    private double amplitude(HostHealth h) {
        return Math.min(1.5, 0.6 + 0.5 * h.severity);
    }

    // ----------------------------------------------------------------------
    // Randomness helpers (all seeded - see the constructor)
    // ----------------------------------------------------------------------

    /** Multiplicative measurement noise, pct = relative sigma in percent. */
    private double jitter(double value, double pct) {
        return value * (1.0 + (pct / 100.0) * random.nextGaussian());
    }

    /** Mean-1 lognormal multiplier for the wear increment. */
    private double lognormalNoise() {
        double s = config.getWearNoiseSigma();
        return Math.exp(s * random.nextGaussian() - 0.5 * s * s);
    }

    private double uniform(double min, double max) {
        return min + random.nextDouble() * (max - min);
    }

    private int repairTicks() {
        int lo = config.getRepairTicksMin();
        int hi = config.getRepairTicksMax();
        return lo + random.nextInt(Math.max(1, hi - lo + 1));
    }

    private static double clamp(double v, double lo, double hi) {
        return Math.max(lo, Math.min(hi, v));
    }

    // ----------------------------------------------------------------------
    // Audit accessors. Used ONLY to write the audit_* block of the exported
    // CSV and to print the end-of-run summary. Never fed to the predictor.
    // ----------------------------------------------------------------------

    public double getWear(int nodeId) {
        return nodeId < healthByNode.size() ? healthByNode.get(nodeId).wear : 0.0;
    }

    public HostFaultMode getFaultMode(int nodeId) {
        return nodeId < healthByNode.size()
                ? healthByNode.get(nodeId).mode
                : HostFaultMode.NONE;
    }

    public int getFailureCount(int nodeId) {
        return nodeId < healthByNode.size() ? healthByNode.get(nodeId).failureCount : 0;
    }

    public void printSummary() {

        System.out.println("\n--- Host degradation summary (latent ground truth) ---");

        for (EdgeNode node : digitalTwin.getNodes()) {

            int id = node.getId();

            if (id >= healthByNode.size()) {
                continue;
            }

            HostHealth h = healthByNode.get(id);

            System.out.printf(
                    "Host %2d | failures=%d | susceptibility=%.2f | final wear=%.3f "
                    + "| final state=%s | active mechanism=%s%n",
                    id, h.failureCount, h.susceptibility, h.wear, h.state, h.mode);
        }
    }
}
