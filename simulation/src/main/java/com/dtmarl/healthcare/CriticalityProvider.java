package com.dtmarl.healthcare;

/**
 * Strategy seam for computing the clinical criticality of a
 * {@link HealthcareTask} from its owning {@link Patient}.
 *
 * <p>This interface is the single extension point that keeps Sprint 5's
 * deterministic placeholder scoring separate from the real, model-driven
 * scoring that later sprints will introduce. Sprint 5 ships one
 * implementation — {@link CriticalityManager} — that uses a transparent,
 * reproducible formula over patient attributes. When HTCF, a Health
 * Severity Index, or any learned model becomes available, a new
 * implementation can be dropped in with no change to {@code HealthcareTask},
 * the broker, or the scheduler.</p>
 *
 * <p><b>Contract:</b> implementations must be pure and side-effect-free with
 * respect to their inputs — they read the patient and task and return a
 * fresh {@link CriticalityResult}; they must never mutate either argument.
 * Caching the result onto the task is the caller's responsibility.</p>
 */
public interface CriticalityProvider {

    /**
     * Evaluates the criticality of a task for a given patient.
     *
     * @param patient the owning patient (authoritative clinical state)
     * @param task    the task being scored
     * @param atTime  simulation time (seconds) of the evaluation, recorded
     *                as provenance on the returned result
     * @return a fresh, immutable {@link CriticalityResult}; never {@code null}
     */
    CriticalityResult evaluate(Patient patient, HealthcareTask task, double atTime);
}
