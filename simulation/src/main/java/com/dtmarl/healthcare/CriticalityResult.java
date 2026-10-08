package com.dtmarl.healthcare;

/**
 * Immutable snapshot of a criticality evaluation for one
 * {@link HealthcareTask}.
 *
 * <p>A {@code CriticalityResult} pairs the two numbers scheduling cares
 * about — {@code clinicalSeverity} (how sick the patient is) and
 * {@code priorityScore} (how urgently the task should run) — with
 * provenance: which {@link CriticalityProvider} produced them and at what
 * simulation time. Provenance matters because later sprints will swap the
 * placeholder scorer for HTCF/HSI-driven scoring, and every cached result
 * must be traceable to the model that produced it.</p>
 *
 * <p>The object is immutable so it can be cached on a task, mirrored onto a
 * {@link com.dtmarl.ai.digitaltwin.TaskTwin}, and logged without any risk
 * of a stale or concurrently-mutated value.</p>
 */
public final class CriticalityResult {

    private final double clinicalSeverity;
    private final double priorityScore;
    private final String source;
    private final double timestamp;

    /**
     * Creates a criticality result.
     *
     * @param clinicalSeverity computed clinical severity (higher = sicker)
     * @param priorityScore    computed scheduling priority (higher = sooner)
     * @param source           identifier of the provider that produced this
     *                         result (e.g. {@code "CriticalityManager/v1"})
     * @param timestamp        simulation time (seconds) the result was computed
     */
    public CriticalityResult(double clinicalSeverity,
                             double priorityScore,
                             String source,
                             double timestamp) {
        this.clinicalSeverity = clinicalSeverity;
        this.priorityScore = priorityScore;
        this.source = source;
        this.timestamp = timestamp;
    }

    /** @return computed clinical severity (higher = sicker) */
    public double getClinicalSeverity() {
        return clinicalSeverity;
    }

    /** @return computed scheduling priority (higher = scheduled sooner) */
    public double getPriorityScore() {
        return priorityScore;
    }

    /** @return identifier of the provider that produced this result */
    public String getSource() {
        return source;
    }

    /** @return simulation time (seconds) the result was computed */
    public double getTimestamp() {
        return timestamp;
    }

    @Override
    public String toString() {
        return String.format(
                "CriticalityResult{severity=%.3f, priority=%.3f, source=%s, t=%.2f}",
                clinicalSeverity, priorityScore, source, timestamp
        );
    }
}
