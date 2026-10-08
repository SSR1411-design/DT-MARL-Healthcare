package com.dtmarl.ai.migration;

/**
 * Immutable audit record of one migration request.
 *
 * <p>Carries exactly the fields Sprint 6 requires to be recorded for every
 * migration: source, destination, task, task criticality, migration cost,
 * whether the migration was preemptive, and whether it happened before the
 * source failed.</p>
 *
 * <h2>Two fields are evaluation-only and must never be fed back</h2>
 *
 * <p>{@code sourceFailedAfterwards} and {@code secondsUntilSourceFailure} can
 * only be known <em>after</em> the source host's fate is observed. They are
 * filled in by {@link MigrationLog#finaliseAgainstFailures} once the run is
 * over, purely so the "tasks protected before failure" metric can be computed.
 * Reading them at decision time would be exactly the look-ahead the project
 * forbids, which is why they are not settable through the constructor used at
 * decision time and why the class carries this warning.</p>
 */
public final class MigrationRecord {

    private final double time;
    private final int taskId;
    private final int patientId;
    private final double clinicalSeverity;
    private final int sourceNodeId;
    private final int destNodeId;
    private final double migrationCost;
    private final boolean preemptive;
    private final String reason;
    private final MigrationOutcome outcome;
    private final double sourceRiskAtDecision;
    private final double destRiskAtDecision;
    private final int collateralTasksMoved;

    /** EVALUATION ONLY - see class comment. */
    private boolean sourceFailedAfterwards;
    /** EVALUATION ONLY - see class comment. NaN when the source never failed. */
    private double secondsUntilSourceFailure = Double.NaN;

    /**
     * Creates a decision-time record. Every argument is knowable at
     * {@code time}; nothing here reads the future.
     *
     * @param time                 simulation time of the request (s)
     * @param taskId               healthcare task id
     * @param patientId            patient the task belongs to
     * @param clinicalSeverity     Sprint 5 clinical severity of the task
     * @param sourceNodeId         host index the task was on
     * @param destNodeId           host index requested
     * @param migrationCost        relative cost charged for this move
     * @param preemptive           true when triggered by predicted risk rather
     *                             than by an observed symptom or actual failure
     * @param reason               short human-readable trigger description
     * @param outcome              what the manager actually did
     * @param sourceRiskAtDecision predicted_failure_risk of the source at
     *                             {@code time}
     * @param destRiskAtDecision   predicted_failure_risk of the destination at
     *                             {@code time}
     * @param collateralTasksMoved other tasks that moved because they shared
     *                             the migrated VM (see MigrationManager)
     */
    public MigrationRecord(double time,
                           int taskId,
                           int patientId,
                           double clinicalSeverity,
                           int sourceNodeId,
                           int destNodeId,
                           double migrationCost,
                           boolean preemptive,
                           String reason,
                           MigrationOutcome outcome,
                           double sourceRiskAtDecision,
                           double destRiskAtDecision,
                           int collateralTasksMoved) {
        this.time = time;
        this.taskId = taskId;
        this.patientId = patientId;
        this.clinicalSeverity = clinicalSeverity;
        this.sourceNodeId = sourceNodeId;
        this.destNodeId = destNodeId;
        this.migrationCost = migrationCost;
        this.preemptive = preemptive;
        this.reason = reason;
        this.outcome = outcome;
        this.sourceRiskAtDecision = sourceRiskAtDecision;
        this.destRiskAtDecision = destRiskAtDecision;
        this.collateralTasksMoved = collateralTasksMoved;
    }

    /** @return simulation time of the request (s) */
    public double getTime() {
        return time;
    }

    /** @return healthcare task id */
    public int getTaskId() {
        return taskId;
    }

    /** @return patient the task belongs to */
    public int getPatientId() {
        return patientId;
    }

    /** @return Sprint 5 clinical severity of the task */
    public double getClinicalSeverity() {
        return clinicalSeverity;
    }

    /** @return host index the task was on */
    public int getSourceNodeId() {
        return sourceNodeId;
    }

    /** @return host index requested */
    public int getDestNodeId() {
        return destNodeId;
    }

    /** @return relative cost charged for this move */
    public double getMigrationCost() {
        return migrationCost;
    }

    /** @return true when triggered by predicted risk, not by a symptom */
    public boolean isPreemptive() {
        return preemptive;
    }

    /** @return short human-readable trigger description */
    public String getReason() {
        return reason;
    }

    /** @return what the manager actually did */
    public MigrationOutcome getOutcome() {
        return outcome;
    }

    /** @return predicted_failure_risk of the source at decision time */
    public double getSourceRiskAtDecision() {
        return sourceRiskAtDecision;
    }

    /** @return predicted_failure_risk of the destination at decision time */
    public double getDestRiskAtDecision() {
        return destRiskAtDecision;
    }

    /** @return tasks that moved as collateral because they shared the VM */
    public int getCollateralTasksMoved() {
        return collateralTasksMoved;
    }

    /**
     * EVALUATION ONLY. Whether the source host later failed within the
     * protection window.
     *
     * @return true when this migration is credited with protecting the task
     */
    public boolean isSourceFailedAfterwards() {
        return sourceFailedAfterwards;
    }

    /**
     * EVALUATION ONLY. Called from {@link MigrationLog#finaliseAgainstFailures}
     * after the run, never during it.
     *
     * @param failed  whether the source failed within the window
     * @param seconds lead time from migration to that failure
     */
    void setSourceFailure(boolean failed, double seconds) {
        this.sourceFailedAfterwards = failed;
        this.secondsUntilSourceFailure = seconds;
    }

    /** @return lead time in seconds, or NaN when the source never failed */
    public double getSecondsUntilSourceFailure() {
        return secondsUntilSourceFailure;
    }

    /** @return CSV header matching {@link #toCsvRow()} */
    public static String csvHeader() {
        return "time,taskId,patientId,clinicalSeverity,sourceNodeId,destNodeId,"
             + "migrationCost,preemptive,outcome,applied,sourceRisk,destRisk,"
             + "collateralTasksMoved,reason,sourceFailedAfterwards,"
             + "secondsUntilSourceFailure";
    }

    /** @return one CSV row for this record */
    public String toCsvRow() {
        return String.format(
                "%.3f,%d,%d,%.4f,%d,%d,%.4f,%d,%s,%d,%.6f,%.6f,%d,%s,%d,%s",
                time, taskId, patientId, clinicalSeverity,
                sourceNodeId, destNodeId, migrationCost,
                preemptive ? 1 : 0, outcome, outcome.isApplied() ? 1 : 0,
                sourceRiskAtDecision, destRiskAtDecision, collateralTasksMoved,
                reason.replace(',', ';'),
                sourceFailedAfterwards ? 1 : 0,
                Double.isNaN(secondsUntilSourceFailure)
                        ? "" : String.format("%.3f", secondsUntilSourceFailure)
        );
    }

    @Override
    public String toString() {
        return String.format(
                "Migration{t=%.1f task=%d sev=%.2f n%d->n%d %s preemptive=%s "
                + "srcRisk=%.3f dstRisk=%.3f}",
                time, taskId, clinicalSeverity, sourceNodeId, destNodeId,
                outcome, preemptive, sourceRiskAtDecision, destRiskAtDecision
        );
    }
}
