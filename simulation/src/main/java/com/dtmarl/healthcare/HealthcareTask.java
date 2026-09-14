package com.dtmarl.healthcare;

import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModel;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * A CloudSim cloudlet enriched with healthcare-domain metadata.
 *
 * <p><b>Design:</b> this class <em>extends</em> {@link CloudletSimple}
 * rather than wrapping it, so every existing CloudSim Plus API — the
 * broker, datacenter, VM scheduler, telemetry — keeps working unchanged;
 * a {@code HealthcareTask} <em>is-a</em> {@code Cloudlet}. All
 * healthcare-specific state rides on the subclass and is invisible to the
 * core simulation.</p>
 *
 * <p><b>Cached criticality.</b> The clinical severity and priority score
 * are <em>computed elsewhere</em> (by a {@link CriticalityProvider}, from
 * the owning {@link Patient}) and cached here as plain fields. Scheduling
 * (Sprint 5) and MARL (Sprint 6) need synchronous, O(1) reads of these
 * values every tick; the task never computes them itself.</p>
 *
 * <p><b>Extensibility.</b> Core fields are typed; anything not yet
 * modelled (TGNN embeddings, uncertainty bounds, recovery hints) goes into
 * the open {@link #getAttributes() attributes} map so new signals can be
 * added without changing this class or its consumers.</p>
 */
public class HealthcareTask extends CloudletSimple {

    /** Sentinel used when the task has not been placed on any edge node. */
    public static final int UNASSIGNED_NODE = -1;

    /** Default failure probability before any prediction has been stamped. */
    public static final double DEFAULT_FAILURE_PROBABILITY = 0.0;

    /** Default failure confidence before any prediction has been stamped. */
    public static final double DEFAULT_FAILURE_CONFIDENCE = 0.0;

    // ----- Identity -----------------------------------------------------

    private final int healthcareTaskId;
    private final int patientId;

    // ----- Clinical criticality (cached; computed by CriticalityProvider)

    private double clinicalSeverity;
    private double priorityScore;

    // ----- Timing -------------------------------------------------------

    private final double arrivalTime;
    private double deadline;

    // ----- Failure prediction (cached; stamped from PredictionGateway) --

    private double failureProbability;
    private double failureConfidence;

    // ----- Placement / lifecycle ---------------------------------------

    private int assignedEdgeNodeId;
    private int migrationCount;
    private TaskState taskState;

    private final List<ExecutionEvent> executionHistory;

    // ----- Extensible metadata -----------------------------------------

    private final Map<String, Object> attributes;

    /**
     * Creates a healthcare task backed by a CloudSim cloudlet.
     *
     * @param healthcareTaskId  domain id for this task (independent of the
     *                          CloudSim cloudlet id assigned at submission)
     * @param patientId         id of the owning {@link Patient}
     * @param length            cloudlet length in Million Instructions (MI)
     * @param pesNumber         number of CPU cores (PEs) the task needs
     * @param utilizationModel  CloudSim CPU utilization model
     * @param arrivalTime       simulation time (seconds) the task arrived
     * @param deadline          simulation time (seconds) by which the task
     *                          should complete
     */
    public HealthcareTask(int healthcareTaskId,
                          int patientId,
                          long length,
                          int pesNumber,
                          UtilizationModel utilizationModel,
                          double arrivalTime,
                          double deadline) {

        // Preserve exact CloudSim workload semantics via the parent ctor.
        super(length, pesNumber, utilizationModel);

        this.healthcareTaskId = healthcareTaskId;
        this.patientId = patientId;
        this.arrivalTime = arrivalTime;
        this.deadline = deadline;

        this.clinicalSeverity = 0.0;
        this.priorityScore = 0.0;

        this.failureProbability = DEFAULT_FAILURE_PROBABILITY;
        this.failureConfidence = DEFAULT_FAILURE_CONFIDENCE;

        this.assignedEdgeNodeId = UNASSIGNED_NODE;
        this.migrationCount = 0;
        this.taskState = TaskState.CREATED;

        this.executionHistory = new ArrayList<>();
        this.attributes = new HashMap<>();
    }

    // ----- Identity accessors ------------------------------------------

    /** @return domain id for this task (not the CloudSim cloudlet id) */
    public int getHealthcareTaskId() {
        return healthcareTaskId;
    }

    /** @return id of the owning {@link Patient} */
    public int getPatientId() {
        return patientId;
    }

    // ----- Criticality accessors ---------------------------------------

    /** @return cached clinical severity (higher = more critical patient) */
    public double getClinicalSeverity() {
        return clinicalSeverity;
    }

    /**
     * Caches the clinical severity computed by a {@link CriticalityProvider}.
     *
     * @param clinicalSeverity newly computed severity
     */
    public void setClinicalSeverity(double clinicalSeverity) {
        this.clinicalSeverity = clinicalSeverity;
    }

    /** @return cached scheduling priority (higher = scheduled sooner) */
    public double getPriorityScore() {
        return priorityScore;
    }

    /**
     * Caches the priority score computed by a {@link CriticalityProvider}.
     *
     * @param priorityScore newly computed priority
     */
    public void setPriorityScore(double priorityScore) {
        this.priorityScore = priorityScore;
    }

    /**
     * Convenience method to apply a full {@link CriticalityResult} in one
     * call, keeping the cached severity and priority in sync.
     *
     * @param result the computed criticality result to cache
     */
    public void applyCriticality(CriticalityResult result) {
        this.clinicalSeverity = result.getClinicalSeverity();
        this.priorityScore = result.getPriorityScore();
    }

    // ----- Timing accessors --------------------------------------------

    /** @return simulation time (seconds) the task arrived */
    public double getArrivalTime() {
        return arrivalTime;
    }

    /** @return simulation time (seconds) by which the task should complete */
    public double getDeadline() {
        return deadline;
    }

    /**
     * @param deadline new completion deadline (seconds)
     */
    public void setDeadline(double deadline) {
        this.deadline = deadline;
    }

    // ----- Failure-prediction accessors --------------------------------

    /** @return cached probability that this task's node fails soon (0..1) */
    public double getFailureProbability() {
        return failureProbability;
    }

    /**
     * @param failureProbability probability stamped from a prediction source
     */
    public void setFailureProbability(double failureProbability) {
        this.failureProbability = failureProbability;
    }

    /** @return cached confidence in the failure probability (0..1) */
    public double getFailureConfidence() {
        return failureConfidence;
    }

    /**
     * @param failureConfidence confidence stamped from a prediction source
     */
    public void setFailureConfidence(double failureConfidence) {
        this.failureConfidence = failureConfidence;
    }

    // ----- Placement / lifecycle accessors -----------------------------

    /** @return edge node the task is assigned to, or {@link #UNASSIGNED_NODE} */
    public int getAssignedEdgeNodeId() {
        return assignedEdgeNodeId;
    }

    /**
     * @param assignedEdgeNodeId edge node the task is now assigned to
     */
    public void setAssignedEdgeNodeId(int assignedEdgeNodeId) {
        this.assignedEdgeNodeId = assignedEdgeNodeId;
    }

    /** @return number of times this task has been migrated */
    public int getMigrationCount() {
        return migrationCount;
    }

    /** @return current lifecycle state */
    public TaskState getTaskState() {
        return taskState;
    }

    /**
     * @param taskState new lifecycle state
     */
    public void setTaskState(TaskState taskState) {
        this.taskState = taskState;
    }

    /**
     * @return an unmodifiable view of this task's execution history,
     *         oldest event first
     */
    public List<ExecutionEvent> getExecutionHistory() {
        return Collections.unmodifiableList(executionHistory);
    }

    // ----- Utility methods ---------------------------------------------

    /**
     * Appends an event to this task's execution history.
     *
     * @param event the event to record (must not be null)
     */
    public void recordExecutionEvent(ExecutionEvent event) {
        executionHistory.add(event);
    }

    /**
     * Records a migration from the current node to a new node: increments
     * the migration counter, updates the assigned node, and appends a
     * {@link TaskState#MIGRATING} event to the history.
     *
     * <p>Sprint 5 does not <em>trigger</em> migrations; this method exists
     * so that Sprint 6/7 migration logic has a single, consistent entry
     * point that keeps counter, placement, and history in sync.</p>
     *
     * @param toEdgeNodeId edge node the task is migrating to
     * @param atTime       simulation time (seconds) of the migration
     * @param reason       short human-readable reason for the migration
     */
    public void incrementMigration(int toEdgeNodeId, double atTime, String reason) {
        int fromNode = this.assignedEdgeNodeId;

        this.migrationCount++;

        recordExecutionEvent(new ExecutionEvent(
                atTime, this.taskState, TaskState.MIGRATING,
                fromNode, toEdgeNodeId, reason
        ));

        this.assignedEdgeNodeId = toEdgeNodeId;
        this.taskState = TaskState.MIGRATING;
    }

    /**
     * @return the live, mutable extensible-metadata map (never null).
     *         Use for signals not yet promoted to typed fields.
     */
    public Map<String, Object> getAttributes() {
        return attributes;
    }

    @Override
    public String toString() {
        return "HealthcareTask{" +
                "healthcareTaskId=" + healthcareTaskId +
                ", patientId=" + patientId +
                ", severity=" + clinicalSeverity +
                ", priority=" + priorityScore +
                ", state=" + taskState +
                ", node=" + assignedEdgeNodeId +
                ", migrations=" + migrationCount +
                '}';
    }
}
