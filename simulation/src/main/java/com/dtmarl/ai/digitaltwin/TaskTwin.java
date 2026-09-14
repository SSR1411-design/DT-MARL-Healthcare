package com.dtmarl.ai.digitaltwin;

import com.dtmarl.healthcare.ExecutionEvent;
import com.dtmarl.healthcare.HealthcareTask;
import com.dtmarl.healthcare.TaskState;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Digital-twin mirror of a single {@link HealthcareTask}.
 *
 * <p>The Digital Twin already mirrors compute hosts ({@link EdgeNode}),
 * network links ({@link NetworkLink}), and IoMT devices ({@link DeviceNode}).
 * A {@code TaskTwin} extends that same mirroring pattern to the workload
 * layer, giving Sprint 6 (MARL) and Sprint 7 (self-healing) a read-only,
 * queryable view of each task's placement, lifecycle, clinical priority, and
 * failure risk without reaching into the live CloudSim objects.</p>
 *
 * <p><b>Mirror, not owner.</b> A twin never computes anything; it holds a
 * snapshot copied from its source task by {@link #syncFrom(HealthcareTask)}.
 * The authoritative state always lives on the {@link HealthcareTask}. This
 * keeps the twin cheap to read every tick and impossible to accidentally
 * mutate the real task through.</p>
 */
public class TaskTwin {

    private final int healthcareTaskId;
    private final int patientId;

    private int currentEdgeNodeId;
    private TaskState taskState;

    private double priorityScore;
    private double clinicalSeverity;

    private double failureProbability;
    private double failureConfidence;

    private int migrationCount;

    private final List<ExecutionEvent> executionHistory;

    /**
     * Creates a task twin and immediately mirrors the given task's state.
     *
     * @param task the source task to mirror (must not be null)
     */
    public TaskTwin(HealthcareTask task) {
        this.healthcareTaskId = task.getHealthcareTaskId();
        this.patientId = task.getPatientId();
        this.executionHistory = new ArrayList<>();
        syncFrom(task);
    }

    /**
     * Copies the current state of the source task into this twin. Safe to
     * call every simulation tick; it overwrites the mirrored fields and
     * refreshes the execution-history snapshot.
     *
     * @param task the source task to mirror (must not be null)
     */
    public void syncFrom(HealthcareTask task) {
        this.currentEdgeNodeId = task.getAssignedEdgeNodeId();
        this.taskState = task.getTaskState();
        this.priorityScore = task.getPriorityScore();
        this.clinicalSeverity = task.getClinicalSeverity();
        this.failureProbability = task.getFailureProbability();
        this.failureConfidence = task.getFailureConfidence();
        this.migrationCount = task.getMigrationCount();

        this.executionHistory.clear();
        this.executionHistory.addAll(task.getExecutionHistory());
    }

    /** @return domain id of the mirrored task */
    public int getHealthcareTaskId() {
        return healthcareTaskId;
    }

    /** @return id of the patient owning the mirrored task */
    public int getPatientId() {
        return patientId;
    }

    /** @return edge node the task is on, or {@link HealthcareTask#UNASSIGNED_NODE} */
    public int getCurrentEdgeNodeId() {
        return currentEdgeNodeId;
    }

    /** @return mirrored lifecycle state */
    public TaskState getTaskState() {
        return taskState;
    }

    /** @return mirrored scheduling priority */
    public double getPriorityScore() {
        return priorityScore;
    }

    /** @return mirrored clinical severity */
    public double getClinicalSeverity() {
        return clinicalSeverity;
    }

    /** @return mirrored near-term failure probability (0..1) */
    public double getFailureProbability() {
        return failureProbability;
    }

    /** @return mirrored confidence in the failure probability (0..1) */
    public double getFailureConfidence() {
        return failureConfidence;
    }

    /** @return mirrored migration count */
    public int getMigrationCount() {
        return migrationCount;
    }

    /**
     * @return an unmodifiable snapshot of the mirrored task's execution
     *         history, oldest event first
     */
    public List<ExecutionEvent> getExecutionHistory() {
        return Collections.unmodifiableList(executionHistory);
    }

    @Override
    public String toString() {
        return "TaskTwin{" +
                "taskId=" + healthcareTaskId +
                ", patientId=" + patientId +
                ", node=" + currentEdgeNodeId +
                ", state=" + taskState +
                ", priority=" + priorityScore +
                ", pFail=" + failureProbability +
                ", migrations=" + migrationCount +
                '}';
    }
}
