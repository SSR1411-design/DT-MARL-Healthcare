package com.dtmarl.healthcare;

/**
 * An immutable record of a single noteworthy moment in a
 * {@link HealthcareTask}'s life — a state change, a migration, or any
 * other event worth auditing.
 *
 * <p>The execution history built from these events is the causal trail
 * that Sprint 7 (self-healing) uses to reason about recovery, and that
 * evaluation/graphing consumes after a run. Because it is immutable, an
 * event can be safely shared between a task and its {@link
 * com.dtmarl.ai.digitaltwin.TaskTwin} mirror without defensive copying.</p>
 */
public final class ExecutionEvent {

    private final double time;
    private final TaskState fromState;
    private final TaskState toState;
    private final int fromEdgeNodeId;
    private final int toEdgeNodeId;
    private final String description;

    /**
     * Creates an execution event.
     *
     * @param time           simulation time (seconds) at which the event occurred
     * @param fromState      task state before the event (may equal {@code toState})
     * @param toState        task state after the event
     * @param fromEdgeNodeId edge node the task was on before the event, or
     *                       {@link HealthcareTask#UNASSIGNED_NODE} if none
     * @param toEdgeNodeId   edge node the task is on after the event, or
     *                       {@link HealthcareTask#UNASSIGNED_NODE} if none
     * @param description    short human-readable reason/label for the event
     */
    public ExecutionEvent(double time,
                          TaskState fromState,
                          TaskState toState,
                          int fromEdgeNodeId,
                          int toEdgeNodeId,
                          String description) {

        this.time = time;
        this.fromState = fromState;
        this.toState = toState;
        this.fromEdgeNodeId = fromEdgeNodeId;
        this.toEdgeNodeId = toEdgeNodeId;
        this.description = description;
    }

    /** @return simulation time (seconds) at which the event occurred */
    public double getTime() {
        return time;
    }

    /** @return task state before the event */
    public TaskState getFromState() {
        return fromState;
    }

    /** @return task state after the event */
    public TaskState getToState() {
        return toState;
    }

    /** @return edge node id before the event (or {@link HealthcareTask#UNASSIGNED_NODE}) */
    public int getFromEdgeNodeId() {
        return fromEdgeNodeId;
    }

    /** @return edge node id after the event (or {@link HealthcareTask#UNASSIGNED_NODE}) */
    public int getToEdgeNodeId() {
        return toEdgeNodeId;
    }

    /** @return short human-readable reason/label for the event */
    public String getDescription() {
        return description;
    }

    @Override
    public String toString() {
        return String.format(
                "[t=%.2f] %s -> %s (node %d -> %d) : %s",
                time, fromState, toState, fromEdgeNodeId, toEdgeNodeId, description
        );
    }
}
