package com.dtmarl.healthcare;

/**
 * Lifecycle states of a {@link HealthcareTask} as it moves through the
 * simulation.
 *
 * <p>The state machine is intentionally coarse for Sprint 5 — it records
 * <em>where in its lifecycle</em> a task currently is, without prescribing
 * how transitions happen. Later sprints attach behaviour to these states:
 * Sprint 6 (MARL) will read {@link #RUNNING}/{@link #MIGRATING} when
 * choosing actions, and Sprint 7 (self-healing) will drive
 * {@link #RECOVERING} during predictive recovery.</p>
 *
 * <p>Adding a new state here must never silently change existing behaviour;
 * consumers should always handle unknown/other states defensively.</p>
 */
public enum TaskState {

    /** Task object created but not yet submitted to the broker. */
    CREATED,

    /** Submitted and waiting for placement / execution to begin. */
    QUEUED,

    /** Actively executing on an assigned edge node / VM. */
    RUNNING,

    /** In the process of being moved to a different edge node (Sprint 6/7). */
    MIGRATING,

    /** Finished successfully. Terminal. */
    COMPLETED,

    /** Ended due to a failure and not (yet) recovered. Terminal for Sprint 5. */
    FAILED,

    /** Undergoing predictive self-healing / recovery (Sprint 7). */
    RECOVERING
}
