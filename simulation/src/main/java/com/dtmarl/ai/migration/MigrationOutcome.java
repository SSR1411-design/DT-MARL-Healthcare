package com.dtmarl.ai.migration;

/**
 * What actually happened when a migration was requested.
 *
 * <p>This enum exists so the caller can never mistake "we asked" for "it
 * moved". Several of these outcomes are refusals, and they are refusals on
 * purpose: CloudSim Plus supports migrating a <em>VM</em> between hosts, not
 * plucking a single running cloudlet out of a VM. Where the request cannot be
 * honoured the manager says so and changes nothing, rather than incrementing a
 * counter and reporting a migration that never occurred.</p>
 */
public enum MigrationOutcome {

    /**
     * A real CloudSim VM migration was requested via
     * {@code Datacenter.requestVmMigration(vm, host)}. The VM hosting the task
     * will be moved to the destination host by the simulator, and the running
     * cloudlet moves with it. This is the genuine mechanism.
     */
    VM_MIGRATION_REQUESTED,

    /**
     * The task had not started executing, so it was retargeted to a VM on the
     * destination host with {@code DatacenterBroker.bindCloudletToVm}. No state
     * transfer is involved; this is the cheap "preemptive reroute" case.
     */
    REROUTED_BEFORE_START,

    /** Destination equals the current host: nothing to do. */
    NO_OP_SAME_HOST,

    /** The destination host index is outside the cluster. */
    REJECTED_UNKNOWN_DESTINATION,

    /** The destination host is powered off or failed. */
    REJECTED_DESTINATION_INACTIVE,

    /**
     * The destination host cannot fit the VM (PEs, RAM, BW or storage), as
     * judged by {@code Host.isSuitableForVm}. Migrating anyway would produce a
     * placement CloudSim would reject.
     */
    REJECTED_DESTINATION_UNSUITABLE,

    /** The VM is already in the middle of a migration. */
    REJECTED_ALREADY_MIGRATING,

    /** The task has no created VM and no free VM exists on the destination. */
    REJECTED_NO_VM_AVAILABLE,

    /** The task has already finished, so there is nothing to protect. */
    REJECTED_TASK_FINISHED,

    /** Migrations are disabled on the datacenter. */
    REJECTED_MIGRATIONS_DISABLED;

    /** @return whether this outcome actually changed placement */
    public boolean isApplied() {
        return this == VM_MIGRATION_REQUESTED || this == REROUTED_BEFORE_START;
    }
}
