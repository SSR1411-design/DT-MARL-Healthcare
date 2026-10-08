package com.dtmarl.ai.migration;

import com.dtmarl.healthcare.HealthcareTask;

import org.cloudsimplus.allocationpolicies.VmAllocationPolicy;
import org.cloudsimplus.allocationpolicies.migration.VmAllocationPolicyMigrationStaticThreshold;
import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.selectionpolicies.VmSelectionPolicyMinimumUtilization;
import org.cloudsimplus.vms.Vm;

import java.util.Collections;
import java.util.List;
import java.util.Map;

/**
 * The smallest practical task-migration mechanism for this repository, built on
 * CloudSim Plus's own migration support.
 *
 * <h2>What actually moves, and the granularity mismatch</h2>
 *
 * <p>CloudSim Plus migrates <b>VMs between hosts</b>
 * ({@code Datacenter.requestVmMigration}). It has no operation that lifts a
 * single running cloudlet out of one VM and drops it into another. Sprint 6's
 * decision problem is expressed per <em>task</em>. Those two granularities do
 * not line up perfectly, and this class does not pretend otherwise:</p>
 *
 * <ul>
 *   <li>A task that has <b>not started</b> is retargeted with
 *       {@code bindCloudletToVm} to a free VM on the destination host. Exact,
 *       cheap, no state transfer - the "preemptive reroute" case.</li>
 *   <li>A task that <b>is running</b> is moved by migrating the VM it occupies.
 *       This is a real CloudSim migration and the running cloudlet genuinely
 *       relocates. But if that VM also hosts other tasks, they move too. The
 *       count of such passengers is recorded in
 *       {@link MigrationRecord#getCollateralTasksMoved()} rather than ignored,
 *       because it is a real cost of the mechanism.</li>
 *   <li>Anything that cannot be honoured is <b>refused</b> with a specific
 *       {@link MigrationOutcome} and no state change. No counter is incremented
 *       for a move that did not happen.</li>
 * </ul>
 *
 * <p>In the configuration this project runs (2 VMs per host, one task per VM
 * slot at a time) the collateral count is usually zero, but it is measured
 * rather than assumed.</p>
 *
 * <h2>Relationship to the Python MARL environment</h2>
 *
 * <p>The MARL policy is trained against the trace-driven Python environment
 * ({@code python-ai/marl/env.py}), which models migration as
 * checkpoint-and-resume with an explicit transfer latency. That environment is
 * a <b>replay</b> of recorded Digital Twin telemetry, not a closed loop with a
 * live CloudSim instance. This class is the Java-side mechanism that a decision
 * would be executed through, and it is verified to work by
 * {@link MigrationDemo}. Sprint 6 does <b>not</b> claim closed-loop
 * co-simulation: no trained policy drives this manager during the dataset run.
 * Saying so plainly is the point - the alternative would be a fake
 * integration.</p>
 *
 * <p>Nothing here is wired into the Sprint 1-5 dataset-generation path, so the
 * exported telemetry and failure logs are unchanged by its presence.</p>
 */
public final class MigrationManager {

    private final Datacenter datacenter;
    private final DatacenterBroker broker;
    private final List<Host> hosts;
    private final List<Vm> vms;
    private final MigrationLog log;

    /**
     * Creates a migration manager over an existing cluster.
     *
     * @param datacenter the datacenter owning the hosts
     * @param broker     broker that submitted the tasks
     * @param hosts      host list, indexed the same way node ids are
     * @param vms        every VM in the cluster
     * @param log        destination for audit records
     */
    public MigrationManager(Datacenter datacenter,
                            DatacenterBroker broker,
                            List<Host> hosts,
                            List<Vm> vms,
                            MigrationLog log) {
        this.datacenter = datacenter;
        this.broker = broker;
        this.hosts = hosts;
        this.vms = vms;
        this.log = log;
    }

    /** @return the audit log */
    public MigrationLog getLog() {
        return log;
    }

    /**
     * Builds the {@link VmAllocationPolicy} a {@link Datacenter} must be
     * constructed with for {@link #enableMigrations()} to have any effect.
     *
     * <p><b>This is not optional plumbing.</b> {@code DatacenterSimple}'s default
     * {@code VmAllocationPolicySimple} does not implement
     * {@code isVmMigrationSupported()}, so {@code Datacenter.enableMigrations()}
     * logs <i>"It was requested to enable VM migrations but the
     * VmAllocationPolicySimple doesn't support that"</i>, returns without setting
     * the flag, and every subsequent request is refused with
     * {@link MigrationOutcome#REJECTED_MIGRATIONS_DISABLED}. The requirement is
     * invisible at the call site, which is why it lives here next to the code
     * that depends on it.</p>
     *
     * <p><b>Autonomous migrations are switched off.</b> The returned policy
     * overrides {@code getOptimizedAllocationMap} to return an empty map, so
     * CloudSim's own load-balancing heuristics never move a VM of their own
     * accord - neither the over-utilisation path nor the under-utilisation
     * consolidation path (whose threshold defaults to 0.35 and cannot be set to
     * 0). Only the mechanism CloudSim provides is wanted, not its policy: every
     * move in this project is requested explicitly by a decision-maker, and a
     * datacenter that also migrated on its own would make it impossible to
     * attribute an outcome to the decision that was actually taken. Returning an
     * empty map is exactly what the non-migration policies do, so this removes
     * behaviour rather than inventing any.</p>
     *
     * <p>The over-utilisation threshold argument is still required by the
     * constructor and must lie strictly inside (0, 1); it is set high and is
     * never consulted, because the optimisation entry point is disabled above.</p>
     *
     * @return a migration-capable allocation policy that initiates no migrations
     *         on its own
     */
    public static VmAllocationPolicy migrationCapablePolicy() {
        return new VmAllocationPolicyMigrationStaticThreshold(
                new VmSelectionPolicyMinimumUtilization(), 0.99) {
            @Override
            public Map<Vm, Host> getOptimizedAllocationMap(
                    List<? extends Vm> vmList) {
                return Collections.emptyMap();
            }
        };
    }

    /**
     * Turns on CloudSim's migration support. Off by default in
     * {@code DatacenterSimple}, and left off unless a caller explicitly wants
     * migrations, so the Sprint 1-5 run is untouched.
     *
     * @return this manager, for chaining
     */
    public MigrationManager enableMigrations() {
        datacenter.enableMigrations();
        return this;
    }

    /**
     * Requests that {@code task} be moved to host {@code destNodeId}.
     *
     * <p>All risk values passed in are decision-time readings supplied by the
     * caller; this method performs no prediction of its own and reads no future
     * state.</p>
     *
     * @param task       the task to protect
     * @param destNodeId destination host index
     * @param now        current simulation time (s)
     * @param cost       relative migration cost to charge
     * @param preemptive true when the trigger was predicted risk rather than an
     *                   observed symptom
     * @param reason     short trigger description for the audit trail
     * @param sourceRisk predicted_failure_risk of the current host at {@code now}
     * @param destRisk   predicted_failure_risk of the destination at {@code now}
     * @return the record that was appended to the log, including the outcome
     */
    public MigrationRecord requestMigration(HealthcareTask task,
                                            int destNodeId,
                                            double now,
                                            double cost,
                                            boolean preemptive,
                                            String reason,
                                            double sourceRisk,
                                            double destRisk) {

        int sourceNodeId = resolveNodeId(task);

        MigrationOutcome outcome = MigrationOutcome.REJECTED_NO_VM_AVAILABLE;
        int collateral = 0;

        if (task.getStatus() == Cloudlet.Status.SUCCESS
                || task.getStatus() == Cloudlet.Status.FAILED
                || task.getStatus() == Cloudlet.Status.CANCELED) {

            outcome = MigrationOutcome.REJECTED_TASK_FINISHED;

        } else if (destNodeId < 0 || destNodeId >= hosts.size()) {

            outcome = MigrationOutcome.REJECTED_UNKNOWN_DESTINATION;

        } else if (destNodeId == sourceNodeId) {

            outcome = MigrationOutcome.NO_OP_SAME_HOST;

        } else {
            Host dest = hosts.get(destNodeId);

            if (!dest.isActive()) {
                outcome = MigrationOutcome.REJECTED_DESTINATION_INACTIVE;
            } else {
                Vm vm = task.getVm();
                boolean placed = vm != null && vm != Vm.NULL && vm.isCreated();

                if (!placed) {
                    // Not yet running anywhere: retarget the cloudlet.
                    Vm target = findFreeVmOn(destNodeId);
                    if (target == null) {
                        outcome = MigrationOutcome.REJECTED_NO_VM_AVAILABLE;
                    } else if (broker.bindCloudletToVm(task, target)) {
                        task.incrementMigration(destNodeId, now, reason);
                        outcome = MigrationOutcome.REROUTED_BEFORE_START;
                    } else {
                        outcome = MigrationOutcome.REJECTED_NO_VM_AVAILABLE;
                    }
                } else if (vm.isInMigration()) {
                    outcome = MigrationOutcome.REJECTED_ALREADY_MIGRATING;
                } else if (!datacenter.isMigrationsEnabled()) {
                    outcome = MigrationOutcome.REJECTED_MIGRATIONS_DISABLED;
                } else if (!dest.isSuitableForVm(vm)) {
                    outcome = MigrationOutcome.REJECTED_DESTINATION_UNSUITABLE;
                } else {
                    // Count the passengers before asking CloudSim to move the VM.
                    collateral = Math.max(0, runningCloudletsOn(vm) - 1);
                    datacenter.requestVmMigration(vm, dest);
                    task.incrementMigration(destNodeId, now, reason);
                    outcome = MigrationOutcome.VM_MIGRATION_REQUESTED;
                }
            }
        }

        MigrationRecord record = new MigrationRecord(
                now, task.getHealthcareTaskId(), task.getPatientId(),
                task.getClinicalSeverity(), sourceNodeId, destNodeId,
                outcome.isApplied() ? cost : 0.0, preemptive, reason, outcome,
                sourceRisk, destRisk, collateral);

        log.add(record);
        return record;
    }

    /**
     * @param task task to locate
     * @return host index the task currently occupies, or
     *         {@link HealthcareTask#UNASSIGNED_NODE}
     */
    public int resolveNodeId(HealthcareTask task) {
        Vm vm = task.getVm();
        if (vm == null || vm == Vm.NULL || !vm.isCreated()) {
            return HealthcareTask.UNASSIGNED_NODE;
        }
        Host h = vm.getHost();
        if (h == null || h == Host.NULL) {
            return HealthcareTask.UNASSIGNED_NODE;
        }
        int i = hosts.indexOf(h);
        return i >= 0 ? i : HealthcareTask.UNASSIGNED_NODE;
    }

    /** @return a created VM on {@code nodeId} with no executing cloudlets */
    private Vm findFreeVmOn(int nodeId) {
        Host host = hosts.get(nodeId);
        for (Vm vm : vms) {
            if (!vm.isCreated() || vm.getHost() != host) {
                continue;
            }
            if (vm.getCloudletScheduler().getCloudletExecList().isEmpty()) {
                return vm;
            }
        }
        return null;
    }

    /** @return how many cloudlets are executing on {@code vm} */
    private int runningCloudletsOn(Vm vm) {
        return vm.getCloudletScheduler().getCloudletExecList().size();
    }
}
