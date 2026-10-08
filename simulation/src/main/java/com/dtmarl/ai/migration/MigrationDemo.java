package com.dtmarl.ai.migration;

import com.dtmarl.healthcare.CriticalityManager;
import com.dtmarl.healthcare.HealthcareTask;
import com.dtmarl.healthcare.Patient;

import org.cloudsimplus.brokers.DatacenterBroker;
import org.cloudsimplus.brokers.DatacenterBrokerSimple;
import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;
import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.List;

/**
 * Standalone verification that {@link MigrationManager} performs a <b>real</b>
 * CloudSim Plus migration, not a bookkeeping change.
 *
 * <pre>
 * mvn -q compile exec:java -Dexec.mainClass=com.dtmarl.ai.migration.MigrationDemo
 * </pre>
 *
 * <p><b>Why this is a separate main rather than a hook in the main run.</b>
 * The Sprint 1-5 pipeline exists to produce a reproducible telemetry and
 * failure dataset, and the host predictor is trained on it. Injecting
 * migrations into that run would change the dataset and silently invalidate the
 * trained model. So the mechanism is proved here, in a small self-contained
 * simulation, and the dataset run is left byte-for-byte as it was.</p>
 *
 * <p>The check is deliberately falsifiable: it records the host a running
 * cloudlet's VM occupies, requests a migration, runs the simulation on, and
 * asserts the VM's host actually changed. If CloudSim refused the move the demo
 * prints FAIL, because a migration mechanism that cannot be shown to move
 * anything is worth nothing.</p>
 */
public final class MigrationDemo {

    private MigrationDemo() {
    }

    private static final int HOSTS = 3;
    private static final int PES_PER_HOST = 8;
    private static final long HOST_MIPS = 1000;
    private static final int VMS = 3;
    private static final long TASK_LENGTH_MI = 200_000;

    /**
     * Runs the demo.
     *
     * @param args ignored
     */
    public static void main(String[] args) {

        CloudSimPlus simulation = new CloudSimPlus();

        List<Host> hosts = new ArrayList<>();
        for (int h = 0; h < HOSTS; h++) {
            List<Pe> pes = new ArrayList<>();
            for (int p = 0; p < PES_PER_HOST; p++) {
                pes.add(new PeSimple(HOST_MIPS));
            }
            Host host = new HostSimple(16_384, 100_000, 1_000_000, pes);
            host.setId(h);
            hosts.add(host);
        }

        // A migration-capable allocation policy is REQUIRED. With
        // DatacenterSimple's default VmAllocationPolicySimple, CloudSim refuses
        // enableMigrations() outright and every request comes back
        // REJECTED_MIGRATIONS_DISABLED. See MigrationManager#migrationCapablePolicy,
        // which also explains why the policy's own autonomous migrations are off.
        Datacenter datacenter = new DatacenterSimple(
                simulation, hosts, MigrationManager.migrationCapablePolicy());
        datacenter.setSchedulingInterval(1);

        DatacenterBroker broker = new DatacenterBrokerSimple(simulation);

        // Three VMs, sized so any single host can hold all of them. Where the
        // allocation policy actually puts them is NOT assumed: it is a power-
        // efficiency First Fit and in practice packs all three onto host 0. The
        // demo therefore reads the source host back with resolveNodeId() at
        // request time and derives the destination from it, instead of asserting
        // a placement CloudSim never promised.
        List<Vm> vms = new ArrayList<>();
        for (int v = 0; v < VMS; v++) {
            Vm vm = new VmSimple(HOST_MIPS, 2);
            vm.setRam(2048).setBw(1000).setSize(10_000);
            vm.setId(v);
            vms.add(vm);
        }
        broker.submitVmList(vms);

        // A single long healthcare task so it is still running when we move it.
        // Patient attributes are set the same way Sprint 5's CloudletManager
        // does it (HSI / vitals-instability / age), so criticality comes from the
        // existing CriticalityManager rather than being invented here.
        CriticalityManager criticality = new CriticalityManager();
        Patient patient = new Patient(0);
        patient.setAttribute(CriticalityManager.ATTR_HSI, 0.9);
        patient.setAttribute(CriticalityManager.ATTR_VITALS_INSTABILITY, 0.8);
        patient.setAttribute(CriticalityManager.ATTR_AGE, 71.0);
        HealthcareTask task = new HealthcareTask(
                0, patient.getPatientId(), TASK_LENGTH_MI, 2,
                new UtilizationModelDynamic(0.5), 0.0, 600.0);
        task.applyCriticality(criticality.evaluate(patient, task, 0.0));
        broker.submitCloudletList(List.of(task));

        MigrationLog log = new MigrationLog();
        MigrationManager migrations =
                new MigrationManager(datacenter, broker, hosts, vms, log)
                        .enableMigrations();

        final int[] sourceHost = {-1};
        final int[] destRequested = {-1};
        final boolean[] requested = {false};

        simulation.addOnClockTickListener(info -> {

            double now = info.getTime();

            // Request one migration, once, after the task is genuinely running.
            if (!requested[0] && now >= 5.0) {

                Vm vm = task.getVm();
                if (vm == null || vm == Vm.NULL || !vm.isCreated()) {
                    return;
                }
                boolean running = !vm.getCloudletScheduler()
                        .getCloudletExecList().isEmpty();
                if (!running) {
                    return;
                }

                int src = migrations.resolveNodeId(task);
                int dst = (src + 1) % HOSTS;
                sourceHost[0] = src;
                destRequested[0] = dst;

                System.out.printf(
                        "t=%.1f  task %d (severity %.3f) is RUNNING on host %d, "
                        + "progress %.1f%%%n",
                        now, task.getHealthcareTaskId(),
                        task.getClinicalSeverity(), src,
                        100.0 * task.getFinishedLengthSoFar() / TASK_LENGTH_MI);

                MigrationRecord rec = migrations.requestMigration(
                        task, dst, now, 1.0, true,
                        "demo: predicted_failure_risk high on source",
                        0.90, 0.10);

                System.out.printf("t=%.1f  requested -> %s%n", now, rec);
                requested[0] = true;
            }
        });

        simulation.terminateAt(400.0);
        simulation.start();

        int finalHost = migrations.resolveNodeId(task);
        Vm vm = task.getVm();
        Host host = (vm == null || vm == Vm.NULL) ? null : vm.getHost();
        int hostIdx = host == null ? -1 : hosts.indexOf(host);

        System.out.println();
        System.out.println("=".repeat(70));
        System.out.println("MIGRATION MECHANISM VERIFICATION");
        System.out.println("=".repeat(70));
        System.out.printf("  source host at request time : %d%n", sourceHost[0]);
        System.out.printf("  destination requested       : %d%n", destRequested[0]);
        System.out.printf("  VM host after simulation    : %d%n", hostIdx);
        System.out.printf("  task assigned node          : %d%s%n", finalHost,
                finalHost < 0 ? "  (VMs already destroyed at teardown; "
                        + "not a migration failure)" : "");
        System.out.printf("  task migrationCount         : %d%n",
                task.getMigrationCount());
        System.out.printf("  task final status           : %s%n",
                task.getStatus());
        System.out.println();
        System.out.print(log.summary());

        boolean moved = hostIdx == destRequested[0]
                && destRequested[0] != sourceHost[0]
                && log.applied() == 1;

        System.out.println("-".repeat(70));
        if (moved) {
            System.out.println("PASS  a running healthcare task's VM was moved "
                    + "to the requested host by CloudSim Plus.");
            System.out.println("      This is a real migration: "
                    + "Datacenter.requestVmMigration, not a counter update.");
        } else {
            System.out.println("FAIL  the VM did not end up on the requested "
                    + "host. Outcome(s) above explain why; nothing was "
                    + "recorded as applied that did not happen.");
        }
        System.out.println();
        System.out.println("SCOPE: this proves the Java-side mechanism works. "
                + "It is NOT closed-loop co-simulation -");
        System.out.println("       no trained MAPPO policy drives this manager "
                + "during the dataset run. See MigrationManager.");
    }
}
