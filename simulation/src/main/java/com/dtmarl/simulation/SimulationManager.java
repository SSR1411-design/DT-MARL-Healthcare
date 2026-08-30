package com.dtmarl.simulation;

import com.dtmarl.ai.digitaltwin.DigitalTwinManager;
import com.dtmarl.ai.prediction.DeviceHistoryCollector;
import com.dtmarl.ai.prediction.HistoryCollector;
import com.dtmarl.ai.prediction.PredictionGateway;
import com.dtmarl.ai.prediction.PredictionGateways;
import com.dtmarl.ai.prediction.PredictionResult;
import com.dtmarl.broker.BrokerManager;
import com.dtmarl.broker.HealthcareBroker;
import com.dtmarl.cloudlet.CloudletManager;
import com.dtmarl.datacenter.DatacenterManager;
import com.dtmarl.failure.DeviceFailureManager;
import com.dtmarl.failure.FailureManager;
import com.dtmarl.failure.HostDegradationConfig;
import com.dtmarl.failure.HostDegradationManager;
import com.dtmarl.failure.NetworkFailureManager;
import com.dtmarl.healthcare.CriticalityManager;
import com.dtmarl.healthcare.HealthcareTask;
import com.dtmarl.healthcare.TaskState;
import com.dtmarl.host.HostManager;
import com.dtmarl.scheduling.TaskPriorityRanker;
import com.dtmarl.vm.VmManager;

import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.vms.Vm;

import java.util.List;

public class SimulationManager {

    /**
     * Master RNG seed. Every stochastic failure/degradation manager derives
     * its own stream from this, so a run is fully reproducible - the previous
     * unseeded Randoms meant the exported dataset could never be regenerated.
     */
    public static final long SIM_SEED = 20260817L;

    /**
     * Hard cap on simulated seconds. The workload normally finishes first;
     * this just guarantees the run cannot spin indefinitely once hosts start
     * failing and recovering.
     */
    public static final double MAX_SIMULATION_SECONDS = 1500.0;

    /** Number of IoMT devices mirrored in the twin (Sprint 3.75). */
    public static final int DEVICE_COUNT = 10;

    /**
     * Per-tick Digital Twin dump. Prints ~10 lines per node per tick, which
     * at 10 nodes x 1000+ ticks is 100k+ lines of console noise, so it is off
     * by default; a compact progress line is printed periodically instead.
     */
    public static final boolean VERBOSE_TICK_TELEMETRY = false;

    /** Interval (ticks) of the compact progress line when not verbose. */
    public static final int PROGRESS_PRINT_INTERVAL = 100;

    private final CloudSimPlus simulation;

    public SimulationManager() {

        simulation = new CloudSimPlus();

        // ==========================================
        // Create Infrastructure
        // ==========================================

        HostManager hostManager =
                new HostManager();

        List<Host> hosts =
                hostManager.createHosts();

        DatacenterManager datacenterManager =
                new DatacenterManager(
                        simulation,
                        hosts
                );

        // ==========================================
        // Create Digital Twin (Hosts + Network Links + Devices)
        // ==========================================

        DigitalTwinManager digitalTwin =
                new DigitalTwinManager();

        digitalTwin.mirrorHosts(hosts);
        digitalTwin.mirrorNetworkLinks(hosts.size());
        digitalTwin.mirrorDevices(DEVICE_COUNT, hosts.size());

        // ==========================================
        // Create Failure Managers
        // ==========================================
        //
        // Each gets its own derived seed so adding/removing one manager does
        // not perturb the others' random streams.

        FailureManager failureManager =
                new FailureManager(hosts, digitalTwin, SIM_SEED);

        NetworkFailureManager networkFailureManager =
                new NetworkFailureManager(digitalTwin, failureManager, SIM_SEED + 1);

        DeviceFailureManager deviceFailureManager =
                new DeviceFailureManager(digitalTwin, failureManager, SIM_SEED + 2);

        // ==========================================
        // Progressive host degradation (the failure-generation model)
        // ==========================================
        //
        // Replaces the old "instant death at a scheduled time or on a coin
        // flip" behaviour. Hosts now accumulate latent wear, develop a fault
        // mechanism, show measurable symptoms, and only then cross a rising
        // failure hazard. Failure instants are NOT scheduled anywhere - they
        // emerge from the wear trajectory, which is what makes the exported
        // willFailSoon label predictable from telemetry without any leakage.
        //
        // Everything here is configuration; nothing is hardcoded in the loop.

        HostDegradationConfig degradationConfig = new HostDegradationConfig()
                // ~1 fault onset per host per ~1700 ticks * susceptibility,
                // so over a ~1200-tick run most hosts develop at least one.
                .setFaultOnsetProbabilityPerTick(0.0018)
                .setEpisodeWearPerTick(0.010)
                .setBaseWearPerTick(0.00015)
                .setWearNoiseSigma(0.45)
                .setSeverityRange(0.5, 2.0)
                .setSusceptibilityRange(0.6, 1.8)
                .setDegradingWearThreshold(0.25)
                .setCriticalWearThreshold(0.70)
                .setHazardScale(0.010)
                .setHazardShape(4.0)
                // A minority of real failures have no usable precursor at all
                // (PSU pop, kernel panic, OOM-kill storm). Keeping them in
                // means the dataset's Bayes-optimal recall stays honestly
                // below 100% instead of every failure being foreseeable.
                // NOTE: at 0.10 this seed happened to draw 0 abrupt episodes
                // out of 29 (a ~5% tail event), which would have made every
                // single failure predictable. 0.15 is the design value and
                // makes the task harder, not easier.
                .setAbruptFailureProbability(0.15)
                .setAbruptWearMultiplier(40.0)
                .setTelemetryNoisePercent(2.5)
                .setRecoveryEnabled(true)
                .setRepairTicksRange(12, 45)
                .setImperfectRepairRetention(0.30);

        HostDegradationManager hostDegradationManager =
                new HostDegradationManager(
                        digitalTwin, failureManager, degradationConfig, SIM_SEED + 3);

        // Overload is now a genuine SYMPTOM: baseline CPU sits near 50% and
        // only a developing fault pushes it past this threshold. (It used to
        // be set to 45%, below the constant baseline, so every host was
        // flagged "overloaded" from t=0 forever and the column was useless.)
        failureManager.setOverloadCpuThreshold(90.0);

        // Cyber-attack tests (Sprint 3.5): kept, on two different nodes at
        // different times. These are independent confounders - the
        // degradation overlay deliberately steps aside while a link is under
        // attack so the attack signature stays intact. The second one targets
        // host 4, which never fails in this run, so the dataset contains
        // "link anomaly with no failure behind it" as well as "link anomaly
        // caused by a dying NIC". A model that simply equates network noise
        // with impending failure has to get that pair wrong.
        networkFailureManager.scheduleCyberAttack(0, 25.0, 5.0);
        networkFailureManager.scheduleCyberAttack(4, 700.0, 8.0);

        // NOTE: scheduleFailure(), enableRandomFailures() and
        // enableRandomLinkFailures() are intentionally NOT used any more.
        // Both were constant-hazard processes with no observable precursor -
        // exactly the defect that made the previous dataset unlearnable. The
        // APIs are left in place for other experiments.

        // Device-layer tests (Sprint 3.75) - unchanged.
        deviceFailureManager.scheduleDropout(3, 12.0);
        deviceFailureManager.scheduleSensorFault(5, 20.0);
        deviceFailureManager.enableBatteryDrain(1.5);
        deviceFailureManager.enableRandomDropouts(0.005);
        deviceFailureManager.enableRandomSensorFaults(0.005);

        // ==========================================
        // Create Broker
        // ==========================================

        BrokerManager brokerManager =
                new BrokerManager(simulation);

        // ==========================================
        // Create VMs
        // ==========================================

        VmManager vmManager =
                new VmManager();

        brokerManager.getBroker()
                .submitVmList(
                        vmManager.createVms()
                );

        // ==========================================
        // Create Healthcare Tasks (Sprint 5)
        // ==========================================
        //
        // Tasks are now HealthcareTasks carrying a Patient, a clinical
        // criticality score, and a deadline. Criticality is computed by a
        // deterministic CriticalityManager (no ML). The HealthcareBroker
        // WRAPS the existing BrokerManager and submits tasks in clinical
        // priority order via the TaskPriorityRanker.

        CriticalityManager criticalityManager =
                new CriticalityManager();

        CloudletManager cloudletManager =
                new CloudletManager(criticalityManager);

        List<HealthcareTask> healthcareTasks =
                cloudletManager.createHealthcareTasks();

        // Mirror the workload into the Digital Twin (task layer).
        digitalTwin.mirrorTasks(healthcareTasks);

        // Failure-prediction seam. Picks up simulation/predicted_risk.csv if the
        // Python side has exported it (marl/export_risk_csv.py); otherwise falls
        // back to the inert Sprint 5 placeholder, so a checkout without the
        // export behaves exactly as Sprint 5 did.
        PredictionGateway predictionGateway =
                PredictionGateways.fromDefaultLocation(simulation::clock);

        TaskPriorityRanker priorityRanker =
                new TaskPriorityRanker();

        HealthcareBroker healthcareBroker =
                new HealthcareBroker(brokerManager, priorityRanker);

        healthcareBroker.submitHealthcareTasks(healthcareTasks);

        System.out.println(
                "Infrastructure Created Successfully!"
        );

        // ==========================================
        // Sprint 4: rolling telemetry history collectors,
        // window size = 10 ticks, for both hosts and devices.
        // ==========================================

        HistoryCollector historyCollector =
                new HistoryCollector(digitalTwin, failureManager, 10);

        // Latent wear is exported only into the audit_* columns, never into
        // the observable feature block. See HistoryCollector.exportLabeledCsv.
        historyCollector.setHealthAuditSource(hostDegradationManager);

        DeviceHistoryCollector deviceHistoryCollector =
                new DeviceHistoryCollector(digitalTwin, failureManager, 10);

        // ==========================================
        // Continuously check compute + network/attack +
        // device conditions, sync the twin, and collect
        // rolling telemetry history (host AND device), on
        // every simulation clock tick.
        // ==========================================
        //
        // ORDER MATTERS and has changed:
        //
        //   1. syncWithHosts        - pull the raw CloudSim baseline. This
        //                             OVERWRITES the twin unconditionally, so
        //                             it must come FIRST or it would erase
        //                             the degradation overlay.
        //   2. host degradation     - accrue wear, maybe fail/recover, write
        //                             the symptom overlay onto the twin.
        //   3. failure manager      - overload detection, now reading the
        //                             degradation-aware twin CPU.
        //   4. network + device     - link/attack and IoMT layers.
        //   5. collectors           - snapshot whatever the twin now says.
        //
        // Previously syncWithHosts ran AFTER the failure checks, which was
        // harmless when nothing wrote to the twin but would silently discard
        // every degradation signal.

        simulation.addOnClockTickListener(info -> {

            double t = info.getTime();

            // CloudSim fires one last tick while it is tearing the datacenter
            // down, after every VM has been destroyed. Every host then reports
            // 0% CPU / 0 tasks while still flagged active, which is a shutdown
            // artifact rather than telemetry, so it is not recorded.
            if (t > MAX_SIMULATION_SECONDS) {
                return;
            }

            digitalTwin.syncWithHosts(hosts);

            hostDegradationManager.checkAndTrigger(t);

            failureManager.checkAndTriggerFailures(t);
            networkFailureManager.checkAndTrigger(t);
            deviceFailureManager.checkAndTrigger(t);

            historyCollector.collect(t);
            deviceHistoryCollector.collect(t);

            if (VERBOSE_TICK_TELEMETRY) {

                System.out.println(
                        "\n[Simulation Time = "
                                + String.format("%.2f", t)
                                + "s] Digital Twin snapshot:"
                );

                digitalTwin.printStatus();

            } else if (((int) t) % PROGRESS_PRINT_INTERVAL == 0) {

                System.out.printf(
                        "[t=%.0fs] up=%d/%d  failed=%s%n",
                        t,
                        hosts.size() - failureManager.getFailedHosts().size(),
                        hosts.size(),
                        failureManager.getFailedHosts());
            }
        });

        // ==========================================
        // Sprint 5: SEPARATE clock-tick listener for the healthcare task
        // layer. Kept distinct from the telemetry listener above so that
        // existing failure/history/CSV behaviour is completely unchanged.
        // Each tick we: resolve where each task is placed, stamp the
        // (placeholder) failure prediction for that node onto the task,
        // update lifecycle state, then refresh the task twins.
        // ==========================================

        simulation.addOnClockTickListener(info -> {

            // Same shutdown-tick guard as the telemetry listener. VMs are
            // already destroyed by then, so resolveEdgeNodeId() would return
            // UNASSIGNED_NODE for every task and the final task dump would
            // report all 40 tasks as QUEUED on node -1. No Sprint 5 logic is
            // changed; this only stops the teardown tick from overwriting the
            // last real placement.
            if (info.getTime() > MAX_SIMULATION_SECONDS) {
                return;
            }

            for (HealthcareTask task : healthcareTasks) {

                int nodeId = resolveEdgeNodeId(task, hosts);
                task.setAssignedEdgeNodeId(nodeId);

                if (nodeId != HealthcareTask.UNASSIGNED_NODE) {
                    PredictionResult prediction =
                            predictionGateway.getPrediction(nodeId);
                    task.setFailureProbability(prediction.getFailureProbability());
                    task.setFailureConfidence(prediction.getFailureConfidence());
                }

                task.setTaskState(mapLifecycleState(task));
            }

            digitalTwin.syncTasks(healthcareTasks);
        });

        // ==========================================
        // Start Simulation
        // ==========================================

        simulation.terminateAt(MAX_SIMULATION_SECONDS);

        simulation.start();

        System.out.println(
                "Simulation Finished!"
        );

        System.out.printf(
                "Simulated %.2f seconds of cluster time.%n",
                simulation.clock());

        // ==========================================
        // Final Digital Twin State
        // ==========================================

        System.out.println(
                "\nFinal Digital Twin State (last snapshot before shutdown):"
        );

        digitalTwin.printStatus();

        // Sprint 5: healthcare task-layer mirror (separate from the
        // infrastructure telemetry above).
        digitalTwin.printTaskStatus();

        System.out.println(
                "\nTotal failed nodes during this run: "
                        + failureManager.getFailedHosts()
        );

        System.out.println(
                "Total failed devices during this run: "
                        + deviceFailureManager.getFailedDevices()
        );

        hostDegradationManager.printSummary();

        // ==========================================
        // Export unified failure log, plus BOTH Sprint 4
        // labeled telemetry-history datasets (host + device).
        // ==========================================

        failureManager.exportEventsToCsv("failure_log.csv");
        historyCollector.exportLabeledCsv("failure_history.csv", 10.0);
        deviceHistoryCollector.exportLabeledCsv("device_failure_history.csv", 10.0);
    }

    public CloudSimPlus getSimulation() {

        return simulation;
    }

    /**
     * Resolves the edge-node index a task is currently placed on by mapping
     * its bound VM's host back to its position in the {@code hosts} list —
     * the same index space the Digital Twin's {@link
     * com.dtmarl.ai.digitaltwin.EdgeNode}s use.
     *
     * @param task  the task to locate
     * @param hosts the ordered host list (index == twin node id)
     * @return the host index, or {@link HealthcareTask#UNASSIGNED_NODE} if
     *         the task is not yet bound to a created VM/host
     */
    private int resolveEdgeNodeId(HealthcareTask task, List<Host> hosts) {

        Vm vm = task.getVm();

        if (vm == null || !vm.isCreated()) {
            return HealthcareTask.UNASSIGNED_NODE;
        }

        Host host = vm.getHost();

        if (host == null || host == Host.NULL) {
            return HealthcareTask.UNASSIGNED_NODE;
        }

        int index = hosts.indexOf(host);
        return index >= 0 ? index : HealthcareTask.UNASSIGNED_NODE;
    }

    /**
     * Derives a coarse {@link TaskState} for a task from its CloudSim
     * cloudlet status, without changing any CloudSim behaviour. Migration
     * and recovery states are driven by later sprints, not inferred here.
     *
     * @param task the task to classify
     * @return the mapped lifecycle state
     */
    private TaskState mapLifecycleState(HealthcareTask task) {

        if (task.isFinished()) {
            return TaskState.COMPLETED;
        }

        if (task.getAssignedEdgeNodeId() == HealthcareTask.UNASSIGNED_NODE) {
            return TaskState.QUEUED;
        }

        return TaskState.RUNNING;
    }
}