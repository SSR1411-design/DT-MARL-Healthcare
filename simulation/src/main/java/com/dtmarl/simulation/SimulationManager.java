package com.dtmarl.simulation;

import com.dtmarl.ai.digitaltwin.DigitalTwinManager;
import com.dtmarl.ai.prediction.CsvPredictionGateway;
import com.dtmarl.ai.prediction.DeviceHistoryCollector;
import com.dtmarl.ai.prediction.HistoryCollector;
import com.dtmarl.ai.prediction.PredictionGateway;
import com.dtmarl.ai.prediction.PredictionResult;
import com.dtmarl.broker.BrokerManager;
import com.dtmarl.broker.HealthcareBroker;
import com.dtmarl.cloudlet.CloudletManager;
import com.dtmarl.datacenter.DatacenterManager;
import com.dtmarl.failure.DeviceFailureManager;
import com.dtmarl.failure.FailureManager;
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
        digitalTwin.mirrorDevices(10, hosts.size());

        // ==========================================
        // Create Failure Managers
        // ==========================================

        FailureManager failureManager =
                new FailureManager(hosts, digitalTwin);

        NetworkFailureManager networkFailureManager =
                new NetworkFailureManager(digitalTwin, failureManager);

        DeviceFailureManager deviceFailureManager =
                new DeviceFailureManager(digitalTwin, failureManager);

        // Compute-layer test (Sprint 3): Node 1 fully fails at t=15s.
        failureManager.scheduleFailure(1, 15.0);
        failureManager.setOverloadCpuThreshold(45.0);

        // Network-layer test (Sprint 3.5): Node 2's link degrades
        // (latency/packet-loss) between t=5s and t=10s.
        networkFailureManager.scheduleLinkDegradation(2, 5.0, 5.0);

        // Cyber-attack test (Sprint 3.5): Node 0 gets attacked between
        // t=25s and t=30s (bandwidth spike + high packet loss signature).
        networkFailureManager.scheduleCyberAttack(0, 25.0, 5.0);

        failureManager.enableRandomFailures(0.01);
        networkFailureManager.enableRandomLinkFailures(0.01);

        // Device-layer tests (Sprint 3.75)
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

        // Failure-prediction seam (Sprint 5 placeholder: no inference/IO).
        PredictionGateway predictionGateway =
                new CsvPredictionGateway();

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

        DeviceHistoryCollector deviceHistoryCollector =
                new DeviceHistoryCollector(digitalTwin, failureManager, 10);

        // ==========================================
        // Continuously check compute + network/attack +
        // device conditions, sync the twin, and collect
        // rolling telemetry history (host AND device), on
        // every simulation clock tick.
        // ==========================================

        simulation.addOnClockTickListener(info -> {

            failureManager.checkAndTriggerFailures(info.getTime());
            networkFailureManager.checkAndTrigger(info.getTime());
            deviceFailureManager.checkAndTrigger(info.getTime());

            digitalTwin.syncWithHosts(hosts);

            historyCollector.collect(info.getTime());
            deviceHistoryCollector.collect(info.getTime());

            System.out.println(
                    "\n[Simulation Time = "
                            + String.format("%.2f", info.getTime())
                            + "s] Digital Twin snapshot:"
            );

            digitalTwin.printStatus();
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

        simulation.start();

        System.out.println(
                "Simulation Finished!"
        );

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