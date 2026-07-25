package com.dtmarl.simulation;

import com.dtmarl.ai.digitaltwin.DigitalTwinManager;
import com.dtmarl.ai.prediction.DeviceHistoryCollector;
import com.dtmarl.ai.prediction.HistoryCollector;
import com.dtmarl.broker.BrokerManager;
import com.dtmarl.cloudlet.CloudletManager;
import com.dtmarl.datacenter.DatacenterManager;
import com.dtmarl.failure.DeviceFailureManager;
import com.dtmarl.failure.FailureManager;
import com.dtmarl.failure.NetworkFailureManager;
import com.dtmarl.host.HostManager;
import com.dtmarl.vm.VmManager;

import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.hosts.Host;

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
        // Create Healthcare Tasks
        // ==========================================

        CloudletManager cloudletManager =
                new CloudletManager();

        brokerManager.getBroker()
                .submitCloudletList(
                        cloudletManager.createCloudlets()
                );

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
}