package com.dtmarl.simulation;

import com.dtmarl.ai.digitaltwin.DigitalTwinManager;
import com.dtmarl.broker.BrokerManager;
import com.dtmarl.cloudlet.CloudletManager;
import com.dtmarl.datacenter.DatacenterManager;
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
        // Create Digital Twin (Hosts + Network Links)
        // ==========================================

        DigitalTwinManager digitalTwin =
                new DigitalTwinManager();

        digitalTwin.mirrorHosts(hosts);
        digitalTwin.mirrorNetworkLinks(hosts.size());

        // ==========================================
        // Create Failure Managers
        // ==========================================

        FailureManager failureManager =
                new FailureManager(hosts, digitalTwin);

        NetworkFailureManager networkFailureManager =
                new NetworkFailureManager(digitalTwin, failureManager);

        // Compute-layer test (Sprint 3): Node 1 fully fails at t=15s.
        failureManager.scheduleFailure(1, 15.0);
        failureManager.setOverloadCpuThreshold(45.0);

        // Network-layer test (Sprint 3.5): Node 2's link degrades
        // (latency/packet-loss) between t=5s and t=10s.
        networkFailureManager.scheduleLinkDegradation(2, 5.0, 5.0);

        // Cyber-attack test (Sprint 3.5): Node 0 gets attacked between
        // t=25s and t=30s (bandwidth spike + high packet loss signature).
        networkFailureManager.scheduleCyberAttack(0, 25.0, 5.0);

        // Random modes OFF by default. Enable later for full experiments:
        // failureManager.enableRandomFailures(0.005);
        // networkFailureManager.enableRandomLinkFailures(0.005);

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
        // Continuously check compute + network/attack
        // conditions AND sync the Digital Twin, on
        // every clock tick.
        // ==========================================

        simulation.addOnClockTickListener(info -> {

            failureManager.checkAndTriggerFailures(info.getTime());
            networkFailureManager.checkAndTrigger(info.getTime());

            digitalTwin.syncWithHosts(hosts);

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

        // ==========================================
        // Export unified failure/network/attack log
        // ==========================================

        failureManager.exportEventsToCsv("failure_log.csv");
    }

    public CloudSimPlus getSimulation() {

        return simulation;
    }
}