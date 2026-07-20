package com.dtmarl.simulation;

import com.dtmarl.broker.BrokerManager;
import com.dtmarl.cloudlet.CloudletManager;
import com.dtmarl.datacenter.DatacenterManager;
import com.dtmarl.host.HostManager;
import com.dtmarl.vm.VmManager;

import org.cloudsimplus.core.CloudSimPlus;

public class SimulationManager {

    private final CloudSimPlus simulation;

    public SimulationManager() {

        simulation = new CloudSimPlus();

        // Create Infrastructure
        HostManager hostManager = new HostManager();
        DatacenterManager datacenterManager =
                new DatacenterManager(simulation, hostManager.createHosts());

        // Create Broker
        BrokerManager brokerManager =
                new BrokerManager(simulation);

        // Create VMs
        VmManager vmManager = new VmManager();

        brokerManager.getBroker()
                .submitVmList(vmManager.createVms());

        // Create Tasks
        CloudletManager cloudletManager =
                new CloudletManager();

        brokerManager.getBroker()
                .submitCloudletList(cloudletManager.createCloudlets());

        System.out.println("Infrastructure Created Successfully!");

        simulation.start();

        System.out.println("Simulation Finished!");
    }

    public CloudSimPlus getSimulation() {
        return simulation;
    }
}