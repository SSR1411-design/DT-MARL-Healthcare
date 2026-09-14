package com.dtmarl.host;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.power.models.PowerModelHostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.vm.VmSchedulerSpaceShared;

import java.util.ArrayList;
import java.util.List;

public class HostManager {

    /**
     * Number of edge hosts. Raised from 3 to 10 so the failure dataset
     * contains enough independent nodes for leave-one-node-out validation
     * (with 3 nodes a single held-out node was a third of the data).
     */
    public static final int HOST_COUNT = 10;

    public List<Host> createHosts() {

        List<Host> hosts = new ArrayList<>();

        for (int hostId = 0; hostId < HOST_COUNT; hostId++) {

            List<Pe> peList = new ArrayList<>();

            // Each Host has 4 CPU cores
            for (int i = 0; i < 4; i++) {
                peList.add(new PeSimple(1000));
            }

            Host host = new HostSimple(
                    16384,     // RAM: 16 GB
                    100000,    // Bandwidth
                    1000000,   // Storage
                    peList
            );

            host.setVmScheduler(
                    new VmSchedulerSpaceShared()
            );

            // Power model: max power draw at 100% CPU = 200W,
            // static/idle power draw at 0% CPU = 50W.
            // CloudSim scales linearly between these based on
            // real-time CPU utilization.
            host.setPowerModel(
                    new PowerModelHostSimple(200, 50)
            );

            hosts.add(host);

            System.out.println(
                    "Host " + hostId + " created successfully!"
            );
        }

        return hosts;
    }
}