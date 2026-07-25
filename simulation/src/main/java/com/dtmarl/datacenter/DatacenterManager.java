package com.dtmarl.datacenter;

import org.cloudsimplus.core.CloudSimPlus;
import org.cloudsimplus.datacenters.Datacenter;
import org.cloudsimplus.datacenters.DatacenterSimple;
import org.cloudsimplus.hosts.Host;

import java.util.List;

public class DatacenterManager {

    private final Datacenter datacenter;

    public DatacenterManager(CloudSimPlus simulation, List<Host> hosts) {

        datacenter = new DatacenterSimple(simulation, hosts);

        // Forces CloudSim Plus to generate a periodic "clock tick" event
        // every 1 simulated second, regardless of whether any cloudlet
        // start/finish events are happening. Without this, tick events
        // only fire when discrete events occur — which is rare with a
        // short workload — leaving almost no telemetry history for
        // Sprint 4's rolling-window collectors to work with.
        datacenter.setSchedulingInterval(1);
    }

    public Datacenter getDatacenter() {
        return datacenter;
    }
}