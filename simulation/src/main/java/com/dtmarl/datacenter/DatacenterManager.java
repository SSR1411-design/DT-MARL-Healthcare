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
    }

    public Datacenter getDatacenter() {
        return datacenter;
    }
}