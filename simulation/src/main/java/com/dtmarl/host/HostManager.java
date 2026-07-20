package com.dtmarl.host;

import org.cloudsimplus.hosts.Host;
import org.cloudsimplus.hosts.HostSimple;
import org.cloudsimplus.resources.Pe;
import org.cloudsimplus.resources.PeSimple;
import org.cloudsimplus.schedulers.vm.VmSchedulerSpaceShared;

import java.util.ArrayList;
import java.util.List;

public class HostManager {

    public List<Host> createHosts() {

        List<Pe> peList = new ArrayList<>();

        for(int i = 0; i < 4; i++) {
            peList.add(new PeSimple(1000));
        }

        Host host = new HostSimple(
                16384,
                100000,
                1000000,
                peList
        );

        host.setVmScheduler(new VmSchedulerSpaceShared());

        List<Host> hosts = new ArrayList<>();
        hosts.add(host);

        return hosts;
    }
}