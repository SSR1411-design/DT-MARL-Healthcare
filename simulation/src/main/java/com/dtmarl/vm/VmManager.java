package com.dtmarl.vm;

import com.dtmarl.host.HostManager;

import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.List;

public class VmManager {

    /** VMs per host. Kept at 2 so every host keeps the same nominal load. */
    public static final int VMS_PER_HOST = 2;

    public List<Vm> createVms() {
        return createVms(HostManager.HOST_COUNT);
    }

    /** One fixed-capacity VM set per controlled physical host. */
    public List<Vm> createVms(int hostCount) {

        if (hostCount <= 0) {
            throw new IllegalArgumentException("hostCount must be positive");
        }

        List<Vm> vmList = new ArrayList<>();

        int vmCount = hostCount * VMS_PER_HOST;

        for (int vmId = 0; vmId < vmCount; vmId++) {

            Vm vm = new VmSimple(
                    1000,  // CPU capacity
                    2      // Number of CPU cores
            );

            vm.setRam(2048)       // 2 GB RAM
                    .setBw(1000)  // Bandwidth
                    .setSize(10000); // Storage

            vmList.add(vm);

            System.out.println(
                    "VM " + vmId + " created successfully!"
            );
        }

        return vmList;
    }
}
