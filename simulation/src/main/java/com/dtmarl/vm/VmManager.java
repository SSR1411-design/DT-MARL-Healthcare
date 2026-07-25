package com.dtmarl.vm;

import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.List;

public class VmManager {

    public List<Vm> createVms() {

        List<Vm> vmList = new ArrayList<>();

        // Create 6 Virtual Machines
        for (int vmId = 0; vmId < 6; vmId++) {

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