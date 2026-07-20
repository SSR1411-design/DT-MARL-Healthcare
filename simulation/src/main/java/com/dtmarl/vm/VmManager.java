package com.dtmarl.vm;

import org.cloudsimplus.vms.Vm;
import org.cloudsimplus.vms.VmSimple;

import java.util.ArrayList;
import java.util.List;

public class VmManager {

    public List<Vm> createVms() {

        List<Vm> vmList = new ArrayList<>();

        Vm vm = new VmSimple(1000, 2);

        vm.setRam(2048)
                .setBw(1000)
                .setSize(10000);

        vmList.add(vm);

        return vmList;
    }
}