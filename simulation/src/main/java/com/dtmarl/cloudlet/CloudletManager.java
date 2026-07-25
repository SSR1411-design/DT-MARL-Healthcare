package com.dtmarl.cloudlet;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;

import java.util.ArrayList;
import java.util.List;

public class CloudletManager {

    public List<Cloudlet> createCloudlets() {

        List<Cloudlet> cloudlets = new ArrayList<>();

        // Bumped from 10 tasks / 10000 MI to 40 tasks / 40000 MI so the
        // workload keeps the datacenter busy well past t=30s — long
        // enough for all three Sprint 3/3.5 scheduled failure/attack
        // events to actually trigger, and to accumulate 30+ ticks per
        // node for Sprint 4's sequence_length=10 windows.
        for (int taskId = 0; taskId < 40; taskId++) {

            Cloudlet cloudlet = new CloudletSimple(
                    40000,
                    2,
                    new UtilizationModelDynamic(0.5)
            );

            cloudlets.add(cloudlet);

            System.out.println(
                    "Healthcare Task " + taskId +
                    " created successfully!"
            );
        }

        return cloudlets;
    }
}