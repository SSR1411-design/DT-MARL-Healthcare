package com.dtmarl.cloudlet;

import org.cloudsimplus.cloudlets.Cloudlet;
import org.cloudsimplus.cloudlets.CloudletSimple;
import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;

import java.util.ArrayList;
import java.util.List;

public class CloudletManager {

    public List<Cloudlet> createCloudlets() {

        List<Cloudlet> cloudlets = new ArrayList<>();

        Cloudlet cloudlet =
                new CloudletSimple(
                        10000,
                        2,
                        new UtilizationModelDynamic(0.5)
                );

        cloudlets.add(cloudlet);

        return cloudlets;
    }
}