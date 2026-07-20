package com.dtmarl.ai.common;

import com.dtmarl.ai.digitaltwin.DigitalTwinManager;
import com.dtmarl.ai.digitaltwin.EdgeNode;

import java.util.ArrayList;
import java.util.List;

public class StateBuilder {

    public double[] buildState(DigitalTwinManager twin) {

        List<Double> state = new ArrayList<>();

        for (EdgeNode node : twin.getNodes()) {

            state.add(node.getCpuUsage());
            state.add(node.getRamUsage());
            state.add(node.getBandwidthUsage());
            state.add(node.getEnergyConsumption());
            state.add((double) node.getRunningTasks());
        }

        double[] result = new double[state.size()];

        for(int i = 0; i < state.size(); i++)
            result[i] = state.get(i);

        return result;
    }
}