package com.dtmarl;

import com.dtmarl.simulation.SimulationManager;

import org.cloudsimplus.util.Log;

import ch.qos.logback.classic.Level;

public class Main {

    public static void main(String[] args) {

        // The run is now ~1500 simulated seconds across 10 hosts. At INFO the
        // CloudSim core alone emits hundreds of thousands of lines, which
        // buries the failure/degradation events we actually need to inspect.
        Log.setLevel(Level.WARN);

        System.out.println("=================================");
        System.out.println("DT-MARL Healthcare Framework");
        System.out.println("=================================");

        new SimulationManager();

        System.out.println("=================================");
        System.out.println("Execution Complete");
        System.out.println("=================================");
    }
}