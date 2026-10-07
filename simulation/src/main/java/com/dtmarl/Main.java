package com.dtmarl;

import com.dtmarl.simulation.SimulationManager;
import com.dtmarl.simulation.Sprint11ScenarioConfig;

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

        try {
            if (args.length == 0) {
                // Historical behaviour: use the frozen R2-compatible defaults.
                new SimulationManager();
            } else if (args.length == 1) {
                new SimulationManager(Sprint11ScenarioConfig.fromProperties(
                        java.nio.file.Path.of(args[0])));
            } else {
                throw new IllegalArgumentException(
                        "usage: Main [path/to/sprint11-simulator.properties]");
            }
        } catch (java.io.IOException exception) {
            throw new IllegalArgumentException("cannot read Sprint 11 scenario properties", exception);
        }

        System.out.println("=================================");
        System.out.println("Execution Complete");
        System.out.println("=================================");
    }
}
