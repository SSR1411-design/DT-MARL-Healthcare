package com.dtmarl;

import com.dtmarl.simulation.SimulationManager;

public class Main {

    public static void main(String[] args) {

        System.out.println("=================================");
        System.out.println("DT-MARL Healthcare Framework");
        System.out.println("=================================");

        new SimulationManager();

        System.out.println("=================================");
        System.out.println("Execution Complete");
        System.out.println("=================================");
    }
}