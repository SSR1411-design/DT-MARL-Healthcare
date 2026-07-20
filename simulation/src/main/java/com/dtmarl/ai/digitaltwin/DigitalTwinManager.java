package com.dtmarl.ai.digitaltwin;

import java.util.ArrayList;
import java.util.List;

public class DigitalTwinManager {

    private final List<EdgeNode> edgeNodes;

    public DigitalTwinManager() {
        edgeNodes = new ArrayList<>();
    }

    public void addNode(EdgeNode node) {
        edgeNodes.add(node);
    }

    public List<EdgeNode> getNodes() {
        return edgeNodes;
    }

    public EdgeNode getNode(int id) {

        for(EdgeNode node : edgeNodes) {
            if(node.getId() == id)
                return node;
        }

        return null;
    }

    public void printStatus() {

        System.out.println("\n========== DIGITAL TWIN ==========");

        for(EdgeNode node : edgeNodes) {

            System.out.println("Node ID : " + node.getId());
            System.out.println("CPU     : " + node.getCpuUsage());
            System.out.println("RAM     : " + node.getRamUsage());
            System.out.println("BW      : " + node.getBandwidthUsage());
            System.out.println("Energy  : " + node.getEnergyConsumption());
            System.out.println("Tasks   : " + node.getRunningTasks());
            System.out.println("Alive   : " + node.isActive());

            System.out.println("--------------------------------");
        }

    }
}