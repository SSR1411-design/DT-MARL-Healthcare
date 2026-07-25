package com.dtmarl.ai.digitaltwin;

public class EdgeNode {

    private int id;

    private double cpuUsage;

    private double ramUsage;

    private double bandwidthUsage;

    private double energyConsumption;

    private int runningTasks;

    private boolean active;

    // NEW: true when the node is overloaded (high sustained CPU) but has
    // NOT fully failed. Distinct from `active` so we can tell apart
    // "struggling" from "dead" in later sprints (prediction/MARL need this).
    private boolean degraded;

    public EdgeNode(int id) {

        this.id = id;

        cpuUsage = 0;

        ramUsage = 0;

        bandwidthUsage = 0;

        energyConsumption = 0;

        runningTasks = 0;

        active = true;

        degraded = false;
    }

    public int getId() {
        return id;
    }

    public double getCpuUsage() {
        return cpuUsage;
    }

    public void setCpuUsage(double cpuUsage) {
        this.cpuUsage = cpuUsage;
    }

    public double getRamUsage() {
        return ramUsage;
    }

    public void setRamUsage(double ramUsage) {
        this.ramUsage = ramUsage;
    }

    public double getBandwidthUsage() {
        return bandwidthUsage;
    }

    public void setBandwidthUsage(double bandwidthUsage) {
        this.bandwidthUsage = bandwidthUsage;
    }

    public double getEnergyConsumption() {
        return energyConsumption;
    }

    public void setEnergyConsumption(double energyConsumption) {
        this.energyConsumption = energyConsumption;
    }

    public int getRunningTasks() {
        return runningTasks;
    }

    public void setRunningTasks(int runningTasks) {
        this.runningTasks = runningTasks;
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }

    public boolean isDegraded() {
        return degraded;
    }

    public void setDegraded(boolean degraded) {
        this.degraded = degraded;
    }
}