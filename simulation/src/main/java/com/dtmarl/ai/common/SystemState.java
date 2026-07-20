package com.dtmarl.ai.common;

public class SystemState {

    private double averageCpu;

    private double averageRam;

    private double averageBandwidth;

    private double averageEnergy;

    private int totalTasks;

    public SystemState() {
    }

    public double getAverageCpu() {
        return averageCpu;
    }

    public void setAverageCpu(double averageCpu) {
        this.averageCpu = averageCpu;
    }

    public double getAverageRam() {
        return averageRam;
    }

    public void setAverageRam(double averageRam) {
        this.averageRam = averageRam;
    }

    public double getAverageBandwidth() {
        return averageBandwidth;
    }

    public void setAverageBandwidth(double averageBandwidth) {
        this.averageBandwidth = averageBandwidth;
    }

    public double getAverageEnergy() {
        return averageEnergy;
    }

    public void setAverageEnergy(double averageEnergy) {
        this.averageEnergy = averageEnergy;
    }

    public int getTotalTasks() {
        return totalTasks;
    }

    public void setTotalTasks(int totalTasks) {
        this.totalTasks = totalTasks;
    }
}