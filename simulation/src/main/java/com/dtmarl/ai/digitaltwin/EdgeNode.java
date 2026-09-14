package com.dtmarl.ai.digitaltwin;

public class EdgeNode {

    /**
     * Health lifecycle of the mirrored host, following the same "explicit
     * mode enum on the twin" pattern already used by
     * {@link DeviceNode.FailureMode}.
     *
     * NORMAL     - no active fault mechanism, or one that has not yet
     *              produced any observable effect (latent incubation).
     * DEGRADING  - a fault mechanism is measurably affecting telemetry.
     * CRITICAL   - degradation is severe; failure is likely soon.
     * FAILED     - host deactivated, telemetry flat-lined.
     * RECOVERING - repair in progress (host still down, will come back).
     *
     * IMPORTANT: this is LATENT GROUND TRUTH maintained by the simulator for
     * validation and plotting. It is deliberately NOT part of the telemetry
     * the failure predictor consumes - a real deployment has no oracle that
     * announces "this host is now DEGRADING", and feeding it to the model
     * would be circular. See TelemetrySnapshot / HistoryCollector, where it
     * is written only into the audit_* block of the exported CSV.
     */
    public enum HealthState {
        NORMAL, DEGRADING, CRITICAL, FAILED, RECOVERING
    }

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

    // Latent health lifecycle (see HealthState). Maintained by
    // HostDegradationManager / FailureManager, never by syncWithHosts.
    private HealthState healthState;

    public EdgeNode(int id) {

        this.id = id;

        cpuUsage = 0;

        ramUsage = 0;

        bandwidthUsage = 0;

        energyConsumption = 0;

        runningTasks = 0;

        active = true;

        degraded = false;

        healthState = HealthState.NORMAL;
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

    public HealthState getHealthState() {
        return healthState;
    }

    public void setHealthState(HealthState healthState) {
        this.healthState = healthState;
    }
}