package com.dtmarl.ai.digitaltwin;

/**
 * Digital Twin representation of a single IoMT device (wearable sensor,
 * heart-rate monitor, ECG patch, pulse oximeter, etc). Distinct from
 * EdgeNode (compute) and NetworkLink (communication) — this is the
 * device layer the physiological data actually originates from, per
 * the KBS paper's Device Layer (Section 3, Fig. 1).
 */
public class DeviceNode {

    public enum FailureMode {
        NONE,
        BATTERY_DEPLETED,
        DROPOUT,
        SENSOR_FAULT
    }

    private final int id;

    // Which EdgeNode this device currently reports to.
    private final int assignedEdgeNodeId;

    private double batteryLevel;       // 0-100 %
    private boolean connected;         // link to its edge node is up
    private double signalQuality;      // 0-100 %, higher = better
    private double signalNoise;        // 0-100 %, higher = worse
    private double lastHeartbeatTime;  // simulation time of last successful heartbeat

    private boolean active;            // overall alive/dead
    private FailureMode failureMode;

    public DeviceNode(int id, int assignedEdgeNodeId) {

        this.id = id;
        this.assignedEdgeNodeId = assignedEdgeNodeId;

        this.batteryLevel = 100.0;
        this.connected = true;
        this.signalQuality = 95.0;
        this.signalNoise = 5.0;
        this.lastHeartbeatTime = 0.0;

        this.active = true;
        this.failureMode = FailureMode.NONE;
    }

    public int getId() {
        return id;
    }

    public int getAssignedEdgeNodeId() {
        return assignedEdgeNodeId;
    }

    public double getBatteryLevel() {
        return batteryLevel;
    }

    public void setBatteryLevel(double batteryLevel) {
        this.batteryLevel = batteryLevel;
    }

    public boolean isConnected() {
        return connected;
    }

    public void setConnected(boolean connected) {
        this.connected = connected;
    }

    public double getSignalQuality() {
        return signalQuality;
    }

    public void setSignalQuality(double signalQuality) {
        this.signalQuality = signalQuality;
    }

    public double getSignalNoise() {
        return signalNoise;
    }

    public void setSignalNoise(double signalNoise) {
        this.signalNoise = signalNoise;
    }

    public double getLastHeartbeatTime() {
        return lastHeartbeatTime;
    }

    public void setLastHeartbeatTime(double lastHeartbeatTime) {
        this.lastHeartbeatTime = lastHeartbeatTime;
    }

    public boolean isActive() {
        return active;
    }

    public void setActive(boolean active) {
        this.active = active;
    }

    public FailureMode getFailureMode() {
        return failureMode;
    }

    public void setFailureMode(FailureMode failureMode) {
        this.failureMode = failureMode;
    }

    /**
     * How long (in simulation seconds) since this device last checked
     * in. Useful directly as a Sprint 4 predictive-failure feature —
     * rising staleness is often an earlier signal than the dropout
     * event itself.
     */
    public double heartbeatStaleness(double currentTime) {
        return currentTime - lastHeartbeatTime;
    }
}