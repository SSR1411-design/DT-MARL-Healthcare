package com.dtmarl.ai.prediction;

/**
 * One tick's worth of telemetry for a single IoMT device (Sprint 3.75
 * DeviceNode), mirroring TelemetrySnapshot's role for hosts/links but
 * for the device layer.
 */
public class DeviceTelemetrySnapshot {

    public final double time;
    public final int deviceId;

    public final double batteryLevel;
    public final boolean connected;
    public final boolean active;
    public final double signalQuality;
    public final double signalNoise;
    public final double heartbeatStaleness;

    public DeviceTelemetrySnapshot(double time, int deviceId,
                                    double batteryLevel, boolean connected,
                                    boolean active, double signalQuality,
                                    double signalNoise, double heartbeatStaleness) {

        this.time = time;
        this.deviceId = deviceId;
        this.batteryLevel = batteryLevel;
        this.connected = connected;
        this.active = active;
        this.signalQuality = signalQuality;
        this.signalNoise = signalNoise;
        this.heartbeatStaleness = heartbeatStaleness;
    }

    public String toCsvRow() {
        return String.format(
                "%.2f,%d,%.4f,%d,%d,%.4f,%.4f,%.4f",
                time, deviceId,
                batteryLevel, connected ? 1 : 0, active ? 1 : 0,
                signalQuality, signalNoise, heartbeatStaleness
        );
    }

    public static String csvHeader() {
        return "time,deviceId,battery,connected,active,signalQuality,signalNoise,heartbeatStaleness";
    }
}