package com.dtmarl.ai.prediction;

import com.dtmarl.ai.digitaltwin.EdgeNode;

/**
 * One tick's worth of telemetry for a single node, combining compute
 * (EdgeNode) and network (NetworkLink) state. This is the raw feature
 * vector that both the rolling history buffer and the CSV exporter
 * work with.
 *
 * The 14 fields written by {@link #toCsvRow()} are the OBSERVABLE ones - the
 * only ones a real monitoring agent could read, and therefore the only ones
 * a predictor may consume.
 *
 * {@link #healthState} and {@link #wear} are LATENT ground truth carried
 * alongside for validation. They are excluded from toCsvRow() on purpose and
 * are exported only into the clearly-marked audit_* block appended by
 * {@link HistoryCollector#exportLabeledCsv}.
 */
public class TelemetrySnapshot {

    public final double time;
    public final int nodeId;

    public final double cpuUsage;
    public final double ramUsage;
    public final double bandwidthUsage;
    public final double energyConsumption;
    public final int runningTasks;
    public final boolean active;
    public final boolean degraded;

    public final boolean linkUp;
    public final double linkBandwidthMbps;
    public final double linkLatencyMs;
    public final double linkPacketLossPercent;
    public final boolean underAttack;

    // ---- latent, audit-only: NEVER part of toCsvRow() ---------------------
    public final EdgeNode.HealthState healthState;
    public final double wear;

    public TelemetrySnapshot(double time, int nodeId,
                              double cpuUsage, double ramUsage,
                              double bandwidthUsage, double energyConsumption,
                              int runningTasks, boolean active, boolean degraded,
                              boolean linkUp, double linkBandwidthMbps,
                              double linkLatencyMs, double linkPacketLossPercent,
                              boolean underAttack,
                              EdgeNode.HealthState healthState, double wear) {

        this.time = time;
        this.nodeId = nodeId;
        this.cpuUsage = cpuUsage;
        this.ramUsage = ramUsage;
        this.bandwidthUsage = bandwidthUsage;
        this.energyConsumption = energyConsumption;
        this.runningTasks = runningTasks;
        this.active = active;
        this.degraded = degraded;
        this.linkUp = linkUp;
        this.linkBandwidthMbps = linkBandwidthMbps;
        this.linkLatencyMs = linkLatencyMs;
        this.linkPacketLossPercent = linkPacketLossPercent;
        this.underAttack = underAttack;
        this.healthState = healthState;
        this.wear = wear;
    }

    /**
     * Feature row for CSV export (label is appended separately by the
     * collector once it knows the future outcome).
     */
    public String toCsvRow() {
        return String.format(
                "%.2f,%d,%.4f,%.4f,%.4f,%.4f,%d,%d,%d,%d,%.4f,%.4f,%.4f,%d",
                time, nodeId,
                cpuUsage, ramUsage, bandwidthUsage, energyConsumption,
                runningTasks, active ? 1 : 0, degraded ? 1 : 0,
                linkUp ? 1 : 0, linkBandwidthMbps, linkLatencyMs,
                linkPacketLossPercent, underAttack ? 1 : 0
        );
    }

    public static String csvHeader() {
        return "time,nodeId,cpu,ram,bandwidth,energy,runningTasks,active,degraded," +
                "linkUp,linkBandwidthMbps,linkLatencyMs,linkPacketLoss,underAttack";
    }
}