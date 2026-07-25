package com.dtmarl.ai.digitaltwin;

/**
 * Digital Twin representation of the communication link belonging to a
 * given node (edge server). This is intentionally decoupled from
 * CloudSim's internal Host/Vm objects: CloudSim Plus's basic simulation
 * mode has no real inter-host network link to degrade, so this state is
 * driven synthetically by NetworkFailureManager and consumed later by
 * the MARL agent's state space (Sprint 5/6) and prediction model
 * (Sprint 4) as an independent signal from compute health.
 */
public class NetworkLink {

    private final int nodeId;

    private boolean up;

    private double bandwidthMbps;
    private double latencyMs;
    private double packetLossPercent;

    private boolean underAttack;

    // Baseline "healthy" values, used to reset after recovery.
    public static final double NORMAL_BANDWIDTH_MBPS = 100.0;
    public static final double NORMAL_LATENCY_MS = 10.0;
    public static final double NORMAL_PACKET_LOSS_PERCENT = 0.0;

    public NetworkLink(int nodeId) {
        this.nodeId = nodeId;
        this.up = true;
        this.bandwidthMbps = NORMAL_BANDWIDTH_MBPS;
        this.latencyMs = NORMAL_LATENCY_MS;
        this.packetLossPercent = NORMAL_PACKET_LOSS_PERCENT;
        this.underAttack = false;
    }

    public int getNodeId() {
        return nodeId;
    }

    public boolean isUp() {
        return up;
    }

    public void setUp(boolean up) {
        this.up = up;
    }

    public double getBandwidthMbps() {
        return bandwidthMbps;
    }

    public void setBandwidthMbps(double bandwidthMbps) {
        this.bandwidthMbps = bandwidthMbps;
    }

    public double getLatencyMs() {
        return latencyMs;
    }

    public void setLatencyMs(double latencyMs) {
        this.latencyMs = latencyMs;
    }

    public double getPacketLossPercent() {
        return packetLossPercent;
    }

    public void setPacketLossPercent(double packetLossPercent) {
        this.packetLossPercent = packetLossPercent;
    }

    public boolean isUnderAttack() {
        return underAttack;
    }

    public void setUnderAttack(boolean underAttack) {
        this.underAttack = underAttack;
    }

    public void resetToNormal() {
        this.up = true;
        this.bandwidthMbps = NORMAL_BANDWIDTH_MBPS;
        this.latencyMs = NORMAL_LATENCY_MS;
        this.packetLossPercent = NORMAL_PACKET_LOSS_PERCENT;
        this.underAttack = false;
    }
}