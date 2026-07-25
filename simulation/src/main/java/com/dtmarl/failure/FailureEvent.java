package com.dtmarl.failure;

public class FailureEvent {

    public enum Type {
        HOST_FAILURE,
        OVERLOAD_START,
        OVERLOAD_END,
        NETWORK_FAILURE,
        NETWORK_RECOVERED,
        LINK_DEGRADED_START,
        LINK_DEGRADED_END,
        CYBER_ATTACK_START,
        CYBER_ATTACK_END
    }

    public enum Cause {
        SCHEDULED,
        RANDOM,
        SYSTEM,
        ATTACK
    }

    private final int nodeId;
    private final double time;
    private final Type type;
    private final Cause cause;
    private final double cpuAtEvent;

    public FailureEvent(int nodeId, double time, Type type, Cause cause, double cpuAtEvent) {
        this.nodeId = nodeId;
        this.time = time;
        this.type = type;
        this.cause = cause;
        this.cpuAtEvent = cpuAtEvent;
    }

    public int getNodeId() {
        return nodeId;
    }

    public double getTime() {
        return time;
    }

    public Type getType() {
        return type;
    }

    public Cause getCause() {
        return cause;
    }

    public double getCpuAtEvent() {
        return cpuAtEvent;
    }

    /**
     * One CSV row: time,nodeId,type,cause,cpuAtEvent
     */
    public String toCsvRow() {
        return String.format("%.2f,%d,%s,%s,%.2f",
                time, nodeId, type, cause, cpuAtEvent);
    }

    @Override
    public String toString() {
        return "[" + type + "] Node " + nodeId +
                " at t=" + String.format("%.2f", time) +
                "s (cause=" + cause + ", cpu=" + cpuAtEvent + "%)";
    }
}