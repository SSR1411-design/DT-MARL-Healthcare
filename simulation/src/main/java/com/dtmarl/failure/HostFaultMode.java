package com.dtmarl.failure;

/**
 * Physical fault mechanisms a host can develop.
 *
 * Each mechanism carries a SIGNATURE: how strongly it pushes each observable
 * telemetry channel as the underlying wear accumulates. This is what makes
 * the generated dataset non-trivial:
 *
 *   * different failures move different SUBSETS of the signals, so no single
 *     column is a giveaway and a model must combine channels;
 *   * the shape exponent makes some mechanisms ramp early and gently
 *     (exponent < 1, e.g. thermal creep) and others stay quiet then spike
 *     late (exponent > 1, e.g. a memory leak that only hurts once the
 *     working set no longer fits), so "how far along am I" is a real
 *     inference problem rather than a fixed threshold.
 *
 * Weights are dimensionless multipliers in [0,1] applied to the per-channel
 * amplitudes configured in {@link HostDegradationConfig}.
 */
public enum HostFaultMode {

    /** No fault mechanism active. */
    NONE(0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0.0),

    /**
     * Cooling degradation / dust build-up. The CPU throttles and retries,
     * so utilisation and power climb together, early and gradually.
     */
    THERMAL_THROTTLING(1.00, 0.15, 0.10, 1.00, 0.70, 0.30, 0.20, 0.20, 0.70, 0.00),

    /**
     * Unreclaimed memory. RAM climbs steadily; CPU follows late as
     * allocation/GC pressure builds. Deliberately quiet until the end.
     */
    MEMORY_LEAK(0.45, 1.00, 0.05, 0.25, 0.90, 0.20, 0.10, 0.05, 1.60, 0.00),

    /**
     * Failing storage: rising I/O wait keeps tasks resident longer, which
     * inflates the queue, the transfer volume and moderately the latency.
     */
    DISK_IO_SATURATION(0.55, 0.35, 0.80, 0.40, 1.00, 0.50, 0.25, 0.35, 1.10, 0.00),

    /**
     * Network interface / cabling fault. Almost invisible on the compute
     * side; dominated by link latency, packet loss and bandwidth fade.
     */
    NIC_DEGRADATION(0.15, 0.05, 0.90, 0.10, 0.35, 1.00, 1.00, 1.00, 1.00, 0.00),

    /**
     * Ageing PSU / brown-out. Power is the leading indicator and it
     * OSCILLATES rather than ramping monotonically, so a first-difference
     * feature sees a very different pattern from the other mechanisms.
     */
    POWER_SUPPLY_INSTABILITY(0.25, 0.10, 0.15, 1.00, 0.30, 0.35, 0.30, 0.25, 0.90, 0.12);

    private final double cpuWeight;
    private final double ramWeight;
    private final double bandwidthWeight;
    private final double energyWeight;
    private final double taskWeight;
    private final double latencyWeight;
    private final double packetLossWeight;
    private final double bandwidthFadeWeight;
    private final double shapeExponent;
    private final double energyOscillation;

    HostFaultMode(double cpuWeight,
                  double ramWeight,
                  double bandwidthWeight,
                  double energyWeight,
                  double taskWeight,
                  double latencyWeight,
                  double packetLossWeight,
                  double bandwidthFadeWeight,
                  double shapeExponent,
                  double energyOscillation) {

        this.cpuWeight = cpuWeight;
        this.ramWeight = ramWeight;
        this.bandwidthWeight = bandwidthWeight;
        this.energyWeight = energyWeight;
        this.taskWeight = taskWeight;
        this.latencyWeight = latencyWeight;
        this.packetLossWeight = packetLossWeight;
        this.bandwidthFadeWeight = bandwidthFadeWeight;
        this.shapeExponent = shapeExponent;
        this.energyOscillation = energyOscillation;
    }

    /** Host CPU utilisation rise. */
    public double getCpuWeight() {
        return cpuWeight;
    }

    /** Host RAM utilisation rise. */
    public double getRamWeight() {
        return ramWeight;
    }

    /** Host bandwidth utilisation rise. */
    public double getBandwidthWeight() {
        return bandwidthWeight;
    }

    /** Host power draw rise. */
    public double getEnergyWeight() {
        return energyWeight;
    }

    /** Extra resident tasks (queue backing up). */
    public double getTaskWeight() {
        return taskWeight;
    }

    /** Uplink latency rise. */
    public double getLatencyWeight() {
        return latencyWeight;
    }

    /** Uplink packet loss rise. */
    public double getPacketLossWeight() {
        return packetLossWeight;
    }

    /** Usable uplink bandwidth fade. */
    public double getBandwidthFadeWeight() {
        return bandwidthFadeWeight;
    }

    /**
     * Curvature of the degradation trajectory. &lt;1 ramps early and
     * saturates; &gt;1 stays flat then accelerates near the end.
     */
    public double getShapeExponent() {
        return shapeExponent;
    }

    /** Amplitude of the non-monotonic power oscillation (fraction of base). */
    public double getEnergyOscillation() {
        return energyOscillation;
    }

    /** The mechanisms that can actually be drawn at fault onset. */
    public static HostFaultMode[] mechanisms() {
        return new HostFaultMode[]{
                THERMAL_THROTTLING,
                MEMORY_LEAK,
                DISK_IO_SATURATION,
                NIC_DEGRADATION,
                POWER_SUPPLY_INSTABILITY,
        };
    }
}
