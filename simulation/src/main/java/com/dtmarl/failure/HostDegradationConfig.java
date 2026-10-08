package com.dtmarl.failure;

/**
 * Tunable parameters for {@link HostDegradationManager}.
 *
 * All values are defaults-with-setters, matching the configuration style
 * already used by the other failure managers (see
 * {@link NetworkFailureManager#enableRandomLinkFailures(double)} and
 * {@link DeviceFailureManager#enableBatteryDrain(double)}) - nothing is
 * hardcoded inside the per-tick loop, and SimulationManager overrides only
 * what it needs.
 *
 * Units: "per tick" means per simulation clock tick, which the datacenter's
 * scheduling interval fixes at 1 simulated second.
 */
public class HostDegradationConfig {

    // ---- wear accumulation -------------------------------------------------

    /**
     * Background wear per tick with no fault mechanism active. Kept small on
     * purpose: over a full run it must NOT be able to reach the degradation
     * threshold on its own, otherwise perfectly healthy nodes would start
     * showing symptoms.
     */
    private double baseWearPerTick = 0.00015;

    /** Probability per tick that a healthy host develops a fault mechanism. */
    private double faultOnsetProbabilityPerTick = 0.0006;

    /** Wear per tick once a mechanism is active (scaled by severity/load). */
    private double episodeWearPerTick = 0.010;

    /**
     * Multiplier applied to the wear rate for "abrupt" episodes - a
     * deliberately configurable minority of failures that give almost no
     * warning (power spike, kernel panic). Keeping these in the dataset is
     * what stops the Bayes-optimal recall from being a fake 100%.
     */
    private double abruptWearMultiplier = 40.0;

    /** Fraction of episodes that are abrupt. */
    private double abruptFailureProbability = 0.10;

    /** Lognormal sigma of the per-tick wear increment (trajectory jitter). */
    private double wearNoiseSigma = 0.45;

    /** Per-episode severity range; scales both wear rate and symptom size. */
    private double severityMin = 0.5;
    private double severityMax = 2.0;

    /** Per-host permanent susceptibility range (hardware lottery). */
    private double susceptibilityMin = 0.6;
    private double susceptibilityMax = 1.8;

    // ---- health-state thresholds ------------------------------------------

    /** Wear at which symptoms become observable (NORMAL -> DEGRADING). */
    private double degradingWearThreshold = 0.25;

    /** Wear at which the host is considered CRITICAL. */
    private double criticalWearThreshold = 0.70;

    // ---- failure hazard ----------------------------------------------------

    /**
     * Weibull-like hazard: P(fail this tick) = scale * wear^shape. With
     * shape &gt; 1 the host is almost safe while lightly worn and increasingly
     * likely to die as wear approaches 1, which is what makes the remaining
     * time-to-failure inferable from the observed trajectory instead of
     * fixed. Wear reaching 1.0 always fails.
     */
    private double hazardScale = 0.010;
    private double hazardShape = 4.0;

    // ---- observable symptom amplitudes ------------------------------------

    /** Max added CPU utilisation, percentage points. */
    private double cpuRisePercent = 45.0;

    /** Max added RAM utilisation, percentage points. */
    private double ramRisePercent = 55.0;

    /** Max added host bandwidth utilisation, percentage points. */
    private double bandwidthUtilRisePercent = 25.0;

    /** Max power rise as a fraction of the nominal draw. */
    private double energyRiseFraction = 0.55;

    /** Max extra resident tasks from queue back-pressure. */
    private int maxTaskPressure = 6;

    /** Max uplink latency as a multiple of nominal. */
    private double latencyRiseFactor = 12.0;

    /** Max uplink packet loss, percent. */
    private double maxPacketLossPercent = 35.0;

    /** Max usable-bandwidth fade as a fraction of nominal. */
    private double bandwidthFadeFraction = 0.70;

    /** Relative measurement noise applied to every telemetry channel, percent. */
    private double telemetryNoisePercent = 2.5;

    // ---- recovery ---------------------------------------------------------

    /** Whether failed hosts are repaired and returned to service. */
    private boolean recoveryEnabled = true;

    /** Repair duration range, in ticks. */
    private int repairTicksMin = 12;
    private int repairTicksMax = 45;

    /**
     * Imperfect repair: fraction of accumulated wear that SURVIVES a repair.
     * 0 would be a factory-new replacement; 0.3 means a repaired host is
     * measurably more fragile than a fresh one, so repeat offenders emerge
     * naturally.
     */
    private double imperfectRepairRetention = 0.30;

    // ---- getters / setters ------------------------------------------------

    public double getBaseWearPerTick() {
        return baseWearPerTick;
    }

    public HostDegradationConfig setBaseWearPerTick(double v) {
        this.baseWearPerTick = v;
        return this;
    }

    public double getFaultOnsetProbabilityPerTick() {
        return faultOnsetProbabilityPerTick;
    }

    public HostDegradationConfig setFaultOnsetProbabilityPerTick(double v) {
        this.faultOnsetProbabilityPerTick = v;
        return this;
    }

    public double getEpisodeWearPerTick() {
        return episodeWearPerTick;
    }

    public HostDegradationConfig setEpisodeWearPerTick(double v) {
        this.episodeWearPerTick = v;
        return this;
    }

    public double getAbruptWearMultiplier() {
        return abruptWearMultiplier;
    }

    public HostDegradationConfig setAbruptWearMultiplier(double v) {
        this.abruptWearMultiplier = v;
        return this;
    }

    public double getAbruptFailureProbability() {
        return abruptFailureProbability;
    }

    public HostDegradationConfig setAbruptFailureProbability(double v) {
        this.abruptFailureProbability = v;
        return this;
    }

    public double getWearNoiseSigma() {
        return wearNoiseSigma;
    }

    public HostDegradationConfig setWearNoiseSigma(double v) {
        this.wearNoiseSigma = v;
        return this;
    }

    public double getSeverityMin() {
        return severityMin;
    }

    public double getSeverityMax() {
        return severityMax;
    }

    public HostDegradationConfig setSeverityRange(double min, double max) {
        this.severityMin = min;
        this.severityMax = max;
        return this;
    }

    public double getSusceptibilityMin() {
        return susceptibilityMin;
    }

    public double getSusceptibilityMax() {
        return susceptibilityMax;
    }

    public HostDegradationConfig setSusceptibilityRange(double min, double max) {
        this.susceptibilityMin = min;
        this.susceptibilityMax = max;
        return this;
    }

    public double getDegradingWearThreshold() {
        return degradingWearThreshold;
    }

    public HostDegradationConfig setDegradingWearThreshold(double v) {
        this.degradingWearThreshold = v;
        return this;
    }

    public double getCriticalWearThreshold() {
        return criticalWearThreshold;
    }

    public HostDegradationConfig setCriticalWearThreshold(double v) {
        this.criticalWearThreshold = v;
        return this;
    }

    public double getHazardScale() {
        return hazardScale;
    }

    public HostDegradationConfig setHazardScale(double v) {
        this.hazardScale = v;
        return this;
    }

    public double getHazardShape() {
        return hazardShape;
    }

    public HostDegradationConfig setHazardShape(double v) {
        this.hazardShape = v;
        return this;
    }

    public double getCpuRisePercent() {
        return cpuRisePercent;
    }

    public HostDegradationConfig setCpuRisePercent(double v) {
        this.cpuRisePercent = v;
        return this;
    }

    public double getRamRisePercent() {
        return ramRisePercent;
    }

    public HostDegradationConfig setRamRisePercent(double v) {
        this.ramRisePercent = v;
        return this;
    }

    public double getBandwidthUtilRisePercent() {
        return bandwidthUtilRisePercent;
    }

    public HostDegradationConfig setBandwidthUtilRisePercent(double v) {
        this.bandwidthUtilRisePercent = v;
        return this;
    }

    public double getEnergyRiseFraction() {
        return energyRiseFraction;
    }

    public HostDegradationConfig setEnergyRiseFraction(double v) {
        this.energyRiseFraction = v;
        return this;
    }

    public int getMaxTaskPressure() {
        return maxTaskPressure;
    }

    public HostDegradationConfig setMaxTaskPressure(int v) {
        this.maxTaskPressure = v;
        return this;
    }

    public double getLatencyRiseFactor() {
        return latencyRiseFactor;
    }

    public HostDegradationConfig setLatencyRiseFactor(double v) {
        this.latencyRiseFactor = v;
        return this;
    }

    public double getMaxPacketLossPercent() {
        return maxPacketLossPercent;
    }

    public HostDegradationConfig setMaxPacketLossPercent(double v) {
        this.maxPacketLossPercent = v;
        return this;
    }

    public double getBandwidthFadeFraction() {
        return bandwidthFadeFraction;
    }

    public HostDegradationConfig setBandwidthFadeFraction(double v) {
        this.bandwidthFadeFraction = v;
        return this;
    }

    public double getTelemetryNoisePercent() {
        return telemetryNoisePercent;
    }

    public HostDegradationConfig setTelemetryNoisePercent(double v) {
        this.telemetryNoisePercent = v;
        return this;
    }

    public boolean isRecoveryEnabled() {
        return recoveryEnabled;
    }

    public HostDegradationConfig setRecoveryEnabled(boolean v) {
        this.recoveryEnabled = v;
        return this;
    }

    public int getRepairTicksMin() {
        return repairTicksMin;
    }

    public int getRepairTicksMax() {
        return repairTicksMax;
    }

    public HostDegradationConfig setRepairTicksRange(int min, int max) {
        this.repairTicksMin = min;
        this.repairTicksMax = max;
        return this;
    }

    public double getImperfectRepairRetention() {
        return imperfectRepairRetention;
    }

    public HostDegradationConfig setImperfectRepairRetention(double v) {
        this.imperfectRepairRetention = v;
        return this;
    }
}
