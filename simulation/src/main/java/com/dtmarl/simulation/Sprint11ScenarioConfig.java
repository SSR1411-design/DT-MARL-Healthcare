package com.dtmarl.simulation;

import com.dtmarl.failure.HostDegradationConfig;

import java.io.IOException;
import java.io.Reader;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Properties;

/**
 * Immutable physical configuration for an isolated Sprint 11 simulation.
 *
 * <p>The no-argument simulator still uses the historical R2 values.  A
 * properties-backed configuration is opt-in and writes its outputs only into
 * the configured new trial directory.  This class deliberately configures
 * {@link HostDegradationConfig}; it never enables the legacy random-host
 * failure API.</p>
 */
public final class Sprint11ScenarioConfig {

    public static final String SCHEMA_VERSION = "sprint11.v1";
    public static final String FAILURE_MODE = "progressive_stochastic_wear";

    private final long simulatorSeed;
    private final int hostCount;
    private final int edgeNodeCount;
    private final int deviceCount;
    private final int taskCount;
    private final int patientCount;
    private final double bandwidthMbps;
    private final double failureIntensity;
    private final double maxSimulationSeconds;
    private final Path outputDirectory;
    private final boolean requireEmptyOutputDirectory;
    private final boolean allowLegacyDefaultRiskExport;
    private final HostDegradationConfig degradationConfig;

    private Sprint11ScenarioConfig(long simulatorSeed, int hostCount, int edgeNodeCount,
                                   int deviceCount, int taskCount, int patientCount,
                                   double bandwidthMbps, double failureIntensity,
                                   double maxSimulationSeconds, Path outputDirectory,
                                   boolean requireEmptyOutputDirectory,
                                   boolean allowLegacyDefaultRiskExport,
                                   HostDegradationConfig degradationConfig) {
        this.simulatorSeed = simulatorSeed;
        this.hostCount = hostCount;
        this.edgeNodeCount = edgeNodeCount;
        this.deviceCount = deviceCount;
        this.taskCount = taskCount;
        this.patientCount = patientCount;
        this.bandwidthMbps = bandwidthMbps;
        this.failureIntensity = failureIntensity;
        this.maxSimulationSeconds = maxSimulationSeconds;
        this.outputDirectory = outputDirectory;
        this.requireEmptyOutputDirectory = requireEmptyOutputDirectory;
        this.allowLegacyDefaultRiskExport = allowLegacyDefaultRiskExport;
        this.degradationConfig = degradationConfig;
        validate();
    }

    /** Exact configuration used by the existing R2 simulator path. */
    public static Sprint11ScenarioConfig r2Defaults() {
        return new Sprint11ScenarioConfig(
                20260817L, 10, 10, 10, 40, 10, 100.0, 1.0, 1500.0,
                Path.of("."), false, true, degradationDefaults());
    }

    /** Load a generated Sprint 11 property file; no defaulting is permitted. */
    public static Sprint11ScenarioConfig fromProperties(Path path) throws IOException {
        Properties properties = new Properties();
        try (Reader reader = Files.newBufferedReader(path)) {
            properties.load(reader);
        }
        requireEquals(properties, "schema_version", SCHEMA_VERSION);
        requireEquals(properties, "failure_mode", FAILURE_MODE);
        String seedText = required(properties, "simulator_seed");
        if ("TRIAL_SEED_REQUIRED".equals(seedText)) {
            throw new IllegalArgumentException("scenario template is not runnable: simulator_seed is required");
        }
        String outputText = required(properties, "output_directory");
        if (outputText.contains("TRIAL_OUTPUT_REQUIRED")) {
            throw new IllegalArgumentException("scenario template is not runnable: trial output directory is required");
        }
        // A newly generated trace cannot already have OOF predictions aligned
        // to it.  Sprint 11 trace generation therefore uses neutral Java-side
        // task predictions rather than silently loading simulation/predicted_risk.csv.
        String riskStatus = properties.getProperty("risk_alignment_status", "unavailable").trim();
        if (!"unavailable".equals(riskStatus)) {
            throw new IllegalArgumentException(
                    "Sprint 11 trace generation requires risk_alignment_status=unavailable; "
                    + "run aligned OOF regeneration after the trace exists");
        }
        HostDegradationConfig degradation = new HostDegradationConfig()
                .setFaultOnsetProbabilityPerTick(doubleValue(properties, "failure.fault_onset_probability_per_tick"))
                .setBaseWearPerTick(doubleValue(properties, "failure.background_wear_per_tick"))
                .setEpisodeWearPerTick(doubleValue(properties, "failure.fault_episode_wear_per_tick"))
                .setWearNoiseSigma(doubleValue(properties, "failure.wear_noise_sigma"))
                .setSusceptibilityRange(doubleValue(properties, "failure.susceptibility_min"),
                                        doubleValue(properties, "failure.susceptibility_max"))
                .setSeverityRange(doubleValue(properties, "failure.severity_min"),
                                  doubleValue(properties, "failure.severity_max"))
                .setHazardScale(doubleValue(properties, "failure.hazard_scale"))
                .setHazardShape(doubleValue(properties, "failure.hazard_shape"))
                .setAbruptFailureProbability(doubleValue(properties, "failure.abrupt_failure_probability"))
                .setAbruptWearMultiplier(doubleValue(properties, "failure.abrupt_wear_multiplier"))
                .setDegradingWearThreshold(doubleValue(properties, "failure.degrading_wear_threshold"))
                .setCriticalWearThreshold(doubleValue(properties, "failure.critical_wear_threshold"))
                .setRepairTicksRange(intValue(properties, "failure.repair_ticks_min"),
                                     intValue(properties, "failure.repair_ticks_max"))
                .setImperfectRepairRetention(doubleValue(properties, "failure.imperfect_repair_retention"));
        return new Sprint11ScenarioConfig(
                longValue(properties, "simulator_seed"),
                intValue(properties, "host_count"), intValue(properties, "edge_node_count"),
                intValue(properties, "device_count"), intValue(properties, "task_count"),
                intValue(properties, "patient_count"), doubleValue(properties, "bandwidth_mbps"),
                doubleValue(properties, "failure_intensity"), 1500.0,
                Path.of(outputText).toAbsolutePath().normalize(), true, false, degradation);
    }

    private static HostDegradationConfig degradationDefaults() {
        return new HostDegradationConfig()
                .setFaultOnsetProbabilityPerTick(0.0018)
                .setEpisodeWearPerTick(0.010)
                .setBaseWearPerTick(0.00015)
                .setWearNoiseSigma(0.45)
                .setSeverityRange(0.5, 2.0)
                .setSusceptibilityRange(0.6, 1.8)
                .setDegradingWearThreshold(0.25)
                .setCriticalWearThreshold(0.70)
                .setHazardScale(0.010)
                .setHazardShape(4.0)
                .setAbruptFailureProbability(0.15)
                .setAbruptWearMultiplier(40.0)
                .setTelemetryNoisePercent(2.5)
                .setRecoveryEnabled(true)
                .setRepairTicksRange(12, 45)
                .setImperfectRepairRetention(0.30);
    }

    private void validate() {
        if (simulatorSeed <= 0 || hostCount <= 0 || edgeNodeCount <= 0 || deviceCount <= 0
                || taskCount <= 0 || patientCount <= 0 || patientCount > taskCount) {
            throw new IllegalArgumentException("scenario counts/seed are invalid");
        }
        if (hostCount != edgeNodeCount) {
            throw new IllegalArgumentException("current simulator requires host_count == edge_node_count");
        }
        if (!Double.isFinite(bandwidthMbps) || bandwidthMbps <= 0
                || !Double.isFinite(failureIntensity) || failureIntensity <= 0) {
            throw new IllegalArgumentException("scenario bandwidth/failure intensity are invalid");
        }
        if (degradationConfig.getFaultOnsetProbabilityPerTick() < 0
                || degradationConfig.getFaultOnsetProbabilityPerTick() > 1
                || degradationConfig.getAbruptFailureProbability() < 0
                || degradationConfig.getAbruptFailureProbability() > 1
                || degradationConfig.getSusceptibilityMin() <= 0
                || degradationConfig.getSusceptibilityMin() > degradationConfig.getSusceptibilityMax()
                || degradationConfig.getSeverityMin() <= 0
                || degradationConfig.getSeverityMin() > degradationConfig.getSeverityMax()
                || degradationConfig.getHazardScale() < 0
                || degradationConfig.getHazardShape() <= 0
                || degradationConfig.getRepairTicksMin() <= 0
                || degradationConfig.getRepairTicksMin() > degradationConfig.getRepairTicksMax()) {
            throw new IllegalArgumentException("progressive degradation parameters are invalid");
        }
    }

    /** Create the isolated output directory and reject any prior trace files. */
    public void prepareOutputDirectory() throws IOException {
        if (!requireEmptyOutputDirectory) {
            return;
        }
        Files.createDirectories(outputDirectory);
        for (String filename : new String[]{"failure_log.csv", "failure_history.csv", "device_failure_history.csv"}) {
            if (Files.exists(outputDirectory.resolve(filename))) {
                throw new IOException("refusing to overwrite existing Sprint 11 output: "
                        + outputDirectory.resolve(filename));
            }
        }
    }

    public long getSimulatorSeed() { return simulatorSeed; }
    public int getHostCount() { return hostCount; }
    public int getEdgeNodeCount() { return edgeNodeCount; }
    public int getDeviceCount() { return deviceCount; }
    public int getTaskCount() { return taskCount; }
    public int getPatientCount() { return patientCount; }
    public double getBandwidthMbps() { return bandwidthMbps; }
    public double getFailureIntensity() { return failureIntensity; }
    public double getMaxSimulationSeconds() { return maxSimulationSeconds; }
    public Path getOutputDirectory() { return outputDirectory; }
    public boolean allowsLegacyDefaultRiskExport() { return allowLegacyDefaultRiskExport; }
    public HostDegradationConfig getDegradationConfig() { return degradationConfig; }

    private static String required(Properties properties, String key) {
        String value = properties.getProperty(key);
        if (value == null || value.trim().isEmpty()) {
            throw new IllegalArgumentException("missing required scenario property: " + key);
        }
        return value.trim();
    }

    private static void requireEquals(Properties properties, String key, String expected) {
        if (!expected.equals(required(properties, key))) {
            throw new IllegalArgumentException("unsupported " + key);
        }
    }

    private static int intValue(Properties properties, String key) {
        return Integer.parseInt(required(properties, key));
    }

    private static long longValue(Properties properties, String key) {
        return Long.parseLong(required(properties, key));
    }

    private static double doubleValue(Properties properties, String key) {
        double value = Double.parseDouble(required(properties, key));
        if (!Double.isFinite(value)) {
            throw new IllegalArgumentException("non-finite property: " + key);
        }
        return value;
    }
}
