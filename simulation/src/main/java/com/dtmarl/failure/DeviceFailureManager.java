package com.dtmarl.failure;

import com.dtmarl.ai.digitaltwin.DeviceNode;
import com.dtmarl.ai.digitaltwin.DigitalTwinManager;

import java.util.HashMap;
import java.util.HashSet;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Simulates the three IoMT device-layer failure modes from Sprint 3.75:
 *
 *   1) Battery depletion - continuous per-tick drain; device fails once
 *      it hits 0%. Drain rate is exposed as a tunable so it can later
 *      be calibrated against a real device-failure dataset instead of
 *      an arbitrary number (same pattern used for ClusterData
 *      calibration in Sprint 3).
 *   2) Dropout            - connectivity to the edge node is lost
 *      (scheduled or random), heartbeat stops updating.
 *   3) Sensor fault        - device stays "connected" at the network
 *      level but its physiological readings can no longer be trusted
 *      (scheduled or random).
 *
 * All three are treated as terminal for the device (same modeling
 * choice as FailureManager's HOST_FAILURE) — once triggered, a device
 * is done for the rest of the run. Events are logged into the SAME
 * unified FailureManager event log used by Sprints 3 and 3.5, so
 * Sprint 8/11 get one CSV covering compute + network + device events.
 */
public class DeviceFailureManager {

    private final DigitalTwinManager digitalTwin;
    private final FailureManager failureManager;

    private final Set<Integer> failedDevices = new HashSet<>();

    // Battery drain
    private boolean batteryDrainEnabled = false;
    private double batteryDrainRatePerTick = 0.0;

    // Dropout
    private final Map<Integer, Double> scheduledDropouts = new HashMap<>();
    private boolean randomDropoutsEnabled = false;
    private double randomDropoutProbabilityPerTick = 0.0;

    // Sensor fault
    private final Map<Integer, Double> scheduledSensorFaults = new HashMap<>();
    private boolean randomSensorFaultsEnabled = false;
    private double randomSensorFaultProbabilityPerTick = 0.0;

    private final Random random;

    public DeviceFailureManager(DigitalTwinManager digitalTwin, FailureManager failureManager) {
        this(digitalTwin, failureManager, 0L);
    }

    /** @param seed explicit RNG seed so a run is reproducible. */
    public DeviceFailureManager(DigitalTwinManager digitalTwin,
                                FailureManager failureManager,
                                long seed) {
        this.digitalTwin = digitalTwin;
        this.failureManager = failureManager;
        this.random = new Random(seed);
    }

    // ==========================================
    // Configuration
    // ==========================================

    /**
     * Enables continuous battery drain across all devices.
     * `ratePerTick` is a tunable hook: set this from a real device
     * telemetry dataset if/when you get one downloaded, instead of an
     * arbitrary constant — same calibration pattern as Sprint 3's
     * ClusterData-derived failure probability.
     */
    public void enableBatteryDrain(double ratePerTick) {
        this.batteryDrainEnabled = true;
        this.batteryDrainRatePerTick = ratePerTick;

        System.out.println(
                "Battery drain enabled at " + ratePerTick + " % per tick per device."
        );
    }

    public void scheduleDropout(int deviceId, double atTime) {
        scheduledDropouts.put(deviceId, atTime);

        System.out.println(
                "Scheduled device dropout: Device " + deviceId +
                " will drop out at t=" + atTime + "s"
        );
    }

    public void enableRandomDropouts(double probabilityPerTick) {
        this.randomDropoutsEnabled = true;
        this.randomDropoutProbabilityPerTick = probabilityPerTick;
    }

    public void scheduleSensorFault(int deviceId, double atTime) {
        scheduledSensorFaults.put(deviceId, atTime);

        System.out.println(
                "Scheduled sensor fault: Device " + deviceId +
                " will fault at t=" + atTime + "s"
        );
    }

    public void enableRandomSensorFaults(double probabilityPerTick) {
        this.randomSensorFaultsEnabled = true;
        this.randomSensorFaultProbabilityPerTick = probabilityPerTick;
    }

    // ==========================================
    // Per-tick checks
    // ==========================================

    public void checkAndTrigger(double currentTime) {

        for (DeviceNode device : digitalTwin.getDevices()) {

            int deviceId = device.getId();

            if (failedDevices.contains(deviceId)) {
                continue;
            }

            if (batteryDrainEnabled) {

                double newLevel = device.getBatteryLevel() - batteryDrainRatePerTick;
                device.setBatteryLevel(Math.max(newLevel, 0));

                if (device.getBatteryLevel() <= 0) {
                    triggerFailure(device, currentTime,
                            FailureEvent.Type.DEVICE_BATTERY_DEPLETED,
                            FailureEvent.Cause.SYSTEM,
                            DeviceNode.FailureMode.BATTERY_DEPLETED);
                    continue;
                }
            }

            Double scheduledDropout = scheduledDropouts.get(deviceId);

            if (scheduledDropout != null && currentTime >= scheduledDropout) {
                triggerFailure(device, currentTime,
                        FailureEvent.Type.DEVICE_DROPOUT,
                        FailureEvent.Cause.SCHEDULED,
                        DeviceNode.FailureMode.DROPOUT);
                continue;
            }

            if (randomDropoutsEnabled &&
                    random.nextDouble() < randomDropoutProbabilityPerTick) {

                triggerFailure(device, currentTime,
                        FailureEvent.Type.DEVICE_DROPOUT,
                        FailureEvent.Cause.RANDOM,
                        DeviceNode.FailureMode.DROPOUT);
                continue;
            }

            Double scheduledFault = scheduledSensorFaults.get(deviceId);

            if (scheduledFault != null && currentTime >= scheduledFault) {
                triggerFailure(device, currentTime,
                        FailureEvent.Type.DEVICE_SENSOR_FAULT,
                        FailureEvent.Cause.SCHEDULED,
                        DeviceNode.FailureMode.SENSOR_FAULT);
                continue;
            }

            if (randomSensorFaultsEnabled &&
                    random.nextDouble() < randomSensorFaultProbabilityPerTick) {

                triggerFailure(device, currentTime,
                        FailureEvent.Type.DEVICE_SENSOR_FAULT,
                        FailureEvent.Cause.RANDOM,
                        DeviceNode.FailureMode.SENSOR_FAULT);
                continue;
            }

            // Healthy tick: heartbeat updates, signal jitters mildly
            // around baseline to look like real sensor noise.
            device.setLastHeartbeatTime(currentTime);

            double jitter = (random.nextDouble() - 0.5) * 4; // +/-2%
            device.setSignalQuality(clamp(device.getSignalQuality() + jitter, 0, 100));
            device.setSignalNoise(clamp(100 - device.getSignalQuality(), 0, 100));
        }
    }

    private void triggerFailure(DeviceNode device, double currentTime,
                                 FailureEvent.Type type, FailureEvent.Cause cause,
                                 DeviceNode.FailureMode mode) {

        device.setActive(false);
        device.setConnected(false);
        device.setFailureMode(mode);

        failedDevices.add(device.getId());

        FailureEvent event = new FailureEvent(
                device.getId(), currentTime, type, cause, device.getBatteryLevel()
        );

        failureManager.logEvent(event);

        System.out.println("\n@@@ DEVICE FAILURE @@@ " + event +
                " [mode=" + mode + "]\n");
    }

    private double clamp(double value, double min, double max) {
        return Math.max(min, Math.min(max, value));
    }

    public boolean hasFailed(int deviceId) {
        return failedDevices.contains(deviceId);
    }

    public Set<Integer> getFailedDevices() {
        return failedDevices;
    }
}