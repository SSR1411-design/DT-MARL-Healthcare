package com.dtmarl.ai.prediction;

import com.dtmarl.ai.digitaltwin.DeviceNode;
import com.dtmarl.ai.digitaltwin.DigitalTwinManager;
import com.dtmarl.failure.FailureEvent;
import com.dtmarl.failure.FailureManager;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayDeque;
import java.util.ArrayList;
import java.util.Deque;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

/**
 * Device-layer counterpart to HistoryCollector (Sprint 4). Collects a
 * rolling window of DeviceTelemetrySnapshots per device, and exports a
 * labeled CSV where each row is tagged with whether that device fails
 * (dropout, battery depletion, or sensor fault) within the next
 * `lookaheadSeconds`.
 */
public class DeviceHistoryCollector {

    private final DigitalTwinManager digitalTwin;
    private final FailureManager failureManager;

    private final int windowSize;

    private final Map<Integer, Deque<DeviceTelemetrySnapshot>> windows = new HashMap<>();
    private final List<DeviceTelemetrySnapshot> allSnapshots = new ArrayList<>();

    public DeviceHistoryCollector(DigitalTwinManager digitalTwin,
                                   FailureManager failureManager,
                                   int windowSize) {

        this.digitalTwin = digitalTwin;
        this.failureManager = failureManager;
        this.windowSize = windowSize;
    }

    public void collect(double currentTime) {

        for (DeviceNode device : digitalTwin.getDevices()) {

            DeviceTelemetrySnapshot snap = new DeviceTelemetrySnapshot(
                    currentTime, device.getId(),
                    device.getBatteryLevel(), device.isConnected(),
                    device.isActive(), device.getSignalQuality(),
                    device.getSignalNoise(), device.heartbeatStaleness(currentTime)
            );

            allSnapshots.add(snap);

            windows.computeIfAbsent(device.getId(), k -> new ArrayDeque<>())
                    .addLast(snap);

            Deque<DeviceTelemetrySnapshot> w = windows.get(device.getId());

            if (w.size() > windowSize) {
                w.removeFirst();
            }
        }
    }

    public List<DeviceTelemetrySnapshot> getWindow(int deviceId) {
        return new ArrayList<>(windows.getOrDefault(deviceId, new ArrayDeque<>()));
    }

    /**
     * Labels each row 1 if a DEVICE_DROPOUT, DEVICE_BATTERY_DEPLETED,
     * or DEVICE_SENSOR_FAULT event for that device occurs within
     * `lookaheadSeconds` after the snapshot's time.
     */
    public void exportLabeledCsv(String filePath, double lookaheadSeconds) {

        Map<Integer, List<Double>> failureTimesByDevice = new HashMap<>();

        for (FailureEvent event : failureManager.getEventLog()) {

            boolean isDeviceFailure =
                    event.getType() == FailureEvent.Type.DEVICE_DROPOUT
                    || event.getType() == FailureEvent.Type.DEVICE_BATTERY_DEPLETED
                    || event.getType() == FailureEvent.Type.DEVICE_SENSOR_FAULT;

            if (isDeviceFailure) {
                failureTimesByDevice
                        .computeIfAbsent(event.getNodeId(), k -> new ArrayList<>())
                        .add(event.getTime());
            }
        }

        try (FileWriter writer = new FileWriter(filePath)) {

            writer.write(DeviceTelemetrySnapshot.csvHeader() + ",willFailSoon\n");

            for (DeviceTelemetrySnapshot snap : allSnapshots) {

                List<Double> failTimes =
                        failureTimesByDevice.getOrDefault(snap.deviceId, new ArrayList<>());

                boolean willFailSoon = false;

                for (double t : failTimes) {
                    if (t > snap.time && t <= snap.time + lookaheadSeconds) {
                        willFailSoon = true;
                        break;
                    }
                }

                writer.write(snap.toCsvRow() + "," + (willFailSoon ? 1 : 0) + "\n");
            }

            System.out.println(
                    "\nLabeled device-failure dataset exported to: " + filePath +
                    " (" + allSnapshots.size() + " rows, lookahead=" + lookaheadSeconds + "s)"
            );

        } catch (IOException e) {
            System.out.println("Failed to export device history: " + e.getMessage());
        }
    }
}