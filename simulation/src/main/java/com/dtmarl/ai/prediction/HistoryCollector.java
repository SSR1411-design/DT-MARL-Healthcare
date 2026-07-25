package com.dtmarl.ai.prediction;

import com.dtmarl.ai.digitaltwin.DigitalTwinManager;
import com.dtmarl.ai.digitaltwin.EdgeNode;
import com.dtmarl.ai.digitaltwin.NetworkLink;
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
 * Collects a rolling window of TelemetrySnapshots per node (Sprint 4:
 * "Historical state collection"), and exports a labeled training CSV
 * where each row is tagged with whether the node fails within the
 * next `lookaheadSeconds` after that row. That label is what the
 * failure prediction model (Python side) learns to predict.
 */
public class HistoryCollector {

    private final DigitalTwinManager digitalTwin;
    private final FailureManager failureManager;

    private final int windowSize;

    // Rolling window per node, most-recent-last.
    private final Map<Integer, Deque<TelemetrySnapshot>> windows = new HashMap<>();

    // Every snapshot ever taken, kept so we can label it retroactively
    // once we know whether a failure happened within the lookahead.
    private final List<TelemetrySnapshot> allSnapshots = new ArrayList<>();

    public HistoryCollector(DigitalTwinManager digitalTwin,
                             FailureManager failureManager,
                             int windowSize) {

        this.digitalTwin = digitalTwin;
        this.failureManager = failureManager;
        this.windowSize = windowSize;
    }

    /**
     * Call once per simulation tick, after digitalTwin.syncWithHosts()
     * and after both failure managers have run their checks.
     */
    public void collect(double currentTime) {

        for (EdgeNode node : digitalTwin.getNodes()) {

            NetworkLink link = digitalTwin.getLink(node.getId());

            TelemetrySnapshot snap = new TelemetrySnapshot(
                    currentTime, node.getId(),
                    node.getCpuUsage(), node.getRamUsage(),
                    node.getBandwidthUsage(), node.getEnergyConsumption(),
                    node.getRunningTasks(), node.isActive(), node.isDegraded(),
                    link != null && link.isUp(),
                    link != null ? link.getBandwidthMbps() : 0,
                    link != null ? link.getLatencyMs() : 0,
                    link != null ? link.getPacketLossPercent() : 0,
                    link != null && link.isUnderAttack()
            );

            allSnapshots.add(snap);

            windows.computeIfAbsent(node.getId(), k -> new ArrayDeque<>())
                    .addLast(snap);

            Deque<TelemetrySnapshot> w = windows.get(node.getId());

            if (w.size() > windowSize) {
                w.removeFirst();
            }
        }
    }

    /**
     * Current rolling window for a node, oldest-first. Size may be less
     * than windowSize early in the run.
     */
    public List<TelemetrySnapshot> getWindow(int nodeId) {
        return new ArrayList<>(windows.getOrDefault(nodeId, new ArrayDeque<>()));
    }

    /**
     * Exports every collected snapshot with a binary "willFailSoon"
     * label: 1 if a HOST_FAILURE or NETWORK_FAILURE event for that
     * node occurs within `lookaheadSeconds` after this snapshot's time,
     * else 0. This is the supervised label Sprint 4's model trains on.
     */
    public void exportLabeledCsv(String filePath, double lookaheadSeconds) {

        // Collect failure times per node for quick lookup.
        Map<Integer, List<Double>> failureTimesByNode = new HashMap<>();

        for (FailureEvent event : failureManager.getEventLog()) {

            boolean isFailure =
                    event.getType() == FailureEvent.Type.HOST_FAILURE
                    || event.getType() == FailureEvent.Type.NETWORK_FAILURE;

            if (isFailure) {
                failureTimesByNode
                        .computeIfAbsent(event.getNodeId(), k -> new ArrayList<>())
                        .add(event.getTime());
            }
        }

        try (FileWriter writer = new FileWriter(filePath)) {

            writer.write(TelemetrySnapshot.csvHeader() + ",willFailSoon\n");

            for (TelemetrySnapshot snap : allSnapshots) {

                List<Double> failTimes =
                        failureTimesByNode.getOrDefault(snap.nodeId, new ArrayList<>());

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
                    "\nLabeled failure-prediction dataset exported to: " + filePath +
                    " (" + allSnapshots.size() + " rows, lookahead=" + lookaheadSeconds + "s)"
            );

        } catch (IOException e) {
            System.out.println("Failed to export labeled history: " + e.getMessage());
        }
    }
}