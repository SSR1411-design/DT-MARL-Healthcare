package com.dtmarl.ai.prediction;

import com.dtmarl.ai.digitaltwin.DigitalTwinManager;
import com.dtmarl.ai.digitaltwin.EdgeNode;
import com.dtmarl.ai.digitaltwin.NetworkLink;
import com.dtmarl.failure.FailureEvent;
import com.dtmarl.failure.FailureManager;
import com.dtmarl.failure.HostDegradationManager;

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

    /**
     * Optional source of LATENT host health, used only to fill the audit_*
     * columns of the export (see {@link #setHealthAuditSource}). Null-safe:
     * without it the audit columns are written as unknown/-1 and the
     * observable columns plus the label are unaffected.
     */
    private HostDegradationManager healthAuditSource;

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
     * Attaches the latent-wear source used for the audit_* export columns.
     *
     * This exists so the generated dataset can be VERIFIED (does degradation
     * really precede failure? do trajectories differ?) without those latent
     * values ever entering the observable feature block. The Python pipeline
     * selects features by an explicit name whitelist and the audit_ prefix
     * keeps them out of it.
     */
    public void setHealthAuditSource(HostDegradationManager source) {
        this.healthAuditSource = source;
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
                    link != null && link.isUnderAttack(),
                    node.getHealthState(),
                    healthAuditSource != null
                            ? healthAuditSource.getWear(node.getId())
                            : -1.0
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
     * Exports every collected snapshot with a binary "willFailSoon" label.
     *
     * ------------------------------------------------------------------
     * LABEL DEFINITION (this is the ONLY place labels are produced)
     * ------------------------------------------------------------------
     * For a snapshot of node n at time t:
     *
     *   willFailSoon = 1  iff  there exists a HOST_FAILURE or
     *                          NETWORK_FAILURE event for node n at time f
     *                          with  t &lt; f &lt;= t + lookaheadSeconds
     *   willFailSoon = 0  otherwise
     *
     * Properties that make this a legitimate supervised target:
     *
     *   * It is computed RETROSPECTIVELY, after the simulation has finished,
     *     from the event log - never during feature generation.
     *   * It shares no code, no random draw and no parameter with the
     *     degradation model that produced the features. Nothing in
     *     HostDegradationManager can observe it.
     *   * The failure instants it reads were themselves decided by a hazard
     *     draw on the tick they occurred, so no "scheduled failure time"
     *     existed at feature-generation time that could have been encoded.
     *   * The comparison is strict on the left (t &lt; f), so the failure tick
     *     itself and every tick after it are labelled 0 - a post-mortem row
     *     is not a prediction opportunity.
     *
     * The label is therefore predictable ONLY to the extent that the
     * simulated telemetry genuinely trends towards failure, which is exactly
     * what is being tested.
     *
     * ------------------------------------------------------------------
     * COLUMNS
     * ------------------------------------------------------------------
     * 1..14  observable telemetry (TelemetrySnapshot.csvHeader())
     * 15     willFailSoon                 - the target
     * 16+    audit_* - LATENT ground truth for validation and plotting only.
     *        These describe the simulator's internal state, including the
     *        realised failure time, and MUST NOT be used as model inputs.
     *        The audit_ prefix keeps them out of the Python feature
     *        whitelists by construction.
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

        int positives = 0;

        try (FileWriter writer = new FileWriter(filePath)) {

            writer.write(TelemetrySnapshot.csvHeader() + ",willFailSoon"
                    + ",audit_healthState,audit_wear,audit_nextFailureTime"
                    + ",audit_secondsToFailure,audit_predictionHorizon\n");

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

                if (willFailSoon) {
                    positives++;
                }

                // ---- audit block (not features) --------------------------
                // Next failure of this node strictly after t, at any distance.
                double nextFailure = -1;

                for (double t : failTimes) {
                    if (t > snap.time && (nextFailure < 0 || t < nextFailure)) {
                        nextFailure = t;
                    }
                }

                double secondsToFailure = nextFailure < 0
                        ? -1
                        : nextFailure - snap.time;

                writer.write(String.format(
                        "%s,%d,%s,%.4f,%.2f,%.2f,%.2f",
                        snap.toCsvRow(),
                        willFailSoon ? 1 : 0,
                        snap.healthState,
                        snap.wear,
                        nextFailure,
                        secondsToFailure,
                        lookaheadSeconds) + "\n");
            }

            System.out.println(
                    "\nLabeled failure-prediction dataset exported to: " + filePath +
                    " (" + allSnapshots.size() + " rows, lookahead=" + lookaheadSeconds + "s, "
                    + positives + " positive / " + (allSnapshots.size() - positives)
                    + " negative)"
            );

        } catch (IOException e) {
            System.out.println("Failed to export labeled history: " + e.getMessage());
        }
    }
}