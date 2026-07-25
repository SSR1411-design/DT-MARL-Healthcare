package com.dtmarl.failure;

import com.dtmarl.ai.digitaltwin.DigitalTwinManager;
import com.dtmarl.ai.digitaltwin.NetworkLink;

import java.util.HashMap;
import java.util.Map;
import java.util.Random;

/**
 * Simulates communication-layer conditions independent of host compute
 * failures (Sprint 3):
 *
 *   1) Full link failure  - communication with a node is lost entirely
 *      (bandwidth -> 0, latency -> very high, packet loss -> 100%)
 *   2) Link degradation    - temporary latency spike / packet loss,
 *      short of a full outage
 *   3) Cyber-attack        - a distinct anomaly signature: simultaneous
 *      abnormal bandwidth SPIKE (e.g. flood traffic) + high packet loss,
 *      which is what makes it distinguishable from (1) a clean hardware
 *      failure (everything flatlines to zero) and from (2) natural
 *      degradation (single-metric, gradual, tied to real conditions).
 *
 * All events are pushed into the same FailureManager event log, so
 * Sprint 8/11 can export one unified CSV across compute + network +
 * attack events for evaluation.
 */
public class NetworkFailureManager {

    private final DigitalTwinManager digitalTwin;
    private final FailureManager failureManager;

    private final Map<Integer, Double> scheduledLinkFailures = new HashMap<>();
    private final Map<Integer, double[]> scheduledDegradations = new HashMap<>(); // [startTime, duration]
    private final Map<Integer, double[]> scheduledAttacks = new HashMap<>();      // [startTime, duration]

    private final Map<Integer, Boolean> degradedActive = new HashMap<>();
    private final Map<Integer, Boolean> attackActive = new HashMap<>();

    private boolean randomLinkFailuresEnabled = false;
    private double randomLinkFailureProbabilityPerTick = 0.0;

    private final Random random = new Random();

    public NetworkFailureManager(DigitalTwinManager digitalTwin, FailureManager failureManager) {
        this.digitalTwin = digitalTwin;
        this.failureManager = failureManager;
    }

    // ==========================================
    // Configuration
    // ==========================================

    public void scheduleLinkFailure(int nodeId, double atTime) {
        scheduledLinkFailures.put(nodeId, atTime);

        System.out.println(
                "Scheduled network failure: Node " + nodeId +
                " link will fail at t=" + atTime + "s"
        );
    }

    public void scheduleLinkDegradation(int nodeId, double atTime, double durationSeconds) {
        scheduledDegradations.put(nodeId, new double[]{atTime, durationSeconds});

        System.out.println(
                "Scheduled link degradation: Node " + nodeId +
                " degraded from t=" + atTime + "s for " + durationSeconds + "s"
        );
    }

    public void scheduleCyberAttack(int nodeId, double atTime, double durationSeconds) {
        scheduledAttacks.put(nodeId, new double[]{atTime, durationSeconds});

        System.out.println(
                "Scheduled cyber-attack: Node " + nodeId +
                " attacked from t=" + atTime + "s for " + durationSeconds + "s"
        );
    }

    public void enableRandomLinkFailures(double probabilityPerTick) {
        this.randomLinkFailuresEnabled = true;
        this.randomLinkFailureProbabilityPerTick = probabilityPerTick;
    }

    // ==========================================
    // Per-tick checks
    // ==========================================

    public void checkAndTrigger(double currentTime) {

        for (NetworkLink link : digitalTwin.getLinks()) {

            int nodeId = link.getNodeId();

            if (!link.isUp()) {
                continue; // already failed, nothing more to check for this node
            }

            // Full link failure (scheduled)
            Double scheduledFailTime = scheduledLinkFailures.get(nodeId);

            if (scheduledFailTime != null && currentTime >= scheduledFailTime) {
                triggerLinkFailure(link, currentTime, FailureEvent.Cause.SCHEDULED);
                continue;
            }

            // Full link failure (random)
            if (randomLinkFailuresEnabled &&
                    random.nextDouble() < randomLinkFailureProbabilityPerTick) {

                triggerLinkFailure(link, currentTime, FailureEvent.Cause.RANDOM);
                continue;
            }

            // Degradation window
            handleDegradation(link, nodeId, currentTime);

            // Cyber-attack window
            handleAttack(link, nodeId, currentTime);
        }
    }

    private void triggerLinkFailure(NetworkLink link, double currentTime, FailureEvent.Cause cause) {

        link.setUp(false);
        link.setBandwidthMbps(0);
        link.setLatencyMs(9999);
        link.setPacketLossPercent(100);
        link.setUnderAttack(false);

        FailureEvent event = new FailureEvent(
                link.getNodeId(), currentTime,
                FailureEvent.Type.NETWORK_FAILURE, cause, 0
        );

        failureManager.logEvent(event);

        System.out.println("\n### NETWORK FAILURE ### " + event + "\n");
    }

    private void handleDegradation(NetworkLink link, int nodeId, double currentTime) {

        double[] window = scheduledDegradations.get(nodeId);

        if (window == null) {
            return;
        }

        double startTime = window[0];
        double endTime = window[0] + window[1];

        boolean shouldBeDegraded = currentTime >= startTime && currentTime < endTime;
        boolean isCurrentlyDegraded = degradedActive.getOrDefault(nodeId, false);

        if (shouldBeDegraded && !isCurrentlyDegraded) {

            link.setLatencyMs(NetworkLink.NORMAL_LATENCY_MS * 3);
            link.setPacketLossPercent(20);

            degradedActive.put(nodeId, true);

            FailureEvent event = new FailureEvent(
                    nodeId, currentTime,
                    FailureEvent.Type.LINK_DEGRADED_START, FailureEvent.Cause.SYSTEM, 0
            );

            failureManager.logEvent(event);

            System.out.println("\n~~~ LINK DEGRADED ~~~ " + event + "\n");

        } else if (!shouldBeDegraded && isCurrentlyDegraded) {

            link.setLatencyMs(NetworkLink.NORMAL_LATENCY_MS);
            link.setPacketLossPercent(NetworkLink.NORMAL_PACKET_LOSS_PERCENT);

            degradedActive.put(nodeId, false);

            FailureEvent event = new FailureEvent(
                    nodeId, currentTime,
                    FailureEvent.Type.LINK_DEGRADED_END, FailureEvent.Cause.SYSTEM, 0
            );

            failureManager.logEvent(event);

            System.out.println("\n~~~ LINK RECOVERED ~~~ " + event + "\n");
        }
    }

    private void handleAttack(NetworkLink link, int nodeId, double currentTime) {

        double[] window = scheduledAttacks.get(nodeId);

        if (window == null) {
            return;
        }

        double startTime = window[0];
        double endTime = window[0] + window[1];

        boolean shouldBeAttacked = currentTime >= startTime && currentTime < endTime;
        boolean isCurrentlyAttacked = attackActive.getOrDefault(nodeId, false);

        if (shouldBeAttacked && !isCurrentlyAttacked) {

            // Attack signature: abnormal bandwidth SPIKE (flood traffic)
            // combined with high packet loss at the same time. This
            // simultaneous spike+loss pattern is what should later let
            // Sprint 4's prediction model tell this apart from a clean
            // hardware failure (both flatline) or normal degradation
            // (single metric, no spike).
            link.setBandwidthMbps(NetworkLink.NORMAL_BANDWIDTH_MBPS * 5);
            link.setLatencyMs(NetworkLink.NORMAL_LATENCY_MS * 8);
            link.setPacketLossPercent(40);
            link.setUnderAttack(true);

            attackActive.put(nodeId, true);

            FailureEvent event = new FailureEvent(
                    nodeId, currentTime,
                    FailureEvent.Type.CYBER_ATTACK_START, FailureEvent.Cause.ATTACK, 0
            );

            failureManager.logEvent(event);

            System.out.println("\n!!! CYBER ATTACK DETECTED !!! " + event + "\n");

        } else if (!shouldBeAttacked && isCurrentlyAttacked) {

            link.resetToNormal();

            attackActive.put(nodeId, false);

            FailureEvent event = new FailureEvent(
                    nodeId, currentTime,
                    FailureEvent.Type.CYBER_ATTACK_END, FailureEvent.Cause.ATTACK, 0
            );

            failureManager.logEvent(event);

            System.out.println("\n!!! ATTACK ENDED !!! " + event + "\n");
        }
    }
}