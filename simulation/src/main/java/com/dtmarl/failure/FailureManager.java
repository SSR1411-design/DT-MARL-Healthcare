package com.dtmarl.failure;

import com.dtmarl.ai.digitaltwin.DigitalTwinManager;
import com.dtmarl.ai.digitaltwin.EdgeNode;

import org.cloudsimplus.hosts.Host;

import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Random;
import java.util.Set;

/**
 * Simulates two distinct failure categories on the CloudSim infrastructure
 * and reflects them onto the Digital Twin in real time:
 *
 *   1) Full host failure (node goes completely down)
 *   2) Resource overload / degradation (node stays alive but is under
 *      sustained heavy load, short of full failure)
 *
 * All events are logged as structured FailureEvent records that can be
 * exported to CSV for later analysis / evaluation graphs.
 */
public class FailureManager {

    private final List<Host> hosts;
    private final DigitalTwinManager digitalTwin;

    private final Map<Integer, Double> scheduledFailures = new HashMap<>();
    private final Set<Integer> failedHosts = new HashSet<>();

    private boolean randomFailuresEnabled = false;
    private double randomFailureProbabilityPerTick = 0.0;

    // Overload detection
    private double overloadCpuThreshold = 90.0; // percent
    private final Set<Integer> currentlyOverloaded = new HashSet<>();

    private final List<FailureEvent> eventLog = new ArrayList<>();

    private final Random random;

    public FailureManager(List<Host> hosts, DigitalTwinManager digitalTwin) {
        this(hosts, digitalTwin, 0L);
    }

    /**
     * @param seed RNG seed. Seeded explicitly so a run is reproducible - the
     *             previous unseeded Random made the exported dataset
     *             impossible to regenerate.
     */
    public FailureManager(List<Host> hosts, DigitalTwinManager digitalTwin, long seed) {
        this.hosts = hosts;
        this.digitalTwin = digitalTwin;
        this.random = new Random(seed);
    }

    // ==========================================
    // Configuration
    // ==========================================

    public void scheduleFailure(int hostId, double atTime) {
        scheduledFailures.put(hostId, atTime);

        System.out.println(
                "Scheduled failure: Edge Node " + hostId +
                " will fail at simulation time " + atTime + "s"
        );
    }

    public void enableRandomFailures(double probabilityPerTick) {
        this.randomFailuresEnabled = true;
        this.randomFailureProbabilityPerTick = probabilityPerTick;

        System.out.println(
                "Random failures enabled with probability " +
                probabilityPerTick + " per tick per host."
        );
    }

    public void disableRandomFailures() {
        this.randomFailuresEnabled = false;
    }

    public void setOverloadCpuThreshold(double percent) {
        this.overloadCpuThreshold = percent;
    }

        /**
     * Allows other failure-domain managers (e.g. NetworkFailureManager)
     * to write into this same unified event log, so a single CSV export
     * covers compute + network + attack events together.
     */
    public void logEvent(FailureEvent event) {
        eventLog.add(event);
    }

    // ==========================================
    // Per-tick checks
    // ==========================================

    /**
     * Called on every simulation clock tick. Checks scheduled/random full
     * failures AND overload/degradation conditions for every host.
     */
    public void checkAndTriggerFailures(double currentTime) {

        for (int hostId = 0; hostId < hosts.size(); hostId++) {

            if (failedHosts.contains(hostId)) {
                continue;
            }

            Double scheduledTime = scheduledFailures.get(hostId);

            if (scheduledTime != null && currentTime >= scheduledTime) {
                triggerFailure(hostId, currentTime, FailureEvent.Cause.SCHEDULED);
                continue;
            }

            if (randomFailuresEnabled &&
                    random.nextDouble() < randomFailureProbabilityPerTick) {

                triggerFailure(hostId, currentTime, FailureEvent.Cause.RANDOM);
                continue;
            }

            checkOverload(hostId, currentTime);
        }
    }

    private void triggerFailure(int hostId, double currentTime, FailureEvent.Cause cause) {

        Host host = hosts.get(hostId);
        double cpuAtEvent = host.getCpuPercentUtilization() * 100;

        host.setActive(false);

        EdgeNode node = digitalTwin.getNode(hostId);

        if (node != null) {
            node.setActive(false);
            node.setDegraded(false); // failed supersedes degraded
        }

        failedHosts.add(hostId);
        currentlyOverloaded.remove(hostId);

        FailureEvent event = new FailureEvent(
                hostId, currentTime, FailureEvent.Type.HOST_FAILURE, cause, cpuAtEvent
        );

        eventLog.add(event);

        System.out.println("\n*** FAILURE *** " + event + "\n");
    }

    private void checkOverload(int hostId, double currentTime) {

        // Read the CPU from the Digital Twin, not straight from CloudSim.
        // The twin carries the degradation overlay applied by
        // HostDegradationManager earlier in the same tick, so overload is now
        // a genuine consequence of a developing fault (a throttling host
        // really does saturate) instead of a constant.
        EdgeNode node = digitalTwin.getNode(hostId);

        Host host = hosts.get(hostId);

        double cpuPercent = node != null
                ? node.getCpuUsage()
                : host.getCpuPercentUtilization() * 100;

        boolean isOverloaded = cpuPercent >= overloadCpuThreshold;
        boolean wasOverloaded = currentlyOverloaded.contains(hostId);

        if (isOverloaded && !wasOverloaded) {

            currentlyOverloaded.add(hostId);

            if (node != null) {
                node.setDegraded(true);
            }

            FailureEvent event = new FailureEvent(
                    hostId, currentTime, FailureEvent.Type.OVERLOAD_START,
                    FailureEvent.Cause.SYSTEM, cpuPercent
            );

            eventLog.add(event);

            System.out.println("\n>>> OVERLOAD START >>> " + event + "\n");

        } else if (!isOverloaded && wasOverloaded) {

            currentlyOverloaded.remove(hostId);

            if (node != null) {
                node.setDegraded(false);
            }

            FailureEvent event = new FailureEvent(
                    hostId, currentTime, FailureEvent.Type.OVERLOAD_END,
                    FailureEvent.Cause.SYSTEM, cpuPercent
            );

            eventLog.add(event);

            System.out.println("\n<<< OVERLOAD END <<< " + event + "\n");
        }
    }

    // ==========================================
    // Wear-driven failure / recovery
    //
    // Entry points for HostDegradationManager, which owns the progressive
    // degradation model. Kept here so that ALL host up/down transitions go
    // through one place and land in one event log.
    // ==========================================

    /**
     * Fails a host because accumulated wear crossed the failure hazard.
     *
     * Called by {@link HostDegradationManager} at the moment the hazard draw
     * succeeds - the failure instant is decided on the tick it happens, it is
     * never scheduled in advance, so no future timestamp exists that could
     * leak into the telemetry.
     *
     * @param mode     the fault mechanism that killed the host
     * @param wear     latent wear at the moment of failure
     * @param severity severity of the episode
     * @param leadTime seconds of OBSERVABLE degradation that preceded this
     *                 failure (0 for an abrupt failure with no warning)
     */
    public void triggerWearFailure(int hostId,
                                   double currentTime,
                                   HostFaultMode mode,
                                   double wear,
                                   double severity,
                                   double leadTime) {

        if (failedHosts.contains(hostId)) {
            return;
        }

        Host host = hosts.get(hostId);
        double cpuAtEvent = host.getCpuPercentUtilization() * 100;

        host.setActive(false);

        EdgeNode node = digitalTwin.getNode(hostId);

        if (node != null) {
            node.setActive(false);
            node.setDegraded(false);
            node.setHealthState(EdgeNode.HealthState.FAILED);
        }

        failedHosts.add(hostId);
        currentlyOverloaded.remove(hostId);

        FailureEvent event = new FailureEvent(
                hostId, currentTime, FailureEvent.Type.HOST_FAILURE,
                FailureEvent.Cause.WEAR, cpuAtEvent
        );

        eventLog.add(event);

        System.out.printf(
                "%n*** FAILURE *** Host %d at t=%.2f (mechanism=%s, wear=%.3f, "
                + "severity=%.2f, observable degradation lead=%.1fs)%n%n",
                hostId, currentTime, mode, wear, severity, leadTime);
    }

    /**
     * Returns a repaired host to service.
     *
     * Host.setActive(true) applies immediately in CloudSim Plus when the
     * host has no startup delay, and it does NOT throw here because
     * Host.setFailed() is never used by this project - deactivation is what
     * models the outage.
     *
     * @param retainedWear wear left over after an imperfect repair, logged so
     *                     repeat-offender behaviour is auditable
     */
    public void recoverHost(int hostId, double currentTime, double retainedWear) {

        if (!failedHosts.contains(hostId)) {
            return;
        }

        Host host = hosts.get(hostId);
        host.setActive(true);

        EdgeNode node = digitalTwin.getNode(hostId);

        if (node != null) {
            node.setActive(true);
            node.setDegraded(false);
            node.setHealthState(EdgeNode.HealthState.NORMAL);
        }

        failedHosts.remove(hostId);

        FailureEvent event = new FailureEvent(
                hostId, currentTime, FailureEvent.Type.HOST_RECOVERED,
                FailureEvent.Cause.REPAIR, retainedWear * 100
        );

        eventLog.add(event);

        System.out.printf(
                "+++ RECOVERED +++ Host %d back online at t=%.2f "
                + "(retained wear=%.3f)%n",
                hostId, currentTime, retainedWear);
    }

    // ==========================================
    // Accessors / Logging export
    // ==========================================

    public boolean hasFailed(int hostId) {
        return failedHosts.contains(hostId);
    }

    public Set<Integer> getFailedHosts() {
        return failedHosts;
    }

    public List<FailureEvent> getEventLog() {
        return eventLog;
    }

    /**
     * Writes the full structured event log to a CSV file, so Sprint 8/11
     * can consume it for graphs and evaluation.
     */
    public void exportEventsToCsv(String filePath) {

        try (FileWriter writer = new FileWriter(filePath)) {

            writer.write("time,nodeId,type,cause,cpuAtEvent\n");

            for (FailureEvent event : eventLog) {
                writer.write(event.toCsvRow() + "\n");
            }

            System.out.println(
                    "\nFailure event log exported to: " + filePath +
                    " (" + eventLog.size() + " events)"
            );

        } catch (IOException e) {
            System.out.println(
                    "Failed to export failure event log: " + e.getMessage()
            );
        }
    }
}