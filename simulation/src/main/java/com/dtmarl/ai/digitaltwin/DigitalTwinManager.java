package com.dtmarl.ai.digitaltwin;

import com.dtmarl.healthcare.HealthcareTask;

import org.cloudsimplus.hosts.Host;

import java.util.ArrayList;
import java.util.List;

public class DigitalTwinManager {

    private final List<EdgeNode> edgeNodes;
    private final List<NetworkLink> networkLinks;
    private final List<DeviceNode> devices;

    // Sprint 5: workload-layer mirror. One TaskTwin per submitted
    // HealthcareTask, kept in sync alongside the existing host/link/device
    // twins. Purely additive — none of the existing twins are affected.
    private final List<TaskTwin> taskTwins;

    public DigitalTwinManager() {
        edgeNodes = new ArrayList<>();
        networkLinks = new ArrayList<>();
        devices = new ArrayList<>();
        taskTwins = new ArrayList<>();
    }

    public void addNode(EdgeNode node) {
        edgeNodes.add(node);
    }

    /**
     * Creates one EdgeNode per physical Host in the simulation, so the
     * Digital Twin always has a 1:1 mirror of the CloudSim infrastructure.
     * Node id == Host list index, which is also used by syncWithHosts()
     * to match twin nodes back to their real host.
     */
    public void mirrorHosts(List<Host> hosts) {

        for (int i = 0; i < hosts.size(); i++) {
            addNode(new EdgeNode(i));
        }

        System.out.println(
                "Digital Twin mirrored " + hosts.size() + " edge node(s)."
        );
    }

    /**
     * Creates one NetworkLink per node, mirroring the communication link
     * each edge node relies on. Independent of host compute state.
     */
    public void mirrorNetworkLinks(int nodeCount) {

        for (int i = 0; i < nodeCount; i++) {
            networkLinks.add(new NetworkLink(i));
        }

        System.out.println(
                "Digital Twin mirrored " + nodeCount + " network link(s)."
        );
    }

    /**
     * Creates `deviceCount` IoMT device twins (Sprint 3.75), assigning
     * each one round-robin to one of the `edgeNodeCount` edge nodes it
     * reports to — mirroring how wearables/sensors in the real
     * architecture connect to their nearest edge server.
     */
    public void mirrorDevices(int deviceCount, int edgeNodeCount) {

        for (int i = 0; i < deviceCount; i++) {

            int assignedEdgeNodeId = edgeNodeCount > 0 ? i % edgeNodeCount : 0;

            devices.add(new DeviceNode(i, assignedEdgeNodeId));
        }

        System.out.println(
                "Digital Twin mirrored " + deviceCount + " IoMT device(s)."
        );
    }

    /**
     * Pulls live resource-utilization metrics from the real CloudSim Host
     * objects into the matching EdgeNode twin. Call this after simulation
     * execution so the twin reflects the actual system state instead of
     * staying at 0.
     */
    public void syncWithHosts(List<Host> hosts) {

        for (int i = 0; i < hosts.size(); i++) {

            Host host = hosts.get(i);
            EdgeNode node = getNode(i);

            if (node == null) {
                continue;
            }

            double cpuPercent = host.getCpuPercentUtilization() * 100;

            double ramPercent = host.getRam()
                    .getPercentUtilization() * 100;

            double bwPercent = host.getBw()
                    .getPercentUtilization() * 100;

            node.setCpuUsage(cpuPercent);
            node.setRamUsage(ramPercent);
            node.setBandwidthUsage(bwPercent);

            double powerWatts = host.getPowerModel().getPower();
            node.setEnergyConsumption(powerWatts);

            node.setRunningTasks(host.getVmList().size());

            node.setActive(host.isActive());
        }

        System.out.println(
                "Digital Twin synchronized with " + hosts.size() + " host(s)."
        );
    }

    public List<EdgeNode> getNodes() {
        return edgeNodes;
    }

    public EdgeNode getNode(int id) {

        for(EdgeNode node : edgeNodes) {
            if(node.getId() == id)
                return node;
        }

        return null;
    }

    public List<NetworkLink> getLinks() {
        return networkLinks;
    }

    public NetworkLink getLink(int nodeId) {

        for (NetworkLink link : networkLinks) {
            if (link.getNodeId() == nodeId)
                return link;
        }

        return null;
    }

    public List<DeviceNode> getDevices() {
        return devices;
    }

    public DeviceNode getDevice(int deviceId) {

        for (DeviceNode device : devices) {
            if (device.getId() == deviceId)
                return device;
        }

        return null;
    }

    // ==========================================================
    // Sprint 5: workload-layer (task) mirror
    // ==========================================================

    /**
     * Creates one {@link TaskTwin} per submitted {@link HealthcareTask},
     * mirroring the workload the way {@link #mirrorHosts(List)} mirrors the
     * infrastructure. Call once after the tasks have been built.
     *
     * @param tasks the healthcare tasks to mirror (must not be null)
     */
    public void mirrorTasks(List<HealthcareTask> tasks) {

        for (HealthcareTask task : tasks) {
            taskTwins.add(new TaskTwin(task));
        }

        System.out.println(
                "Digital Twin mirrored " + tasks.size() + " healthcare task(s)."
        );
    }

    /**
     * Refreshes every {@link TaskTwin} from its source task. Call on each
     * clock tick (after infrastructure sync) so the workload mirror stays
     * current. Twins are matched to tasks by healthcare task id.
     *
     * @param tasks the live healthcare tasks to sync from (must not be null)
     */
    public void syncTasks(List<HealthcareTask> tasks) {

        for (HealthcareTask task : tasks) {

            TaskTwin twin = getTaskTwin(task.getHealthcareTaskId());

            if (twin != null) {
                twin.syncFrom(task);
            }
        }
    }

    /** @return the live list of task twins (never null; may be empty) */
    public List<TaskTwin> getTaskTwins() {
        return taskTwins;
    }

    /**
     * @param healthcareTaskId domain id of the task to look up
     * @return the matching {@link TaskTwin}, or {@code null} if none
     */
    public TaskTwin getTaskTwin(int healthcareTaskId) {

        for (TaskTwin twin : taskTwins) {
            if (twin.getHealthcareTaskId() == healthcareTaskId)
                return twin;
        }

        return null;
    }

    /**
     * Prints the workload-layer mirror. Kept <em>separate</em> from
     * {@link #printStatus()} so existing infrastructure telemetry output is
     * byte-for-byte unchanged.
     */
    public void printTaskStatus() {

        if (taskTwins.isEmpty()) {
            return;
        }

        System.out.println("\n---------- HEALTHCARE TASK LAYER ----------");

        for (TaskTwin twin : taskTwins) {

            System.out.println("Task ID    : " + twin.getHealthcareTaskId());
            System.out.println("Patient    : " + twin.getPatientId());
            System.out.println("Edge Node  : " + twin.getCurrentEdgeNodeId());
            System.out.println("State      : " + twin.getTaskState());
            System.out.println("Priority   : " + twin.getPriorityScore());
            System.out.println("Severity   : " + twin.getClinicalSeverity());
            System.out.println("P(fail)    : " + twin.getFailureProbability());
            System.out.println("Confidence : " + twin.getFailureConfidence());
            System.out.println("Migrations : " + twin.getMigrationCount());
            System.out.println("--------------------------------");
        }
    }

    public void printStatus() {

        System.out.println("\n========== DIGITAL TWIN ==========");

        for(EdgeNode node : edgeNodes) {

            System.out.println("Node ID : " + node.getId());
            System.out.println("CPU     : " + node.getCpuUsage());
            System.out.println("RAM     : " + node.getRamUsage());
            System.out.println("BW      : " + node.getBandwidthUsage());
            System.out.println("Energy  : " + node.getEnergyConsumption());
            System.out.println("Tasks   : " + node.getRunningTasks());
            System.out.println("Alive   : " + node.isActive());
            System.out.println("Degraded: " + node.isDegraded());

            NetworkLink link = getLink(node.getId());

            if (link != null) {
                System.out.println("Link Up : " + link.isUp());
                System.out.println("Link BW : " + link.getBandwidthMbps() + " Mbps");
                System.out.println("Latency : " + link.getLatencyMs() + " ms");
                System.out.println("PktLoss : " + link.getPacketLossPercent() + " %");
                System.out.println("Attacked: " + link.isUnderAttack());
            }

            System.out.println("--------------------------------");
        }

        if (!devices.isEmpty()) {

            System.out.println("\n---------- IoMT DEVICE LAYER ----------");

            for (DeviceNode device : devices) {

                System.out.println("Device ID   : " + device.getId());
                System.out.println("Edge Node   : " + device.getAssignedEdgeNodeId());
                System.out.println("Battery     : " + device.getBatteryLevel() + " %");
                System.out.println("Connected   : " + device.isConnected());
                System.out.println("Signal Qual : " + device.getSignalQuality() + " %");
                System.out.println("Signal Noise: " + device.getSignalNoise() + " %");
                System.out.println("Last HB Time: " + device.getLastHeartbeatTime() + "s");
                System.out.println("Alive       : " + device.isActive());
                System.out.println("FailureMode : " + device.getFailureMode());
                System.out.println("--------------------------------");
            }
        }
    }
}