package com.dtmarl.broker;

import com.dtmarl.healthcare.HealthcareTask;
import com.dtmarl.scheduling.TaskPriorityRanker;

import org.cloudsimplus.brokers.DatacenterBroker;

import java.util.List;

/**
 * Healthcare-aware submission facade that <em>wraps</em> the existing
 * {@link BrokerManager} instead of replacing it.
 *
 * <p><b>Why wrap, not replace.</b> {@link BrokerManager} owns the concrete
 * CloudSim {@link DatacenterBroker} and every existing call site
 * (VM submission, telemetry) depends on it. This class adds one capability
 * on top — submitting {@link HealthcareTask}s in clinical-priority order —
 * without touching the broker's construction or its other responsibilities.
 * The wrapped {@link BrokerManager} remains fully usable directly, so this
 * change is strictly additive and backwards compatible.</p>
 *
 * <p><b>Responsibility.</b> Ordering the tasks is delegated to
 * {@link TaskPriorityRanker}; actually enqueuing them is delegated to the
 * CloudSim broker. This class only composes the two, keeping a single,
 * clear reason to change (how healthcare tasks are submitted).</p>
 */
public class HealthcareBroker {

    private final BrokerManager brokerManager;
    private final TaskPriorityRanker priorityRanker;

    /**
     * Creates a healthcare broker over an existing broker manager.
     *
     * @param brokerManager  the broker manager to wrap (must not be null)
     * @param priorityRanker the ranker used to order tasks before submission
     */
    public HealthcareBroker(BrokerManager brokerManager,
                            TaskPriorityRanker priorityRanker) {
        this.brokerManager = brokerManager;
        this.priorityRanker = priorityRanker;
    }

    /**
     * Ranks the given healthcare tasks by clinical priority and submits
     * them to the wrapped CloudSim broker in that order.
     *
     * <p>Because {@link HealthcareTask} is a {@code Cloudlet}, the ranked
     * list is accepted directly by
     * {@link DatacenterBroker#submitCloudletList(List)} — no conversion or
     * copying of the CloudSim workload is required.</p>
     *
     * @param tasks the healthcare tasks to submit (must not be null)
     * @return the ranked list that was submitted, in submission order
     */
    public List<HealthcareTask> submitHealthcareTasks(List<HealthcareTask> tasks) {
        List<HealthcareTask> ranked = priorityRanker.rank(tasks);
        brokerManager.getBroker().submitCloudletList(ranked);
        return ranked;
    }

    /**
     * @return the wrapped {@link BrokerManager}, so callers retain full
     *         access to the underlying broker for VMs, telemetry, etc.
     */
    public BrokerManager getBrokerManager() {
        return brokerManager;
    }

    /**
     * @return the underlying CloudSim {@link DatacenterBroker}, a
     *         convenience shortcut equivalent to
     *         {@code getBrokerManager().getBroker()}
     */
    public DatacenterBroker getBroker() {
        return brokerManager.getBroker();
    }
}
