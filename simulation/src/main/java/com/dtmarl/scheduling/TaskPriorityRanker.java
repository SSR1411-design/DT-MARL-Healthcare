package com.dtmarl.scheduling;

import com.dtmarl.healthcare.HealthcareTask;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;

/**
 * Stateless ranker that orders {@link HealthcareTask}s for submission so
 * that the most clinically urgent work is placed first.
 *
 * <p><b>Ordering policy.</b> Tasks are compared on three keys, in order:</p>
 * <ol>
 *   <li>{@link HealthcareTask#getPriorityScore() priority score}, descending
 *       — higher-priority (sicker / more urgent) tasks first;</li>
 *   <li>{@link HealthcareTask#getDeadline() deadline}, ascending — among
 *       equal-priority tasks, the one due soonest goes first;</li>
 *   <li>{@link HealthcareTask#getArrivalTime() arrival time}, ascending — a
 *       stable, fair tie-break so ordering is deterministic.</li>
 * </ol>
 *
 * <p>The ranker holds no state and never mutates its inputs; it returns a
 * new sorted list and exposes the underlying {@link Comparator} so callers
 * can reuse the exact same policy elsewhere. This is deliberately a plain
 * priority heuristic, not a learned policy — the MARL scheduler of Sprint 6
 * will replace the ordering decision while keeping this class available as
 * a reproducible baseline.</p>
 */
public class TaskPriorityRanker {

    /**
     * Shared comparator implementing the three-key ordering policy
     * documented on this class.
     */
    private static final Comparator<HealthcareTask> PRIORITY_ORDER =
            Comparator
                    .comparingDouble(HealthcareTask::getPriorityScore).reversed()
                    .thenComparingDouble(HealthcareTask::getDeadline)
                    .thenComparingDouble(HealthcareTask::getArrivalTime);

    /**
     * Returns a new list containing the given tasks in priority order. The
     * input list is not modified.
     *
     * @param tasks the tasks to rank (must not be {@code null})
     * @return a new, sorted list ordered by the policy on this class
     */
    public List<HealthcareTask> rank(List<HealthcareTask> tasks) {
        List<HealthcareTask> ranked = new ArrayList<>(tasks);
        ranked.sort(PRIORITY_ORDER);
        return ranked;
    }

    /**
     * @return the {@link Comparator} implementing the priority ordering
     *         policy, for callers that need to sort in place or compose it
     */
    public Comparator<HealthcareTask> comparator() {
        return PRIORITY_ORDER;
    }
}
