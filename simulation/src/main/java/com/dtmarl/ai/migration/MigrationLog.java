package com.dtmarl.ai.migration;

import com.dtmarl.failure.FailureEvent;

import java.io.IOException;
import java.io.PrintWriter;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Collects {@link MigrationRecord}s and computes the migration metrics
 * Sprint 6 reports.
 *
 * <p>The counters here are deliberately split into <em>requested</em> versus
 * <em>applied</em>. A run in which the policy asked for 90 migrations and
 * CloudSim honoured 12 is a materially different run from one in which 90 moves
 * happened, and collapsing the two would overstate what the mechanism did.</p>
 */
public final class MigrationLog {

    private final List<MigrationRecord> records = new ArrayList<>();

    /** Adds a record. */
    public void add(MigrationRecord record) {
        records.add(record);
    }

    /** @return an unmodifiable view of every request, in time order */
    public List<MigrationRecord> getRecords() {
        return Collections.unmodifiableList(records);
    }

    /** @return number of migration requests made */
    public int requested() {
        return records.size();
    }

    /** @return number of requests that actually changed placement */
    public int applied() {
        return (int) records.stream()
                .filter(r -> r.getOutcome().isApplied()).count();
    }

    /** @return applied migrations driven by predicted risk */
    public int appliedPreemptive() {
        return (int) records.stream()
                .filter(r -> r.getOutcome().isApplied() && r.isPreemptive())
                .count();
    }

    /** @return applied migrations driven by an observed symptom or failure */
    public int appliedReactive() {
        return applied() - appliedPreemptive();
    }

    /** @return count of applied migrations by outcome kind */
    public long countByOutcome(MigrationOutcome outcome) {
        return records.stream()
                .filter(r -> r.getOutcome() == outcome).count();
    }

    /** @return total migration cost charged across applied migrations */
    public double totalAppliedCost() {
        return records.stream()
                .filter(r -> r.getOutcome().isApplied())
                .mapToDouble(MigrationRecord::getMigrationCost).sum();
    }

    /**
     * @return applied migrations whose source host later failed within the
     *         protection window - i.e. tasks moved out of harm's way in time
     */
    public int protectedBeforeFailure() {
        return (int) records.stream()
                .filter(r -> r.getOutcome().isApplied()
                             && r.isSourceFailedAfterwards()).count();
    }

    /**
     * EVALUATION ONLY. Marks each applied migration according to whether its
     * source host failed within {@code windowSeconds} afterwards.
     *
     * <p>This reads the failure log <em>after the fact</em>. It is the one place
     * in the project where migration data is compared against future failures,
     * it runs strictly in the post-run metrics path, and its output never
     * re-enters a decision. Calling it mid-run and feeding the result to a
     * policy would be the look-ahead the project forbids.</p>
     *
     * @param failures      the run's failure event log
     * @param windowSeconds how long after a migration a source failure still
     *                      counts as having been avoided
     */
    public void finaliseAgainstFailures(List<FailureEvent> failures,
                                        double windowSeconds) {
        for (MigrationRecord r : records) {
            if (!r.getOutcome().isApplied()) {
                continue;
            }
            double best = Double.NaN;
            for (FailureEvent e : failures) {
                if (e.getType() != FailureEvent.Type.HOST_FAILURE) {
                    continue;
                }
                if (e.getNodeId() != r.getSourceNodeId()) {
                    continue;
                }
                double lead = e.getTime() - r.getTime();
                if (lead >= 0.0 && lead <= windowSeconds
                        && (Double.isNaN(best) || lead < best)) {
                    best = lead;
                }
            }
            r.setSourceFailure(!Double.isNaN(best), best);
        }
    }

    /**
     * Writes every record to CSV.
     *
     * @param path destination file
     * @throws IOException if the file cannot be written
     */
    public void exportCsv(Path path) throws IOException {
        Path parent = path.toAbsolutePath().getParent();
        if (parent != null) {
            Files.createDirectories(parent);
        }
        try (PrintWriter w = new PrintWriter(Files.newBufferedWriter(path))) {
            w.println(MigrationRecord.csvHeader());
            for (MigrationRecord r : records) {
                w.println(r.toCsvRow());
            }
        }
    }

    /** @return a one-block human-readable summary */
    public String summary() {
        StringBuilder sb = new StringBuilder();
        sb.append(String.format(
                "Migrations: %d requested, %d applied (%d preemptive, "
                + "%d reactive), cost %.2f, protected-before-failure %d%n",
                requested(), applied(), appliedPreemptive(), appliedReactive(),
                totalAppliedCost(), protectedBeforeFailure()));
        for (MigrationOutcome o : MigrationOutcome.values()) {
            long n = countByOutcome(o);
            if (n > 0) {
                sb.append(String.format("  %-34s %d%n", o, n));
            }
        }
        return sb.toString();
    }
}
