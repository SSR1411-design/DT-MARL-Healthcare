package com.dtmarl.ai.prediction;

import java.io.BufferedReader;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.function.DoubleSupplier;

/**
 * {@link PredictionGateway} backed by a per-(node, time) failure-risk table
 * exported from the Python pipeline.
 *
 * <p><b>What this replaces.</b> {@link CsvPredictionGateway} is the Sprint 5
 * placeholder: it returns {@link PredictionResult#NEUTRAL} for every node and
 * performs no I/O. This class is the first implementation that carries a real
 * predicted risk across the seam. It reads the CSV written by
 * {@code python-ai/marl/export_risk_csv.py}, whose rows are the out-of-fold
 * scores of the host failure predictor — every window scored by a model that
 * never trained on its temporal block.</p>
 *
 * <p><b>Expected CSV format</b> (header required, column order free):</p>
 * <pre>
 * nodeId,time,predicted_failure_risk,prediction_uncertainty
 * 0,9.0,0.0231,0.0
 * 0,10.0,0.0244,0.0
 * </pre>
 *
 * <h2>Causality</h2>
 *
 * <p>Lookup is a <b>step hold on the most recent row at or before the current
 * clock</b>. It never interpolates forward and never reads a row with a
 * timestamp greater than {@code clock.getAsDouble()}. Before the first
 * exported timestamp for a node the gateway returns
 * {@link PredictionResult#NEUTRAL}, because the predictor needs a
 * {@code sequence_length}-long observation window before it can score
 * anything and inventing a value for that warm-up period would be fabricating
 * a prediction.</p>
 *
 * <h2>The confidence field is deliberately left at zero</h2>
 *
 * <p>{@link PredictionResult#getFailureConfidence()} returns {@code 0.0} from
 * this gateway, and that is not an oversight. The predictor emits an
 * <em>uncalibrated</em> risk score: the sigmoid of a logit, pooled across five
 * cross-validation folds, with no reliability curve fitted. There is
 * therefore no quantity in the pipeline that honestly deserves the name
 * "confidence" yet. Populating the field with {@code 1 - uncertainty} would
 * report perfect confidence, since Sprint 6 exports the uncertainty column as
 * a reserved zero. Sprint 6.5 is where an uncertainty estimate is actually
 * produced; until then the field stays at zero and callers must read
 * {@link PredictionResult#getFailureProbability()} as a
 * <em>predicted_failure_risk</em>, not as a calibrated probability.</p>
 *
 * <h2>Failing loudly, then standing aside</h2>
 *
 * <p>A malformed or unreadable file is a configuration error, not something to
 * paper over: the constructor throws. Deciding whether the file exists at all
 * is the caller's job — see {@link PredictionGateways#forRiskExport}, which
 * falls back to the inert Sprint 5 gateway when no export is present, so a
 * checkout without the Python artefacts behaves exactly as it did before.</p>
 */
public final class RiskCsvPredictionGateway implements PredictionGateway {

    /** Column names accepted for the risk value, in order of preference. */
    private static final String[] RISK_COLUMNS = {
            "predicted_failure_risk", "risk", "failure_risk"
    };

    private final DoubleSupplier clock;
    private final double[] times;          // ascending, shared by all nodes
    private final double[][] risk;         // [node][timeIndex]
    private final double[][] uncertainty;  // [node][timeIndex]
    private final Path source;

    /** Cursor per node: index of the last row returned, for cheap scanning. */
    private final int[] cursor;

    /**
     * Loads a risk export.
     *
     * @param csv   path to the exported risk table
     * @param clock supplier of the current simulation time in seconds,
     *              typically {@code simulation::clock}
     * @throws IOException              if the file cannot be read
     * @throws IllegalArgumentException if the header or contents are malformed
     */
    public RiskCsvPredictionGateway(Path csv, DoubleSupplier clock)
            throws IOException {

        this.clock = clock;
        this.source = csv;

        List<int[]> nodeIdx = new ArrayList<>();
        List<double[]> vals = new ArrayList<>();
        List<Double> timeList = new ArrayList<>();

        int maxNode = -1;

        try (BufferedReader r = Files.newBufferedReader(csv)) {

            String header = r.readLine();
            if (header == null) {
                throw new IllegalArgumentException(
                        "empty risk export: " + csv);
            }

            String[] cols = header.split(",");
            int cNode = indexOf(cols, "nodeId");
            int cTime = indexOf(cols, "time");
            int cRisk = -1;
            for (String candidate : RISK_COLUMNS) {
                cRisk = indexOf(cols, candidate);
                if (cRisk >= 0) {
                    break;
                }
            }
            int cUnc = indexOf(cols, "prediction_uncertainty");

            if (cNode < 0 || cTime < 0 || cRisk < 0) {
                throw new IllegalArgumentException(
                        "risk export must have nodeId, time and a risk column; "
                        + "found header: " + header);
            }

            String line;
            int lineNo = 1;
            while ((line = r.readLine()) != null) {
                lineNo++;
                if (line.isBlank()) {
                    continue;
                }
                String[] f = line.split(",");
                try {
                    int node = Integer.parseInt(f[cNode].trim());
                    double t = Double.parseDouble(f[cTime].trim());
                    double p = Double.parseDouble(f[cRisk].trim());
                    double u = (cUnc >= 0 && cUnc < f.length)
                            ? Double.parseDouble(f[cUnc].trim())
                            : 0.0;
                    nodeIdx.add(new int[]{node});
                    vals.add(new double[]{t, p, u});
                    timeList.add(t);
                    maxNode = Math.max(maxNode, node);
                } catch (RuntimeException ex) {
                    throw new IllegalArgumentException(
                            "malformed risk export at line " + lineNo
                            + " of " + csv + ": " + line, ex);
                }
            }
        }

        if (vals.isEmpty()) {
            throw new IllegalArgumentException(
                    "risk export has a header but no rows: " + csv);
        }

        // Unique ascending timestamps shared across nodes.
        double[] uniq = timeList.stream().mapToDouble(Double::doubleValue)
                .distinct().sorted().toArray();
        this.times = uniq;

        int nNodes = maxNode + 1;
        this.risk = new double[nNodes][uniq.length];
        this.uncertainty = new double[nNodes][uniq.length];
        this.cursor = new int[nNodes];

        // -1 marks "no prediction exported for this (node, time)", which is
        // how the pre-warm-up period stays distinguishable from a genuine
        // risk of 0.0.
        for (double[] row : this.risk) {
            Arrays.fill(row, -1.0);
        }
        Arrays.fill(this.cursor, -1);

        for (int i = 0; i < vals.size(); i++) {
            int node = nodeIdx.get(i)[0];
            double[] v = vals.get(i);
            int ti = Arrays.binarySearch(uniq, v[0]);
            if (ti < 0) {
                continue;                       // unreachable: uniq came from v
            }
            this.risk[node][ti] = v[1];
            this.uncertainty[node][ti] = v[2];
        }
    }

    private static int indexOf(String[] cols, String name) {
        for (int i = 0; i < cols.length; i++) {
            if (cols[i].trim().equalsIgnoreCase(name)) {
                return i;
            }
        }
        return -1;
    }

    /**
     * {@inheritDoc}
     *
     * <p>Returns the most recent exported risk for {@code nodeId} at or before
     * the current simulation clock, or {@link PredictionResult#NEUTRAL} when
     * the node is unknown or no row exists yet at this time.</p>
     */
    @Override
    public PredictionResult getPrediction(int nodeId) {

        if (nodeId < 0 || nodeId >= risk.length) {
            return PredictionResult.NEUTRAL;
        }

        double now = clock.getAsDouble();
        int ti = latestIndexAtOrBefore(now);
        if (ti < 0) {
            return PredictionResult.NEUTRAL;    // before the first export
        }

        // Step back to the most recent tick that actually has a value for
        // this node. Never steps forward: the search starts at `now`.
        double[] byTime = risk[nodeId];
        int k = ti;
        while (k >= 0 && byTime[k] < 0.0) {
            k--;
        }
        if (k < 0) {
            return PredictionResult.NEUTRAL;
        }

        cursor[nodeId] = k;

        // Confidence stays 0.0 by design — see the class comment. Sprint 6
        // has no calibrated uncertainty to report.
        return new PredictionResult(nodeId, byTime[k], 0.0);
    }

    /** Binary search for the last exported timestamp {@code <= now}. */
    private int latestIndexAtOrBefore(double now) {
        int i = Arrays.binarySearch(times, now);
        if (i >= 0) {
            return i;
        }
        return -i - 2;                          // insertion point - 1
    }

    /**
     * @return the reserved uncertainty value most recently returned for
     *         {@code nodeId}, which Sprint 6 always exports as 0.0. Present so
     *         Sprint 6.5 can begin consuming it without a signature change.
     */
    public double getReservedUncertainty(int nodeId) {
        if (nodeId < 0 || nodeId >= risk.length || cursor[nodeId] < 0) {
            return 0.0;
        }
        return uncertainty[nodeId][cursor[nodeId]];
    }

    /** @return number of nodes covered by the export */
    public int getNodeCount() {
        return risk.length;
    }

    /** @return number of distinct exported timestamps */
    public int getTimestampCount() {
        return times.length;
    }

    /** @return the file this gateway was loaded from */
    public Path getSource() {
        return source;
    }

    @Override
    public String toString() {
        return String.format(
                "RiskCsvPredictionGateway{source=%s, nodes=%d, timestamps=%d, "
                + "t=[%.1f..%.1f], UNCALIBRATED, confidence not populated}",
                source.getFileName(), risk.length, times.length,
                times[0], times[times.length - 1]
        );
    }
}
