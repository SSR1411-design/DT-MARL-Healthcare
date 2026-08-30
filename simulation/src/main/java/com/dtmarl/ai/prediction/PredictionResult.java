package com.dtmarl.ai.prediction;

/**
 * Immutable failure-prediction reading for a single edge node.
 *
 * <p>A {@code PredictionResult} carries the two numbers scheduling and
 * self-healing consume — {@code failureProbability} (the chance the node
 * fails in the near horizon) and {@code failureConfidence} (how much to
 * trust that probability) — together with the node it refers to.</p>
 *
 * <p><b>Boundary object.</b> This type is the value that crosses the
 * {@link PredictionGateway} seam. Sprint 5 produces it from a neutral
 * placeholder; a later sprint may produce it from live HTCF/TGNN inference.
 * Keeping it immutable means a reading can be stamped onto a task or a twin
 * and logged without any risk of later mutation.</p>
 */
public final class PredictionResult {

    /** Neutral reading used when no prediction is available for a node. */
    public static final PredictionResult NEUTRAL =
            new PredictionResult(-1, 0.0, 0.0);

    private final int nodeId;
    private final double failureProbability;
    private final double failureConfidence;

    /**
     * Creates a prediction result.
     *
     * @param nodeId             edge node this reading refers to
     * @param failureProbability probability of near-term failure (0..1)
     * @param failureConfidence  confidence in the probability (0..1)
     */
    public PredictionResult(int nodeId,
                            double failureProbability,
                            double failureConfidence) {
        this.nodeId = nodeId;
        this.failureProbability = failureProbability;
        this.failureConfidence = failureConfidence;
    }

    /** @return edge node this reading refers to */
    public int getNodeId() {
        return nodeId;
    }

    /** @return probability of near-term failure (0..1) */
    public double getFailureProbability() {
        return failureProbability;
    }

    /** @return confidence in the probability (0..1) */
    public double getFailureConfidence() {
        return failureConfidence;
    }

    @Override
    public String toString() {
        return String.format(
                "PredictionResult{node=%d, p=%.3f, conf=%.3f}",
                nodeId, failureProbability, failureConfidence
        );
    }
}
