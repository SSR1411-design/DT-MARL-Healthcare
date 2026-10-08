package com.dtmarl.ai.prediction;

/**
 * Strategy seam for obtaining a failure prediction for an edge node.
 *
 * <p>This interface isolates the rest of the system from <em>how</em> a
 * failure probability is produced. Sprint 5 ships a neutral placeholder
 * ({@link CsvPredictionGateway}) that performs no inference and no file I/O;
 * a later sprint can supply an implementation backed by the offline HTCF/TGNN
 * pipeline or live inference, with no change to callers.</p>
 *
 * <p><b>Contract:</b> implementations must be synchronous, side-effect-free,
 * and safe to call every simulation tick. They must never return
 * {@code null}; when no prediction is available they return
 * {@link PredictionResult#NEUTRAL}.</p>
 */
public interface PredictionGateway {

    /**
     * Returns the current failure prediction for the given edge node.
     *
     * @param nodeId edge node to predict for
     * @return a non-null {@link PredictionResult}; {@link PredictionResult#NEUTRAL}
     *         when no prediction is available
     */
    PredictionResult getPrediction(int nodeId);
}
