package com.dtmarl.ai.prediction;

/**
 * Sprint 5 placeholder {@link PredictionGateway}.
 *
 * <p><b>Deliberately inert.</b> This gateway performs <em>no</em> failure
 * inference, opens <em>no</em> CSV file, and starts <em>no</em> Python
 * process. It exists only to occupy the prediction seam so that the rest of
 * Sprint 5 (task stamping, digital-twin mirroring) can be wired end-to-end
 * against a stable, side-effect-free source. For every node it returns
 * {@link PredictionResult#NEUTRAL}, i.e. zero probability and zero
 * confidence, which leaves scheduling behaviour unchanged.</p>
 *
 * <p>The class is named for its <em>future</em> role: a later sprint will
 * replace this body with logic that reads the offline HTCF/TGNN prediction
 * CSVs produced by the Python pipeline (or queries live inference) and maps
 * them to {@link PredictionResult}s. Because it sits behind
 * {@link PredictionGateway}, that swap requires no change to any caller.</p>
 */
public class CsvPredictionGateway implements PredictionGateway {

    /**
     * {@inheritDoc}
     *
     * <p>Sprint 5 behaviour: always returns {@link PredictionResult#NEUTRAL}
     * regardless of {@code nodeId}. No inference is performed.</p>
     */
    @Override
    public PredictionResult getPrediction(int nodeId) {
        return PredictionResult.NEUTRAL;
    }
}
