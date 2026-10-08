package com.dtmarl.ai.prediction;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.function.DoubleSupplier;

/**
 * Chooses a {@link PredictionGateway} based on what is actually present on
 * disk.
 *
 * <p><b>Why a factory rather than a constructor call.</b> The risk export is
 * produced by the Python side of the project. A fresh checkout, or a run made
 * before the export step, will not have it. Wiring
 * {@link RiskCsvPredictionGateway} in unconditionally would turn a missing
 * optional artefact into a crash and would make the Java simulation
 * un-runnable on its own. Wiring it in and silently swallowing the failure
 * would be worse: the run would look predictive while every reading was
 * neutral.</p>
 *
 * <p>So the rule is explicit and announced on stdout: <b>if the export exists,
 * use it; if not, fall back to the inert Sprint 5 placeholder and say so.</b>
 * With no export present the simulation behaves exactly as it did in Sprint 5,
 * which is what makes this addition safe to drop into the existing pipeline —
 * the regenerated dataset is unchanged.</p>
 */
public final class PredictionGateways {

    /** Default location of the Python-side export, relative to the repo root. */
    public static final String DEFAULT_RISK_CSV = "simulation/predicted_risk.csv";

    private PredictionGateways() {
    }

    /**
     * Returns a risk-backed gateway if {@code csv} exists and parses, and the
     * inert Sprint 5 placeholder otherwise.
     *
     * @param csv   path to the exported risk table; may be absent
     * @param clock supplier of current simulation time in seconds
     * @return a non-null gateway, never throwing for a merely absent file
     */
    public static PredictionGateway forRiskExport(Path csv, DoubleSupplier clock) {

        if (csv == null || !Files.isRegularFile(csv)) {
            System.out.printf(
                    "[prediction] no risk export at %s -> using inert "
                    + "CsvPredictionGateway (all readings NEUTRAL, "
                    + "Sprint 5 behaviour unchanged)%n",
                    csv == null ? DEFAULT_RISK_CSV : csv);
            return new CsvPredictionGateway();
        }

        try {
            RiskCsvPredictionGateway g =
                    new RiskCsvPredictionGateway(csv, clock);
            System.out.printf("[prediction] %s%n", g);
            System.out.println(
                    "[prediction] readings are UNCALIBRATED "
                    + "predicted_failure_risk values, not probabilities; "
                    + "confidence is not populated in Sprint 6");
            return g;
        } catch (IOException | IllegalArgumentException ex) {
            // A present-but-broken file is a real error worth surfacing, but it
            // must not take down a simulation whose primary job is unrelated.
            System.out.printf(
                    "[prediction] FAILED to load %s (%s) -> falling back to "
                    + "inert CsvPredictionGateway%n",
                    csv, ex.getMessage());
            return new CsvPredictionGateway();
        }
    }

    /**
     * Convenience overload using {@link #DEFAULT_RISK_CSV} resolved against the
     * process working directory and its parent, so the simulation works whether
     * it is launched from the repository root or from {@code simulation/}.
     *
     * @param clock supplier of current simulation time in seconds
     * @return a non-null gateway
     */
    public static PredictionGateway fromDefaultLocation(DoubleSupplier clock) {

        Path here = Path.of(DEFAULT_RISK_CSV);
        if (Files.isRegularFile(here)) {
            return forRiskExport(here, clock);
        }
        Path up = Path.of("..").resolve(DEFAULT_RISK_CSV).normalize();
        if (Files.isRegularFile(up)) {
            return forRiskExport(up, clock);
        }
        Path local = Path.of("predicted_risk.csv");
        if (Files.isRegularFile(local)) {
            return forRiskExport(local, clock);
        }
        return forRiskExport(here, clock);      // announces the fallback
    }
}
