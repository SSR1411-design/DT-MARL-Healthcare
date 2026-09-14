package com.dtmarl.healthcare;

/**
 * Sprint 5 reference implementation of {@link CriticalityProvider}.
 *
 * <p><b>Deterministic, not learned.</b> This manager computes criticality
 * with a transparent, reproducible weighted formula over well-known patient
 * attributes. It contains <em>no</em> machine learning, no HTCF inference,
 * and no randomness — given the same patient and task it always returns the
 * same result. That determinism is intentional: it makes Sprint 5 runs
 * reproducible and gives later sprints a stable baseline to compare a
 * learned scorer against.</p>
 *
 * <p><b>Scoring model.</b> Clinical severity is a weighted sum of three
 * normalised patient signals — a Health Severity Index proxy, vital-sign
 * instability, and age — each read defensively from the patient's attribute
 * map with a neutral default when absent. The scheduling priority then
 * boosts severity by how close the task is to its deadline, so that an
 * equally-sick patient whose task is more urgent ranks higher.</p>
 *
 * <p>All tunable numbers are named constants; there are no magic numbers in
 * the formula. Swapping this class for an HTCF/HSI-driven provider later
 * requires no change to callers (see {@link CriticalityProvider}).</p>
 */
public class CriticalityManager implements CriticalityProvider {

    /** Provenance tag stamped onto every {@link CriticalityResult} produced. */
    public static final String SOURCE_ID = "CriticalityManager/v1";

    // ----- Patient attribute keys this provider understands ------------

    /** Attribute key for a normalised Health Severity Index proxy (0..1). */
    public static final String ATTR_HSI = "hsi";

    /** Attribute key for a normalised vital-sign instability signal (0..1). */
    public static final String ATTR_VITALS_INSTABILITY = "vitalsInstability";

    /** Attribute key for patient age in years. */
    public static final String ATTR_AGE = "age";

    // ----- Severity weights (sum to 1.0) -------------------------------

    /** Weight applied to the Health Severity Index proxy. */
    public static final double WEIGHT_HSI = 0.60;

    /** Weight applied to vital-sign instability. */
    public static final double WEIGHT_VITALS = 0.30;

    /** Weight applied to the normalised age term. */
    public static final double WEIGHT_AGE = 0.10;

    // ----- Normalisation / defaults ------------------------------------

    /** Neutral default for a normalised signal that is absent (0..1 scale). */
    public static final double DEFAULT_NORMALISED_SIGNAL = 0.5;

    /** Default age (years) used when the patient has no age attribute. */
    public static final double DEFAULT_AGE_YEARS = 50.0;

    /** Age (years) mapped to 1.0 when normalising the age term. */
    public static final double AGE_NORMALISATION_CEILING = 100.0;

    // ----- Priority shaping --------------------------------------------

    /**
     * Maximum multiplicative urgency boost applied to severity when a task
     * is exactly at its deadline. A task with slack far beyond
     * {@link #URGENCY_HORIZON_SECONDS} gets no boost.
     */
    public static final double MAX_URGENCY_BOOST = 0.5;

    /**
     * Time-to-deadline (seconds) at or beyond which no urgency boost is
     * applied. Inside this horizon the boost scales linearly toward
     * {@link #MAX_URGENCY_BOOST} as the deadline approaches.
     */
    public static final double URGENCY_HORIZON_SECONDS = 300.0;

    /**
     * {@inheritDoc}
     *
     * <p>Computes severity from patient attributes and shapes it into a
     * priority using the task's remaining time to deadline.</p>
     */
    @Override
    public CriticalityResult evaluate(Patient patient, HealthcareTask task, double atTime) {
        double severity = computeClinicalSeverity(patient);
        double priority = computePriorityScore(severity, task, atTime);
        return new CriticalityResult(severity, priority, SOURCE_ID, atTime);
    }

    /**
     * Weighted, normalised sum of the patient's clinical signals.
     *
     * @param patient the owning patient
     * @return clinical severity in the range [0, 1]
     */
    private double computeClinicalSeverity(Patient patient) {
        double hsi = patient.getNumericAttribute(ATTR_HSI, DEFAULT_NORMALISED_SIGNAL);
        double vitals = patient.getNumericAttribute(
                ATTR_VITALS_INSTABILITY, DEFAULT_NORMALISED_SIGNAL);
        double ageNormalised = normaliseAge(
                patient.getNumericAttribute(ATTR_AGE, DEFAULT_AGE_YEARS));

        return WEIGHT_HSI * clamp01(hsi)
                + WEIGHT_VITALS * clamp01(vitals)
                + WEIGHT_AGE * ageNormalised;
    }

    /**
     * Shapes clinical severity into a scheduling priority by boosting it
     * for tasks whose deadline is near.
     *
     * @param severity computed clinical severity (0..1)
     * @param task     the task being scored
     * @param atTime   current simulation time (seconds)
     * @return priority score (>= severity)
     */
    private double computePriorityScore(double severity, HealthcareTask task, double atTime) {
        double timeToDeadline = task.getDeadline() - atTime;
        double urgency = urgencyBoost(timeToDeadline);
        return severity * (1.0 + urgency);
    }

    /**
     * Linear urgency boost: {@link #MAX_URGENCY_BOOST} at (or past) the
     * deadline, decaying to zero at {@link #URGENCY_HORIZON_SECONDS} of slack.
     *
     * @param timeToDeadline seconds until the deadline (may be negative)
     * @return urgency boost in [0, {@link #MAX_URGENCY_BOOST}]
     */
    private double urgencyBoost(double timeToDeadline) {
        if (timeToDeadline <= 0.0) {
            return MAX_URGENCY_BOOST;
        }
        if (timeToDeadline >= URGENCY_HORIZON_SECONDS) {
            return 0.0;
        }
        double closeness = 1.0 - (timeToDeadline / URGENCY_HORIZON_SECONDS);
        return MAX_URGENCY_BOOST * closeness;
    }

    /**
     * Maps an age in years onto [0, 1] against a fixed ceiling.
     *
     * @param ageYears patient age in years
     * @return normalised age in [0, 1]
     */
    private double normaliseAge(double ageYears) {
        return clamp01(ageYears / AGE_NORMALISATION_CEILING);
    }

    /**
     * Clamps a value into the closed interval [0, 1].
     *
     * @param value the value to clamp
     * @return {@code value} constrained to [0, 1]
     */
    private double clamp01(double value) {
        if (value < 0.0) {
            return 0.0;
        }
        if (value > 1.0) {
            return 1.0;
        }
        return value;
    }
}
