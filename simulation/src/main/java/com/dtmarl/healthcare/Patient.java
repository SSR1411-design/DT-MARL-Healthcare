package com.dtmarl.healthcare;

import java.util.HashMap;
import java.util.Map;

/**
 * Domain representation of a patient whose physiological data drives the
 * healthcare workload.
 *
 * <p>A {@code Patient} is the authoritative source of <em>clinical
 * state</em> in the system. It is deliberately decoupled from the
 * compute-side {@link HealthcareTask}: one patient generates many tasks
 * over time, and a patient's condition can evolve independently of any
 * single task. This separation is what lets later sprints wire real
 * clinical signals in without touching the task/scheduling code.</p>
 *
 * <p><b>Future compatibility.</b> Rather than hard-coding every possible
 * medical field, this class exposes an open {@link #getAttributes()
 * attributes} map. HTCF outputs, a Health Severity Index, raw vitals, or
 * any other per-patient signal can be attached under well-known keys with
 * zero schema change. Stable, frequently-used signals can later be
 * promoted to typed fields without breaking callers that used the map.</p>
 */
public class Patient {

    private final int patientId;

    /**
     * Open, extensible per-patient metadata (vitals, HTCF output, HSI,
     * etc.). Never null; may be empty.
     */
    private final Map<String, Object> attributes;

    /**
     * Creates a patient with an empty attribute map.
     *
     * @param patientId unique identifier for this patient
     */
    public Patient(int patientId) {
        this.patientId = patientId;
        this.attributes = new HashMap<>();
    }

    /** @return unique identifier for this patient */
    public int getPatientId() {
        return patientId;
    }

    /**
     * @return the live, mutable attribute map for this patient. Callers
     *         may read and write directly; the map is never null.
     */
    public Map<String, Object> getAttributes() {
        return attributes;
    }

    /**
     * Attaches or overwrites a single attribute.
     *
     * @param key   attribute name (e.g. {@code "hsi"}, {@code "heartRate"})
     * @param value attribute value (any type)
     */
    public void setAttribute(String key, Object value) {
        attributes.put(key, value);
    }

    /**
     * Reads a single attribute.
     *
     * @param key attribute name
     * @return the stored value, or {@code null} if absent
     */
    public Object getAttribute(String key) {
        return attributes.get(key);
    }

    /**
     * Reads a numeric attribute with a fallback, so scoring code can stay
     * simple and defensive when a signal has not been populated yet.
     *
     * @param key          attribute name
     * @param defaultValue value returned when the key is absent or non-numeric
     * @return the stored number as a {@code double}, or {@code defaultValue}
     */
    public double getNumericAttribute(String key, double defaultValue) {
        Object value = attributes.get(key);

        if (value instanceof Number number) {
            return number.doubleValue();
        }

        return defaultValue;
    }

    @Override
    public String toString() {
        return "Patient{id=" + patientId + ", attributes=" + attributes.keySet() + "}";
    }
}
