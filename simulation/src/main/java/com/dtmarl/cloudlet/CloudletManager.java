package com.dtmarl.cloudlet;

import com.dtmarl.healthcare.CriticalityManager;
import com.dtmarl.healthcare.CriticalityProvider;
import com.dtmarl.healthcare.CriticalityResult;
import com.dtmarl.healthcare.HealthcareTask;
import com.dtmarl.healthcare.Patient;

import org.cloudsimplus.utilizationmodels.UtilizationModelDynamic;

import java.util.ArrayList;
import java.util.List;

/**
 * Builds the healthcare workload submitted to the simulation.
 *
 * <p><b>Sprint 5 change.</b> Previously this manager produced 40 identical
 * bare {@code CloudletSimple}s. It now produces 40 {@link HealthcareTask}s —
 * each backed by the <em>same</em> CloudSim workload parameters as before,
 * so datacenter/failure/telemetry behaviour is unchanged — but additionally
 * carrying a {@link Patient}, a clinical criticality score, and a deadline.
 * Because {@code HealthcareTask} <em>is-a</em> {@code Cloudlet}, the broker
 * and every downstream CloudSim API keep working without modification.</p>
 *
 * <p>Patient clinical attributes are generated <em>deterministically</em>
 * from the task index (no randomness), so runs remain reproducible and the
 * priority ordering is stable. Criticality itself is delegated to the
 * injected {@link CriticalityProvider}; this class never scores tasks
 * itself.</p>
 */
public class CloudletManager {

    // ----- Workload parameters (unchanged from the pre-Sprint-5 values) -

    /** Number of healthcare tasks to create. */
    public static final int TASK_COUNT = 40;

    /** Cloudlet length in Million Instructions (MI). */
    public static final long TASK_LENGTH_MI = 40000;

    /** Number of CPU cores (PEs) each task requires. */
    public static final int TASK_PES = 2;

    /** Constant CPU utilization fraction for each task. */
    public static final double TASK_CPU_UTILIZATION = 0.5;

    // ----- Healthcare metadata generation ------------------------------

    /** All Sprint 5 tasks arrive at the start of the simulation. */
    public static final double ARRIVAL_TIME_SECONDS = 0.0;

    /** Baseline deadline (seconds) applied to every task. */
    public static final double BASE_DEADLINE_SECONDS = 60.0;

    /** Extra deadline slack (seconds) added per task index for spread. */
    public static final double DEADLINE_SLACK_PER_TASK_SECONDS = 5.0;

    /** Number of distinct synthetic patients tasks are spread across. */
    public static final int PATIENT_COUNT = 10;

    private final CriticalityProvider criticalityProvider;

    /**
     * Creates a cloudlet manager that scores tasks with the given provider.
     *
     * @param criticalityProvider provider used to stamp clinical criticality
     *                            onto each task (must not be null)
     */
    public CloudletManager(CriticalityProvider criticalityProvider) {
        this.criticalityProvider = criticalityProvider;
    }

    /**
     * Creates the healthcare workload: one {@link HealthcareTask} per task
     * index, each linked to a {@link Patient} with deterministic clinical
     * attributes and stamped with a {@link CriticalityResult}.
     *
     * @return the list of created healthcare tasks (unranked; submission
     *         ordering is the broker's responsibility)
     */
    public List<HealthcareTask> createHealthcareTasks() {

        List<HealthcareTask> tasks = new ArrayList<>();

        for (int taskId = 0; taskId < TASK_COUNT; taskId++) {

            int patientId = taskId % PATIENT_COUNT;
            Patient patient = buildPatient(patientId);

            double deadline = BASE_DEADLINE_SECONDS
                    + taskId * DEADLINE_SLACK_PER_TASK_SECONDS;

            HealthcareTask task = new HealthcareTask(
                    taskId,
                    patientId,
                    TASK_LENGTH_MI,
                    TASK_PES,
                    new UtilizationModelDynamic(TASK_CPU_UTILIZATION),
                    ARRIVAL_TIME_SECONDS,
                    deadline
            );

            CriticalityResult criticality = criticalityProvider.evaluate(
                    patient, task, ARRIVAL_TIME_SECONDS);
            task.applyCriticality(criticality);

            tasks.add(task);

            System.out.println(
                    "Healthcare Task " + taskId +
                    " created successfully! (patient=" + patientId +
                    ", priority=" + String.format("%.3f", task.getPriorityScore()) +
                    ")"
            );
        }

        return tasks;
    }

    /**
     * Builds a patient with deterministic, reproducible clinical attributes
     * derived from the patient id. No randomness is used.
     *
     * @param patientId the patient identifier
     * @return a populated {@link Patient}
     */
    private Patient buildPatient(int patientId) {

        Patient patient = new Patient(patientId);

        // Spread signals across [0,1] / a plausible age range using the id,
        // so different patients get different — but fixed — criticality.
        double fraction = PATIENT_COUNT > 0
                ? (double) patientId / PATIENT_COUNT
                : 0.0;

        patient.setAttribute(CriticalityManager.ATTR_HSI, fraction);
        patient.setAttribute(CriticalityManager.ATTR_VITALS_INSTABILITY, 1.0 - fraction);
        patient.setAttribute(CriticalityManager.ATTR_AGE, 20.0 + patientId * 6.0);

        return patient;
    }
}
