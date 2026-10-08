"""
Python port of the Sprint 5 clinical-criticality layer.

This is a PORT, not a reimplementation: the constants, the weighted sum, the
urgency boost and the patient-attribute generation are transcribed from

    simulation/src/main/java/com/dtmarl/healthcare/CriticalityManager.java
    simulation/src/main/java/com/dtmarl/cloudlet/CloudletManager.java

and nothing about the scoring model is re-designed here. `self_check()`
reproduces the priority scores the Java run printed for tasks 0..5
(0.448, 0.495, 0.542, 0.589, 0.634, 0.679), which is the proof that the port
is exact rather than merely similar.

Clinical severity depends only on patient attributes, so it is identical to
Sprint 5 in every respect. Priority additionally depends on the deadline,
and the replay environment uses a slack-based deadline (see
EnvConfig.sla_slack_factor for why), so priority values in the environment
differ from the Java run's while the formula, constants and ordering rule are
unchanged. `sprint5_deadline` keeps Sprint 5's absolute deadline available so
the self-check can validate against the Java output.
"""

from dataclasses import dataclass, field
from typing import Dict, List

# ---- CriticalityManager constants (verbatim) ------------------------------
SOURCE_ID = "CriticalityManager/v1(python-port)"

ATTR_HSI = "hsi"
ATTR_VITALS_INSTABILITY = "vitalsInstability"
ATTR_AGE = "age"

WEIGHT_HSI = 0.60
WEIGHT_VITALS = 0.30
WEIGHT_AGE = 0.10

DEFAULT_NORMALISED_SIGNAL = 0.5
DEFAULT_AGE_YEARS = 50.0
AGE_NORMALISATION_CEILING = 100.0

MAX_URGENCY_BOOST = 0.5
URGENCY_HORIZON_SECONDS = 300.0

# ---- CloudletManager constants (verbatim) --------------------------------
BASE_DEADLINE_SECONDS = 60.0
DEADLINE_SLACK_PER_TASK_SECONDS = 5.0
PATIENT_COUNT = 10


def clamp01(value: float) -> float:
    return 0.0 if value < 0.0 else (1.0 if value > 1.0 else value)


def normalise_age(age_years: float) -> float:
    return clamp01(age_years / AGE_NORMALISATION_CEILING)


def clinical_severity(attributes: Dict[str, float]) -> float:
    """CriticalityManager.computeClinicalSeverity, transcribed."""
    hsi = attributes.get(ATTR_HSI, DEFAULT_NORMALISED_SIGNAL)
    vitals = attributes.get(ATTR_VITALS_INSTABILITY, DEFAULT_NORMALISED_SIGNAL)
    age_n = normalise_age(attributes.get(ATTR_AGE, DEFAULT_AGE_YEARS))
    return (WEIGHT_HSI * clamp01(hsi)
            + WEIGHT_VITALS * clamp01(vitals)
            + WEIGHT_AGE * age_n)


def urgency_boost(time_to_deadline: float) -> float:
    """CriticalityManager.urgencyBoost, transcribed."""
    if time_to_deadline <= 0.0:
        return MAX_URGENCY_BOOST
    if time_to_deadline >= URGENCY_HORIZON_SECONDS:
        return 0.0
    closeness = 1.0 - (time_to_deadline / URGENCY_HORIZON_SECONDS)
    return MAX_URGENCY_BOOST * closeness


def priority_score(severity: float, deadline: float, at_time: float) -> float:
    """CriticalityManager.computePriorityScore, transcribed."""
    return severity * (1.0 + urgency_boost(deadline - at_time))


@dataclass
class PatientTask:
    """
    One HealthcareTask + its Patient, with the Sprint 5 criticality attached.

    `deadline` is the environment deadline (slack-based, see EnvConfig);
    `sprint5_deadline` is Sprint 5's absolute value, retained for validation.
    """

    task_id: int
    patient_id: int
    attributes: Dict[str, float]
    severity: float
    deadline: float
    sprint5_deadline: float
    arrival_time: float
    length_mi: float

    def priority_at(self, at_time: float) -> float:
        return priority_score(self.severity, self.deadline, at_time)

    def sprint5_priority_at(self, at_time: float) -> float:
        return priority_score(self.severity, self.sprint5_deadline, at_time)


def build_patient_tasks(n_tasks: int,
                        n_patients: int = PATIENT_COUNT,
                        task_length_mi: float = 600_000.0,
                        arrivals: List[float] = None,
                        env_deadlines: List[float] = None) -> List[PatientTask]:
    """
    Recreate Sprint 5's deterministic patient/task generation.

    From CloudletManager.createHealthcareTasks:

        patientId = taskId % PATIENT_COUNT
        fraction  = patientId / PATIENT_COUNT
        hsi                = fraction
        vitalsInstability  = 1 - fraction
        age                = 20 + 6 * patientId
        deadline           = 60 + 5 * taskId
    """
    tasks: List[PatientTask] = []
    for task_id in range(n_tasks):
        patient_id = task_id % n_patients
        fraction = patient_id / n_patients
        attrs = {
            ATTR_HSI: fraction,
            ATTR_VITALS_INSTABILITY: 1.0 - fraction,
            ATTR_AGE: 20.0 + patient_id * 6.0,
        }
        sev = clinical_severity(attrs)
        s5_deadline = (BASE_DEADLINE_SECONDS
                       + task_id * DEADLINE_SLACK_PER_TASK_SECONDS)
        arrival = 0.0 if arrivals is None else float(arrivals[task_id])
        deadline = s5_deadline if env_deadlines is None else float(env_deadlines[task_id])
        tasks.append(PatientTask(
            task_id=task_id,
            patient_id=patient_id,
            attributes=attrs,
            severity=sev,
            deadline=deadline,
            sprint5_deadline=s5_deadline,
            arrival_time=arrival,
            length_mi=task_length_mi,
        ))
    return tasks


# Sprint 5 ordering rule, from scheduling/TaskPriorityRanker.PRIORITY_ORDER:
# priority desc, then deadline asc, then arrival asc.
def priority_order_key(task: PatientTask, at_time: float):
    return (-task.priority_at(at_time), task.deadline, task.arrival_time)


# ---- validation ----------------------------------------------------------

# Priorities printed by the Java run (simulation/run_out.log) for tasks 0..5
# at t = 0. Used as the port's ground truth.
JAVA_PRIORITIES_T0 = [0.448, 0.495, 0.542, 0.589, 0.634, 0.679]


def self_check(verbose: bool = True) -> bool:
    """Reproduce the Java priorities exactly (to printing precision)."""
    tasks = build_patient_tasks(len(JAVA_PRIORITIES_T0))
    ok = True
    for t, expected in zip(tasks, JAVA_PRIORITIES_T0):
        got = t.sprint5_priority_at(0.0)
        match = abs(got - expected) < 5e-4
        ok = ok and match
        if verbose:
            print(f"  task {t.task_id}: severity={t.severity:.4f} "
                  f"deadline={t.sprint5_deadline:.1f} "
                  f"priority={got:.4f}  java={expected:.3f}  "
                  f"{'OK' if match else 'MISMATCH'}")
    return ok


if __name__ == "__main__":
    print("Sprint 5 criticality port self-check "
          "(expected: Java priorities for tasks 0..5)")
    print("PASS" if self_check() else "FAIL")
