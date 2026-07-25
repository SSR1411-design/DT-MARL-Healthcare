import gzip
import json
import csv
from pathlib import Path
from collections import Counter


# ============================================================
# 1. PROJECT PATHS
# ============================================================

# Project root directory
project_root = Path.cwd()

# Raw Google ClusterData2019 dataset
raw_file = (
    project_root
    / "datasets"
    / "google"
    / "clusterdata2019"
    / "raw"
    / "collection_events-000000000000.json.gz"
)

# Processed output directory
processed_dir = (
    project_root
    / "datasets"
    / "google"
    / "clusterdata2019"
    / "processed"
)

# Create processed folder if it doesn't exist
processed_dir.mkdir(parents=True, exist_ok=True)

# Output CSV file
output_file = processed_dir / "failure_statistics.csv"


# ============================================================
# 2. EVENT TYPE MAPPING
# ============================================================

# Based on Google ClusterData2019 EventType definition
EVENT_TYPES = {
    "0": "SUBMIT",
    "1": "QUEUE",
    "2": "ENABLE",
    "3": "SCHEDULE",
    "4": "EVICT",
    "5": "FAIL",
    "6": "FINISH",
    "7": "KILL",
    "8": "LOST",
    "9": "UPDATE_PENDING",
    "10": "UPDATE_RUNNING"
}


# ============================================================
# 3. CHECK RAW DATASET
# ============================================================

print("===================================")
print("Google ClusterData2019 Processor")
print("===================================")

print("\nRaw dataset:")
print(raw_file)

if not raw_file.exists():
    print("\nERROR: Dataset file was not found!")
    print("Please check that the raw .gz file exists.")
    exit()

print("\nDataset found successfully!")
print("Processing events...")


# ============================================================
# 4. PROCESS DATA
# ============================================================

event_counts = Counter()
total_events = 0

with gzip.open(raw_file, "rt", encoding="utf-8") as file:

    for line in file:

        # Skip empty lines
        if not line.strip():
            continue

        # Convert JSON line into Python dictionary
        event = json.loads(line)

        # Get event type
        event_type = event.get("type")

        # Count event
        event_counts[event_type] += 1

        # Count total events
        total_events += 1


# ============================================================
# 5. GET IMPORTANT EVENT COUNTS
# ============================================================

submit_events = event_counts.get("0", 0)
queue_events = event_counts.get("1", 0)
enable_events = event_counts.get("2", 0)
schedule_events = event_counts.get("3", 0)
evict_events = event_counts.get("4", 0)
fail_events = event_counts.get("5", 0)
finish_events = event_counts.get("6", 0)
kill_events = event_counts.get("7", 0)
lost_events = event_counts.get("8", 0)
update_pending_events = event_counts.get("9", 0)
update_running_events = event_counts.get("10", 0)


# ============================================================
# 6. CALCULATE FAILURE STATISTICS
# ============================================================

if total_events > 0:
    failure_event_rate = fail_events / total_events
    failure_event_percentage = failure_event_rate * 100
else:
    failure_event_rate = 0
    failure_event_percentage = 0


# ============================================================
# 7. PRINT RESULTS
# ============================================================

print("\n===================================")
print("Processing Complete!")
print("===================================")

print(f"\nTotal events: {total_events}")

print("\nEvent counts:")
print(f"SUBMIT          : {submit_events}")
print(f"QUEUE           : {queue_events}")
print(f"ENABLE          : {enable_events}")
print(f"SCHEDULE        : {schedule_events}")
print(f"EVICT           : {evict_events}")
print(f"FAIL            : {fail_events}")
print(f"FINISH          : {finish_events}")
print(f"KILL            : {kill_events}")
print(f"LOST            : {lost_events}")
print(f"UPDATE_PENDING  : {update_pending_events}")
print(f"UPDATE_RUNNING  : {update_running_events}")

print("\nFailure statistics:")
print(f"Failure events        : {fail_events}")
print(f"Failure event rate    : {failure_event_rate:.6f}")
print(f"Failure event percent : {failure_event_percentage:.4f}%")


# ============================================================
# 8. CREATE PROCESSED CSV
# ============================================================

rows = [
    ["metric", "value"],

    ["total_events", total_events],

    ["submit_events", submit_events],
    ["queue_events", queue_events],
    ["enable_events", enable_events],
    ["schedule_events", schedule_events],
    ["evict_events", evict_events],
    ["fail_events", fail_events],
    ["finish_events", finish_events],
    ["kill_events", kill_events],
    ["lost_events", lost_events],
    ["update_pending_events", update_pending_events],
    ["update_running_events", update_running_events],

    ["failure_event_rate", failure_event_rate],
    ["failure_event_percentage", failure_event_percentage]
]


# ============================================================
# 9. SAVE CSV
# ============================================================

with open(
    output_file,
    "w",
    newline="",
    encoding="utf-8"
) as csv_file:

    writer = csv.writer(csv_file)

    writer.writerows(rows)


# ============================================================
# 10. FINAL MESSAGE
# ============================================================

print("\n===================================")
print("CSV FILE CREATED SUCCESSFULLY!")
print("===================================")

print("\nOutput file:")
print(output_file)