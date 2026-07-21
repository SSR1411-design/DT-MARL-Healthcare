from pathlib import Path

import pandas as pd
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DATASET = (
    BASE_DIR
    / "datasets"
    / "healthcare_iot"
    / "healthcare_iot_target_dataset_5000.csv"
)

OUTPUT_DATASET = (
    BASE_DIR
    / "datasets"
    / "healthcare_iot"
    / "clean_healthcare_iot.csv"
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

print("Loading dataset...")

df = pd.read_csv(INPUT_DATASET)

print(df.shape)

# --------------------------------------------------
# Timestamp
# --------------------------------------------------

print("Converting Timestamp...")

df["Timestamp"] = pd.to_datetime(df["Timestamp"])

# --------------------------------------------------
# Sort Data
# --------------------------------------------------

print("Sorting by Patient and Time...")

df = df.sort_values(
    by=["Patient_ID", "Timestamp"]
).reset_index(drop=True)

# --------------------------------------------------
# Time Features
# --------------------------------------------------

df["Hour"] = df["Timestamp"].dt.hour
df["Day"] = df["Timestamp"].dt.day
df["Month"] = df["Timestamp"].dt.month

# --------------------------------------------------
# Remove Duplicate Battery Column
# --------------------------------------------------

if (
    "Battery_Level (%)" in df.columns
    and
    "Device_Battery_Level (%)" in df.columns
):

    if df["Battery_Level (%)"].equals(
        df["Device_Battery_Level (%)"]
    ):

        print("Duplicate Battery Column Removed")

        df.drop(
            columns=["Battery_Level (%)"],
            inplace=True
        )

# --------------------------------------------------
# Encode Sensor Type
# --------------------------------------------------

sensor_encoder = LabelEncoder()

df["Sensor_Type"] = sensor_encoder.fit_transform(
    df["Sensor_Type"]
)

# --------------------------------------------------
# Encode Target
# --------------------------------------------------

target_encoder = LabelEncoder()

df["Target_Health_Status"] = target_encoder.fit_transform(
    df["Target_Health_Status"]
)

# --------------------------------------------------
# Save
# --------------------------------------------------

df.to_csv(
    OUTPUT_DATASET,
    index=False
)

print()

print("Saved Successfully")

print(OUTPUT_DATASET)