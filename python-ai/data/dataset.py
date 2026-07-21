from pathlib import Path

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from torch.utils.data import Dataset
from torch.utils.data import DataLoader


# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[2]

DATASET = (
    BASE_DIR
    / "datasets"
    / "healthcare_iot"
    / "clean_healthcare_iot.csv"
)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class HealthcareSequenceDataset(Dataset):

    def __init__(self, X, y):

        self.X = torch.tensor(
            X,
            dtype=torch.float32
        )

        self.y = torch.tensor(
            y,
            dtype=torch.long
        )

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        return self.X[idx], self.y[idx]


# --------------------------------------------------
# Sequence Builder
# --------------------------------------------------

def build_sequences(df, sequence_length=10):

    sequences = []

    labels = []

    feature_columns = [
        c for c in df.columns
        if c not in [
            "Patient_ID",
            "Timestamp",
            "Target_Health_Status",
            "Sensor_ID",
            "Battery_Level (%)"
        ]
    ]
    print("\nFeature Columns:")
    for col in feature_columns:
        print(col)

    grouped = df.groupby("Patient_ID")

    total_patients = 0

    for patient_id, patient_df in grouped:

        total_patients += 1

        patient_df = patient_df.sort_values("Timestamp")

        features = patient_df[feature_columns].values

        target = patient_df["Target_Health_Status"].values

        if len(patient_df) < sequence_length:

            continue

        for i in range(
            len(patient_df) - sequence_length + 1
        ):

            sequences.append(
                features[i:i+sequence_length]
            )

            labels.append(
                target[i+sequence_length-1]
            )

    print()

    print("Patients :", total_patients)

    print("Sequences :", len(sequences))

    return (
        np.array(sequences),
        np.array(labels)
    )


# --------------------------------------------------
# DataLoader
# --------------------------------------------------

def get_dataloaders(
    batch_size=32,
    sequence_length=10
):

    df = pd.read_csv(
        DATASET
    )

    X, y = build_sequences(
        df,
        sequence_length
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = StandardScaler()

    train_shape = X_train.shape
    test_shape = X_test.shape

    X_train = scaler.fit_transform(
        X_train.reshape(-1, train_shape[-1])
    ).reshape(train_shape)

    X_test = scaler.transform(
        X_test.reshape(-1, test_shape[-1])
    ).reshape(test_shape)

    train_dataset = HealthcareSequenceDataset(
        X_train,
        y_train
    )

    test_dataset = HealthcareSequenceDataset(
        X_test,
        y_test
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    return train_loader, test_loader


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    train_loader, test_loader = get_dataloaders()

    X, y = next(iter(train_loader))

    print()

    print("Batch Shape")

    print(X.shape)

    print()

    print("Label Shape")

    print(y.shape)