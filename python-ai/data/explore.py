import pandas as pd
from pathlib import Path

# ----------------------------
# Dataset Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parents[2]

IOT_DATASET = BASE_DIR / "datasets" / "healthcare_iot" / "healthcare_iot_target_dataset_5000.csv"
HEALTHCARE_DATASET = BASE_DIR / "datasets" / "healthcare" / "healthcare_dataset.csv"


def analyze_dataset(file_path, name):
    print("=" * 70)
    print(f"{name}")
    print("=" * 70)

    df = pd.read_csv(file_path)

    print("\nShape:")
    print(df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData Types:")
    print(df.dtypes)

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nDuplicate Rows:")
    print(df.duplicated().sum())

    print("\nStatistical Summary:")
    print(df.describe(include='all'))

    return df


def main():

    analyze_dataset(IOT_DATASET, "Healthcare IoT Dataset")

    print("\n\n")

    analyze_dataset(HEALTHCARE_DATASET, "Healthcare Dataset")


if __name__ == "__main__":
    main()