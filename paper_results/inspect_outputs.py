from pathlib import Path
import pandas as pd
import json

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

print("=" * 80)
print("DT-MARL RESEARCH OUTPUT INSPECTOR")
print("=" * 80)

csv_files = sorted(DATA.glob("*.csv"))
json_files = sorted(DATA.glob("*.json"))

print(f"\nCSV files found: {len(csv_files)}")
print(f"JSON files found: {len(json_files)}")

# ---------------------------------------------------------------------
# CSV INVENTORY
# ---------------------------------------------------------------------

for path in csv_files:
    print("\n" + "-" * 80)
    print(f"FILE: {path.name}")

    try:
        df = pd.read_csv(path)

        print(f"Rows:    {len(df):,}")
        print(f"Columns: {len(df.columns)}")

        print("\nColumns:")
        for col in df.columns:
            dtype = df[col].dtype
            non_null = df[col].notna().sum()
            print(f"  - {col} [{dtype}] ({non_null:,} non-null)")

        numeric = df.select_dtypes(include="number")

        if not numeric.empty:
            print("\nNumeric summary:")
            summary = numeric.describe().T

            for col, row in summary.iterrows():
                print(
                    f"  {col}: "
                    f"min={row['min']:.6g}, "
                    f"max={row['max']:.6g}, "
                    f"mean={row['mean']:.6g}"
                )

    except Exception as exc:
        print(f"ERROR reading {path.name}: {exc}")

# ---------------------------------------------------------------------
# JSON INVENTORY
# ---------------------------------------------------------------------

for path in json_files:
    print("\n" + "-" * 80)
    print(f"FILE: {path.name}")

    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)

        if isinstance(obj, dict):
            print("Top-level keys:")
            for key, value in obj.items():
                if isinstance(value, (dict, list)):
                    print(f"  - {key}: {type(value).__name__}")
                else:
                    print(f"  - {key}: {value}")

        elif isinstance(obj, list):
            print(f"Top-level list length: {len(obj)}")

    except Exception as exc:
        print(f"ERROR reading {path.name}: {exc}")

# ---------------------------------------------------------------------
# EXISTING FIGURES
# ---------------------------------------------------------------------

FIGURES = ROOT / "figures"

if FIGURES.exists():
    figures = sorted(
        p for p in FIGURES.iterdir()
        if p.suffix.lower() in {".png", ".pdf", ".jpg", ".jpeg"}
    )

    print("\n" + "=" * 80)
    print("EXISTING FIGURES")
    print("=" * 80)

    for fig in figures:
        print(f"  {fig.name}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)
print("\nNo training was run.")
print("No checkpoints were modified.")
print("No source artifacts were modified.")