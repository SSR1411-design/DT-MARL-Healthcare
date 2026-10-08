from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

print("=" * 80)
print("DT-MARL PAPER DATA INSPECTION")
print("=" * 80)

for path in sorted(DATA.glob("*.csv")):
    print("\n" + "-" * 80)
    print(f"FILE: {path.name}")
    print("-" * 80)

    try:
        df = pd.read_csv(path)

        print(f"Rows    : {len(df)}")
        print(f"Columns : {len(df.columns)}")

        print("\nCOLUMN NAMES:")
        for col in df.columns:
            print(f"  - {col}")

        print("\nFIRST 5 ROWS:")
        print(df.head().to_string(index=False))

        print("\nNUMERIC SUMMARY:")
        numeric = df.select_dtypes(include="number")

        if not numeric.empty:
            print(numeric.describe().round(4).to_string())
        else:
            print("  No numeric columns.")

    except Exception as e:
        print(f"ERROR READING {path.name}: {e}")

print("\n" + "=" * 80)
print("INSPECTION COMPLETE")
print("=" * 80)