from pathlib import Path
import sys

import torch
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from models.failure_predictor import FailurePredictor
from data.failure_dataset import FEATURE_COLUMNS

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = FailurePredictor(num_features=12).to(device)
model.load_state_dict(
    torch.load(ROOT / "saved_models" / "failure_predictor.pth", map_location=device)
)
model.eval()

CSV_PATH = ROOT.parent / "simulation" / "failure_history.csv"
SEQ_LEN = 10
THRESHOLD = 0.5

df = pd.read_csv(CSV_PATH)

lead_times = []

for node_id, group in df.groupby("nodeId"):

    group = group.sort_values("time").reset_index(drop=True)
    features = group[FEATURE_COLUMNS].values.astype(np.float32)

    # Find the actual failure tick (first "active"==0 after being 1).
    failure_idx = None
    for i in range(1, len(group)):
        if group.loc[i - 1, "active"] == 1 and group.loc[i, "active"] == 0:
            failure_idx = i
            break

    if failure_idx is None or failure_idx < SEQ_LEN:
        continue

    first_alert_time = None

    for end in range(SEQ_LEN, failure_idx + 1):
        window = features[end - SEQ_LEN:end]
        x = torch.tensor(window, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            prob = torch.sigmoid(model(x)).item()

        if prob > THRESHOLD:
            first_alert_time = group.loc[end - 1, "time"]
            break

    failure_time = group.loc[failure_idx, "time"]

    if first_alert_time is not None:
        lead = failure_time - first_alert_time
        lead_times.append(lead)
        print(f"Node {node_id}: alerted at t={first_alert_time:.2f}, "
              f"failed at t={failure_time:.2f}, lead={lead:.2f}s")
    else:
        print(f"Node {node_id}: no early alert before failure at t={failure_time:.2f}")

if lead_times:
    print(f"\nAverage prediction lead time: {np.mean(lead_times):.2f}s over {len(lead_times)} failures")
else:
    print("\nNo lead times measured — check threshold or training data.")