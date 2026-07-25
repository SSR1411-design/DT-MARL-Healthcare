from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from data.failure_dataset import get_dataloaders
from models.failure_predictor import FailurePredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)

CSV_PATH = ROOT.parent / "simulation" / "failure_history.csv"

train_loader, test_loader = get_dataloaders(
    csv_path=str(CSV_PATH),
    sequence_length=10,
    batch_size=32
)

model = FailurePredictor(num_features=12).to(device)

# Class imbalance handling — failures are rare events, so weight the
# positive class up rather than treating it like a balanced dataset.
labels = train_loader.dataset.dataset.y.numpy()

pos = max(int(labels.sum()), 1)
neg = max(int(len(labels) - labels.sum()), 1)

pos_weight = torch.tensor(neg / pos, dtype=torch.float32).to(device)

print("Positive samples :", pos)
print("Negative samples :", neg)
print("Pos weight :", pos_weight.item())

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.0003
)

epochs = 20

for epoch in range(epochs):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device).float()

        optimizer.zero_grad()

        logits = model(X)

        loss = criterion(logits, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        preds = (torch.sigmoid(logits) > 0.5).long()

        total += y.size(0)
        correct += (preds == y.long()).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{epochs}"
        f" | Loss = {running_loss/len(train_loader):.4f}"
        f" | Accuracy = {accuracy:.2f}%"
    )

save_dir = ROOT / "saved_models"
save_dir.mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    save_dir / "failure_predictor.pth"
)

print()
print("Training Complete!")
print("Model Saved!")

model.eval()

test_correct = 0
test_total = 0

with torch.no_grad():
    for X, y in test_loader:

        X = X.to(device)
        y = y.to(device).float()

        logits = model(X)
        preds = (torch.sigmoid(logits) > 0.5).long()

        test_total += y.size(0)
        test_correct += (preds == y.long()).sum().item()

if test_total > 0:
    print(f"Test Accuracy = {100 * test_correct / test_total:.2f}%")