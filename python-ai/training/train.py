from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim

from collections import Counter


# --------------------------------------------------
# Fix imports
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]

sys.path.append(str(ROOT))

from data.dataset import get_dataloaders
from models.htcf import HTCF

# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print()

print("Device :", device)

# --------------------------------------------------
# Data
# --------------------------------------------------

train_loader, test_loader = get_dataloaders(
    batch_size=32,
    sequence_length=10
)

# --------------------------------------------------
# Model
# --------------------------------------------------

model = HTCF().to(device)

import numpy as np

labels = train_loader.dataset.y.numpy()

class_counts = np.bincount(labels)

print("Class Counts :", class_counts)

weights = 1.0 / class_counts

weights = weights / weights.sum()

weights = torch.tensor(
    weights,
    dtype=torch.float32
).to(device)

print("Class Weights :", weights)

criterion = nn.CrossEntropyLoss(weight=weights)

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

epochs = 10

# --------------------------------------------------
# Training
# --------------------------------------------------

for epoch in range(epochs):

    model.train()

    running_loss = 0

    correct = 0

    total = 0

    for X, y in train_loader:

        X = X.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        outputs = model(X)

        loss = criterion(outputs, y)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, predicted = torch.max(outputs, 1)

        total += y.size(0)

        correct += (predicted == y).sum().item()

    accuracy = 100 * correct / total

    print(
        f"Epoch {epoch+1}/{epochs}"
        f" | Loss = {running_loss/len(train_loader):.4f}"
        f" | Accuracy = {accuracy:.2f}%"
    )

# --------------------------------------------------
# Save
# --------------------------------------------------

save_dir = ROOT / "saved_models"
save_dir.mkdir(exist_ok=True)

torch.save(
    model.state_dict(),
    save_dir / "htcf_model.pth"
)

print()

print("Training Complete!")

print("Model Saved!")

model.eval()

predictions = []

with torch.no_grad():
    for X, y in train_loader:
        X = X.to(device)
        outputs = model(X)
        preds = torch.argmax(outputs, dim=1)
        predictions.extend(preds.cpu().numpy())

print(Counter(predictions))