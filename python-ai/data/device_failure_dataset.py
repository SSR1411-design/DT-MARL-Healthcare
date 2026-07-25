import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

DEVICE_FEATURE_COLUMNS = [
    "battery", "connected", "active",
    "signalQuality", "signalNoise", "heartbeatStaleness"
]


class DeviceFailureHistoryDataset(Dataset):
    """
    Same sliding-window approach as FailureHistoryDataset, but built
    from device_failure_history.csv (Sprint 3.75 IoMT telemetry)
    instead of host/network telemetry.
    """

    def __init__(self, csv_path, sequence_length=10):

        df = pd.read_csv(csv_path)

        self.sequence_length = sequence_length

        sequences = []
        labels = []

        for device_id, group in df.groupby("deviceId"):

            group = group.sort_values("time").reset_index(drop=True)

            features = group[DEVICE_FEATURE_COLUMNS].values.astype(np.float32)
            targets = group["willFailSoon"].values.astype(np.int64)

            for i in range(len(group) - sequence_length + 1):
                seq = features[i:i + sequence_length]
                label = targets[i + sequence_length - 1]

                sequences.append(seq)
                labels.append(label)

        self.X = torch.tensor(np.stack(sequences), dtype=torch.float32)
        self.y = torch.tensor(np.array(labels), dtype=torch.int64)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_device_dataloaders(csv_path, sequence_length=10, batch_size=32, train_split=0.8):

    dataset = DeviceFailureHistoryDataset(csv_path, sequence_length)

    train_size = int(len(dataset) * train_split)
    test_size = len(dataset) - train_size

    train_ds, test_ds = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader