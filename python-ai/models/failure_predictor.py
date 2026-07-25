import torch
import torch.nn as nn

from models.transformer import TransformerEncoder
from models.bilstm import BiLSTM


class FailurePredictor(nn.Module):
    """
    Same HTCF backbone (Transformer attention + BiLSTM) as your HSI
    classifier, repointed at failure prediction. `num_features` lets
    this same class serve both the host/network predictor (12
    features) and the device/IoMT predictor (6 features) — two
    separately trained instances, same architecture.
    """

    def __init__(self, num_features=12):

        super().__init__()

        # TransformerEncoder's param is `input_size`, matching your
        # existing transformer.py signature.
        self.transformer = TransformerEncoder(input_size=num_features)

        self.bilstm = BiLSTM()

        self.classifier = nn.Sequential(

            nn.Linear(256, 128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128, 64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64, 1)  # single failure-probability logit
        )

    def forward(self, x):

        # (Batch, Seq, num_features)
        x = self.transformer(x)

        # (Batch, Seq, 256)
        x = self.bilstm(x)

        # Mean pooling -> (Batch, 256)
        x = torch.mean(x, dim=1)

        output = self.classifier(x)  # (Batch, 1)

        return output.squeeze(-1)  # (Batch,)


if __name__ == "__main__":

    sample = torch.randn(32, 10, 12)  # (batch, seq_len, num_features)

    model = FailurePredictor()

    output = model(sample)

    print()
    print("Input Shape")
    print(sample.shape)
    print()
    print("Output Shape")
    print(output.shape)