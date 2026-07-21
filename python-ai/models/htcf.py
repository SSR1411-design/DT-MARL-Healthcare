import torch
import torch.nn as nn

from models.transformer import TransformerEncoder
from models.bilstm import BiLSTM


class HTCF(nn.Module):

    def __init__(self):

        super().__init__()

        self.transformer = TransformerEncoder()

        self.bilstm = BiLSTM()

        self.classifier = nn.Sequential(

            nn.Linear(256,128),

            nn.ReLU(),

            nn.Dropout(0.3),

            nn.Linear(128,64),

            nn.ReLU(),

            nn.Dropout(0.2),

            nn.Linear(64,2)

        )

    def forward(self,x):

        # -------------------------
        # Transformer
        # -------------------------

        x = self.transformer(x)

        # Shape
        # (Batch,10,128)

        # -------------------------
        # BiLSTM
        # -------------------------

        x = self.bilstm(x)

        # Shape
        # (Batch,10,256)

        # -------------------------
        # Mean Pooling
        # -------------------------

        x = torch.mean(x,dim=1)

        # Shape
        # (Batch,256)

        output = self.classifier(x)

        return output


if __name__ == "__main__":

    sample = torch.randn(
        32,
        10,
        11
    )

    model = HTCF()

    output = model(sample)

    print()

    print("Input Shape")

    print(sample.shape)

    print()

    print("Output Shape")

    print(output.shape)