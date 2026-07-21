import math

import torch
import torch.nn as nn


# --------------------------------------------------
# Positional Encoding
# --------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):

        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1).float()

        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        return x + self.pe[:, :x.size(1)]


# --------------------------------------------------
# Transformer Encoder
# --------------------------------------------------

class TransformerEncoder(nn.Module):

    def __init__(
            self,
            input_size=11,
            d_model=128,
            n_heads=4,
            num_layers=2,
            ff_dim=256,
            dropout=0.1
    ):

        super().__init__()

        # Project 11 features -> 128-dimensional embedding
        self.embedding = nn.Linear(
            input_size,
            d_model
        )

        self.position = PositionalEncoding(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu"
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

    def forward(self, x):

        # (Batch, Seq, 11)
        x = self.embedding(x)

        # (Batch, Seq, 128)
        x = self.position(x)

        # (Batch, Seq, 128)
        x = self.encoder(x)

        return x


# --------------------------------------------------
# Test
# --------------------------------------------------

if __name__ == "__main__":

    sample = torch.randn(
        32,
        10,
        11
    )

    model = TransformerEncoder()

    output = model(sample)

    print()

    print("Input Shape")

    print(sample.shape)

    print()

    print("Output Shape")

    print(output.shape)