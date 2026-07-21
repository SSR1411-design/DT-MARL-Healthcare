import torch
import torch.nn as nn


class BiLSTM(nn.Module):

    def __init__(
            self,
            input_size=128,
            hidden_size=128,
            num_layers=2,
            dropout=0.2
    ):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout
        )

    def forward(self, x):

        output, (hidden, cell) = self.lstm(x)

        return output


if __name__ == "__main__":

    sample = torch.randn(
        32,
        10,
        128
    )

    model = BiLSTM()

    output = model(sample)

    print()

    print("Input Shape")

    print(sample.shape)

    print()

    print("Output Shape")

    print(output.shape)