model.py
import torch
import torch.nn as nn


class LSTMRegressor(nn.Module):
    """
    LSTM-based regressor for time series forecasting.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, (hn, cn) = self.lstm(x)
        # Take the last hidden state
        last_hidden = out[:, -1, :]
        out = self.fc(last_hidden)
        return out
