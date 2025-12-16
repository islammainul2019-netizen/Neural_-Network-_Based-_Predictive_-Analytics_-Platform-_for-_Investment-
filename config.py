config.py

from dataclasses import dataclass


@dataclass
class Config:
    # Data settings
    ticker: str = "AAPL"
    start_date: str = "2015-01-01"
    end_date: str = "2025-01-01"
    lookback_window: int = 60       # days used as input sequence
    forecast_horizon: int = 5       # days ahead to predict

    # Train/validation/test split (on time axis)
    val_size: float = 0.15
    test_size: float = 0.15

    # Training hyperparameters
    batch_size: int = 32
    num_epochs: int = 30
    learning_rate: float = 1e-3
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2

    # Reproducibility
    random_seed: int = 42
