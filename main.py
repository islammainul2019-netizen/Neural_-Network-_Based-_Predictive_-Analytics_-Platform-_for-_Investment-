main.py

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from config import Config
from data import prepare_datasets
from trainer import (
    set_seed,
    build_dataloaders,
    create_model,
    train_model,
    evaluate_model,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Neural Network Based Predictive Analytics Platform for Investment and Market Trend Forecasting"
    )
    parser.add_argument("--ticker", type=str, default="AAPL", help="Ticker symbol")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Start date")
    parser.add_argument("--end", type=str, default="2025-01-01", help="End date")
    parser.add_argument("--lookback", type=int, default=60, help="Lookback window")
    parser.add_argument("--horizon", type=int, default=5, help="Forecast horizon (days ahead)")
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--device", type=str, default="auto", help="cpu / cuda / auto")

    return parser.parse_args()


def select_device(device_arg: str) -> torch.device:
    if device_arg == "cpu":
        return torch.device("cpu")
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Fallback
    return torch.device("cpu")


def plot_predictions(dates, true_vals, pred_vals, ticker: str, horizon: int, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(dates, true_vals, label="True future log-return")
    plt.plot(dates, pred_vals, label="Predicted future log-return", linestyle="--")
    plt.title(f"{ticker} - {horizon}-day Ahead Log-Return Forecast")
    plt.xlabel("Date")
    plt.ylabel("Log-return")
    plt.legend()
    plt.tight_layout()
    out_path = os.path.join(out_dir, f"{ticker}_forecast.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Saved prediction plot to: {out_path}")


def main():
    args = parse_args()
    device = select_device(args.device)
    print(f"Using device: {device}")

    # Build config
    config = Config(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        lookback_window=args.lookback,
        forecast_horizon=args.horizon,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
    )

    set_seed(config.random_seed)

    print("Preparing datasets...")
    splits, targets, scalers, dates = prepare_datasets(config)
    feature_scaler = scalers["feature_scaler"]
    target_scaler = scalers["target_scaler"]

    # Dates align with all sequences; split as in splits
    n_train = len(splits["train"])
    n_val = len(splits["val"])
    n_test = len(splits["test"])

    dates_train = dates[:n_train]
    dates_val = dates[n_train: n_train + n_val]
    dates_test = dates[n_train + n_val:]

    # Create loaders
    loaders = build_dataloaders(splits, targets, config, device)

    # Model
    input_size = splits["train"].shape[-1]
    model = create_model(input_size=input_size, config=config, device=device)

    print("Training model...")
    model, train_metrics = train_model(model, loaders, config, device)
    print(f"Best validation loss (MSE, scaled): {train_metrics['best_val_loss']:.6f}")

    print("Evaluating model on test set...")
    test_metrics, y_true, y_pred = evaluate_model(
        model, loaders, target_scaler, device
    )

    print("Test metrics (inverse scaled):")
    print(f"  RMSE: {test_metrics['rmse']:.6f}")
    print(f"  MAE: {test_metrics['mae']:.6f}")
    print(f"  Directional accuracy: {test_metrics['directional_accuracy']:.4f}")

    # Plot predictions vs true on test set
    plot_predictions(
        dates_test,
        y_true,
        y_pred,
        ticker=config.ticker,
        horizon=config.forecast_horizon,
        out_dir="outputs",
    )


if __name__ == "__main__":
    main()
