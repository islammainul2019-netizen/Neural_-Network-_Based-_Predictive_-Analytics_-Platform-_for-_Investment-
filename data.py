data.py
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import yfinance as yf
from typing import Tuple, Dict
from config import Config


def download_price_data(config: Config) -> pd.DataFrame:
    """
    Download daily OHLCV data from Yahoo Finance.
    """
    df = yf.download(
        config.ticker,
        start=config.start_date,
        end=config.end_date,
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        raise ValueError(f"No data downloaded for ticker {config.ticker}")

    # Ensure Adj Close exists
    if "Adj Close" not in df.columns:
        df["Adj Close"] = df["Close"]

    df = df.dropna()
    return df


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construct basic technical features for forecasting.
    Target will be future log-return based on Adj Close.
    """
    data = df.copy()

    # Log returns
    data["log_return"] = np.log(data["Adj Close"]).diff()

    # 10-day rolling volatility of log returns
    data["volatility_10"] = data["log_return"].rolling(window=10).std()

    # 10-day momentum (percentage change)
    data["momentum_10"] = data["Adj Close"].pct_change(periods=10)

    # 20-day moving average
    data["ma_20"] = data["Adj Close"].rolling(window=20).mean()

    # Drop initial NaNs
    data = data.dropna()

    # Keep only relevant columns
    feature_cols = ["Adj Close", "log_return", "volatility_10", "momentum_10", "ma_20"]
    data = data[feature_cols].dropna()

    return data


def create_sequences(
    features: np.ndarray,
    target: np.ndarray,
    lookback: int,
    horizon: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert time series into supervised learning sequences.

    X[i] = features[t-lookback : t]
    y[i] = target[t + horizon - 1]
    """
    X, y = [], []
    n_samples = len(features)

    for t in range(lookback, n_samples - horizon + 1):
        X_seq = features[t - lookback: t]
        y_val = target[t + horizon - 1]
        X.append(X_seq)
        y.append(y_val)

    X = np.array(X)
    y = np.array(y).reshape(-1, 1)

    return X, y


def scale_and_split(
    X: np.ndarray,
    y: np.ndarray,
    config: Config,
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, np.ndarray],
    Dict[str, StandardScaler],
]:
    """
    Scale features and targets, then split into train/val/test
    along the time axis (no shuffling).
    """
    n_total = len(X)
    n_test = int(n_total * config.test_size)
    n_val = int(n_total * config.val_size)
    n_train = n_total - n_val - n_test

    if n_train <= 0:
        raise ValueError("Train set size is non-positive. Adjust val/test sizes.")

    # Flatten sequences for feature scaling
    n_seq, seq_len, n_feat = X.shape
    X_flat = X.reshape(n_seq * seq_len, n_feat)

    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Fit on training portion only
    X_train_flat = X_flat[: n_train * seq_len]
    y_train = y[:n_train]

    X_train_scaled_flat = feature_scaler.fit_transform(X_train_flat)
    y_train_scaled = target_scaler.fit_transform(y_train)

    # Transform all
    X_scaled_flat = feature_scaler.transform(X_flat)
    y_scaled = target_scaler.transform(y)

    X_scaled = X_scaled_flat.reshape(n_seq, seq_len, n_feat)

    # Split by time
    X_train = X_scaled[:n_train]
    y_train = y_scaled[:n_train]

    X_val = X_scaled[n_train: n_train + n_val]
    y_val = y_scaled[n_train: n_train + n_val]

    X_test = X_scaled[n_train + n_val:]
    y_test = y_scaled[n_train + n_val:]

    splits = {
        "train": X_train,
        "val": X_val,
        "test": X_test,
    }
    targets = {
        "train": y_train,
        "val": y_val,
        "test": y_test,
    }
    scalers = {
        "feature_scaler": feature_scaler,
        "target_scaler": target_scaler,
    }

    return splits, targets, scalers


def prepare_datasets(config: Config):
    """
    Full pipeline: download data, build features, create sequences,
    scale and split.
    """
    df = download_price_data(config)
    features_df = build_features(df)

    features = features_df.values
    # Use log_return as base target (index 1 in feature_cols)
    target = features_df["log_return"].values

    X, y = create_sequences(
        features=features,
        target=target,
        lookback=config.lookback_window,
        horizon=config.forecast_horizon,
    )

    splits, targets, scalers = scale_and_split(X, y, config)
    dates = features_df.index[config.lookback_window + config.forecast_horizon - 1:]

    return splits, targets, scalers, dates[-len(X):]  # align dates with X sequences
