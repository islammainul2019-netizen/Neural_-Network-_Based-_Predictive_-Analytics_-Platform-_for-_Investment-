trainer.py
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple
from config import Config
from model import LSTMRegressor


def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_dataloaders(
    splits: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    config: Config,
    device: torch.device,
) -> Dict[str, DataLoader]:
    """
    Convert numpy arrays into PyTorch DataLoaders.
    """
    loaders = {}
    for split in ["train", "val", "test"]:
        X = splits[split]
        y = targets[split]

        if len(X) == 0:
            loaders[split] = None
            continue

        X_tensor = torch.tensor(X, dtype=torch.float32, device=device)
        y_tensor = torch.tensor(y, dtype=torch.float32, device=device)

        dataset = TensorDataset(X_tensor, y_tensor)
        shuffle = split == "train"
        loader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            shuffle=shuffle,
            drop_last=False,
        )
        loaders[split] = loader

    return loaders


def create_model(input_size: int, config: Config, device: torch.device) -> LSTMRegressor:
    model = LSTMRegressor(
        input_size=input_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
    )
    model.to(device)
    return model


def train_model(
    model: LSTMRegressor,
    loaders: Dict[str, DataLoader],
    config: Config,
    device: torch.device,
) -> Tuple[LSTMRegressor, Dict[str, float]]:
    """
    Train the model and return trained model + best validation loss.
    """
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")
    best_state_dict = None

    for epoch in range(1, config.num_epochs + 1):
        model.train()
        train_losses = []

        train_loader = loaders["train"]
        if train_loader is None:
            raise ValueError("Train loader is None. Check dataset sizes.")

        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            preds = model(X_batch)
            loss = criterion(preds, y_batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        avg_train_loss = float(np.mean(train_losses))

        # Validation
        model.eval()
        val_loader = loaders["val"]
        val_losses = []

        if val_loader is not None:
            with torch.no_grad():
                for X_batch, y_batch in val_loader:
                    preds = model(X_batch)
                    loss = criterion(preds, y_batch)
                    val_losses.append(loss.item())

            avg_val_loss = float(np.mean(val_losses))
        else:
            avg_val_loss = float("nan")

        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)

        print(
            f"Epoch {epoch}/{config.num_epochs} "
            f"- Train Loss: {avg_train_loss:.6f} "
            f"- Val Loss: {avg_val_loss:.6f}"
        )

        # Track best model on validation loss
        if not np.isnan(avg_val_loss) and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state_dict = model.state_dict()

    # Restore best weights if we have them
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)

    metrics = {"best_val_loss": best_val_loss}
    return model, metrics


def evaluate_model(
    model: LSTMRegressor,
    loaders: Dict[str, DataLoader],
    target_scaler,
    device: torch.device,
):
    """
    Evaluate on the test set: RMSE, MAE, direction accuracy.
    Returns dict + arrays of true/pred values (inverse scaled).
    """
    test_loader = loaders["test"]
    if test_loader is None:
        raise ValueError("Test loader is None. Check dataset sizes.")

    model.eval()
    all_preds = []
    all_true = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            preds = model(X_batch)
            all_preds.append(preds.cpu().numpy())
            all_true.append(y_batch.cpu().numpy())

    preds_scaled = np.vstack(all_preds)
    true_scaled = np.vstack(all_true)

    preds = target_scaler.inverse_transform(preds_scaled)
    true = target_scaler.inverse_transform(true_scaled)

    mse = np.mean((preds - true) ** 2)
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(preds - true)))

    # Directional accuracy: sign(pred) == sign(true)
    pred_sign = np.sign(preds)
    true_sign = np.sign(true)
    direction_acc = float(np.mean(pred_sign == true_sign))

    metrics = {
        "rmse": rmse,
        "mae": mae,
        "directional_accuracy": direction_acc,
    }

    return metrics, true.ravel(), preds.ravel()
