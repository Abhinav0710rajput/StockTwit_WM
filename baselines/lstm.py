"""
Shared LSTM baseline.

A single LSTM processes the fixed-roster feature tensor (B, T, K*D),
with a linear head predicting the next step's features.
Has cross-ticker coupling (via the shared hidden state) but no explicit
latent regime separation. Requires zero-imputed fixed-roster data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class SharedLSTM(nn.Module):
    def __init__(
        self,
        n_tickers: int,
        feature_dim: int = 5,
        hidden_dim: int = 512,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.n_tickers   = n_tickers
        self.feature_dim = feature_dim
        input_dim        = n_tickers * feature_dim

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            dropout=dropout if n_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_dim, input_dim)

    def forward(
        self,
        x: torch.Tensor,   # (B, T, K, D)
    ) -> torch.Tensor:
        B, T, K, D = x.shape
        x_flat = x.reshape(B, T, K * D)
        out, _ = self.lstm(x_flat)          # (B, T, hidden)
        pred_flat = self.head(out)           # (B, T, K*D)
        return pred_flat.reshape(B, T, K, D)


def train_lstm_baseline(
    model: SharedLSTM,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int = 30,
    lr: float = 3e-4,
    device: torch.device = torch.device("cpu"),
    output_path: str | Path | None = None,
) -> SharedLSTM:
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    best_val = float("inf")
    for epoch in range(1, max_epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"LSTM epoch {epoch}", leave=False):
            x = batch["features"].to(device)          # (B, T+k, K, D)
            # shift: predict x_{t+1} from x_{1:t}
            inp    = x[:, :-1]   # (B, T+k-1, K, D)
            target = x[:, 1:]    # (B, T+k-1, K, D)
            pred = model(inp)
            loss = nn.functional.mse_loss(pred, target)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            train_loss += loss.item()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                inp, target = x[:, :-1], x[:, 1:]
                val_loss += nn.functional.mse_loss(model(inp), target).item()

        print(f"[LSTM] epoch {epoch}: train={train_loss/len(train_loader):.4f}  val={val_loss/len(val_loader):.4f}")

        if val_loss < best_val:
            best_val = val_loss
            if output_path:
                torch.save(model.state_dict(), output_path)

    return model


@torch.no_grad()
def predict_lstm(
    model: SharedLSTM,
    context: torch.Tensor,   # (T_ctx, K, D)
    steps: int,
    device: torch.device,
) -> np.ndarray:
    """
    Auto-regressively predict `steps` steps ahead.
    Returns (steps, K, D).
    """
    model.eval()
    model = model.to(device)
    history = context.unsqueeze(0).to(device)   # (1, T_ctx, K, D)
    preds = []
    for _ in range(steps):
        pred = model(history)[:, -1:]           # (1, 1, K, D)
        preds.append(pred[0, 0].cpu().numpy())
        history = torch.cat([history[:, 1:], pred], dim=1)
    return np.stack(preds)
