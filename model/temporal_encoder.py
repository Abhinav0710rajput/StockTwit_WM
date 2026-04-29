"""
Stage-2 encoder: temporal attention over a rolling window of snapshot embeddings.

Input  : (B, k+1, d_enc)  — window of a_t embeddings with positional encoding
Output : e_t (B, d_enc)   — contextualized embedding fed to RSSM posterior
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPosEnc(nn.Module):
    def __init__(self, d_enc: int, max_len: int = 64) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_enc)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_enc, 2).float() * (-math.log(10000.0) / d_enc))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalEncoder(nn.Module):
    def __init__(
        self,
        d_enc: int,
        n_heads: int = 4,
        n_layers: int = 1,
        dropout: float = 0.1,
        max_window: int = 32,
    ) -> None:
        super().__init__()
        self.pos_enc = SinusoidalPosEnc(d_enc, max_len=max_window + 1)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_enc,
            nhead=n_heads,
            dim_feedforward=d_enc * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_enc)

    def forward(self, window: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        window : (B, W, d_enc)   W ≤ k+1, may be zero-padded at start

        Returns
        -------
        e_t : (B, d_enc)   representation of current context
        """
        h = self.pos_enc(window)               # (B, W, d_enc)
        h = self.transformer(h)
        h = self.norm(h)
        return h[:, -1]                        # take the last (most recent) token
