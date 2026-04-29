"""
Factorized decoder.

PresenceHead  : p_i = σ( h(z_t) · e_i_ret )
                h: R^z_dim → R^E  (learned projection)

FeatureHead   : 4 separate MLPs, each concat(z_t, e_i_dec) → scalar
                MLP_1 → log_attn      (linear)
                MLP_2 → bullish_rate  (sigmoid)
                bearish_rate = 1 - bullish_rate   (exact constraint, no MLP)
                MLP_3 → unlabeled_rate (sigmoid)
                MLP_4 → attn_growth   (5 * tanh)

Output feature order: [log_attn, bullish_rate, bearish_rate, unlabeled_rate, attn_growth]
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _feature_mlp(z_dim: int, e_dim: int, hidden: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(z_dim + e_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, hidden),
        nn.SiLU(),
        nn.Linear(hidden, 1),
    )


class PresenceHead(nn.Module):
    def __init__(self, z_dim: int, embed_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(z_dim, embed_dim)

    def forward(
        self,
        z: torch.Tensor,      # (B, z_dim)
        e_ret: torch.Tensor,  # (vocab_size, E)
    ) -> torch.Tensor:
        """Returns presence logits (B, vocab_size). Apply sigmoid for probabilities."""
        h = self.proj(z)                    # (B, E)
        return h @ e_ret.T                  # (B, vocab_size)


class FeatureHead(nn.Module):
    def __init__(self, z_dim: int, embed_dim: int, hidden: int) -> None:
        super().__init__()
        self.mlp_log_attn   = _feature_mlp(z_dim, embed_dim, hidden)
        self.mlp_bullish    = _feature_mlp(z_dim, embed_dim, hidden)
        self.mlp_unlabeled  = _feature_mlp(z_dim, embed_dim, hidden)
        self.mlp_growth     = _feature_mlp(z_dim, embed_dim, hidden)

    def forward(
        self,
        z: torch.Tensor,        # (B, z_dim)
        e_dec: torch.Tensor,    # (B, N, E)  decode embeddings for active tickers
    ) -> torch.Tensor:
        """
        Returns predicted features (B, N, 5) in order:
        [log_attn, bullish_rate, bearish_rate, unlabeled_rate, attn_growth]
        """
        B, N, E = e_dec.shape
        z_exp = z.unsqueeze(1).expand(B, N, -1)   # (B, N, z_dim)
        inp   = torch.cat([z_exp, e_dec], dim=-1)  # (B, N, z_dim + E)

        log_attn    = self.mlp_log_attn(inp)                   # (B, N, 1)
        bullish     = torch.sigmoid(self.mlp_bullish(inp))     # (B, N, 1)
        bearish     = 1.0 - bullish                            # (B, N, 1)  exact
        unlabeled   = torch.sigmoid(self.mlp_unlabeled(inp))   # (B, N, 1)
        growth      = 5.0 * torch.tanh(self.mlp_growth(inp))  # (B, N, 1)

        return torch.cat([log_attn, bullish, bearish, unlabeled, growth], dim=-1)  # (B, N, 5)
