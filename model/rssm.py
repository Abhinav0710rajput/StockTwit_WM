"""
Recurrent State-Space Model core.

Components
----------
GRU          : h_t = GRU(h_{t-1}, s_{t-1})   deterministic path
Posterior    : s_t ~ q_φ(s | h_t, e_t)        diagonal Gaussian (training)
Prior        : ŝ_t ~ p_θ(s | h_t)             diagonal Gaussian (inference)

The KL divergence between posterior and prior is computed analytically.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def reparameterize(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


def kl_divergence(
    post_mean: torch.Tensor,
    post_logvar: torch.Tensor,
    prior_mean: torch.Tensor,
    prior_logvar: torch.Tensor,
) -> torch.Tensor:
    """
    Analytical KL( N(post_mean, post_var) || N(prior_mean, prior_var) )
    Summed over latent dimension, shape (B,).
    """
    prior_var  = prior_logvar.exp()
    post_var   = post_logvar.exp()
    kl = 0.5 * (
        prior_logvar - post_logvar
        + (post_var + (post_mean - prior_mean) ** 2) / prior_var.clamp(min=1e-6)
        - 1.0
    )
    return kl.sum(dim=-1)   # (B,)


def _make_mlp(in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Linear(in_dim, hidden_dim),
        nn.SiLU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers += [
        nn.Linear(hidden_dim, hidden_dim),
        nn.SiLU(),
    ]
    if dropout > 0.0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)


def make_time_embedding(week_idx: torch.Tensor, dim: int = 8) -> torch.Tensor:
    """
    Sinusoidal embedding of week-of-year (0–51).
    Encodes seasonality at annual / semi-annual / quarterly / monthly periods.
    dim must equal 2 × n_periods (default 8 = 4 periods).
    """
    periods = torch.tensor([52.0, 26.0, 13.0, 4.0], device=week_idx.device)
    t = week_idx.unsqueeze(-1).float() / periods   # (..., 4)
    return torch.cat([t.sin(), t.cos()], dim=-1)   # (..., 8)


class RSSM(nn.Module):
    # Time embedding is fed into the GRU input so h_t encodes temporal position.
    # The prior p(s_t | h_t) then remains a pure function of recurrent memory —
    # it can predict seasonal patterns because h_t carries them, not because it
    # was handed the calendar directly.
    TIME_DIM = 8

    def __init__(
        self,
        h_dim: int,      # GRU hidden size
        s_dim: int,      # stochastic latent size
        d_enc: int,      # encoder output size (input to posterior)
        mlp_hidden: int = 256,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.h_dim = h_dim
        self.s_dim = s_dim

        # GRU input: cat(s_{t-1}, time_emb_t) so h_t encodes temporal position
        self.gru = nn.GRUCell(input_size=s_dim + self.TIME_DIM, hidden_size=h_dim)

        # Posterior: concat(h_t, e_t) → (mean, logvar)
        self.posterior_net = _make_mlp(h_dim + d_enc, mlp_hidden, 2 * s_dim, dropout=dropout)

        # Prior: h_t only — pure world model prediction from recurrent memory
        self.prior_net = _make_mlp(h_dim, mlp_hidden, 2 * s_dim, dropout=dropout)

    # ------------------------------------------------------------------
    def gru_step(self, h: torch.Tensor, s: torch.Tensor, week_idx: torch.Tensor) -> torch.Tensor:
        """h_t = GRU(h_{t-1}, cat(s_{t-1}, time_emb_t))"""
        time_emb = make_time_embedding(week_idx, dim=self.TIME_DIM)
        return self.gru(torch.cat([s, time_emb], dim=-1), h)

    # ------------------------------------------------------------------
    def posterior(
        self, h: torch.Tensor, e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns s_t, mean, logvar."""
        out = self.posterior_net(torch.cat([h, e], dim=-1))
        mean, logvar = out.chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 2)
        s = reparameterize(mean, logvar)
        return s, mean, logvar

    # ------------------------------------------------------------------
    def prior(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns ŝ_t, mean, logvar."""
        out = self.prior_net(h)
        mean, logvar = out.chunk(2, dim=-1)
        logvar = logvar.clamp(-10, 2)
        s = reparameterize(mean, logvar)
        return s, mean, logvar

    def prior_mean(self, h: torch.Tensor) -> torch.Tensor:
        """Deterministic prior prediction (use at eval for stable metrics)."""
        out = self.prior_net(h)
        mean, _ = out.chunk(2, dim=-1)
        return mean

    # ------------------------------------------------------------------
    def init_state(self, batch_size: int, device: torch.device) -> tuple:
        h = torch.zeros(batch_size, self.h_dim, device=device)
        s = torch.zeros(batch_size, self.s_dim, device=device)
        return h, s
