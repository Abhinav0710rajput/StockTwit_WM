"""
TwitWave: full world model wiring all components together.

forward_train  — runs the full ELBO computation over a batch of chunks
forward_step_prior — single prior step for inference rollout
context_phase  — warm up h_T, s_T from observed sequence (posterior)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .embeddings import TickerEmbeddings
from .set_encoder import SetEncoder
from .temporal_encoder import TemporalEncoder
from .rssm import RSSM, kl_divergence
from .decoder import PresenceHead, FeatureHead


@dataclass
class ModelConfig:
    vocab_size: int
    embed_dim: int = 128
    d_enc: int = 256
    h_dim: int = 256
    s_dim: int = 128
    n_heads: int = 4
    n_layers: int = 2
    window_k: int = 4
    mlp_hidden: int = 256
    feature_dim: int = 5
    top_k: int = 100
    dropout: float = 0.1

    @property
    def z_dim(self) -> int:
        return self.h_dim + self.s_dim


class TwitWave(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg

        self.embeddings     = TickerEmbeddings(cfg.vocab_size, cfg.embed_dim)
        self.set_encoder    = SetEncoder(
            input_dim=cfg.feature_dim + cfg.embed_dim,
            d_enc=cfg.d_enc,
            n_heads=cfg.n_heads,
            n_layers=cfg.n_layers,
            dropout=cfg.dropout,
        )
        self.temporal_enc   = TemporalEncoder(
            d_enc=cfg.d_enc,
            n_heads=cfg.n_heads,
            dropout=cfg.dropout,
        )
        self.rssm           = RSSM(
            h_dim=cfg.h_dim,
            s_dim=cfg.s_dim,
            d_enc=cfg.d_enc,
            mlp_hidden=cfg.mlp_hidden,
        )
        self.presence_head  = PresenceHead(cfg.z_dim, cfg.embed_dim)
        self.feature_head   = FeatureHead(cfg.z_dim, cfg.embed_dim, cfg.mlp_hidden)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _encode_step(
        self,
        feat: torch.Tensor,    # (B, N, 5)
        ids: torch.Tensor,     # (B, N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run Stage-1 encoder for a single time step."""
        e_dec = self.embeddings.get_dec(ids)              # (B, N, E)
        pad_mask = (ids == 0)                             # (B, N) True=padding
        inp = torch.cat([feat, e_dec], dim=-1)            # (B, N, 5+E)
        a_t, A_t = self.set_encoder(inp, key_padding_mask=pad_mask)
        return a_t, A_t

    def _build_window(
        self,
        a_cache: list[torch.Tensor],
        k: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Stack last k+1 snapshot embeddings, zero-pad if not enough history."""
        needed = k + 1
        if len(a_cache) >= needed:
            window = a_cache[-needed:]
        else:
            pad = [torch.zeros_like(a_cache[0])] * (needed - len(a_cache))
            window = pad + a_cache
        return torch.stack(window, dim=1)   # (B, k+1, d_enc)

    # ------------------------------------------------------------------
    # training forward
    # ------------------------------------------------------------------

    def forward_train(
        self,
        features: torch.Tensor,    # (B, T+k, top_k, 5)
        ticker_ids: torch.Tensor,  # (B, T+k, top_k)
        presence: torch.Tensor,    # (B, T+k, vocab_size)
        window_k: int | None = None,
    ) -> dict:
        """
        Run the full RSSM loop over the sequence.
        The first window_k steps are warm-up (losses not accumulated).

        Returns a dict with keys:
          presence_logits  (B, T, vocab_size)
          feat_pred        (B, T, top_k, 5)
          feat_true        (B, T, top_k, 5)
          presence_true    (B, T, vocab_size)
          post_mean        (B, T, s_dim)
          post_logvar      (B, T, s_dim)
          prior_mean       (B, T, s_dim)
          prior_logvar     (B, T, s_dim)
          kl_t             list of T scalars
          A_t_list         list of T tensors (B, N, N)
        """
        if window_k is None:
            window_k = self.cfg.window_k

        B, total_T, N, D = features.shape
        T = total_T - window_k           # loss steps
        device = features.device

        h, s = self.rssm.init_state(B, device)
        a_cache: list[torch.Tensor] = []

        # accumulators for loss steps only
        pres_logits_list, feat_pred_list, feat_true_list = [], [], []
        pres_true_list = []
        pm_list, plv_list, qm_list, qlv_list, kl_list, A_list = [], [], [], [], [], []

        for t in range(total_T):
            feat_t = features[:, t]        # (B, N, 5)
            ids_t  = ticker_ids[:, t]      # (B, N)

            # Stage 1
            a_t, A_t = self._encode_step(feat_t, ids_t)
            a_cache.append(a_t)

            # Stage 2
            window = self._build_window(a_cache, window_k, device)
            e_t = self.temporal_enc(window)

            # GRU
            h = self.rssm.gru_step(h, s)

            # Posterior + Prior
            s, post_mean, post_logvar = self.rssm.posterior(h, e_t)
            _, prior_mean, prior_logvar = self.rssm.prior(h)

            # KL
            kl = kl_divergence(post_mean, post_logvar, prior_mean, prior_logvar)

            # Latent state
            z_t = torch.cat([h, s], dim=-1)   # (B, z_dim)

            # Decode — only during loss steps
            if t >= window_k:
                e_ret_all = self.embeddings.all_ret()          # (vocab_size, E)
                pres_logits = self.presence_head(z_t, e_ret_all)  # (B, vocab_size)

                e_dec_active = self.embeddings.get_dec(ids_t)  # (B, N, E)
                feat_pred = self.feature_head(z_t, e_dec_active)  # (B, N, 5)

                pres_logits_list.append(pres_logits)
                feat_pred_list.append(feat_pred)
                feat_true_list.append(feat_t)
                pres_true_list.append(presence[:, t])
                pm_list.append(prior_mean)
                plv_list.append(prior_logvar)
                qm_list.append(post_mean)
                qlv_list.append(post_logvar)
                kl_list.append(kl.mean())
                A_list.append(A_t)

        def _stack(lst):
            return torch.stack(lst, dim=1)

        return {
            "presence_logits": _stack(pres_logits_list),    # (B, T, vocab_size)
            "feat_pred":       _stack(feat_pred_list),      # (B, T, N, 5)
            "feat_true":       _stack(feat_true_list),      # (B, T, N, 5)
            "presence_true":   _stack(pres_true_list),      # (B, T, vocab_size)
            "post_mean":       _stack(qm_list),             # (B, T, s_dim)
            "post_logvar":     _stack(qlv_list),
            "prior_mean":      _stack(pm_list),             # (B, T, s_dim)
            "prior_logvar":    _stack(plv_list),
            "kl_t":            kl_list,                     # list[T] of scalars
            "A_t_list":        A_list,                      # list[T] of (B,N,N)
        }

    # ------------------------------------------------------------------
    # inference
    # ------------------------------------------------------------------

    def forward_step_prior(
        self,
        h: torch.Tensor,   # (B, h_dim)
        s: torch.Tensor,   # (B, s_dim)
        use_mean: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Single prior step for multi-step rollout.

        Returns
        -------
        h_new        : (B, h_dim)
        s_new        : (B, s_dim)
        z_new        : (B, z_dim)
        pres_logits  : (B, vocab_size)
        """
        h_new = self.rssm.gru_step(h, s)

        if use_mean:
            s_new = self.rssm.prior_mean(h_new)
        else:
            s_new, _, _ = self.rssm.prior(h_new)

        z_new = torch.cat([h_new, s_new], dim=-1)
        e_ret_all = self.embeddings.all_ret()
        pres_logits = self.presence_head(z_new, e_ret_all)
        return h_new, s_new, z_new, pres_logits

    def decode_features(
        self,
        z: torch.Tensor,      # (B, z_dim)
        ticker_ids: torch.Tensor,  # (B, N)
    ) -> torch.Tensor:
        """Decode features for given ticker ids from latent state z."""
        e_dec = self.embeddings.get_dec(ticker_ids)   # (B, N, E)
        return self.feature_head(z, e_dec)             # (B, N, 5)

    # ------------------------------------------------------------------
    # context phase
    # ------------------------------------------------------------------

    @torch.no_grad()
    def context_phase(
        self,
        features: torch.Tensor,    # (T, N, 5)  — single sequence, no batch dim
        ticker_ids: torch.Tensor,  # (T, N)
        window_k: int | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Run posterior encoder on observed sequence to warm up h_T, s_T.
        Returns (h_T, s_T) ready for inference rollout.
        """
        if window_k is None:
            window_k = self.cfg.window_k

        device = features.device
        T = features.shape[0]

        # Add batch dim
        features   = features.unsqueeze(0)    # (1, T, N, 5)
        ticker_ids = ticker_ids.unsqueeze(0)  # (1, T, N)

        h, s = self.rssm.init_state(1, device)
        a_cache: list[torch.Tensor] = []

        for t in range(T):
            feat_t = features[:, t]
            ids_t  = ticker_ids[:, t]
            a_t, _ = self._encode_step(feat_t, ids_t)
            a_cache.append(a_t)
            window = self._build_window(a_cache, window_k, device)
            e_t = self.temporal_enc(window)
            h = self.rssm.gru_step(h, s)
            s, _, _ = self.rssm.posterior(h, e_t)

        return h, s   # (1, h_dim), (1, s_dim)

    # ------------------------------------------------------------------
    # param count
    # ------------------------------------------------------------------

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
