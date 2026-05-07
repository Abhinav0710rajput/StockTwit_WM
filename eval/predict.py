"""
Inference: context phase + multi-step prior rollout.

Predictor.build_context       — panel DataFrame → (feat_seq, ids_seq) tensors
Predictor.context_phase       — warm up h_T, s_T from observed data
Predictor.rollout             — predict H steps ahead using prior only
Predictor.get_last_latent     — extract z_T = [h_T; s_T] after context phase
Predictor.compute_kl_sequence — KL(q||p) at every context step
Predictor.extract_attention   — A_t (cross-ticker attention) at the last context step
Predictor.get_presence_logits — presence logits + true presence for a target week
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import Tensor

from model.twit_wave import TwitWave
from data.vocab import Vocabulary

FEAT_COLS = ["log_attention", "bullish_rate", "bearish_rate", "unlabeled_rate", "attn_growth"]


class Predictor:
    def __init__(
        self,
        model: TwitWave,
        device: torch.device,
        norm_stats: dict | None = None,
    ) -> None:
        self.model      = model
        self.device     = device
        self.norm_stats = norm_stats
        self.model.eval()

    # ------------------------------------------------------------------
    def build_context(
        self,
        ctx_panel: pd.DataFrame,
        vocab: Vocabulary,
        max_len: int = 52,
    ) -> tuple[Tensor, Tensor]:
        """
        Convert a panel DataFrame to (feat_seq, ids_seq) tensors.

        Takes the last max_len weeks; within each week keeps up to
        model.cfg.top_k tickers sorted by log_attention descending.

        Returns
        -------
        feat_seq : (T, N, 5) float32  normalised features
        ids_seq  : (T, N)   int64     vocab indices (0 = padding)
        """
        top_k = self.model.cfg.top_k
        weeks = sorted(ctx_panel["week"].unique())[-max_len:]
        if not weeks:
            raise ValueError("build_context: empty context panel")

        feat_list, ids_list = [], []
        for week in weeks:
            wdf = (
                ctx_panel[ctx_panel["week"] == week]
                .nlargest(top_k, "log_attention")
                .reset_index(drop=True)
            )
            n = len(wdf)

            ids  = np.zeros(top_k, dtype=np.int64)
            feat = np.zeros((top_k, 5), dtype=np.float32)

            raw = wdf[FEAT_COLS].values[:n].astype(np.float32)
            if self.norm_stats is not None:
                mu  = self.norm_stats["mean"].astype(np.float32)
                sig = self.norm_stats["std"].astype(np.float32)
                raw = (raw - mu) / (sig + 1e-8)

            feat[:n] = raw
            ids[:n]  = [vocab.encode(s) for s in wdf["symbol"].values[:n]]

            feat_list.append(feat)
            ids_list.append(ids)

        feat_seq = torch.tensor(np.stack(feat_list), dtype=torch.float32)
        ids_seq  = torch.tensor(np.stack(ids_list),  dtype=torch.long)
        return feat_seq, ids_seq

    # ------------------------------------------------------------------
    @torch.no_grad()
    def context_phase(
        self,
        features: Tensor,    # (T, N, 5)
        ticker_ids: Tensor,  # (T, N)
    ) -> tuple[Tensor, Tensor]:
        """Run posterior on observed sequence. Returns (h_T, s_T)."""
        h, s = self.model.context_phase(
            features.to(self.device),
            ticker_ids.to(self.device),
        )
        return h, s   # (1, h_dim), (1, s_dim)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        h: Tensor,
        s: Tensor,
        steps: int,
        top_k: int | None = None,
        use_mean: bool = True,
    ) -> list[dict]:
        """
        Roll forward `steps` prior steps from (h, s).

        Returns a list of dicts per step:
          ticker_ids     : (K,)          predicted active ticker indices
          features       : (K, 5)        predicted (normalised) features
          presence_probs : (vocab_size,) presence probabilities
          z              : (z_dim,)      latent state
        """
        if top_k is None:
            top_k = self.model.cfg.top_k

        results = []
        for _ in range(steps):
            h, s, z, pres_logits = self.model.forward_step_prior(h, s, use_mean=use_mean)
            probs   = torch.sigmoid(pres_logits[0])   # (vocab_size,)
            top_ids = probs.topk(top_k).indices       # (K,)
            feat    = self.model.decode_features(z, top_ids.unsqueeze(0))[0]  # (K, 5)
            results.append({
                "ticker_ids":     top_ids.cpu().numpy(),
                "features":       feat.cpu().numpy(),
                "presence_probs": probs.cpu().numpy(),
                "z":              z[0].cpu().numpy(),
            })
        return results

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_last_latent(
        self,
        feat_seq: Tensor,
        ids_seq: Tensor,
    ) -> np.ndarray:
        """Return z_T = [h_T; s_T] after the context phase. Shape: (z_dim,)."""
        h, s = self.context_phase(feat_seq, ids_seq)
        z = torch.cat([h[0], s[0]], dim=-1)
        return z.cpu().numpy()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def compute_kl_sequence(
        self,
        feat_seq: Tensor,
        ids_seq: Tensor,
    ) -> np.ndarray:
        """
        Step through the posterior encoder and return KL(q||p) at every step.

        Returns float array of shape (T,).
        """
        from model.rssm import kl_divergence

        feat = feat_seq.unsqueeze(0).to(self.device)   # (1, T, N, 5)
        ids  = ids_seq.unsqueeze(0).to(self.device)    # (1, T, N)
        T    = feat.shape[1]
        wk   = self.model.cfg.window_k

        h, s = self.model.rssm.init_state(1, self.device)
        a_cache: list[Tensor] = []
        kls: list[float] = []

        for t in range(T):
            a_t, _ = self.model._encode_step(feat[:, t], ids[:, t])
            a_cache.append(a_t)
            window = self.model._build_window(a_cache, wk, self.device)
            e_t    = self.model.temporal_enc(window)
            h      = self.model.rssm.gru_step(h, s)
            s, qm, qlv = self.model.rssm.posterior(h, e_t)
            _, pm, plv = self.model.rssm.prior(h)
            kl = kl_divergence(qm, qlv, pm, plv)
            kls.append(float(kl.mean().item()))

        return np.array(kls)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def extract_attention(
        self,
        feat_seq: Tensor,
        ids_seq: Tensor,
    ) -> Tensor | None:
        """
        Run the context phase and return the cross-ticker attention matrix A_t
        at the final step. Shape: (N, N).

        Returns None if the sequence is empty.
        """
        feat = feat_seq.unsqueeze(0).to(self.device)
        ids  = ids_seq.unsqueeze(0).to(self.device)
        T    = feat.shape[1]
        wk   = self.model.cfg.window_k

        h, s = self.model.rssm.init_state(1, self.device)
        a_cache: list[Tensor] = []
        last_A: Tensor | None = None

        for t in range(T):
            a_t, A_t = self.model._encode_step(feat[:, t], ids[:, t])
            a_cache.append(a_t)
            last_A = A_t[0]   # (N, N) — keep overwriting; we want the last one
            window = self.model._build_window(a_cache, wk, self.device)
            e_t    = self.model.temporal_enc(window)
            h      = self.model.rssm.gru_step(h, s)
            s, _, _ = self.model.rssm.posterior(h, e_t)

        return last_A

    # ------------------------------------------------------------------
    @torch.no_grad()
    def get_presence_logits(
        self,
        feat_seq: Tensor,
        ids_seq: Tensor,
        panel: pd.DataFrame,
        week,
        vocab: Vocabulary,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Run context phase, take one prior step, return (logits, true_presence).

        logits        : (vocab_size,) float32
        true_presence : (vocab_size,) float32 binary
        """
        h, s = self.context_phase(feat_seq, ids_seq)
        _, _, _, pres_logits = self.model.forward_step_prior(h, s)
        logits = pres_logits[0].cpu().numpy()

        vocab_size = self.model.cfg.vocab_size
        true_pres  = np.zeros(vocab_size, dtype=np.float32)
        for sym in panel.loc[panel["week"] == week, "symbol"].values:
            idx = vocab.encode(sym)
            if idx > 0:
                true_pres[idx] = 1.0

        return logits, true_pres
