"""
Inference: context phase + multi-step prior rollout.

Predictor.context_phase  — warm up h_T, s_T from observed data
Predictor.rollout        — predict H steps ahead using prior only
Predictor.decode_for_true_tickers — decode features for ground-truth ticker set
                                     (decouples presence error from feature error)
"""

from __future__ import annotations

import torch
import numpy as np

from model.twit_wave import TwitWave
from data.vocab import Vocabulary


class Predictor:
    def __init__(self, model: TwitWave, vocab: Vocabulary, device: torch.device) -> None:
        self.model  = model
        self.vocab  = vocab
        self.device = device
        self.model.eval()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def context_phase(
        self,
        features: torch.Tensor,    # (T_ctx, N, 5)
        ticker_ids: torch.Tensor,  # (T_ctx, N)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run posterior on observed sequence. Returns (h_T, s_T)."""
        features   = features.to(self.device)
        ticker_ids = ticker_ids.to(self.device)
        h, s = self.model.context_phase(features, ticker_ids)
        return h, s   # (1, h_dim), (1, s_dim)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rollout(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
        steps: int,
        top_k: int = 100,
        use_mean: bool = True,
    ) -> list[dict]:
        """
        Roll forward `steps` time steps using the prior.

        Returns a list of dicts per step:
          ticker_ids : (top_k,)  int  — predicted active ticker indices
          features   : (top_k, 5) float — predicted features
          presence_probs : (vocab_size,) float — presence probabilities
          z : (z_dim,) float — latent state
        """
        results = []
        for _ in range(steps):
            h, s, z, pres_logits = self.model.forward_step_prior(h, s, use_mean=use_mean)
            probs = torch.sigmoid(pres_logits[0])         # (vocab_size,)
            top_ids = probs.topk(top_k).indices           # (top_k,)

            feat = self.model.decode_features(
                z, top_ids.unsqueeze(0)
            )[0]    # (top_k, 5)

            results.append({
                "ticker_ids":      top_ids.cpu().numpy(),
                "features":        feat.cpu().numpy(),
                "presence_probs":  probs.cpu().numpy(),
                "z":               z[0].cpu().numpy(),
            })

        return results

    # ------------------------------------------------------------------
    @torch.no_grad()
    def decode_for_true_tickers(
        self,
        h: torch.Tensor,
        s: torch.Tensor,
        true_ticker_ids: torch.Tensor,  # (N,) int
        use_mean: bool = True,
    ) -> tuple[np.ndarray, torch.Tensor, torch.Tensor]:
        """
        One prior step, then decode features for the ground-truth ticker set.
        Decouples presence error from feature reconstruction error.

        Returns (features_pred (N,5), h_new, s_new).
        """
        h_new, s_new, z, _ = self.model.forward_step_prior(h, s, use_mean=use_mean)
        ids = true_ticker_ids.to(self.device).unsqueeze(0)   # (1, N)
        feat = self.model.decode_features(z, ids)[0]          # (N, 5)
        return feat.cpu().numpy(), h_new, s_new
