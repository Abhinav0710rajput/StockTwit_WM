"""
Stage-1 encoder: cross-ticker self-attention within a single time step.

Input  : (B, N, D+E)  — ticker features concat'd with e_dec embeddings
Output : a_t (B, d_enc)         — pooled snapshot embedding
         A_t (B, N, N)          — mean attention weights across heads (for eval)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SetEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,    # D + E  (feature dim + embed dim)
        d_enc: int,        # output/hidden dim
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_proj = nn.Linear(input_dim, d_enc)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_enc,
            nhead=n_heads,
            dim_feedforward=d_enc * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,          # pre-norm for stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(d_enc)

        # We need raw attention weights — attach hooks on the last layer
        self._attn_weights: torch.Tensor | None = None
        self._register_attn_hook()

    def _register_attn_hook(self) -> None:
        last_layer = self.transformer.layers[-1].self_attn

        def hook(module, input, output):
            # output is (attn_output, attn_weights) when need_weights=True
            if isinstance(output, tuple) and output[1] is not None:
                self._attn_weights = output[1].detach()

        last_layer.register_forward_hook(hook)

    def forward(
        self,
        x: torch.Tensor,              # (B, N, D+E)
        key_padding_mask: torch.Tensor | None = None,  # (B, N) True=ignore
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        a_t : (B, d_enc)   pooled snapshot embedding
        A_t : (B, N, N)    mean attention weights (or zeros if unavailable)
        """
        B, N, _ = x.shape

        # Force last TransformerEncoderLayer to return attention weights
        last = self.transformer.layers[-1]
        last.self_attn.need_weights = True
        last.self_attn.average_attn_weights = True

        h = self.input_proj(x)         # (B, N, d_enc)
        h = self.transformer(h, src_key_padding_mask=key_padding_mask)
        h = self.norm(h)               # (B, N, d_enc)

        # Mean pool over non-padded positions
        if key_padding_mask is not None:
            active = (~key_padding_mask).float().unsqueeze(-1)  # (B, N, 1)
            a_t = (h * active).sum(dim=1) / active.sum(dim=1).clamp(min=1)
        else:
            a_t = h.mean(dim=1)        # (B, d_enc)

        if self._attn_weights is not None:
            A_t = self._attn_weights   # (B, N, N)
        else:
            A_t = torch.zeros(B, N, N, device=x.device)

        return a_t, A_t
