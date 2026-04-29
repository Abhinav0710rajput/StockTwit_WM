"""
Two separate ticker embedding tables:
  e_ret : used only by the presence head (retrieval geometry)
  e_dec : used by the set encoder + feature MLPs (discriminative geometry)
"""

from __future__ import annotations

import torch
import torch.nn as nn


class TickerEmbeddings(nn.Module):
    def __init__(self, vocab_size: int, embed_dim: int, padding_idx: int = 0) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.e_ret = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self.e_dec = nn.Embedding(vocab_size, embed_dim, padding_idx=padding_idx)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.e_ret.weight, mean=0.0, std=0.01)
        nn.init.normal_(self.e_dec.weight, mean=0.0, std=0.01)
        # zero out padding row
        with torch.no_grad():
            self.e_ret.weight[0].fill_(0.0)
            self.e_dec.weight[0].fill_(0.0)

    def get_ret(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (...) → (..., E)"""
        return self.e_ret(ids)

    def get_dec(self, ids: torch.Tensor) -> torch.Tensor:
        """ids: (...) → (..., E)"""
        return self.e_dec(ids)

    def all_ret(self) -> torch.Tensor:
        """Return full e_ret table: (vocab_size, E)"""
        return self.e_ret.weight

    def all_dec(self) -> torch.Tensor:
        """Return full e_dec table: (vocab_size, E)"""
        return self.e_dec.weight
