"""
PyTorch Dataset for Twit Wave RSSM.

Each sample is a contiguous sequence of T+window_k weeks.
The first window_k weeks are warm-up for the Stage-2 temporal encoder
(no loss computed on them). The last T weeks are the training target.

Returned tensors per sample
---------------------------
Dynamic mode (for RSSM):
  features    : (T+k, top_k, 5)   float32  — normalised feature matrix
  ticker_ids  : (T+k, top_k)      int64    — vocab indices of active tickers
  presence    : (T+k, vocab_size)  float32  — binary, 1 if ticker active that week

Fixed mode (for baselines):
  features    : (T+k, K, 5)       float32  — zero-imputed fixed roster
  ticker_ids  : (T+k, K)          int64    — fixed roster vocab indices
  presence    : (T+k, K)          float32  — same as dynamic but over fixed K
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .vocab import Vocabulary

FEATURE_COLS = [
    "log_attention",
    "bullish_rate",
    "bearish_rate",
    "unlabeled_rate",
    "attn_growth",
]


class TwitWaveDataset(Dataset):
    def __init__(
        self,
        panel: pd.DataFrame,
        vocab: Vocabulary,
        split: Literal["train", "val", "test1", "test2"],
        mode: Literal["dynamic", "fixed"] = "dynamic",
        chunk_len: int = 52,          # T — loss weeks
        window_k: int = 4,            # Stage-2 warm-up weeks
        top_k: int = 100,
        norm_stats: dict | None = None,  # {"mean": arr, "std": arr} computed on train
        fixed_roster: list[str] | None = None,  # for baseline mode
    ) -> None:
        self.vocab = vocab
        self.mode = mode
        self.chunk_len = chunk_len
        self.window_k = window_k
        self.top_k = top_k

        # --- date splits ---
        split_ranges = {
            "train": ("2008-01-01", "2018-12-31"),
            "val":   ("2019-01-01", "2019-12-31"),
            "test1": ("2020-01-01", "2020-06-30"),
            "test2": ("2021-01-01", "2021-03-31"),
        }
        lo, hi = split_ranges[split]
        panel = panel.copy()
        panel["week"] = pd.to_datetime(panel["week"])
        mask = (panel["week"] >= lo) & (panel["week"] <= hi)
        self.panel = panel[mask].sort_values(["week", "symbol"]).reset_index(drop=True)

        self.weeks = sorted(self.panel["week"].unique())
        self.total_weeks = len(self.weeks)
        self.seq_len = chunk_len + window_k   # total weeks per sample

        # --- normalization ---
        if norm_stats is None:
            norm_stats = self._compute_norm_stats()
        self.mean = norm_stats["mean"]   # (5,)
        self.std  = norm_stats["std"]    # (5,)

        # --- fixed roster (baseline mode) ---
        if mode == "fixed":
            if fixed_roster is None:
                # top-K tickers by total msg_count across split
                top = (
                    self.panel.groupby("symbol")["msg_count"]
                    .sum()
                    .nlargest(top_k)
                    .index.tolist()
                )
                fixed_roster = sorted(top)
            self.fixed_roster = fixed_roster
            self.fixed_ids = torch.tensor(
                vocab.encode_list(fixed_roster), dtype=torch.long
            )   # (K,)
        else:
            self.fixed_roster = None
            self.fixed_ids = None

        # --- build week → feature matrix lookup (fast) ---
        self._week_data: dict[pd.Timestamp, pd.DataFrame] = {}
        for wk, grp in self.panel.groupby("week"):
            self._week_data[wk] = grp.reset_index(drop=True)

    # ------------------------------------------------------------------
    def _compute_norm_stats(self) -> dict:
        vals = self.panel[FEATURE_COLS].values.astype(np.float32)
        mean = vals.mean(axis=0)
        std  = vals.std(axis=0) + 1e-6
        return {"mean": mean, "std": std}

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return max(0, self.total_weeks - self.seq_len + 1)

    # ------------------------------------------------------------------
    def __getitem__(self, idx: int) -> dict:
        weeks_slice = self.weeks[idx : idx + self.seq_len]

        if self.mode == "dynamic":
            return self._get_dynamic(weeks_slice)
        else:
            return self._get_fixed(weeks_slice)

    # ------------------------------------------------------------------
    def _get_dynamic(self, weeks: list) -> dict:
        T = len(weeks)
        feat_list, ids_list, pres_list = [], [], []

        for wk in weeks:
            grp = self._week_data.get(wk)
            n = min(len(grp), self.top_k) if grp is not None else 0

            # features (top_k, 5)
            feat = np.zeros((self.top_k, 5), dtype=np.float32)
            if n > 0:
                vals = grp[FEATURE_COLS].values[:n].astype(np.float32)
                vals = (vals - self.mean) / self.std
                feat[:n] = vals

            # ticker ids (top_k,)
            ids = np.zeros(self.top_k, dtype=np.int64)
            if n > 0:
                syms = grp["symbol"].tolist()[:n]
                ids[:n] = self.vocab.encode_list(syms)

            # presence (vocab_size,)
            pres = np.zeros(self.vocab.size, dtype=np.float32)
            if n > 0:
                pres[ids[:n]] = 1.0

            feat_list.append(feat)
            ids_list.append(ids)
            pres_list.append(pres)

        return {
            "features":   torch.from_numpy(np.stack(feat_list)),   # (T, top_k, 5)
            "ticker_ids": torch.from_numpy(np.stack(ids_list)),    # (T, top_k)
            "presence":   torch.from_numpy(np.stack(pres_list)),   # (T, vocab_size)
        }

    # ------------------------------------------------------------------
    def _get_fixed(self, weeks: list) -> dict:
        K = len(self.fixed_roster)
        T = len(weeks)
        feat_list, pres_list = [], []

        sym_to_col = {s: i for i, s in enumerate(self.fixed_roster)}

        for wk in weeks:
            grp = self._week_data.get(wk)
            feat = np.zeros((K, 5), dtype=np.float32)
            pres = np.zeros(K, dtype=np.float32)

            if grp is not None:
                for _, row in grp.iterrows():
                    col = sym_to_col.get(row["symbol"])
                    if col is not None:
                        vals = row[FEATURE_COLS].values.astype(np.float32)
                        feat[col] = (vals - self.mean) / self.std
                        pres[col] = 1.0

            feat_list.append(feat)
            pres_list.append(pres)

        ids = self.fixed_ids.unsqueeze(0).expand(T, -1)   # (T, K)

        return {
            "features":   torch.from_numpy(np.stack(feat_list)),   # (T, K, 5)
            "ticker_ids": ids.clone(),
            "presence":   torch.from_numpy(np.stack(pres_list)),   # (T, K)
        }


# ------------------------------------------------------------------
# Collate functions
# ------------------------------------------------------------------

def collate_dynamic(batch: list[dict]) -> dict:
    return {
        "features":   torch.stack([b["features"]   for b in batch]),
        "ticker_ids": torch.stack([b["ticker_ids"] for b in batch]),
        "presence":   torch.stack([b["presence"]   for b in batch]),
    }


def collate_fixed(batch: list[dict]) -> dict:
    return collate_dynamic(batch)
