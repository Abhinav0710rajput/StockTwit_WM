"""
Reduced-rank VAR baseline.

Fits a Vector Autoregression on the fixed-roster log_attention series,
with a low-rank constraint on the coefficient matrix (via SVD truncation).
Captures linear cross-ticker coupling but assumes static, stationary dynamics.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR
from tqdm import tqdm


class ReducedRankVAR:
    def __init__(self, maxlags: int = 4, rank: int = 10) -> None:
        self.maxlags = maxlags
        self.rank    = rank
        self.model   = None
        self.result  = None
        self.tickers: list[str] = []

    def fit(
        self,
        panel: pd.DataFrame,
        tickers: list[str],
        log_attn_col: str = "log_attention",
        week_col: str = "week",
        symbol_col: str = "symbol",
    ) -> None:
        """
        Pivot panel to (T, K) matrix and fit VAR.
        Missing values (ticker not active that week) are zero-filled.
        """
        self.tickers = tickers
        pivoted = (
            panel[panel[symbol_col].isin(tickers)]
            .pivot_table(index=week_col, columns=symbol_col, values=log_attn_col)
            .reindex(columns=tickers)
            .fillna(0.0)
            .sort_index()
        )
        self._train_weeks = pivoted.index.tolist()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model  = VAR(pivoted.values)
            self.result = self.model.fit(maxlags=self.maxlags, ic="aic")

        # Apply low-rank truncation to coefficient matrices
        self._apply_low_rank()
        print(f"[VAR] fitted lag={self.result.k_ar}, rank={self.rank}, K={len(tickers)}")

    def _apply_low_rank(self) -> None:
        """Truncate VAR coefficient matrices to rank r via SVD."""
        if self.result is None:
            return
        coefs = self.result.coefs.copy()   # (p, K, K)
        for l in range(coefs.shape[0]):
            U, S, Vt = np.linalg.svd(coefs[l], full_matrices=False)
            r = min(self.rank, len(S))
            coefs[l] = (U[:, :r] * S[:r]) @ Vt[:r]
        self._coefs = coefs
        self._intercept = self.result.coefs_exog[0] if self.result.coefs_exog is not None else np.zeros(len(self.tickers))

    def forecast(self, last_obs: np.ndarray, steps: int = 1) -> np.ndarray:
        """
        last_obs : (p, K) — last p observations
        Returns (steps, K) forecast.
        """
        p = self._coefs.shape[0]
        K = len(self.tickers)
        history = last_obs[-p:].copy()   # (p, K)
        preds = []
        for _ in range(steps):
            y_hat = self._intercept.copy()
            for l in range(p):
                y_hat += history[-(l + 1)] @ self._coefs[l].T
            preds.append(y_hat)
            history = np.vstack([history[1:], y_hat])
        return np.stack(preds)   # (steps, K)
