"""
Per-ticker ARIMA baseline.

Fits independent ARIMA(p,d,q) models for each ticker's log_attention series.
Cannot model cross-ticker dynamics — serves as the floor in the ablation ladder.
Operates on the fixed-roster data format (zero-imputed).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from tqdm import tqdm


class PerTickerARIMA:
    def __init__(self, order: tuple = (2, 0, 1)) -> None:
        self.order = order
        self.models: dict[str, object] = {}   # ticker → fitted ARIMAResults
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
        Fit ARIMA independently per ticker on training data.

        panel should be the training split of the panel DataFrame.
        """
        self.tickers = tickers
        for ticker in tqdm(tickers, desc="ARIMA fit"):
            series = (
                panel[panel[symbol_col] == ticker]
                .sort_values(week_col)[log_attn_col]
                .values
            )
            if len(series) < 10:
                self.models[ticker] = None
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    m = ARIMA(series, order=self.order)
                    self.models[ticker] = m.fit()
                except Exception:
                    self.models[ticker] = None

    def forecast(self, steps: int) -> dict[str, np.ndarray]:
        """
        Forecast `steps` steps ahead for each ticker.
        Returns dict: ticker → (steps,) array of log_attention predictions.
        """
        preds = {}
        for ticker in self.tickers:
            m = self.models.get(ticker)
            if m is None:
                preds[ticker] = np.zeros(steps)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    preds[ticker] = m.forecast(steps=steps)
        return preds

    def predict_panel(
        self,
        panel_test: pd.DataFrame,
        steps_ahead: int = 1,
        log_attn_col: str = "log_attention",
        week_col: str = "week",
        symbol_col: str = "symbol",
    ) -> pd.DataFrame:
        """
        Rolling one-step-ahead prediction over the test panel.
        Returns DataFrame with columns: week, symbol, pred_log_attn, true_log_attn.
        """
        rows = []
        weeks = sorted(panel_test[week_col].unique())

        for ticker in tqdm(self.tickers, desc="ARIMA predict"):
            series = (
                panel_test[panel_test[symbol_col] == ticker]
                .sort_values(week_col)
            )
            m = self.models.get(ticker)
            if m is None or len(series) == 0:
                continue

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # In-sample + forecast
                fitted = m.fittedvalues
                fc = m.forecast(steps=len(series))
                for i, (_, row) in enumerate(series.iterrows()):
                    rows.append({
                        "week":           row[week_col],
                        "symbol":         ticker,
                        "pred_log_attn":  float(fc[i]) if i < len(fc) else 0.0,
                        "true_log_attn":  float(row[log_attn_col]),
                    })

        return pd.DataFrame(rows)
