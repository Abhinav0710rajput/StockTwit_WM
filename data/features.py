"""
Feature engineering: raw parquet → ticker×week panel.

Reads feature_wo_messages parquet (partitioned year=*/month=*),
groups by (symbol, week_start), computes 4 raw counts and 5 model features,
then writes data/processed/panel.parquet.
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd
import numpy as np


FEATURE_COLS = [
    "log_attention",
    "bullish_rate",
    "bearish_rate",
    "unlabeled_rate",
    "attn_growth",
]

RAW_COUNT_COLS = ["msg_count", "user_count", "bullish_count", "labeled_count"]


def build_panel(
    parquet_dir: str | Path,
    output_path: str | Path,
    start_year: int = 2008,
    end_year: int = 2022,
    top_k: int = 100,
) -> pd.DataFrame:
    """
    Build the ticker×week panel from raw parquet files.

    Parameters
    ----------
    parquet_dir : path to feature_wo_messages parquet root (year=*/month=*)
    output_path : where to save panel.parquet
    start_year / end_year : inclusive year bounds
    top_k : number of top tickers per week to keep

    Returns
    -------
    DataFrame with columns: symbol, week, msg_count, user_count,
        bullish_count, labeled_count, log_attention, bullish_rate,
        bearish_rate, unlabeled_rate, attn_growth
    """
    parquet_dir = Path(parquet_dir)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    glob = (parquet_dir / "year=*" / "month=*" / "*.parquet").as_posix()

    con = duckdb.connect()

    # --- Step 1: aggregate to symbol × week ---
    raw = con.execute(f"""
        WITH base AS (
            SELECT
                UNNEST(regexp_extract_all(symbol_list, '''([^'']+)''', 1)) AS symbol,
                user_id,
                sentiment,
                date_trunc('week', created_at::TIMESTAMP)::DATE AS week
            FROM read_parquet('{glob}', hive_partitioning = true)
            WHERE year >= {start_year} AND year <= {end_year}
              AND symbol_list IS NOT NULL
              AND symbol_list != '[]'
        )
        SELECT
            symbol,
            week,
            COUNT(*)                                              AS msg_count,
            COUNT(DISTINCT user_id)                               AS user_count,
            SUM(CASE WHEN sentiment = 'Bullish' THEN 1 ELSE 0 END) AS bullish_count,
            SUM(CASE WHEN sentiment IS NOT NULL THEN 1 ELSE 0 END)  AS labeled_count
        FROM base
        WHERE symbol != ''
        GROUP BY symbol, week
        ORDER BY week, symbol
    """).df()

    print(f"[features] raw panel: {len(raw):,} rows, {raw['symbol'].nunique():,} symbols, "
          f"{raw['week'].nunique():,} weeks")

    # --- Step 2: compute features ---
    raw["log_attention"] = np.log1p(raw["msg_count"].astype(float))
    raw["bullish_rate"] = np.where(
        raw["labeled_count"] > 0,
        raw["bullish_count"] / raw["labeled_count"],
        0.0,
    )
    raw["bearish_rate"] = 1.0 - raw["bullish_rate"]
    raw["unlabeled_rate"] = 1.0 - np.where(
        raw["msg_count"] > 0,
        raw["labeled_count"] / raw["msg_count"],
        1.0,
    )

    # --- Step 3: keep only top_k tickers per week ---
    raw = raw.sort_values(["week", "msg_count"], ascending=[True, False])
    raw["rank"] = raw.groupby("week")["msg_count"].rank(method="first", ascending=False)
    panel = raw[raw["rank"] <= top_k].copy()
    panel = panel.drop(columns=["rank"])

    # --- Step 4: attn_growth (WoW % change in log_attention) ---
    panel = panel.sort_values(["symbol", "week"]).reset_index(drop=True)
    panel["prev_log_attn"] = panel.groupby("symbol")["log_attention"].shift(1)
    panel["attn_growth"] = (
        (panel["log_attention"] - panel["prev_log_attn"])
        / (panel["prev_log_attn"].abs() + 1e-6)
    ).clip(-5.0, 5.0)
    # First appearance of a ticker → growth = 0
    panel["attn_growth"] = panel["attn_growth"].fillna(0.0)
    panel = panel.drop(columns=["prev_log_attn"])

    # --- Step 5: save ---
    panel = panel.sort_values(["week", "symbol"]).reset_index(drop=True)
    panel.to_parquet(output_path, index=False)
    print(f"[features] panel saved: {len(panel):,} rows → {output_path}")

    return panel
