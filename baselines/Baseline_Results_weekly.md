# Baseline Results — Weekly Data

## Setup

- **Data:** `data/processed_week/`
- **Granularity:** Weekly aggregation
- **Train:** 2008-01-01 → 2018-12-31 (554 weeks)
- **Val:** 2019-01-01 → 2019-12-31 (52 weeks)
- **Test1:** 2020-01-06 → 2020-06-29 (26 weeks) — COVID crash
- **Test2:** 2020-10-05 → 2021-06-28 (39 weeks) — GME squeeze
- **Roster:** Top-100 tickers by training message count → 65 common tickers across all splits
- **Horizons:** h=1 (next week), h=4 (~1 month), h=13 (~1 quarter)
- **Primary metric:** Spearman ρ (rank correlation of log_attention across tickers)

> **Note:** P@100 and AUC-ROC virality not computed for weekly baselines.
> Val split results not available for weekly baselines.

---

## ARIMA — PerTickerARIMA(order=(2,0,1))

Independent ARIMA model per ticker. No cross-ticker dynamics, no regime awareness.

### Test1 — COVID Crash

| Horizon | MSE | MAE | Spearman ρ |
|---------|-----|-----|------------|
| h=1 (1 week) | 0.5378 | 0.5386 | 0.5956 |
| h=4 (1 month) | 0.5706 | 0.5537 | 0.5366 |
| h=13 (1 quarter) | 1.1493 | 0.7938 | 0.4273 |

### Test2 — GME Squeeze

| Horizon | MSE | MAE | Spearman ρ |
|---------|-----|-----|------------|
| h=1 (1 week) | 2.9953 | 0.8538 | 0.4457 |
| h=4 (1 month) | 2.0926 | 0.8238 | 0.4611 |
| h=13 (1 quarter) | 3.8198 | 1.0578 | 0.3482 |

---

## VAR — ReducedRankVAR(maxlags=1, rank=10)

Vector Autoregression on fixed roster. Linear cross-ticker coupling, static dynamics.
> maxlags reduced to 1 due to limited training weeks relative to ticker count (65 tickers, 554 weeks).

### Test1 — COVID Crash

| Horizon | MSE | MAE | Spearman ρ |
|---------|-----|-----|------------|
| h=1 (1 week) | 47.5777 | 5.8185 | -0.0250 |
| h=4 (1 month) | 54.0309 | 6.1969 | 0.1034 |
| h=13 (1 quarter) | 52.7401 | 6.0326 | -0.0622 |

### Test2 — GME Squeeze

| Horizon | MSE | MAE | Spearman ρ |
|---------|-----|-----|------------|
| h=1 (1 week) | 39.3817 | 5.2962 | -0.0462 |
| h=4 (1 month) | 51.3546 | 6.3470 | -0.1414 |
| h=13 (1 quarter) | 41.4797 | 5.3113 | -0.0392 |

---

## LSTM — SharedLSTM(hidden=512, layers=2, epochs=100)

Single LSTM on concatenated fixed-roster features. Implicit cross-ticker coupling via shared hidden state. No explicit regime separation.
- SEQ_LEN: 8 weeks
- Device: Apple MPS (M4)
- Optimizer: Adam lr=3e-4 with CosineAnnealingLR

### Test1 — COVID Crash

| Horizon | MSE | MAE | Spearman ρ |
|---------|-----|-----|------------|
| h=1 (1 week) | 5.9190 | 1.8160 | 0.5139 |
| h=4 (1 month) | 7.6435 | 2.0707 | 0.5380 |
| h=13 (1 quarter) | 11.2785 | 2.6092 | 0.5535 |

### Test2 — GME Squeeze

| Horizon | MSE | MAE | Spearman ρ |
|---------|-----|-----|------------|
| h=1 (1 week) | 16.1714 | 3.3477 | 0.3245 |
| h=4 (1 month) | 12.3459 | 2.8106 | 0.4808 |
| h=13 (1 quarter) | 17.2411 | 3.4343 | 0.4190 |

---

## Summary — Spearman ρ

### Test1 — COVID Crash

| Model | h=1 | h=4 | h=13 |
|-------|-----|-----|------|
| ARIMA | **0.596** | **0.537** | 0.427 |
| VAR | -0.025 | 0.103 | -0.062 |
| LSTM | 0.514 | 0.538 | **0.554** |
| TwitWave | ⏳ | ⏳ | ⏳ |

### Test2 — GME Squeeze

| Model | h=1 | h=4 | h=13 |
|-------|-----|-----|------|
| ARIMA | **0.446** | 0.461 | 0.348 |
| VAR | -0.046 | -0.141 | -0.039 |
| LSTM | 0.325 | **0.481** | 0.419 |
| TwitWave | ⏳ | ⏳ | ⏳ |

---

## Key Observations

**ARIMA is the strongest baseline at h=1** on both test sets — simple per-ticker models outperform complex ones at short horizons when the train-test regime gap is large.

**VAR completely collapses** — near-zero and negative Spearman ρ across both test sets. Static linear coefficients trained on 2008-2018 have no generalization to COVID/GME dynamics.

**LSTM beats ARIMA at longer horizons** — LSTM wins at h=13 on COVID (0.55 vs 0.43), showing that cross-ticker coupling helps at longer prediction windows.

**GME is harder than COVID for all models** — ARIMA ρ drops from 0.60 (COVID h=1) to 0.45 (GME h=1). GME is idiosyncratic and nonlinear — exactly what TwitWave's latent regime model is designed to handle.

**20% of roster tickers disappeared** — 35 out of 100 training tickers had zero activity during test periods (e.g. DRYS, FIT, UWTI). This directly demonstrates the limitation of fixed-roster baselines and motivates TwitWave's dynamic set design.
