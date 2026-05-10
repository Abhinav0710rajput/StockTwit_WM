# Baseline Results — Daily Data

## Setup

- **Data:** `data/processed_day/by_split_month/`
- **Granularity:** Daily aggregation
- **Train:** 2008-05-27 → 2018-12-31 (3,870 days)
- **Val:** 2019-01-01 → 2019-12-31 (365 days)
- **Test1:** 2020-01-01 → 2020-06-30 (182 days) — COVID crash
- **Test2:** 2020-10-01 → 2021-06-30 (273 days) — GME squeeze
- **Roster:** Top-100 tickers by training message count → 79 common tickers across all splits
- **Horizons:** h=1 (next day), h=4 (~1 week), h=13 (~2.5 weeks)
- **Primary metric:** Spearman ρ (rank correlation of log_attention across tickers)

> **Note:** P@100 always = 1.0 for baselines (roster size 79 < 100) — not a meaningful metric here.
> P@100 will be meaningful for TwitWave which predicts from a vocab of 1000+ tickers.

---

## ARIMA — PerTickerARIMA(order=(2,0,1))

Independent ARIMA model per ticker. No cross-ticker dynamics, no regime awareness.

### Val — 2019 (Stable Period)

| Horizon | MSE | MAE | Spearman ρ | AUC-ROC |
|---------|-----|-----|------------|---------|
| h=1 (1 day) | 1.2346 | 0.9609 | 0.7290 | 0.8468 |
| h=4 (4 days) | 1.1517 | 0.8897 | 0.5677 | 0.7882 |
| h=13 (13 days) | 1.2042 | 0.8777 | 0.2487 | 0.7513 |

### Test1 — COVID Crash

| Horizon | MSE | MAE | Spearman ρ | AUC-ROC |
|---------|-----|-----|------------|---------|
| h=1 (1 day) | 1.7123 | 1.1358 | 0.4478 | 0.8344 |
| h=4 (4 days) | 1.5292 | 1.0092 | 0.3691 | 0.8024 |
| h=13 (13 days) | 1.6844 | 1.0558 | 0.2290 | 0.6846 |

### Test2 — GME Squeeze

| Horizon | MSE | MAE | Spearman ρ | AUC-ROC |
|---------|-----|-----|------------|---------|
| h=1 (1 day) | 1.2289 | 0.8536 | 0.4922 | 0.7808 |
| h=4 (4 days) | 1.0962 | 0.8844 | 0.3318 | 0.7407 |
| h=13 (13 days) | 2.2204 | 1.1910 | 0.3148 | 0.7385 |

---

## VAR — ReducedRankVAR(maxlags=4, rank=10)

Vector Autoregression on fixed roster. Linear cross-ticker coupling, static dynamics.
> maxlags=4 feasible with daily data (3,870 days, 79 tickers).

### Val — 2019 (Stable Period)

| Horizon | MSE | MAE | Spearman ρ | AUC-ROC |
|---------|-----|-----|------------|---------|
| h=1 (1 day) | 4.0805 | 1.5570 | 0.5177 | — |
| h=4 (4 days) | 7.7180 | 2.3696 | 0.5011 | — |
| h=13 (13 days) | 13.8469 | 3.3190 | 0.3586 | — |

### Test1 — COVID Crash

| Horizon | MSE | MAE | Spearman ρ | AUC-ROC |
|---------|-----|-----|------------|---------|
| h=1 (1 day) | 5.7274 | 1.8671 | 0.3331 | — |
| h=4 (4 days) | 4.3939 | 1.6674 | 0.3643 | — |
| h=13 (13 days) | 18.7592 | 3.5930 | 0.2452 | — |

### Test2 — GME Squeeze

| Horizon | MSE | MAE | Spearman ρ | AUC-ROC |
|---------|-----|-----|------------|---------|
| h=1 (1 day) | 11.0210 | 2.7392 | 0.2405 | — |
| h=4 (4 days) | 6.9443 | 2.0326 | 0.2277 | — |
| h=13 (13 days) | 16.8549 | 3.2179 | 0.0973 | — |

> AUC-ROC not computed for VAR daily — to be added.

---

## LSTM — SharedLSTM(hidden=512, layers=2, epochs=150)

Single LSTM on concatenated fixed-roster features. Implicit cross-ticker coupling via shared hidden state. No explicit regime separation.
- SEQ_LEN: 20 days (~1 month lookback)
- Device: Apple MPS (M4)
- Optimizer: Adam lr=3e-4 with CosineAnnealingLR(T_max=150)
- Best model selected by validation loss

### Val — 2019 (Stable Period)

| Horizon | MSE | MAE | Spearman ρ | AUC-ROC |
|---------|-----|-----|------------|---------|
| h=1 (1 day) | 1.4114 | 0.9289 | 0.7938 | — |
| h=4 (4 days) | 2.9890 | 1.4334 | 0.7762 | — |
| h=13 (13 days) | 2.2139 | 1.1368 | 0.5670 | — |

### Test1 — COVID Crash

| Horizon | MSE | MAE | Spearman ρ | AUC-ROC |
|---------|-----|-----|------------|---------|
| h=1 (1 day) | 2.6267 | 1.2868 | 0.6827 | — |
| h=4 (4 days) | 2.1600 | 1.1553 | 0.6774 | — |
| h=13 (13 days) | 5.9537 | 2.0610 | 0.4994 | — |

### Test2 — GME Squeeze

| Horizon | MSE | MAE | Spearman ρ | AUC-ROC |
|---------|-----|-----|------------|---------|
| h=1 (1 day) | 8.1574 | 2.3433 | 0.5085 | — |
| h=4 (4 days) | 5.9249 | 1.9321 | 0.4861 | — |
| h=13 (13 days) | 7.0242 | 2.2478 | 0.5091 | — |

> AUC-ROC for VAR and LSTM daily to be added after rerun with full eval code.

---

## Summary — Spearman ρ

### Val — 2019 (Stable Period)

| Model | h=1 | h=4 | h=13 |
|-------|-----|-----|------|
| ARIMA | 0.729 | 0.568 | 0.249 |
| VAR | 0.518 | 0.501 | 0.359 |
| LSTM | **0.794** | **0.776** | **0.567** |
| TwitWave | ⏳ | ⏳ | ⏳ |

### Test1 — COVID Crash

| Model | h=1 | h=4 | h=13 |
|-------|-----|-----|------|
| ARIMA | **0.448** | 0.369 | 0.229 |
| VAR | 0.333 | **0.364** | 0.245 |
| LSTM | 0.683 | 0.677 | **0.499** |
| TwitWave | ⏳ | ⏳ | ⏳ |

### Test2 — GME Squeeze

| Model | h=1 | h=4 | h=13 |
|-------|-----|-----|------|
| ARIMA | **0.492** | 0.332 | 0.315 |
| VAR | 0.241 | 0.228 | 0.097 |
| LSTM | 0.509 | **0.486** | **0.509** |
| TwitWave | ⏳ | ⏳ | ⏳ |

### AUC-ROC Virality (ARIMA only, top-20 within 4 days)

| Split | h=1 | h=4 | h=13 |
|-------|-----|-----|------|
| Val | 0.847 | 0.788 | 0.751 |
| Test1 (COVID) | 0.834 | 0.802 | 0.685 |
| Test2 (GME) | 0.781 | 0.741 | 0.739 |

---

## Key Observations

**LSTM dominates on daily data** — beats ARIMA everywhere, especially at longer horizons. With 3,870 training days, LSTM has enough data to learn cross-ticker patterns effectively.

**Val performance is strong across all models** — LSTM reaches ρ=0.79 at h=1 during the stable 2019 period. This is the bar TwitWave needs to beat during normal conditions.

**Performance degrades on regime-shift periods** — all models drop significantly on Test1 and Test2 vs Val. This is the core motivation for TwitWave's latent regime design.

**LSTM h=13 on GME (ρ=0.51)** — this is the key number TwitWave needs to beat to justify the world model approach on the hardest test case.

**VAR still underperforms** — despite having 4 lags with daily data, static linear coefficients can't adapt to regime changes.

**Daily data favors LSTM over ARIMA** — with more training data points, the non-linear LSTM learns better patterns than simple per-ticker autoregression.

**Data sparsity issue** — some tickers have zero messages on certain days, causing valid ticker counts to drop below 79 on individual evaluation days. AUC-ROC computed only on tickers active on each evaluation day.

**21 tickers dropped from roster** — from 100 training tickers to 79 common tickers. Same fixed-roster limitation as weekly, but less severe (79 vs 65 common tickers).
