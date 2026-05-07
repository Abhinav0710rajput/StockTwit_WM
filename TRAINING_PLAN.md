# TwitWave RSSM — Training & Eval Task Plan

**Data:** `data/processed_week/panel_all.parquet`  
**Vocab size:** 1520 tickers | **Train weeks:** 554 (2008–2018) | **Val:** 52 (2019) | **Test1:** 26 (2020 H1) | **Test2:** 39 (2021 Q1)  
**Conda env:** `wm_ml` | **Run all scripts from project root with** `PYTHONPATH=/Users/ush/Desktop/StockTwit_WM`

---

## Phase 0 — Bug Fixes ✅ DONE

All fixed and verified with smoke test on 2026-05-06.

| # | File | Fix |
|---|------|-----|
| 0.1 | `scripts/2_b_train_rssm.py` | `--data_dir` default → `data/processed_week` |
| 0.2 | `scripts/2_b_train_rssm.py` | `seq_len=` → `chunk_len=`, removed `normalise=True` |
| 0.3 | `data/dataset.py` | Added `norm_stats` property exposing `{"mean", "std"}` |
| 0.4 | `scripts/2_b_train_rssm.py` | Trainer now receives a `config` dict; maps `lambda_mse` → `lambda_`, `bce_pos_weight` → `pos_weight`; `--resume` handled via `trainer.load_checkpoint()` |
| 0.5 | `data/__init__.py` | `build_panel` made a lazy import so `duckdb` is not required for training |
| 0.6 | `wm_ml` env | Downgraded `numpy 2.4.3 → 1.26.4` (torch 2.2.2 requires NumPy <2) |

---

## Phase 1 — Smoke Test ✅ DONE

```bash
conda run -n wm_ml env PYTHONPATH=/Users/ush/Desktop/StockTwit_WM \
  python scripts/2_b_train_rssm.py \
  --cfg configs/debug.yaml \
  --out_dir /tmp/tw_smoke_test \
  --seed 42
```

**Expected:** 2 epochs in ~25s on CPU, `best.pt` saved, `kl_log.json` written.  
**Result:** ✅ train_loss=3.01, val_loss=3.51, checkpoint saved.

---

## Phase 2 — Full RSSM Training

### 2.1 Base config (primary run)

```bash
conda run -n wm_ml env PYTHONPATH=/Users/ush/Desktop/StockTwit_WM \
  python scripts/2_b_train_rssm.py \
  --cfg configs/rssm_base.yaml \
  --out_dir outputs/rssm_base \
  --seed 42
```

**Config highlights** (`configs/rssm_base.yaml`):

| Param | Value |
|-------|-------|
| embed_dim | 64 |
| h_dim / s_dim | 512 / 256 (z_dim = 768) |
| n_heads / n_layers | 8 / 2 |
| seq_len (BPTT) | 52 weeks |
| batch_size | 32 |
| max_epochs | 100 |
| β anneal epochs | 30 |
| early stopping patience | 15 |

**Outputs:**
- `outputs/rssm_base/checkpoints/best.pt` — best val ELBO checkpoint
- `outputs/rssm_base/checkpoints/epoch_NNN.pt` — every 5 epochs
- `outputs/rssm_base/logs/kl_log.json` — per-step KL time series
- `outputs/rssm_base/norm_stats.json` — train-set normalisation stats

**Things to watch during training:**
- `val_loss` should decrease and stabilise; if it diverges after epoch ~10 → reduce `lr`
- `kl` should rise slowly from ~0 during β annealing (epochs 0–30), then plateau
- If KL collapses to `free_nats=3.0` and stays there → posterior collapse; increase `beta_end` or reduce model size

### 2.2 Resume from checkpoint (if interrupted)

```bash
conda run -n wm_ml env PYTHONPATH=/Users/ush/Desktop/StockTwit_WM \
  python scripts/2_b_train_rssm.py \
  --cfg configs/rssm_base.yaml \
  --out_dir outputs/rssm_base \
  --resume outputs/rssm_base/checkpoints/best.pt \
  --seed 42
```

### 2.3 Large config (optional, A100 only)

```bash
conda run -n wm_ml env PYTHONPATH=/Users/ush/Desktop/StockTwit_WM \
  python scripts/2_b_train_rssm.py \
  --cfg configs/rssm_large.yaml \
  --out_dir outputs/rssm_large \
  --seed 42
```

---

## Phase 3 — Baseline Training

Train ARIMA, VAR, and LSTM baselines for comparison metrics.

```bash
conda run -n wm_ml env PYTHONPATH=/Users/ush/Desktop/StockTwit_WM \
  python scripts/2_c_train_baselines.py \
  --data_dir data/processed_week
```

**Outputs:** `outputs/baselines/` — per-model forecasts and metrics.  
**Dependency:** Must complete before Phase 4.1.

---

## Phase 4 — Evaluation

Run in order. Each script reads `outputs/rssm_base/checkpoints/best.pt`.

### 4.1 Prediction metrics

```bash
conda run -n wm_ml env PYTHONPATH=/Users/ush/Desktop/StockTwit_WM \
  python scripts/3_a_eval_prediction.py \
  --ckpt outputs/rssm_base/checkpoints/best.pt \
  --cfg configs/rssm_base.yaml \
  --data_dir data/processed_week \
  --out_dir outputs/rssm_base/results
```

**What it computes:**
- Presence AUC-ROC and Precision@100 at horizons 1 / 4 / 13 weeks
- Virality AUC (top-20 tickers within 4-week window)
- Feature MSE vs. ground truth
- All metrics compared against baselines from Phase 3

**Method:** `context_phase()` warms up `(h_T, s_T)` over 52-week context window, then `forward_step_prior()` rolls out for each horizon.

### 4.2 KL analysis

```bash
conda run -n wm_ml env PYTHONPATH=/Users/ush/Desktop/StockTwit_WM \
  python scripts/3_b_eval_kl.py \
  --log outputs/rssm_base/logs/kl_log.json \
  --out_dir outputs/rssm_base/results
```

**What it computes:**
- KL_t time series plot over training
- Spike detection (z-score threshold = 2.0) — flags epochs where the model was "surprised"
- Confirms no posterior collapse

### 4.3 Attention analysis

```bash
conda run -n wm_ml env PYTHONPATH=/Users/ush/Desktop/StockTwit_WM \
  python scripts/3_c_eval_attention.py \
  --ckpt outputs/rssm_base/checkpoints/best.pt \
  --cfg configs/rssm_base.yaml \
  --data_dir data/processed_week \
  --out_dir outputs/rssm_base/results
```

**What it computes:**
- Per-week `A_t` attention weight matrices from the set encoder
- Top-50 most-attended tickers per period
- Interpretability / paper figures for which tickers drive the latent state

### 4.4 Latent clustering

```bash
conda run -n wm_ml env PYTHONPATH=/Users/ush/Desktop/StockTwit_WM \
  python scripts/3_d_eval_latent.py \
  --ckpt outputs/rssm_base/checkpoints/best.pt \
  --cfg configs/rssm_base.yaml \
  --data_dir data/processed_week \
  --out_dir outputs/rssm_base/results
```

**What it computes:**
- t-SNE + UMAP projections of `z_t` latent states
- K-Means (k=5) regime clusters
- Cluster labels aligned with known market events (COVID crash 2020 H1, meme-stock rally 2021 Q1)

### 4.5 Counterfactual analysis

```bash
conda run -n wm_ml env PYTHONPATH=/Users/ush/Desktop/StockTwit_WM \
  python scripts/3_e_eval_counterfactual.py \
  --ckpt outputs/rssm_base/checkpoints/best.pt \
  --cfg configs/rssm_base.yaml \
  --data_dir data/processed_week \
  --out_dir outputs/rssm_base/results
```

**What it computes:**
- Perturbs a single ticker's `log_attention` by +3.0 σ (`cf_delta`)
- Re-rolls the prior and measures presence probability change for all other tickers
- Tests whether the latent space captures contagion / attention spillover dynamics

---

## Output Directory Layout (after full run)

```
outputs/
└── rssm_base/
    ├── config.yaml              # exact config used
    ├── norm_stats.json          # train normalisation stats
    ├── checkpoints/
    │   ├── best.pt              # best val ELBO
    │   └── epoch_NNN.pt        # every 5 epochs
    ├── logs/
    │   └── kl_log.json         # per-step KL series
    └── results/
        ├── prediction_metrics.json
        ├── kl_analysis.*
        ├── attention_analysis.*
        ├── latent_clusters.*
        └── counterfactual.*
```

---

## Quick Reference — Key Commands

| Action | Command |
|--------|---------|
| Smoke test | `python scripts/2_b_train_rssm.py --cfg configs/debug.yaml --out_dir /tmp/tw_smoke` |
| Full train | `python scripts/2_b_train_rssm.py --cfg configs/rssm_base.yaml --out_dir outputs/rssm_base` |
| Resume | add `--resume outputs/rssm_base/checkpoints/best.pt` |
| Baselines | `python scripts/2_c_train_baselines.py --data_dir data/processed_week` |
| Eval prediction | `python scripts/3_a_eval_prediction.py --ckpt outputs/rssm_base/checkpoints/best.pt ...` |
| Eval KL | `python scripts/3_b_eval_kl.py --log outputs/rssm_base/logs/kl_log.json ...` |
| Eval attention | `python scripts/3_c_eval_attention.py --ckpt outputs/rssm_base/checkpoints/best.pt ...` |
| Eval latent | `python scripts/3_d_eval_latent.py --ckpt outputs/rssm_base/checkpoints/best.pt ...` |
| Eval counterfactual | `python scripts/3_e_eval_counterfactual.py --ckpt outputs/rssm_base/checkpoints/best.pt ...` |

> All commands must be run from `/Users/ush/Desktop/StockTwit_WM` with:
> `conda run -n wm_ml env PYTHONPATH=/Users/ush/Desktop/StockTwit_WM`
