# Twit Wave: A World Model for Collective Attention Dynamics in Financial Social Media

**Abhinav Rajput** · NYU Stern School of Business  
*Working paper — targeting Management Science*

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Key Contributions](#2-key-contributions)
3. [Architecture Deep-Dive](#3-architecture-deep-dive)
   - [3.1 Data Representation: Dynamic Ticker Sets](#31-data-representation-dynamic-ticker-sets)
   - [3.2 Two-Stage Set Encoder](#32-two-stage-set-encoder)
   - [3.3 Recurrent State-Space Model (RSSM)](#33-recurrent-state-space-model-rssm)
   - [3.4 Factorized Decoder](#34-factorized-decoder)
   - [3.5 ELBO Training Objective](#35-elbo-training-objective)
   - [3.6 Inference Strategy](#36-inference-strategy)
4. [Repository Structure](#4-repository-structure)
5. [Installation](#5-installation)
6. [Data Pipeline](#6-data-pipeline)
   - [6.1 Raw Data](#61-raw-data)
   - [6.2 Feature Engineering](#62-feature-engineering)
   - [6.3 Temporal Splits](#63-temporal-splits)
7. [Training](#7-training)
   - [7.1 Training the RSSM](#71-training-the-rssm)
   - [7.2 Training Baselines](#72-training-baselines)
   - [7.3 NYU HPC / SLURM](#73-nyu-hpc--slurm)
   - [7.4 Hyperparameter Reference](#74-hyperparameter-reference)
8. [Evaluation](#8-evaluation)
   - [8.1 Predictive Metrics (3a)](#81-predictive-metrics-3a)
   - [8.2 KL Regime Analysis (3b)](#82-kl-regime-analysis-3b)
   - [8.3 Cross-Ticker Attention (3c)](#83-cross-ticker-attention-3c)
   - [8.4 Latent Space Clustering (3d)](#84-latent-space-clustering-3d)
   - [8.5 Counterfactual Probing (3e)](#85-counterfactual-probing-3e)
9. [Baselines](#9-baselines)
10. [Ablation Ladder](#10-ablation-ladder)
11. [Key Results and Hypotheses](#11-key-results-and-hypotheses)
12. [Configuration Reference](#12-configuration-reference)
13. [Reproducing Results](#13-reproducing-results)
14. [Design Decisions and Rationale](#14-design-decisions-and-rationale)
15. [Citation](#15-citation)

---

## 1. Project Overview

**Twit Wave** is a *world model* for the collective attention dynamics of the StockTwits financial social network (2008–2022). The central hypothesis is that retail investor attention — measured by message volume and sentiment across hundreds of ticker symbols — is not an i.i.d. noise process but a low-dimensional latent regime that evolves over time.

The model learns a **latent representation** `z_t` of the attention ecosystem at each week `t`, enabling:

- **Forecasting**: multi-step-ahead prediction of which tickers will trend and what their sentiment profile will be.
- **Regime detection**: the KL-divergence between the posterior and prior `KL(q||p)` spikes at genuine market regime transitions (COVID crash Feb 2020, GME squeeze Jan 2021) without ever seeing price data.
- **Counterfactual reasoning**: perturbing `z_t` in the direction of a specific ticker's attention reveals crowd-out effects on unrelated tickers and contagion effects among narratively related ones — an information-theoretic view of the *finite attention* hypothesis.
- **Cross-ticker coupling**: the set encoder's attention matrix `A_t` measures intrinsic (self-attention) vs. extrinsic (cross-ticker) coupling, directly quantifying whether market co-movements are driven by genuine co-dependence or aggregate regime shifts.

The architecture adapts the **Dreamer / RSSM** family of world models (Hafner et al., 2019) to the domain of *variable-membership sets* — the key novelty over standard time-series world models.

---

## 2. Key Contributions

| Contribution | Description |
|---|---|
| **Dynamic-set world model** | Extends RSSM to a setting where the observed entities (tickers) change each period. Standard baselines (VAR, LSTM) require a fixed roster with zero-imputation; TwitWave is roster-agnostic. |
| **Two-stage set encoder** | Stage 1: cross-ticker self-attention at each time step. Stage 2: temporal attention over a rolling window. Separates intra-step interaction from cross-step temporal memory. |
| **Two-embedding decoder** | Separates the *retrieval* embedding (`e_ret`) used for presence detection from the *feature* embedding (`e_dec`) used for reconstruction. Avoids the geometric conflict between dot-product retrieval and Euclidean feature decoding. |
| **KL as a regime indicator** | Demonstrates that `KL(q||p)` computed from StockTwits attention data alone tracks major market shock dates without access to price, volume, or order-flow data. |
| **Finite-attention hypothesis test** | Counterfactual probing provides the first quantitative measure of attention crowd-out vs. contagion at the ecosystem level. |

---

## 3. Architecture Deep-Dive

### 3.1 Data Representation: Dynamic Ticker Sets

At each week `t`, the model observes the **top-K tickers by message count** — the set `S_t ⊆ V` where `V` is the full vocabulary of ~1,000 tickers ever seen in the training data.

Each ticker `i ∈ S_t` is represented by a 5-dimensional feature vector:

| Feature | Symbol | Definition |
|---|---|---|
| Log attention | `log_attn` | `log(1 + msg_count_i_t)` |
| Bullish rate | `bull` | `bullish_count / labeled_count` |
| Bearish rate | `bear` | `1 - bull` (exact, not independent) |
| Unlabeled rate | `unlab` | `unlabeled_count / msg_count` |
| Attention growth | `growth` | WoW% change in msg_count, clipped to `[-5, 5]`. First appearance → 0. |

The **observation at time `t`** is thus an *unordered set* of (ticker_id, feature_vector) pairs: `X_t = {(i, f_i_t) : i ∈ S_t}`.

**Why dynamic sets matter**: The identity of the top-100 tickers changes week-to-week. An LSTM operating on a fixed `(T, K*D)` tensor would need to zero-impute absent tickers and assign a fixed "slot" to each — destroying the sparsity structure and making the model's hidden state polluted by zeros. TwitWave's set encoder is entirely permutation-equivariant and handles absence natively.

### 3.2 Two-Stage Set Encoder

**Stage 1 — Cross-ticker attention within a step** (`model/set_encoder.py`)

For a single time step with `N_t` active tickers:

```
Input: {(e_dec_i, f_i_t)} for i ∈ S_t   →   shape (N_t, D + embed_dim)
   ↓ Linear projection → (N_t, d_enc)
   ↓ Transformer encoder (n_layers, n_heads, pre-norm)
   ↓ Mean pooling over tickers
Output: a_t ∈ ℝ^{d_enc}   (step-level summary)
        A_t ∈ ℝ^{N_t × N_t}   (cross-ticker attention matrix)
```

`A_t` is the interpretability artefact: diagonal entries capture self-attention (intrinsic dynamics), off-diagonal entries capture cross-ticker coupling (extrinsic / ecosystem dynamics).

**Stage 2 — Temporal attention over a rolling window** (`model/temporal_encoder.py`)

The last `k` step summaries `(a_{t-k+1}, …, a_t)` are fed to a second transformer with sinusoidal positional encodings:

```
Input:  (k, d_enc)   — k step summaries
   ↓ Sinusoidal positional encoding
   ↓ Transformer encoder (n_layers, n_heads)
Output: e_t ∈ ℝ^{d_enc}   (context embedding for RSSM)
```

`e_t` is then used as the emission/observation signal that updates the RSSM's posterior.

### 3.3 Recurrent State-Space Model (RSSM)

The RSSM (`model/rssm.py`) maintains two state components:

- `h_t ∈ ℝ^{h_dim}` — the **deterministic state** (GRU hidden state)
- `s_t ∈ ℝ^{s_dim}` — the **stochastic latent state** (sampled from a Gaussian)
- `z_t = [h_t; s_t] ∈ ℝ^{z_dim}` where `z_dim = h_dim + s_dim`

**Prior** (transition model, no observations):
```
p(s_t | h_t):  MLP(h_t) → (μ_p, log σ_p)
               s_t ~ N(μ_p, σ_p²)
```

**Posterior** (representation model, conditioned on observations):
```
q(s_t | h_t, e_t):  MLP([h_t; e_t]) → (μ_q, log σ_q)
                    s_t ~ N(μ_q, σ_q²)
```

**Deterministic transition**:
```
h_t = GRUCell(z_{t-1}, h_{t-1})
```

The KL divergence `KL(q || p)` at each step is computed analytically (both distributions are diagonal Gaussians) and serves as both the regulariser in the ELBO and the regime-change indicator.

### 3.4 Factorized Decoder

The decoder (`model/decoder.py`) operates from `z_t` and answers two questions:

**Question 1: Which tickers are active?** (Presence head)

Uses the *retrieval* embedding `e_ret` (separate from the encoder's `e_dec`):

```
logit_i = z_t · W_proj · e_ret_i
P(ticker i active | z_t) = σ(logit_i)
Loss: weighted BCE with pos_weight=10 (to correct ~100/1000 class imbalance)
```

**Question 2: What are their features?** (Feature head)

Uses the *feature* embedding `e_dec` (shared with encoder), with a **separate MLP per feature**:

```
For each feature j ∈ {log_attn, bull, unlab, growth}:
    f̂_{i,j} = MLP_j([z_t; e_dec_i])   →   scalar

Activations enforce domain constraints:
  log_attn:  Softplus (non-negative)
  bull:      Sigmoid (∈ [0,1])
  unlab:     Sigmoid (∈ [0,1])
  growth:    5 · Tanh (∈ [-5, 5])

bear = 1 - bull   (exact constraint, not a separate MLP)
```

**Why two separate embeddings?**

The retrieval head performs inner-product lookup: `z_t · e_ret_i`. This geometry rewards embedding directions that are aligned with or orthogonal to `z_t` — a dot-product manifold.

The feature head feeds `[z_t; e_dec_i]` to an MLP: a Euclidean composition. These two geometries are incompatible in a single embedding space — a ticker that is "far" in retrieval geometry might need to be "close" in feature geometry. Two embedding tables cleanly separate these concerns.

**Why separate MLPs per feature?**

Each feature has different marginal distributions, different ranges, and different domain constraints. Sharing a single MLP with a multi-output head would force the model to balance gradients across incommensurable scales. Per-feature MLPs allow each head to learn its own effective learning rate and nonlinearity shape.

### 3.5 ELBO Training Objective

The full ELBO loss (`training/loss.py`):

```
L = L_BCE + λ · L_MSE + β · max(free_nats, KL)

where:
  L_BCE = weighted BCE over all vocab tickers (pos_weight=10)
  L_MSE = MSE on features of *active* tickers only (ticker_ids != 0)
  KL    = Σ_t KL(q(s_t|h_t,e_t) || p(s_t|h_t))   (analytical, diagonal Gaussian)
  β     = linearly annealed from 0.1 → 1.0 over first 30 epochs
  free_nats = 3.0 (prevents posterior collapse early in training)
  λ     = 1.0 (can be tuned to balance BCE and MSE scales)
```

**β-annealing**: Starting with a small KL weight lets the model first learn a useful representation before the prior becomes constraining — a standard technique from β-VAE and the RSSM literature.

**Free nats**: The KL is only penalised when it exceeds 3 nats. This prevents the model from collapsing `q` onto `p` before the decoder has learned enough to make use of the latent code.

**MSE only on active tickers**: Zero-imputed entries in the fixed-roster representation are artefacts of the data format, not real observations. Computing MSE on them would train the model to predict zeros for absent tickers — exactly the wrong inductive bias. The dynamic-set representation makes this explicit: we only decode features for tickers that are actually in `S_t`.

### 3.6 Inference Strategy

**Context phase** (posterior warm-up):
```
Feed T_ctx weeks of observed data through the RSSM in posterior mode.
At the end, we have (h_{T_ctx}, s_{T_ctx}) ≈ "where we are" in latent space.
```

**Prediction phase** (prior rollout, no observations):
```
For step t = T_ctx + 1, …, T_ctx + H:
    h_t = GRUCell(z_{t-1}, h_{t-1})
    μ_p, σ_p = Prior(h_t)
    s_t = μ_p   ← use mean, not sample (deterministic metrics)
    z_t = [h_t; s_t]
    
    # Decode presence: which tickers will be in S_t?
    P(i ∈ S_t) = σ(z_t · W_proj · e_ret_i)   for all i ∈ V
    Ŝ_t = top-K by probability
    
    # Decode features for predicted active set
    f̂_{i,j} = MLP_j([z_t; e_dec_i])   for i ∈ Ŝ_t
```

Using the prior mean (not a sample) for multi-step metrics eliminates sampling variance from the evaluation, giving a fair comparison against deterministic baselines (ARIMA, VAR, LSTM).

---

## 4. Repository Structure

```
StockTwit_WM/
│
├── 0_a_download_data.sh          # Download raw StockTwits data
├── 0_b_csv_to_parquet.py         # Convert CSVs to Parquet format
├── 1_a_data_overview.ipynb       # Exploratory data analysis
│
├── data/
│   ├── features.py               # Raw → (symbol, week, 5-feature) panel
│   ├── vocab.py                  # Stable integer indices for all ticker symbols
│   └── dataset.py                # TwitWaveDataset: dynamic + fixed-roster modes
│
├── model/
│   ├── embeddings.py             # Two-table ticker embeddings (e_ret, e_dec)
│   ├── set_encoder.py            # Stage 1: cross-ticker self-attention + mean pool
│   ├── temporal_encoder.py       # Stage 2: temporal transformer over rolling window
│   ├── rssm.py                   # GRU + Prior + Posterior + KL divergence
│   ├── decoder.py                # Presence head (e_ret) + Feature head (4 MLPs)
│   └── twit_wave.py              # TwitWave nn.Module + ModelConfig dataclass
│
├── training/
│   ├── loss.py                   # ELBO: BCE + λ·MSE + β·max(free_nats, KL)
│   ├── scheduler.py              # BetaScheduler + CosineWarmupScheduler
│   └── trainer.py                # Training loop, checkpointing, wandb, KL logging
│
├── eval/
│   ├── predict.py                # Predictor: context_phase → rollout → decode
│   ├── metrics.py                # MSE, MAE, Spearman ρ, Precision@K, AUC-ROC
│   ├── kl_analysis.py            # KL timeline plots + spike detection + CSV export
│   ├── attention_analysis.py     # A_t extraction, diag vs off-diag, heatmaps
│   ├── latent_clustering.py      # t-SNE, UMAP, k-means, GMM, silhouette
│   ├── counterfactual.py         # Gradient-based z_t perturbation + Δ decode
│   └── residual_correlation.py   # BCE residual cross-ticker correlation diagnostic
│
├── baselines/
│   ├── arima.py                  # Per-ticker ARIMA(p,d,q) — floor baseline
│   ├── var.py                    # Reduced-rank VAR with SVD truncation
│   └── lstm.py                   # Shared LSTM on zero-imputed fixed-roster data
│
├── scripts/
│   ├── 2_a_feature_engineering.py  # Step 2a: build panel + vocab + splits
│   ├── 2_b_train_rssm.py           # Step 2b: train TwitWave RSSM
│   ├── 2_c_train_baselines.py      # Step 2c: train ARIMA, VAR, LSTM
│   ├── 3_a_eval_prediction.py      # Step 3a: MSE, Spearman, P@K, AUC-ROC
│   ├── 3_b_eval_kl.py              # Step 3b: KL timeline + regime spike analysis
│   ├── 3_c_eval_attention.py       # Step 3c: cross-ticker attention analysis
│   ├── 3_d_eval_latent.py          # Step 3d: t-SNE/UMAP + clustering
│   └── 3_e_eval_counterfactual.py  # Step 3e: counterfactual probing
│
├── configs/
│   ├── model/
│   │   ├── rssm_small.yaml    # ~8M params, CPU / L4 debug
│   │   ├── rssm_base.yaml     # ~45M params, A100 40GB (primary paper model)
│   │   └── rssm_large.yaml    # ~180M params, ablation upper bound
│   ├── train/
│   │   ├── train_base.yaml    # Primary training config
│   │   └── train_debug.yaml   # 2-epoch smoke test
│   └── eval/
│       ├── eval_predict.yaml  # Forecast horizon + metric settings
│       └── eval_interp.yaml   # KL/attention/latent/counterfactual settings
│
├── slurm/
│   ├── train_a100.sh          # SLURM script for A100 40GB training
│   ├── train_l4.sh            # SLURM script for L4 24GB training
│   └── eval.sh                # SLURM script running full eval pipeline
│
├── figure_making/
│   └── data/                  # Pre-computed data for paper figures
│
├── Architecture Design/
│   ├── architecture_reference.md   # Full architecture decision record
│   ├── decoder_design.md           # Decoder design deep-dive
│   └── *.pdf                       # Reference papers (RSSM, S4WM, decoder design)
│
├── requirements.txt
└── README.md
```

---

## 5. Installation

### Prerequisites

- Python 3.10 or 3.11
- CUDA 12.1+ (for GPU training)
- Conda or virtualenv

### Setup

```bash
# Clone the repository
git clone https://github.com/Abhinav0710rajput/StockTwit_WM.git
cd StockTwit_WM

# Create and activate a conda environment
conda create -n twit_wave python=3.11 -y
conda activate twit_wave

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Install all other dependencies
pip install -r requirements.txt
```

### Verify the installation

```bash
python -c "
import torch, pandas, statsmodels, sklearn, umap
print('torch:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('GPU:', torch.cuda.get_device_name(0))
"
```

---

## 6. Data Pipeline

### 6.1 Raw Data

The dataset is StockTwits messages from 2008-01 to 2022-12. Each row is a single message with fields:
- `created_at`: timestamp
- `symbol`: ticker symbol (e.g., `AAPL`, `GME`)
- `sentiment_label`: `Bullish`, `Bearish`, or `null`

Raw data should be placed in `data/raw/` as Parquet files. If starting from CSV exports:

```bash
bash 0_a_download_data.sh      # if using the download script
python 0_b_csv_to_parquet.py   # convert CSV → Parquet for faster IO
```

### 6.2 Feature Engineering

```bash
python scripts/2_a_feature_engineering.py \
    --raw_dir  data/raw \
    --out_dir  data/processed \
    --top_k    100 \
    --min_weeks 10
```

This script:
1. Groups messages by `(symbol, week)` using DuckDB for fast aggregation
2. Computes the 5 features per (ticker, week) row
3. Keeps the **top-100 tickers by message count** per week (the dynamic roster)
4. Builds a `Vocabulary` — stable integer IDs for all tickers that appear ≥10 weeks in training
5. Splits into temporal train/val/test1/test2 panels
6. Saves `data/processed/panel_{train,val,test1,test2}.parquet` and `vocab.json`

**Output files:**
```
data/processed/
├── vocab.json              # {"AAPL": 1, "GME": 2, ...}  (padding_idx=0)
├── panel_all.parquet       # Full panel (all splits)
├── panel_train.parquet     # 2008-01 to 2018-12
├── panel_val.parquet       # 2019-01 to 2019-12
├── panel_test1.parquet     # 2020-01 to 2020-06  (COVID)
├── panel_test2.parquet     # 2020-10 to 2021-06  (GME)
└── dataset_stats.json      # Metadata: split sizes, vocab size, etc.
```

### 6.3 Temporal Splits

| Split | Period | Purpose | Key Events |
|---|---|---|---|
| **Train** | 2008-01 to 2018-12 | Model fitting | Financial crisis recovery, bull market |
| **Val** | 2019-01 to 2019-12 | Hyperparameter selection, early stopping | Stable market |
| **Test1** | 2020-01 to 2020-06 | Out-of-sample evaluation | COVID crash (Feb 20), recovery |
| **Test2** | 2020-10 to 2021-06 | Out-of-sample evaluation | GME squeeze (Jan 22), meme stocks |

The val set is deliberately placed adjacent to training (no time gap) because the model must learn temporal dynamics without seeing the future. The two test sets are chosen to stress-test regime generalisation: Test1 is a *macro shock* (correlated crash), Test2 is a *micro idiosyncratic* event (retail-driven meme-stock contagion).

---

## 7. Training

### 7.1 Training the RSSM

```bash
# Primary base model (A100 40GB)
python scripts/2_b_train_rssm.py \
    --model_cfg configs/model/rssm_base.yaml \
    --train_cfg configs/train/train_base.yaml \
    --data_dir  data/processed \
    --out_dir   outputs/rssm_base \
    --wandb

# Debug smoke-test (CPU, ~2 minutes)
python scripts/2_b_train_rssm.py \
    --model_cfg configs/model/rssm_small.yaml \
    --train_cfg configs/train/train_debug.yaml \
    --data_dir  data/processed \
    --out_dir   outputs/debug
```

**Output directory structure after training:**
```
outputs/rssm_base/
├── best_model.pt        # Best checkpoint (lowest val ELBO)
├── last_model.pt        # Final epoch checkpoint
├── model_cfg.yaml       # Exact model config used (for reproducibility)
├── train_cfg.yaml       # Exact train config used
├── norm_stats.json      # Z-score normalisation parameters (fit on train)
└── kl_log.json          # Per-epoch KL values (for training dynamics analysis)
```

**Resume from checkpoint:**
```bash
python scripts/2_b_train_rssm.py \
    --model_cfg configs/model/rssm_base.yaml \
    --train_cfg configs/train/train_base.yaml \
    --out_dir   outputs/rssm_base \
    --resume    outputs/rssm_base/last_model.pt
```

### 7.2 Training Baselines

```bash
python scripts/2_c_train_baselines.py \
    --data_dir data/processed \
    --out_dir  outputs/baselines \
    --top_k    100

# Train only specific baselines
python scripts/2_c_train_baselines.py --models arima var
python scripts/2_c_train_baselines.py --models lstm --lstm_epochs 30
```

**Baselines and their fixed-roster requirement:**

All three baselines (ARIMA, VAR, LSTM) require a *fixed* set of tickers across all time steps. The `2_c_train_baselines.py` script automatically:
1. Selects the top-K tickers by total training-set message count as the fixed roster
2. Saves this roster to `outputs/baselines/fixed_roster.json`
3. Uses `TwitWaveDataset(mode="fixed")` which zero-imputes absent weeks for each ticker

This is the fairest possible comparison: baselines get the same K tickers and the same features, just without the ability to model dynamic membership.

### 7.3 NYU HPC / SLURM

**Available GPU allocations:**
| Node | GPU | VRAM | Cores | RAM | Best for |
|---|---|---|---|---|---|
| `c12m85-a100-1` | A100 SXM4 | 40 GB | 12 | 85 GB | Primary training (rssm_base) |
| `g2-standard-12` | L4 | 24 GB | 6 | 48 GB | Ablations, baselines, eval |

```bash
# Primary training on A100
sbatch slurm/train_a100.sh

# Training on L4 (smaller batch, longer wall time)
sbatch slurm/train_l4.sh

# Full evaluation pipeline (after training)
MODEL_DIR=outputs/rssm_base sbatch slurm/eval.sh

# Monitor jobs
squeue -u $USER
tail -f logs/train_a100_<JOB_ID>.out

# Override configs via environment variables
MODEL_CFG=configs/model/rssm_large.yaml \
TRAIN_CFG=configs/train/train_base.yaml \
OUT_DIR=outputs/rssm_large \
WANDB=true \
sbatch slurm/train_a100.sh
```

**Estimated runtimes on A100 40GB:**
| Model | Epochs | Approx time |
|---|---|---|
| rssm_small (debug) | 2 | ~5 min |
| rssm_base | 100 | ~6-8 hours |
| rssm_large | 100 | ~20 hours |

### 7.4 Hyperparameter Reference

**Model hyperparameters** (in `configs/model/`):

| Parameter | Small | Base | Large | Description |
|---|---|---|---|---|
| `embed_dim` | 32 | 64 | 128 | Ticker embedding dimension (both tables) |
| `d_enc` | 128 | 256 | 512 | Set encoder output dimension |
| `h_dim` | 256 | 512 | 1024 | GRU hidden state dimension |
| `s_dim` | 128 | 256 | 512 | Stochastic latent dimension |
| `z_dim` | 384 | 768 | 1536 | Total latent dim = h_dim + s_dim |
| `n_heads` | 4 | 8 | 16 | Attention heads in set encoder |
| `n_layers` | 1 | 2 | 4 | Transformer layers (set + temporal encoder) |
| `window_k` | 4 | 8 | 16 | Temporal encoder rolling window (weeks) |
| `mlp_hidden` | 128 | 256 | 512 | Hidden units in each feature MLP |

**Training hyperparameters** (in `configs/train/`):

| Parameter | Value | Description |
|---|---|---|
| `lr` | 3e-4 | Adam learning rate |
| `weight_decay` | 1e-5 | L2 regularisation |
| `grad_clip` | 10.0 | Gradient norm clipping |
| `seq_len` | 52 | Sequence length in weeks (~1 year) |
| `batch_size` | 32 | Training batch size (A100) |
| `beta_start` | 0.1 | Initial KL weight |
| `beta_end` | 1.0 | Final KL weight |
| `beta_anneal_epochs` | 30 | Linear β annealing duration |
| `free_nats` | 3.0 | KL free bits threshold |
| `lambda_mse` | 1.0 | MSE loss weight relative to BCE |
| `bce_pos_weight` | 10.0 | BCE positive class weight (class imbalance) |
| `warmup_epochs` | 5 | Linear LR warmup |
| `patience` | 15 | Early stopping patience (val ELBO) |

---

## 8. Evaluation

All evaluation scripts write results to `outputs/eval/{prediction,kl,attention,latent,counterfactual}/`.

### 8.1 Predictive Metrics (3a)

```bash
python scripts/3_a_eval_prediction.py \
    --model_dir     outputs/rssm_base \
    --data_dir      data/processed \
    --baselines_dir outputs/baselines \
    --out_dir       outputs/eval/prediction \
    --horizons      1 4 13 \
    --splits        test1 test2
```

**Metrics computed:**

| Metric | Description | What it tests |
|---|---|---|
| `MSE_log_attn` | MSE on log(1+msg_count) | Raw attention volume forecasting |
| `MAE_log_attn` | MAE on log(1+msg_count) | Robustness to outliers |
| `MSE_feat` | MSE across all 5 features | Full feature reconstruction |
| `Spearman ρ` | Rank correlation of log_attn across tickers | Relative attention ordering |
| `Precision@100` | Fraction of true top-100 correctly predicted | Set membership quality |
| `AUC-ROC virality` | AUC for detecting tickers that enter top-20 within H weeks | Early detection of viral events |

Horizons are 1, 4, 13 weeks (approximately: next week, next month, next quarter).

**Outputs:**
- `metrics_test1.json` / `metrics_test2.json` — per-model, per-horizon scalar metrics
- Side-by-side comparison of TwitWave vs ARIMA, VAR, LSTM

### 8.2 KL Regime Analysis (3b)

```bash
python scripts/3_b_eval_kl.py \
    --model_dir  outputs/rssm_base \
    --data_dir   data/processed \
    --out_dir    outputs/eval/kl \
    --splits     val test1 test2 \
    --spike_z    2.0
```

The KL divergence `KL(q(s_t|h_t,e_t) || p(s_t|h_t))` at each week is the model's **surprise**: how much did the observed attention distribution differ from what the deterministic trajectory predicted?

**Key expected results:**
- Spike on 2020-02-20 (COVID crash begins): latent regime shifts from "bull market" to "crisis"
- Spike on 2021-01-22 (GME squeeze): latent regime shifts to "meme-stock mania"
- Low KL during stable periods (2019, late 2020)
- KL elevated throughout the COVID/GME test splits vs the stable val split

**Outputs:**
- `kl_all_splits.csv` — weekly KL values for all splits
- `kl_timeline.{png,pdf}` — time series with annotated market events
- `spike_stats_{split}.json` — spike count, max KL, Z-threshold percentiles

### 8.3 Cross-Ticker Attention (3c)

```bash
python scripts/3_c_eval_attention.py \
    --model_dir outputs/rssm_base \
    --data_dir  data/processed \
    --out_dir   outputs/eval/attention \
    --top_n     50
```

The cross-ticker attention matrix `A_t ∈ ℝ^{K×K}` is extracted from the last layer of the set encoder for each week. Its interpretation:

- **Diagonal entries** `A_t[i,i]`: how much ticker `i`'s encoding is driven by its own features (intrinsic dynamics)
- **Off-diagonal entries** `A_t[i,j]`: how much ticker `i` attends to ticker `j` (extrinsic coupling)

The **coupling ratio** = mean off-diagonal / mean diagonal. When this ratio increases, the model detects that tickers are becoming more co-dependent — expected during systemic events.

**Outputs:**
- `coupling_{split}.{csv,png,pdf}` — coupling ratio time series with event annotations
- `heatmaps/attn_{split}_{week}.png` — attention heatmaps at selected event weeks

### 8.4 Latent Space Clustering (3d)

```bash
python scripts/3_d_eval_latent.py \
    --model_dir  outputs/rssm_base \
    --data_dir   data/processed \
    --out_dir    outputs/eval/latent \
    --n_clusters 5
```

For each week `t`, extract `z_t = [h_t; s_t]` (the RSSM latent state), reduce to 2D via t-SNE and UMAP, and colour by market era.

**Market eras:**
| Era | Period | Expected cluster | Narrative |
|---|---|---|---|
| Pre-crisis | 2008-2009 | Isolated cluster | Financial crisis, unique dynamics |
| Recovery | 2010-2015 | Core cluster | Steady bull market, low attention vol |
| Vol regime | 2016-2019 | Intermediate | Brexit, trade wars, increasing retail activity |
| COVID | 2020 Q1-Q2 | Outlier cluster | Correlated crash + recovery |
| Meme stocks | 2021 Q1 | Outlier cluster | GME/AMC squeeze, retail-driven |

A high silhouette score (> 0.5) on era labels would validate that the RSSM's latent space learns meaningful market regime structure.

**Outputs:**
- `latent_states.csv` — week, era, t-SNE coords, UMAP coords, cluster labels
- `tsne_by_era.png`, `tsne_by_cluster.png`, `umap_by_era.png`
- `era_silhouette.json` — clustering quality metrics

### 8.5 Counterfactual Probing (3e)

```bash
# Run all predefined experiments
python scripts/3_e_eval_counterfactual.py \
    --model_dir outputs/rssm_base \
    --data_dir  data/processed \
    --out_dir   outputs/eval/counterfactual \
    --run_all_experiments

# Custom experiment
python scripts/3_e_eval_counterfactual.py \
    --model_dir    outputs/rssm_base \
    --target       GME \
    --week         2021-01-22 \
    --delta        3.0 \
    --eval_tickers GME AMC BB NOK TSLA AAPL MSFT SPY

# Include residual correlation diagnostic
python scripts/3_e_eval_counterfactual.py \
    --run_all_experiments \
    --run_residual_corr
```

**Methodology:**

1. Run context phase up to week `t` → get `z_t`
2. Compute `∂(log_attn_target) / ∂z_t` via autograd
3. Perturb: `z̃_t = z_t + δ · grad / ||grad||`
4. Decode features under both `z_t` and `z̃_t` for all eval tickers
5. Report `Δ = f̂(z̃_t) - f̂(z_t)` per ticker per feature

**Predefined experiments:**

| Experiment | Target | Week | δ | Question |
|---|---|---|---|---|
| `gme_squeeze` | GME | 2021-01-22 | +3.0 | Does spiking GME crowd out unrelated tickers? |
| `covid_crash` | SPY | 2020-02-20 | -3.0 | Does crashing SPY propagate to all sectors? |

**BCE residual correlation diagnostic** (when `--run_residual_corr`):

If `z_t` is a sufficient statistic for the joint ticker distribution, the Bernoulli decoder residuals `r_{i,t} = y_{i,t} - σ(logit_{i,t})` should be approximately uncorrelated across tickers. A high mean absolute off-diagonal correlation indicates that `z_t` is missing cross-ticker shared variation.

---

## 9. Baselines

Three baselines are included, forming an ablation ladder from simplest to most expressive:

### ARIMA (floor)

**File:** `baselines/arima.py`  
**Model:** `PerTickerARIMA(order=(2,0,1))`

Fits independent ARIMA models for each ticker. No cross-ticker dynamics, no latent regime. Serves as the absolute floor — any model that cannot beat ARIMA per-ticker is failing to capture even univariate temporal structure.

```python
arima = PerTickerARIMA(order=(2, 0, 1))
arima.fit(panel_train, roster)
preds = arima.forecast(steps=4)
```

### Reduced-Rank VAR

**File:** `baselines/var.py`  
**Model:** `ReducedRankVAR(maxlags=4, rank=10)`

Fits a Vector Autoregression on the `(T, K)` log_attention matrix, then truncates the coefficient matrices to rank `r` via SVD. Captures *linear* cross-ticker coupling but assumes static, stationary dynamics — exactly what the RSSM is designed to go beyond.

The SVD truncation serves two purposes: (1) regularisation for high-dimensional VAR, (2) forcing the coefficient matrices to lie on a low-rank manifold, which is theoretically motivated by the factor structure of financial markets.

```python
var = ReducedRankVAR(maxlags=4, rank=10)
var.fit(panel_train, roster)
forecast = var.forecast(last_obs, steps=4)  # (4, K)
```

### Shared LSTM

**File:** `baselines/lstm.py`  
**Model:** `SharedLSTM(n_tickers=K, feature_dim=5, hidden_dim=512, n_layers=2)`

A single LSTM processes the concatenated `(K*D)` feature vector at each step. The hidden state provides implicit cross-ticker coupling (unlike ARIMA), but there is no explicit latent regime separation — the LSTM cannot disentangle "the whole ecosystem is in a meme-stock regime" from "GME specifically has high attention."

```python
lstm = SharedLSTM(n_tickers=100, feature_dim=5, hidden_dim=512)
preds = predict_lstm(lstm, context, steps=4, device=device)  # (4, K, D)
```

---

## 10. Ablation Ladder

The paper reports results on the following ablation ladder:

| Model | Cross-ticker coupling | Latent regime | Dynamic set | Paper section |
|---|---|---|---|---|
| ARIMA | ✗ | ✗ | ✗ | Baseline |
| VAR | Linear | ✗ | ✗ | Baseline |
| LSTM | Implicit | ✗ | ✗ | Baseline |
| Transformer (ablation) | Full | ✗ | ✗ | Ablation |
| HMM-VAR (ablation) | Linear | Discrete | ✗ | Ablation |
| **TwitWave (RSSM)** | **Full** | **Continuous** | **✓** | **Main** |

To run TwitWave without the stochastic latent (RSSM → deterministic world model), set `s_dim=0` in the model config. The GRU hidden state alone then serves as the latent representation.

---

## 11. Key Results and Hypotheses

### H1: Regime transitions manifest as KL spikes

*Hypothesis:* The KL divergence `KL(q||p)` should spike on known market shock dates.

*Expected result:*
- KL Z-score > 2 on weeks containing: 2020-02-20 (COVID crash), 2021-01-22 (GME squeeze), 2020-03-23 (COVID bottom)
- KL Z-score < 1 during stable val period (2019)

### H2: TwitWave outperforms baselines on viral ticker detection

*Hypothesis:* The dynamic-set world model should have superior AUC-ROC for predicting which tickers will enter the top-20 within 4 weeks.

*Expected result:* TwitWave > LSTM > VAR > ARIMA on virality AUC-ROC, especially on Test2 (GME) where meme-stock dynamics are highly non-linear.

### H3: Attention matrix coupling ratio increases during systemic events

*Hypothesis:* During COVID (correlated macro shock), off-diagonal attention should dominate (everything moves together). During GME (idiosyncratic micro event), diagonal attention should be high for meme stocks but off-diagonal for the broader market.

### H4: Latent space clusters align with market eras

*Hypothesis:* t-SNE/UMAP of `z_t` should show distinct clusters corresponding to: pre-2016 calm, 2016-2019 vol regime, COVID crash, COVID recovery, GME meme-stock period.

*Validation:* Silhouette score on era labels > 0.5.

### H5: GME spike induces crowd-out in unrelated tickers (finite attention)

*Hypothesis:* Perturbing `z_t` in the direction of increasing GME attention should decrease predicted log_attention for unrelated tickers (SPY, AAPL, MSFT) while increasing it for related meme stocks (AMC, BB, NOK).

---

## 12. Configuration Reference

### Model config fields

```yaml
embed_dim:   64    # Both e_ret and e_dec embedding tables
d_enc:       256   # Set encoder output dimension (also temporal encoder input)
h_dim:       512   # GRU hidden state dimension
s_dim:       256   # Stochastic latent dimension (z_dim = h_dim + s_dim)
n_heads:     8     # Multi-head attention in set and temporal encoders
n_layers:    2     # Transformer encoder layers (shared config for both encoders)
window_k:    8     # Temporal encoder rolling window size (weeks)
mlp_hidden:  256   # Hidden units in each of the 4 feature MLPs
feature_dim: 5     # Fixed: [log_attn, bull, bear, unlab, growth]
top_k:       100   # Active tickers per week (used for dataset construction)
dropout:     0.1   # Applied in transformer encoder layers and MLPs
```

### Train config fields

```yaml
seq_len:             52    # Sequence length for BPTT
batch_size:          32
num_workers:         4
lr:                  3e-4
weight_decay:        1e-5
grad_clip:           10.0
max_epochs:          100
warmup_epochs:       5
patience:            15    # Early stopping on val ELBO
beta_start:          0.1
beta_end:            1.0
beta_anneal_epochs:  30
free_nats:           3.0
lambda_mse:          1.0
bce_pos_weight:      10.0
```

---

## 13. Reproducing Results

Complete reproduction from raw data to paper figures:

```bash
# Step 0: Data preparation
bash 0_a_download_data.sh
python 0_b_csv_to_parquet.py

# Step 1: Exploratory analysis (optional)
jupyter notebook 1_a_data_overview.ipynb

# Step 2a: Feature engineering
python scripts/2_a_feature_engineering.py \
    --raw_dir data/raw --out_dir data/processed --top_k 100

# Step 2b: Train RSSM (on HPC)
sbatch slurm/train_a100.sh
# or locally (debug):
python scripts/2_b_train_rssm.py \
    --model_cfg configs/model/rssm_small.yaml \
    --train_cfg configs/train/train_debug.yaml \
    --out_dir outputs/debug

# Step 2c: Train baselines
python scripts/2_c_train_baselines.py \
    --data_dir data/processed --out_dir outputs/baselines

# Step 3: Full evaluation (on HPC after training)
MODEL_DIR=outputs/rssm_base sbatch slurm/eval.sh

# Or run individual eval steps:
python scripts/3_a_eval_prediction.py --model_dir outputs/rssm_base ...
python scripts/3_b_eval_kl.py         --model_dir outputs/rssm_base ...
python scripts/3_c_eval_attention.py  --model_dir outputs/rssm_base ...
python scripts/3_d_eval_latent.py     --model_dir outputs/rssm_base ...
python scripts/3_e_eval_counterfactual.py --model_dir outputs/rssm_base --run_all_experiments
```

**Expected total compute:** ~7-8 GPU-hours on A100 40GB for the base model.

---

## 14. Design Decisions and Rationale

### Why weekly aggregation?

Weekly aggregation provides a good balance between:
- **Signal**: enough messages per week to compute stable statistics for the top-100 tickers
- **Resolution**: fine enough to track regime transitions that unfold over days-to-weeks (COVID crash took ~3 weeks, GME squeeze ~2 weeks)
- **Tractability**: 14 years × 52 weeks = ~728 time steps in training, which is sufficient for BPTT through long sequences

Monthly aggregation would miss the GME dynamics; daily aggregation would be too noisy for small tickers and make the temporal encoder window (k=8) too short-range.

### Why not use price/volume data?

The research question is about *social* attention dynamics, not market microstructure. Including price data would:
1. Make it unclear whether predictive gains come from attention vs. price signals
2. Conflate the *cause* (attention) with the *effect* (price movement)
3. Make the finite-attention hypothesis test less clean

The goal is to demonstrate that attention ecosystem dynamics alone — measured only through message counts and sentiment — contain enough information to identify regime transitions and cross-ticker dependencies.

### Why the RSSM over a pure Transformer?

A temporal Transformer over the full history would require O(T²) attention. More importantly, it has no explicit separation between:
- The *current regime state* (what kind of market are we in?)
- The *stochastic surprise* (how much did this week differ from expectation?)

The RSSM's deterministic/stochastic split is precisely designed to capture this. The deterministic component `h_t` carries the slow-moving regime signal; the stochastic component `s_t` carries the fast-moving surprise. The KL between `q(s_t)` and `p(s_t)` is then interpretable as the weekly regime-shift signal.

### Why a retrieval-style presence decoder?

The vocabulary has ~1,000 tickers but only ~100 are active each week. A dense softmax over all tickers would waste capacity on impossible outputs. The retrieval decoder `σ(z_t · W_proj · e_ret_i)` computes an independent Bernoulli probability per ticker, making it:
- **Scalable**: linear in vocab size, not quadratic
- **Interpretable**: each ticker's presence probability is an explicit function of the latent state and the ticker's learned retrieval embedding
- **Class-imbalance-aware**: upweighting positives in BCE (`pos_weight=10`) is natural for Bernoulli decoders

### Why bearish_rate = 1 - bullish_rate (not a separate MLP)?

In the StockTwits labeling scheme, a tweet is either Bullish, Bearish, or unlabeled. Among *labeled* tweets:
```
bullish_rate + bearish_rate = 1   (exact)
```
Training a separate MLP for bearish_rate would add parameters that the loss function would inevitably push to satisfy this constraint approximately. Instead, we enforce it exactly by construction: `bear = 1 - bull`. This is one fewer MLP, one fewer source of numerical error, and a built-in inductive bias.

---

## 15. Citation

If you use this code or build on this work, please cite:

```bibtex
@article{rajput2025twitwave,
  title   = {Twit Wave: A World Model for Collective Attention Dynamics in Financial Social Media},
  author  = {Rajput, Abhinav},
  journal = {Working paper},
  year    = {2025},
  note    = {NYU Stern School of Business}
}
```

**Reference papers:**

- Hafner, D., Lillicrap, T., et al. (2019). *Dream to Control: Learning Behaviors by Latent Imagination.* ICLR 2020.
- Hafner, D., et al. (2021). *Mastering Atari with Discrete World Models.* ICLR 2021.
- Smith, J., et al. (2023). *S4WM: State Space World Models for Robotic Manipulation.*

---

*For questions or collaboration inquiries: [abhinav1995rajput@gmail.com](mailto:abhinav1995rajput@gmail.com)*
