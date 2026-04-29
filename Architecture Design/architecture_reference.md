# Twit Wave — Full Architecture Reference

> Synthesized from: `twit_wave_architecture_manual.pdf`, `decoder_design.md`, `Reference_Proposal.md`, and design discussions.

---

## 1. Core Thesis

Platform-wide collective attention on StockTwits operates under **latent regime structure** — qualitatively distinct states (normal fragmented attention, COVID crash, meme-stock mania) with fundamentally different cross-topic coupling structures. A world model can learn the latent forces governing these dynamics without explicit regime labels, analogous to how video world models learn latent physical laws from pixel sequences.

Standard sequence models (VAR, LSTM, Transformer) fail here because they assume approximately stationary dynamics and have no mechanism to represent or detect regime transitions. The RSSM factors the problem into encoder → dynamics → decoder, where the latent state implicitly encodes the current regime.

---

## 2. Data

### Input observation at time t

```
X_t ∈ R^{N_t × D}     N_t ≤ 100,   D = 5
```

Each row is one ticker from the **top-100 by that week's message count** (set membership changes each week). The 5 features per ticker:

| Feature | Formula | Constraint |
|---|---|---|
| `log_attention` | `log(1 + msg_count)` | ≥ 0 |
| `bullish_rate` | `bullish_count / labeled_count` | [0, 1] |
| `bearish_rate` | `1 - bullish_rate` | [0, 1], exact constraint |
| `unlabeled_rate` | `1 - (labeled_count / msg_count)` | [0, 1] |
| `attn_growth` | WoW % Δ log_attn, clipped [-5, 5] | [-5, 5] |

Raw intermediate counts: `msg_count`, `user_count`, `bullish_count`, `labeled_count`. `user_count` used for EDA only, not a feature.

### Data splits (temporal, not random)

| Split | Period | Purpose |
|---|---|---|
| Train | 2008–2018 | Growth + maturity phases |
| Validation | 2019 | Pre-COVID; tune hyperparameters |
| Test 1 | 2020 Q1–Q2 | COVID onset — in-distribution regime shift |
| Test 2 | 2021 Q1 | GME meme era — OOD novel regime |

Test 1 tests regime-shift detection. Test 2 tests generalization to a genuinely unprecedented regime.

---

## 3. Embeddings

Each ticker has **two separate learned static embeddings**:

| Embedding | Dim | Used in | Trained by |
|---|---|---|---|
| `e_i^{ret} ∈ R^E` | E | Presence head only | BCE retrieval loss |
| `e_i^{dec} ∈ R^E` | E | Feature MLPs + Encoder set-attention (shared) | Reconstruction loss end-to-end |

### Why two embeddings

The presence head uses dot-product retrieval geometry — it pulls co-popular tickers toward similar directions in embedding space. The feature decoder + encoder need `e_i^{dec}` to be **discriminative** (SPY ≠ QQQ even when always co-popular). Keeping them separate prevents the BCE retrieval loss from corrupting the encode-decode symmetric loop:

```
Encoder:  {concat(x_{i,t}, e_i^{dec})} → set-attn → z_t
Decoder:  MLP(concat(z_t, e_i^{dec})) → d̂_i
```

Gradients from `L_MSE` flow end-to-end through `e_i^{dec}` in both directions.

---

## 4. Full Architecture

### 4.1 Latent state

```
z_t = [h_t, s_t]
```

- `h_t` — **deterministic GRU state**: slow-moving regime memory. Accumulates full sequence history. `h_t = GRU(h_{t-1}, s_{t-1})`
- `s_t` — **stochastic latent**: within-regime variability. Sampled from learned distribution conditioned on `h_t` (and during training, the observation).

Neither alone is sufficient: `h_t` without `s_t` cannot represent multiple possible futures; `s_t` without `h_t` has no long-range memory. No action conditioning (passive observation, not RL).

### 4.2 Two-stage set encoder

**Stage 1 — Cross-ticker self-attention (per time step):**

```
a_t = SetEncoder({concat(x_{i,t}, e_i^{dec})}_{i ∈ active(t)})     a_t ∈ R^{d_enc}
```

Self-attention over the N_t active ticker rows → fixed-dim snapshot embedding `a_t`. Captures cross-ticker interactions within a single week: which tickers co-move, substitute, or cluster. Permutation-invariant over tickers, handles variable N_t naturally.

**Stage 2 — Temporal attention (over recent window):**

```
e_t = TemporalEncoder({a_{t-k}, ..., a_t} + PosEnc)     e_t ∈ R^{d_enc}
```

Attention over a window of k recent snapshot embeddings with positional encoding → contextualized embedding `e_t`. Captures short-range temporal patterns (momentum build-ups, sentiment reversals, multi-week viral cascades) that a single snapshot cannot reveal.

**Division of labor:** Stage 2 handles local temporal patterns (days to weeks). The GRU `h_t` handles global temporal context (months to years). Window length `k` is a hyperparameter to ablate.

### 4.3 Posterior (encoder — hindsight model)

During training, sees both the current observation and the GRU memory:

```
s_t ~ q_φ(s | h_t, e_t)     [diagonal Gaussian]
```

"Cheats" by looking at today's data. All observation information must pass through the stochastic sampling bottleneck to prevent a deterministic shortcut from input to reconstruction.

### 4.4 Prior (transition model — foresight model)

Predicts the latent regime from history alone, without seeing the current observation:

```
ŝ_t ~ p_θ(s | h_t)     [diagonal Gaussian]
```

At inference time this is the only source of `s_t`. The KL loss trains the prior to match the posterior — teaching it to predict regimes from history alone.

### 4.5 Factorized decoder

Reconstructs observations from `z_t = [h_t, s_t]`. Factorizes into two conditionally independent components:

**Presence head** — which tickers are in the top-100:
```
p_i = σ( h(z_t) · e_i^{ret} )
```
- `h: R^|z| → R^E` is a learned linear projection (maps latent into embedding space)
- `e_i^{ret}` stays in `R^E`, not constrained to `z`-space
- At inference: score all tickers, take top-100 by `p_i`

**Feature head** — 4 separate MLPs per active ticker (`y_i = 1`):
```
log_attn_i        = MLP_1(concat(z_t, e_i^{dec}))                  # unbounded
bullish_rate_i    = sigmoid(MLP_2(concat(z_t, e_i^{dec})))          # [0,1]
bearish_rate_i    = 1 - bullish_rate_i                               # exact, no MLP
unlabeled_rate_i  = sigmoid(MLP_3(concat(z_t, e_i^{dec})))          # [0,1]
attn_growth_i     = 5·tanh(MLP_4(concat(z_t, e_i^{dec})))          # [-5,5]
```
Output constraints enforced via activation functions, not loss penalties.

**Note on the Bernoulli decoder:** The independent Bernoulli decoder is a **testable hypothesis** — it claims `z_t` acts as a sufficient statistic for the joint ticker distribution, d-separating tickers conditionally. Successful BCE reconstruction constitutes evidence for the latent-regime hypothesis. Empirical test: measure residual cross-ticker correlations in BCE residuals.

---

## 5. Training

### 5.1 One gradient step (chunk of length T)

```
1. Initialize h_0 = 0, s_0 = 0
2. For t = 1..T:
   a. h_t = GRU(h_{t-1}, s_{t-1})
   b. X_t → stage1 → a_t → stage2 → e_t
   c. s_t ~ q_φ(s | h_t, e_t)          [posterior, reparameterization trick]
   d. ŝ_t ~ p_θ(s | h_t)              [prior]
   e. x̂_t = decoder(h_t, s_t)         [posterior sample used during training]
   f. Accumulate L_recon(t) and KL(t)
3. Average over T, backprop, clip gradients, update all parameters jointly
```

### 5.2 Full ELBO

```
L = (1/T) Σ_t [ L_BCE(t)  +  λ · L_MSE(t)  +  β · max(free_nats, KL[q_φ(s_t) || p_θ(ŝ_t)]) ]
```

**Loss components:**

```
L_BCE = - Σ_{i=1}^{K} [ y_i log p_i + (1-y_i) log(1-p_i) ]     over all K candidate tickers

L_MSE = (1/2) Σ_{i: y_i=1} || d_i - d̂_i ||^2                   only for active tickers
```

**Key training levers:**

| Parameter | Default | Effect |
|---|---|---|
| `β` | 0.1 → 1.0 (anneal) | Too high → trivially predictable latent, loses regime structure. Too low → prior never learns |
| `free_nats` | 3.0 | Clips KL floor. Prevents posterior collapse (without it, model routes everything through `h_t`, `s_t` goes unused) |
| `λ` | Tune on val | Balances BCE vs MSE heads |

### 5.3 Inference (forecasting)

Encoder/posterior completely dropped. Rolls forward using prior only:

```
h_{t+1} = GRU(h_t, s_t)
s_{t+1} ~ p_θ(s | h_{t+1})      [prior only — no observation]
x̂_{t+1} = decoder(h_{t+1}, s_{t+1})
```

The model is **imagining** — this is the defining property of a world model. The KL loss during training prepared the prior for this moment.

---

## 6. Evaluation

### 6.1 Prediction metrics

| Metric | Details |
|---|---|
| MSE / MAE | On `log_attention` forecasts at horizons H ∈ {1, 7, 14, 30} days |
| Spearman ρ | Between predicted and actual ticker popularity rankings |
| AUC-ROC | Virality: will ticker enter top-20 within 7 days? (binary classification) |

### 6.2 KL spike analysis — during training AND test (critical)

The KL divergence `KL_t = KL[q_φ(s_t) || p_θ(ŝ_t)]` is a **learned, continuous regime-transition signal**. It measures how surprised the prior (foresight model) is by what the posterior (hindsight model) infers from observation.

- **Small KL** → dynamics were predictable, the prior anticipated the regime
- **Large KL spike** → posterior saw something the prior didn't expect → regime transition

**During training (2008–2018):** Plot `KL_t` over time. Spikes should cluster around sector rotations, market crashes, and major macro events within the training window. This validates that the model has internalized regime structure from history.

**During Test 1 (2020 Q1–Q2, COVID):** The COVID crash (Feb–Mar 2020) should produce a prominent KL spike — the prior, trained on 2008–2018, did not anticipate this regime shift. The magnitude and timing of the spike is a key interpretability result.

**During Test 2 (2021 Q1, GME):** The meme stock episode should produce an even larger KL spike — genuinely OOD. A spike that precedes or tracks the GME/AMC rally constitutes strong evidence the model detects novel regimes without explicit labels.

**Implementation:** Log `KL_t` at every time step during both training and evaluation passes. Overlay with known event dates for the paper figure.

For V-JEPA variant: use latent prediction error `||s^pred_{t+1} - s^target_{t+1}||` as the analogous surprise measure.

### 6.3 Cross-ticker self-attention matrix — during training AND test (critical)

The Stage 1 set encoder produces attention weights `A_t ∈ R^{N_t × N_t}` (aggregated across heads) at each time step. These directly operationalize the theory's interaction matrix `W_t`.

**Interpretation:**
- Diagonal mass `A_t[i,i]` → proxy for ticker i's **intrinsic appeal** (attending to itself)
- Off-diagonal `A_t[i,j]` → proxy for **extrinsic interaction** (ticker j influences ticker i)

**During training:** Track how `A_t` evolves across the 2008–2018 training window. Expect:
- Normal fragmented periods: sparse, low off-diagonal mass (tickers mostly independent)
- Market stress periods: dense cross-ticker attention (ecosystem-wide coupling increases)

**During test (COVID / GME):** Check if `A_t` structure undergoes a structural shift when entering Test 1 and Test 2 windows. A shift in the attention graph structure (new hubs, different coupling topology) during COVID/GME would directly confirm the ecosystem theory.

**Paper figure:** Heatmaps of `A_t` at 3–4 representative weeks (normal week, COVID onset, GME peak, post-meme stabilization), showing how the interaction structure changes across regimes.

### 6.4 Latent regime clustering

Extract `z_t = [h_t, s_t]` for all time steps across the full 2008–2022 sequence.

- **t-SNE / UMAP** of `z_t` colored by known era (pre-2017 growth, 2017-2019 maturity, 2020 COVID, 2021 meme, 2022 decline)
- **Silhouette score** on `z_t` clustered by known era — quantifies regime separability in latent space
- **Unsupervised clustering** (k-means, GMM) — test if inferred clusters recover or refine known regime boundaries without using era labels

If the model learned meaningful regime structure, temporal clustering should emerge: consecutive time steps within an era should cluster together, with clear transitions at era boundaries.

### 6.5 Counterfactual coupling in latent space

Directly tests the ecosystem hypothesis (finite attention resource).

1. Take a held-out week `t` and encode it to get `z_t`
2. **Perturb** the latent representation of a single ticker (e.g., spike GME's attention component)
3. Decode the perturbed `z_t` and observe how other tickers' predicted features change

Expected results if the ecosystem hypothesis holds:
- Related meme tickers (AMC, BB) increase when GME is spiked
- Unrelated/blue-chip tickers (SPY, AAPL) decrease (crowding-out / finite attention)
- The pattern should be **regime-dependent**: same spike during normal periods causes smaller propagation than during meme-era periods

### 6.6 Theory ↔ architecture empirical tests

| Theoretical claim | Architecture component | Test |
|---|---|---|
| Cross-topic coupling | Stage 1 cross-ticker self-attention | Ablate: per-ticker ARIMA vs. full model |
| Latent regimes | Stochastic `s_t` in RSSM | t-SNE/UMAP of `z_t` colored by era; silhouette score |
| Regime persistence | Deterministic GRU `h_t` | Ablate: no-GRU SSM-only vs. RSSM |
| Regime transitions | KL divergence spikes | Plot `KL_t`; align with COVID/GME/training events |
| Attention as finite resource | Attention-share features | Counterfactual: perturb one ticker, observe substitution |
| `z_t` as sufficient statistic | Independent Bernoulli decoder | Measure residual cross-ticker correlation in BCE residuals |

---

## 7. Ablation ladder

Models form a clean ablation structure — each step isolates one architectural contribution:

| Model | Cross-topic? | Latent state? | Regime-aware? | Tests |
|---|---|---|---|---|
| Per-ticker ARIMA | No | No | No | Baseline floor |
| VAR | Yes | No | No | Value of cross-topic coupling |
| LSTM (shared) | Yes | Implicit | Implicit | Value of recurrence |
| Transformer | Yes | No | No | Latent vs. direct prediction |
| HMM-VAR | Yes | No | Yes (discrete) | Continuous vs. discrete regime |
| **RSSM (ours)** | Yes | Yes | Yes (continuous) | Full model |

---

## 8. Hyperparameters

| Parameter | Range / Default | Notes |
|---|---|---|
| `z_t` (s_t) dimension | {64, 128, 256} | Stochastic state |
| `h_t` dimension | 256 or 512 | GRU hidden size |
| Encoder window `k` | {5, 10, 20} | Temporal attention context length |
| Sequence chunk `T` | 30–90 weeks | Training window |
| `β` (KL weight) | 0.1 → 1.0 (anneal) | Latent consistency pressure |
| Free nats | 3.0 | Prevents posterior collapse |
| `λ` (BCE/MSE balance) | Tune on val | Decoder head balance |
| Top-K ticker pool | 100 | Per-week active tickers |
| Optimizer | Adam, lr=3e-4 | Cosine schedule |
| Batch size | 32 sequences | |

---

## 9. Scaling direction (post-course)

Replace GRU with **S5 layer** (diagonal variant of S4), following S4WM (Deng et al., NeurIPS 2023). Drop-in replacement — only `h_t` computation changes. Everything else (stochastic path, two-stage encoder, factorized decoder, KL objectives) stays identical.

S5 advantages: parallelized training via convolution mode; HiPPO-initialized state matrices for principled long-range retention; training on much longer windows (hundreds of weeks), capturing cross-regime dependencies spanning years.

---

## 10. Full data flow (summary)

```
X_t = {(x_{i,t}, e_i^{dec})}_{i=1}^{N_t}
         │
         ▼
   [Stage 1: Cross-ticker self-attention]
   A_t ∈ R^{N_t × N_t}  ←─── SAVE FOR EVAL (attention matrix evolution)
   a_t = pool(A_t · V)
         │
         ▼
   [Stage 2: Temporal attention over {a_{t-k},...,a_t}]
   e_t ∈ R^{d_enc}
         │
    ┌────┴────┐
    │         │
  [Posterior]  [Prior]
  q_φ(s|h_t,e_t)  p_θ(s|h_t)
    │         │
   s_t       ŝ_t
    │         │
    └──KL─────┘  ←─── SAVE KL_t FOR EVAL (regime transition signal)
         │
       z_t = [h_t, s_t]
         │
    ┌────┴──────────────────────┐
    │                           │
[PRESENCE HEAD]           [FEATURE HEAD]
h(z_t) · e_i^{ret}        MLP_{1-4}(concat(z_t, e_i^{dec}))
→ p_i ∈ [0,1]             → d̂_i ∈ R^5
→ L_BCE                   → L_MSE

         h_t updated via GRU:
         h_{t+1} = GRU(h_t, s_t)
```
