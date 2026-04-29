# Decoder Design: Twit Wave RSSM

## 1. Setup

The full latent state at time `t` is:

```
z_t = [h_t, s_t]
```

where `h_t` is the deterministic recurrent state and `s_t` is the stochastic state.

The observation at time `t` is a variable-size set of the top-100 tickers by that week's message count:

```
X_t = {x_i}   where x_i = [t_i, d_i],   i = 1..N_t,   N_t <= 100
```

- `t_i` — ticker identity
- `d_i ∈ R^D` — continuous feature vector (D=5, see Section 5)

The set of 100 tickers **changes each week** — membership is determined by that week's message count ranking, not a fixed global list.

---

## 2. Embeddings

Each ticker has **two separate learned static embeddings**:

| Embedding | Dim | Used in |
|---|---|---|
| `e_i^{ret} ∈ R^E` | E | Presence head only |
| `e_i^{dec} ∈ R^E` | E | Feature MLPs + Posterior encoder (shared) |

### Why Two Embeddings

**Presence head** (`e_i^{ret}`) uses a dot-product retrieval geometry — it is trained to align with the projected latent state for popular tickers. This imposes a geometric constraint (popular tickers pulled toward similar directions).

**Feature head + encoder** (`e_i^{dec}`) needs to be discriminative — the decoder must distinguish SPY from QQQ even when they are co-popular. Sharing `e_i^{dec}` across the feature MLPs and the encoder creates a symmetric encode-decode loop:

```
Encoder:  concat(x_{i,t}, e_i^{dec})  →  set-attn  →  z_t
Decoder:  MLP(concat(z_t, e_i^{dec})) →  d̂_i
```

Gradients from the reconstruction loss flow end-to-end through `e_i^{dec}` in both directions. Keeping `e_i^{ret}` separate prevents the retrieval BCE loss from corrupting this loop.

---

## 3. Decoder Factorization

The observation likelihood factorizes into two conditionally independent components:

```
p(X_t | z_t) = ∏_{i=1}^{K} p_i^{y_i} (1-p_i)^{1-y_i}   ·   q(d_i | t_i, y_i=1, z_t, d̂_i)
                └──────────────────────────────────────┘       └──────────────────────────────┘
                         Bernoulli (presence)                     Gaussian (features, if present)
```

where `K` is the full candidate ticker pool and `y_i ∈ {0,1}` indicates whether ticker `i` is active (in the top 100) at time `t`.

---

## 4. Presence Head

Scores each candidate ticker against the current latent state via bilinear retrieval:

```
p_i = σ( h(z_t) · e_i^{ret} )
```

- `h: R^|z| → R^E` is a learned linear projection mapping the latent state into the embedding space
- `σ` is sigmoid
- The projection `h` means `e_i^{ret}` does not need to live in `z`-space directly — it lives in `R^E`

At **inference**, to predict which tickers are in the top 100 at `t+1`:
1. Compute predicted latent `z_{t+1} = g(z_t)` via RSSM transition
2. Score all tickers: `s_i = σ(h(z_{t+1}) · e_i^{ret})` over the full vocabulary
3. Take top 100 by `s_i` as the predicted active set

---

## 5. Feature Head

For each active ticker (`y_i = 1`), features are decoded via **five separate MLPs**, each with an activation function that enforces the output constraint:

```
log_attn_i       = MLP_1(concat(z_t, e_i^{dec}))                  # R, unbounded
bullish_rate_i   = sigmoid(MLP_2(concat(z_t, e_i^{dec})))          # [0, 1]
bearish_rate_i   = 1 - bullish_rate_i                               # exact constraint, no head needed
unlabeled_rate_i = sigmoid(MLP_3(concat(z_t, e_i^{dec})))          # [0, 1]
attn_growth_i    = 5 · tanh(MLP_4(concat(z_t, e_i^{dec})))         # [-5, 5]
```

`bearish_rate` has no MLP — it is computed exactly from `bullish_rate`, preserving the constraint that `bullish + bearish = 1` by construction rather than by loss penalty.

The feature likelihood is modeled as a unit-variance Gaussian:

```
q(d_i | t_i, y_i=1, z_t, d̂_i) = N(d_i; d̂_i, I)
```

---

## 6. Reconstruction Loss

Taking the negative log-likelihood:

```
L_recon^(t) = L_BCE  +  λ · L_MSE
```

**Presence loss (BCE)** — evaluated over all K candidate tickers:

```
L_BCE = - Σ_{i=1}^{K} [ y_i log p_i + (1 - y_i) log(1 - p_i) ]
```

**Feature reconstruction loss (MSE)** — evaluated only for active tickers (`y_i = 1`):

```
L_MSE = (1/2) Σ_{i: y_i=1} || d_i - d̂_i ||^2
```

`λ > 0` balances the two heads.

---

## 7. Full ELBO

Over a sequence of length `T`:

```
L = Σ_{t=1}^{T} [ L_recon^(t)  +  β · KL( q_φ(s_t | h_t, X_t) || p_θ(s_t | h_t) ) ]
```

- `q_φ` — posterior (encoder): takes `h_t` and observed `X_t`, uses `e_i^{dec}` in the set encoder
- `p_θ` — prior (transition model): predicts `s_t` from `h_t` alone
- `β` — KL regularization strength (β-VAE style)

**Three hyperparameters to tune:** `λ` (BCE vs MSE balance), `β` (KL strength).

---

## 8. Posterior / Encoder

The encoder compresses the observed `X_t` into the posterior over `s_t`. Each ticker's row is augmented with its decode embedding before set-attention:

```
input tokens:  { concat(x_{i,t}, e_i^{dec}) }_{i=1}^{N_t}   ∈ R^{N_t × (D+E)}
    ↓  set encoder (self-attention + pooling)
aggregate embedding  →  RSSM posterior  →  q_φ(s_t | h_t, X_t)
```

`e_i^{dec}` here tells the set encoder which ticker each row belongs to — without it the encoder sees anonymous feature vectors and cannot distinguish tickers.

---

## 9. Summary of Data Flow

```
                     ┌─────────────────────────────────────┐
                     │           ENCODER (posterior)        │
  X_t + e_i^{dec} →  │  set-attn → pool → RSSM posterior   │ → s_t ~ q_φ
                     └─────────────────────────────────────┘
                                        │
                                      z_t = [h_t, s_t]
                                        │
              ┌─────────────────────────┴──────────────────────────┐
              │                                                      │
    PRESENCE HEAD                                          FEATURE HEAD
  h(z_t) · e_i^{ret}                            MLP_{1..4}(concat(z_t, e_i^{dec}))
  → p_i ∈ [0,1]                                 → d̂_i ∈ R^5  (for y_i=1 only)
  → L_BCE                                        → L_MSE
```
