# Weekly Context Validation Findings

This note summarizes the RSSM/world-model validation results from `notebooks/4_b_RSSM_week_model_validation.ipynb`. The goal is interpretability validation, not baseline predictive comparison.

## Run Context

- Model artifact: `external_models/twitwave-rssm-large/checkpoints/best.pt`
- Weekly data: `data/processed_week/`
- Observed ticker network: `data/processed_week/ticker_network_fact/`
- Notebook output folder: `outputs/eval/model_validation/`
- The checkpoint loaded successfully after using a compatibility loader that infers `gru_input_extra_dim=8`.
- Checkpoint load status: no missing keys and no unexpected keys.

## Observed Ticker-Ticker Fact Network

**Result**

The observed network built from post-level `feature_wo_messages` data contains:

| Quantity | Value |
|---|---:|
| Weeks covered | 762 |
| Date range | 2008-05-26 to 2022-12-26 |
| Node-week rows | 151,337 |
| Edge-week rows | 11,159,469 |
| Message co-mentions | 46,521,911 |
| User-week co-mentions | 140,513,569 |

Edges have two meanings:

- `message_co_mentions`: two tickers appeared in the same message.
- `user_week_co_mentions`: the same user mentioned both tickers during the same week.

**Interpretation for research question**

This dataset is now available as an observed benchmark for validating learned cross-ticker attention. In the paper, this can support the claim that the model attention matrix should be compared against actual user-level co-attention patterns, rather than interpreted only internally.

## Finding 1: KL Regime-Surprise Check

**Research question**

Does the RSSM posterior-prior KL divergence spike around known market/social-attention regime changes such as COVID and GME?

**Result**

| Event week | Context start | KL last | Context mean KL | Context z-score |
|---|---:|---:|---:|---:|
| 2020-02-20 | 2019-02-25 | 53.260 | 57.881 | -0.352 |
| 2020-03-23 | 2019-04-01 | 53.303 | 57.174 | -0.271 |
| 2021-01-22 | 2019-10-28 | 53.695 | 58.194 | -0.356 |
| 2021-01-28 | 2019-11-04 | 54.455 | 57.642 | -0.256 |

**Interpretation**

The expected validation pattern was positive KL surprise at COVID/GME event weeks. The current run does not show that: all selected event-week z-scores are mildly negative relative to their context windows. This weakens the claim that the current checkpoint’s KL signal cleanly identifies known regime transitions.

For paper integration, this should be framed as a mixed or negative validation result unless a revised KL extraction method, broader event window, or retrained checkpoint produces stronger alignment.

## Finding 2: Cross-Ticker Attention Allocation

**Research question**

Does the learned set-encoder attention matrix separate intrinsic ticker dynamics from extrinsic cross-ticker coupling, especially during systemic events?

**Result**

| Label | Week | Diagonal attention | Off-diagonal attention | Coupling ratio |
|---|---:|---:|---:|---:|
| Stable validation | 2019-06-21 | 0.010000 | 0.010000 | 1.000002 |
| COVID | 2020-02-20 | 0.010000 | 0.010000 | 0.999999 |
| GME | 2021-01-22 | 0.010000 | 0.010000 | 1.000000 |

**Interpretation**

The attention matrix is essentially uniform across stable and event weeks. This does not support the hypothesis that the learned attention weights themselves provide a differentiated ticker-ticker interaction matrix.

The observed ticker-ticker fact network is ready for a stronger future check: compare model attention edge rankings with observed co-mention/user-week edge rankings. However, the current attention output does not yet show meaningful variation, so attention-weight interpretability should be presented cautiously.

## Finding 3: Latent Regime Geometry

**Research question**

Does the latent state `z_t` organize weeks into recognizable market/social-attention regimes?

**Result**

The full run extracted:

- Latent matrix shape: `(671, 1280)`
- Era silhouette score: `0.155`

Era counts:

| Era | Weeks |
|---|---:|
| Early | 241 |
| Maturity | 208 |
| Pre-COVID | 157 |
| COVID | 39 |
| Meme | 26 |

**Interpretation**

The latent space shows weak-to-modest era structure. A silhouette score of about `0.155` is above zero, so eras are not completely random in latent space, but the separation is not strong enough to claim clean regime clustering.

For the paper, this can be described as partial evidence that the RSSM latent state captures broad temporal/regime differences, but not as strong evidence of sharply separated regimes.

## Finding 4: GME Counterfactual Probe

**Research question**

If the latent state is perturbed in the direction of increasing GME attention, does the model show crowd-in for related meme stocks and crowd-out for unrelated tickers?

**Result**

All tracked tickers increased in predicted `log_attention`.

| Ticker | Delta log_attention |
|---|---:|
| AAPL | +0.349 |
| BB | +0.341 |
| AMZN | +0.339 |
| TSLA | +0.334 |
| AMC | +0.334 |
| NOK | +0.334 |
| NFLX | +0.334 |
| MSFT | +0.333 |
| SPY | +0.332 |
| GME | +0.330 |

**Interpretation**

The model responds to a positive GME-direction perturbation as a broad market-wide attention increase. This is not the clean finite-attention pattern expected by the theory. Specifically, the model does not show unrelated large-cap names being crowded out while related meme names are crowded in.

For the paper, this should be interpreted as evidence that the current latent direction is mostly a general attention-intensity axis, not a ticker-specific substitution/crowding mechanism.

## Finding 5: COVID / SPY Counterfactual Probe

**Research question**

If SPY is perturbed in a negative shock direction during COVID, does the model propagate broad market stress across related market tickers?

**Result**

All tracked tickers decreased in predicted `log_attention`.

| Ticker | Delta log_attention |
|---|---:|
| SPY | -0.323 |
| TLT | -0.325 |
| VIX | -0.325 |
| TSLA | -0.325 |
| XLE | -0.325 |
| QQQ | -0.325 |
| XLF | -0.325 |
| AMZN | -0.333 |
| GLD | -0.334 |
| AAPL | -0.347 |

**Interpretation**

This is more consistent with a broad systemic shock interpretation: a negative SPY-direction perturbation depresses attention predictions across the tracked market set. However, the response is also quite uniform, suggesting the model is again using a broad latent attention/regime direction rather than fine-grained sector-specific propagation.

For paper integration, this can be described as partial support for broad shock propagation, but not strong evidence of differentiated contagion channels.

## Finding 6: Residual Dependence Diagnostic

**Research question**

Does the latent state `z_t` act as a sufficient statistic for the joint ticker-presence distribution?

**Result**

For Test2:

- Weeks evaluated: 39
- Active tickers in residual correlation matrix: 50
- Mean absolute off-diagonal residual correlation: `0.5094`

**Interpretation**

Residual correlation remains high after conditioning on the model state. If `z_t` fully absorbed shared ticker variation, residual correlations should be closer to zero. This result suggests the current latent state does not fully explain cross-ticker dependence in ticker presence.

For the paper, this should be treated as an important diagnostic limitation: the current model captures some broad latent structure, but leaves substantial shared variation unexplained.

## Overall Interpretation

The validation results are mixed and mostly cautionary.

The strongest positive outcome is infrastructure: the model now loads, the weekly validation notebook runs end to end, and the observed ticker-ticker fact network is available for comparison against learned attention.

The substantive model findings are weaker:

- KL does not spike at the selected COVID/GME weeks.
- Attention weights are nearly uniform and not currently interpretable as differentiated ticker-ticker coupling.
- Latent regime clustering is present but modest.
- Counterfactuals mostly move all tracked tickers in the same direction.
- Residual dependence remains high.

The current checkpoint therefore provides limited support for the stronger interpretability claims. A careful paper draft should present these as diagnostic findings and motivate next steps: improve attention extraction/training, compare learned attention to the observed co-attention network, refine counterfactual directions, and retrain or recalibrate the model so KL and latent dynamics better align with known regime events.

