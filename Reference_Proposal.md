# Project Proposal: World Modeling Topic Ecosystem*

**Jax Li** — NYU &nbsp;&nbsp; **Narayani Sai Pemmaraju** — NYU &nbsp;&nbsp; **Abhinav Rajput** — NYU

April 22, 2026

---

**Research Question: World Modeling Collective Attention Dynamics in Topic Ecosystems.**

Public online social platforms has become an increasingly important component of people's life (Suciu 2021, Pew 2018). Among the questions researchers seek to answer about online communities, understanding collective attention dynamics—how topics gain and lose public attention—has been a central problem due to its scientific and practical importance (see Zhou et al. 2021 for a review). A meaningful and underexplored challenge is that topics on online platforms are not independent. Instead, topics may emerge, evolve, compete with, or reinforce one another, resembling an ecosystem in which collective attention functions as the key resource. This raises an important research question:

> Can collective attention on online platforms be understood and modeled as a topic ecosystem, and if so, how can we empirically test the existence of such dynamics and model them effectively?

In this project, we investigate this question in two steps. First, we develop and test a topic-centric ecosystem perspective of collective attention in online communities. Second, we explore whether world models—a machine learning paradigm designed to capture the dynamics of complex systems—provide an effective framework for modeling such attention ecosystems. We study these questions using a comprehensive dataset of user activity from the financial social media platform StockTwits (Li et al. 2025).

---

## Theory Gap: A platform-wide view of topic prominence[^1] dynamics

There has been extensive research on topic prominence dynamics. For example, prior work has examined which content attributes lead to virality (Rathje and Van Bavel 2025, Berger and Milkman 2010, 2012), how information spreads through online social networks (Cheng et al. 2024, Weng et al. 2013), what outcomes viral-topic events produce (Huberman et al. 2009, Wu and Huberman 2007), and how topic popularity can be interpreted in domains such as financial markets (Cookson et al. 2024).

Despite these advances, a meaningful gap remains in developing a platform-wide understanding of collective attention dynamics. First, many studies focus on a limited subset of activity within large platforms—for example, restricting analysis to a predefined set of users, topics, or short time windows—making it difficult to observe the broader attention landscape across the platform. Second, much of the existing literature adopts a user-centric perspective, where users are treated as the primary unit of analysis. While this approach provides valuable insights into user behavior, it can obscure ecosystem-level attention dynamics because activity on public platforms is often highly skewed toward a small number of very active users. These limitations suggest the need for a complementary perspective that examines collective attention at the platform level, focusing on the interactions and dynamics among topics (or information) themselves.

---

## Theory Development

We seek to develop a theory that models topic prominence through an intrinsic–extrinsic decomposition. Integrating the topic-centric view of online communities (Weng and Lento 2014, Weng and Menczer 2014, 2015) with the emerging literature on attention economics (Meyer et al. 2024), we conceptualize online platforms as dynamic attention systems in which users allocate limited cognitive resources across competing topics. Within this system, topics emerge, rise, interact, cluster, and decline as attention is continuously redistributed across the platform.

Let $p_{i,t}$ denote the prominence of topic $i$ at time $t$, measured as the share of platform activity devoted to that topic.[^2] Topic prominence is directly observable in platform activity, but the mechanisms generating it are not. We propose that the prominence of a topic at any given time arises from two conceptually distinct sources: its intrinsic appeal and extrinsic interaction effects from other topics.

**Intrinsic appeal** reflects the inherent ability of a topic to attract attention independent of the surrounding topical environment. Intrinsic appeal may arise from properties such as novelty, emotional salience, real-world relevance, or cultural significance. For example, breaking news about a well-known public figure may naturally attract more attention than discussion of a niche subject, even in isolation from other topics.

**Extrinsic interaction effects** arise because topics coexist within a shared attention ecosystem. Since user attention is limited, the prominence of one topic can influence the prominence of others. Topics may reinforce one another when they attract overlapping audiences or are frequently discussed together, leading increases in one topic's prominence to elevate related topics. Conversely, topics may compete for attention when they draw from the same audience pool, causing the rise of one topic to crowd out attention to others.

To formalize these mechanisms, we posit that observed topic prominence arises from two latent components: the intrinsic appeal of individual topics and the extrinsic influence exerted by other topics through interaction effects. Let $a_{i,t}$ denote the intrinsic appeal of topic $i$ at time $t$, capturing the inherent ability of the topic to attract attention independent of the surrounding topical environment. Let $w_{ij,t}$ denote the interaction effect of topic $j$ on topic $i$, reflecting how the prominence of one topic may reinforce or compete with another within the shared attention ecosystem.

Under this formulation, the prominence of topic $i$ at time $t$ is generated by combining its intrinsic appeal with the aggregate influence of other topics:

$$p_{i,t} = f\!\left(a_{i,t},\sum_{j \neq i} w_{ij,t}\, p_{j,t}\right),$$

where $f(\cdot)$ represents the mechanism through which intrinsic appeal and cross-topic interaction effects jointly produce observed prominence.

For compact representation, we denote the latent state of the topic ecosystem at time $t$ as

$$z_t = (a_t, W_t),$$

where $a_t = (a_{1,t}, \ldots, a_{K,t})$ is the vector of intrinsic topic appeals and $W_t = (w_{ij,t})$ is the matrix of cross-topic interaction effects. Observed prominence across topics can therefore be expressed as

$$p_t = f(a_t, W_t),$$

where $p_t = (p_{1,t}, p_{2,t}, \ldots, p_{K,t})$ denotes the vector of topic prominence across the platform.

The latent ecosystem state itself evolves over time as new information arrives and user attention shifts across topics. We therefore model the temporal evolution of the system through a transition function

$$z_{t+1} = g(z_t),$$

where $g(\cdot)$ governs how intrinsic topic appeal and cross-topic interaction structure evolve over time.

This formulation conceptualizes online discourse as a coupled dynamical system in which the prominence of topics emerges from both topic-specific intrinsic appeal and relational interactions among topics. Importantly, while topic prominence is directly observable from platform activity, the intrinsic appeal of topics and their interaction structure are latent components of the system. In the empirical section, we leverage world-model-based approaches to learn representations of this latent ecosystem and evaluate whether modeling topic prominence as an interacting dynamical system improves both the prediction and interpretation of platform-wide topic dynamics.

---

## Method Gap: using world model for collective attention dynamics

World models are representation learning models that learn a compressed latent representation of an environment and use it to simulate future states. They have become a dominant paradigm in reinforcement learning and sequential decision-making. Originating with early work on learning dynamics from pixels (Ha and Schmidhuber 2018), the framework has scaled rapidly through the Dreamer line of architectures (Hafner et al. 2024), which use Recurrent State-Space Models (RSSMs) to learn latent dynamics for continuous control across diverse domains. More recently, world models have expanded beyond game environments into robotics (V-JEPA 2; Assran et al. 2025), autonomous driving (GAIA-1; Hu et al. 2023), and video prediction at internet scale. The core architectural feature of factoring a complex system into an encoder that compresses observations into a latent state prediction, a predictor that then makes predictions in the latent state space using context from previous states, and a decoder that reconstructs observations has proven remarkably general.

Meanwhile, the modeling of topic popularity dynamics in online communities has relied on a different methodological toolkit. Classical approaches adapt epidemic diffusion models (SIR-type cascades) to predict whether individual pieces of content will go viral (Zhou et al. 2021; Cheng et al. 2024). More recent work uses standard sequence models such as VAR, LSTM, and Transformer architectures to forecast popularity trajectories from observed features (Zhou et al. 2021). These methods learn a direct mapping from past observations to future observations in the raw feature space.

An illustration behind modeling popularity dynamics using world models comes from recent work on intuitive physics alignment of video models. (Garrido et al. 2025) show that V-JEPA, a model trained to predict future video in a learned representation space rather than in raw pixels, develops an emergent understanding of physical principles such as object permanence and shape constancy, significantly outperforming both pixel-prediction and text-based models on violation-of-expectation benchmarks. The key architectural driver is that predicting in a compressed latent space forces the model to internalize the low-dimensional generative rules (gravity, solidity, continuity) that actually govern the visual dynamics, rather than memorizing surface-level pixel correlations. We view the collective attention ecosystem through an analogous lens: the observed time series of ticker-level popularity, sentiment, and engagement are high-dimensional surface observations generated by a much lower-dimensional set of latent forces, regime-level shifts in market sentiment, speculative momentum, and cross-topic coupling structure. Just as a world model that learns latent physics generalizes to novel object configurations, a world model that learns latent attention dynamics should generalize to novel regime configurations that were absent from training.

Despite the natural fit, world models have not been applied to collective attention dynamics. We argue this gap is both surprising and consequential. The StockTwits attention ecosystem exhibits three properties that make it a textbook world-model problem but a poor fit for standard sequence models.

**First**, the system is non-stationary with latent regime structure: the platform traverses qualitatively distinct phases (growth, maturity, meme-driven speculation) with fundamentally different cross-topic coupling structures. Standard sequence models applied in this field assume approximately stationary dynamics; they have no mechanism to represent or detect these regime transitions.

**Secondly**, the system is high-dimensional with shared latent structure: thousands of tickers evolve simultaneously, but their co-movement is governed by a much lower-dimensional set of latent factors (market sentiment, speculative momentum, sector rotation). World models are designed precisely to learn such compressed latent representations, discarding observation-level noise while preserving the structural dynamics that matter for forecasting.

**Third**, the system exhibits state-dependent dynamics: identical local shocks (e.g., a spike in a single ticker's popularity) propagate differently depending on the current regime, a spike during normal fragmented attention dissipates, while the same spike during a speculative wave cascades across related tickers. World models handle this naturally because their dynamics model conditions on the current latent state, making predictions inherently state-dependent without requiring explicit regime labels.

---

## Data

We handle the research question with the StockTwits dataset proposed by Li et al. (2025). StockTwits is a popular public forum where users post short blogs about the financial market, an analogy of "finance related Twitter" platform. Three unique features of the data make it especially suitable for our question.

1. **Scale and comprehensiveness:** The dataset consist of the full record of StockTwits platform (7M+ users and 550M+ posts over 2008 to 2022). The platform-wide setting avoid most of selection bias and the time range allow us to reveal the regime evolution, making it an ideal testbed for our ecosystem theory and the world model setting requirement.

2. **Topic stability and heterogeneity:** The finance-related nature of the platform provides a distinguishable, stable, and comparable definition of topicallity – each ticker corresponds to a topic. On the other hand, the diverse nature of the companies and indices prevent the ecosystem (world) from being concentrated to a specialized community.

3. **Future replication and enhancement:** The data is publicly available (Li et al. 2025), and has clear directions to complement (i.e. with market data, company context, etc.). So future works can use our results as comparable benchmarks with the same data.

---

## Empirical Design

We propose two world model architectures. Our primary model is an RSSM with an S4/S5 structured state space backbone (Deng et al. 2024) replacing the standard GRU for long-range memory. A set encoder with self-attention compresses the variable-size ticker observation $x_t \in \mathbb{R}^{N_t \times d}$ into a fixed-dimensional embedding, which feeds into the RSSM's deterministic path (slow regime memory) and stochastic path (within-regime variability). A query-based decoder reconstructs per-ticker features conditioned on the latent state and learned ticker embeddings, and the full system is trained end-to-end via ELBO (reconstruction loss plus KL divergence between the posterior and prior). Our second model adapts V-JEPA 2 (Assran et al. 2025): the same set encoder maps observations to latent embeddings, but training proceeds entirely in latent space—a predictor maps the current embedding to the next time step's, with targets from an EMA copy of the encoder and an L1 loss; no decoder is used during pretraining, and a lightweight decoder is trained downstream for prediction. We compare both world models against baselines forming an ablation ladder, per-ticker AR(p), reduced-rank VAR, per-ticker LSTM, PatchTST (Nie et al. 2023), and iTransformer (Liu et al. 2024), isolating the contributions of cross-ticker coupling, nonlinearity, and latent regime-awareness. Evaluation uses temporal splits: training on 2008–2018, validation on 2019, Test 1 on 2020 Q1–Q2 (COVID onset), and Test 2 on 2021 Q1 (meme stock era), with MSE/MAE on log-attention forecasts, Spearman $\rho$ on popularity rankings, and attention growth as a secondary target.

The raw data is at the post level, where each record corresponds to a single user message. We use five columns from the raw `symbols/` CSV files: `message_id` (to count posts), `user_id` (to count distinct users), `sentiment` (user self-labeled as Bullish, Bearish, or unlabeled), `created_at` (to define weekly time windows), and `symbol` (to group by ticker). The full dataset spans 550M+ posts from 7M+ users over 2008–2022.

We aggregate the raw post-level data into a ticker × week panel by grouping rows by symbol and week, and computing four intermediate counts: `msg_count` (total rows), `user_count` (distinct users), `bullish_count` (rows where sentiment = 1), and `labeled_count` (rows where sentiment is not null). From these four counts we derive five features:

1. **Log attention:** $\log(1 + \text{msg\_count})$. This is the primary signal and the main prediction target. The log transform is applied because raw post counts are extremely skewed across tickers and time; the transform compresses scale so that large spikes do not dominate the model.

2. **Bullish rate:** $\text{bullish\_count} / \text{labeled\_count}$. Sentiment is a leading indicator of attention dynamics: when a stock is moving, more users post with conviction, which attracts further attention. Weeks with zero labeled posts are filled with 0.

3. **Bearish rate:** $1 - \text{bullish\_rate}$. Kept separate from bullish rate rather than collapsed into a single net sentiment score, because a 50/50 bullish-bearish split carries different information than a ticker with no labeled posts at all.

4. **Unlabeled rate:** $1 - (\text{labeled\_count} / \text{msg\_count})$. When engagement is intense, users are more likely to stake a directional position and label their post. A rising unlabeled rate may therefore signal declining conviction, while a falling unlabeled rate may signal a strengthening narrative around a ticker.

5. **Attention growth:** week-over-week percentage change in log attention, computed per ticker and clipped to $[-5, 5]$ to handle extreme spikes. A ticker accelerating from 100 to 1000 posts is fundamentally different from one declining from 2000 to 1000, even though both have the same current log attention. This feature captures momentum, which is a well-documented phenomenon in both financial markets and social media attention dynamics.

No text content, price data, or external sources are used. All five features trace back entirely to counting rows and applying simple arithmetic over the `symbols/` directory.

**Prediction target.** Our primary prediction target is log attention $\log(1 + n_{i,t+1})$ for each ticker simultaneously, one week ahead. We additionally evaluate on attention growth as a secondary target, as it captures dynamics independently of baseline popularity levels.

**Models.** We compare seven models in increasing order of complexity. As a zero-parameter floor, we use naive persistence, which predicts next week's log attention as equal to this week's value. We then fit a per-ticker AR(p) model (Box and Jenkins 1970) independently for each ticker using only its own history, capturing within-ticker persistence while assuming tickers are independent. A reduced-rank VAR (Sims 1980) extends this by modeling all tickers jointly with a low-rank coefficient matrix, capturing linear cross-ticker coupling but assuming the coupling structure is static over time. A per-ticker LSTM (Hochreiter and Schmidhuber 1997) adds nonlinearity within each ticker's own history without any cross-ticker coupling, isolating the contribution of nonlinearity from that of ecosystem structure. On the transformer side, PatchTST (Nie et al. 2023) treats each ticker independently by segmenting its series into patch tokens — the nonlinear attention-based analogue of per-ticker AR — while iTransformer (Liu et al. 2024) inverts this design by embedding each ticker's full history as a variate token and using self-attention to capture cross-ticker correlations, making it the nonlinear analogue of VAR. Finally, our RSSM (Hafner et al. 2024) learns a shared latent state $z_t \in \mathbb{R}^d$ jointly across all tickers, and is the only model that simultaneously captures cross-ticker coupling and a time-varying latent regime state.

---

## Evaluation

The evaluation period spans three qualitatively distinct market regimes: the COVID-19 market crash (February–May 2020), the meme stock episode (January–March 2021), and a post-meme stabilization phase (2022). This structure allows us to test whether model performance advantages are concentrated during regime transitions—the central empirical prediction of our framework. We hope to check if these events are captured by our model.

The world model framework provides a set of principled tools for reading the latent space that standard sequence models structurally cannot offer, this is how we distinguish "the model learned real dynamics" from "the model just has more parameters." We evaluate whether the model captures meaningful ecosystem dynamics through four methods.

1. **Regime surprise via KL divergence.** In the RSSM, the KL divergence between the prior (predictor) and posterior (encoder) at each time step, $\text{KL}_t$, quantifies how much the observed data deviates from the model's expected dynamics. We track $\text{KL}_t$ over 2008–2021 and test whether spikes align with known regime transitions (e.g., COVID crash, GME episode, sector rotations). Alignment would indicate that the model has internalized regime structure, as this signal arises endogenously from the learned dynamics rather than from external labels. For the V-JEPA model, we use the latent prediction error $\|z^{\text{pred}}_{t+1} - z^{\text{target}}_{t+1}\|$ as an analogous measure of distributional surprise.

2. **Latent structure and regime clustering.** We extract latent states across all time steps and visualize them using t-SNE/UMAP. If the model learns meaningful regime structure, we expect temporal clustering of latent states corresponding to distinct platform eras. We quantify this using silhouette scores and apply unsupervised clustering (k-means, GMM) to test whether inferred clusters recover or refine known regime boundaries.

3. **Counterfactual coupling in latent space.** To directly test the ecosystem hypothesis, we perform counterfactual interventions in the latent space. By perturbing the latent representation of a single ticker and decoding the resulting state, we examine whether changes propagate to other tickers. Systematic co-movement—e.g., increasing a meme stock's latent prominence leading to gains in related tickers and declines in unrelated ones—would provide evidence that the model has learned shared attention constraints and cross-topic interactions.

4. **Intrinsic vs. extrinsic decomposition via attention structure.** To connect the learned representation to our theoretical framework, we analyze the self-attention patterns within the set encoder. For each time step $t$, attention weights define pairwise interactions between tickers $x_{t,i}$ and $x_{t,j}$. We interpret attention mass on $i \to i$ as a proxy for intrinsic contribution, and cross-ticker attention $i \to j$ as a proxy for extrinsic interaction effects. Aggregating attention weights across heads provides an empirical estimate of the interaction matrix $W_t$. By examining how these interaction patterns evolve over time, we test whether the model separates topic-specific appeal from cross-topic influence and whether shifts in this structure align with changes in observed prominence.

---

## Expected Contributions

Our contributions, if the theory and method hold, lay in three different domains. Firstly for research in online communities and collective behavior, we provide a theory to understand the attention dynamics in large scale platform with long term regime-changing and highly diverse content. Secondly, for the machine learning community, we introduce a promising area that is under studied and is suitable for the thriving paradigm of world model. Lastly, we provide a framework that is highly flexible for future replication and enhancement with a public available dataset.

---

## Limitations and Future Research Plan

We acknowledge several limitations and potential future directions in our paper. On theory, our topic ecosystem theory is validated by pattern identification based on correlations rather than causal inference, future research could be on potential causal relationships among attention dynamics. On data context, although the StockTwits data obtains significant scale and diversity from the diverse nature of the financial tickers and indices, it is still constrained with the investment-related domain, more encyclopedic platforms like X and Reddit could have more complex user and topic patterns. On data complement, the ticker-based topic definition can be complemented with other data such as market data or company context, which leads to potential future works on the application of our theory and method. On feature engineering and selection, we are in lack of information extraction from the blog texts since it is not our primary focus, future adoption of NLP and text-as-data method could lead to valuable insights.

---

## A message for course project setting

Dear Professors,

As reflected in our proposal, our project lies at the intersection of computational social science and machine learning. Our team is approaching this project with the long-term goal of developing it into a research paper that could eventually be submitted to journals such as Management Science. As such, we expect to continue working on it beyond the course timeline.

With that context in mind, we would like to ask two brief questions regarding the course project requirements and whether there may be some flexibility in how our project fits within the rubric.

1. **Research track.** Our project is research-focused and includes a significant machine learning component, but it may not align perfectly with the conventions of purely machine learning venues (e.g., NeurIPS, ICLR, ICML, ACL, CVPR). Instead, it is closer in spirit to computational social science work that combines theory, data analysis, and machine learning. We wanted to check whether this type of project would be acceptable for the course, or whether the expectation is that projects should follow a more traditional ML format.

2. **Final report format and length.** In our project, the theoretical and conceptual framing from social science is closely integrated with the machine learning methodology, as is common in computational social science research. As a result, if we structure the final report as a first draft of a research paper, it may exceed the suggested 8-page limit. If possible, we would appreciate any flexibility regarding format or length. If helpful, we would be happy to include a short guide or pointer table at the beginning of the report to clearly indicate where the components relevant to the course rubric appear.

Of course, we are also happy to adjust the scope or format if needed to better align with the course requirements. We would greatly appreciate any guidance you may have on this. Thank you very much for your time and consideration.

Best regards,
Xingji Jax Li
Narayani Sai Pemmaraju
Abhinav Rajput

---

## References

Assran, M., Bardes, A., Fan, D., Garrido, Q., Howes, R., Mojtaba, Komeili, Muckley, M., Rizvi, A., Roberts, C., Sinha, K., Zholus, A., Arnaud, S., Gejji, A., Martin, A., Hogan, F. R., Dugas, D., Bojanowski, P., Khalidov, V., Labatut, P., Massa, F., Szafraniec, M., Krishnakumar, K., Li, Y., Ma, X., Chandar, S., Meier, F., LeCun, Y., Rabbat, M., and Ballas, N. (2025). V-jepa 2: Self-supervised video models enable understanding, prediction and planning.

Berger, J. and Milkman, K. (2010). Social transmission, emotion, and the virality of online content. *Wharton research paper*, 106:1–52.

Berger, J. and Milkman, K. L. (2012). What makes online content viral? *Journal of marketing research*, 49(2):192–205.

Box, G. E. P. and Jenkins, G. M. (1970). *Time Series Analysis: Forecasting and Control*. Holden-Day.

Cheng, Z., Zhou, F., Xu, X., Zhang, K., Trajcevski, G., Zhong, T., and Yu, P. S. (2024). Information cascade popularity prediction via probabilistic diffusion. *IEEE Transactions on Knowledge and Data Engineering*, 36(12):8541–8555.

Cookson, J. A., Lu, R., Mullins, W., and Niessner, M. (2024). The social signal. *Journal of Financial Economics*, 158:103870.

Garrido, Q., Ballas, N., Assran, M., Bardes, A., Najman, L., Rabbat, M., Dupoux, E., and LeCun, Y. (2025). Intuitive physics understanding emerges from self-supervised pretraining on natural videos.

Ha, D. and Schmidhuber, J. (2018). World models.

Hafner, D., Pasukonis, J., Ba, J., and Lillicrap, T. (2024). Mastering diverse domains through world models.

Hochreiter, S. and Schmidhuber, J. (1997). Long short-term memory. *Neural Computation*, 9(8):1735–1780.

Hu, A., Russell, L., Yeo, H., Murez, Z., Fedoseev, G., Kendall, A., Shotton, J., and Corrado, G. (2023). Gaia-1: A generative world model for autonomous driving.

Huberman, B. A., Romero, D. M., and Wu, F. (2009). Crowdsourcing, attention and productivity. *Journal of Information science*, 35(6):758–765.

Li, X., Al Ansari, N., and Kaufman, A. (2025). Stocktwits: Comprehensive records of a financial social media platform from 2008 to 2022. *Journal of Quantitative Description: Digital Media*, 5.

Liu, Y., Hu, T., Zhang, H., Wu, H., Wang, S., Ma, L., and Long, M. (2024). iTransformer: Inverted transformers are effective for time series forecasting. In *International Conference on Learning Representations*.

Meyer, T., Kerkhof, A., Cennamo, C., and Kretschmer, T. (2024). Competing for attention on digital platforms: The case of news outlets. *Strategic Management Journal*, 45(9):1731–1790.

Nie, Y., Nguyen, N. H., Sinthong, P., and Kalagnanam, J. (2023). A time series is worth 64 words: Long-term forecasting with transformers. In *International Conference on Learning Representations*.

Pew (2018). News use across social media platforms 2018. Pew Research Center: Journalism and Media.

Rathje, S. and Van Bavel, J. J. (2025). The psychology of virality. *Trends in Cognitive Sciences*.

Sims, C. A. (1980). Macroeconomics and reality. *Econometrica*, 48(1):1–48.

Suciu, P. (2021). Americans spent on average more than 1,300 hours on social media last year. *Forbes*. Accessed: 2026-03-31.

Weng, L. and Lento, T. (2014). Topic-based clusters in egocentric networks on facebook. In *Proceedings of the International AAAI Conference on Web and Social Media*, volume 8, pages 623–626.

Weng, L. and Menczer, F. (2014). Topicality and social impact: Diverse messages but focused messengers. *arXiv preprint* arXiv:1402.5443.

Weng, L. and Menczer, F. (2015). Topicality and impact in social media: diverse messages, focused messengers. *PloS one*, 10(2):e0118410.

Weng, L., Menczer, F., and Ahn, Y.-Y. (2013). Virality prediction and community structure in social networks. *Scientific reports*, 3(1):2522.

Wu, F. and Huberman, B. A. (2007). Novelty and collective attention. *Proceedings of the National Academy of Sciences*, 104(45):17599–17601.

Zhou, F., Xu, X., Trajcevski, G., and Zhang, K. (2021). A survey of information cascade analysis: Models, predictions, and recent advances. *ACM Computing Surveys (CSUR)*, 54(2):1–36.

---

## Appendix: Validation with Simulation

---

[^1]: topic prominence reflects the amount of collective attention allocated to a topic. We avoid the term "attention" in the formal notation to prevent confusion with the attention mechanism used in modern machine learning models.

[^2]: measurement is subject to explore and change.

---

*We thank Nicholas Tomlin and Abhinav's adviser for helpful advice and discussions. All errors are our own.*
