# Research References

This directory contains **60 PDFs** organized by topic. Topic-level Markdown distillations summarize the local PDFs and have been verified against local PDF text extraction. For a distilled title/abstract/keyword lookup across all papers, see [index.md](index.md).

One distillation, `prediction-markets/DISTILLATION-polymarket-forecasting.md`, covers both `prediction-markets/` and `polymarket/`. Known organization caveat: `sequence-models/chung2015.pdf` is about stock-selection/trading strategies despite the folder name, and `sequence-models/DISTILLATION.md` documents that mismatch.

## Topic Index

| Folder | PDFs | Distillation | Focus |
|---|---:|---|---|
| `agentic-workflows/` | 1 | [agentic-workflows/DISTILLATION.md](agentic-workflows/DISTILLATION.md) | Options-strategy catalogs as structured inputs for deterministic agent workflows. |
| `derivatives-risk/` | 4 | [derivatives-risk/DISTILLATION.md](derivatives-risk/DISTILLATION.md) | Options convexity, volatility trading, and LLM/model-risk controls. |
| `financial-forecasting/` | 3 | [financial-forecasting/DISTILLATION.md](financial-forecasting/DISTILLATION.md) | NLP, agentic intelligence, and time-series methods for finance. |
| `forecasting-methods/` | 3 | [forecasting-methods/DISTILLATION.md](forecasting-methods/DISTILLATION.md) | General forecasting foundations: Kalman filtering, Gaussian processes, and scoring. |
| `llm-forecasting/` | 5 | [llm-forecasting/DISTILLATION.md](llm-forecasting/DISTILLATION.md) | LLM forecasting benchmarks, financial statement analysis, and scientific synthesis agents. |
| `markov-probability/` | 22 | [markov-probability/DISTILLATION.md](markov-probability/DISTILLATION.md) | Markov chains, HMM/POMDP/MDP methods, MCMC, and transition calibration. |
| `optimization/` | 1 | [optimization/DISTILLATION.md](optimization/DISTILLATION.md) | Fuzzy stochastic prediction with Markov chain transition probabilities. |
| `polymarket/` | 5 | [prediction-markets/DISTILLATION-polymarket-forecasting.md](prediction-markets/DISTILLATION-polymarket-forecasting.md) | Polymarket order books, surveys, trader profitability, and alpha taxonomies. |
| `prediction-markets/` | 9 | [prediction-markets/DISTILLATION-polymarket-forecasting.md](prediction-markets/DISTILLATION-polymarket-forecasting.md) | Prediction-market accuracy, arbitrage, manipulation, governance, and signal quality. |
| `risk-uncertainty/` | 1 | [risk-uncertainty/DISTILLATION.md](risk-uncertainty/DISTILLATION.md) | Uncertainty-about-uncertainty and fat-tail forecasting limits. |
| `sentiment-analysis/` | 3 | [sentiment-analysis/DISTILLATION.md](sentiment-analysis/DISTILLATION.md) | Feature selection, news sentiment, and entity-aware financial text labeling. |
| `sequence-models/` | 1 | [sequence-models/DISTILLATION.md](sequence-models/DISTILLATION.md) | Misfiled stock-selection/trading-strategy paper; not sequence-model evidence. |
| `trading-strategies/` | 2 | [trading-strategies/DISTILLATION.md](trading-strategies/DISTILLATION.md) | Reinforcement-learning trading and transformer-based stock prediction. |
| **Total** | **60** |  |  |

## Per-topic PDF Index

### `agentic-workflows/` — 1 PDF

Focus: Options-strategy catalogs as structured inputs for deterministic agent workflows. Distillation: [agentic-workflows/DISTILLATION.md](agentic-workflows/DISTILLATION.md).

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [ssrn-4896867.pdf](agentic-workflows/ssrn-4896867.pdf) | Trading strategies implemented on python Part I: Options | 2024; Chenjie LI, Maxime LE FLOCH | Not an agent-systems paper, but useful as a structured options-strategy catalog that can drive deterministic agent workflows for payoff explanation, scenario analysis, and implementation checks. |

### `derivatives-risk/` — 4 PDFs

Focus: Options convexity, volatility trading, and LLM/model-risk controls. Distillation: [derivatives-risk/DISTILLATION.md](derivatives-risk/DISTILLATION.md).

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [2602.14350v1.pdf](derivatives-risk/2602.14350v1.pdf) | *Hidden Risks and Optionalities in American Options* | 2026; Noura El Hassan, Bacel Maddah, Nassim N. Taleb | Shows that American options can carry hidden convexity when rates, carry, exercise timing, liquidity, or model parameters are treated as fixed. |
| [risks-14-00089-v2.pdf](derivatives-risk/risks-14-00089-v2.pdf) | *Hidden Optionalities in American Options* | 2026; Noura El Hassan, Bacel Maddah, Nassim Nicholas Taleb | Journal version of the hidden-optionality argument, with simulations across equity puts, currency options, and stochastic-rate dynamics. |
| [ssrn-2715517.pdf](derivatives-risk/ssrn-2715517.pdf) | *A Practical Guide to Quantitative Volatility Trading* | 2016; Daniel Bloch | Broad volatility-trading manual linking multifractal returns, inefficient markets, implied-volatility surfaces, VaR limits, variance swaps, and dispersion trades. |
| [ssrn-5440116.pdf](derivatives-risk/ssrn-5440116.pdf) | *When LLMs Go Abroad: Foreign Bias in AI Financial Predictions* | 2026; Sean Cao, Charles C.Y. Wang, Xiang Yi | Not a derivatives paper, but relevant to model-risk controls because it documents systematic LLM forecast bias from asymmetric information availability. |

### `financial-forecasting/` — 3 PDFs

Focus: NLP, agentic intelligence, and time-series methods for finance. Distillation: [financial-forecasting/DISTILLATION.md](financial-forecasting/DISTILLATION.md).

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [1-s2.0-S1566253524005335-main.pdf](financial-forecasting/1-s2.0-S1566253524005335-main.pdf) | Natural language processing in finance: A survey | 2025; Kelvin Du, Yazhi Zhao, Rui Mao, Frank Xing, Erik Cambria | Maps NLP applications across financial sentiment, narrative processing, forecasting, portfolio management, risk, compliance, ESG, and explainable AI. |
| [2601.11958v1.pdf](financial-forecasting/2601.11958v1.pdf) | Autonomous Market Intelligence: Agentic AI Nowcasting Predicts Stock Returns | 2026; Zefeng Chen, Darcy Pu | Tests a fully agentic LLM nowcasting workflow on Russell 1000 stocks with explicit look-ahead-bias controls. |
| [ssrn-5140015.pdf](financial-forecasting/ssrn-5140015.pdf) | Applications of Time Series Analysis in Quantitative Finance | 2025; Yifan Guo | Reviews ARIMA, GARCH, and machine-learning time-series models for asset prediction, risk management, and portfolio optimization. |

### `forecasting-methods/` — 3 PDFs

Focus: General forecasting foundations: Kalman filtering, Gaussian processes, and scoring. Distillation: [forecasting-methods/DISTILLATION.md](forecasting-methods/DISTILLATION.md).

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [2605.20119v1.pdf](forecasting-methods/2605.20119v1.pdf) | Toto 2.0: Time Series Forecasting Enters the Scaling Era | 2026; Emaad Khwaja, Chris Lettieri, Gerald Woo, Eden Belouadah, Marc Cenac, Guillaume Jarry, Enguerrand Paquin, Xunyi Zhao, Viktoriya Zhukova, Othmane Abou-Amal, Chenghao Liu, Ameet Talwalkar, David Asker | Shows time-series foundation models can improve with scale and reports state-of-the-art benchmark results on BOOM, GIFT-Eval, and TIME. |
| [2605.21041v1.pdf](forecasting-methods/2605.21041v1.pdf) | Conditioning Gaussian Processes on Almost Anything | 2026; Henry B. Moss, Lachlan Astfalck, Thomas Cowperthwaite, Colin Doumont, Sam Willis, Philipp Hennig, Christopher Nemeth, Andrew Zammit-Mangion | Generalizes GP conditioning beyond linear-Gaussian observations using diffusion/ODE sampling and point-wise likelihood guidance. |
| [Kalman1960.pdf](forecasting-methods/Kalman1960.pdf) | A New Approach to Linear Filtering and Prediction Problems | 1960; R. E. Kalman | Foundational state-space filtering and prediction method for noisy dynamic systems, nonstationary statistics, and optimal linear filters. |

### `llm-forecasting/` — 5 PDFs

Focus: LLM forecasting benchmarks, financial statement analysis, and scientific synthesis agents. Distillation: [llm-forecasting/DISTILLATION.md](llm-forecasting/DISTILLATION.md).

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [2409.13740v2.pdf](llm-forecasting/2409.13740v2.pdf) | Language Agents Achieve Superhuman Synthesis of Scientific Knowledge | 2024; Michael D. Skarlinski, Sam Cox, Jon M. Laurent, James D. Braza, Michaela Hinks, Michael J. Hammerling, Manvitha Ponnapati, Samuel G. Rodriques, Andrew D. White | Shows cited language-agent synthesis can match or exceed domain experts on literature-search tasks, relevant to evidence retrieval before forecasts. |
| [2507.04562v3.pdf](llm-forecasting/2507.04562v3.pdf) | Evaluating LLMs on Real-World Forecasting Against Expert Forecasters | 2025; Janna Lu | Benchmarks frontier LLMs on 464 Metaculus questions and compares Brier scores against crowds and expert forecasters. |
| [2603.25040v1.pdf](llm-forecasting/2603.25040v1.pdf) | Intern-S1-Pro: Scientific Multimodal Foundation Model at Trillion Scale | 2026; Intern-S1-Pro Team / many listed authors | Describes a trillion-parameter scientific multimodal foundation model with agent capabilities and specialized scientific-task coverage. |
| [Financial-Statement-Analysis-with-Large-Language-Models.pdf](llm-forecasting/Financial-Statement-Analysis-with-Large-Language-Models.pdf) | Financial Statement Analysis with Large Language Models | 2024; Alex G. Kim, Maximilian Muhn, Valeri V. Nikolaev | Tests GPT-4 on anonymized financial statements for directional earnings-change prediction and trading-strategy construction. |
| [Kong_et_al_2024_Large_language_models.pdf](llm-forecasting/Kong_et_al_2024_Large_language_models.pdf) | A Survey of Large Language Models for Financial Applications: Progress, Prospects and Challenges | 2024; Yuqi Nie, Yaxuan Kong, Xiaowen Dong, John M. Mulvey, H. Vincent Poor, Qingsong Wen, Stefan Zohren | Surveys financial LLM applications across linguistic tasks, sentiment, time series, reasoning, agent-based modeling, and decision support. |

### `markov-probability/` — 22 PDFs

Focus: Markov chains, HMM/POMDP/MDP methods, MCMC, and transition calibration. Distillation: [markov-probability/DISTILLATION.md](markov-probability/DISTILLATION.md).

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [1-s2.0-S0306261914001226-main.pdf](markov-probability/1-s2.0-S0306261914001226-main.pdf) | Optimal charging of an electric vehicle using a Markov decision process | 2014; Emil B. Iversen, Juan M. Morales, Henrik Madsen | Shows stochastic dynamic programming with an inhomogeneous Markov model for user behavior and risk-aware decisions under price uncertainty. |
| [1-s2.0-S0377221723003867-main.pdf](markov-probability/1-s2.0-S0377221723003867-main.pdf) | Partially observable Markov decision process-based optimal maintenance planning with time-dependent observations | 2023; Akash Deep, Shiyu Zhou, Dharmaraj Veeramani, Yong Chen | Useful POMDP/HMM example where hidden degradation states are inferred from time-dependent signals before optimizing actions. |
| [2310.03775v2.pdf](markov-probability/2310.03775v2.pdf) | Hidden Markov Models for Stock Market Prediction | 2025; Luigi Catello, Ludovica Ruggiero, Lucia Schiavone, Mario Valentino | Direct stock-forecasting HMM paper using price observations, MAPE, and directional prediction accuracy. |
| [2510.15205v1.pdf](markov-probability/2510.15205v1.pdf) | Toward Black-Scholes for Prediction Markets: A Unified Kernel and Market-Maker's Handbook | 2025; Shaw Dalen | Provides prediction-market probability dynamics via logit jump-diffusion, risk-neutral drift, volatility/jump calibration, and market-maker risk factors. |
| [2511.03628v1.pdf](markov-probability/2511.03628v1.pdf) | LiveTradeBench: Seeking Real-World Alpha with Large Language Models | 2025; Haofei Yu, Fenghai Li, Jiaxuan You | Frames trading and prediction-market evaluation as sequential decision-making under live uncertainty, delayed/noisy feedback, and multi-asset allocation. |
| [2602.19520v1.pdf](markov-probability/2602.19520v1.pdf) | Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets | 2026; Nam Anh Le | Strong calibration reference: prediction-market prices need domain, horizon, trade-size, and platform recalibration before treating them as probabilities. |
| [40064_2014_Article_1373.pdf](markov-probability/40064_2014_Article_1373.pdf) | A methodology for stochastic analysis of share prices as Markov chains with finite states | 2014; Felix Okoe Mettle, Enoch Nii Boi Quaye, Ravenhill Adjetey Laryea | Finite-state stock-price Markov chain with transition matrices, communicating classes, limiting distributions, and mean return times. |
| [87624-submission2017-5-29.pdf](markov-probability/87624-submission2017-5-29.pdf) | Capturing the Order Imbalance with Hidden Markov Model: A Case of SET50 and KOSPI50 | Unknown publication year; Po-Lin Wu, Wasin Siwasarit | Applies HMMs to intraday order imbalance and liquidity-dependent short-horizon price-movement prediction. |
| [A First Course in Probability and Markov Chains.pdf](markov-probability/A%20First%20Course%20in%20Probability%20and%20Markov%20Chains.pdf) | A First Course in Probability and Markov Chains | 2013; Giuseppe Modica, Laura Poggiolini | Textbook baseline for probability, Markov chains, transition matrices, recurrence, and steady-state reasoning. |
| [A.pdf](markov-probability/A.pdf) | Hidden Markov Models (from Speech and Language Processing draft) | 2026; Daniel Jurafsky, James H. Martin | Concise HMM reference for Markov assumptions, transition matrices, forward-backward learning, and hidden-state inference. |
| [FULLTEXT01.pdf](markov-probability/FULLTEXT01.pdf) | Optimal Order Placement Using Markov Models of Limit Order Books | 2023; Max Oliveberg | Finance microstructure reference for Markov limit-order-book states and optimal execution/order placement. |
| [Kumar_Amer_MEng_2023.pdf](markov-probability/Kumar_Amer_MEng_2023.pdf) | Stock Market Prediction using LSTM and Markov Chain Models: A Case Study of Royal Bank of Canada Stock | 2023; Amer Kumar | Combines LSTM temporal prediction with a three-state Markov chain for transition probabilities, steady-state distribution, and mean hitting times. |
| [Markov Chain Monte Carlo Innovations and Applications.pdf](markov-probability/Markov%20Chain%20Monte%20Carlo%20Innovations%20and%20Applications.pdf) | Markov Chain Monte Carlo: Innovations and Applications | 2005; edited by W. S. Kendall, F. Liang, J.-S. Wang | MCMC primer for simulation, perfect simulation, sequential Monte Carlo, and statistical analysis of generated chains. |
| [Markov Chains From Theory to Implementation and Experimentation.pdf](markov-probability/Markov%20Chains%20From%20Theory%20to%20Implementation%20and%20Experimentation.pdf) | Markov Chains: From Theory to Implementation and Experimentation | 2017; Paul A. Gagniuc | Implementation-oriented Markov chain text covering stochastic matrices, transition probabilities, simulation, and experimentation. |
| [Optimal_electricity_supply_bidding_by_Markov_decision_process.pdf](markov-probability/Optimal_electricity_supply_bidding_by_Markov_decision_process.pdf) | Optimal Electricity Supply Bidding by Markov Decision Process | 2000; Haili Song, Chen-Ching Liu, Jacques Lawarree, Robert W. Dahlgren | Classic MDP formulation for stochastic bidding with transition probabilities, rewards, production constraints, and finite-horizon policy optimization. |
| [Probability and Statistics by Example Markov Chains a Primer in Random Processes and Their Applications.pdf](markov-probability/Probability%20and%20Statistics%20by%20Example%20Markov%20Chains%20a%20Primer%20in%20Random%20Processes%20and%20Their%20Applications.pdf) | Probability and Statistics by Example: II Markov Chains: a Primer in Random Processes and their Applications | 2008; Yuri Suhov, Mark Kelbert | Deep textbook reference for discrete/continuous chains, hitting times, equilibrium, control/POMDP, HMMs, Baum-Welch, and Bayesian Markov-chain analysis. |
| [aviv2005.pdf](markov-probability/aviv2005.pdf) | A Partially Observed Markov Decision Process for Dynamic Pricing | 2005; Yossi Aviv, Amit Pazgal | Revenue-management POMDP for pricing under hidden demand regimes and learning from sales observations. |
| [fnhum-17-1249413.pdf](markov-probability/fnhum-17-1249413.pdf) | Markov chains as a proxy for the predictive memory representations underlying mismatch negativity | 2023; Erich Schroger, Urte Roeber, Nina Coy | Non-finance but useful reminder that transition matrices encode predictive memory/generative expectations over event sequences. |
| [ijfs-06-00036-v2.pdf](markov-probability/ijfs-06-00036-v2.pdf) | Hidden Markov Model for Stock Trading | 2018; Nguyet Nguyen | Practical HMM trading workflow: select number of hidden states using information criteria, predict S&P 500 prices, and validate out-of-sample. |
| [mathematics-13-00778-v2.pdf](markov-probability/mathematics-13-00778-v2.pdf) | Dynamic Modeling of Limit Order Book and Market Maker Strategy Optimization Based on Markov Queue Theory | 2025; Fei Xie, Yang Liu, Changlong Hu, Shenbao Liang | Uses Markov queue theory and HJB optimization for limit-order-book state dynamics and market-maker strategy. |
| [risks-09-00037.pdf](markov-probability/risks-09-00037.pdf) | Calibration of Transition Intensities for a Multistate Model: Application to Long-Term Care | 2021; Manuel L. Esquivel, Gracinda R. Guerreiro, Matilde C. Oliveira, Pedro Corte Real | Strong calibration pattern for continuous-time Markov chains: fit transition intensities to observed one-step probabilities and validate by simulation. |
| [welton2005.pdf](markov-probability/welton2005.pdf) | Estimation of Markov Chain Transition Probabilities and Rates from Fully and Partially Observed Data: Uncertainty Propagation, Evidence Synthesis, and Model Calibration | 2005; Nicky J. Welton, A. E. Ades | Key Bayesian MCMC/evidence-synthesis reference for estimating rates/probabilities from partial observations and propagating transition uncertainty. |

### `optimization/` — 1 PDF

Focus: Fuzzy stochastic prediction with Markov chain transition probabilities. Distillation: [optimization/DISTILLATION.md](optimization/DISTILLATION.md).

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [1-s2.0-S1568494609001537-main.pdf](optimization/1-s2.0-S1568494609001537-main.pdf) | Incorporating the Markov chain concept into fuzzy stochastic prediction of stock indexes | 2010; Yi-Fan Wang, Shihmin Cheng, Mei-Hua Hsu | Shows a compact stock-index forecasting optimizer that combines fuzzy stochastic parameters with Markov rising/falling probabilities to improve prediction accuracy and stop-loss confidence. |

### `polymarket/` — 5 PDFs

Focus: Polymarket order books, surveys, trader profitability, and alpha taxonomies. Distillation: [prediction-markets/DISTILLATION-polymarket-forecasting.md](prediction-markets/DISTILLATION-polymarket-forecasting.md).

Note: this folder shares the combined Polymarket/prediction-market distillation at `prediction-markets/DISTILLATION-polymarket-forecasting.md`.

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [2604.24366v1.pdf](polymarket/2604.24366v1.pdf) | *The Anatomy of a Decentralized Prediction Market: Microstructure Evidence from the Polymarket Order Book* | 2026; Philipp D. Dubach | Order-book microstructure: spreads, depth, trade-direction inference, wash metrics, depth decay near resolution. |
| [ssrn-6161946.pdf](polymarket/ssrn-6161946.pdf) | *Can Polymarket Become a More Accurate “Survey”? A Replacement for Traditional Surveys That Are Easy to Game?* | 2026; Fahril Irkham | Conceptual survey-style paper comparing Polymarket incentives with traditional surveys; summarizes binary contracts, USDC/Polygon/CLOB/oracle mechanics and risks. |
| [ssrn-6443103.pdf](polymarket/ssrn-6443103.pdf) | *Who Wins and Who Loses In Prediction Markets? Evidence from Polymarket* | 2026; Pat Akey, Vincent Grégoire, Nicolas Harvie, Charles Martineau | Large wallet-level profitability study: 2.4M users, $67B volume, 588M trades; top 1% capture 76.5% of gains; aggregate calibration. |
| [ssrn-6624899.pdf](polymarket/ssrn-6624899.pdf) | *Smart Money on Polymarket: A Behavioral Anatomy of 273 Top Prediction-Market Traders* | 2026; authors Unknown in extracted text | Polydata.pro wallet panel; profit concentration, resolution-edge holding, automation/bot fingerprinting, politics as dominant profit pool. |
| [ssrn-6625018.pdf](polymarket/ssrn-6625018.pdf) | *What Is the Alpha on Polymarket? A Methodology of Probability Mispricing and a Taxonomy of Eleven Profitable Regimes* | 2026; authors Unknown in extracted text | Companion to Smart Money; maps eleven profitable regimes to slow information, behavioral bias, microstructure inefficiency, and cross-market inconsistency. |

### `prediction-markets/` — 9 PDFs

Focus: Prediction-market accuracy, arbitrage, manipulation, governance, and signal quality. Distillation: [prediction-markets/DISTILLATION-polymarket-forecasting.md](prediction-markets/DISTILLATION-polymarket-forecasting.md).

Note: `prediction-markets/DISTILLATION-polymarket-forecasting.md` is the primary combined distillation and also covers `polymarket/`.

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [2508.03474v1.pdf](prediction-markets/2508.03474v1.pdf) | *Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets* | 2025; Oriol Saguillo, Vahid Ghafouri, Lucianna Kiffer, Guillermo Suarez-Tangil | Defines/estimates Market Rebalancing and Combinatorial Arbitrage on Polymarket; about $40M realized arbitrage profit. |
| [2601.20452v1.pdf](prediction-markets/2601.20452v1.pdf) | *Manipulation in Prediction Markets: An Agent-based Modeling Experiment* | 2026; Bridget Smart, Ebba Mark, Anne Bastian, Josefina Waugh | Models whale manipulation; distortion scales with capital share and persists with herding / slow learning. |
| [2602.05181v1.pdf](prediction-markets/2602.05181v1.pdf) | *Prediction Laundering: The Illusion of Neutrality, Transparency, and Governance in Polymarket* | 2026; Yasaman Rohanifar, Syed Ishtiaque Ahmed, Sharifa Sultana | Sociotechnical critique of platform selection, probability flattening, capital opacity, and oracle/governance dispute erasure. |
| [2603.03136v1.pdf](prediction-markets/2603.03136v1.pdf) | *The Anatomy of Polymarket: Evidence from the 2024 Presidential Election* | 2026; Kwok Ping Tsang, Zichao Yang | Transaction-level Polygon analysis; decomposes volume/net inflow/gross activity; studies election episodes, whale inflows, arbitrage deviations, Kyle's λ. |
| [2603.03152v2.pdf](prediction-markets/2603.03152v2.pdf) | *Political Shocks and Price Discovery in Prediction Markets: Evidence from the 2024 U.S. Presidential Election* | 2026; Kwok Ping Tsang, Zichao Yang | Event-study of debate, assassination attempt, and Biden dropout; establishes persistence-vs-reversal shock patterns. |
| [AccuracyEfficiencyClintonHuangNov2025.pdf](prediction-markets/AccuracyEfficiencyClintonHuangNov2025.pdf) | *Prediction Markets? The Accuracy and Efficiency of $2.4 Billion in the 2024 Presidential Election* | 2025; Joshua D. Clinton, TzuFeng Huang | Cross-exchange political-market accuracy/efficiency; Polymarket 67% active-market accuracy; arbitrage peaked in final two weeks; negative/weak autocorrelation. |
| [BetaHMMpolymarket-18-1.pdf](prediction-markets/BetaHMMpolymarket-18-1.pdf) | *Predicting Prediction Markets: A Beta-Hidden Markov Modeling Approach* | 2025; Conrad Oskar Voigt | Thesis introducing Groupwise Beta-HMM for interdependent Polymarket contracts; reports 89.3% classification accuracy and regime asymmetries. |
| [futureinternet-17-00487-v2.pdf](prediction-markets/futureinternet-17-00487-v2.pdf) | *Beyond the Polls: Quantifying Early Signals in Decentralized Prediction Markets with Cross-Correlation and Dynamic Time Warping* | 2025; Francisco Cordoba Otalora, Marinos Themistocleous | DPMVF framework; Polymarket led polling shifts by up to 14 days in contested swing states; low-volatility states were low signal. |
| [ssrn-5910522.pdf](prediction-markets/ssrn-5910522.pdf) | *Exploring Decentralized Prediction Markets: Accuracy, Skill, and Bias on Polymarket* | 2025; Felix Reichenbach, Martin Walther | >124M-trade Polymarket dataset; accuracy, default/YES overtrading, <30% profitable traders, persistent skill. |

### `risk-uncertainty/` — 1 PDF

Focus: Uncertainty-about-uncertainty and fat-tail forecasting limits. Distillation: [risk-uncertainty/DISTILLATION.md](risk-uncertainty/DISTILLATION.md).

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [risks-13-00247-v2.pdf](risk-uncertainty/risks-13-00247-v2.pdf) | *The Regress of Uncertainty and the Forecasting Paradox* | 2025; Nassim Nicholas Taleb and Pasquale Cirillo | Formalizes the claim that uncertainty about uncertainty thickens predictive tails, making future risk structurally more extreme than in-sample history suggests. |

### `sentiment-analysis/` — 3 PDFs

Focus: Feature selection, news sentiment, and entity-aware financial text labeling. Distillation: [sentiment-analysis/DISTILLATION.md](sentiment-analysis/DISTILLATION.md).

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [a-review-on-feature-selection-techniques-for-sentiment-virrrmgn.pdf](sentiment-analysis/a-review-on-feature-selection-techniques-for-sentiment-virrrmgn.pdf) | A Review on Feature Selection Techniques for Sentiment Analysis | 2022; Kriti Agarwal | Reviews and tests feature-selection techniques for stock-market Twitter sentiment, showing simple word-frequency filtering can outperform more aggressive reductions for a Naive Bayes classifier. |
| [jrfm-18-00412.pdf](sentiment-analysis/jrfm-18-00412.pdf) | News Sentiment and Stock Market Dynamics: A Machine Learning Investigation | 2025; Milivoje Davidovic, Jacqueline McCleary | Large financial-news study finding that TextBlob/VADER/FinBERT sentiment alone has weak standalone predictive power, while VIX, volume, controls, and weekend/holiday adjustments add more useful signal. |
| [s42521-025-00162-3.pdf](sentiment-analysis/s42521-025-00162-3.pdf) | Financial sentiment analysis with FUNNEL: filtered UNion for NER-based ensemble labeling | 2025; William Nordansjö, Fredrik Fourong, Muhammad Qasim | Provides an entity-aware ensemble-labeling framework that improves stock-specific financial-news labels by combining keyword heuristics with spaCy NER and weighted voting. |

### `sequence-models/` — 1 PDF

Focus: Misfiled stock-selection/trading-strategy paper; not sequence-model evidence. Distillation: [sequence-models/DISTILLATION.md](sequence-models/DISTILLATION.md).

Caveat: the single PDF is stock-selection/trading-strategy evidence, not sequence-model architecture evidence.

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [chung2015.pdf](sequence-models/chung2015.pdf) | The selection of popular trading strategies | 2015; Yi-Tsai Chung, Tung Liang Liao, Yi-Chein Chiang | Despite the folder name, this paper compares popular stock-selection strategies using stochastic dominance rather than sequence models. |

### `trading-strategies/` — 2 PDFs

Focus: Reinforcement-learning trading and transformer-based stock prediction. Distillation: [trading-strategies/DISTILLATION.md](trading-strategies/DISTILLATION.md).

| PDF | Inferred title | Year/authors | Topic/relevance |
|---|---|---|---|
| [1-s2.0-S0952197624015239-main.pdf](trading-strategies/1-s2.0-S0952197624015239-main.pdf) | An adaptive financial trading strategy based on proximal policy optimization and financial signal representation | 2024; Lin Wang, Xuerui Wang | Proposes FSRPPO, combining financial signal representation with PPO to reduce noise, trading frequency, costs, and risk. |
| [2412.20138v7.pdf](trading-strategies/2412.20138v7.pdf) | TradingAgents: Multi-Agents LLM Financial Trading Framework | 2025; Yijia Xiao, Edward Sun, Di Luo, Wei Wang | Proposes specialized LLM agents for fundamental, sentiment, technical, trading, bull/bear research, and risk-management roles. |
