# Markov Probability References Distillation

Purpose: topic-level synthesis of the local PDFs in this folder for Cramer-Short forecasting, probability calibration, and Markov-model tooling.

## Source Coverage

| Source PDF | Inferred title | Year/authors if available | Extraction status | One-sentence relevance |
|---|---|---|---|---|
| `./1-s2.0-S0306261914001226-main.pdf` | Optimal charging of an electric vehicle using a Markov decision process | 2014; Emil B. Iversen, Juan M. Morales, Henrik Madsen | Extracted title page, abstract, keywords | Shows stochastic dynamic programming with an inhomogeneous Markov model for user behavior and risk-aware decisions under price uncertainty. |
| `./1-s2.0-S0377221723003867-main.pdf` | Partially observable Markov decision process-based optimal maintenance planning with time-dependent observations | 2023; Akash Deep, Shiyu Zhou, Dharmaraj Veeramani, Yong Chen | Extracted title page, abstract, keywords | Useful POMDP/HMM example where hidden degradation states are inferred from time-dependent signals before optimizing actions. |
| `./2310.03775v2.pdf` | Hidden Markov Models for Stock Market Prediction | 2025; Luigi Catello, Ludovica Ruggiero, Lucia Schiavone, Mario Valentino | Extracted title page and abstract | Direct stock-forecasting HMM paper using price observations, MAPE, and directional prediction accuracy. |
| `./2510.15205v1.pdf` | Toward Black-Scholes for Prediction Markets: A Unified Kernel and Market-Maker's Handbook | 2025; Shaw Dalen | Extracted title page and abstract | Provides prediction-market probability dynamics via logit jump-diffusion, risk-neutral drift, volatility/jump calibration, and market-maker risk factors. |
| `./2511.03628v1.pdf` | LiveTradeBench: Seeking Real-World Alpha with Large Language Models | 2025; Haofei Yu, Fenghai Li, Jiaxuan You | Extracted title page and abstract-like summary | Frames trading and prediction-market evaluation as sequential decision-making under live uncertainty, delayed/noisy feedback, and multi-asset allocation. |
| `./2602.19520v1.pdf` | Decomposing Crowd Wisdom: Domain-Specific Calibration Dynamics in Prediction Markets | 2026; Nam Anh Le | Extracted title page and abstract | Strong calibration reference: prediction-market prices need domain, horizon, trade-size, and platform recalibration before treating them as probabilities. |
| `./40064_2014_Article_1373.pdf` | A methodology for stochastic analysis of share prices as Markov chains with finite states | 2014; Felix Okoe Mettle, Enoch Nii Boi Quaye, Ravenhill Adjetey Laryea | Extracted title page and abstract | Finite-state stock-price Markov chain with transition matrices, communicating classes, limiting distributions, and mean return times. |
| `./87624-submission2017-5-29.pdf` | Capturing the Order Imbalance with Hidden Markov Model: A Case of SET50 and KOSPI50 | Unknown publication year; Po-Lin Wu, Wasin Siwasarit | Extracted title page and abstract; PDF metadata has 2017 creation date | Applies HMMs to intraday order imbalance and liquidity-dependent short-horizon price-movement prediction. |
| `./A First Course in Probability and Markov Chains.pdf` | A First Course in Probability and Markov Chains | 2013; Giuseppe Modica, Laura Poggiolini | Extracted title/copyright pages | Textbook baseline for probability, Markov chains, transition matrices, recurrence, and steady-state reasoning. |
| `./A.pdf` | Hidden Markov Models (from Speech and Language Processing draft) | 2026; Daniel Jurafsky, James H. Martin | Extracted chapter header and opening sections | Concise HMM reference for Markov assumptions, transition matrices, forward-backward learning, and hidden-state inference. |
| `./FULLTEXT01.pdf` | Optimal Order Placement Using Markov Models of Limit Order Books | 2023; Max Oliveberg | Extracted title pages | Finance microstructure reference for Markov limit-order-book states and optimal execution/order placement. |
| `./Kumar_Amer_MEng_2023.pdf` | Stock Market Prediction using LSTM and Markov Chain Models: A Case Study of Royal Bank of Canada Stock | 2023; Amer Kumar | Extracted title page and abstract | Combines LSTM temporal prediction with a three-state Markov chain for transition probabilities, steady-state distribution, and mean hitting times. |
| `./Markov Chain Monte Carlo Innovations and Applications.pdf` | Markov Chain Monte Carlo: Innovations and Applications | 2005; edited by W. S. Kendall, F. Liang, J.-S. Wang | Extracted title/copyright pages and contents | MCMC primer for simulation, perfect simulation, sequential Monte Carlo, and statistical analysis of generated chains. |
| `./Markov Chains From Theory to Implementation and Experimentation.pdf` | Markov Chains: From Theory to Implementation and Experimentation | 2017; Paul A. Gagniuc | Extracted title/copyright pages and contents | Implementation-oriented Markov chain text covering stochastic matrices, transition probabilities, simulation, and experimentation. |
| `./Optimal_electricity_supply_bidding_by_Markov_decision_process.pdf` | Optimal Electricity Supply Bidding by Markov Decision Process | 2000; Haili Song, Chen-Ching Liu, Jacques Lawarree, Robert W. Dahlgren | Extracted title page and abstract | Classic MDP formulation for stochastic bidding with transition probabilities, rewards, production constraints, and finite-horizon policy optimization. |
| `./Probability and Statistics by Example Markov Chains a Primer in Random Processes and Their Applications.pdf` | Probability and Statistics by Example: II Markov Chains: a Primer in Random Processes and their Applications | 2008; Yuri Suhov, Mark Kelbert | Extracted title/copyright pages and contents | Deep textbook reference for discrete/continuous chains, hitting times, equilibrium, control/POMDP, HMMs, Baum-Welch, and Bayesian Markov-chain analysis. |
| `./aviv2005.pdf` | A Partially Observed Markov Decision Process for Dynamic Pricing | 2005; Yossi Aviv, Amit Pazgal | Extracted publication page and abstract | Revenue-management POMDP for pricing under hidden demand regimes and learning from sales observations. |
| `./fnhum-17-1249413.pdf` | Markov chains as a proxy for the predictive memory representations underlying mismatch negativity | 2023; Erich Schroger, Urte Roeber, Nina Coy | Extracted title page and abstract | Non-finance but useful reminder that transition matrices encode predictive memory/generative expectations over event sequences. |
| `./ijfs-06-00036-v2.pdf` | Hidden Markov Model for Stock Trading | 2018; Nguyet Nguyen | Extracted title page and abstract | Practical HMM trading workflow: select number of hidden states using information criteria, predict S&P 500 prices, and validate out-of-sample. |
| `./mathematics-13-00778-v2.pdf` | Dynamic Modeling of Limit Order Book and Market Maker Strategy Optimization Based on Markov Queue Theory | 2025; Fei Xie, Yang Liu, Changlong Hu, Shenbao Liang | Extracted title page and abstract | Uses Markov queue theory and HJB optimization for limit-order-book state dynamics and market-maker strategy. |
| `./risks-09-00037.pdf` | Calibration of Transition Intensities for a Multistate Model: Application to Long-Term Care | 2021; Manuel L. Esquivel, Gracinda R. Guerreiro, Matilde C. Oliveira, Pedro Corte Real | Extracted title page and abstract | Strong calibration pattern for continuous-time Markov chains: fit transition intensities to observed one-step probabilities and validate by simulation. |
| `./welton2005.pdf` | Estimation of Markov Chain Transition Probabilities and Rates from Fully and Partially Observed Data: Uncertainty Propagation, Evidence Synthesis, and Model Calibration | 2005; Nicky J. Welton, A. E. Ades | Extracted title page and abstract | Key Bayesian MCMC/evidence-synthesis reference for estimating rates/probabilities from partial observations and propagating transition uncertainty. |

## Practical Synthesis for Cramer-Short

- Treat market forecasts as stateful probability processes, not isolated point estimates: transition matrices, hidden regimes, hitting times, and steady-state distributions are reusable primitives.
- Use HMM/POMDP machinery when the useful state is latent: market regime, crowd calibration condition, liquidity state, order-book pressure, or user/investor intent.
- Separate observation models from transition models. Prices, order imbalance, news, volume, and prediction-market trades are noisy emissions, not necessarily the state itself.
- For prediction markets, quoted prices are not automatically calibrated probabilities; recalibrate by horizon, domain, liquidity/trade size, platform, and jump/event risk.
- Decision tools should optimize expected utility/reward under uncertainty, with explicit risk aversion and constraints, rather than only maximizing one-step forecast accuracy.
- In finance experiments, report directional accuracy, calibration, likelihood/log score, transition stability, and trading/portfolio outcomes; no single metric is sufficient.
- Use MCMC/SMC for posterior uncertainty and simulation-based validation when transition rates, hidden states, or calibration curves are only partially observed.

## Modeling Implications

- **Markov chains:** implement finite-state utilities for transition counting, smoothing, n-step transition matrices, communicating/ergodic checks, stationary distributions, hitting/return times, and regime persistence diagnostics.
- **HMMs:** support multi-observation emissions, state-count selection with AIC/BIC/HQC/CAIC-style criteria, forward-backward/Baum-Welch learning, and out-of-sample regime stability checks before trading use.
- **POMDP/MDP:** expose belief-state updates and finite-horizon reward optimization for actions such as trading, abstaining, market making, order placement, and portfolio rebalancing.
- **MCMC/uncertainty:** propagate uncertainty in transition probabilities/rates into forecast intervals and decision policies; avoid presenting fitted matrices as exact.
- **Continuous-time models:** where event timing matters, calibrate intensities/generators to observed one-step probabilities and validate simulated paths against external moments.
- **Prediction-market tooling:** model prices in logit/probability space with drift, volatility, jumps, horizon effects, and cross-event dependence; add recalibration layers before downstream use.
- **Microstructure:** order imbalance and limit-order-book states can be Markov/queue states; action policies should account for liquidity, spread capture, fill probability, and adverse selection.
- **Calibration:** maintain reliability diagrams/slope checks by domain and horizon; flags should warn when prices are compressed toward 50%, extreme, sparse, or platform-specific.

## Reliability and Limitations

**Textbooks and primers.** Modica/Poggiolini, Suhov/Kelbert, Gagniuc, Jurafsky/Martin, and the MCMC lecture notes are best used for definitions, algorithms, and implementation checks. They are broad and reliable for fundamentals, but they do not validate any specific financial alpha.

**Empirical/domain papers.** Stock/HMM, order-book, prediction-market, electricity, maintenance, health, and dynamic-pricing papers provide modeling patterns and cautionary examples. Their results are domain-, data-, horizon-, and cost-structure-dependent; transfer only the modeling discipline unless Cramer-Short reproduces the data conditions.

**Forecasting limitations.** Markov assumptions can underfit long memory, structural breaks, reflexive markets, and exogenous shocks. HMM state labels may be unstable or non-identifiable, and high directional accuracy can still fail after costs, slippage, liquidity constraints, or poor calibration.

**Calibration limitations.** Transition matrices and prediction-market prices should be treated as estimates with uncertainty. Partial observation requires evidence synthesis or posterior inference, and extrapolating one platform/domain/horizon to another is risky.

## Verification Checklist

- [x] `./1-s2.0-S0306261914001226-main.pdf`
- [x] `./1-s2.0-S0377221723003867-main.pdf`
- [x] `./2310.03775v2.pdf`
- [x] `./2510.15205v1.pdf`
- [x] `./2511.03628v1.pdf`
- [x] `./2602.19520v1.pdf`
- [x] `./40064_2014_Article_1373.pdf`
- [x] `./87624-submission2017-5-29.pdf`
- [x] `./A First Course in Probability and Markov Chains.pdf`
- [x] `./A.pdf`
- [x] `./FULLTEXT01.pdf`
- [x] `./Kumar_Amer_MEng_2023.pdf`
- [x] `./Markov Chain Monte Carlo Innovations and Applications.pdf`
- [x] `./Markov Chains From Theory to Implementation and Experimentation.pdf`
- [x] `./Optimal_electricity_supply_bidding_by_Markov_decision_process.pdf`
- [x] `./Probability and Statistics by Example Markov Chains a Primer in Random Processes and Their Applications.pdf`
- [x] `./aviv2005.pdf`
- [x] `./fnhum-17-1249413.pdf`
- [x] `./ijfs-06-00036-v2.pdf`
- [x] `./mathematics-13-00778-v2.pdf`
- [x] `./risks-09-00037.pdf`
- [x] `./welton2005.pdf`
