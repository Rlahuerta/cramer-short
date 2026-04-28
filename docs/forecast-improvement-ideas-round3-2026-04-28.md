# Forecast & Prediction Improvement Ideas — Round 3 (2026-04-28)

> **Status:** review-ready brainstorm. Each idea names: source, what it changes
> in the current pipeline, the *true-improvement gate* (how we'd know it
> beats the existing baseline), and a rough effort estimate. Ranked by
> expected uplift × inverse effort, **excluding** ideas already shipped or
> already covered in `references-deep-dive-2026-04-28.md` and
> `forecast-improvement-review-2026-04-28.md`.
>
> **Already shipped (do not re-propose):** Beta-HMM (W1 P1), logit
> jump-diffusion (W1 P2), domain×horizon recalibration (W1 P3),
> convergence-time signal (W2 P4), range-before-point scaffolding (W2 P5),
> NIG scale-mixture HMM spike (W2 P6), Q→P Girsanov, vol-regime detection,
> entity-aware sentiment.
>
> **Already deferred in prior round (do not re-propose without new evidence):**
> Hurst exponent dial, multi-criterion HMM K-selection, trade-size
> stratified nudging.

---

## 0. TL;DR — Top 8, ranked

| # | Idea | Source | Where it slots in | Effort | Expected uplift |
|---|------|--------|-------------------|--------|-----------------|
| 1 | **On-chain Polymarket trade direction (replace WebSocket inference)** | Dubach 2026 (arXiv 2604.24366) | `src/tools/finance/polymarket-*.ts` | M | 🔥🔥🔥 — public-feed direction is correct only **~59%** of the time vs on-chain truth. Effective half-spread *flips sign* on 67% of markets. Every flow-based signal we derive from the public feed is partially noise. |
| 2 | **Online conformal prediction wrapper for forecast intervals** | Angelopoulos, Candès & Tibshirani 2023 (PID, arXiv 2307.16895); Yang, Candès & Lei 2024 (Bellman BCI, arXiv 2402.05203) | New `src/tools/finance/conformal.ts`, wraps `polymarket_forecast` & `markov_distribution` outputs | M | 🔥🔥 — finite-sample marginal coverage guarantee under arbitrary distribution shift. Replaces our hand-tuned ±band heuristics. |
| 3 | **Hawkes self-exciting jump intensity (extend Wave-1 P2)** | Cestari et al. 2023 (arXiv 2312.16190); Oxenhorn 2022 (arXiv 2205.06338) | Extends `logit-jump-diffusion.ts` from Poisson to Hawkes intensity λ(t) = μ + Σ α·e^{-β(t-t_i)} | M-H | 🔥🔥 — captures jump *clustering*. Today's P2 model treats jumps as independent Poisson, but news shocks cluster (depegging cascades, FOMC follow-through). Hawkes was the original SOTA for LOB return-direction prediction. |
| 4 | **ADWIN drift detector triggers Markov refit** | Yelleti 2025 (arXiv 2504.10229); classic Bifet & Gavaldà 2007 | New `src/utils/drift-detector.ts`; wired into `markov-distribution.ts` history loader | S-M | 🔥🔥 — replaces our fixed 60–90 day rolling window with a "train-only-when-required" policy. ADWIN beat fixed-window baselines on 4/5 fraud datasets. Same mechanism flags regime changes earlier than our current break-detector. |
| 5 | **LASSO-PCA forecast pooling across model variants** | Uniejewski & Maciejowska 2022 (arXiv 2207.04794) | New `src/tools/finance/forecast-pooling.ts` aggregator over {Markov, jump-diffusion, Beta-HMM, NIG} estimates | M | 🔥 — replaces our fixed log-odds combination weights with data-driven LASSO + PCA on past calibration error. Robust to forecaster correlation, outperformed simple-mean / weighted-mean on 4 markets in the source paper. |
| 6 | **Multivariate fractional Brownian motion (mfBm) for realised vol** | Bibinger, Yu & Zhang 2025 (arXiv 2504.15985) | Replace static σ in `markov-distribution.ts` realised-vol layer with mfBm-based forecast | H | 🔥 — beats vector-HAR out-of-sample, especially when component Hurst exponents differ. Synergistic with the deferred Hurst-exponent dial. |
| 7 | **Online rolling controlled SMC / particle filter** | Xue, Finke & Johansen 2025 (arXiv 2508.00696) | Optional drop-in replacement for batch HMM fits in stochastic-volatility mode | H | 🔥 — bounded-cost real-time state inference. Helpful if/when we want sub-daily updates on the trajectory MC. |
| 8 | **Crypto unpredictability sanity-check baseline** | Puoti, Pittorino & Roveri 2025 (arXiv 2502.09079) | New eval harness in `research/spikes/crypto_baseline/` | S | 🔥 — *negative* result paper showing simple naïve baselines often beat ML/DL on crypto. Use as a regression guard: any new BTC forecast must beat a no-change baseline before we ship it. |

Items 1–4 are the highest-ROI work. Item 1 in particular is the most
consequential finding I've seen in this round — our current Polymarket flow
signals derived from the public WebSocket feed have systematic measurement
error that the literature now explicitly documents.

---

## 1. Idea #1 — On-Chain Polymarket Trade Direction (TOP PRIORITY)

**Source.** Dubach (2026), *The Anatomy of a Decentralized Prediction
Market: Microstructure Evidence from the Polymarket Order Book* (arXiv
[2604.24366](https://arxiv.org/abs/2604.24366), April 2026). 30 billion
WebSocket events over 52 days, joined to the on-chain `OrderFilled` event
record on a pre-registered panel of 600 markets.

**The killer finding (verbatim from the abstract).** "Trade direction
inferred from Polymarket's public order-book feed agrees with on-chain
ground truth only **~59%** of the time (panel mean 0.615, 95% CI [0.58,
0.65]), barely above the 50% chance baseline. On the comparable subset of
the top-100 panel, the **effective half-spread changes sign** between feed-
and on-chain directions on **67% of markets** in a first 7-day window and
50% in a second non-overlapping window, with **Kyle's lambda flipping** on
60% and 43% respectively; neither window recovers the on-chain sign at
anything close to the **~80%** rate that Lee–Ready achieves on equity
venues."

**Why this is critical for us.**
- We currently derive several signals from the public Polymarket feed
  (volume, recent activity, trade direction). Any signal that depends on
  knowing *who initiated* a trade — buyer-pressure, lambda-style price
  impact, effective spread — is built on a 59%-correct direction inference.
- The effect is not random noise; the sign of the effective half-spread
  flips on the majority of markets, so we are systematically biasing the
  flow signal.
- The same paper documents three other stylised facts directly relevant to
  us: a **longshot spread premium** (consistent with our domain×horizon
  recalibration in W1 P3), a **depth-concentration profile closer to a
  uniform geometric grid** than top-of-book (changes how we should weight
  small vs large prints), and **depth decay near resolution** with an
  in-category slope of −0.55 on log seconds-to-close (we should *down*-
  weight near-expiry liquidity proxies).

**Implementation plan.**
1. Add an on-chain ingestion path next to the existing WebSocket consumer.
   Use Polymarket's CTF Exchange contract `OrderFilled` event log (the
   replication package the paper releases is open source — pin to that
   schema).
2. Replace direction inference in any flow-derived signal with the on-chain
   `taker` flag.
3. Where on-chain data is unavailable (e.g. cold-cache), explicitly mark
   the flow signal as "low-confidence (public-feed direction)" and reduce
   its log-odds weight by 50%.
4. Keep the WebSocket feed for *price* and *depth* — those are accurate.
   Only the **direction** field is the problem.

**True-improvement gate.**
- Re-run the BTC 14d backtest with on-chain direction. The flow-derived
  short signal's hit-rate should improve from baseline. (We have an
  existing 14d bearish-break SELL gate test in
  `markov-backtest.integration.test.ts` — extend it to include flow
  direction.)
- Sanity check: replicate the 67% sign-flip statistic on our own market
  panel before claiming the fix.

**Effort:** Medium. The on-chain ingestion is new code and new infra
(RPC node or indexer subscription), but the API surface is small.
Approximately 1 PR for ingestion + 1 PR for signal-rewiring.

---

## 2. Idea #2 — Online Conformal Prediction Wrappers

**Sources.**
- Angelopoulos, Candès & Tibshirani (2023), *Conformal PID Control for
  Time Series Prediction* (arXiv [2307.16895](https://arxiv.org/abs/2307.16895))
  — online conformal prediction with **PID-controller** scoring that
  adapts to seasonality, trend, and distribution shift. Improves CDC's
  ensemble forecaster's coverage on 4-week-ahead COVID counts; tested on
  electricity demand and **market returns**.
- Yang, Candès & Lei (2024), *Bellman Conformal Inference: Calibrating
  Prediction Intervals For Time Series* (arXiv
  [2402.05203](https://arxiv.org/abs/2402.05203)) — wraps any forecaster,
  uses dynamic programming on a 1D stochastic control problem to optimise
  *interval length* under a coverage constraint. **Long-term coverage
  guarantee under arbitrary distribution shift and temporal dependence**,
  even with poor multi-step forecasts.

**What this changes.** Today our `markov_distribution` and
`polymarket_forecast` outputs include 5th/95th percentile bounds derived
from Monte Carlo quantiles. Those have *no* coverage guarantee — when the
underlying model is mis-specified (and on crypto it always is), the 90%
band misses ~20–30% of the time in our backtests. Online conformal
wrappers fix this with finite-sample guarantees.

**Implementation plan.**
1. Create `src/tools/finance/conformal.ts` exposing two wrappers:
   - `conformalPID(forecastStream, target=0.9, lr=0.05)` — keeps a rolling
     residual store, adjusts a single quantile threshold via a PID
     controller (the 2307.16895 algorithm).
   - `bellmanConformal(forecastStream, target=0.9, horizon=14)` — solves
     the 1D SCP at each step (the 2402.05203 algorithm). Heavier but
     produces shorter intervals on long horizons.
2. Wire as an optional output enrichment on the forecast tools (don't
   replace existing bounds; add `conformalLow`/`conformalHigh` alongside).
3. Mirror in `research/models/conformal.py` for parity testing.

**True-improvement gate.**
- Run on 6+ months of historical BTC 14d forecasts. Empirical coverage of
  the conformal interval must reach 90% ±2pp; the existing 5/95 quantile
  band should miss this target.
- Average interval width should be ≤ 1.5× the quantile band's width when
  coverage targets are matched (conformal must not pay a huge width cost).

**Effort:** Medium. Pure post-processing, no model surgery. Strong test
coverage available because the algorithms have closed-form behaviour.

---

## 3. Idea #3 — Hawkes Self-Exciting Jump Intensity

**Sources.**
- Cestari, Barchi, Busetto, Marazzina & Formentin (2023), *Hawkes-based
  cryptocurrency forecasting via Limit Order Book data* (arXiv
  [2312.16190](https://arxiv.org/abs/2312.16190)) — Hawkes + continuous
  output error on USDT/USD LOB, surpassed benchmark models in **both
  prediction accuracy and cumulative profit** across 50 Monte Carlo
  scenarios.
- Oxenhorn (2022), *A Multivariate Hawkes Process Model for
  Stablecoin–Cryptocurrency Depegging Event Dynamics* (arXiv
  [2205.06338](https://arxiv.org/abs/2205.06338)) — multivariate
  mutually-exciting Hawkes captures USDT depeg → BTC contagion.

**What this changes in our W1 P2 jump-diffusion.** Today
`logit-jump-diffusion.ts` uses an i.i.d. Bernoulli per day for jump
arrival, parameterised by Polymarket-implied jump probability. That
treats jumps as **memoryless**, but real markets show clear clustering:
one jump raises the conditional intensity of further jumps for hours/days.
A self-exciting Hawkes intensity

```
λ(t) = μ + Σ_{t_i < t} α · exp(−β · (t − t_i))
```

decays back toward baseline μ but spikes whenever a recent jump occurs.
On Polymarket, this maps directly to "FOMC surprise → multiple resolved
follow-on markets in the next 24h" or "BTC depeg event → cluster of
crypto-related markets reprice".

**Implementation plan.**
1. Add `HawkesIntensity` class in `src/tools/finance/hawkes.ts` with two
   parameters (α excitation, β decay) calibrated by MLE on past
   inter-jump intervals.
2. Modify `logit-jump-diffusion.ts` to accept an optional `intensitySpec`
   (defaults to current Bernoulli for back-compat). When `intensitySpec
   === "hawkes"`, the daily jump probability is the Hawkes integral over
   that day given recent simulation history.
3. For the multivariate variant (Oxenhorn): cross-asset excitation matrix
   between BTC ↔ ETH ↔ SOL as a Phase-2 follow-on.

**True-improvement gate.**
- Backtest BTC 14d trajectory MC during a known clustered-event window
  (e.g. March 2023 banking crisis week, or any FOMC week). Hawkes-driven
  paths must produce realised-volatility quantiles that better envelope
  the actual spot path than i.i.d. Poisson-driven paths.
- Statistical test: Ljung–Box on simulated jump-time inter-arrival
  intervals must show no remaining autocorrelation; the i.i.d. baseline
  fails this test on real BTC data.

**Effort:** Medium-High. Hawkes MLE is well-known but numerically fiddly
(branching-ratio constraint α/β < 1 to avoid explosion). Mirror to Python
spike first, then port.

---

## 4. Idea #4 — ADWIN Drift Detector Triggers Markov Refit

**Source.** Yelleti (2025), *ROSFD: Robust Online Streaming Fraud
Detection with Resilience to Concept Drift in Data Streams* (arXiv
[2504.10229](https://arxiv.org/abs/2504.10229)) — empirical comparison of
DDM, EDDM, and ADWIN drift detectors. **ADWIN won across 4/5 datasets**
with a "train-only-when-required" policy that drastically reduced retrain
frequency without AUC loss. Reinforces the well-established Bifet &
Gavaldà 2007 ADWIN result.

**What this changes.** Today `markov-distribution.ts` re-fits the
transition matrix on every call using a fixed 60–90 day rolling window.
This conflates two failure modes:
1. The window is **too short** during a stable regime → noisy transition
   estimates.
2. The window is **too long** across a regime change → blends pre- and
   post-break dynamics.

ADWIN over the realised-return stream gives a principled adaptive window.
When ADWIN detects no drift, the window grows (= more data, less noise);
when it flags drift, the window snaps back to just the post-drift
observations.

**Implementation plan.**
1. New `src/utils/adwin.ts` — pure-TS ADWIN window (~150 lines, no
   dependencies). Stores compressed return buckets, runs the
   Hoeffding-bound test on bucket means.
2. Wrap the historical loader in `markov-distribution.ts`: ask ADWIN for
   "earliest non-drifted timestamp", trim history accordingly.
3. Surface a `windowSizeUsed` and `driftDetectedAt` field in the output
   metadata (already have a structural-break flag — this is the same
   semantic, but principled rather than threshold-tuned).
4. Mirror in `research/models/adwin.py`.

**True-improvement gate.**
- Synthetic drift test (TDD-friendly): generate 100 days of regime A
  followed by 50 days of regime B with different μ/σ. ADWIN must detect
  the change point within ≤5 days of the true change; the resulting
  Markov refit on the trimmed window must produce a transition matrix
  closer in Frobenius norm to the regime-B truth than the fixed-window
  estimator.
- Real-data: backtest BTC 14d forecasts across known regime breaks (e.g.
  Nov 2022 FTX collapse). ADWIN-trimmed forecasts should have lower
  90%-band miss rate than fixed-window forecasts during the post-break
  fortnight.

**Effort:** Small-Medium. ADWIN is a well-understood algorithm with a
clean TS port and natural unit tests.

---

## 5. Idea #5 — LASSO-PCA Forecast Pooling Across Model Variants

**Source.** Uniejewski & Maciejowska (2022), *LASSO Principal Component
Averaging — a fully automated approach for point forecast pooling* (arXiv
[2207.04794](https://arxiv.org/abs/2207.04794)). Tested on 650 hourly
day-ahead electricity price forecasts across 4 markets; outperforms
simple-mean, AW/WAW, plain LASSO and plain PCA on MAE.

**What this changes.** We currently combine signals via a fixed weighted
log-odds with manually-tuned per-asset-class weights (Idea #3 in W1 P3
addressed *recalibration* of Polymarket but not *pooling* across
model families). LASSO-PCA selects calibration-window-conditional
weights automatically, robust to forecaster correlation.

**Implementation plan.**
1. Build a calibration store (already kicking around: `.cramer-short/
   memory/` daily files include past forecasts and outturns) as a matrix
   of past predictions × past outturns.
2. New `src/tools/finance/forecast-pooling.ts` runs LASSO on PCA
   components of that matrix, returns weights per source.
3. Replace the fixed `weights` in `weightedLogOdds` with the LASSO-PCA
   output when ≥30 historical (forecast, outturn) pairs are available;
   fall back to current fixed weights otherwise.

**True-improvement gate.**
- Hold-out 30-day window. Compare LASSO-PCA pooled forecast Brier score
  against fixed-weight log-odds. Required: ≥3pp Brier improvement on at
  least 60% of holdout days.

**Effort:** Medium. Needs a real calibration store. Can defer behind a
feature flag.

---

## 6. Idea #6 — Multivariate Fractional Brownian Motion (mfBm)

**Source.** Bibinger, Yu & Zhang (2025), *Modeling and Forecasting
Realized Volatility with Multivariate Fractional Brownian Motion* (arXiv
[2504.15985](https://arxiv.org/abs/2504.15985)). "**mfBm-based forecasts
outperform the (vector) HAR model**" out-of-sample, with biggest gains
when component Hurst exponents differ.

**What this changes.** Synergistic with the previously-deferred Hurst
exponent regime dial (Idea #8 in `references-deep-dive`). Instead of
treating the single-asset Hurst as a regime indicator, we'd estimate a
*joint* Hurst structure across BTC / ETH / SPY / GLD and use the
time-reversibility test from the paper to decide whether the
optimal-forecast formula applies. Adds proper cross-asset
realised-volatility forecasts that drop into our trajectory MC's σ input.

**True-improvement gate.** Out-of-sample MSE on realised-vol forecasts
must beat HAR baseline by ≥5% on a 12-month BTC/ETH/SPY backtest.

**Effort:** High. Estimation routines are non-trivial (consistent +
asymptotically normal estimators are derived but require careful
implementation). Best as a Python research spike before any TS port.

---

## 7. Idea #7 — Online Rolling Controlled SMC (Particle Filter)

**Source.** Xue, Finke & Johansen (2025), *Online Rolling Controlled
Sequential Monte Carlo* (arXiv [2508.00696](https://arxiv.org/abs/2508.00696)).
Real-time particle-filter inference for general state-space HMMs with
**bounded computational cost** via rolling window.

**What this changes.** Our current HMM (`hmm.ts`) is batch-fit on every
call. For sub-daily updates (intraday Polymarket trade flow), this is
expensive. ORCSMC gives O(1) cost per new observation while maintaining
calibration on stochastic-volatility, linear-Gaussian, and neuroscience
models.

**Effort:** High; **deferred** unless we add an intraday update path. The
batch HMM works fine at daily cadence.

---

## 8. Idea #8 — Crypto Unpredictability Sanity Baseline

**Source.** Puoti, Pittorino & Roveri (2025), *Quantifying Cryptocurrency
Unpredictability* (arXiv [2502.09079](https://arxiv.org/abs/2502.09079)).
Negative-result paper: across BTC/ETH/LTC/BNB/XRP, **simpler models
(naïve random-walk) consistently outperform ML and deep-learning models**
across all forecast horizons in MAPE/RMSE.

**Why include a negative result.** Because every new shipped forecast
component should have to beat a no-change baseline. Today we don't enforce
this — it would be embarrassing to ship the next ten ideas only to find
some of them are worse than `next_close = current_close`.

**Implementation plan.**
1. New `research/spikes/crypto_baseline/naive_baseline.py` — implements
   the four baselines from §3 of the paper (Naïve, Mean, Drift, Seasonal).
2. New CI guard in `src/tools/finance/markov-backtest.integration.test.ts`
   that any forecast claim ("Markov beats X") must include a comparison
   against the matching naïve baseline on the same window.
3. Add a section to `docs/forecast-implementation-review.md` whenever a
   new model ships, listing baseline comparison numbers.

**True-improvement gate.** This *is* the gate. Every other idea on this
list must clear it before we consider the work complete.

**Effort:** Small. Pure infra. Should land first because items 1–6 will
need it.

---

## 9. Lower-priority ideas (recorded for later)

| Idea | Source | Why deferred |
|------|--------|--------------|
| Self-Calibrating Conformal Prediction (Venn-Abers + conformal) | van der Laan & Alaa 2024 (arXiv 2402.07307) | Subsumed by Idea #2; we'd add this as a v2 wrapper after the basic conformal interval lands. |
| JANET joint multi-step conformal regions | English et al. 2024 (arXiv 2407.06390) | Same family as Idea #2; revisit when we need joint coverage across the 14-day horizon (currently we treat each day's interval independently). |
| WCPS under covariate shift | Jonkers et al. 2024 (arXiv 2404.15018) | Useful when our calibration set differs systematically from prediction set (e.g. backtesting across a regime break). Subsumed by Idea #4 (drift detection). |
| HF duration forecasting via self-exciting flexible residual point process | Lee 2026 (arXiv 2604.00346) | Intraday cadence; revisit alongside Idea #7. |
| SpotV2Net — GAT for multivariate intraday spot vol | Brini & Toscano 2024 (arXiv 2401.06249) | Heavy infra; only if we go intraday. |
| GNN for multivariate realised vol with spillover | Zhang et al. 2023 (arXiv 2308.01419) | Same — intraday/spillover is a separate research axis. |
| BTC-denominated prediction markets infra | Shabashev 2025 (arXiv 2509.11990) | Out-of-scope for forecasting; relevant if we ever build a market-making or execution layer. |
| Conformal prediction sets aid human decisions | Cresswell et al. 2024 (arXiv 2401.13744) | Confirms UX value of Idea #2 outputs in the TUI; not a model change. |

---

## 10. Cross-cutting themes

1. **The Polymarket public feed is partially broken for direction-based
   signals.** This dwarfs every other finding in this round. Idea #1
   should land before any work that consumes Polymarket flow.
2. **Distribution-free guarantees are within reach.** Conformal prediction
   (Idea #2) gives us a finite-sample coverage guarantee no parametric
   model can match. This should be a permanent wrapper on every
   probabilistic output we ship.
3. **Memory matters in two more places.** Today we have memory of
   (a) returns (rolling window) and (b) Polymarket prices (decay). We do
   *not* have memory of (c) jump arrivals (Idea #3, Hawkes) or
   (d) regime breaks (Idea #4, ADWIN). Both are cheap to add.
4. **Always beat naïve.** Idea #8 is the smallest item but the most
   important guard. Land it first.

---

## 11. Suggested execution plan

**Wave 3 (recommended order):**
- **W3a — Naïve-baseline CI guard** (Idea #8) — small, lands first, becomes
  the gate for everything else.
- **W3b — On-chain Polymarket trade direction** (Idea #1) — this is the
  highest-priority bug fix in our entire pipeline; ship before any new
  flow-based feature.
- **W3c — ADWIN drift detector** (Idea #4) — small TS file + Python mirror,
  unblocks better Markov windowing.
- **W3d — Online conformal PID wrapper** (Idea #2) — pure post-processing,
  parallel-safe with W3b/W3c.

**Wave 4 (research spikes first):**
- **W4a — Hawkes intensity** (Idea #3) — Python spike; gate on
  Ljung–Box + envelope tests before TS port.
- **W4b — LASSO-PCA pooling** (Idea #5) — needs the calibration store to
  exist first.

**Wave 5 (heavier research):**
- **W5a — mfBm realised-vol** (Idea #6) — Python spike only until/unless
  it beats vector-HAR.
- **W5b — ORCSMC particle filter** (Idea #7) — only if/when we add
  intraday cadence.

---

## 12. Reference inventory (new this round)

All searched via the arXiv MCP server on 2026-04-28; not yet downloaded
locally to `references/` — recommend pulling the top 8 PDFs into the
appropriate sub-folders before implementation.

Prediction markets / microstructure
- arXiv:2604.24366 — Dubach (2026), Polymarket order-book microstructure ⭐⭐⭐

Conformal prediction
- arXiv:2307.16895 — Angelopoulos, Candès & Tibshirani (2023), Conformal PID ⭐⭐
- arXiv:2402.05203 — Yang, Candès & Lei (2024), Bellman Conformal Inference ⭐⭐
- arXiv:2402.07307 — van der Laan & Alaa (2024), Self-Calibrating Conformal
- arXiv:2407.06390 — English et al. (2024), JANET (joint multi-step)
- arXiv:2404.15018 — Jonkers et al. (2024), WCPS under covariate shift
- arXiv:2401.13744 — Cresswell et al. (2024), Conformal sets aid humans

Hawkes / point processes
- arXiv:2312.16190 — Cestari et al. (2023), Hawkes LOB crypto forecasting ⭐⭐
- arXiv:2205.06338 — Oxenhorn (2022), Multivariate Hawkes for stablecoin depeg
- arXiv:2604.00346 — Lee (2026), self-exciting flexible residual PP

Drift detection / online learning
- arXiv:2504.10229 — Yelleti (2025), ROSFD ADWIN comparison ⭐⭐

Forecast pooling
- arXiv:2207.04794 — Uniejewski & Maciejowska (2022), LASSO-PCA averaging ⭐

Realised-volatility models
- arXiv:2504.15985 — Bibinger, Yu & Zhang (2025), mfBm realised vol ⭐
- arXiv:2401.06249 — Brini & Toscano (2024), SpotV2Net GAT
- arXiv:2308.01419 — Zhang et al. (2023), GNN multivariate realised vol
- arXiv:2602.19732 — Cipollini et al. (2026), VOLARE archive (data infra)

Particle filters / SMC
- arXiv:2508.00696 — Xue, Finke & Johansen (2025), ORCSMC ⭐
- arXiv:2511.04975 — Zhumekenov et al. (2025), SMC for low-noise filtering
- arXiv:2207.09590 — Mastrototaro & Olsson (2022), ALVar variance estimator
- arXiv:2504.09875 — Amri et al. (2025), Particle HMC

Sanity baselines
- arXiv:2502.09079 — Puoti et al. (2025), crypto unpredictability ⭐ (lands first)

---

*Prepared for review. No code changes in this PR — pure planning document.*
