# Markov Chain Prediction System

Cramer-Short includes a Markov chain probability distribution model for asset price
forecasting. Instead of returning a single price target, it produces a full
probability distribution — P(price > X) at many price levels — along with
confidence intervals, action signals, and a confidence score you can use to
filter predictions.

**Source:** `src/tools/finance/markov-distribution.ts`

---

## Table of Contents

1. [Overview](#overview)
2. [How the Model Works](#how-the-model-works)
3. [Interpreting Results](#interpreting-results)
4. [Selective Prediction Strategy](#selective-prediction-strategy)
5. [Per-Asset Reliability Profiles](#per-asset-reliability-profiles)
6. [Known Limitations](#known-limitations)
7. [Backtest Methodology](#backtest-methodology)
8. [Code Examples](#code-examples)
9. [Configuration & Tuning](#configuration--tuning)

---

## Overview

The Markov distribution model answers questions like "What's the probability
SPY finishes above $600 in 14 days?" by combining three independent signals:

1. **Markov regime transitions** — classifies recent history into bull/bear/
   sideways regimes, then projects forward via matrix exponentiation.
2. **Hidden Markov Model (HMM)** — a Baum-Welch Gaussian HMM fitted on daily
   returns for an independent directional + volatility forecast.
3. **Ensemble momentum indicators** — mean-reversion z-score, SMA crossover,
   and volatility compression signals.

These are calibrated through Bayesian shrinkage toward an empirical base rate
and packaged into actionable outputs.

### When to use it

- Probability assessment for a specific price target and horizon
- Understanding the range of likely outcomes (confidence intervals)
- Generating BUY / HOLD / SELL signals grounded in a full distribution
- Filtering predictions by confidence score (selective prediction)

### Key outputs

| Output | Description |
|--------|-------------|
| `distribution` | Array of `{price, probability}` points — a survival curve |
| `actionSignal` | BUY / HOLD / SELL recommendation with expected return |
| `predictionConfidence` | 0–1 score for selective prediction filtering |
| `metadata` | Regime state, HMM info, ensemble consensus, structural-break diagnostics, experimental flags |

---

## How the Model Works

### Regime Detection

The model classifies each trading day into one of three regimes: **Bull**,
**Bear**, or **Sideways**.

Classification uses *adaptive* thresholds derived from the asset's own return
distribution — not fixed cutoffs. The threshold is set at half the median
absolute daily return, so roughly 30–40% of days fall into each of bull and
bear, with 20–30% sideways, regardless of whether the asset is SPY (daily vol
~1%) or BTC (daily vol ~4%).

```
Bull:     daily return > threshold
Bear:     daily return < −threshold
Sideways: |daily return| ≤ threshold
```

Why three states instead of five? An earlier version included high-volatility
bull/bear states, but with a 120-day walk-forward window those states had only
2–4 observations each — their transition rows were dominated by the Dirichlet
prior with zero directional information. Collapsing to three states
concentrates ~40 observations per state and yields 5–10× more reliable
transition estimates.

### Transition Matrix

Once every day is labelled with a regime, the model counts transitions (bull →
bear, bear → sideways, etc.) and builds a 3×3 stochastic matrix where entry
`P[i][j]` = probability of moving from regime `i` to regime `j` on the next
day.

Key details:

- **Dirichlet prior** for smoothing — the smoothing constant scales inversely
  with sample size (`α = max(0.01, 5/N)`). Small samples get more
  regularization; large samples let the data dominate.
- **Exponential decay** — recent transitions are weighted more heavily than
  older ones (default decay rate 0.97). This lets the matrix adapt to changing
  market conditions.
- **N-step projection** — to forecast `h` days ahead, the matrix is raised to
  the `h`-th power via repeated squaring. The resulting row gives regime
  probabilities at the horizon given the current regime.
- **Structural break detection** — the training window is split in half and
  the two sub-matrices are compared via chi-squared divergence. If a break is
  detected, the model falls back to a conservative default matrix and widens
  confidence intervals by 50%.
- **Experimental 4-state sideways split** — an optional override that
  decomposes the sideways regime into chop (high vol, mean-reverting) and coil
  (low vol, accumulating). Activated only when the resulting matrix is
  non-degenerate: all self-loop probabilities must be ≥ 1/3. If this
  ergodicity guard fails, the override is silently skipped and the standard
  3-state matrix is used.

### Ensemble Prediction

Three independent signals contribute to the final distribution:

| Signal | What it measures | Weight mechanism |
|--------|-----------------|------------------|
| **Markov chain** | Regime persistence and transition probabilities | Primary signal — regime-conditional P(up) |
| **HMM (Baum-Welch)** | Latent states from daily returns | Weighted blend (0.25–0.70 depending on data length and asset) |
| **Momentum ensemble** | Mean-reversion z-score, SMA5/SMA20 crossover, vol compression | Applied as drift adjustment when ≥2 of 3 signals agree |

The momentum ensemble includes multi-lookback confirmation (10d, 20d, 40d). If
all lookbacks agree on direction, the trend is considered robust across
timeframes, which increases the confidence score.

### Regime-Conditional Up-Rates

When computing the conditional probability of a positive return for each
regime, the model uses a **lookforward window** — it checks whether returns
*after* day `i` are positive, not including day `i` itself. This avoids a
circular leak where a bull day would always contribute a positive return to
its own label. The lookforward iterates `j = i+1` to `i+horizon` inclusive,
and the training window is sized so the last valid start index is always
included (`maxStart = n - horizon` with `<` not `<=`).

### Calibration

Raw model probabilities tend to be overconfident. The calibration step applies
**Bayesian shrinkage** to pull predictions toward a center value:

```
calibrated_P = κ × center + (1 − κ) × raw_P
```

Where:

- **κ (kappa)** is the shrinkage coefficient (0.15–0.55). Lower = trust the
  model more. Adjusted down for ensemble consensus, more data, HMM
  convergence, and trending regimes. Adjusted up for sideways regimes.
- **center** is the calibration target — not a fixed 0.5, but a blend of:
  - **Regime-conditional P(up)**: empirical frequency of positive returns
    following each regime, weighted by the n-step transition probabilities.
  - **Unconditional base rate**: simple fraction of recent up-days.
  - Blending weights depend on asset type: ETFs use 80% regime-conditional
    (their regime model is reliable), crypto uses only 40% (noisier).
- **Asset-type scaling**: ETFs get κ × 0.85 (less shrinkage), crypto gets
  κ × 1.3 (more shrinkage toward base rate).

After shrinkage, a monotonicity pass ensures P(> X) is non-increasing in X,
and a base-rate floor prevents the model from predicting "down" in strongly
bullish markets without strong evidence.

### Confidence Scoring

The prediction confidence score (0–1) combines five base factors plus a
base-rate alignment adjustment:

| Factor | Weight | How it works |
|--------|--------|--------------|
| Decisiveness | 30% | `|P(up) − 0.5| × 2` — distance from coin flip (uses raw, pre-calibration P(up)) |
| Ensemble consensus | 15% | Fraction of momentum/MR/crossover signals that agree (0–3 → 0–1) |
| HMM convergence | 10% | Binary: did Baum-Welch converge? |
| Regime stability | 15% | Consecutive days in the current regime / 20 (saturates at 20 days) |
| Multi-lookback momentum | 10% | Fraction of lookback windows (10d/20d/40d) agreeing on direction |
| Base-rate alignment | +20% / −8% | Boost when the calibrated direction agrees with the empirical base rate; mild penalty when it fights the base rate |

Additional implementation heuristics:

- **Short-horizon crypto with near-zero R²** gets a small additive confidence bump
  before multiplicative penalties.

Penalties / multipliers applied after the weighted sum:

- **Structural break** detected: multiply by 0.6 by default
- **Short-horizon crypto with ≥2 trusted anchors and non-bad R²**: use a lighter
  break penalty of 0.8 instead of 0.6
- **Experimental Phase 4 flag** (`trendPenaltyOnlyBreakConfidence`, backtests only):
  keep the structural-break penalty only for break+trending windows (`bull` /
  `bear`) and skip it for break+`sideways` windows. This is **off by default**
  and does not change the public tool behavior unless explicitly enabled in the
  backtest pipeline.
- **Crypto** asset type: multiply by 0.7, or 0.85 for the short-horizon anchored
  crypto carve-out
- **Commodity** asset type: multiply by 0.85
- **ETF** asset type: multiply by 1.1
- **High volatility** (daily vol > 2%): linear penalty ramping from 1.0 at 2%
  to 0.7 at 8%

The Phase 4 experiment is exposed in metadata as
`trendPenaltyOnlyBreakConfidenceActive`. In the real walk-forward pipeline
validation (6 tickers × 3 horizons, `warmup=120`, `stride=5`), enabling the flag
changed exactly **313 break+sideways** steps and **0 of 874 break+trending**
steps, which confirms the implementation only touched the intended contexts.

`trendPenaltyOnlyBreakConfidenceActive` is **run-level provenance**: it means the
experimental policy was enabled for that prediction run. It does **not** imply
that every flagged step skipped the break penalty; break+trending steps still
retain the default break penalty under this experiment.

---

## Interpreting Results

### Probability Distribution

The `distribution` array is a **survival function** — a set of
`{price, probability}` points where `probability` = P(asset price > this
level at the horizon).

- Sorted ascending by price, with probability non-increasing.
- Covers roughly ±3 standard deviations from the current price (20+ points).
- `P(> currentPrice) > 0.5` → the model leans bullish.
- `P(> currentPrice) < 0.5` → the model leans bearish.

**Example:** If `P(> $580) = 0.73` for SPY at 14 days, the model estimates a
73% chance SPY finishes above $580.

### Confidence Intervals

Each distribution point includes `lowerBound` and `upperBound` fields
representing the **90% confidence interval** (5th and 95th percentiles from
Monte Carlo simulation).

- Wider CIs for volatile assets (TSLA, BTC) — reflecting genuine uncertainty.
- Narrower CIs for stable assets (GLD, SPY).
- CIs are widened by 50% when a structural break is detected in the training
  window.

**Walk-forward backtest CI coverage:** 93% at the 90% level (well-calibrated —
slightly conservative).

### Action Signals

The `actionSignal` object provides a concrete BUY / HOLD / SELL
recommendation:

| Field | Description |
|-------|-------------|
| `recommendation` | `'BUY'` / `'HOLD'` / `'SELL'` |
| `expectedReturn` | Probability-weighted expected return (e.g., 0.023 = +2.3%) |
| `riskRewardRatio` | Expected upside / expected downside (> 1 favours upside) |
| `confidence` | `'HIGH'` / `'MEDIUM'` / `'LOW'` (conviction relative to threshold) |
| `actionLevels.targetPrice` | Where P(> price) ≈ 30% — profit-taking level |
| `actionLevels.stopLoss` | Where P(> price) ≈ 90% — strong support level |
| `actionLevels.medianPrice` | Where P(> price) ≈ 50% — expected median outcome |
| `actionLevels.bullCase` | 80th-percentile upside |
| `actionLevels.bearCase` | 20th-percentile downside |

**How recommendations are decided:**

- Expected return is computed by integrating over the full distribution
  (trapezoid rule on the survival curve).
- **Dynamic thresholds** scale with asset volatility and horizon:
  `threshold = scaleFactor × dailyVol × √horizon`. This avoids the problem of
  fixed thresholds producing 74% HOLD predictions for low-vol assets.
- `BUY` when expected return exceeds the buy threshold.
- `SELL` when expected return is below the negative sell threshold.
- `HOLD` otherwise.

### Prediction Confidence (0–1)

| Range | Interpretation |
|-------|---------------|
| > 0.5 | **High confidence** — signals strongly agree, decisive probability |
| 0.3–0.5 | **Medium confidence** — most signals agree, reasonable to act on |
| < 0.3 | **Low confidence** — signals disagree or model is near random |

⚠️ **Recommendation:** Only act on predictions with confidence ≥ 0.3 for
directional bets. Use confidence ≥ 0.4 for high-conviction trades.

---

## Selective Prediction Strategy

The aggregate directional accuracy of 63% includes many uncertain predictions
that dilute overall performance. By filtering on the confidence score, you
trade coverage (fewer predictions) for accuracy (higher hit rate).

### Risk-Coverage (RC) Curve

| Confidence ≥ | Accuracy | Coverage | Predictions |
|--------------|----------|----------|-------------|
| 0.0 | 63% | 100% | 484 |
| 0.2 | 65% | 54% | 262 |
| 0.3 | 65% | 26% | 124 |
| 0.4 | 80% | 5% | 25 |
| 0.5 | 93% | 3% | 15 |

### Recommended Strategies

- **Baseline filter (conf ≥ 0.3):** Use as the minimum threshold for any
  directional bet. Cuts out 74% of uncertain predictions while maintaining 65%
  accuracy on the remainder.
- **High-conviction only (conf ≥ 0.4):** 80% accuracy on 25 predictions.
  Suitable for concentrated positions where being wrong is costly.
- **Very selective (conf ≥ 0.5):** 93% accuracy but only 15 predictions over
  the full backtest period. Statistically limited (see
  [Known Limitations](#known-limitations)).

The intuition: when the model's regime detection, HMM, and momentum signals
all agree *and* the probability is far from 0.5, the prediction is
substantially more reliable than the 63% baseline.

---

## Per-Asset Reliability Profiles

The current horizon audit for the live reporting surface tested
`{5, 7, 10, 14, 20, 30, 45, 60}` using `walkForward`, `WARMUP=120`, and
`STRIDE=10`. Use the tables below to separate **confirmed canonical coverage**
from **exploratory candidates that are not yet canonical**.

### Confirmed canonical coverage

These rows are backed by the committed fixture audit and are safe for
reviewer-visible summaries.

| Asset | Good tested horizon(s) | Weak / unsupported horizon(s) | Notes |
|-------|-------------------------|-------------------------------|-------|
| **SPY** | **5d, 10d, 14d, 20d, 30d** | 45–60d not Markov-specific | Confirmed positive from the short to mid horizon band. |
| **QQQ** | **14d, 20d, 30d** | 5–10d limited; 45–60d not Markov-specific | Strongest confirmed tech/index case. |
| **GLD** | **20d, 30d** | 5–14d weaker; 45–60d not Markov-specific | Clearly strong from 20d onward. |
| **VOO** | **14d, 20d, 30d** | 5–10d still exploratory; 45–60d not Markov-specific | First-wave canonical validation confirms the mid-horizon SPY-like pattern. |
| **DIA** | **30d** | 5–20d still exploratory; 45–60d not Markov-specific | First-wave canonical validation only clears the confirmation bar at 30d. |
| **VTI** | **14d, 20d, 30d** | 5–10d still exploratory; 45–60d not Markov-specific | First-wave canonical validation confirms a broad-market mid-horizon signal. |
| **IAU** | **7d, 10d, 14d, 20d, 30d** | 5d still exploratory; 45–60d not Markov-specific | Strongest first-wave addition; especially strong at 20d and 30d. |
| **MSFT** | **5d, 14d** | 7d, 10d, 20d, 30d still exploratory; 45–60d not Markov-specific | Second-wave canonical validation only supports selective short / mid-horizon confirmation. |
| **NVDA** | **5d, 14d, 20d** | 7d, 10d, 30d still exploratory; 45–60d not Markov-specific | Second-wave canonical validation supports selective short / mid-horizon coverage. |
| **GOOGL** | **7d** | 5d, 10d, 14d, 20d, 30d still exploratory; 45–60d not Markov-specific | Second-wave canonical validation confirms only a narrow short-horizon case. |
| **AMZN** | **20d, 30d** | 5d, 7d, 10d, 14d still exploratory; 45–60d not Markov-specific | Second-wave canonical validation supports the mid-horizon band only. |
| **BTC-USD** | **None** in `{5, 7, 10, 14, 20, 30, 45, 60}` | All tested horizons | Best observed directional accuracy was **49.2% at 30d**, still too weak for a “good horizon” claim. |
| **AAPL** | **None in canonical coverage** | All tested canonical horizons | Not good enough for a canonical positive claim. |
| **TSLA** | **None in canonical coverage** | All tested canonical horizons | Not good enough for a canonical positive claim. |

### Exploratory candidates (not yet canonical)

The broader liquid-asset sweep found additional promising names, but these do
**not** yet have committed fixture coverage. They must not be cited as already
confirmed.

| Tier | Assets | Why they matter | Next step |
|------|--------|-----------------|-----------|
| **Residual first-wave exploratory horizons** | **VOO 5d/7d/10d; DIA 5d/7d/10d/14d/20d; VTI 5d/7d/10d; IAU 5d** | First-wave canonical validation improved confidence, but these specific shorter or weaker horizons remain below the confirmed-docs bar. | Keep as exploratory unless a later canonical rerun clears them decisively. |
| **Residual second-wave exploratory horizons** | **MSFT 7d/10d/20d/30d; NVDA 7d/10d/30d; GOOGL 5d/10d/14d/20d/30d; AMZN 5d/7d/10d/14d** | Second-wave canonical validation improved confidence, but these specific single-name horizons remain below the confirmed-docs bar. | Keep as exploratory unless a later canonical rerun clears them decisively. |

⚠️ Exploratory sweeps are useful for ranking what to validate next, not for
claiming confirmed Markov coverage. Keep the distinction explicit.

First-wave canonical validation confirms **VOO at 14d/20d/30d, DIA at 30d,
VTI at 14d/20d/30d, and IAU at 7d/10d/14d/20d/30d**. Shorter or weaker
first-wave results remain exploratory rather than confirmed.

Second-wave canonical validation confirms **horizon-specific single-name coverage
for MSFT (5d, 14d), NVDA (5d, 14d, 20d), GOOGL (7d), and AMZN (20d, 30d)**.
Other tested single-name horizons remain exploratory, and `45–60d` horizons
remain out of scope.

The current canonical multi-horizon backtest (`swing-trade-backtest.test.ts`) covers
6 horizons × 3 tickers with 36 signals per ticker-horizon. Key results:

### Canonical multi-horizon results (confidence ≥ 0.25)

| Ticker | Best Horizon | Sharpe | Win Rate | Directional Accuracy |
|--------|-------------|--------|----------|---------------------|
| SPY | 14d | 1.60 | 72.2% | 68.8% |
| QQQ | 20d | 1.57 | 76.5% | 73.3% |
| GLD | 30d | 2.61 | 88.9% | 88.9% |

**Aggregate (all tickers, all horizons):** Sharpe 1.34 vs 1.09 unfiltered, 71.4% win rate.

The older "Historical 14d / 30d snapshot" table (below) comes from a different
backtest run with a narrower dataset. It remains as historical context only.

### Historical 14d / 30d backtest snapshot (legacy)

| Ticker | Directional Accuracy | CI Coverage (90%) |
|--------|---------------------|--------------------|
| SPY | 76% / 72% | 81% / 83% |
| GLD | 73% / 89% | 97% / 86% |
| QQQ | 73% / 69% | 89% / 89% |
| AAPL | 54% / 67% | 92% / 94% |
| TSLA | 51% / 58% | 97% / 97% |
| BTC | 42% / 54%* | 100% / 100% |

⚠️ **BTC and TSLA show near-random or anti-correlated directional accuracy.**
The regime model can be actively misleading for highly volatile assets. Use the
confidence score to filter, or rely only on CI coverage for these tickers.

\* Historical 14d / 30d snapshot from the narrower backtest documented in this
guide. The later full horizon audit still found **BTC-USD weak at every tested
horizon**, with a best observed directional accuracy of **49.2% at 30d**.

---

## Known Limitations

⚠️ **Bullish bias.** The model was trained on 2022–2025 data, which was
predominantly bullish. The base-rate floor and calibration center both reflect
this. Performance may degrade in a sustained bear market not represented in the
training window.

⚠️ **No fundamental data.** The model uses only price history — it cannot
capture earnings surprises, news events, macro announcements, or sector
rotation. Combine with Cramer-Short's other tools (SEC filings, financial metrics,
web search) for a complete picture.

⚠️ **Sample size.** The walk-forward backtest has n = 36–60 per
ticker-horizon combination. The 95% confidence interval on the aggregate 63%
accuracy (n = 484) is approximately [59%, 67%]. Individual ticker results have
wider statistical uncertainty.

⚠️ **BTC/TSLA anti-correlation.** The model shows negative directional skill
on these assets. The regime model can be anti-informative when volatility
overwhelms regime persistence.

⚠️ **Small n at high confidence.** The 93% accuracy at confidence ≥ 0.5 is
based on only 15 predictions. This is not statistically significant — the 95%
CI on that 93% spans roughly [68%, 100%]. Treat as encouraging but
unvalidated.

⚠️ **Non-stationary markets.** Performance is expected to degrade during
regime changes not seen in training data (e.g., a sustained deflationary
crash, a market structure change, or a new asset class entering the training
universe).

⚠️ **Horizon limits.** The current canonical evidence supports the **14–30d**
band broadly for non-crypto assets, with **SPY / QQQ already strong by 14d**,
**GLD clearly strong from 20d onward**, and **IAU now confirmed from 7d
through 30d**. Short horizons are still mixed overall: **SPY**, **MSFT**, and
**NVDA** have selective 5d confirmation, **IAU** is confirmed from 7d onward,
and **GOOGL** is only confirmed at 7d. **VOO** and **VTI** become review-safe
from 14d onward, **AMZN** from 20d onward, and **DIA** only clears the
confirmation bar at 30d. **BTC-USD has no good tested horizon** in
`{5, 7, 10, 14, 20, 30, 45, 60}`; its best observed directional accuracy was
**49.2% at 30d**. At **45–60d**, average `markovWeight` is near zero, so those
outcomes should not be attributed to the Markov chain signal itself. Beyond
60d, the chain mixes toward its stationary distribution and loses directional
information.

---

## Backtest Methodology

The model was validated using a **walk-forward backtest** — the same design
used in production, with no look-ahead.

### Design

| Parameter | Value |
|-----------|-------|
| Warmup window | 120 trading days |
| Stride | 10 trading days (non-overlapping prediction windows) |
| Tickers | SPY, QQQ, GLD, AAPL, TSLA, BTC |
| Horizons | 14 days, 30 days |
| Configurations | 6 tickers × 2 horizons = 12 |
| Predictions per config | ~36–60 |
| Total predictions | 484 |

At each step, the model sees only the trailing 120 days of price history,
makes a prediction, then walks forward 10 days. No future data is used at any
point.

### Metrics

| Metric | Value | Description |
|--------|-------|-------------|
| Aggregate Directional Accuracy | 63% | % of predictions where predicted direction matches actual |
| 90% CI Coverage | 93% | % of actual prices falling within the predicted 90% CI |
| Brier Score | 0.247 | Mean squared error of P(up) vs. actual binary outcome (lower = better) |
| Selective conf ≥ 0.4 | 80% accuracy, 5% coverage | High-confidence subset |
| Selective conf ≥ 0.5 | 93% accuracy, 3% coverage | Very-high-confidence subset |

### Stressed Backtests

The model was also tested against synthetic scenarios:

- **Crash** — sudden 20% drawdown
- **V-recovery** — crash followed by sharp rebound
- **Sideways** — extended low-volatility range-bound market
- **Volatility spike** — sudden doubling of daily volatility
- **Regime flip** — abrupt transition from bull to bear (or reverse)

These stress tests validate that confidence intervals widen appropriately, the
structural break detector fires, and the model degrades gracefully rather than
producing dangerously confident wrong predictions.

---

## Code Examples

### Basic Usage

```typescript
import { computeMarkovDistribution } from './tools/finance/markov-distribution.js';

const result = await computeMarkovDistribution({
  ticker: 'SPY',
  horizon: 14,
  currentPrice: 580,
  historicalPrices: prices, // 120+ daily closes, oldest first
  polymarketMarkets: [],   // optional: Polymarket threshold anchors
});

// Action signal
console.log(result.actionSignal.recommendation); // 'BUY' | 'HOLD' | 'SELL'
console.log(result.actionSignal.expectedReturn);  // e.g., 0.023 (2.3%)
console.log(result.actionSignal.riskRewardRatio); // e.g., 1.4

// Prediction confidence (use for selective filtering)
console.log(result.predictionConfidence); // 0.0 – 1.0

// Full probability distribution (survival curve)
for (const point of result.distribution) {
  console.log(`P(SPY > $${point.price}) = ${(point.probability * 100).toFixed(1)}%`);
}

// Key price levels
const { targetPrice, stopLoss, medianPrice } = result.actionSignal.actionLevels;
console.log(`Target: $${targetPrice}, Stop: $${stopLoss}, Median: $${medianPrice}`);
```

### Selective Prediction (Confidence Filtering)

```typescript
const result = await computeMarkovDistribution({ /* ... */ });

// Only act when confidence is high enough
if (result.predictionConfidence >= 0.3) {
  console.log(`Signal: ${result.actionSignal.recommendation}`);
  console.log(`Expected return: ${(result.actionSignal.expectedReturn * 100).toFixed(2)}%`);
  console.log(`Confidence: ${result.predictionConfidence.toFixed(2)}`);
} else {
  console.log('Low confidence — abstaining from prediction.');
}
```

### Reading the Probability at a Specific Price

```typescript
import {
  computeMarkovDistribution,
  interpolateSurvival,
} from './tools/finance/markov-distribution.js';

const result = await computeMarkovDistribution({ /* ... */ });

// What's the probability SPY finishes above $600?
const pAbove600 = interpolateSurvival(result.distribution, 600);
console.log(`P(SPY > $600) = ${(pAbove600 * 100).toFixed(1)}%`);
```

### Inspecting Model Diagnostics

```typescript
const result = await computeMarkovDistribution({ /* ... */ });
const meta = result.metadata;

console.log(`Current regime: ${meta.regimeState}`);
console.log(`HMM converged: ${meta.hmm?.converged}`);
console.log(`Ensemble consensus: ${meta.ensemble.consensus}/3`);
console.log(`Structural break: ${meta.structuralBreakDetected}`);
console.log(`Trend-only break confidence active: ${meta.trendPenaltyOnlyBreakConfidenceActive}`);
console.log(`Sparse states: ${meta.sparseStates.join(', ') || 'none'}`);
console.log(`Goodness-of-fit p-value: ${meta.goodnessOfFit?.pValue.toFixed(3)}`);
console.log(`Out-of-sample R²: ${meta.outOfSampleR2?.toFixed(4)}`);
```

---

## Configuration & Tuning

### Asset Profiles

The model applies per-asset-class parameter overrides. These are selected
automatically based on the ticker symbol.

| Parameter | ETF | Equity | Crypto |
|-----------|-----|--------|--------|
| Kappa multiplier | 0.85× (less shrinkage) | 1.0× (baseline) | 1.3× (more shrinkage) |
| HMM weight multiplier | 1.1× (trust HMM more) | 0.9× | 0.5× (HMM less reliable) |
| Student-t degrees of freedom | 5 (lighter tails) | 4 | 3 (fattest tails) |
| Transition matrix decay rate | 0.97 | 0.96 | 0.94 (faster adaptation) |

The degrees-of-freedom parameter (`ν`, nu) is used in the Student-t survival
function for probability calibration and is resolved from the asset profile.
It is threaded consistently through all internal calls (calibration,
distribution interpolation, and Monte Carlo CI bounds) so the same nu value
applies throughout.

**Recognized tickers:**
- **ETFs:** SPY, QQQ, IWM, DIA, VOO, VTI, GLD, SLV, TLT, XLF, XLK, ARKK,
  and ~40 more common US ETFs.
- **Crypto:** Any ticker containing BTC, ETH, SOL, DOGE, XRP, or ending in
  `-USD` / `USDT`.
- **Equity:** Everything else (default).

### Horizon Guidance

| Horizon | Backtest Coverage | Notes |
|---------|-------------------|-------|
| 5–10 days | Mixed | Confirmed for **SPY**, **MSFT 5d**, **NVDA 5d**, **GOOGL 7d**, and much of **IAU**; still noisy or exploratory for many other assets |
| **14 days** | **Well-tested** | Earliest broad defensible horizon for **SPY / QQQ / VOO / VTI**; **IAU**, **MSFT**, and **NVDA** are also confirmed here; BTC remains weak |
| **20 days** | **Well-tested** | Start of defensible good territory for non-crypto; **GLD**, **IAU**, **VTI**, **VOO**, **NVDA**, and **AMZN** all have confirmed coverage here |
| **30 days** | **Well-tested** | Practical sweet spot for non-crypto; **DIA** only clears the confirmed bar here, **AMZN** also joins here, and BTC is still weak at **49.2%** |
| 45–60 days | Limited | Average `markovWeight` is near zero; do not market wins as Markov-specific |
| 90+ days | Not recommended | Model loses directional information |

### Default Parameters

All defaults are optimized for the walk-forward backtest and should not need
adjustment for typical use. Key internal defaults:

- **Dirichlet smoothing:** `α = max(0.01, 5/N)` — auto-tuned per window size
- **Transition decay:** 0.94–0.97 depending on asset type
- **Calibration kappa:** 0.15–0.55 (adaptive per prediction)
- **HMM:** 3-state Gaussian HMM, max 50 Baum-Welch iterations, convergence
  threshold 1e-3
- **Minimum data:** 60 returns for HMM fitting, 30 transitions for matrix
  estimation, 120+ daily prices recommended
- **Monte Carlo simulations:** 1,000 per distribution point (for CI bounds)
- **Grid:** 20+ price levels spanning approximately ±3σ from current price

### Experimental Backtest Flags

These switches are for research / backtest plumbing, not the public
`markov_distribution` tool schema. They are disabled unless a caller in the
backtest harness opts in explicitly.

#### `trendPenaltyOnlyBreakConfidence?: boolean`

- **Default (`false`)**: every structural-break window gets the usual confidence
  penalty (`×0.6`, or `×0.8` in the short-horizon anchored-crypto carve-out).
- **Experimental (`true`)**: keep that penalty only when the break window is
  still trending (`bull` / `bear`). If the break window is `sideways`, skip the
  structural-break confidence penalty entirely.
- **Metadata / reporting**: surfaced as
  `metadata.trendPenaltyOnlyBreakConfidenceActive`, propagated into
  `BacktestStep.trendPenaltyOnlyBreakConfidenceActive`, and summarized at the
  report level as run provenance.

**Phase 4 implementation verification (real end-to-end walk-forward, not just
post-hoc rescoring):**

| Slice | Baseline | `trendPenaltyOnlyBreakConfidence=true` |
|-------|----------|----------------------------------------|
| Overall RC @ conf ≥ 0.2 | 64.5% acc / 57.9% cov | **65.4% acc / 65.8% cov** |
| Overall RC @ conf ≥ 0.3 | 62.6% acc / 15.4% cov | **65.5% acc / 28.3% cov** |
| Break-context RC @ conf ≥ 0.2 | 64.9% acc / 54.2% cov | **65.9% acc / 62.9% cov** |
| Break-context RC @ conf ≥ 0.3 | 60.7% acc / 9.4% cov | **65.7% acc / 23.8% cov** |

Those results are directionally consistent with the earlier Phase 3 offline
ablation, but the feature remains **experimental and non-default** until it is
explicitly promoted.

### Polymarket Anchors

When Polymarket threshold markets are available (e.g., "Will AAPL exceed $200
by March?"), the model uses them as real-money anchors:

- Raw probabilities are corrected for **YES-bias** (multiplicative adjustment
  based on Reichenbach & Walther, 2025).
- Trust scoring based on liquidity and market age filters out thin markets.
- Optional **Kalshi cross-platform validation** flags price levels where
  Polymarket and Kalshi disagree by more than 5 percentage points.
- When anchors are available, the distribution blends Markov projections with
  anchor probabilities, weighted by the spectral gap (mixing time) of the
  transition matrix.

The model works without any Polymarket data — pass an empty array for
`polymarketMarkets` to use pure Markov + HMM + momentum signals.

---

## Day-by-Day Price Trajectory

The `trajectory` option generates a day-by-day price forecast with expected price,
90% confidence intervals, and directional probability for each day up to the horizon.

### Usage

Set `trajectory: true` when calling the tool. Optionally specify `trajectoryDays`
(1–30, defaults to the horizon value):

```
"Give me a 7-day price trajectory for AAPL"
"Show me day-by-day BTC forecast for the next 14 days"
```

### Output Format

```
═══ 7-DAY PRICE TRAJECTORY: AAPL ═══
Current: $213.00 | Regime: bull | Confidence: 0.42

Day │ Expected │     90% CI Range    │ P(up) │ Return
────┼──────────┼─────────────────────┼───────┼────────
  1 │  $213.40 │ $211.90 – $214.90   │   54% │ +0.19%
  2 │  $213.90 │ $211.10 – $216.70   │   56% │ +0.42%
  3 │  $214.30 │ $210.20 – $218.40   │   57% │ +0.61%
  4 │  $214.80 │ $209.50 – $220.10   │   58% │ +0.84%
  5 │  $215.20 │ $208.60 – $221.80   │   59% │ +1.03%
  6 │  $215.70 │ $207.80 – $223.60   │   60% │ +1.27%
  7 │  $216.10 │ $206.90 – $225.30   │   61% │ +1.46%

📈 Trend: Bullish drift, CI widens ~$2.50/day
⚠️  Point estimates are probability-weighted means, not forecasts.
    The CI range is the honest measure of uncertainty.
```

### Columns Explained

| Column | Description |
|--------|-------------|
| **Day** | Trading days from today (1 = tomorrow) |
| **Expected** | Regime-weighted expected price (mean of Monte Carlo paths) |
| **90% CI Range** | 5th–95th percentile of simulated prices at that day |
| **P(up)** | Probability price exceeds current price at this horizon |
| **Return** | Cumulative expected return from current price |

### How It Works

1. **Shared Monte Carlo paths**: 1,000 random walks are generated using Student-t
   variates (fat tails, ν determined by regime). Each path is sampled at every day.
2. **Analytical drift/vol**: At each day d, `computeHorizonDriftVol(d, ...)` provides
   the regime-weighted drift and volatility from the transition matrix. The
   volatility term includes both within-regime variance (`E[σ²]`) and the
   between-regime variance of mean returns (`Var(μ)`), so mixed-regime
   forecasts appropriately widen to reflect regime uncertainty.
3. **CI bounds**: 5th and 95th percentiles of the Monte Carlo price distribution at day d.
4. **P(up)**: From the Student-t survival function `S(currentPrice; drift_d, vol_d)`.
5. **Regime**: Most likely regime state from `P^d` (d-step transition matrix).

### Key Properties

- **CI widths monotonically increase** — uncertainty grows with time (guaranteed by
  shared random walks).
- **Compatible with calibration** — the drift-based calibration preserves the S-shape
  of the survival curve at each day.
- **Maximum 30 days** — beyond this, weekly or monthly summaries would be more useful.
  For longer horizons, use the standard single-snapshot mode.

### Interpreting the Trajectory

- **P(up) near 50%**: No strong directional signal — focus on the CI range.
- **P(up) > 60%**: Moderate bullish signal. The expected price is above current.
- **Wide CI**: High uncertainty — the range matters more than the point estimate.
- **Narrow CI**: Relatively stable asset — the expected price is more meaningful.

### Limitations

- Trajectory assumes **no regime changes** within the forecast window. For volatile
  assets (BTC, TSLA), a regime flip mid-trajectory is possible but not modeled.
- **Day 1-2 signals are very weak** — P(up) will be close to 50% for most assets.
- The trajectory is **not a prediction** — it's a probability-weighted projection.
  Always quote the CI range, not just the expected price.
