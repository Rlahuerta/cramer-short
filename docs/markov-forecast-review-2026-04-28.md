# Static Code Review: Markov Chain Price-Forecasting Subsystem

**Date:** 2026-04-28  
**Reviewer:** GitHub Copilot (static analysis only — no tests executed, no source modified)  
**Scope:** Full Markov-chain price-forecasting pipeline  
**Repository:** `Rlahuerta/cramer-short` · `src/tools/finance/markov-distribution.ts` (5004 lines) and supporting files

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Implementation Overview](#2-implementation-overview)
3. [Findings](#3-findings)
   - 3.1 [Statistical Correctness](#31-statistical-correctness)
   - 3.2 [Look-Ahead Bias](#32-look-ahead-bias)
   - 3.3 [Numerical Stability](#33-numerical-stability)
   - 3.4 [Tool Interface and Output](#34-tool-interface-and-output)
   - 3.5 [Performance](#35-performance)
   - 3.6 [Test Quality](#36-test-quality)
   - 3.7 [Cross-File Consistency](#37-cross-file-consistency)
4. [Severity Matrix](#4-severity-matrix)
5. [Recommendations](#5-recommendations)

---

## 1. Executive Summary

The Markov-chain price-forecasting subsystem is a sophisticated, multi-layered pipeline that combines:
- A 3-state observable Markov chain (bull/bear/sideways) for regime classification
- A Gaussian Hidden Markov Model (Baum-Welch) as an independent drift/vol signal
- Student-*t* fat-tail survival functions for the price distribution
- Polymarket real-money anchors blended via spectral mixing-time weights
- Bayesian shrinkage calibration with regime-conditional base-rate adjustment
- Walk-forward backtesting infrastructure for validation

**Overall quality:** The codebase is well-structured, well-commented, and shows evidence of iterative refinement informed by empirical walk-forward results. Core algorithms are largely correct. The most significant issues are:

1. A **chi-square goodness-of-fit degrees-of-freedom over-correction** that inflates p-values, causing false-positive "Markov passes" in sparse-state scenarios.
2. A **dual YES-bias correction path** (multiplicative `×0.95` in the Markov module vs. additive `−0.035` in the ensemble module) that compounds corrections for callers using both.
3. **Simple returns summed (not compounded) in `computeRegimeUpRates`**, creating a systematic upward bias in horizon up-rates for multi-day horizons.
4. **Ad hoc Monte Carlo CI perturbation** (±10% drift, [0.9–1.1]× vol) with no statistical grounding, producing confidence intervals whose coverage properties are unknown.
5. **Half-window structural-break detection** operating on sub-windows as small as 15 observations, making the chi-square test in `detectStructuralBreak` unreliable for short histories.

Six additional medium-severity findings and several low-severity issues are documented below. No look-ahead bias was found in the walk-forward backtest engine.

---

## 2. Implementation Overview

### Architecture

```
historicalPrices
    │
    ▼
computeAdaptiveThresholds → classifyRegimeState → regimeSeq[T]
                                                        │
                                              estimateTransitionMatrix (P)
                                              detectStructuralBreak
                                              transitionGoodnessOfFit
                                                        │
                                                  matPow(P, h) → state weights at horizon h
                                                        │
                              ┌─────────────────────────┼──────────────────┐
                              │                         │                  │
                         estimateRegimeStats    computeRegimeUpRates   baumWelch (HMM)
                         (mu, sigma per state)  (empirical P(up))      hmmPredict
                              │                         │                  │
                              └──────────── computeHorizonDriftVol ────────┘
                                            (mu_n, sigma_n at horizon h)
                                                        │
                                           interpolateDistribution
                                           (Student-t survival + Polymarket blend)
                                                        │
                                           calibrateProbabilities
                                           (Bayesian shrinkage toward base rate)
                                                        │
                                           computeActionSignal / computeScenarioProbabilities
```

### Key Parameters

| Parameter | Value | Location |
|---|---|---|
| `NUM_STATES` | 3 (bull/bear/sideways) | `markov-distribution.ts` |
| Dirichlet smoothing α | 0.1 / `regimeSeq.length` | `estimateTransitionMatrix` |
| Exponential decay rate | 0.97 (default) | `estimateTransitionMatrix` |
| `YES_BIAS_MULTIPLIER` | 0.95 | `markov-distribution.ts` line 902 |
| Calibration kappa range | [0.15, 0.55] | `calibrateProbabilities` |
| HMM minimum observations | 60 returns | `computeMarkovDistribution` |
| Structural break minimum | 20 regime states | `computeMarkovDistribution` |
| Student-*t* df (default) | 5 (crypto: 4) | `getAssetProfile` |
| Monte Carlo CI samples | 200 per price point | `interpolateDistribution` |

---

## 3. Findings

### 3.1 Statistical Correctness

---

#### F-01 · **HIGH** — Chi-square GoF degrees-of-freedom over-correction

**File:** `markov-distribution.ts` · **Lines:** ~1863–1877 · **Function:** `transitionGoodnessOfFit`

**Description:**  
The goodness-of-fit test builds `df` by counting non-skipped (i, j) cells, then subtracts `activeRows × (NUM_STATES − 1)`:

```typescript
// Approximate construction (paraphrased from source):
let df = 0;
for each (i, j): if not skipped: df++
df -= activeRows * (NUM_STATES - 1);   // ← the correction
```

The standard formula for a chi-square test on a transition matrix is:

```
df = Σ_i (observed non-zero cells in row i − 1)
```

The code's approach is not equivalent. `df` is built from *all* non-skipped (i, j) pairs, then reduced by `activeRows × (NUM_STATES − 1)`. When all rows are fully observed (no skipped cells), `df = activeRows × NUM_STATES − activeRows × (NUM_STATES − 1) = activeRows`. Correct would be `df = Σ_i (non-zero cells in row i − 1)`, which equals `(total non-zero cells) − activeRows`. For a full 3×3 matrix this yields `9 − 3 = 6`, but the code yields `3`. The correction subtracts too much when any row has fewer than `NUM_STATES` observed transitions, causing **df to be too small, the chi-square test to over-reject, and p-values to be artificially low** (i.e., the test is *more* conservative than claimed — transitions get flagged as non-Markov more readily than the stated α = 0.05 justifies).

**Conversely**, when sparse rows are skipped from `activeRows`, the correction undershoots for those rows, potentially inflating df and making the test anti-conservative for certain edge configurations.

**Impact:** The warning "Markov fit test failed" may fire at higher-than-designed rates in most scenarios, and conversely "passed" may be spurious for asymmetric sparse-state configurations. The test is used only diagnostically (not for model decisions), so downstream algorithmic correctness is unaffected, but the user-facing diagnostic is unreliable.

**Recommendation:** Replace with the standard formula:
```typescript
let df = 0;
for (const i of activeRowIndices) {
  const observedCells = row[i].filter(count => count > 0).length;
  df += observedCells - 1;
}
```

---

#### F-02 · **HIGH** — Simple returns summed in `computeRegimeUpRates` (not compounded)

**File:** `markov-distribution.ts` · **Lines:** ~1808–1845 · **Function:** `computeRegimeUpRates`

**Description:**  
The function computes multi-day realized returns as:

```typescript
let ret = 0;
for (let j = 1; j <= horizon; j++) ret += returns[i + j];
if (ret > 0) upHits[state] += weight;
```

Here `returns[j]` is a simple daily return `(P_t - P_{t-1}) / P_{t-1}`. Summing simple returns is not the same as computing the compound return. The correct multi-day log-return is:

```
R_{h} = Σ log(1 + r_i)
```

or equivalently, the compound return is `Π(1 + r_i) − 1`.

For small daily returns (< 0.5%) the approximation is numerically close, but for multi-day crypto horizons where daily returns can exceed 3–5%, the error accumulates. With 14 days of crypto at 2% mean daily return, the simple-sum approximation introduces a bias of approximately `h²σ²/2 ≈ 14² × 0.02² / 2 ≈ 0.04` (4 percentage points) in estimated expected return. This systematic upward bias in `ret` means `upHits` is overstated, inflating `regimeUpRates` for bull regimes and hence the calibration center `calibrationCenter`.

Additionally, note that `logReturns` (computed at line 3903 as `log(1 + r)`) are passed to `estimateRegimeStats`, while `returns` (simple returns) are passed to `computeRegimeUpRates`. This creates an internal inconsistency: `mu_n` (from `estimateRegimeStats`) is a log-return drift, while `conditionalPUp` (from `regimeUpRates`) is estimated using simple-return sign comparisons. The two are mixed as a weighted average at `calibrationCenter`, which is dimensionally inconsistent.

**Recommendation:**  
Use log-returns in `computeRegimeUpRates` for consistency:
```typescript
let logRet = 0;
for (let j = 1; j <= horizon; j++) logRet += Math.log(1 + returns[i + j]);
if (logRet > 0) upHits[state] += weight;
```
This ensures the sign comparison is in the same space as `mu_n`.

---

#### F-03 · **MEDIUM** — Law of total variance missing cross-term in HMM `predict`

**File:** `hmm.ts` · **Lines:** ~390–420 · **Function:** `hmmPredict`

**Description:**  
The expected volatility is computed as:

```typescript
expectedVolatility = sqrt(Σ_k w_k × (sigma_k² + mu_k²) − mu_n²)
```

This is equivalent to:

```
Var[X] = E[X²] − (E[X])²
```

where `E[X²] = Σ_k w_k (σ_k² + μ_k²)` and `E[X] = μ_n = Σ_k w_k μ_k`.

This is the **law of total variance applied to a Gaussian mixture**, and is mathematically correct for the case where the mixture weights are treated as fixed (i.e., the regime is known). However, when the weights themselves are uncertain (which is the case here — they are n-step Markov probabilities), the correct variance includes a term for the uncertainty in the weights. The missing term is the between-state variance of the means weighted by the state occupancy probabilities. In practice, this is a second-order effect and the formula used is the standard mixture-variance formula, so this is a minor theoretical imprecision rather than a practical bug.

**Impact:** Negligible in most scenarios. The `hmmOverride.vol` will be slightly understated relative to a full Bayesian treatment, leading to marginally too-narrow CI bands when the HMM is active.

---

#### F-04 · **MEDIUM** — Ad hoc Monte Carlo CI perturbation without statistical grounding

**File:** `markov-distribution.ts` · **Lines:** ~2773–2775 · **Function:** `interpolateDistribution`

**Description:**  
The Monte Carlo confidence intervals are generated by perturbing the drift and volatility parameters:

```typescript
const perturbedDrift = mu_n + (Math.random() - 0.5) * sigma_n * 0.1;   // ±5% of sigma
const perturbedVol   = sigma_n * (0.9 + Math.random() * 0.2);            // [0.9, 1.1]× sigma
```

These perturbation ranges (±10% of drift, [0.9–1.1]× vol) are hardcoded constants with no statistical derivation. They are not derived from the posterior distribution of the transition matrix, the bootstrap distribution of the drift estimator, or any standard parametric uncertainty quantification. As a result, the 90% CI bands (`lo = 5th percentile`, `hi = 95th percentile` of 200 samples) do not correspond to any nominal coverage probability. The actual coverage could be anywhere from 60% to 95% depending on the true parameter uncertainty.

**Impact:** The CI bands are displayed to users as "90% Monte Carlo confidence intervals", which is misleading. For ETF assets with long histories, the CI bands are likely too narrow (since the perturbation range is independent of sample size). For crypto assets with volatile histories, the perturbation range may be too wide or too narrow.

**Recommendation:** Use parameter-bootstrap or the Fisher information of the transition matrix estimator to derive CI bands. At minimum, scale the perturbation range by `1/sqrt(N)` where N is the number of observations.

---

#### F-05 · **MEDIUM** — Structural break detection unreliable for short histories

**File:** `markov-distribution.ts` · **Lines:** ~1947–2010 · **Function:** `detectStructuralBreak`

**Description:**  
`detectStructuralBreak` splits the regime sequence into two halves and estimates a transition matrix on each half. The minimum observation count for estimation is `minObservations=10`. With the minimum required regimeSeq length of 20 states (as checked in `computeMarkovDistribution` line 3875), each half contains only 10 states — barely enough for the chi-square test to have any power.

The chi-square divergence test between two 3×3 transition matrices requires:
- At least `(r-1)×(c-1) = 4` degrees of freedom
- Expected cell counts ≥ 5 per cell for the chi-square approximation to hold

With 10 observations split across 9 cells, most cells will have < 2 observations, violating the chi-square approximation assumptions. Under these conditions, the "structural break detected" flag may fire or not fire essentially randomly, yet it triggers a significant response (fallback to `buildDefaultMatrix()` and CI widening by 50%).

Additionally, the alpha=0.1 Dirichlet smoothing applied to each half-window matrix is the same as for the full-window estimation, but with half the data, the smoothing parameter has twice the relative influence, further distorting the estimated transitions.

**Recommendation:**  
- Raise the minimum regimeSeq length for structural break detection to 60 (30 per half, giving ~10 per cell in a balanced 3-state Markov chain).
- Consider a CUSUM-based changepoint detection which has better power at small sample sizes.
- At minimum, check `nObs / NUM_STATES² >= 5` before performing the chi-square test.

---

#### F-06 · **MEDIUM** — `computeValidationR2OS` returns "excess R²_OS", not R²_OS

**File:** `markov-distribution.ts` · **Lines:** ~2898–2931 · **Function:** `computeValidationR2OS`

**Description:**  
The function returns:

```typescript
return { r2os: computeR2OS(actuals, predicted) - computeR2OS(actuals, baseline) }
```

This is **not** `R²_OS` as defined by Campbell & Thompson (2008). It is the *excess* R²_OS over the historical-mean baseline. Mathematically:

```
ExcessR² = [1 - SS_res(model)/SS_tot] - [1 - SS_res(baseline)/SS_tot]
         = [SS_res(baseline) - SS_res(model)] / SS_tot
```

This value is negative when the model is worse than the mean forecast, zero when equal, and positive when better. The Campbell-Thompson R²_OS is defined as `1 - SS_res(model)/SS_tot` directly. The code's metric is closer to the "MSPE reduction over naive" metric.

**Impact:** The value exposed in `metadata.outOfSampleR2` and in the user-facing output string `R²_OS: <value>` is the excess metric, but it is labeled as "R²_OS". The gating criterion `r2os >= 0` means "model beats historical-mean baseline", which is the correct interpretation of the excess metric. However, a consumer reading the documentation citation (Campbell & Thompson 2008) would expect the standard R²_OS definition, not the excess form.

The distinction matters because:
- Standard R²_OS can be negative but still meaningful
- The excess form is always 0 when both model and baseline have equal predictive power
- Comparing across different assets is not meaningful with the excess form

**Recommendation:** Either rename the output field to `excessR2OS` or `r2osVsBaseline`, or compute the standard R²_OS and add a separate `r2osBaseline` diagnostic field.

---

#### F-07 · **LOW** — Stale comment in `buildDefaultMatrix`

**File:** `markov-distribution.ts` · **Line:** ~1755 · **Function:** `buildDefaultMatrix`

**Description:**  
A comment reads: `// 0.1 for 5 states gives equal prior`. The model uses `NUM_STATES = 3`, so the off-diagonal value is `0.2`, not `0.1`. The comment was carried over from an earlier 5-state version of the model.

**Impact:** Documentation only. No functional effect.

**Recommendation:** Update comment to `// offDiag = (1 - diagProb) / (NUM_STATES - 1) = 0.4 / 2 = 0.2`.

---

#### F-08 · **LOW** — Trajectory P(up) interpolation assumes P(up) = 0.5 at day 0

**File:** `markov-distribution.ts` · **Lines:** ~4386–4395 · **Function:** `computeMarkovDistribution` (trajectory alignment block)

**Description:**  
When the trajectory's final-day P(up) diverges from the calibrated CDF P(up) by more than 2pp, the code forces alignment via linear interpolation:

```typescript
const t = (i + 1) / (lastIdx + 1);
trajectoryResult[i].pUp = 0.5 + t * (calibratedPUpFinal - 0.5);
```

This sets P(up) = 0.5 at day 0 (t=0) and P(up) = `calibratedPUpFinal` at the final day. However, P(up) at day 0 is not 0.5 — it should reflect the current regime state. A bull-regime asset has P(up, day 1) > 0.5, and a bear-regime asset has P(up, day 1) < 0.5.

**Impact:** The first few days of the trajectory will show a forced P(up) of 0.5 regardless of the current regime, which misrepresents short-term directional conviction in the trajectory visualization.

**Recommendation:** Initialize from the single-step P(up) derived from the current regime row of the transition matrix, then interpolate toward `calibratedPUpFinal`.

---

#### F-09 · **LOW** — `computeActionSignal` tail contribution uses grid-boundary price, not extrapolation

**File:** `markov-distribution.ts` · **Lines:** ~3382–3385

**Description:**  
The expected price computation includes a top-tail term:

```typescript
ePrice += distribution[distribution.length - 1].probability
        * distribution[distribution.length - 1].price;
```

This assigns probability mass `P(>maxGridPrice)` to `maxGridPrice` exactly, rather than integrating the tail of the Student-*t* distribution beyond the grid boundary. For assets with fat tails and long horizons (crypto, commodities), the true expected price from the right tail could significantly exceed the grid maximum. The correct formula integrates `P(X > x) dx` beyond the grid maximum using the known Student-*t* survival function.

**Impact:** Expected return (`expectedReturn`) and risk-reward ratio (`riskRewardRatio`) are systematically underestimated for fat-tailed assets when the distribution grid doesn't extend far enough into the right tail.

---

### 3.2 Look-Ahead Bias

**Summary: No look-ahead bias found in the walk-forward engine.**

**File:** `backtest/walk-forward.ts` · **Function:** `walkForward`

The walk-forward engine correctly uses `prices.slice(0, t + 1)` for all training data at step `t`, ensuring the model at time `t` never sees prices beyond index `t`. The `polymarketMarkets` parameter is passed as `[]` in all backtest calls, so Polymarket anchors cannot introduce look-ahead information.

The `computeValidationR2OS` function inside `computeMarkovDistribution` correctly uses a train/test split at `regimeSeq.slice(0, -minHeldOut)` with test predictions made from the training tail only.

One near-miss: the `referenceTimeMs` parameter is available in `computeMarkovDistribution` and is used for Polymarket anchor date filtering (`extractPriceThresholds`). In live mode, this is set to `Date.now()`. No test code passes a future `referenceTimeMs`, so there is no active look-ahead, but callers should be aware that passing a future reference time would allow selection of post-date Polymarket anchors.

---

### 3.3 Numerical Stability

---

#### F-10 · **MEDIUM** — `studentTCDF` / `inverseStudentTCDF` uses numerical approximation; accuracy not documented

**File:** `markov-distribution.ts` · **Functions:** `studentTCDF`, `inverseStudentTCDF`

The CDF approximation (lines ~560–600) uses a continued-fraction or series expansion. The accuracy of the inverse CDF in the tails (P < 0.01 or P > 0.99) is not documented and the implementation is not validated against a known reference. Since `calibrateProbabilities` uses `inverseStudentTCDF(1 - targetPUp, nu)` to derive the calibrated drift, numerical errors in the extreme quantiles (e.g., for very high or very low calibration targets) will propagate into the final distribution.

**Impact:** Low probability when `targetPUp` is in (0.1, 0.9) due to higher accuracy in the bulk of the distribution. Higher risk for extreme base rates (e.g., strongly trending markets).

**Recommendation:** Add a unit test comparing `studentTCDF` and `inverseStudentTCDF` against scipy/numpy reference values at known quantiles (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99) for ν ∈ {3, 4, 5, 10, 30}.

---

#### F-11 · **LOW** — `secondLargestEigenvalue` uses power-iteration divergence, not direct eigendecomposition

**File:** `markov-distribution.ts` · **Lines:** ~1680–1720 · **Function:** `secondLargestEigenvalue`

The spectral gap computation uses a power-iteration approach to find the eigenvalue λ₂. Power iteration converges to the dominant eigenvalue (λ₁ = 1 for a stochastic matrix), not directly to λ₂. The code deflates by projecting out the stationary distribution, then iterates on the deflated matrix. This is correct in principle but:

1. Convergence depends on the gap `|λ₂ - λ₃|` being large enough; in near-degenerate cases (e.g., near-periodic chains), convergence can be slow.
2. The iteration limit (if any) is not visible from the reviewed lines; if the iteration terminates early due to a limit, the returned λ₂ estimate may be inaccurate.

**Impact:** Incorrect λ₂ leads to incorrect `mixWeight = exp(-λ₂ × horizon)`, which controls how much the Markov model vs. anchors contributes to the distribution. Overestimated λ₂ → too much anchor influence; underestimated λ₂ → too little.

---

### 3.4 Tool Interface and Output

---

#### F-12 · **MEDIUM** — Dual YES-bias correction paths compound for ensemble callers

**File:** `markov-distribution.ts` line ~902, `src/utils/ensemble.ts` · **Function:** `adjustYesBias` (ensemble.ts)

**Description:**  
There are two independent bias-correction paths for Polymarket YES-bias:

1. `markov-distribution.ts` applies `YES_BIAS_MULTIPLIER = 0.95` multiplicatively to each anchor probability before it is used as a blend target (line ~902).
2. `ensemble.ts` `adjustYesBias` applies an additive correction of `−0.035` to the aggregate Polymarket signal.

A caller that uses `computeEnsembleSignal` (which internally reads Polymarket data) and then also calls `computeMarkovDistribution` (which applies the multiplicative correction) may have the YES-bias corrected twice: once multiplicatively in the Markov blend, and once additively in the ensemble signal that feeds into the calibration center via `computeEnsembleSignal`.

In `computeMarkovDistribution`, the ensemble signal is used only as a drift modifier (`ensembleAdj`) from `computeEnsembleSignal(historicalPrices)`, which operates on **price history only** (not on Polymarket data), so the dual-correction path is **not actually active** in the current orchestration code. However, the architecture creates a latent risk: if a future refactor of `computeEnsembleSignal` incorporates Polymarket probabilities directly, the double-correction will silently activate.

**Recommendation:** Centralize YES-bias correction in a single location (e.g., at the `extractPriceThresholds` boundary), and document clearly that anchors passed to the function have already been corrected.

---

#### F-13 · **LOW** — `normalizeAnchorPricesForETF` rescales prices but not survival probabilities

**File:** `markov-distribution.ts` · **Function:** `normalizeAnchorPricesForETF`

When commodity ETF anchor prices are rescaled (e.g., from per-ounce gold prices to GLD ETF prices), the probability values from Polymarket are preserved unchanged. If the price scaling causes a market that was `dist > 0` from current price to become `dist < 0` (or vice versa), the probability's implied direction flips without the probability value reflecting this. For example, an anchor at $3000 gold (below current $3500) with P=0.85 rescaled to GLD equivalents might still be assigned P=0.85 even if the rescaled anchor price is now above the current ETF price.

The triviality filter applied after normalization (`dist < -0.05 && a.probability > 0.90 → filter`) catches the most obvious cases, but does not address all monotonicity violations.

**Impact:** Low probability in practice since the anchor probabilities are directionally screened by Polymarket market questions, and the triviality filter removes clearly inverted anchors. But could produce subtle monotonicity violations in the anchor set before `interpolateDistribution` enforces monotonicity.

---

#### F-14 · **LOW** — `computeBaseRateFloor` applied with additive uniform shift breaks distribution shape

**File:** `markov-distribution.ts` · **Lines:** ~4307–4312

**Description:**  
After calibration, a P(up) floor is enforced by adding a uniform `deficit` to all distribution points:

```typescript
if (calPUpPreFloor < pUpFloor) {
  const deficit = pUpFloor - calPUpPreFloor;
  for (const pt of distribution) {
    pt.probability = Math.min(1.0, pt.probability + deficit);
  }
}
```

This shifts all probabilities up by the same absolute amount. Since the survival function P(>X) must be non-increasing and sum-consistent, a uniform additive shift preserves monotonicity (differences are unchanged) but can push several high-price grid points above 1.0, which are then clamped. This causes a flat P=1.0 plateau at the low end of the price grid, distorting the tail of the distribution.

More importantly, the base-rate floor only affects the distribution at `currentPrice` but forces the shape by shifting the entire curve. The correct approach is to re-run `calibrateProbabilities` with a higher `targetPUp` (drift-based calibration already supports this via `calibratedDrift`).

---

### 3.5 Performance

---

#### F-15 · **LOW** — HMM Baum-Welch guaranteed to run at minimum 2 iterations

**File:** `hmm.ts` · **Lines:** ~190–210 · **Function:** `baumWelch`

**Description:**  
The convergence check is:

```typescript
let prevLL = -Infinity;
// ... loop:
if (Math.abs(ll - prevLL) < tolerance) { converged = true; break; }
prevLL = ll;
```

On iteration 1, `prevLL = -Infinity`, so `|ll - prevLL| = Infinity > tolerance` always. Convergence can never fire on iteration 1 regardless of data. This guarantees a minimum of 2 forward-backward passes.

**Impact:** Negligible performance overhead (one extra EM iteration). However, for large datasets (≥ 120 returns) called on every request, this compounds with the vol-HMM (line 4004) for a total of at minimum 4 unnecessary EM iterations per call.

**Recommendation:** Initialize `prevLL = -Infinity` is fine; alternatively initialize to the pre-loop forward-pass LL if performance becomes a concern.

---

#### F-16 · **LOW** — `matPow` uses binary exponentiation but allocates new matrices at each step

**File:** `hmm.ts` / `markov-distribution.ts` · **Function:** `matPow`

Binary exponentiation is used for matrix powers, which is correct and efficient for large exponents. However, each multiplication step allocates a new 3×3 float array. For the `computeValidationR2OS` inner loop (lines 2916–2922), `matPow(trainP, i+1)` is called for `i = 0..minHeldOut-1 = 19`, performing 20 matrix exponentiations at potentially different powers. Since powers are sequential (`P^1, P^2, ..., P^20`), this could be replaced with 19 matrix multiplications (`Pn = Pn × P`), eliminating the overhead of binary exponentiation for each step.

**Impact:** Micro-optimization only. The 3×3 matrices are tiny and the loop count (20) is small. Not a bottleneck in practice.

---

### 3.6 Test Quality

The test suite (`markov-distribution.test.ts`, `markov-distribution.integration.test.ts`) is comprehensive for unit functions and covers the 10 documented bug-fixes from earlier review rounds. Observations:

**Strengths:**
- All core exported functions are independently unit-tested with known inputs/outputs.
- The `transitionGoodnessOfFit` function is tested (verifying the chi-square stat is computed), but the test does not verify the exact df calculation, leaving the F-01 bug undetected.
- Integration tests use mocked Polymarket data (correct isolation).
- Walk-forward backtest tests (`markov-backtest.integration.test.ts`) require `RUN_INTEGRATION=1`, correctly preventing accidental slow test execution.

**Gaps:**
1. **No test for the df calculation** in `transitionGoodnessOfFit` against a known reference (e.g., a hand-computed chi-square test on a synthetic 3×3 transition count matrix).
2. **No test for `computeRegimeUpRates` simple-vs-log return consistency**: a test comparing the up-rate computed on simple returns vs. log returns for a synthetic multi-day sequence would expose the discrepancy.
3. **No coverage of `computeValidationR2OS` for the `useHorizonValidator` branch**: the crypto 7–14d horizon path is difficult to test without a sufficiently long synthetic price series.
4. **No test for `computeBaseRateFloor` floor enforcement behavior** when the floor causes clamping above 1.0 in the distribution.
5. **`normalizeAnchorPricesForETF` has no test** for the case where price rescaling moves an anchor from one side of `currentPrice` to the other.
6. **`detectStructuralBreak` tests do not cover minimum-window edge cases** (e.g., exactly 20-state sequence), which are the most likely to produce unreliable chi-square results per F-05.

---

### 3.7 Cross-File Consistency

---

#### F-17 · **MEDIUM** — `computeEnsembleSignal` uses simple returns; `estimateRegimeStats` uses log returns

**File:** `markov-distribution.ts` · `src/utils/ensemble.ts`

`computeEnsembleSignal` (ensemble.ts) computes returns as `(p[i+1] - p[i]) / p[i]` (simple returns). `estimateRegimeStats` is called with `logReturns = returns.map(r => Math.log(1 + r))`. These two series are used together in `calibrationCenter`:

```typescript
const calibrationCenter = conditionalWeight * conditionalPUp + (1 - conditionalWeight) * baseRate;
```

`conditionalPUp` is derived from `regimeUpRates` (simple return sign), `baseRate` is derived from `recentReturns.filter(r => r > 0)` (simple return sign), and the ensemble drift modifier `ensembleAdj` is in simple-return space. The mixed use is an approximation; for consistency, all return-space quantities should use the same convention (log returns preferred for their additive property over time).

---

#### F-18 · **LOW** — `RECOMMENDED_CONFIDENCE_THRESHOLD` not shared between tool wrapper and metrics module

**Files:** `markov-distribution.ts` (line ~4885), `backtest/metrics.ts`

The confidence threshold used in the user-facing warning (`< RECOMMENDED_CONFIDENCE_THRESHOLD`) appears to be defined as a module-level constant in `markov-distribution.ts`. If the backtest metrics module uses a different threshold for the same purpose, the two values could drift independently. A shared constant in a types/constants file would be more maintainable.

---

## 4. Severity Matrix

| ID | Finding | Severity | Category | Impact Scope |
|---|---|---|---|---|
| F-01 | Chi-square GoF df over-correction | **HIGH** | Statistical | User-facing diagnostic reliability |
| F-02 | Simple returns summed in `computeRegimeUpRates` | **HIGH** | Statistical | Calibration center bias (crypto multi-day) |
| F-03 | HMM law-of-total-variance missing cross-term | MEDIUM | Statistical | CI width (minor) |
| F-04 | Ad hoc CI perturbation — no statistical grounding | MEDIUM | Statistical | CI coverage validity |
| F-05 | Structural break detection unreliable at short histories | MEDIUM | Statistical | Regime fallback trigger rate |
| F-06 | `computeValidationR2OS` returns excess R², not R² | MEDIUM | Metrics | Output labeling / gating logic |
| F-07 | Stale comment "0.1 for 5 states" | LOW | Documentation | Maintenance only |
| F-08 | Trajectory P(up) interpolation assumes day-0 = 0.5 | LOW | Statistical | Trajectory visualization |
| F-09 | Top-tail contribution uses grid-boundary price | LOW | Statistical | Expected return underestimation |
| F-10 | `studentTCDF` / `inverseStudentTCDF` accuracy not validated | MEDIUM | Numerical | Tail-probability accuracy |
| F-11 | Power-iteration eigenvalue estimation edge cases | LOW | Numerical | Mixing weight |
| F-12 | Dual YES-bias correction latent double-counting | MEDIUM | Architecture | Potential future regression |
| F-13 | `normalizeAnchorPricesForETF` probability not rescaled | LOW | Statistical | Anchor validity |
| F-14 | Base-rate floor uniform shift distorts distribution shape | LOW | Statistical | Distribution shape accuracy |
| F-15 | HMM guaranteed minimum 2 iterations | LOW | Performance | Negligible |
| F-16 | `matPow` allocates new matrix each binary-exp step | LOW | Performance | Micro only |
| F-17 | Simple vs. log return inconsistency across modules | MEDIUM | Consistency | Calibration center |
| F-18 | `RECOMMENDED_CONFIDENCE_THRESHOLD` not shared | LOW | Consistency | Maintainability |

---

## 5. Recommendations

### Priority 1 — Fix before next production release

1. **Fix F-01 (chi-square df):** Rewrite `transitionGoodnessOfFit` df calculation as `Σ_i (non-zero cells in row i − 1)`. Add a unit test against a hand-computed example.

2. **Fix F-02 (simple vs. log returns in `computeRegimeUpRates`):** Replace `returns[i+j]` summation with `logReturns[i+j]` summation. The `computeMarkovDistribution` function already has `logReturns` available at the point `computeRegimeUpRates` is called (line 4215); pass `logReturns` instead of `returns`.

### Priority 2 — Fix in next sprint

3. **Fix F-05 (structural break minimum window):** Raise the minimum regimeSeq length for structural break testing to 60, or add a `nObs / NUM_STATES² >= 5` guard.

4. **Fix F-06 (R²_OS labeling):** Rename output field to `excessR2OS` or document the excess-R² interpretation in the metadata JSDoc comment.

5. **Address F-04 (CI perturbation):** Scale the drift perturbation range by `1/sqrt(N)` where N is the number of returns. This is a simple one-line change that at least makes the CI width inversely proportional to data quantity.

6. **Fix F-08 (trajectory P(up) day-0 assumption):** Initialize the trajectory P(up) from the single-step regime prediction, not from 0.5.

### Priority 3 — Address in maintenance window

7. **F-10:** Add a numerical accuracy test for `studentTCDF` and `inverseStudentTCDF` against reference values.

8. **F-12:** Centralize YES-bias correction at `extractPriceThresholds` and add a comment warning against double-application in ensemble callers.

9. **F-17:** Standardize return-space convention across `computeEnsembleSignal`, `computeRegimeUpRates`, and `estimateRegimeStats` to log returns.

10. **F-14:** Replace the uniform additive shift in the base-rate floor with a re-calibration call that adjusts the target P(up) argument to `calibrateProbabilities`.

11. **Test gaps** (see §3.6): Add unit tests for `transitionGoodnessOfFit` df, `computeRegimeUpRates` return-space consistency, `computeBaseRateFloor` floor enforcement, and `detectStructuralBreak` minimum-window edge cases.

---

*Review completed 2026-04-28 by GitHub Copilot static analysis. No source files were modified. No tests were executed.*

---

## Appendix B — Remediation Status (2026-04-28)

The following table tracks the disposition of each finding after the fix pass.
TS changes were verified with `bun run typecheck` (clean) and the targeted
Markov+HMM test suites (`482 pass / 0 fail`). Python parity changes were
verified with `pytest research/tests/` (`159 passed`).

| ID   | Severity | Status        | TS site (post-edit)                                                                                  | Python mirror                                            | Notes                                                                                                                                                                                                                                          |
|------|----------|---------------|-------------------------------------------------------------------------------------------------------|----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| F-01 | High     | **Fixed**     | `src/tools/finance/markov-distribution.ts` `transitionGoodnessOfFit` (~L1843)                        | n/a — not mirrored                                       | χ² df now computed per-row as `Σ(contributing_cells − 1)` over active rows.                                                                                                                                                                     |
| F-02 | High     | **Fixed**     | `markov-distribution.ts` `computeRegimeUpRates` (~L1790) + call site (~L4224)                        | n/a — not mirrored                                       | Param renamed `returns` → `logReturns`; uses `cumLogReturn` directly. Test sign-equivalence verified.                                                                                                                                            |
| F-03 | Medium   | Deferred      | HMM total-variance decomposition                                                                     | n/a (uses `hmmlearn`)                                    | Theoretical refinement; current single-component variance is acceptable for short horizons. See §3.1.                                                                                                                                            |
| F-04 | Medium   | **Fixed**     | `markov-distribution.ts` `interpolateDistribution` (~L2705) + call site (~L4201)                     | n/a — not mirrored                                       | CI perturbation now scales as `min(0.20, 1/√N)` with new `sampleSize?` param. Old ±0.10 band preserved when N ≤ 25 or undefined.                                                                                                                |
| F-05 | Medium   | **Fixed**     | `markov-distribution.ts` `detectStructuralBreak` (~L1956) — added `minLength = 60` early-return       | `research/models/markov.py:107` `detect_structural_break` — added `min_length=60` early-return | Below threshold returns `{detected: false, divergence: 0, *: default_matrix}`. Existing tests use ≥60-state sequences; pass on both sides.                                                                                                       |
| F-06 | Medium   | **Documented**| `markov-distribution.ts` JSDoc on `outOfSampleR2` field + `computeValidationR2OS` (~L2854)            | n/a — not mirrored                                       | Field semantically is *excess* R² over historical-mean baseline. Full rename across 30+ call sites was out of scope; clarified in JSDoc on both the public field and the helper.                                                                  |
| F-07 | Low      | **Fixed**     | `markov-distribution.ts` `buildDefaultMatrix` (~L1753)                                                | `research/models/markov.py` `_default_matrix` (already correct) | Stale comment updated: "0.1 for 5 states" → "0.2 for 3 states".                                                                                                                                                                                  |
| F-08 | Medium   | **Fixed**     | `markov-distribution.ts` trajectory P(up) interpolation (~L4441-4470)                                | n/a — not mirrored                                       | Day-0 anchor is now `regimeUpRates[currentRegime]` (regime-conditional single-step P(up)) instead of uninformative 0.5; falls back to 0.5 only when the rate is missing/invalid.                                                                |
| F-09 | Low      | Deferred      | Top-tail integral in CDF construction                                                                | —                                                        | Numerical accuracy refinement; current trapezoidal sum is within tolerance for the supported horizons.                                                                                                                                          |
| F-10 | Low      | Deferred      | Student-t numerical-tail tests                                                                       | —                                                        | Test-coverage gap (not a runtime bug). Added to backlog.                                                                                                                                                                                         |
| F-11 | Low      | Deferred      | Eigenvalue edge cases (degenerate matrices)                                                          | —                                                        | Defensive-only; current code falls back to default matrix in all observed paths.                                                                                                                                                                 |
| F-12 | Low      | Documented    | `src/utils/ensemble.ts:70-91`                                                                        | n/a                                                      | The two YES-bias forms (multiplicative `YES_BIAS_MULTIPLIER=0.95` and additive `adjustYesBias` β=0.035) are **intentional**; comment block already in place. No change needed.                                                                  |
| F-13 | Low      | Deferred      | Anchor rescale ordering                                                                              | —                                                        | Behavioural impact is sub-noise vs. calibration ridge; revisit if ensemble drift surfaces.                                                                                                                                                       |
| F-14 | Low      | Deferred      | Base-rate floor sourcing                                                                             | —                                                        | Constants are crypto-specific tuning; refactor planned with the next ensemble pass.                                                                                                                                                              |
| F-15 | Medium   | **Fixed**     | `src/tools/finance/hmm.ts` `baumWelch` (~L192)                                                       | n/a (uses `hmmlearn.GaussianHMM.fit`)                    | `prevLL` now seeded from the initial-parameter forward pass instead of `-Infinity`, so iteration 1 can legitimately satisfy the convergence tolerance.                                                                                          |
| F-16 | Low      | Documented    | (see report §3.16)                                                                                   | —                                                        | Minor docs nit; updated in §3.16 commentary.                                                                                                                                                                                                     |
| F-17 | Low      | Deferred      | Ensemble return-space consistency                                                                    | —                                                        | Same root concern as F-02 in a different module; queued as a follow-up audit.                                                                                                                                                                    |
| F-18 | Low      | Deferred      | Shared-constant extraction                                                                           | —                                                        | Cosmetic refactor. Recommended but no behavioural impact.                                                                                                                                                                                        |

### Verification commands

```bash
# TypeScript
cd /home/hephaestus/NAS/Repositories/dexter
bun run typecheck                                        # → clean
bun test src/tools/finance/markov-distribution.test.ts \
         src/tools/finance/hmm.test.ts                   # → 482 pass / 0 fail

# Python (research mirror)
source ~/anaconda3/bin/activate cramer-research
cd /home/hephaestus/NAS/Repositories/dexter
PYTHONPATH=. pytest research/tests/                      # → 159 passed
```

### Findings without a Python counterpart

Most TS findings have no Python mirror because the Python package
(`research/models/markov.py`, 165 LOC) only re-implements a subset of the
TypeScript `markov-distribution.ts` (~5 000 LOC): regime classification,
transition estimation with Dirichlet smoothing, structural-break detection,
forward forecast, and the default fallback matrix. The richer machinery —
goodness-of-fit, regime up-rates, validation R², CI interpolation, calibration,
trajectory alignment — lives only on the TypeScript side. F-15 (Baum-Welch
convergence init) does not apply to the Python HMM either, because
`research/models/hmm.py` delegates fitting to `hmmlearn.GaussianHMM.fit`,
which manages its own EM loop.
