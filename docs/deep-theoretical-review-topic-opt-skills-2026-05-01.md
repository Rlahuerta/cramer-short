# Deep Theoretical Code Review — topic/opt-skills

**Date:** 2026-05-01
**Scope:** Verification of theoretical correctness for the four core implementations on
`topic/opt-skills` against the papers they cite. Each module was cross-referenced
against its claimed theoretical foundation using arXiv MCP review. When the
summary mentions adjacent backtest-helper approximations, those are called out
explicitly rather than treated as part of the four core implementation audits.

---

## 1. Conformal PID — conformal.ts

**Foundation:** Angelopoulos, Candes & Tibshirani (2023), "Conformal PID Control for
Time Series Prediction." arXiv:2307.16895.

### 1.1 PID Update Rule

The paper's PID-controller update (Eq. in §2-§3):

```
q_{t+1} = q_t + η · (K_P · e_t + K_I · I_t + K_D · (e_t − e_{t−1}))
```
where `e_t = (1 − covered_t) − α` is the miscoverage error, and `I_t` is the
integral term with optional decay γ.

Code (conformal.ts:102-110):
```typescript
const err = 1 - covered;  // 1 if missed, 0 if hit
const bias = err - this.alpha;                // e_t
this.integral = this.gamma * this.integral + bias;  // I_t = γ·I_{t-1} + e_t
const derivative = bias - this.prevBias;       // e_t − e_{t-1}
const update = learningRate * (
  this.kp * bias +
  this.ki * this.integral +
  this.kd * derivative
);
this.q = Math.max(0, this.q + update);
```

**Verdict: MATCH.** The PID update exactly implements the paper's formulation.
The `Math.max(0, ...)` guard preserves the theoretical non-negativity constraint
on the radius.

### 1.2 Integral Decay Default

The paper §3 explicitly recommends γ < 1 ("integral decay") to prevent windup
during extended high-error periods. Code (conformal.ts:82):
`this.gamma = opts.integralDecay ?? 1.0;` — default 1.0 disables anti-windup.

The implementation report's attempted change to 0.99 failed the repository's
empirical invariant (radius > 1.4). Crucially, this **invariant is not a
theoretical requirement** — the paper only requires q ≥ 0. The threshold of 1.4
is a repository-specific calibration choice.

### 1.3 AdaptiveConformalPID

The `AdaptiveConformalPID` subclass (lines 139-247) adds structural-break-aware
conformal updates with break detection, cooloff window, and inflated radius
during breaks. This is novel engineering beyond the paper's scope. No
theoretical issues with the extension itself.

### 1.4 Conformal PID + HMM Refitting: Critical Theoretical Gap

Angelopoulos et al. (2023) analyze an online conformal controller wrapped around
a forecaster whose behavior evolves over time, but the paper does not directly
analyze the repository's walk-forward pattern where the Markov/HMM forecaster is
periodically re-fit and can effectively restart after structural-break logic.
In `walk-forward.ts:265-301`, that means the conformal controller is reused
across forecast updates even though the underlying prediction rule is not fixed
in the narrow sense of the paper's setup.

The `AdaptiveConformalPID` break-mode acceleration is a pragmatic mitigation,
but the paper does not directly justify this refit/restart integration. That is
the most significant theoretical gap in the current implementation.

### 1.5 Gaps

| Gap | Severity |
|-----|----------|
| Conformal coverage theory is not directly established for the repo's refit/restart walk-forward integration | **High** — the paper does not directly analyze this setting |
| integralDecay default 1.0 disables the paper's anti-windup | Minor — documented, tradeoff accepted |
| Radius > 1.4 test invariant is empirical, not from paper | Minor — test hygiene issue |

---

## 2. CRPS and Murphy-Winkler Decomposition — metrics.ts

**Foundation:** Waghmare & Ziegel (2025), "Proper scoring rules for estimation and
forecast evaluation." arXiv:2504.01781; Bolin & Wallin (2019), "Local scale
invariance and robustness of proper scoring rules." arXiv:1912.05642.

### 2.1 CRPS Closed Form

For a normal forecast distribution N(μ, σ²) and observation y, the CRPS is:
```
CRPS(N(μ,σ²), y) = σ · [z · (2Φ(z) − 1) + 2φ(z) − 1/√π]
```
where z = (y − μ)/σ (Gneiting & Raftery 2007, eq. 17).

Code (metrics.ts:412-415):
```typescript
function normalCrps(observed: number, mean: number, sigma: number): number {
  const z = (observed - mean) / sigma;
  return sigma * (z * (2 * normalCdf(z) - 1) + 2 * normalPdf(z) - 1 / SQRT_PI);
}
```

**Verdict: MATCH.** Exact implementation of the closed-form CRPS for a normal
predictive distribution.

### 2.2 Scaled CRPS

Bolin & Wallin (2019) define scaled CRPS as CRPS divided by the forecast's local
scale, achieving local scale invariance. Code (metrics.ts:434-438) divides each
step's CRPS by `deriveIntervalScale(step)` — the interval-implied σ. **Verdict: MATCH.**

### 2.3 Murphy-Winkler Interval Score Decomposition

For a (1−α) central prediction interval [L, U] and observation y:
```
S_α(L,U; y) = (U−L) + (2/α)(L−y)·𝟙_{y<L} + (2/α)(y−U)·𝟙_{y>U}
```

Code (metrics.ts:445-489):
```typescript
const penaltyScale = 2 / alpha;                          // = 20 for 90% CI
if (step.realizedPrice < lower) {
  lowerPenaltySum += penaltyScale * (lower - step.realizedPrice);
} else if (step.realizedPrice > upper) {
  upperPenaltySum += penaltyScale * (step.realizedPrice - upper);
}
```
**Verdict: MATCH.** Exact Murphy-Winkler decomposition.

### 2.4 Distribution Approximation

CRPS uses a normal approximation derived from the interval width. The actual
forecast is a multi-component mixture (HMM states + RND anchors + trajectory
MC). **Verdict: APPROXIMATION** — loses skewness and tail information.

### 2.5 Gaps

| Gap | Severity |
|-----|----------|
| CRPS uses normal approx where true forecast is a mixture | Medium — documented |
| Interval-based σ for scaled CRPS proxy | Low — preserves ranking intent |

---

## 3. Student-t HMM Emissions — hmm.ts

**Foundation:** Geweke (1993), "Bayesian Treatment of the Independent Student-t
Linear Model." Journal of Applied Econometrics.

### 3.1 Conjugate Prior Derivation

Geweke (1993) derives the Normal-Gamma conjugate prior for Gaussian observations.
The predictive posterior is Student-t with:
```
μ_n = (κ₀μ₀ + n·x̄) / (κ₀ + n)
σ²_n = β_n · (κ_n + 1) / (α_n · κ_n)
ν_n = 2 · α_n
```
where κ_n = κ₀ + n, α_n = α₀ + n/2,
β_n = β₀ + ½SS + ½·(κ₀·n/κ_n)·(x̄ − μ₀)².

Code (hmm.ts:198-206):
```typescript
location = (priorKappa * priorMean + weightedSum) / kappaN;
scale = Math.sqrt(Math.max(betaN * (kappaN + 1) / (alphaN * kappaN), 1e-12));
degreesOfFreedom = 2 * alphaN;
```

**Verdict: MATCH.** Exact implementation of Geweke's Normal-Gamma conjugate updates.

### 3.2 Weighted Sufficient Statistics

`nEff`, `weightedSum`, and `weightedSs` use HMM posterior probabilities as soft
weights. The theoretically rigorous approach would be full Bayesian HMM (MCMC).
Using gamma pseudo-counts is a standard empirical Bayes shortcut but understates
posterior uncertainty. Geweke (1993) assumes IID observations; the HMM case is
not addressed. **Verdict: APPROXIMATION.**

### 3.3 Prior Hyperparameter Selection

`resolveStudentTPriorHyperparameters` (hmm.ts:230-265) uses excess kurtosis to
calibrate priorAlpha and sparsePriorBoost for low-nEff states. Engineering, not
from Geweke (1993). **Verdict: reasonable heuristic, not paper-grounded.**

### 3.4 Gaps

| Gap | Severity |
|-----|----------|
| Gamma weights as pseudo-counts, not full Bayesian HMM | Medium — standard shortcut |
| Excess kurtosis heuristic not paper-grounded | Low — reasonable engineering |

---

## 4. Soft Regime Weighting

**Foundation:** Blake, Gandhi & Jakkula (2025), "Improving S&P 500 Volatility
Forecasting through Regime-Switching Methods." arXiv:2510.03236.

### 4.1 What the Paper Claims

Coefficient-based soft clustering consistently outperforms hard Markov switching
for S&P 500 volatility. Method: Mood test segmentation + Bayesian GMM clusters +
XGBoost-predicted soft regime weights.

### 4.2 What the Code Implements

The code captures the paper's **soft-over-hard insight** through HMM posterior
probabilities from the forward-backward pass. Posterior entropy is used as the
control signal:

```typescript
const effectivePosteriorEntropy = Math.max(posteriorEntropy, forecastEntropy);
const softRegimeConfidenceMultiplier = Math.max(0.65, 1 - effectivePosteriorEntropy * 0.35);
const softRegimeCiScale = 1 + effectivePosteriorEntropy * 0.35;
```

When entropy is high: confidence is reduced (floor 0.65), CI is widened (up to
1.35x), and HMM weight is reduced via `hmmWeight * max(0.5, 1 - entropy * 0.4)`.

Current code goes further than a pure confidence overlay. It also blends soft
current-regime and forecast-regime mixtures directly into the forecast path:

```typescript
const softBlendedMixture = params.enableSoftRegimeWeighting === true && !rndMixture && softCurrentRegimeMixture
  ? blendRegimeMixtures(softBaseMixture, softCurrentRegimeMixture, softTransitionBlendWeight)
  : undefined;
const effectiveMixture = rndMixture ?? softBlendedMixture ?? mixture;
const forecastAdjustedStateWeights = params.enableSoftRegimeWeighting === true && softForecastRegimeMixture
  ? blendStateWeightVectors(baseForecastStateWeights, regimeMixtureToArray(softForecastRegimeMixture), softTransitionBlendWeight)
  : baseForecastStateWeights;
```

So the implemented mechanism changes not only metadata and confidence, but also
the **raw distribution path**, **forecast state weights**, and therefore
**expected return**. Existing tests verify:

- disabled parity when the flag is omitted or explicitly false,
- metadata for posterior/forecast mixtures,
- lower confidence and wider CI on choppy series,
- movement in expected return and raw survival probabilities when enabled.

**Verdict: CONCEPTUAL MATCH, DIFFERENT MECHANISM.** The repository does not
implement the paper's Mood-test + Bayesian GMM + XGBoost pipeline. Instead it
uses posterior-probability mixture blending plus entropy-based modulation —
architecturally simpler, but materially stronger than a metadata-only heuristic.

### 4.3 Gaps

| Gap | Severity |
|-----|----------|
| Magic constants (0.35, 0.65, 0.4, 0.5) are empirically chosen rather than paper-derived | Medium — pragmatic calibration |
| Does not implement the paper's Mood-test + Bayesian GMM + XGBoost regime pipeline | Low — different but still aligned with the paper's soft-weighting intuition |

---

## 5. Summary Assessment

### 5.1 Correct Implementations

| Module | Paper | Verdict |
|--------|-------|---------|
| PID update rule (conformal.ts:102-110) | Angelopoulos et al. (2307.16895) | Exact match |
| CRPS closed form (metrics.ts:412-415) | Gneiting & Raftery via Waghmare & Ziegel | Exact match |
| Scaled CRPS (metrics.ts:434-438) | Bolin & Wallin (1912.05642) | Correct local scaling rule within the interval-normal approximation |
| Murphy-Winkler decomposition (metrics.ts:445-489) | Murphy & Winkler (1984) | Exact match |
| Student-t posterior updates (hmm.ts:198-206) | Geweke (1993) | Exact local posterior-update match, with HMM state-weighting caveat handled separately |
| Forward-backward (hmm.ts:313-373) | Rabiner (1989) | Correct |

### 5.2 Approximations (Honest, Documented)

| Approximation | Limitation |
|---------------|------------|
| Normal CRPS for mixture forecast | Loses multi-modality/tail info |
| Gamma weights as pseudo-counts | Undersells joint posterior uncertainty |
| Soft-regime entropy constants | Empirically calibrated |

**Adjacent backtest-helper approximation (outside the four-core-module scope)**

| Approximation | Limitation |
|---------------|------------|
| 0.995/0.005 survival thresholds for CI extraction in `walk-forward.ts` / `replay.ts` | Coverage-tuned helper thresholds, not literal 0.5% / 99.5% forecast percentiles |

### 5.3 Theoretical Gaps

| Gap | Significance |
|-----|--------------|
| Conformal PID + HMM refitting sits outside the paper's directly analyzed forecaster setting | **Most significant.** The current integration is reasonable engineering, but not directly justified by the cited theory. |
| integralDecay = 1.0 disables paper's anti-windup | Minor, tradeoff accepted |
| Ad-hoc entropy-to-CI/confidence mapping | Medium, captures Blake et al. insight |

### 5.4 Verdict

The code correctly implements the mathematical formulas from the papers it cites.
The PID update, CRPS, Murphy-Winkler decomposition, and Student-t conjugate
updates are exact matches to their theoretical foundations.

The primary theoretical concern is the **conformal integration setting**. The
Angelopoulos et al. (2023) analysis motivates adaptive online conformal control,
but it does not directly analyze this repository's walk-forward refit/restart
pattern for the underlying Markov/HMM forecaster. The break-mode acceleration is
reasonable mitigation, but it is the repository's own engineering rather than a
paper-derived extension with explicit guarantees. The empirical backtest
provides practical validation; theory provides intuition rather than a direct
proof for this exact integration.

---

## References (Papers Reviewed)

1. Angelopoulos, A.N., Candes, E.J., & Tibshirani, R.J. (2023). "Conformal PID Control for Time Series Prediction." arXiv:2307.16895.
2. Waghmare, K. & Ziegel, J. (2025). "Proper scoring rules for estimation and forecast evaluation." arXiv:2504.01781.
3. Bolin, D. & Wallin, J. (2019). "Local scale invariance and robustness of proper scoring rules." arXiv:1912.05642.
4. Geweke, J. (1993). "Bayesian Treatment of the Independent Student-t Linear Model." Journal of Applied Econometrics.
5. Blake, A.C., Gandhi, N.A., & Jakkula, A.R. (2025). "Improving S&P 500 Volatility Forecasting through Regime-Switching Methods." arXiv:2510.03236.
6. Lee, J., Xu, C., & Xie, Y. (2024). "Kernel-based Optimally Weighted Conformal Time-Series Prediction." arXiv:2405.16828.
7. Koch, D., Jeleskovic, V., & Younas, Z.I. (2024). "Modelling and Predicting the Conditional Variance of Bitcoin Daily Returns." arXiv:2401.03393.
8. Ammann, K., Adam, T., & Koslik, J.-O. (2026). "Non-Homogeneous Markov-Switching Generalized Additive Models." arXiv:2601.03760.
9. Gneiting, T. & Raftery, A.E. (2007). "Strictly proper scoring rules, prediction, and estimation." JASA.
10. Murphy, A.H. & Winkler, R.L. (1984). "Probability forecasting in meteorology." JASA.
