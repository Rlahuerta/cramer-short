# Forecast Implementation Review Guide

**Scope:** `markov-distribution.ts`, `polymarket-forecast.ts`, `hmm.ts`, `ensemble.ts` and their Python mirrors.

**Purpose:** Code-review reference for the forecasting subsystem. Covers architecture, critical algorithms, test strategy, known issues, and TS/Python parity status.

---

## 1. Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    FORECASTING PIPELINE                      │
├─────────────────────────────────────────────────────────────┤
│  INPUT LAYER                                                 │
│  ├── Price history (Financial Datasets → Binance → Yahoo)   │
│  ├── Polymarket markets (Gamma API, tag-slug inference)      │
│  ├── News sentiment (social scrapers, Fear & Greed)          │
│  └── Fundamental targets (analyst 1y price targets)            │
├─────────────────────────────────────────────────────────────┤
│  MODEL LAYER                                                 │
│  ├── Observable Markov (bull/bear/sideways regime model)      │
│  ├── Gaussian HMM (Baum-Welch EM, volatility clustering)      │
│  ├── Polymarket signal (YES-bias correction, quality score) │
│  └── Ensemble blending (weighted signal combination)          │
├─────────────────────────────────────────────────────────────┤
│  OUTPUT LAYER                                                │
│  ├── Survival distribution: P(price > X) at many thresholds   │
│  ├── Trajectory: day-by-day expected price + CI             │
│  ├── Scenarios: Down>5%, Down3-5%, Flat, Up3-5%, Up>5%     │
│  └── Action signal: BUY/HOLD/SELL with confidence grade     │
└─────────────────────────────────────────────────────────────┘
```

### File Map

| Component | TypeScript | Python | Notebook |
|-----------|-----------|--------|----------|
| Price fetching | `api.ts`, `binance.ts` | `research/data/prices.py` | 01_Data_Pipeline |
| Observable Markov | `markov-distribution.ts:3947-4011` | `research/models/markov.py` | 02_Markov_Engine |
| Gaussian HMM | `hmm.ts` | `research/models/hmm.py` | 10_HMM_Regimes |
| Trajectory | `markov-distribution.ts` | `research/models/trajectory.py` | 08_Trajectory |
| **Jump-diffusion** | **`jump-diffusion.ts`** | **`research/models/jump_diffusion.py`** | — |
| Q→P transformation | `rnd-integration.ts` | `research/models/rnd.py` | — |
| Polymarket signal | `polymarket-forecast.ts` | `research/models/ensemble.py` | 03_Polymarket_Signals |
| Ensemble blend | `ensemble.ts` | `research/models/ensemble.py` | 04_Ensemble_Blending |
| Walk-forward backtest | `markov-backtest.integration.test.ts` | `research/backtest/walk_forward.py` | 09_Backtest |
| Metrics | — | `research/backtest/metrics.py` | 09_Backtest |

---

## 2. Observable Markov Chain (`markov.py` / `markov-distribution.ts`)

### 2.1 Regime Classification

**Algorithm:** Adaptive threshold = `0.5 × median(|returns|)`, floored at `0.001`.

```python
def classify_regime_series(returns, return_threshold_multiplier=0.5):
    median_abs = median(|returns|)
    threshold = max(0.001, 0.5 * median_abs)
    # Bull:  return > threshold
    # Bear:  return < -threshold
    # Sideways: |return| <= threshold
```

**Why this works:** With a 120-day window and daily vol ~1-4%, the median absolute return is ~0.5-2%. The multiplier yields ~30-40% bull, ~30-40% bear, ~20-30% sideways regardless of asset class.

**Code-review checkpoints:**
- [ ] Threshold is floored at `0.001` — prevents degenerate cases where all returns are classified as bull on near-zero vol assets
- [ ] Returns must be sorted for median — `np.sort(np.abs(arr))` is O(n log n), could be `np.partition` for O(n)
- [ ] The TS side uses the same formula — verify parity with `test_markov_parity.py`

### 2.2 Transition Matrix Estimation

**Algorithm:** Count transitions with exponential decay + Dirichlet smoothing.

```
α = max(0.01, 5/N)          # Dirichlet prior, auto-tuned by sample size
weight = decay_rate^age     # age = 0 for most recent transition
counts[i][j] = α + Σ weight  # for each observed i→j transition
P[i][j] = counts[i][j] / Σ_k counts[i][k]
```

**Code-review checkpoints:**
- [ ] `min_observations=30` guard — returns identity-like default matrix if insufficient data
- [ ] Default matrix: diagonal = 0.6, off-diagonal = 0.2 — matches TS `DEFAULT_TRANSITION_MATRIX`
- [ ] Exponential decay `0.97^age` means the half-life is ~23 days (`ln(0.5)/ln(0.97)`)
- [ ] The Python side normalizes with `row_sums[row_sums == 0] = 1.0` — prevents div-by-zero but silently masks data issues

### 2.3 Structural Break Detection

**Algorithm:** Split returns into first/second half. Estimate transition matrix for each. Compute Frobenius norm of the difference.

```
divergence = Σ_ij (P_first[i][j] - P_second[i][j])²
detected = divergence > 0.05
```

**Known limitation:** The threshold `0.05` is arbitrary. It was chosen empirically on a small sample. There's no statistical test (no p-value, no bootstrap).

**Code-review checkpoints:**
- [ ] `alpha=0.1` is passed to both halves — with half the data, this is `max(0.01, 5/(N/2))` = `10/N`, which is more aggressive smoothing than the full-window estimate
- [ ] The test uses only 10 min observations for each half — very noisy on short windows

---

## 3. Gaussian HMM (`hmm.py` / `hmm.ts`)

### 3.1 Core Design

The TypeScript side implements Baum-Welch EM, forward-backward, and Viterbi from scratch. The Python side wraps `hmmlearn.GaussianHMM` with `covariance_type="diag"`.

**Key difference:** TS implements the algorithm manually; Python delegates to a battle-tested library. This is intentional — `hmmlearn` handles numerical stability (scaling, log-space computations) better than a from-scratch implementation.

### 3.2 Critical Finding: Volatility Clustering, Not Regime Detection

**This is the most important thing to understand when reviewing this code.**

Gaussian HMMs maximize likelihood by fitting Gaussian distributions to observations. In financial returns, **variance differences between regimes are orders of magnitude larger than mean differences**. As a result, EM finds that one broad Gaussian explains the bulk of data more efficiently than splitting by mean.

**Empirical evidence (synthetic + real data):**

| Asset | Daily Std | Largest State | Status |
|-------|-----------|---------------|--------|
| BTC | 2.47% | 98.3% | COLLAPSED |
| SPY | 1.14% | 100.0% | COLLAPSED |
| AAPL | 1.84% | 98.9% | COLLAPSED |
| BTC (structured) | 2.65% | 99.4% | COLLAPSED |

Even when 15% "bull" days (+2%) and 10% "bear" days (-2%) are manually injected into synthetic data, the HMM still assigns 98-100% of days to a single state.

**Implication for code review:**
- [ ] The `fit_hmm_regime_model()` bridge function was **removed** because it misleadingly mapped HMM states to "bull/bear/sideways"
- [ ] The value is in **continuous forecasts** (`expected_return`, `expected_volatility`), not hard labels
- [ ] `fit_2state_return_hmm()` provides a cleaner "calm/volatile" split with explicit volatility-based labeling
- [ ] Reviewers should check that `hmm_override` blending in `compute_horizon_drift_vol` and `walk_forward` uses the continuous forecasts, not state labels

### 3.3 State Ordering

After fitting, means are sorted ascending and all parameters are permuted accordingly:
```python
order = np.argsort(means)
means = means[order]
stds = stds[order]
pi = pi[order]
A = A[np.ix_(order, order)]
```

**Code-review checkpoints:**
- [ ] Verify the permutation is applied consistently to ALL parameters (means, stds, pi, A)
- [ ] Check that `covars_` is updated before sorting — `stds = sqrt(covars)`
- [ ] The TS side does NOT sort by mean — it sorts by a different heuristic. This is a known parity gap.

### 3.4 Convergence Handling

```python
try:
    model.fit(obs)
    converged = model.monitor_.converged
except Exception:
    converged = False
```

**Code-review checkpoints:**
- [ ] `hmmlearn` can raise on singular covariance matrices — the try/except catches this but returns `converged=False` with no diagnostic info
- [ ] The TS side has its own convergence check (`log-likelihood delta < tolerance`) — verify that `model.monitor_.converged` is equivalent
- [ ] On non-convergence, the caller in `walk_forward` silently skips the HMM override — this is correct fallback behavior

---

## 4. Polymarket Forecast (`polymarket-forecast.ts` / `ensemble.py`)

### 4.1 Signal Pipeline

```
Polymarket markets → Filter by relevance → Quality score → YES-bias correction
                                                          ↓
Sentiment score ───────────────────────────────────────→ Ensemble blend
Fundamental return ─────────────────────────────────────→
Options skew ──────────────────────────────────────────→
Markov return ─────────────────────────────────────────→
```

### 4.2 YES-Bias Correction

**Research basis:** Reichenbach & Walther (2025) found systematic YES-overtrading across 124M trades.

**Implementation:**
```typescript
const YES_BIAS_MULTIPLIER = 0.965;  // Multiplicative form
// Additive form: -3.5pp when p > 0.5
```

**Code-review checkpoints:**
- [ ] The TS side uses multiplicative form (`rawProbability * 0.965`) for survival interpolation (log-spaced)
- [ ] The Python side uses additive form (`p - 0.035` when `p > 0.5`) in `adjust_yes_bias()`
- [ ] These are NOT equivalent for all p values — at p=0.5: multiplicative gives 0.4825, additive gives 0.465
- [ ] The difference is small (1.75pp at p=0.5) but reviewers should be aware

### 4.3 Quality Scoring

Each market gets a quality score (0-100) based on:
- Liquidity (`volume24h`)
- Age (`ageDays`) — older markets are more trusted
- Price spike detection (whale penalty: -50%)
- Transitory move detection (30% discount)

**Code-review checkpoints:**
- [ ] Whale threshold: `|delta| > 0.08` AND `volume24h < $100K` — the volume threshold is hardcoded
- [ ] Transitory move: original move > 10pp, reversal > 50% of original — requires both 2-4h and 24-48h snapshots
- [ ] The snapshot history is stored in `.cramer-short/polymarket-snapshots.jsonl` — this file grows unbounded (no pruning)

### 4.4 Ensemble Blending

Default weights:
```
Polymarket:   0.40 × quality_score
Sentiment:    0.20
Fundamental:  0.25
Options:      0.15
Markov:       0.20
```

**Code-review checkpoints:**
- [ ] Weights sum to 1.2, not 1.0 — this is intentional (Markov and Polymarket are the primary signals)
- [ ] Weights are renormalized after filtering out missing signals
- [ ] The quality score is applied ONLY to Polymarket — other signals get full weight regardless of reliability

---

## 5. Trajectory Computation (`trajectory.py` / `markov-distribution.ts`)

### 5.1 Monte Carlo Design

**Critical optimization:** Uses a SINGLE shared set of MC paths sampled at each day, ensuring:
1. Monotonically widening CIs
2. ~7× speedup over independent simulations per day

```python
# Shared paths
paths = np.zeros((n_samples, days))
for s in range(n_samples):
    cum_log_return = 0.0
    for d in range(days):
        z = student_t_ppf(np.random.random(), nu)
        cum_log_return += drift_1d + z * scaled_vol
        paths[s, d] = cum_log_return
```

**Code-review checkpoints:**
- [ ] `student_t_ppf` with `np.random.random()` uses inverse transform sampling — this is correct for Student-t
- [ ] `scaled_vol = mc_vol * sqrt((nu-2)/nu)` — this corrects for the variance of Student-t (var = nu/(nu-2))
- [ ] The same `paths` array is reused for all days — ensures path consistency
- [ ] Expected price uses MC median when `empirical_daily_vol` is provided, else analytical `exp(mu_n)`

### 5.2 HMM Override Blending

```python
if hmm_override:
    w = hmm_override["weight"]
    mu_n = w * (horizon * hmm_drift) + (1 - w) * mu_n
    sigma_n = w * (hmm_vol * sqrt(horizon)) + (1 - w) * sigma_n
```

**Code-review checkpoints:**
- [ ] The HMM drift is NOT annualized — it's the daily expected return from `predict()`
- [ ] `hmm_vol` is also daily — scaled by `sqrt(horizon)` to get horizon volatility
- [ ] Weight is clamped to [0, 1] by `np.clip` in the caller
- [ ] Asset profiles have different default weights: ETF=1.1, equity=0.9, crypto=0.5, commodity=0.7

---

## 6. Walk-Forward Backtest (`walk_forward.py` / `markov-backtest.integration.test.ts`)

### 6.1 Design

Slides a window over historical prices:
1. Fit regime model on `window_returns[start-warmup:start]`
2. Compute forecast for horizon days ahead
3. Record predicted vs realized outcome
4. Slide forward by `stride` days

### 6.2 HMM Integration

When `use_hmm=True`:
1. Fit 3-state HMM on each window
2. Compute `predict()` for horizon drift/vol
3. Fit 2-state vol HMM for scale factor
4. Blend via `hmm_override` into `compute_horizon_drift_vol`

**Code-review checkpoints:**
- [ ] HMM is re-fit on EVERY window — this is expensive (EM is iterative)
- [ ] On short windows (< 60 days), HMM may not converge — the fallback skips the override
- [ ] The `asset_profile` parameter selects the HMM weight multiplier

### 6.3 Metrics

| Metric | Python | TS | Notes |
|--------|--------|-----|-------|
| Brier score | `metrics.brier_score()` | — | Mean squared error of probabilistic forecasts |
| Directional accuracy | `metrics.directional_accuracy()` | — | Fraction of correct direction calls |
| CI coverage | `metrics.ci_coverage()` | — | Fraction of realized prices inside predicted CI |
| Bootstrap CI | `metrics.bootstrap_directional_ci()` | — | 1000-sample bootstrap for metric uncertainty |
| Calibration table | `metrics.calibration_table()` | — | Reliability diagram bins |

**Code-review checkpoints:**
- [ ] Brier score uses binary outcomes (1 if return > 0, else 0) — this is standard but loses magnitude information
- [ ] Bootstrap uses `np.random.choice` with replacement — standard non-parametric bootstrap
- [ ] CI coverage target is 68-95% — the 90% CI from trajectory uses 5th/95th percentiles, so ~90% target

---

## 7. Parity Test Status

### 7.1 Test Coverage

| Module | Tests | Status |
|--------|-------|--------|
| Observable Markov | 21 | PASSING |
| Ensemble | 64 | PASSING |
| Trajectory | 20 | PASSING |
| HMM | 26 | PASSING |
| Backtest | 10 | PASSING |
| **Total** | **141** | **ALL PASSING** |

### 7.2 Known Parity Gaps

These are intentional or documented differences between TS and Python:

| Aspect | TypeScript | Python | Gap |
|--------|-----------|--------|-----|
| HMM implementation | Custom forward-backward | `hmmlearn.GaussianHMM` | TS is from-scratch; Python uses library |
| HMM state ordering | By mean return | By mean return | Same after sorting |
| YES-bias correction | Multiplicative (0.965×) | Additive (-3.5pp) | Small difference (~1.75pp at p=0.5) |
| Transition matrix default | `[[0.6,0.2,0.2], ...]` | Same | Parity verified |
| Structural break threshold | `0.05` (Frobenius) | Same | Parity verified |
| Adaptive threshold | `0.5 × median(\|r\|)` | Same | Parity verified |
| Student-t nu | Asset-dependent (3-5) | Same via `ASSET_PROFILES` | Parity verified |

---

## 8. Common Code Review Issues

### 8.1 Numerical Stability

**Issue:** `hmmlearn` can produce singular covariance matrices on short or noisy data.
**Mitigation:** The `min_std=1e-4` floor in `baum_welch()` prevents zero variances.
**Review check:** Verify that `covars_` is floored before `sqrt()`.

### 8.2 Silent Failures

**Issue:** Multiple functions return default values on failure without logging.
**Examples:**
- `estimate_transition_matrix()` returns default matrix on `len(states) < min_observations`
- `fit_volatility_hmm()` returns `1.0` on non-convergence
- `walk_forward()` catches all exceptions per-step and appends to `result.errors`

**Review check:** Verify that callers check `converged` flags and `len(errors)`.

### 8.3 Assumption of Stationarity

**Issue:** The Markov model assumes regime transitions are stationary over the estimation window.
**Violation:** Markets have structural breaks (regime shifts).
**Mitigation:** `detect_structural_break()` flags non-stationarity but does NOT re-estimate.

**Review check:** Verify that structural break detection actually triggers a confidence penalty in the output.

### 8.4 Look-Ahead Bias Risk

**Issue:** The walk-forward backtest uses `prices[start+warmup:start+warmup+horizon]` as the realized outcome.
**Risk:** If the backtest is used for hyperparameter tuning, repeated evaluation on the same data creates overfitting.
**Mitigation:** The `stride` parameter (default 10) reduces overlap between windows.

**Review check:** Verify that `stride >= horizon` for independent test points.

---

## 9. Checklist for New Reviewers

Before approving changes to the forecasting subsystem, verify:

- [ ] All 141 parity tests pass (`pytest research/tests/`)
- [ ] HMM changes include a test on synthetic data with known labels
- [ ] Notebook `10_HMM_Regimes.ipynb` executes end-to-end
- [ ] `walk_forward` with `use_hmm=True` runs without errors on BTC data
- [ ] No new silent failures — all try/except blocks log or return diagnostic flags
- [ ] Docstrings explain the volatility-clustering behavior of HMMs
- [ ] Changes to transition matrix or ensemble weights are mirrored in BOTH TS and Python
- [ ] New parameters have defaults that preserve backward compatibility

---

## 10. Merton Jump-Diffusion MC Step (Phase B)

### 10.1 Overview

`src/tools/finance/jump-diffusion.ts` (mirror: `research/models/jump_diffusion.py`) adds an optional
compound Poisson jump term to the trajectory Monte Carlo, implementing the Merton (1976) model:

```
dS_t / S_t = (μ − λ·κ) dt + σ dW_t + (J − 1) dN_t
```

- `N_t` — Poisson process; per-event intensity `λ_e` (jumps / day).
- `log(J) ~ N(μ_J, σ_J²)` — log-normal jump size.
- `κ = exp(μ_J + σ_J²/2) − 1` — expected percentage jump.
- `λ·κ` — **drift compensator** that keeps `E[dS/S] = μ dt` without double-counting the
  expected jump impact.

### 10.2 `jumpSpec` Parameter Contract

`computeTrajectory` accepts an optional 12th argument `jumpSpec?: JumpEventSpec[]`.

```typescript
interface JumpEventSpec {
  id: string;           // provenance (e.g. Polymarket slug)
  dailyIntensity: number;  // λ_e — physical-measure, post Q→P transform
  meanLogJump: number;  // μ_J — usually negative (down-jump)
  stdLogJump: number;   // σ_J — uncertainty in jump size
}
```

**`hasJumps` gate invariant:**  
If `jumpSpec` is `undefined` or empty, the `hasJumps` flag is `false`. No extra RNG calls are
made; the trajectory is byte-identical to the pre-jump-diffusion path. This preserves all 453
existing `markov-distribution` trajectory test assertions.

### 10.3 Drift Compensator

`jumpDriftCompensator(events)` returns `Σ_e λ_e · (exp(μ_J,e + σ_J,e²/2) − 1)`.

This value is subtracted from the per-step drift *before* the Brownian term is added, so the
unconditional expected daily return stays constant regardless of whether a jump fires.

### 10.4 Polymarket → λ Pipeline

```
Polymarket price (Q-measure)
  ↓ transformQToP(qProb, drift, rf, σ, days)    [rnd-integration.ts]
P-measure event probability
  ↓ polymarketProbToHazard(pProb, horizon)       [jump-diffusion.ts]
Daily Poisson intensity λ_e = −ln(1 − p) / horizon
```

The Q→P step must precede the hazard conversion; reversing the order re-introduces the
systematic bearish bias that Phase A removes.

### 10.5 Default Priors (`JUMP_DEFAULTS`)

| Asset class | `meanLogJump` | `stdLogJump` | Calibration source |
|---|---|---|---|
| `etf` | −0.04 | 0.02 | SPY/QQQ 90-day tail percentile, 2020–2024 |
| `equity` | −0.05 | 0.03 | Single-stock 90-day tail percentile |
| `crypto` | −0.08 | 0.05 | BTC/ETH 90-day tail percentile |
| `commodity` | −0.05 | 0.03 | GLD/USO 90-day tail percentile |
| `geopolitics` | −0.10 | 0.06 | Theory-based (±10% shock, wide uncertainty) |

### 10.6 Metadata Fields

When `enableJumpDiffusion: true`:
- `metadata.jumpDiffusionApplied: boolean` — always set; `true` iff jump spec was active.
- `metadata.jumpDiffusion?: { compensatorPerDay, events[] }` — populated only when
  `trajectory: true` is also set; provides per-event provenance for audit.

### 10.7 Activation

Set `params.enableJumpDiffusion = true` and provide a `jumpEvents` array in
`computeMarkovDistribution`. The feature defaults to **off** to preserve backward
compatibility with all existing callers.

To surface Polymarket markets as jump events, call `extractJumpEventMarkets()` (in
`polymarket.ts`) to filter the raw market list, then `buildJumpEventSpec()` (in
`jump-diffusion.ts`) to apply the Q→P transform and hazard conversion in one step.

### 10.8 Test Coverage

| File | Count | What it tests |
|---|---|---|
| `jump-diffusion.test.ts` | 13 + 4 = 17 | Pure-math helpers + geopolitics default |
| `jump-diffusion.parity.test.ts` | 5 | TS ↔ Python numerical parity |
| `polymarket-jump-extract.test.ts` | 9 | `extractJumpEventMarkets` quality filters |
| `markov-distribution.test.ts` | 4 | `jumpDiffusionApplied` metadata flag |
| `research/tests/test_jump_diffusion.py` | 13 + 4 = 17 | Python mirror |

### 10.9 Open Tasks (before enabling in production)

- **B7** — Re-run walk-forward backtest harness with `enableJumpDiffusion: true` and one
  representative `jumpEvents` config. If avg P(up) drifts > 1 pp versus the jump-free
  baseline, recalibrate `YES_BIAS_MULTIPLIER` in `src/utils/ensemble.ts`.
- **§5.4** — `settings.json` `forecasting.enableJumpDiffusion` is now schema-validated
  but not yet wired through `computeMarkovDistribution` as a default override; callers
  must pass `params.enableJumpDiffusion` explicitly.

---

## 11. References

- `docs/markov-prediction-guide.md` — User-facing prediction guide
- `docs/markov-when-to-use.md` — When to use (and not use) the Markov tool
- `docs/polymarket-history-guide.md` — Snapshot history and whale detection
- `docs/python-research-mirror-plan.md` — Original implementation plan
- `src/tools/finance/markov-distribution.ts` — TypeScript implementation
- `src/tools/finance/hmm.ts` — TypeScript HMM
- `research/models/markov.py` — Python observable Markov
- `research/models/hmm.py` — Python HMM
- `research/models/trajectory.py` — Python trajectory
- `research/backtest/walk_forward.py` — Python backtest harness
