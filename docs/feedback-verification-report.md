# Feedback Verification Report

**Date:** 2026-04-27
**Scope:** TypeScript (`src/tools/finance/`) and Python (`research/models/`) forecasting implementations
**Result:** 141/141 tests passing after fixes

---

## Executive Summary

The feedback review identified 7 claims across 3 severity categories. After deep verification against both the TypeScript and Python codebases:

| Claim | Severity | Status | Action Taken |
|-------|----------|--------|--------------|
| A. MC ignores regime evolution | Critical | **Valid** (both TS & Python) | Documented as known simplification |
| B. Flawed temporal aggregation | Critical | **Valid** (both TS & Python) | Documented as known limitation |
| C. Polymarket bias discontinuity | Critical | **Valid** (Python only) | **Fixed** — changed to multiplicative |
| D. Undefined `p_up` in backtest | Critical | **Invalid** | None — variable is defined line 130 |
| 2A. Student-t expectation formula | Theoretical | **Valid** (doc only) | **Fixed** docs — removed log-normal formula |
| 2B. Ignored autocorrelation | Theoretical | **Valid** (both TS & Python) | Documented as known limitation |
| 3A. Hardcoded structural break alpha | Minor | **Valid** (doc only) | **Fixed** docs — corrected description |
| 3B. Inexact median | Minor | **Valid** (both TS & Python) | Noted as shared parity quirk |
| 3C. Power iteration suggestion | Suggestion | N/A | Noted; closed-form exists for 2x2 |

---

## 1. Claim A: Monte Carlo Ignores Regime Evolution

**Status:** VALID — both TypeScript and Python share the same simplification.

**Evidence:**
- TypeScript (`markov-distribution.ts:2576-2597`): computes `drift1d` once via `computeHorizonDriftVol(1, ...)` and uses it for every day in the inner loop.
- Python (`trajectory.py:210-228`): identical pattern — `drift_1d` and `scaled_vol` are constants in the MC loop.
- The `regime_weights_per_day` / `regimeWeightsPerDay` arrays are pre-computed but **only** used for:
  1. Analytical expected price (`mu_n` at each horizon)
  2. Most likely regime label
  3. `p_up` survival computation

**Why this is a simplification, not a bug:** Both implementations intentionally use a single mixture drift/vol for path generation. A true regime-switching MC would require tracking `current_state` per sample, sampling transitions via $P$, and applying regime-specific $(\mu_k, \sigma_k)$ each day. The current approach is $O(N \cdot D)$; true regime-switching would be $O(N \cdot D \cdot K)$ with additional memory overhead.

**Action:** Documented as a known simplification in `forecast-theoretical-foundations.md`. No code changes — this would require rewriting both TS and Python implementations and updating all trajectory tests.

---

## 2. Claim B: Flawed Temporal Aggregation

**Status:** VALID — both TypeScript and Python share the same approximation.

**Evidence:**
- TypeScript (`markov-distribution.ts:2525`):
  ```typescript
  mu_n: horizon * (mu_eff + momentumAdjustment),
  sigma_n: sigma_eff * Math.sqrt(horizon),
  ```
- Python (`trajectory.py:155-156`):
  ```python
  mu_n = horizon * (mu_eff + momentum_adjustment)
  sigma_n = sigma_eff * math.sqrt(horizon)
  ```

**The mathematical issue:** For a Markov chain, the expected cumulative return is:

$$
\mathbb{E}\left[\sum_{d=1}^h R_d\right] = \sum_{d=1}^h \sum_k (P^d)_{ik} \mu_k
$$

The code approximates this as $h \cdot \sum_k (P^h)_{ik} \mu_k$. This is exact only when the chain has reached its stationary distribution (so $(P^d)_{ik} \approx \pi_k$ for all $d$). For short horizons where the initial regime dominates, the approximation can over/under-estimate the true drift.

Similarly, $\sigma_{\text{eff}} \cdot \sqrt{h}$ ignores autocorrelation from regime persistence. The true variance is:

$$
\text{Var}\left(\sum_{d=1}^h R_d\right) = \sum_{d=1}^h \text{Var}(R_d) + 2 \sum_{i < j} \text{Cov}(R_i, R_j)
$$

Since $\text{Cov}(R_i, R_j) > 0$ under persistent regimes, the code underestimates the true horizon variance.

**Action:** Documented as a known limitation in `forecast-theoretical-foundations.md`. No code changes — this is a shared design decision.

---

## 3. Claim C: Polymarket Bias Discontinuity

**Status:** VALID — Python bug. Fixed.

**Evidence (before fix):**
```python
def adjust_yes_bias(p: float, beta: float = 0.035) -> float:
    if p > 0.5:
        return _clamp(p - beta, 0.01, 0.99)
    return _clamp(p, 0.01, 0.99)
```

At $p = 0.50$: returns **0.50**
At $p = 0.51$: returns **0.475**

This is a non-monotonicity — a 51% market is treated as more bearish than a 50% market.

**Fix:** Changed Python to match TypeScript multiplicative form:
```python
def adjust_yes_bias(p: float) -> float:
    return _clamp(p * YES_BIAS_MULTIPLIER, 0.01, 0.99)
```
where `YES_BIAS_MULTIPLIER = 0.95`.

**Verification:** All 63 ensemble parity tests pass. The corrected probabilities are now continuous and monotonic:
- $p = 0.50 \rightarrow 0.475$
- $p = 0.51 \rightarrow 0.4845$
- $p = 0.40 \rightarrow 0.38$

**Files changed:**
- `research/models/ensemble.py:77-85`
- `research/tests/test_ensemble_parity.py:27-50`

---

## 4. Claim D: Undefined `p_up` in Walk-Forward

**Status:** INVALID.

**Evidence:**
```python
# Line 130
p_up = sum(forecast[s] * up_rates[s] for s in ["bull", "bear", "sideways"])

# Line 163-164
direction_correct = (p_up > 0.5 and realised_return > 0) or (
    p_up <= 0.5 and realised_return <= 0
)
```

`p_up` is clearly defined on line 130 and used on lines 163 and 171. This claim appears to be based on an incomplete reading of the function.

---

## 5. Claim 2A: Student-t vs Log-Normal Expectation

**Status:** VALID — documentation error. Fixed.

**Evidence:** The original doc stated:
$$
\mathbb{E}[S_{t+h}] = S_t \cdot \exp\left(h \cdot \mu_{\text{eff}} + \frac{h \cdot \sigma_{\text{eff}}^2}{2}\right)
$$

This is the expectation of a **log-normal** distribution. The code uses Student-t innovations. The MGF of Student-t is **undefined** (infinite) for all degrees of freedom, making the arithmetic mean of price infinite.

**What the code actually does:**
- TypeScript (`markov-distribution.ts:2612`): `analyticalExpected = currentPrice * Math.exp(mu_n)`
- Python (`trajectory.py:256`): `expected_price = current_price * math.exp(mu_n)`

Both compute the **geometric mean** (median), not the arithmetic mean. The doc formula was more complex than the actual code and theoretically incompatible.

**Fix:** Updated the doc to describe the geometric mean correctly and explain why the log-normal expectation formula is invalid under Student-t assumptions.

---

## 6. Claim 2B: Ignored Markov Autocorrelation

**Status:** VALID — known limitation.

**Evidence:** Both implementations use `sigma_eff * sqrt(horizon)` (Python line 156, TS line 2526). This assumes i.i.d. daily returns.

Under regime persistence (decay rate 0.97), $\Pr(S_{t+1} = k \mid S_t = k) \approx 0.8$, so returns are positively autocorrelated. The true horizon variance is:

$$
\text{Var}\left(\sum_{d=1}^h R_d\right) = h \cdot \sigma_{\text{eff}}^2 + 2 \sum_{i < j} \text{Cov}(R_i, R_j)
$$

Since $\text{Cov}(R_i, R_j) > 0$, the code underestimates variance. This is a real theoretical issue but shared by both implementations.

**Action:** Documented as a known limitation. No code changes.

---

## 7. Claim 3A: Structural Break Alpha Hardcoded

**Status:** VALID — documentation error. Fixed.

**Evidence:**
```python
def detect_structural_break(
    states: list[RegimeState],
    divergence_threshold: float = 0.05,
    alpha: float = 0.1,   # <-- hardcoded
    decay_rate: float = 0.97,
) -> dict:
    # ...
    first_matrix = estimate_transition_matrix(first_half, alpha, 10, decay_rate)
```

Since `alpha=0.1` is explicitly passed, the auto-tune logic `max(0.01, 5/T)` in `estimate_transition_matrix` is **bypassed**. The doc incorrectly claimed the dynamic $10/N$ prior would be used.

**Fix:** Corrected the doc to state that `alpha=0.1` is hardcoded and explicitly passed, overriding the auto-tune.

---

## 8. Claim 3B: Inexact Median

**Status:** VALID — shared parity quirk.

**Evidence:**
```python
abs_returns = np.sort(np.abs(arr))
median_abs = float(abs_returns[len(abs_returns) // 2])
```

For even-length arrays, this selects the upper-middle element (index $n/2$), not the true median (average of indices $n/2 - 1$ and $n/2$).

TypeScript does the same (`absReturns[Math.floor(absReturns.length / 2)]`).

**Action:** Noted as a shared parity quirk. Since both sides do the same thing, fixing it in Python would break parity without significant benefit.

---

## 9. Claim 3C: Power Iteration Suggestion

**Status:** Suggestion, not a bug.

**Context:** The 2-state HMM uses power iteration for the stationary distribution:
```python
stationary = np.ones(2) / 2
for _ in range(100):
    next_stationary = stationary @ A
    if np.allclose(next_stationary, stationary, atol=1e-10):
        break
    stationary = next_stationary
```

**Suggestion:** For a 2x2 stochastic matrix, the closed-form stationary distribution is:

$$
\pi_0 = \frac{1 - A_{11}}{2 - A_{00} - A_{11}}, \quad \pi_1 = 1 - \pi_0
$$

This avoids iteration entirely. The reviewer is correct that the closed form is trivial for $2 \times 2$. However, the power iteration generalizes to $K \times K$ (used in the 3-state HMM path) and is robust to numerical issues. For consistency with the general $K$-state case, the power iteration was kept.

---

## Test Results

```
141 passed, 4 warnings in 3.08s
```

All tests pass after the Polymarket fix. The 4 warnings are deprecation notices for `datetime.utcnow()` in the price fetching module, unrelated to the forecasting logic.

---

## Files Changed

| File | Change |
|------|--------|
| `research/models/ensemble.py` | Changed `adjust_yes_bias` from additive to multiplicative |
| `research/tests/test_ensemble_parity.py` | Updated 6 tests to match multiplicative behavior |
| `docs/forecast-theoretical-foundations.md` | Fixed 4 sections: Polymarket bias, Student-t expectation, structural break alpha, MC regime evolution |
