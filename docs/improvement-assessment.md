# Improvement Guide Assessment & Implementation Plan

**Date:** 2026-04-27
**Scope:** Critical evaluation of proposed improvements to the Dexter forecasting pipeline

---

## Summary Verdict

| Suggestion | Valid? | Priority | Risk | Action |
|-----------|--------|----------|------|--------|
| 1.1 Per-day mixture drift/vol | Yes | **P0** | Medium | Implement |
| 1.2 Correct temporal aggregation | Yes | **P0** | Medium | Implement with 1.1 |
| 1.3 Polymarket discontinuity | Already fixed | — | — | Done |
| 1.4 Undefined `p_up` | **No** | — | — | False claim |
| 2.1 Remove analytical expectation | Partially | P2 | Low | Doc only |
| 2.2 Student-t CIs in backtest | Yes | **P1** | Medium | Implement |
| 2.3 Formalize shrinkage docs | Yes | P3 | None | Doc only |
| 3.1 Exact median | **Risky** | — | High | Reject (breaks TS parity) |
| 3.2 Dynamic structural break alpha | **Risky** | — | High | Reject (breaks TS parity) |
| 3.3 Closed-form stationary dist | Yes | P2 | Low | Implement |

**Net: 4 implement, 1 already done, 2 reject (false claim / parity break), 3 docs-only**

---

## Detailed Assessment

### P0: Per-Day Mixture Drift/Vol in MC (Suggestion 1.1 + 1.2)

**Why this is the highest-value change:**

The current code computes a single 1-day mixture drift/vol and applies it statically:

```python
dv1 = compute_horizon_drift_vol(1, ...)  # Single 1-day mixture
drift_1d = dv1["mu_n"]  # Applied to ALL days
mc_vol = empirical_daily_vol if empirical_daily_vol else dv1["sigma_n"]

for d in range(days):
    cum_log_return += drift_1d + z * scaled_vol  # Same every day
```

This ignores that the regime distribution evolves. On day 1, if you start in "bull", the regime distribution is heavily weighted toward bull. By day 7, it converges toward the stationary distribution. The daily expected drift changes each day.

**The correct approach:** Use the pre-computed `regime_weights_per_day[d]` to compute per-day mixture parameters:

```python
daily_drifts = np.zeros(days)
daily_vols = np.zeros(days)

for d in range(days):
    weights = regime_weights_per_day[d]
    mu_obs = sum(weights[i] * regime_stats[state].mean_return for i, state in enumerate(REGIME_STATES))
    var_of_means = sum(weights[i] * (regime_stats[state].mean_return - mu_obs)**2 for i, state in enumerate(REGIME_STATES))
    expected_var = sum(weights[i] * regime_stats[state].std_return**2 for i, state in enumerate(REGIME_STATES))
    daily_drifts[d] = mu_obs
    daily_vols[d] = math.sqrt(expected_var + var_of_means)
```

Then in the MC loop, use `daily_drifts[d]` and `daily_vols[d]` instead of constants. This also fixes temporal aggregation (1.2) because the cumulative drift naturally becomes `sum(daily_drifts)`.

**Implementation plan:**
1. Modify `compute_trajectory` to compute `daily_drifts` and `daily_vols` arrays before the MC loop.
2. Replace the static `drift_1d` and `scaled_vol` in the MC inner loop with per-day values.
3. Update the TypeScript `computeTrajectory` identically for parity.
4. Update tests: existing `test_compute_trajectory_ci_widening` and `test_compute_trajectory_day1_near_current` may need adjustment.

**Parity note:** Both TS and Python share the same simplification, so fixing both is required.

---

### P1: Student-t CIs in Walk-Forward Backtest (Suggestion 2.2)

**Why this is valuable:**

The current backtest computes CIs using the Gaussian approximation:

```python
sigma = dv["sigma_n"]
ci_lower = current_price * (1 - 1.96 * sigma)
ci_upper = current_price * (1 + 1.96 * sigma)
```

This is a log-normal CI (`price ~ lognormal`), but the model actually uses Student-t innovations with heavy tails. A 95% Gaussian CI will be too narrow for Student-t data with $\nu = 3$ (crypto), causing systematic under-coverage.

**The correct approach:** Generate actual MC paths in the backtest (or reuse them from `compute_trajectory`) and use percentiles:

```python
traj = compute_trajectory(current_price, horizon, P, regime_stats, current_regime, n_samples=1000)
prices_at_horizon = [pt.upper_bound for pt in traj]  # or from paths directly
ci_lower = np.percentile(prices_at_horizon, 5)
ci_upper = np.percentile(prices_at_horizon, 95)
```

Alternatively, since `compute_trajectory` already generates paths, we can extract them:

```python
# After calling compute_trajectory, the paths are internally computed
# We need to either:
# a) Return paths from compute_trajectory, or
# b) Call it with sufficient n_samples and use the day=horizon point
```

**Implementation plan:**
1. Modify `compute_trajectory` to optionally return the raw `paths` array.
2. In `walk_forward`, call `compute_trajectory` with `n_samples=1000` and extract percentiles from `paths[:, -1]`.
3. Update `BacktestStep` to store `ci_method: "mc"` for tracking.
4. Update tests for `test_walk_forward_ci_coverage_reasonable`.

**Note:** This will make the backtest slower (currently it only does analytical drift/vol, no MC). The `compute_trajectory` call adds `O(1000 * 7)` per window. With ~50 windows per backtest, this is manageable (~350K operations).

---

### P2: Closed-Form Stationary Distribution (Suggestion 3.3)

**Why this is safe and clean:**

For a 2x2 right-stochastic matrix $A$, the stationary distribution has a closed form:

$$
\pi_0 = \frac{1 - A_{11}}{2 - A_{00} - A_{11}}, \quad \pi_1 = 1 - \pi_0
$$

The current power iteration:
```python
stationary = np.ones(2) / 2
for _ in range(100):
    next_stationary = stationary @ A
    if np.allclose(next_stationary, stationary, atol=1e-10):
        break
    stationary = next_stationary
```

This is unnecessary for 2x2 and may not converge if eigenvalues are close to 1. The closed form is exact and $O(1)$.

**Implementation plan:**
1. Replace the power iteration in `fit_2state_return_hmm` with the closed form.
2. Keep the power iteration as a fallback for $K > 2$ (for future generalization).
3. Update tests — they should still pass since the result is identical (just faster).

**Code:**
```python
denom = 2.0 - A[0, 0] - A[1, 1]
if abs(denom) > 1e-8:
    pi_0 = (1.0 - A[1, 1]) / denom
    stationary = np.array([pi_0, 1.0 - pi_0])
else:
    stationary = np.array([0.5, 0.5])
```

---

### P2: Remove Analytical Expectation Docs (Suggestion 2.1)

**Why this is partially valid:**

The suggestion correctly identifies that the log-normal expectation formula is incompatible with Student-t innovations. However:
- The **code** already avoids this: it uses `current_price * math.exp(mu_n)` (geometric mean), NOT the log-normal mean.
- The **documentation** was already fixed in the previous turn to remove the incorrect log-normal formula.
- The "convexity bias" argument is valid but the code already handles it correctly.

**Action:** No code changes needed. The doc fix is already done.

---

### P3: Formalize Polymarket Shrinkage Docs (Suggestion 2.3)

**Why this is low priority:**

This is a pure documentation improvement. The math is correct but the current quality score is already functional. Adding the epistemic shrinkage framing makes the documentation more rigorous but doesn't change behavior.

**Action:** Add to `docs/forecast-theoretical-foundations.md` Section 4.2.

---

## Rejected Suggestions

### 1.4 Undefined `p_up` — FALSE CLAIM

**Evidence:** `p_up` is defined on line 130 of `walk_forward.py`:
```python
p_up = sum(forecast[s] * up_rates[s] for s in ["bull", "bear", "sideways"])
```

This is used on lines 163 and 171. The claim is based on an incomplete reading of the function.

**Verdict:** Reject. No action needed.

### 3.1 Exact Median — BREAKS PARITY

**Evidence:** Both TS and Python use the same upper-middle-element approach:
- Python: `abs_returns[len(abs_returns) // 2]`
- TS: `absReturns[Math.floor(absReturns.length / 2)]`

Using `np.median` would produce different thresholds for even-length arrays, breaking parity tests.

**Verdict:** Reject unless also changed in TS.

### 3.2 Dynamic Structural Break Alpha — BREAKS PARITY

**Evidence:** Both TS and Python hardcode `alpha=0.1` in `detect_structuralBreak`/`detect_structural_break`. Changing Python to `alpha=None` would make the test behavior diverge from TS.

**Verdict:** Reject unless also changed in TS. Even then, the hardcoded approach is intentional for stability.

---

## Implementation Plan

### Phase 1: P0 — Per-Day Mixture in MC Trajectory (1.1 + 1.2)

**Files:**
- `research/models/trajectory.py` — Python implementation
- `src/tools/finance/markov-distribution.ts` — TypeScript parity
- `research/tests/test_trajectory_parity.py` — update tests

**Steps:**
1. Add `daily_drifts` and `daily_vols` computation before MC loop in `compute_trajectory`.
2. Modify MC inner loop to use `daily_drifts[d]` and `daily_vols[d]`.
3. Apply HMM override per-day (distribute the weight across days).
4. Mirror changes in TypeScript.
5. Update tests: `test_compute_trajectory_ci_widening` should still pass but values may shift slightly.

### Phase 2: P1 — Student-t CIs in Backtest (2.2)

**Files:**
- `research/models/trajectory.py` — return paths
- `research/backtest/walk_forward.py` — use MC CIs
- `research/tests/test_backtest_parity.py` — update tests

**Steps:**
1. Add optional `return_paths: bool = False` to `compute_trajectory`.
2. In `walk_forward`, call `compute_trajectory(..., return_paths=True)`.
3. Extract `paths[:, -1]` percentiles for CI bounds.
4. Update `BacktestStep` to include `ci_method`.
5. Update `test_walk_forward_ci_coverage_reasonable` — coverage should improve (closer to 90%).

### Phase 3: P2 — Stationary Distribution Optimization (3.3)

**Files:**
- `research/models/hmm.py` — replace power iteration
- `research/tests/test_hmm_parity.py` — verify tests still pass

**Steps:**
1. Replace 100-iteration loop with closed-form 2x2 solution.
2. Add fallback to power iteration if denom is near-zero.
3. Run tests — all should pass.

### Phase 4: P3 — Documentation Updates (2.3)

**Files:**
- `docs/forecast-theoretical-foundations.md`

**Steps:**
1. Add epistemic shrinkage framing to Section 4.2.
2. Update Section 3.1 to document the per-day mixture approach.

---

## Test Impact

| Change | Tests Affected | Expected Impact |
|--------|---------------|-----------------|
| P0: Per-day mixture | `test_trajectory_parity.py` (6 tests) | Values shift slightly, CIs may widen |
| P1: MC CIs in backtest | `test_backtest_parity.py` (4 tests) | CI coverage should improve toward target |
| P2: Closed-form stationary | `test_hmm_parity.py` (2 tests) | No impact (result identical) |
| P3: Doc updates | None | — |

---

## Risk Analysis

| Risk | Mitigation |
|------|-----------|
| TS/Python parity drift | Mirror every Python change in TS immediately |
| Trajectory output changes | Update expected values in tests, document shifts |
| Backtest slowdown from MC | Profile before/after; consider caching trajectories |
| Regime persistence variance | The per-day approach actually REDUCES this risk by properly accounting for time-varying drift |
