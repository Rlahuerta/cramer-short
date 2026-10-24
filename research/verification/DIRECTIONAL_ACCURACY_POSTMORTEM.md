# Directional Accuracy Postmortem — Markov Backtest

**Date**: 2026-06-02
**Status**: DRAFT — for review
**Question**: Why has the Markov backtest directional accuracy dropped after the recent fixes?

---

## 1. Root Cause: p_up Computation Architecture Change

The commit `0606a5c` ("fixing the markov chain implementation", Mon Jun 1) rewrote `_window_forecaster.py` and fundamentally changed how `p_up` is computed. This is not a bug — it is a **methodological pivot** from non-parametric empirical counting to parametric trajectory-based estimation.

### Before (empirical counting)

```python
# Old code in compute_window_forecast
up_rates = compute_regime_up_rates(
    regimes, log_returns, horizon=horizon, decay_rate=decay_rate,
)
forecast = compute_markov_forecast(P, current_regime, horizon)
p_up = sum(forecast[state] * up_rates[state] for state in ["bull", "bear", "sideways"])
```

**Method**: `p_up = Σ P(regime_i at horizon) × P(up | regime_i)`

Where `P(up | regime_i)` is computed empirically: for every day in history where regime = `i`, look forward `horizon` days and count what fraction had positive cumulative log returns. This is **non-parametric** and directly answers "when we're in this regime, what's the actual historical probability of being up?"

### After (trajectory-based)

```python
# New code in compute_window_forecast
traj = compute_trajectory(window_prices[-1], horizon, P, regime_stats, current_regime, ...)
horizon_point = traj[-1]
p_up = horizon_point.p_up  # from simulation.py line 150
```

Where in `simulation.py`:
```python
p_up = student_t_survival(current_price, current_price, mu_n, sigma_n, nu)
```

**Method**: `p_up = P(S > S₀ | Student-t(μ_n, σ_n, ν))` where `μ_n` and `σ_n` come from the Monte Carlo trajectory's terminal accumulated drift and volatility.

This is **parametric**: it assumes the terminal log-return distribution is Student-t with parameters estimated from the trajectory simulation.

### Why This Matters

These are fundamentally different estimators:

| Aspect | Empirical Counting (OLD) | Trajectory Survival (NEW) |
|--------|--------------------------|---------------------------|
| **Distribution assumption** | None — actual historical frequencies | Student-t with ν=5 degrees of freedom |
| **Tail behavior** | Captures actual extreme events | Controlled by ν — may not match empirical tails |
| **Skew capture** | Natural — if regime has asymmetric up/down, reflected directly | Imposed by Student-t shape around μ_n |
| **Robustness to outliers** | Outliers affect count directly (robust to magnitude) | Outliers inflate σ_n, pulling p_up toward 0.5 |
| **Data requirements** | Needs enough historical samples per regime | Needs stable μ_n, σ_n estimates from trajectory |
| **Theoretical justification** | Directly answers "P(up)" | Assumes log-return distribution form |

---

## 2. Why Empirical Counting Typically Wins for Directional Accuracy

### 2.1 Non-Parametric Advantage

Empirical counting asks a simple question: "When regime was X, what fraction of future returns were positive?" This is the frequentist probability of direction — exactly what `direction_correct` scores against. It makes no distributional assumptions.

Trajectory survival converts regime-conditioned means and standard deviations into a survival probability. This adds two layers of approximation:
1. **Moment-based approximation**: μ_n and σ_n capture only the first two moments of the regime-conditioned distribution. If the distribution is skewed (e.g., bull regimes have fat right tails), the mean/std pair loses this information.
2. **Distributional approximation**: Student-t with ν=5 is a specific parametric family. Real regime-conditioned returns may have different tail shapes.

### 2.2 Literature Support

The literature review (`references/markov-probability/DISTILLATION.md`) shows:

- **Mettle et al. (2014)**: Uses 3-state discrete Markov chain with counting-based transition estimation. Directional prediction is based on state, not parametric survival.
- **Nguyen (2018)**: HMM for stock trading — uses actual price-level predictions, not survival-converted probabilities.
- **Catello et al. (2025)**: Reports directional prediction accuracy (DPA) of 40-63% across stocks. Their method uses HMM with Gaussian Mixture emissions — the emission distribution captures the full within-regime variation, not just mean/std.
- **Kumar & Amer (2023)**: LSTM + 3-state Markov — the LSTM handles temporal patterns, the Markov chain provides structure. Neither converts mean/std to survival.

No paper in the review converts regime-conditioned moments to directional probability via a parametric survival function. The standard approach is to use the regime state itself as the signal (e.g., "if in bull regime, predict up") or to count empirical frequencies.

### 2.3 Specific Issue with Student-t Survival

The terminal `p_up = student_t_survival(S₀, S₀, μ_n, σ_n, ν)` evaluates:
```
P(S_T > S₀) = 1 - F_t(0 | μ_n, σ_n, ν) where F_t is Student-t CDF
```

For ν=5 and μ_n (the horizon drift accumulated from daily drifts):
- If μ_n ≈ 0: p_up ≈ 0.5 regardless of regime distribution shape
- If μ_n > 0: p_up > 0.5, but the magnitude depends on σ_n (wider distribution → p_up closer to 0.5)
- If the regime has strong right skew: the Student-t with only μ_n and σ_n cannot capture this

The empirical approach captures skew naturally: if bull regimes have fat right tails and thin left tails, the fraction of positive returns reflects this. The Student-t approach only sees the mean, which may be pulled right by outliers but the probability is limited by the standard deviation.

### 2.4 σ_n Inflation from Trajectory

The trajectory simulation computes:
```python
mu_n = float(np.sum(daily_drifts[:d]))   # sum of d daily drifts
sigma_n = math.sqrt(float(np.sum(daily_vols[:d] ** 2)))  # sqrt of sum of squared vols
```

`sigma_n` grows as `sqrt(horizon)` by construction (law of total variance via mixture). But the actual horizon-ahead distribution may have lower variance than this implies because of mean reversion or regime stabilization. The `sqrt(horizon)` scaling assumes independent innovations, which conflicts with the Markov model's purpose (capturing regime persistence).

The result: `sigma_n` is systematically too large for the horizon, which pushes `p_up` toward 0.5 (the uninformative prior). This **suppresses directional accuracy** by making the model less decisive.

---

## 3. Were the Phase 1-3 Fixes Harmful?

### Fixes That Could Affect Directional Accuracy

| Fix | Affects Default Path? | Impact on Directional Accuracy |
|-----|----------------------|-------------------------------|
| HMM return mismatch (Phase 1.1) | Only with `use_hmm=True` | ✅ Correct — eliminates drift bias when HMM enabled |
| GARCH+HMM preservation (Phase 1.2) | Only with GARCH+HMM both enabled | ✅ Correct — fixes volatility when both active |
| CI scaling asymmetry (Phase 1.3) | Only affects CI bounds | ⬜ No effect on p_up or direction_correct |
| Mixing weight sign (Phase 1.4) | Only affects anchor blending in interpolation | ⬜ Not called in default backtest path |
| `compute_regime_up_rates` integration (Phase 2.1) | Only with `use_empirical_up_rates=True` | ⬜ Default is False — not active |
| CI percentile indexing (Phase 3.4) | Changes percentile indices | ⬜ Only affects CI bounds, not p_up |
| Entropy z-score std (Phase 3.2) | Changes entropy_ci_scale | ⬜ Only affects CI modulation, not p_up |
| Transition matrix validation (Phase 2.2) | Adds assertions only | ⬜ No output change |
| Monotonic CI enforcement (Phase 2.5) | Only affects CI bounds | ⬜ No p_up change |

**Verdict**: None of the Phase 1-3 fixes change the default (no HMM, no GARCH, no empirical up-rates) backtest path's `p_up` or directional accuracy. The accuracy change is entirely due to the architectural change in commit `0606a5c`.

---

## 4. Literature Benchmarks for Directional Accuracy

The literature sets expectations:

| Source | Model | Directional Accuracy |
|--------|-------|---------------------|
| Catello et al. (2025) | HMM + GMM | 40–63% across stocks |
| Nguyen (2018) | HMM (4-state) | Outperforms historical mean |
| Bulla et al. | Markov switching | ~41% after costs |
| General consensus | Any Markov model | **50–55% is normal; 60%+ is rare** |

**Key insight**: Stock returns are nearly unpredictable in direction (random walk). A Markov model that achieves 52-55% directional accuracy is performing well. If the old code was getting >55% and the new code is getting <52%, the drop is real and the old approach was empirically better.

---

## 5. Diagnostic Plan (Before Any Code Changes)

### 5.1 Confirm the Drop Exists

Run the same backtest with the old code (checkout `0606a5c~1`) and new code (current HEAD):
```bash
# Old code
git stash && git checkout 0606a5c~1
python -m research.backtest.markov_tool_backtest --ticker BTC --horizon 7 --from 2024-01-01

# New code
git checkout main  # or current branch
python -m research.backtest.markov_tool_backtest --ticker BTC --horizon 7 --from 2024-01-01
```

Compare `direction_correct` rate between both runs.

### 5.2 Isolate the p_up Change

Test the same backtest with `use_empirical_up_rates=True` (restores the old methodology):
```bash
# This uses the Phase 2.1 integration to restore empirical p_up
python -m research.backtest.markov_tool_backtest --ticker BTC --horizon 7 --from 2024-01-01 --use-empirical-up-rates
```

If directional accuracy recovers with this flag, the root cause is confirmed.

### 5.3 Check p_up Distribution Shape

Add a diagnostic to log the distribution of `p_up` values:
- Old approach: should show values away from 0.5 (more decisive)
- New approach: may cluster closer to 0.5 (underconfident)
- If the new `p_up` values are more tightly clustered around 0.5, the model is underconfident

### 5.4 Check σ_n Horizon Scaling

Verify whether `sigma_n` is growing faster than empirical horizon volatility:
- Compute the empirical standard deviation of `horizon`-period returns
- Compare against the trajectory's `sigma_n`
- If `sigma_n` is systematically larger, this inflates variance and pushes `p_up` → 0.5

---

## 6. Improvement Options (for Review)

### Option A: Restore Empirical Counting as Default

Revert `p_up` to use `compute_regime_up_rates` by default while keeping the trajectory for `predicted_return` and CI bounds.

**Pros**: Restores the empirically better directional accuracy; maintains TS parity; well-supported by literature.
**Cons**: Loses the coherence argument (p_up from same distribution as CI); requires running both trajectories and empirical up-rates.
**Effort**: ~5 lines (change default flag).

### Option B: Hybrid Approach

Compute both and use a quality-weighted blend:
```python
empirical_p_up = sum(forecast[state] * up_rates[state] for state in REGIME_STATES)
trajectory_p_up = horizon_point.p_up
p_up = w * empirical_p_up + (1-w) * trajectory_p_up
```

Where `w` depends on data sufficiency (more data → higher empirical weight).

**Pros**: Gets best of both; handles sparse data gracefully.
**Cons**: Adds complexity; another parameter to tune.
**Effort**: ~15 lines.

### Option C: Calibrate Trajectory p_up

Instead of raw `student_t_survival`, calibrate the trajectory `p_up` using Platt scaling or isotonic regression on historical out-of-sample data. This would adjust the parametric estimate to match empirical frequencies.

**Pros**: Keeps the elegant trajectory-based approach; addresses the distribution mismatch.
**Cons**: Requires calibration data; adds complexity; may overfit.
**Effort**: ~50 lines (new calibration module).

### Option D: Reduce σ_n Inflation

The `sqrt(horizon)` scaling assumes IID innovations. For a Markov-modulated process with persistent regimes, the actual horizon variance is lower. Options:
- Use `sigma_n = sigma_obs * sqrt(horizon) * sqrt(1 - ρ^2)` where ρ is the autocorrelation
- Estimate horizon variance from the trajectory paths directly (already available)
- Use the empirical standard deviation of horizon returns instead of the parametric formula

**Pros**: Addresses a fundamental theoretical issue; improves both p_up and CI calibration.
**Cons**: Requires careful derivation; may affect other parts of the pipeline.
**Effort**: ~10 lines if using trajectory path quantiles directly.

---

## 7. Recommendation

**Immediate (before any code changes)**:
1. Run the diagnostic tests in Section 5 to confirm the root cause
2. Test with `use_empirical_up_rates=True` to verify recovery

**Short-term**:
- **Option A** (restore empirical counting as default) is the safest fix. It restores the known-good methodology while preserving the trajectory for CI/expected price.
- **Option D** (reduce σ_n inflation) addresses a genuine theoretical weakness regardless of which p_up method is chosen.

**Medium-term**:
- Validate the `decay_rate=0.97` and `return_threshold_multiplier=0.5` against held-out data
- Add minimum data requirements enforcement (45+ transitions before estimating, not 30)
- Consider replacing fixed ν=5 with data-driven degrees-of-freedom estimation

---

## 8. Literature Alignment Summary

| Aspect | Implementation | Literature Consensus | Gap |
|--------|---------------|---------------------|-----|
| p_up computation (new) | Student-t survival from trajectory μ_n, σ_n | Empirical counting preferred | 🔴 High — documented mismatch |
| 3-state threshold | `0.5 × median(|r|)` | Data-driven (AIC/BIC) or sign-based | 🟡 Medium — unvalidated heuristic |
| Decay rate | 0.97 fixed | Not calibrated in literature | 🟡 Medium — introduces time-inhomogeneity bias |
| Min observations | 30 (below 45-transition floor) | ≥45 transitions for 3×3 | 🟡 Medium — underpowered estimates |
| σ_n horizon scaling | `sqrt(horizon)` assumes IID | Should account for regime autocorrelation | 🟡 Medium — inflates variance |
| HMM integration | Optional overlay | Standard approach (not overlay) | 🟢 Low — HMM is optional |
| ν=5 degrees of freedom | Hardcoded | Data-driven preferred | 🟢 Low — acceptable default |
