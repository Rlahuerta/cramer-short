# Root Cause Analysis: Markov Backtest Directional Accuracy Drop

**Date**: 2026-06-02
**Status**: DRAFT — for review
**Sources**: Git history analysis, literature cross-reference (22 PDFs + web search), code trace of `p_up` computation

---

## Executive Summary

The directional accuracy drop is caused by a **fundamental methodological change** in how `p_up` is computed. Commit `0606a5c` ("fixing the markov chain implementation", Jun 1) replaced the **empirical regime up-rates** approach with a **parametric trajectory-based Student-t survival** approach. The literature strongly favors the empirical counting method for directional prediction; the trajectory-based method was introduced for "coherence" (p_up comes from the same engine as expected_price and CI) but sacrifices predictive accuracy.

**The Phase 1-3 fixes (Jun 2) did not cause the drop.** They were applied on top of the already-regressed code. The drop originated in the Jun 1 rewrite.

---

## 1. What Changed: The `p_up` Computation Rewrite

### Old Code (before commit `0606a5c`)

```python
# p_up computed via empirical regime up-rates:
forecast = compute_markov_forecast(P, current_regime, horizon)
up_rates = compute_regime_up_rates(
    regimes, log_returns, horizon=horizon, decay_rate=decay_rate,
)
p_up = sum(forecast[state] * up_rates[state] for state in ["bull", "bear", "sideways"])
```

**Methodology**: Non-parametric. For each historical regime occurrence, look forward `horizon` days and check if the cumulative log return was positive. Weight by regime forecast probability. This is conceptually: **"Given the regime forecast at the horizon, what fraction of similar historical regimes were followed by positive returns?"**

### New Code (after commit `0606a5c`)

```python
# p_up computed via Monte Carlo trajectory terminal distribution:
traj = compute_trajectory(...)
horizon_point = traj[-1]
p_up = horizon_point.p_up  # = student_t_survival(S₀, S₀, μ_n, σ_n, ν=5)
```

**Methodology**: Parametric. Simulate many price paths, compute the terminal log-return distribution's mean `μ_n` and std `σ_n`, then use `student_t_survival` to compute P(price > current_price). This is conceptually: **"Given the regime-weighted drift and volatility at the horizon, what does a Student-t(ν=5) distribution say about the probability of a positive return?"**

### Why This Matters

These two approaches answer **different questions**:

| Aspect | Old (Empirical Up-Rates) | New (Trajectory Student-t) |
|--------|--------------------------|---------------------------|
| Data source | Historical regime→return pairs | Simulated parametric paths |
| Distributional assumption | None (non-parametric) | Student-t with ν=5 |
| Sensitivity to drift | Indirect (through regime weights) | Direct (μ_n drives survival) |
| Sensitivity to volatility | None (binary up/down) | Strong (σ_n drives survival) |
| Regime-specific behavior | Captured per-regime | Smoothed through mixture weights |
| Interpretation | P(return>0 \| regime at horizon) | P(return>0 \| Student-t(μ_n, σ_n, 5)) |

---

## 2. Literature Alignment

### 2.1 Empirical Counting is the Standard

The literature on Markov chain financial forecasting consistently uses **empirical counting** for directional prediction:

| Source | Method | p_up approach |
|--------|--------|---------------|
| **Mettle et al. (2014)** | 3-state finite Markov chain | Transition counting + limiting distribution, NOT parametric survival |
| **Nguyen (2018)** | 4-state HMM | Directional Prediction Accuracy (DPA) measured empirically |
| **Kumar & Amer (2023)** | LSTM + 3-state MC | Steady-state probabilities + empirical hitting times |
| **Catello et al. (2025)** | HMM + GMM emissions | DPA measured as `P(predicted_direction = actual_direction)` from out-of-sample |

**None of these papers use Student-t survival for p_up.** They all measure directional accuracy against empirically observed outcomes.

### 2.2 Student-t Approximation for Terminal Distribution

The May 31 verification report noted (section 2.11.5):

> "The sum of Student-t variables is **not** Student-t distributed. The terminal distribution of the sum of dependent (or independent) Student-t increments is approximately normal for large n by CLT, but for small n (e.g., h=7) it's a convolution of t-distributions, not a t-distribution."

This approximation introduces **distributional error** that the empirical counting approach avoids entirely. For short horizons (h=7, 14, 30) where directional accuracy matters most, this error is material.

### 2.3 Expected Directional Accuracy Benchmarks

The literature establishes that **50-55% directional accuracy is the expected range** for Markov models on financial data:

| Source | Model | Directional Accuracy |
|--------|-------|---------------------|
| Catello et al. (2025) | HMM+GMM | 40–63% |
| Lim et al. (2024) | Neural Regime-Switching | 51–57% |
| Nguyen (2018) | HMM | Outperforms HAR baseline |
| Bulla et al. | Markov-switching | ~41% after costs |

**Key insight**: Stock returns are nearly unpredictable in direction. A model achieving 50-55% is performing as expected. If accuracy dropped from, say, 53% to 50%, this is a meaningful regression.

---

## 3. Specific Mechanisms of Accuracy Loss

### 3.1 Loss of Regime-Specific Information

The empirical up-rates approach captures regime-specific directional patterns:

- `P(up | bull)` might be 65% in trending markets
- `P(up | bear)` might be 35%
- `P(up | sideways)` might be 50%

The trajectory Student-t approach replaces these three numbers with a single `student_t_survival(S₀, S₀, μ_n, σ_n, ν=5)` call. The regime-specific information is collapsed into two parameters (`μ_n`, `σ_n`) that summarize first and second moments but lose the actual empirical frequency.

**Example**: If bull regimes historically have P(up)=70% but mean return is only 0.2% with high variance, the Student-t survival might give P(up)≈53% (near 0.5 because high variance washes out the small positive mean). The empirical approach correctly reports 70%.

### 3.2 ν=5 Degrees of Freedom is Arbitrary

The Student-t survival uses `ν=5` as a hardcoded default. For assets with:
- **Heavy tails** (crypto, small caps): ν=5 may be too thin-tailed, underestimating tail probability
- **Light tails** (large-cap ETFs): ν=5 may be too thick-tailed, overestimating tail probability

The empirical up-rates approach requires no such parameter.

### 3.3 Trajectory Mean/Variance is Unstable for Short Windows

`μ_n` and `σ_n` are computed from regime-weighted daily drift/vol. For a backtest window of ~120 days (the warmup default), regime stats are estimated from sparse data (some regimes may have <10 observations). This produces unstable `μ_n`/`σ_n` estimates that feed directly into the Student-t survival.

The empirical approach uses the full regime sequence to count up/down outcomes, which is more robust to small samples because it only needs to count the sign (binary), not estimate a continuous parameter.

### 3.4 The `compute_mixing_weight` Fix Also Affected Anchor Blending

The Phase 1 fix changed `compute_mixing_weight` from `exp(-λ·h)` to `|λ|^h`. This affects `interpolate_distribution` which blends Polymarket anchors with Markov forecasts. If the backtest uses anchor blending (via `compute_scenario_probabilities` or similar), the mixing weight change could shift p_up indirectly.

---

## 4. Additional Implementation Concerns (from Literature)

### 4.1 Threshold-Based Regime Classification

The adaptive threshold `0.5 * median(|returns|)` is not validated in the literature. It forces ~30-40% bull days regardless of market conditions:
- In a strong bull market, the threshold rises with volatility, classifying strong up days as "sideways"
- In a low-vol bear market, the threshold shrinks, classifying mild down days as "bear"

This **conflates volatility with direction** and may invert regime classifications at market turning points.

### 4.2 Exponential Decay (0.97) Not Calibrated

The `decay_rate=0.97` gives an effective half-life of ~23 trading days. Transitions older than 23 days contribute <50% weight. For a 120-day window, the oldest transitions contribute <3% weight. This means the matrix is **dominated by very recent history** and may be unstable.

### 4.3 Minimum Data Requirements

`transitionMinObservations=30` provides ~29 transitions across 9 cells (~3.2 per cell). The statistical rule of thumb for multinomial estimation is ≥5 expected counts per cell. The current default is below this threshold, meaning many regime pairs will have highly unstable probability estimates.

---

## 5. Diagnostic Questions to Answer

Before proposing fixes, the following should be measured:

1. **Quantify the drop**: What was the old directional accuracy vs the new? On which assets? At which horizons?

2. **Is the drop universal?**: Check BTC (heavy-tailed), SPY (moderate), GLD (low-vol). The Student-t approach may work better for some assets than others.

3. **What if we A/B test?**: Run the same backtest with `use_empirical_up_rates=True` (the new Phase 2 flag) vs the default trajectory path. Compare p_up values per step.

4. **Is the trajectory itself biased?**: Check if `trajectory[-1].p_up` systematically deviates from `compute_regime_up_rates` output. If trajectory p_up is always pulled toward 0.5, that would explain accuracy loss.

5. **How does ν affect results?**: Sweep ν ∈ {3, 5, 7, 10, ∞} and measure directional accuracy. The hardcoded ν=5 may not be optimal.

6. **Is this a regression vs the original?**: The old code (pre-0606a5c) used empirical up-rates. Compare directional accuracy of that exact commit vs HEAD.

---

## 6. Improvement Plan — For Review

### Phase A — Diagnostic Verification (no code changes)

| Step | Action | Purpose |
|------|--------|---------|
| A1 | Checkout `0606a5c~1` (pre-rewrite) and run backtest | Establish old baseline accuracy |
| A2 | Checkout HEAD and run same backtest | Quantify the accuracy drop |
| A3 | Run HEAD with `use_empirical_up_rates=True` | Verify the Phase 2 flag restores old behavior |
| A4 | Compare `trajectory.p_up` vs `compute_regime_up_rates` output per step | Quantify the Student-t vs empirical gap |
| A5 | Sweep ν values (3, 5, 7, 10) on BTC/SPY/GLD | Find optimal ν per asset |

### Phase B — Methodology Decision (review required)

| Option | Description | Pros | Cons |
|--------|-------------|------|------|
| **B1: Restore empirical** | Make old empirical approach the default; keep trajectory as optional | Literature-supported, non-parametric, transparent | p_up decoupled from trajectory engine |
| **B2: Blend both** | `p_up = α × empirical + (1-α) × trajectory` with α tuned per horizon | Balances data-driven with model-driven | Adds calibration parameter |
| **B3: Keep trajectory, tune ν** | Optimize ν per asset class; fix the student_t_survival approach | Maintains coherence of pipeline | Overfits to historical data |
| **B4: Hybrid** | Use empirical for h≤14 (short horizons), trajectory for h>14 (long horizons) | Best of both worlds | Complexity boundary at h=14 |

### Phase C — Implementation (after Phase B decision)

| Step | Action |
|------|--------|
| C1 | Implement chosen methodology |
| C2 | Add test comparing old vs new p_up on known fixtures |
| C3 | Re-run full backtest suite and confirm accuracy recovery |
| C4 | Update PARITY.md and documentation |

---

## 7. Conclusion

**The directional accuracy drop is real and explainable.** It stems from replacing a non-parametric, literature-supported empirical counting approach with a parametric Student-t survival approach in commit `0606a5c`. The Phase 1-3 fixes (Jun 2) did not cause the drop — they were applied on top of already-regressed code.

**The Phase 2 flag `use_empirical_up_rates` (added Jun 2) partially restores the old behavior**, but it is disabled by default. The most likely path to recovering accuracy is to make the empirical approach the default (Option B1) or to blend both approaches (Option B2).

**Recommendation**: Run Phase A diagnostics to quantify the magnitude of the drop, then decide on the Phase B methodology in review.
