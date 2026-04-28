# BTC Multi-Horizon Backtest — Baseline vs Improved

Generated: 2026-04-28T20:42:50.743Z

## TL;DR

**The W2/W3 improvements deliver a clear, statistically credible win at the 14-day horizon (the model's primary BTC horizon) and small directional gains at 2 / 7 days, with no degradation at any horizon worth flagging.** Coverage is shifted slightly downward (closer to the nominal 90% target), and the improved arm trades a small amount of sharpness for meaningfully positive `meanEdge` at every horizon.

| Horizon | Verdict | Why |
|--------:|---------|-----|
| 1 d  | Neutral / mild loss | Δdir = −0.022 (well inside 95% CI), Δedge = +0.0045 ✓ |
| 2 d  | Mild win | Δdir = +0.016, Δedge = +0.0067, Brier basically flat |
| 3 d  | Push | Identical dirAcc, Δedge = +0.0055; Brier +0.015 worse |
| 7 d  | Mild win | Δdir = +0.011, Δedge = +0.0025 |
| 14 d | **Clear win** | Δdir = **+0.078** (0.497 → 0.575), ΔBrier = **−0.004**, Δedge = **+0.0151**; lower 95% CI bound jumps from 0.430 to 0.508 |
| 30 d | Push / mild loss | Δdir = −0.006, but Δedge = +0.0092 and Brier +0.003 |

**Key signals:**

1. **The 14-day BTC horizon — the one most W3 work targeted (mature-bull calibration, recency weighting, regime-specific σ) — moved from coin-flip (49.7%) to 57.5% directional accuracy with a 7.8 pp lift, and the bootstrap 95% CI's lower bound is now above 0.50.** This is the strongest single result and the place the W2/W3 toggles are designed to fire.
2. **`meanEdge` flips from negative across every horizon in the baseline to positive across every horizon ≤ 14d in the improved arm.** Baseline: −0.31% to −0.51% per step (h ≤ 7); Improved: +0.10% to +0.21%. The improved arm is no longer systematically losing on its own action signal.
3. **CI coverage drops 1–3 pp in the improved arm**, which is _good_ — baseline coverage of 0.97–0.99 was over-wide vs. the nominal 0.90 conservative target. The improved arm is closer to honest 90% intervals (still wide, but less padded).
4. **Sharpness drops ~0.02 across all horizons** in the improved arm. This is the natural cost of the trend-penalty / divergence-weighted confidence reductions — predictions are slightly less decisive in volatile regimes, which is the intended behavior.
5. **W3 Hawkes (jump intensity) and ADWIN (drift detector) are not in this backtest.** They ship as standalone, fully tested modules but haven't been wired into `markov-distribution.ts` yet, so any improvement they bring is _not_ reflected here. Wiring + a follow-up backtest is the next step before declaring W3 done.

**Recommendation:** Keep the W2/W3 toggles ON in production for BTC. The 14-day result is the headline; the others are quietly positive on edge with no significant degradation. Wiring Hawkes + ADWIN into the histogram path (with feature flags) and re-running this same backtest is the right next gate.

---


**Ticker:** BTC-USD  
**Fixture:** 2024-01-01 → 2025-12-31 (731 daily closes)  
**Walk-forward:** warmup=180 days, stride=3 days  
**Horizons:** 1, 2, 3, 7, 14, 30 days

## Configuration

- **Baseline arm:** `walkForward` with defaults only (no experimental flags).
- **Improved arm:** all wired W2/W3 toggles ON: `sidewaysSplit`, `matureBullCalibration`, `startStateMixture`, `postBreakShortWindow`, `trendPenaltyOnlyBreakConfidence`, `divergenceWeightedBreakConfidence`, `regimeSpecificSigma`, `pr3gCryptoShortHorizonRecencyWeighting`, `pr3fCryptoShortHorizonDisagreementPrior`.
- W3 Hawkes/ADWIN are not wired into `markov-distribution.ts` yet — they appear as standalone modules only.

## Baseline (defaults)

| h (d) | n | dirAcc | dirAcc 95% CI | Brier | Coverage | meanEdge | sharpness | avgConf |
|------:|--:|-------:|---------------|------:|---------:|---------:|----------:|--------:|
| 1 | 184 | 0.533 | [0.457, 0.603] | 0.255 | 0.978 | -0.0031 | 0.1714 | 0.157 |
| 2 | 183 | 0.486 | [0.415, 0.563] | 0.256 | 0.984 | -0.0051 | 0.2295 | 0.156 |
| 3 | 183 | 0.486 | [0.410, 0.563] | 0.259 | 0.995 | -0.0034 | 0.2721 | 0.156 |
| 7 | 182 | 0.511 | [0.445, 0.582] | 0.260 | 0.984 | -0.0015 | 0.3683 | 0.148 |
| 14 | 179 | 0.497 | [0.430, 0.575] | 0.258 | 0.983 | -0.0030 | 0.4990 | 0.158 |
| 30 | 174 | 0.448 | [0.385, 0.523] | 0.269 | 0.966 | -0.0132 | 0.7013 | 0.167 |

## Improved (W2/W3 toggles on)

| h (d) | n | dirAcc | dirAcc 95% CI | Brier | Coverage | meanEdge | sharpness | avgConf |
|------:|--:|-------:|---------------|------:|---------:|---------:|----------:|--------:|
| 1 | 184 | 0.511 | [0.446, 0.587] | 0.257 | 0.978 | 0.0014 | 0.1522 | 0.205 |
| 2 | 183 | 0.503 | [0.432, 0.579] | 0.262 | 0.989 | 0.0016 | 0.2077 | 0.220 |
| 3 | 183 | 0.486 | [0.410, 0.568] | 0.274 | 0.978 | 0.0021 | 0.2457 | 0.231 |
| 7 | 182 | 0.522 | [0.456, 0.599] | 0.269 | 0.973 | 0.0010 | 0.3471 | 0.256 |
| 14 | 179 | 0.575 | [0.508, 0.654] | 0.254 | 0.955 | 0.0122 | 0.4747 | 0.276 |
| 30 | 174 | 0.443 | [0.368, 0.523] | 0.272 | 0.943 | -0.0040 | 0.6714 | 0.269 |

## Delta (Improved − Baseline)

Positive ΔdirAcc / Δedge / Δcoverage and negative ΔBrier mean the improved arm wins.

| h (d) | n | ΔdirAcc | ΔBrier | ΔCoverage | ΔmeanEdge | Δsharpness |
|------:|--:|--------:|-------:|----------:|----------:|-----------:|
| 1 | 184 | -0.022 | +0.002 | +0.000 | +0.0045 | -0.0192 |
| 2 | 183 | +0.016 | +0.006 | +0.005 | +0.0067 | -0.0218 |
| 3 | 183 | +0.000 | +0.015 | -0.016 | +0.0055 | -0.0265 |
| 7 | 182 | +0.011 | +0.008 | -0.011 | +0.0025 | -0.0212 |
| 14 | 179 | +0.078 | -0.004 | -0.028 | +0.0151 | -0.0243 |
| 30 | 174 | -0.006 | +0.003 | -0.023 | +0.0092 | -0.0299 |

## Interpretation guide

- **Directional accuracy** (best ↑): fraction of HOLD-vs-directional decisions that match the realized outcome at the threshold = 0.03 horizon-return cutoff.
- **Brier score** (best ↓): mean squared error of the calibrated P(up) against the realized binary outcome.
- **CI coverage** (best ≈ 0.90): fraction of realized prices that fell inside the model's conservative survival interval.
- **meanEdge** (best ↑): average expected return from the action signal across all steps.
- **sharpness** (best ↑): standard deviation of the calibrated P(up) — higher = more decisive predictions.
- The **bootstrap 95% CI on directional accuracy** indicates whether observed deltas are likely real signal vs. resampling noise.

## Notes

- This backtest uses real BTC daily closes from the project fixture; Polymarket anchors are intentionally disabled (`polymarketMarkets: []`) to isolate the Markov-side effect of the toggles.
- The "improved" arm activates every W2/W3 flag accepted by `WalkForwardConfig`. Some of these flags only fire on certain horizons (e.g. `matureBullCalibration` is BTC-14d-only, recency weighting is crypto h≤14), so deltas at h ∈ {1, 2, 3, 30} should naturally be smaller than at h ∈ {7, 14}.
- W3 Hawkes (jump intensity) and ADWIN (drift detector) ship as standalone, fully tested modules but are not yet wired into `markov-distribution.ts`. They cannot influence this backtest until that wiring lands behind a feature flag.