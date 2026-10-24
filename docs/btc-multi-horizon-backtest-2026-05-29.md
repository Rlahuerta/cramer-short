# BTC Multi-Horizon Backtest — Baseline vs Improved

Generated: 2026-05-29T15:03:05.361Z

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
| 1 | 184 | 0.609 | [0.538, 0.674] | 0.255 | 0.978 | -0.0056 | 0.1712 | 0.400 |
| 2 | 183 | 0.541 | [0.475, 0.617] | 0.255 | 0.984 | -0.0083 | 0.2287 | 0.401 |
| 3 | 183 | 0.579 | [0.503, 0.656] | 0.258 | 0.995 | -0.0068 | 0.2698 | 0.401 |
| 7 | 182 | 0.500 | [0.434, 0.571] | 0.260 | 0.989 | -0.0126 | 0.3630 | 0.387 |
| 14 | 179 | 0.486 | [0.413, 0.564] | 0.263 | 0.983 | -0.0112 | 0.4931 | 0.410 |
| 30 | 174 | 0.443 | [0.368, 0.511] | 0.273 | 0.960 | -0.0003 | 0.6902 | 0.290 |

## Improved (W2/W3 toggles on)

| h (d) | n | dirAcc | dirAcc 95% CI | Brier | Coverage | meanEdge | sharpness | avgConf |
|------:|--:|-------:|---------------|------:|---------:|---------:|----------:|--------:|
| 1 | 184 | 0.663 | [0.598, 0.728] | 0.256 | 0.978 | -0.0057 | 0.1558 | 0.324 |
| 2 | 183 | 0.546 | [0.486, 0.623] | 0.259 | 0.989 | -0.0080 | 0.2110 | 0.331 |
| 3 | 183 | 0.497 | [0.426, 0.574] | 0.263 | 0.978 | -0.0077 | 0.2494 | 0.337 |
| 7 | 182 | 0.495 | [0.423, 0.571] | 0.260 | 0.973 | -0.0087 | 0.3495 | 0.360 |
| 14 | 179 | 0.525 | [0.453, 0.603] | 0.249 | 0.961 | -0.0018 | 0.4745 | 0.383 |
| 30 | 174 | 0.483 | [0.408, 0.557] | 0.251 | 0.937 | 0.0072 | 0.6623 | 0.253 |

## Delta (Improved − Baseline)

Positive ΔdirAcc / Δedge / Δcoverage and negative ΔBrier mean the improved arm wins.

| h (d) | n | ΔdirAcc | ΔBrier | ΔCoverage | ΔmeanEdge | Δsharpness |
|------:|--:|--------:|-------:|----------:|----------:|-----------:|
| 1 | 184 | +0.054 | +0.001 | +0.000 | -0.0001 | -0.0154 |
| 2 | 183 | +0.005 | +0.004 | +0.005 | +0.0003 | -0.0177 |
| 3 | 183 | -0.082 | +0.005 | -0.016 | -0.0010 | -0.0204 |
| 7 | 182 | -0.005 | -0.000 | -0.016 | +0.0039 | -0.0135 |
| 14 | 179 | +0.039 | -0.014 | -0.022 | +0.0094 | -0.0186 |
| 30 | 174 | +0.040 | -0.021 | -0.023 | +0.0076 | -0.0279 |

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