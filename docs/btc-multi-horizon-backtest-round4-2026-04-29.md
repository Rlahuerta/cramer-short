# BTC Multi-Horizon Backtest — Round 4 (R4 feature flags)

Generated: 2026-04-29T07:20:44.174Z

**Ticker:** BTC-USD  
**Fixture:** 2024-01-01 → 2025-12-31 (731 daily closes)  
**Walk-forward:** warmup=180 days, stride=3 days  
**Horizons:** 1, 2, 3, 7, 14, 30 days

## TL;DR

Focus on the **Δ R4 − Hawkes+ADWIN** table. Positive ΔdirAcc / Δedge
and negative ΔBrier on top of the already-improved+HA arm confirm the R4
flags add real signal beyond W3R2.

## Configuration

- **baseline:** default flags only.
- **improved:** all Round-1 W2/W3 toggles: `sidewaysSplit`, `matureBullCalibration`, `startStateMixture`, `postBreakShortWindow`, `trendPenaltyOnlyBreakConfidence`, `divergenceWeightedBreakConfidence`, `regimeSpecificSigma`, `pr3gCryptoShortHorizonRecencyWeighting`, `pr3fCryptoShortHorizonDisagreementPrior`.
- **improved+HA:** improved + W3R2 wirings: `enableAdwinTrim` (δ=0.05), `enableHawkesIntensity` (σ=3.0).
- **improved+R4:** improved+HA + Round-4: `enableGarchVol`, `enableKswinTrim` (α=0.005), `enableCrossAssetBias` (peers: SPY/GLD/QQQ, λ=0.005).

> **Note on Regime Platt (R4 Idea 3):** omitted from this single-pass walk-forward
> because fitting the recalibrator requires a separate validation set of
> (pUp, regime, realized) triples — unavailable without a two-pass approach. The
> unit tests verify its correctness; production use fits on a prior window.

> **Cross-asset Lasso peers:** SPY (US equities), GLD (gold), QQQ (tech).  Daily
> log-returns are computed from the same fixture and passed alongside BTC closes.
> The Lasso λ=0.005 regularises away noise so only genuine co-movement survives.

## Baseline (defaults)

| h (d) | n | dirAcc | dirAcc 95% CI | Brier | Coverage | meanEdge | sharpness | avgConf |
|------:|--:|-------:|---------------|------:|---------:|---------:|----------:|--------:|
| 1 | 184 | 0.533 | [0.457, 0.603] | 0.255 | 0.978 | -0.0031 | 0.1714 | 0.157 |
| 2 | 183 | 0.486 | [0.415, 0.563] | 0.256 | 0.984 | -0.0051 | 0.2295 | 0.156 |
| 3 | 183 | 0.486 | [0.410, 0.563] | 0.259 | 0.995 | -0.0034 | 0.2721 | 0.156 |
| 7 | 182 | 0.511 | [0.445, 0.582] | 0.260 | 0.984 | -0.0015 | 0.3683 | 0.148 |
| 14 | 179 | 0.497 | [0.430, 0.575] | 0.258 | 0.983 | -0.0030 | 0.4990 | 0.158 |
| 30 | 174 | 0.448 | [0.385, 0.523] | 0.269 | 0.966 | -0.0132 | 0.7013 | 0.167 |

## Improved (Round-1 W2/W3 toggles ON)

| h (d) | n | dirAcc | dirAcc 95% CI | Brier | Coverage | meanEdge | sharpness | avgConf |
|------:|--:|-------:|---------------|------:|---------:|---------:|----------:|--------:|
| 1 | 184 | 0.511 | [0.446, 0.587] | 0.257 | 0.978 | 0.0014 | 0.1522 | 0.205 |
| 2 | 183 | 0.503 | [0.432, 0.579] | 0.262 | 0.989 | 0.0016 | 0.2077 | 0.220 |
| 3 | 183 | 0.486 | [0.410, 0.568] | 0.274 | 0.978 | 0.0021 | 0.2457 | 0.231 |
| 7 | 182 | 0.522 | [0.456, 0.599] | 0.269 | 0.973 | 0.0010 | 0.3471 | 0.256 |
| 14 | 179 | 0.575 | [0.508, 0.654] | 0.254 | 0.955 | 0.0122 | 0.4747 | 0.276 |
| 30 | 174 | 0.443 | [0.368, 0.523] | 0.272 | 0.943 | -0.0040 | 0.6714 | 0.269 |

## Improved + Hawkes + ADWIN (W3R2)

| h (d) | n | dirAcc | dirAcc 95% CI | Brier | Coverage | meanEdge | sharpness | avgConf |
|------:|--:|-------:|---------------|------:|---------:|---------:|----------:|--------:|
| 1 | 184 | 0.511 | [0.446, 0.587] | 0.257 | 0.978 | 0.0014 | 0.1522 | 0.205 |
| 2 | 183 | 0.503 | [0.432, 0.579] | 0.262 | 0.989 | 0.0016 | 0.2077 | 0.220 |
| 3 | 183 | 0.486 | [0.410, 0.568] | 0.274 | 0.978 | 0.0021 | 0.2457 | 0.231 |
| 7 | 182 | 0.522 | [0.456, 0.599] | 0.269 | 0.973 | 0.0010 | 0.3471 | 0.256 |
| 14 | 179 | 0.575 | [0.508, 0.654] | 0.254 | 0.955 | 0.0122 | 0.4747 | 0.276 |
| 30 | 174 | 0.443 | [0.368, 0.523] | 0.272 | 0.943 | -0.0040 | 0.6714 | 0.269 |

## Improved + R4 flags (GARCH + KSWIN + Lasso)

| h (d) | n | dirAcc | dirAcc 95% CI | Brier | Coverage | meanEdge | sharpness | avgConf |
|------:|--:|-------:|---------------|------:|---------:|---------:|----------:|--------:|
| 1 | 184 | 0.516 | [0.451, 0.598] | 0.256 | 0.978 | 0.0015 | 0.1517 | 0.208 |
| 2 | 183 | 0.508 | [0.443, 0.585] | 0.260 | 0.989 | 0.0019 | 0.2074 | 0.224 |
| 3 | 183 | 0.486 | [0.410, 0.563] | 0.274 | 0.978 | 0.0017 | 0.2451 | 0.235 |
| 7 | 182 | 0.516 | [0.451, 0.593] | 0.271 | 0.973 | 0.0009 | 0.3473 | 0.259 |
| 14 | 179 | 0.570 | [0.503, 0.642] | 0.260 | 0.955 | 0.0118 | 0.4738 | 0.278 |
| 30 | 174 | 0.443 | [0.368, 0.523] | 0.279 | 0.943 | -0.0040 | 0.6678 | 0.272 |

## Δ Improved − Baseline (Round-1 cumulative gain recap)

| h (d) | n | ΔdirAcc | ΔBrier | ΔCoverage | ΔmeanEdge | Δsharpness |
|------:|--:|--------:|-------:|----------:|----------:|-----------:|
| 1 | 184 | -0.022 | +0.002 | +0.000 | +0.0045 | -0.0192 |
| 2 | 183 | +0.016 | +0.006 | +0.005 | +0.0067 | -0.0218 |
| 3 | 183 | +0.000 | +0.015 | -0.016 | +0.0055 | -0.0265 |
| 7 | 182 | +0.011 | +0.008 | -0.011 | +0.0025 | -0.0212 |
| 14 | 179 | +0.078 | -0.004 | -0.028 | +0.0151 | -0.0243 |
| 30 | 174 | -0.006 | +0.003 | -0.023 | +0.0092 | -0.0299 |

## Δ Hawkes+ADWIN − Improved  *(W3R2 ablation)*

| h (d) | n | ΔdirAcc | ΔBrier | ΔCoverage | ΔmeanEdge | Δsharpness |
|------:|--:|--------:|-------:|----------:|----------:|-----------:|
| 1 | 184 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |
| 2 | 183 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |
| 3 | 183 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |
| 7 | 182 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |
| 14 | 179 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |
| 30 | 174 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |

## Δ R4 − Hawkes+ADWIN  *(the R4 ablation that matters)*

| h (d) | n | ΔdirAcc | ΔBrier | ΔCoverage | ΔmeanEdge | Δsharpness |
|------:|--:|--------:|-------:|----------:|----------:|-----------:|
| 1 | 184 | +0.005 | -0.001 | +0.000 | +0.0001 | -0.0006 |
| 2 | 183 | +0.005 | -0.002 | +0.000 | +0.0004 | -0.0003 |
| 3 | 183 | +0.000 | -0.001 | +0.000 | -0.0004 | -0.0006 |
| 7 | 182 | -0.005 | +0.002 | +0.000 | -0.0001 | +0.0002 |
| 14 | 179 | -0.006 | +0.006 | +0.000 | -0.0004 | -0.0009 |
| 30 | 174 | +0.000 | +0.007 | +0.000 | +0.0000 | -0.0035 |

## Δ R4 − Baseline (cumulative gain over defaults)

| h (d) | n | ΔdirAcc | ΔBrier | ΔCoverage | ΔmeanEdge | Δsharpness |
|------:|--:|--------:|-------:|----------:|----------:|-----------:|
| 1 | 184 | -0.016 | +0.001 | +0.000 | +0.0047 | -0.0198 |
| 2 | 183 | +0.022 | +0.005 | +0.005 | +0.0070 | -0.0221 |
| 3 | 183 | +0.000 | +0.015 | -0.016 | +0.0051 | -0.0270 |
| 7 | 182 | +0.005 | +0.010 | -0.011 | +0.0024 | -0.0210 |
| 14 | 179 | +0.073 | +0.002 | -0.028 | +0.0148 | -0.0252 |
| 30 | 174 | -0.006 | +0.010 | -0.023 | +0.0092 | -0.0334 |

## Notes

- Fixture: 2024-01-01 → 2025-12-31 real BTC daily closes, warmup=180 days, stride=3 days.
- `polymarketMarkets` is empty so Hawkes fires only on endogenous 3σ BTC returns.
- GARCH: `fitGarch11` on the history window; per-day scalar clamped [0.33, 3.0].
- KSWIN: operates on |log-return| (variance proxy) at α=0.005; runs after ADWIN.
- Cross-asset Lasso: per-day bias clipped to [-0.05, +0.05] so no single peer dominates.
- A neutral or negative Δ on the R4 arm is informative: it means these flags need
  different defaults or the fixture period lacks the regimes where they help most.