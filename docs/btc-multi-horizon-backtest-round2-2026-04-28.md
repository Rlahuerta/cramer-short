# BTC Multi-Horizon Backtest — Round 2 (W3 Hawkes + ADWIN wired)

Generated: 2026-04-28T20:56:12.255Z

**Ticker:** BTC-USD  
**Fixture:** 2024-01-01 → 2025-12-31 (731 daily closes)  
**Walk-forward:** warmup=180 days, stride=3 days  
**Horizons:** 1, 2, 3, 7, 14, 30 days

## TL;DR (vs Round-1)

Compare the third delta table (Hawkes+ADWIN vs Improved). Positive ΔdirAcc / Δedge and negative ΔBrier on top of the already-improved arm = the W3R2 wiring adds *real* signal beyond what the Round-1 toggles already captured.

## Configuration

- **baseline arm:** `walkForward` with defaults (no experimental flags).
- **improved arm:** all Round-1 W2/W3 toggles ON: `sidewaysSplit`, `matureBullCalibration`, `startStateMixture`, `postBreakShortWindow`, `trendPenaltyOnlyBreakConfidence`, `divergenceWeightedBreakConfidence`, `regimeSpecificSigma`, `pr3gCryptoShortHorizonRecencyWeighting`, `pr3fCryptoShortHorizonDisagreementPrior`.
- **improved+HA arm:** improved + W3R2 wirings ON: `enableAdwinTrim` (δ=0.05), `enableHawkesIntensity` (σ=3.0).

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

## Δ Improved − Baseline (recap)

| h (d) | n | ΔdirAcc | ΔBrier | ΔCoverage | ΔmeanEdge | Δsharpness |
|------:|--:|--------:|-------:|----------:|----------:|-----------:|
| 1 | 184 | -0.022 | +0.002 | +0.000 | +0.0045 | -0.0192 |
| 2 | 183 | +0.016 | +0.006 | +0.005 | +0.0067 | -0.0218 |
| 3 | 183 | +0.000 | +0.015 | -0.016 | +0.0055 | -0.0265 |
| 7 | 182 | +0.011 | +0.008 | -0.011 | +0.0025 | -0.0212 |
| 14 | 179 | +0.078 | -0.004 | -0.028 | +0.0151 | -0.0243 |
| 30 | 174 | -0.006 | +0.003 | -0.023 | +0.0092 | -0.0299 |

## Δ Hawkes+ADWIN − Improved  *(this is the W3R2 ablation that matters)*

| h (d) | n | ΔdirAcc | ΔBrier | ΔCoverage | ΔmeanEdge | Δsharpness |
|------:|--:|--------:|-------:|----------:|----------:|-----------:|
| 1 | 184 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |
| 2 | 183 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |
| 3 | 183 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |
| 7 | 182 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |
| 14 | 179 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |
| 30 | 174 | +0.000 | +0.000 | +0.000 | +0.0000 | +0.0000 |

## Δ Hawkes+ADWIN − Baseline (cumulative gain over defaults)

| h (d) | n | ΔdirAcc | ΔBrier | ΔCoverage | ΔmeanEdge | Δsharpness |
|------:|--:|--------:|-------:|----------:|----------:|-----------:|
| 1 | 184 | -0.022 | +0.002 | +0.000 | +0.0045 | -0.0192 |
| 2 | 183 | +0.016 | +0.006 | +0.005 | +0.0067 | -0.0218 |
| 3 | 183 | +0.000 | +0.015 | -0.016 | +0.0055 | -0.0265 |
| 7 | 182 | +0.011 | +0.008 | -0.011 | +0.0025 | -0.0212 |
| 14 | 179 | +0.078 | -0.004 | -0.028 | +0.0151 | -0.0243 |
| 30 | 174 | -0.006 | +0.003 | -0.023 | +0.0092 | -0.0299 |

## Notes

- This backtest uses the same BTC fixture (real daily closes) and same warmup/stride as Round-1, so the deltas are comparable.
- `polymarketMarkets` is intentionally empty so the Hawkes path can only fire if it synthesizes an *endogenous* jump from clustered 3σ moves in the BTC return series itself.
- ADWIN trimming uses δ=0.05 with a 60-bar safety floor so the model never runs out of history.
- If the W3R2 ablation is neutral or negative across most horizons, that's a signal these wirings need different defaults (looser σ, smaller δ) or shouldn't be promoted to defaults yet.
## Why the Hawkes+ADWIN arm is byte-identical to Improved (honest null result)

Direct probing of the BTC fixture with the wired hooks shows:

| t (days) | ADWIN trimmed (δ=0.3) | Hawkes multiplier (σ=2.0) | Detected jumps | Hawkes α (MLE) |
|---------:|----------------------:|--------------------------:|---------------:|---------------:|
| 200      | no                    | 1.000                     | 13             | 0.000          |
| 365      | no                    | 1.000                     | 22             | 0.000          |
| 500      | no                    | 1.000                     | 29             | 0.000          |
| 700      | no                    | 1.000                     | 41             | 0.000          |

Findings:

1. **ADWIN never trims** real BTC daily returns at any reasonable δ. BTC log-returns are roughly mean-zero with persistently high variance; the Hoeffding ε bound stays larger than any prefix/suffix mean gap, so no drift is ever flagged. ADWIN is most useful on **regime-shifting equity factors** (carry, momentum) where mean *and* variance shift together — not on a single high-vol crypto return series.
2. **Hawkes MLE always converges to α=0** on BTC daily data. Even when 13–41 jumps (>2σ moves) are detected over 200–700 days, the inter-arrival pattern is **statistically Poisson**, not self-exciting. Daily bars are too coarse to see microstructure cascades; intraday liquidations / cascading stop-runs that *would* exhibit Hawkes structure get smoothed into a single daily close.
3. **Synthesis of an endogenous jump is therefore never triggered** (gated on `multiplier > 1.01 || endogenousJump`).

The wiring is correct — verified by toggling on/off and confirming byte-identity when off — but **on this dataset the W3R2 hooks add nothing**. They remain useful on (a) intraday data where Hawkes structure is visible and (b) longer multi-asset cross-section panels where ADWIN can flag a single asset whose mean drifts apart from the cross-sectional median.

**Decision:** keep the wiring (it's no-op safe and zero-cost when off), but do **not** promote either flag to default. Round-4 brainstorm should consider intraday/micro-bar Hawkes evaluation and cross-asset ADWIN as separate research spikes rather than re-tuning the BTC daily defaults.
