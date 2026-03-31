# Trump Pressure Index — Design Specification

## Overview

The Trump Pressure Index is a composite metric that tracks when political and market stress may force Trump to reverse policy — a "TACO moment" (Trump Always Chickens Out). It extends Deutsche Bank strategist Maximilian Uleer's methodology with Polymarket prediction markets, gas prices, social sentiment, and Markov chain regime modeling.

## Background

Deutsche Bank's original index tracks the 1-month change in four inputs:
1. **S&P 500** — stock market performance ("his favorite report card")
2. **1-year inflation expectations** — voter wallet pain
3. **Approval rating** — direct political pressure
4. **10-year Treasury yield** — government borrowing cost

Dexter's implementation extends this with three additional data sources for real-time pressure amplification.

## Architecture

### Data Flow

```
7 Input Sources → Z-Score Normalization → Weighted Composite → 4-State Markov → TACO Probability
```

### Input Components (7 total)

| Component | Source | Weight | Direction | Group |
|-----------|--------|--------|-----------|-------|
| S&P 500 (SPY) | Financial Datasets API | 20% | -1 (decline = pressure) | DB Core |
| 10Y Treasury Yield | FRED (DGS10) | 15% | +1 (rising = pressure) | DB Core |
| Inflation Expectations | FRED (T5YIE breakeven) | 15% | +1 (rising = pressure) | DB Core |
| Approval Rating | Polymarket proxy | 10% | -1 (falling = pressure) | DB Core |
| Gas Prices (UGA ETF) | Financial Datasets API | 12% | +1 (rising = pressure) | Extension |
| Policy Reversal Prob | Polymarket search | 15% | +1 (rising = pressure) | Extension |
| Social Sentiment | Reddit + X aggregation | 13% | -1 (bearish = pressure) | Extension |

**DB Core: 60%** | **Extensions: 40%**

### Z-Score Normalization

Each input is converted to a 1-month rolling change, then standardized via a 90-day rolling Z-score window. The `direction` field flips the sign so that positive Z always means "more pressure on Trump."

### Quality-Adaptive Weights

When inputs are unavailable, their weight is redistributed proportionally within the same group (db_core or extension). If an entire group is missing, redistribution falls back to all available components. This ensures the composite is always normalized to sum to 1.0.

### 4-State Markov Regime Model

| Regime | Score Range | Description |
|--------|------------|-------------|
| 🟢 LOW | < 0.5σ | Markets calm, policy on course |
| 🟡 MODERATE | 0.5–1.5σ | Growing unease, rhetoric shifts |
| 🟠 ELEVATED | 1.5–2.0σ | Significant stress, adjustments likely |
| 🔴 CRITICAL | ≥ 2.0σ | TACO event probable within 5–10 days |

The model uses:
- **Non-absorbing states** — regimes transition naturally via probabilities
- **Dirichlet smoothing** (α=0.1) for sparse transition estimation
- **Monte Carlo simulation** (1000 paths, 30-day horizon) for regime forecasts
- **Structural break detection** (Frobenius divergence between half-windows)

### TACO Probability Blender

```
TACO_prob = 0.4 × P_markov + 0.6 × P_polymarket
```

- **P_markov**: Probability of de-escalation (regime dropping to LOW or MODERATE) from Monte Carlo simulation
- **P_polymarket**: Average probability across policy-reversal Polymarket markets (tariff pause, trade deal, etc.) with YES-bias correction (×0.95)

### Alert Mode

When pressure score ≥ 2.0σ (CRITICAL), the tool triggers a `⚠️ TACO ALERT` that can be surfaced in watchlist briefings.

## Historical TACO Landmarks

| Date | Event | Score | Regime | Outcome |
|------|-------|-------|--------|---------|
| 2025-04-09 | Liberation Day tariff pause | 2.8σ | CRITICAL | S&P +9.5% single day |
| 2025-05-12 | US-China de-escalation | 2.3σ | CRITICAL | Tariffs 145%→30% |
| 2025-07-10 | UK trade deal | 1.5σ | ELEVATED | First bilateral deal |
| 2025-01-20 | Inauguration (baseline) | 0.0σ | LOW | Markets at highs |
| 2026-01-15 | Auto tariff exemption | 1.8σ | ELEVATED | Sector exemption |

## Resolved Design Decisions

1. **Output format**: Probability only (no time-to-TACO estimate)
2. **Policy threads**: Aggregated into one composite score
3. **Alert mode**: Auto-trigger in briefings when CRITICAL
4. **Markov states**: Non-absorbing (natural transitions)
5. **Weekends/holidays**: Non-market data (approval, sentiment) injected; market inputs carry forward

## File Structure

```
src/tools/finance/trump-pressure-index.ts       — Core tool + computation
src/tools/finance/trump-pressure-index.test.ts   — 70 unit tests
src/skills/trump-pressure/SKILL.md               — Guided analysis workflow
docs/superpowers/specs/2026-03-31-trump-pressure-index.md — This spec
```

## References

- Uleer, M. (2026). Deutsche Bank Pressure Index methodology.
- Nguyen (2018, IJFS): 4-state HMM for S&P 500.
- Welton & Ades (2005): Dirichlet priors for transition matrix estimation.
- Reichenbach & Walther (2025): YES-bias in Polymarket (124M trades).
- Davidovic & McCleary (2025, JRFM): Sentiment alpha for return prediction.
