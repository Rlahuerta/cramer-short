# When to Use the Markov Model

**Last Updated:** 2026-04-04
**Coverage:** 14 tickers, 6 horizons (5–30d)
**Status:** Coverage-Complete (Option A approved)

---

## Quick Answer

**Use this model when:**
- ✅ You want directional forecasts for **broad-market ETFs** (SPY, QQQ, VOO, VTI, DIA) or **gold ETFs** (GLD, IAU)
- ✅ Your horizon is **14–30 days** (the model's sweet spot)
- ✅ You accept **~60–75% directional accuracy** with explicit confidence intervals
- ✅ You want probability-weighted scenarios, not point forecasts

**Do NOT use this model when:**
- ❌ You trade **crypto** (BTC-USD is negative at all horizons)
- ❌ You need **<7d** or **>30d** forecasts (5–10d is noisy, 45–60d has no Markov signal)
- ❌ You trade **single-name equities** expecting reliable signals (AAPL/TSLA never clear confirmation bar)
- ❌ You want certainty (the model shows **probability distributions**, not guarantees)

---

## Supported Assets

### ✅ Strong Coverage (Confirmed)

| Asset | Type | Best Horizons | Typical Accuracy |
|-------|------|---------------|------------------|
| **SPY** | S&P 500 ETF | 5d, 7d, 10d, 14d, 20d, 30d | 65–75% |
| **QQQ** | Nasdaq-100 ETF | 14d, 20d, 30d | 65–73% |
| **GLD** | Gold ETF | 20d, 30d | 70–89% |
| **VOO** | S&P 500 ETF | 14d, 20d, 30d | 68–73% |
| **VTI** | Total Market ETF | 14d, 20d, 30d | 69–70% |
| **IAU** | Gold ETF | 7d, 10d, 14d, 20d, 30d | 73–89% |
| **DIA** | Dow Jones ETF | 30d only | 78% |

### ⚠️ Selective Coverage (Use with Confidence Filter)

| Asset | Type | Best Horizons | Notes |
|-------|------|---------------|-------|
| **MSFT** | Megacap Tech | 5d, 14d | Selective confirmation only |
| **NVDA** | Semiconductor | 5d, 14d, 20d | Strong at 5d/14d/20d, weaker at 30d |
| **AMZN** | Megacap Tech | 20d, 30d | Mid-horizon band only |
| **GOOGL** | Megacap Tech | 7d only | Narrow confirmation |

### ❌ Weak Coverage (Do Not Rely On)

| Asset | Type | Why |
|-------|------|-----|
| **AAPL** | Single-name equity | Never clears confirmation bar across any horizon |
| **TSLA** | High-volatility equity | Marginal across all horizons; volatility overwhelms signal |
| **BTC-USD** | Crypto | Negative at all horizons; best result 49.2% at 30d |

---

## Confidence Thresholds

The model outputs a **confidence score** (0–1) for each prediction. Use this to trade coverage for accuracy:

| Threshold | Accuracy | Coverage | When to Use |
|-----------|----------|----------|-------------|
| **≥ 0.25** (Recommended) | ~66% | ~44% | Default — balances accuracy and coverage |
| **≥ 0.30** | ~71% | ~30% | When you need higher confidence and can skip more predictions |
| **≥ 0.40** | ~75% | ~15% | Only for high-conviction trades; most predictions filtered out |
| **No threshold** | ~60% | 100% | Full coverage, but includes weak signals |

**What the warnings mean:**
- `⚠️ Low confidence (0.18 < 0.25)` — This prediction is below the recommended threshold; accuracy drops to ~55% in this range. Consider waiting for a higher-confidence signal.

---

## What the Output Means

### Key Metrics

| Metric | What It Is | What to Look For |
|--------|------------|------------------|
| **P(up)** | Probability price goes up | >55% = bullish lean, <45% = bearish lean |
| **90% CI** | 90% confidence interval for horizon price | Honest uncertainty measure; wider = more uncertain |
| **Confidence** | Model's certainty in this prediction | ≥0.25 = recommended threshold |
| **Brier Score** | Calibration quality (lower is better) | <0.35 = well-calibrated |
| **Directional Accuracy** | % of correct direction calls | 60–75% for confirmed assets |

### How to Read the Decision Card

```
📊 Markov Distribution: SPY | Horizon: 14d
Current: $520.50 | Regime: bull
Anchors: 3 trusted | Anchor quality: HIGH
Mixing: 85% Markov / 15% Anchors

🎯 Recommendation: BUY
  P(up): 68% | Expected: $535.20 (+2.8%)
  90% CI: $510.30 – $560.10
  Confidence: 0.32 ✓ (above 0.25 threshold)
```

**Interpretation:**
- **BUY** signal with 68% probability of upside
- **Expected return** +2.8% over 14 days
- **90% CI** shows the honest range of outcomes ($510–$560)
- **Confidence 0.32** is above the 0.25 threshold — this is a reliable signal

---

## Failure Modes (When the Model Is Wrong)

The model fails predictably in these scenarios:

| Scenario | Why | What to Do |
|----------|-----|------------|
| **Earnings week** | No fundamental data; can't capture earnings surprises | Check earnings calendar; reduce position size |
| **Macro shocks** (Fed, CPI) | Trained on 2022–2025 data; may miss regime shifts | Wait 1–2 days post-announcement for regime to stabilize |
| **Crypto volatility** | BTC regime model is anti-informative | Do not use for crypto; use CI coverage only |
| **Low confidence (<0.20)** | Model is uncertain; accuracy drops to ~50% | Skip these predictions; wait for ≥0.25 |
| **45–60d horizons** | `markovWeight` near zero; no Markov signal | Do not use beyond 30d |

---

## Practical Examples

### ✅ Good Use Case

> "I want to know if SPY is likely to be higher in 20 days for a swing trade."

**Verdict:** ✅ Use it — SPY 20d is confirmed (70%+ accuracy), horizon is in sweet spot.

### ⚠️ Marginal Use Case

> "I want a GOOGL 30d forecast for earnings next month."

**Verdict:** ⚠️ Use with caution — GOOGL 30d is exploratory, and earnings week adds fundamental risk. Check confidence score; if <0.25, skip it.

### ❌ Bad Use Case

> "I want BTC 14d direction for a crypto trade."

**Verdict:** ❌ Do not use — BTC is negative at all horizons (49.2% best case).

---

## How to Improve Your Results

1. **Filter by confidence ≥ 0.25** — skips the weakest ~56% of predictions, raises accuracy from ~60% to ~66%
2. **Stick to confirmed assets** — SPY, QQQ, GLD, IAU, VOO, VTI have the most reliable signals
3. **Use 14–30d horizons** — the model's sweet spot; avoid 5–10d (noisy) and 45–60d (no signal)
4. **Combine with fundamentals** — use Cramer-Short's other tools (SEC filings, financial metrics) for earnings/macro context
5. **Respect the CI** — the 90% confidence interval is the honest measure of uncertainty; don't over-interpret point estimates

---

## Evidence Base

All claims are backed by:
- **84 asset × horizon cells** validated via walk-forward backtest
- **160 integration tests** passed (0 failures)
- **Coverage matrix** at `.sisyphus/plans/canonical-coverage-matrix.md`
- **Milestone summary** at `.sisyphus/plans/coverage-milestone-summary.md`

**Commit:** `0fc931b` — record coverage-complete decision (Option A)

---

## What's Next

**Current status:** Coverage-Complete — no automatic expansion planned.

**If you need missing exposures:**
- Small-cap: `IWM` (not yet validated)
- Bonds: `TLT` (not yet validated)
- International: `EFA`, `EEM` (not yet validated)

These would require a separate gap-filling wave with explicit justification.
