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
- ✅ The run returns a **calibrated Markov distribution** (a non-abstain action signal)

**Do NOT use this model when:**
- ❌ You trade **crypto** (BTC-USD is negative at all horizons; any broader commentary the agent provides is fallback, not Markov output)
- ❌ You need **<7d** or **>30d** forecasts (5–10d is noisy, 45–60d has no Markov signal)
- ❌ You trade **single-name equities** expecting reliable signals (AAPL/TSLA never clear confirmation bar)
- ❌ The run **abstains or returns no calibrated distribution** — treat that as a skip, not edge
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
| **BTC-USD** | Crypto | Negative at all horizons; best result 49.2% at 30d. Any broader trade commentary the agent provides for BTC is fallback, not validated Markov output. |

---

## The Confidence Score

The model outputs a **`predictionConfidence` score** (0–1) for each prediction. **Do not filter or size trades by it.** On the live-config benchmark and 8 held-out tickers (2026-11-25) the score did not predict whether a BUY/SELL call was right: the highest-scored fifth of calls was the least accurate, and no threshold separated better calls from worse ones. The model's calls were also right slightly less often than always calling "up" on the same days (59.6% vs 61.2%), so the overall accuracy mostly reflects the 2024–25 bull market. The tool no longer warns below a confidence threshold.

Earlier versions of this guide listed accuracy and coverage figures per threshold (measured 2026-04-04); they did not hold up on live-length windows.

**What the warnings mean:**
- **Abstain or no calibrated distribution** — The model declined to produce a calibrated result (typically because trusted anchors are missing or the regime signal is too weak). This is not an edge. Skip the trade rather than interpreting fallback text as a signal.

---

## What the Output Means

### Key Metrics

| Metric | What It Is | What to Look For |
|--------|------------|------------------|
| **P(up)** | Probability price goes up | >55% = bullish lean, <45% = bearish lean |
| **90% CI** | 90% confidence interval for horizon price | Honest uncertainty measure; wider = more uncertain |
| **Confidence** | Model-agreement score (0–1) | Not a reliability signal: it did not predict accuracy (2026-11-25) |
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
  Confidence: 0.32
```

**Interpretation:**
- **BUY** signal with 68% probability of upside
- **Expected return** +2.8% over 14 days
- **90% CI** shows the honest range of outcomes ($510–$560)

---

## Failure Modes (When the Model Is Wrong)

The model fails predictably in these scenarios:

| Scenario | Why | What to Do |
|----------|-----|------------|
| **Earnings week** | No fundamental data; can't capture earnings surprises | Check earnings calendar; reduce position size |
| **Macro shocks** (Fed, CPI) | Trained on 2022–2025 data; may miss regime shifts | Wait 1–2 days post-announcement for regime to stabilize |
| **Crypto volatility** | BTC regime model is anti-informative | Do not use for crypto; any commentary is fallback, not Markov |
| **Abstain / no calibrated distribution** | Trusted anchors missing or regime signal too weak | Skip. Do not treat absence of output or fallback text as a signal |
| **45–60d horizons** | `markovWeight` near zero; no Markov signal | Do not use beyond 30d |

---

## Practical Examples

### ✅ Good Use Case

> "I want to know if SPY is likely to be higher in 20 days for a swing trade."

**Verdict:** ✅ Use it — SPY 20d is confirmed (70%+ accuracy), horizon is in sweet spot. If the run returns a calibrated distribution, act on it. If it abstains or returns no calibrated output, skip.

### ⚠️ Marginal Use Case

> "I want a GOOGL 30d forecast for earnings next month."

**Verdict:** ⚠️ Use with caution — GOOGL only has confirmed coverage at 7d; 30d is exploratory, and earnings week adds fundamental risk. If the model abstains, treat it as a skip rather than a signal.

### ❌ Bad Use Case

> "I want BTC 14d direction for a crypto trade."

**Verdict:** ❌ Do not use — BTC is negative at all horizons (49.2% best case). Any broader trade commentary the agent may still provide for BTC is fallback reasoning, not validated Markov output. Do not rely on it as a Markov edge.

---

## How to Improve Your Results

1. **Stick to confirmed assets** — SPY, QQQ, GLD, IAU, VOO, VTI have the most reliable signals
2. **Use 14–30d horizons** — the model's sweet spot; avoid 5–10d (noisy) and 45–60d (no signal)
3. **Combine with fundamentals** — use Cramer-Short's other tools (SEC filings, financial metrics) for earnings/macro context
4. **Respect the CI** — the 90% confidence interval is the honest measure of uncertainty; don't over-interpret point estimates

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
