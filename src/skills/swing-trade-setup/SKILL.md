---
name: swing-trade-setup
description: Identify high-probability swing trade setups with entry signals, stop-loss levels, and risk-based position sizing.
parameters:
  horizon:
    type: number
    description: "Forecast horizon in trading days"
    default: 14
    min: 5
    max: 30
---

# Swing Trade Setup Skill

Use this skill when the user asks about swing trade opportunities, entry points, or setup analysis for a specific ticker.

## Workflow

### Step 1 — Fetch Market Context

Call `get_market_data` to gather current price action and trend context:

**Query:** `"[TICKER] current price 52-week range moving averages volume trend"`

**Extract:**
- Current price and day change %
- 52-week high/low (distance from each)
- SMA20, SMA50, SMA200 (price vs. moving averages)
- Volume vs. average (unusual activity flag)

### Step 2 — Technical Setup Analysis

Identify the setup pattern from price action:

**Bullish setups:**
- Pullback to support (SMA20/50, prior resistance turned support)
- Breakout above consolidation (range-bound → breakout)
- Golden cross (SMA50 crosses above SMA200)
- Higher lows + higher highs (uptrend confirmation)

**Bearish setups:**
- Breakdown below support
- Lower highs + lower lows (downtrend)
- Death cross (SMA50 crosses below SMA200)

**Output:** Setup type (e.g., "pullback to SMA50 support", "breakout above $X consolidation")

### Step 3 — Markov Distribution Forecast

Call `markov_distribution` to get regime-aware probability distribution:

```
markov_distribution({
  ticker: "[TICKER]",
  horizon: {{horizon}},
  historicalPrices: [...],  // from get_market_data or fetchHistoricalPrices
  polymarketMarkets: []     // omit if unavailable
})
```

**Extract from result:**
- `regimeState` — current regime (bull/bear/sideways)
- `actionSignal.recommendation` — BUY/HOLD/SELL
- `actionSignal.buyProbability` / `sellProbability`
- `actionSignal.actionLevels.targetPrice` — upside target
- `actionSignal.actionLevels.stopLoss` — downside stop

Do not grade the setup or the size by `predictionConfidence`: in backtests (2026-11-25) the score did not predict whether a forecast was right, and the highest-scored fifth was the least accurate.

### Step 4 — Entry Signal Confirmation

Check for confluence signals:

**Volume confirmation:**
- Breakout volume > 1.5× average = strong signal
- Pullback volume < average = healthy (no distribution)

**Momentum confirmation:**
- RSI 40–60 = neutral room to run
- RSI > 70 = overbought (caution on long entries)
- RSI < 30 = oversold (caution on short entries)

**Markov confirmation:**
- Regime state matches setup direction (bull regime + bullish setup = confluence)
- `buyProbability > 0.55` for long setups, `sellProbability > 0.45` for short setups

### Step 5 — Risk/Reward Calculation

Compute setup quality:

```
entryPrice = current price (or limit order at support level)
targetPrice = from Markov actionLevels.targetPrice
stopLoss = from Markov actionLevels.stopLoss (or technical level: recent swing low/high)

upside = (targetPrice - entryPrice) / entryPrice
downside = (entryPrice - stopLoss) / entryPrice
riskReward = upside / downside
```

**Minimum acceptable:** Risk/reward ≥ 2:1

### Step 6 — Position Sizing Recommendation

Risk a fixed share of the portfolio per trade; the stop distance sets the size:

```
risk = 1% of portfolio (standard swing trade risk)
positionSize = risk / downside
```

### Step 7 — Build Setup Summary

Present in this format:

```
## [TICKER] Swing Trade Setup

**Setup Type:** [pullback/breakout/breakdown/etc.]
**Direction:** Long / Short

**Entry:**
- Current price: $X.XX
- Preferred entry: $X.XX (limit at support/resistance)

**Targets:**
- Target 1: $X.XX (+X%)
- Target 2: $X.XX (+X%)

**Stop-Loss:** $X.XX (−X%)

**Risk/Reward:** X:1

**Position Size:** X% portfolio risk

**Catalysts:** [earnings date, sector tailwinds, etc.]

**Flags:**
- ✓ Regime confluence (bull regime + bullish setup)
- ✓ Volume confirmation (breakout on 2.1× average volume)
```

### Step 8 — Exit Strategy

Define exit rules:

**Profit-taking:**
- Scale out 50% at Target 1, trail remainder
- Full exit if Markov regime flips (bull → bear or vice versa)

**Stop-loss:**
- Hard stop at stopLoss level
- OR: Mental stop if setup thesis invalidated (e.g., breakdown below support on high volume)

**Time-based exit:**
- If no follow-through in 5–10 trading days, re-evaluate
- Markov horizon is 14 days — exit or adjust if thesis hasn't played out

---

## Example Output

```
## NVDA Swing Trade Setup

**Setup Type:** Pullback to SMA50 support
**Direction:** Long

**Entry:**
- Current price: $118.50
- Preferred entry: $116–117 (SMA50 confluence)

**Targets:**
- Target 1: $128 (+9.8%)
- Target 2: $135 (+15.6%)

**Stop-Loss:** $109 (−8.1%)

**Risk/Reward:** 3.2:1

**Position Size:** 1% portfolio risk (standard size)

**Catalysts:** Earnings in 12 days, AI sector momentum

**Flags:**
- ✓ Regime confluence (bull regime + bullish pullback)
- ✓ Volume drying up on pullback (healthy, no distribution)
- ✓ Markov buyProbability: 0.64

**Exit Strategy:**
- Scale 50% at $128, trail to $122
- Hard stop at $109
- Re-evaluate if no follow-through in 7 days
```

---

**Do not run full analysis unless explicitly requested.** This skill provides setup identification and risk parameters — deeper fundamental analysis (DCF, peer comparison) requires separate skills.
