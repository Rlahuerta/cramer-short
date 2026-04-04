# Trading Skills — Practitioner's Guide

**Last Updated:** 2026-04-04  
**New Skills:** `swing-trade-setup`, `position-sizing`  
**Integration:** Markov Chain prediction confidence (threshold: 0.25)

---

## Overview

Cramer-Short includes two new trading workflow skills that integrate **Markov Chain prediction confidence** into entry signals and position sizing decisions. These skills use the `markov_distribution` tool to produce confidence-weighted recommendations.

| Skill | Use When | Key Output |
|-------|----------|------------|
| **`swing-trade-setup`** | You want to identify high-probability swing trade entries | Entry price, stop-loss, targets, risk/reward, confidence-weighted size |
| **`position-sizing`** | You want to calculate optimal position size for a ticker | Shares to buy, portfolio %, confidence-adjusted risk, concentration checks |

Both skills use the **Markov confidence threshold of 0.25** to separate high/low-confidence regimes:
- **≥ 0.25**: Recommended threshold — ~66% accuracy at ~44% coverage
- **< 0.25**: Low confidence — accuracy drops to ~55%; reduce size or skip

---

## Skill: `swing-trade-setup`

### What It Does

Identifies high-probability swing trade setups with:
- Entry signals (pullback, breakout, breakdown)
- Stop-loss and price targets from Markov distribution
- Risk/reward calculation
- Confidence-weighted position sizing recommendation
- Volume and momentum confirmation checks

### How to Invoke

**Via `/skills` command:**
```
/skills swing-trade-setup
```

Then provide the ticker when prompted.

**Direct prompt:**
```
Use the swing-trade-setup skill for NVDA
```

**With parameters:**
```
Use the swing-trade-setup skill for AAPL with a 20-day horizon
```

### Example Prompts

#### Basic Setup Analysis
```
Use the swing-trade-setup skill for SPY
```

**Expected output:**
```
## SPY Swing Trade Setup

**Setup Type:** Pullback to SMA50 support
**Direction:** Long
**Confidence:** High (predictionConfidence: 0.42)

**Entry:**
- Current price: $520.50
- Preferred entry: $515–517 (SMA50 confluence)

**Targets:**
- Target 1: $535 (+2.8%)
- Target 2: $548 (+5.3%)

**Stop-Loss:** $508 (−2.4%)

**Risk/Reward:** 3.1:1

**Position Size:** 1.8% portfolio risk (high confidence → 1.2× standard size)

**Flags:**
- ✓ Regime confluence (bull regime + bullish pullback)
- ✓ Volume drying up on pullback (healthy, no distribution)
- ✓ Markov buyProbability: 0.64
```

#### Setup with Horizon Specification
```
Use the swing-trade-setup skill for QQQ with a 14-day horizon
```

**Why specify horizon:** Different horizons may have different confidence levels. QQQ 14d is confirmed (strong), but QQQ 5d is exploratory (weaker).

#### Setup Comparison
```
Compare swing trade setups for SPY vs QQQ vs IAU — which has the highest Markov confidence?
```

**Use case:** You have limited capital and want to allocate to the highest-conviction setup.

### When to Use This Skill

✅ **Good use cases:**
- You're considering a swing trade (5–30 day hold)
- You want objective entry/exit levels
- You want to know if Markov confidence supports the trade
- You're reviewing your watchlist for new opportunities

❌ **Bad use cases:**
- Day trading (horizon < 5 days)
- Long-term investing (horizon > 60 days)
- Crypto tickers (BTC-USD is negative at all horizons)
- Earnings week (model has no fundamental data)

### Output Interpretation

| Field | What to Look For |
|-------|-----------------|
| **Confidence** | ≥ 0.25 is recommended threshold; < 0.25 = reduce size or skip |
| **Risk/Reward** | ≥ 2:1 for standard setups; ≥ 3:1 for low-confidence setups |
| **Regime confluence** | Bull regime + bullish setup = stronger signal |
| **Volume confirmation** | Breakout volume > 1.5× average = strong; Pullback volume < average = healthy |
| **Markov buyProbability** | > 0.55 for long setups; < 0.45 for short setups |

### Confidence-Weighted Sizing

The skill automatically adjusts position size based on Markov confidence:

| Confidence | Size Adjustment | When to Use |
|------------|----------------|-------------|
| ≥ 0.40 (High) | 1.2–1.5× standard | High-conviction setups |
| 0.25–0.40 (Medium) | 1.0× standard | Standard setups |
| < 0.25 (Low) | 0.5–0.8× standard | Speculative; reduce exposure |

**Example:** If your standard risk is 1% per trade:
- High confidence (0.42): Risk 1.5%
- Medium confidence (0.32): Risk 1.0%
- Low confidence (0.18): Risk 0.5% or skip

---

## Skill: `position-sizing`

### What It Does

Calculates optimal position size based on:
- Portfolio risk metrics (VaR, Sharpe, correlation)
- Markov prediction confidence
- Individual position risk (downside from stop-loss)
- Concentration risk checks
- Kelly-based and risk-based sizing methods

### How to Invoke

**Via `/skills` command:**
```
/skills position-sizing
```

Then provide the ticker and portfolio value when prompted.

**Direct prompt:**
```
Use the position-sizing skill for NVDA with a $100,000 portfolio
```

**With existing watchlist:**
```
Use the position-sizing skill for AAPL using my watchlist
```

### Example Prompts

#### Basic Position Sizing
```
Use the position-sizing skill for MSFT with a $50,000 portfolio
```

**Expected output:**
```
## MSFT Position Sizing Recommendation

**Portfolio Context:**
- Portfolio value: $50,000
- Current portfolio volatility: 16% (annualized)
- Portfolio VaR (95%): 2.4%
- Portfolio Sharpe: 1.1

**Markov Confidence:**
- Prediction confidence: 0.38 (Medium)
- Regime state: bull
- Buy probability: 61%
- Expected return: +5.2%

**Position Sizing (Risk-Based Method):**
- Entry price: $420.50
- Stop-loss: $395.00 (−6.1% downside)
- Risk per trade: 1.0% (standard size)
- **Recommended position: 80 shares ($33,640)**
- **Position size: 13.4% of portfolio**

**Concentration Check:**
- Correlation with existing holdings: medium (0.52 avg with tech)
- Sector exposure: 32% post-trade (technology)
- Flags: none

**Risk Management:**
- Maximum loss at stop: $2,040 (4.1% of portfolio)
- Position beta: 1.15 (vs. S&P 500)
- Contribution to portfolio VaR: +0.4%
```

#### Sizing with Concentration Concerns
```
Use the position-sizing skill for NVDA — I already have 25% in tech stocks
```

**Expected output:** Will flag sector concentration and recommend reducing size.

#### Kelly vs. Risk-Based Comparison
```
Use the position-sizing skill for GLD and show both Kelly and risk-based methods
```

**Use case:** You want to compare aggressive (Kelly) vs. conservative (risk-based) sizing.

### When to Use This Skill

✅ **Good use cases:**
- You're sizing a new swing trade position
- You want to rebalance based on conviction levels
- You're unsure how much to allocate to a ticker
- You want to check concentration risk before adding

❌ **Bad use cases:**
- Day trading (different risk profile)
- Options position sizing (requires Greeks-based calculation)
- Long-term buy-and-hold (use DCF + fundamental conviction instead)

### Output Interpretation

| Field | What to Look For |
|-------|-----------------|
| **Markov Confidence** | ≥ 0.25 recommended; < 0.25 = reduce size |
| **Position size %** | 10–20% typical for swing trades; > 25% = concentration risk |
| **Sector exposure** | > 30% in one sector = concentration flag |
| **Correlation** | > 0.7 with existing holdings = reduce size by 30% |
| **Max loss at stop** | Should be ≤ 2% of portfolio for standard trades |

### Sizing Scenarios

The skill provides three scenarios:

| Scenario | When to Use |
|----------|-------------|
| **Conservative** | Portfolio VaR > 15%, or predictionConfidence < 0.25, or high correlation |
| **Base (recommended)** | Standard swing trade with medium confidence (0.25–0.40) |
| **Aggressive** | High conviction (confidence ≥ 0.40) and low correlation |

**Example:**
```
**Sizing Scenarios:**

| Scenario | Shares | Position Value | % Portfolio | Max Loss |
|----------|--------|----------------|-------------|----------|
| Conservative | 60 | $25,230 | 12.6% | $1,530 |
| Base (recommended) | 90 | $37,845 | 18.9% | $2,295 |
| Aggressive | 120 | $50,460 | 25.2% | $3,060 |
```

---

## Integration: Using Both Skills Together

### Workflow: Setup → Sizing

1. **Run `swing-trade-setup`** to identify the setup and get initial size recommendation
2. **Run `position-sizing`** to validate against your full portfolio context

**Example:**
```
# Step 1: Identify setup
Use the swing-trade-setup skill for AMZN

# Step 2: Validate sizing
Use the position-sizing skill for AMZN with a $75,000 portfolio
```

### Workflow: Compare Multiple Setups

```
# Run setup analysis for multiple tickers
Use the swing-trade-setup skill for SPY
Use the swing-trade-setup skill for QQQ
Use the swing-trade-setup skill for IAU

# Compare confidence scores and allocate to highest
```

### Workflow: Portfolio Rebalancing

```
# Get sizing for all watchlist positions
Use the position-sizing skill for each ticker in my watchlist

# Identify over/under-weight positions
# Rebalance toward higher Markov confidence names
```

---

## Markov Confidence Threshold Guide

Both skills use the **0.25 confidence threshold** as a key decision point:

### What the Threshold Means

| Confidence | Accuracy | Coverage | Action |
|------------|----------|----------|--------|
| ≥ 0.40 | ~75% | ~15% | High-conviction; increase size |
| 0.25–0.40 | ~66% | ~44% | Standard; use base size |
| < 0.25 | ~55% | ~56% | Low-conviction; reduce size or skip |

### How Skills Use the Threshold

**`swing-trade-setup`:**
- Adjusts position size multiplier (0.5× to 1.5×)
- Adds warning flag if < 0.25
- Recommends tighter stop-loss for low-confidence setups

**`position-sizing`:**
- Applies confidence factor to risk-per-trade calculation
- Caps max position size at 5% for confidence < 0.25
- Shows explicit warning: "⚠️ Low Markov confidence — accuracy drops to ~55%"

### Supported Assets

Both skills work best with **confirmed canonical coverage**:

| Asset | Confidence Reliability |
|-------|----------------------|
| **SPY, QQQ, VOO, VTI** | High — confirmed at 14–30d horizons |
| **GLD, IAU** | High — confirmed at 7–30d horizons |
| **DIA** | Medium — confirmed at 30d only |
| **MSFT, NVDA, GOOGL, AMZN** | Medium — selective horizon confirmation |
| **AAPL, TSLA, BTC-USD** | Low — never clear confirmation bar |

---

## Advanced Examples

### Example 1: High-Conviction Setup
```
Use the swing-trade-setup skill for SPY with a 20-day horizon

# Output shows:
# - predictionConfidence: 0.45 (High)
# - Risk/Reward: 3.5:1
# - Regime: bull, confluence: yes

# Follow up with:
Use the position-sizing skill for SPY with a $100,000 portfolio

# Output shows:
# - Recommended: 2.0% risk (1.5× standard due to high confidence)
# - Position: 38 shares ($19,760)
```

### Example 2: Low-Confidence Setup (Skip or Reduce)
```
Use the swing-trade-setup skill for BTC-USD with a 14-day horizon

# Output shows:
# - predictionConfidence: 0.15 (Low)
# - Warning: "⚠️ Low Markov confidence — accuracy drops to ~55%"
# - Recommendation: Reduce size to 0.5× or skip

# Decision: Skip the trade or use minimum size
```

### Example 3: Concentration Risk Flag
```
Use the position-sizing skill for NVDA with a $50,000 portfolio

# Output shows:
# - Current tech exposure: 35%
# - Post-trade exposure: 48%
# - Flag: "⚠️ Sector concentration > 30% — consider reducing to 18% position size"

# Decision: Use Conservative scenario (11% position) instead of Base (22%)
```

### Example 4: Multi-Ticker Allocation
```
# Identify best setups
Use the swing-trade-setup skill for SPY
Use the swing-trade-setup skill for QQQ
Use the swing-trade-setup skill for GLD

# Allocate capital to highest confidence
# SPY: 0.42, QQQ: 0.38, GLD: 0.35
# Split capital: 40% SPY, 35% QQQ, 25% GLD

# Size each position
Use the position-sizing skill for SPY with $40,000 allocation
Use the position-sizing skill for QQQ with $35,000 allocation
Use the position-sizing skill for GLD with $25,000 allocation
```

---

## Troubleshooting

### "Markov confidence is low (< 0.25) — what should I do?"

**Options:**
1. **Skip the trade** — wait for higher-confidence signal
2. **Reduce size** — use 0.5× standard position size
3. **Tighten stop-loss** — reduce downside risk to compensate
4. **Shorten horizon** — try 7d or 14d instead of 30d (or vice versa)

### "The skill recommends 25% position size — is that too much?"

**Check:**
- Sector concentration (is this ticker in an already-heavy sector?)
- Correlation with existing holdings (> 0.7 = reduce size)
- Portfolio VaR (if > 15%, use Conservative scenario)

**Rule of thumb:** Cap single positions at 20–25% for swing trades.

### "My ticker isn't showing Markov confidence — why?"

**Possible reasons:**
- Ticker not in canonical coverage (e.g., small-cap, international)
- Insufficient price history (< 120 days)
- API failure (check `FINANCIAL_DATASETS_API_KEY`)

**Workaround:** Use the skill without Markov integration — it will fall back to technical analysis only.

---

## Related Documentation

- **Markov Model Guide:** [`markov-prediction-guide.md`](markov-prediction-guide.md)
- **When to Use Markov:** [`markov-when-to-use.md`](markov-when-to-use.md)
- **Watchlist Commands:** [`watchlist.md`](watchlist.md)

---

## Skill Files

- **`swing-trade-setup`:** `/data/Repositories/dexter/src/skills/swing-trade-setup/SKILL.md`
- **`position-sizing`:** `/data/Repositories/dexter/src/skills/position-sizing/SKILL.md`

**Commit:** `b35d6da` — "Integrate Markov confidence into watchlist and add trading skills"
