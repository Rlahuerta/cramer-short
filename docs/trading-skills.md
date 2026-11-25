# Trading Skills — Practitioner's Guide

**Last Updated:** 2026-11-25  
**Skills:** `swing-trade-setup`, `position-sizing`  
**Integration:** Markov stop-loss, targets and regime from `markov_distribution`

---

## Overview

Cramer-Short includes two trading workflow skills that use the `markov_distribution` tool for stop-loss levels, price targets and the current regime, and size positions by risk: a fixed share of the portfolio divided by the distance to the stop.

| Skill | Use When | Key Output |
|-------|----------|------------|
| **`swing-trade-setup`** | You want to identify high-probability swing trade entries | Entry price, stop-loss, targets, risk/reward, risk-based size |
| **`position-sizing`** | You want to calculate optimal position size for a ticker | Shares to buy, portfolio %, risk per trade, concentration checks |

**Neither skill uses the Markov `predictionConfidence` score.** On the live-config benchmark and 8 held-out tickers (2026-11-25), the score did not predict whether a BUY/SELL call was right: the highest-scored fifth of calls was the least accurate, and no threshold separated better calls from worse ones. Earlier versions of these skills scaled position size by the score and cited accuracy figures for its threshold; those figures did not hold up.

---

## Skill: `swing-trade-setup`

### What It Does

Identifies high-probability swing trade setups with:
- Entry signals (pullback, breakout, breakdown)
- Stop-loss and price targets from Markov distribution
- Risk/reward calculation
- Risk-based position sizing recommendation
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

**With parameters (CLI syntax):**
```
/skills swing-trade-setup NVDA --horizon 20
```

**With parameters (natural language):**
```
Use the swing-trade-setup skill for AAPL with a 20-day horizon
```

### Available Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `horizon` | number | 14 | 5–30 | Markov forecast horizon in trading days |

**When to override defaults:**
- **Longer horizon (20–30d):** For swing trades with wider targets

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

**Entry:**
- Current price: $520.50
- Preferred entry: $515–517 (SMA50 confluence)

**Targets:**
- Target 1: $535 (+2.8%)
- Target 2: $548 (+5.3%)

**Stop-Loss:** $508 (−2.4%)

**Risk/Reward:** 3.1:1

**Position Size:** 1% portfolio risk (standard size)

**Flags:**
- ✓ Regime confluence (bull regime + bullish pullback)
- ✓ Volume drying up on pullback (healthy, no distribution)
- ✓ Markov buyProbability: 0.64
```

#### Setup with Horizon Specification
```
Use the swing-trade-setup skill for QQQ with a 14-day horizon
```

**Why specify horizon:** Markov coverage differs by horizon. QQQ 14d is confirmed (strong), but QQQ 5d is exploratory (weaker).

#### Setup Comparison
```
Compare swing trade setups for SPY vs QQQ vs IAU — which has the best risk/reward?
```

**Use case:** You have limited capital and want to allocate to the best setup.

### When to Use This Skill

✅ **Good use cases:**
- You're considering a swing trade (5–30 day hold)
- You want objective entry/exit levels
- You're reviewing your watchlist for new opportunities

❌ **Bad use cases:**
- Day trading (horizon < 5 days)
- Long-term investing (horizon > 60 days)
- Crypto tickers (BTC-USD is negative at all horizons)
- Earnings week (model has no fundamental data)

### Output Interpretation

| Field | What to Look For |
|-------|-----------------|
| **Risk/Reward** | ≥ 2:1 |
| **Regime confluence** | Bull regime + bullish setup = stronger signal |
| **Volume confirmation** | Breakout volume > 1.5× average = strong; Pullback volume < average = healthy |
| **Markov buyProbability** | > 0.55 for long setups; < 0.45 for short setups |

### Risk-Based Sizing

The skill risks a fixed share of the portfolio per trade (1% by default); the stop distance sets the size:

```
positionSize = risk / downside
```

**Example:** With 1% risk and a stop 2.4% below entry, the position is 1% / 2.4% ≈ 42% of the portfolio in notional value, above the 25% single-position cap, so cap it (see the concentration rules in `position-sizing`).

---

## Skill: `position-sizing`

### What It Does

Calculates optimal position size based on:
- Portfolio risk metrics (VaR, Sharpe, correlation)
- The Markov stop-loss and regime
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

**With parameters (CLI syntax):**
```
/skills position-sizing NVDA --horizon 20 --portfolioValue 75000 --riskPerTrade 0.015
```

**With parameters (natural language):**
```
Use the position-sizing skill for MSFT with a $75,000 portfolio and 1.5% risk per trade
Use the position-sizing skill for GLD with horizon 30 days
```

### Available Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `horizon` | number | 14 | 5–30 | Markov forecast horizon in trading days |
| `portfolioValue` | number | 100000 | 10k–10M | Total portfolio value in USD |
| `riskPerTrade` | number | 0.01 | 0.005–0.03 | Base risk per trade (1% = 0.01) |

**When to override defaults:**
- **`portfolioValue`:** Always set to your actual portfolio value for accurate sizing
- **`riskPerTrade`:** Increase to 0.015–0.02 for an aggressive approach; decrease to 0.005 for a conservative one
- **`horizon`:** Match to your intended hold period (14d standard, 20–30d for wider targets)

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

**Markov Forecast:**
- Regime state: bull
- Buy probability: 61%
- Expected return: +5.2%

**Position Sizing (Risk-Based Method):**
- Entry price: $420.50
- Stop-loss: $395.00 (−6.1% downside)
- Risk per trade: 1.0% (standard size)
- **Recommended position: 19 shares ($7,990)**
- **Position size: 16.0% of portfolio**

**Concentration Check:**
- Correlation with existing holdings: medium (0.52 avg with tech)
- Sector exposure: 32% post-trade (technology)
- Flags: none

**Risk Management:**
- Maximum loss at stop: $485 (1.0% of portfolio)
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
- You're unsure how much to allocate to a ticker
- You want to check concentration risk before adding

❌ **Bad use cases:**
- Day trading (different risk profile)
- Options position sizing (requires Greeks-based calculation)
- Long-term buy-and-hold (use DCF + fundamental conviction instead)

### Output Interpretation

| Field | What to Look For |
|-------|-----------------|
| **Position size %** | 10–20% typical for swing trades; > 25% = concentration risk |
| **Sector exposure** | > 30% in one sector = concentration flag |
| **Correlation** | > 0.7 with existing holdings = reduce size by 30% |
| **Max loss at stop** | Should be ≤ 2% of portfolio for standard trades |

### Sizing Scenarios

The skill provides three scenarios:

| Scenario | When to Use |
|----------|-------------|
| **Conservative** | Portfolio VaR > 15%, or the position adds concentration |
| **Base (recommended)** | Standard swing trade |
| **Aggressive** | 1.5× base risk, only with low correlation to existing holdings |

**Example:**
```
**Sizing Scenarios:**

| Scenario | Shares | Position Value | % Portfolio | Max Loss |
|----------|--------|----------------|-------------|----------|
| Conservative | 12 | $5,046 | 10.1% | $306 |
| Base (recommended) | 19 | $7,990 | 16.0% | $485 |
| Aggressive | 29 | $12,195 | 24.4% | $740 |
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

# Compare risk/reward and regime confluence
```

### Workflow: Portfolio Rebalancing

```
# Get sizing for all watchlist positions
Use the position-sizing skill for each ticker in my watchlist

# Identify over/under-weight positions against the concentration rules
```

---

## Supported Assets

Both skills work best where the Markov tool has **confirmed canonical coverage**:

| Asset | Coverage |
|-------|----------|
| **SPY, QQQ, VOO, VTI** | High — confirmed at 14–30d horizons |
| **GLD, IAU** | High — confirmed at 7–30d horizons |
| **DIA** | Medium — confirmed at 30d only |
| **MSFT, NVDA, GOOGL, AMZN** | Medium — selective horizon confirmation |
| **AAPL, TSLA, BTC-USD** | Low — never clear confirmation bar |

---

## Advanced Examples

### Example 1: Setup Then Sizing
```
Use the swing-trade-setup skill for SPY with a 20-day horizon

# Output shows:
# - Risk/Reward: 3.5:1
# - Regime: bull, confluence: yes

# Follow up with:
Use the position-sizing skill for SPY with a $100,000 portfolio

# Parameterized usage (CLI)
/skills swing-trade-setup SPY --horizon 20
/skills position-sizing SPY --portfolioValue 100000 --riskPerTrade 0.02
```

### Example 2: Concentration Risk Flag
```
Use the position-sizing skill for NVDA with a $50,000 portfolio

# Output shows:
# - Current tech exposure: 35%
# - Post-trade exposure: 48%
# - Flag: "⚠️ Sector concentration > 30% — consider the Conservative size"

# Decision: Use the Conservative scenario instead of Base
```

### Example 3: Multi-Ticker Allocation
```
# Identify setups
Use the swing-trade-setup skill for SPY
Use the swing-trade-setup skill for QQQ
Use the swing-trade-setup skill for GLD

# Size each position with the same risk per trade; the stops set the sizes
Use the position-sizing skill for SPY with $40,000 allocation
Use the position-sizing skill for QQQ with $35,000 allocation
Use the position-sizing skill for GLD with $25,000 allocation
```

### Example 4: Conservative Exploration
```
/skills swing-trade-setup TSLA --horizon 7
/skills position-sizing TSLA --portfolioValue 50000 --riskPerTrade 0.005

# Uses 0.5% base risk (half of standard) for a speculative name
```

---

## Troubleshooting

### "Should I skip trades when Markov `predictionConfidence` is low?"

No: the score did not predict whether calls were right (see the Overview). Judge the setup by risk/reward, regime confluence and volume, and size it by risk.

### "The skill recommends 25% position size — is that too much?"

**Check:**
- Sector concentration (is this ticker in an already-heavy sector?)
- Correlation with existing holdings (> 0.7 = reduce size)
- Portfolio VaR (if > 15%, use Conservative scenario)

**Rule of thumb:** Cap single positions at 20–25% for swing trades.

### "My ticker isn't showing a Markov forecast — why?"

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

- **`swing-trade-setup`:** `src/skills/swing-trade-setup/SKILL.md`
- **`position-sizing`:** `src/skills/position-sizing/SKILL.md`
