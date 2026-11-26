---
name: position-sizing
description: Calculate optimal position size based on portfolio risk, the Markov stop-loss and regime, and individual position risk characteristics.
parameters:
  horizon:
    type: number
    description: "Markov forecast horizon in trading days"
    default: 14
    min: 5
    max: 30
  portfolioValue:
    type: number
    description: "Total portfolio value in USD (e.g. 100000 for $100k)"
    default: 100000
    min: 10000
    max: 10000000
  riskPerTrade:
    type: number
    description: "Base risk per trade as decimal (e.g. 0.01 for 1%)"
    default: 0.01
    min: 0.005
    max: 0.03
---

# Position Sizing Skill

Use this skill when the user asks about position sizing, portfolio allocation, or risk management for a specific ticker or portfolio.

## Workflow

### Step 1 — Gather Portfolio Context

Call `portfolio_risk` to understand current portfolio risk characteristics:

```
portfolio_risk({ tickers: [...] })  // specified tickers
portfolio_risk({ watchlist_entries: [...] })  // user's watchlist entries from context
```

**Extract:**
- Current portfolio volatility (annualized)
- Portfolio VaR (95%) and CVaR (95%)
- Portfolio Sharpe ratio
- Correlation matrix (identify concentration risk)
- Max drawdown

**If user didn't specify tickers:** Ask for watchlist or portfolio holdings first.

### Step 2 — Fetch Individual Position Risk

For the target ticker, gather risk metrics:

**Call `get_market_data`:**
- Current price, 52-week range
- Beta (vs. S&P 500)
- Average volume, market cap

**Call `markov_distribution` for the stop-loss level and regime:**

```
markov_distribution({
  ticker: "[TICKER]",
  horizon: {{horizon}},
  historicalPrices: [...],
  polymarketMarkets: []
})
```

**Active parameters:** `horizon = {{horizon}}`, `portfolioValue = {{portfolioValue}}`, `riskPerTrade = {{riskPerTrade}}`

**Extract:**
- `actionSignal.actionLevels.stopLoss` — downside risk level
- `actionSignal.sellProbability` — probability of downside
- `metadata.regimeState` — current regime (bull/bear/sideways)
- `scenarios.expectedReturn` — expected return over horizon

Do not scale the size by `predictionConfidence`: in backtests (2026-11-25) the score did not predict whether a forecast was right, and the highest-scored fifth was the least accurate.

### Step 3 — Calculate Position Risk Parameters

**Downside risk (for position sizing):**
```
downsideRisk = (currentPrice - stopLoss) / currentPrice
```

**Regime adjustment:**
- Bear regime → reduce max position size by 20% (defensive stance)
- Sideways regime → reduce by 10% (lower expected follow-through)
- Bull regime → no adjustment

### Step 4 — Apply Position Sizing Method

Use the **Kelly-based fractional sizing** approach (conservative Kelly for swing trades):

**Method A: Risk-based sizing (recommended for most users)**

```
portfolioValue = {{portfolioValue}}  // active parameter
riskPerTrade = {{riskPerTrade}}  // active parameter (e.g. 0.01 for 1%)
downsideRisk = (entryPrice - stopLoss) / entryPrice

positionValue = (portfolioValue × riskPerTrade) / downsideRisk
shares = positionValue / entryPrice
positionSizePct = positionValue / portfolioValue × 100
```

**Example:**
- Portfolio: ${{portfolioValue}}
- Entry: $100, Stop: $92 (8% downside)
- Position value = (${{portfolioValue}} × {{riskPerTrade}}) / 0.08 = $12,500 at the default parameters
- Shares: 125
- Position size: 12.5% of portfolio

**Method B: Kelly criterion (aggressive, for experienced traders)**

```
winProbability = diagnostics.callTrackRecord.hitRate   // the measured hit rate of this call; skip Method B if it is null
lossProbability = 1 - winProbability
winLossRatio = upside / downside  // from risk/reward calculation

kellyFraction = winProbability - (lossProbability / winLossRatio)
fractionalKelly = kellyFraction × 0.5  // half-Kelly for safety
fractionalKelly = max(0, min(0.25, fractionalKelly))  // cap at 25% portfolio

positionValue = portfolioValue × fractionalKelly
```

Do not plug in `buyProbability`/`sellProbability`: they are the model's estimates and did not beat the market's own up/down share over 2021–2025. The track record's hit rate sits close to its base rate, so Kelly here sizes a bet on the market's drift, not on a model edge; say so when you show it.

**Recommendation:** Use Method A for most users. Method B can produce aggressive sizes — only for experienced traders with strong risk tolerance.

### Step 5 — Check Portfolio Concentration

Before finalizing recommendation, check for concentration risk:

**Call `portfolio_risk` correlation matrix:**
- If new position correlates > 0.7 with existing holdings → flag concentration risk
- If portfolio already has > 20% in same sector → flag sector concentration
- If single position would exceed 25% of portfolio → flag overconcentration

**Adjustments:**
- High correlation (> 0.7) with existing position → reduce size by 30%
- Sector concentration (> 20%) → reduce size by 20%
- Single position > 25% → cap at 25% maximum

### Step 6 — Build Position Sizing Recommendation

Present in this format:

```
## [TICKER] Position Sizing Recommendation

**Portfolio Context:**
- Portfolio value: $X,XXX
- Current portfolio volatility: XX% (annualized)
- Portfolio VaR (95%): X.X%
- Portfolio Sharpe: X.XX

**Markov Forecast:**
- Regime state: [bull/bear/sideways]
- Buy probability: XX%
- Expected return: +X.X%

**Position Sizing (Risk-Based Method):**
- Entry price: $X.XX
- Stop-loss: $X.XX (−X.X% downside)
- Risk per trade: X%
- **Recommended position: X shares ($X,XXX)**
- **Position size: X.X% of portfolio**

**Concentration Check:**
- Correlation with existing holdings: [low/medium/high]
- Sector exposure: X% (post-trade)
- Flags: [none / concentration warning]

**Alternative (Kelly Method):**
- Kelly fraction: X.X%
- Half-Kelly: X.X%
- **Kelly-based position: X shares ($X,XXX)**

**Risk Management:**
- Maximum loss at stop: $X,XXX (X% of portfolio)
- Position beta: X.XX (vs. S&P 500)
- Contribution to portfolio VaR: +X.X%
```

### Step 7 — Provide Sizing Scenarios

Show conservative/base/aggressive options:

```
**Sizing Scenarios:**

| Scenario | Shares | Position Value | % Portfolio | Max Loss |
|----------|--------|----------------|-------------|----------|
| Conservative | XXX | $X,XXX | X% | $XXX |
| Base (recommended) | XXX | $X,XXX | X% | $XXX |
| Aggressive | XXX | $X,XXX | X% | $XXX |

**Recommendation:** Use Base scenario for standard swing trade. Conservative if portfolio VaR > 15% or the position adds concentration. Aggressive (1.5× base risk) only with low correlation to existing holdings.
```

---

## Example Output

```
## NVDA Position Sizing Recommendation

**Portfolio Context:**
- Portfolio value: $100,000
- Current portfolio volatility: 18% (annualized)
- Portfolio VaR (95%): 2.8%
- Portfolio Sharpe: 1.2

**Markov Forecast:**
- Regime state: bull
- Buy probability: 64%
- Expected return: +8.2%

**Position Sizing (Risk-Based Method):**
- Entry price: $118.50
- Stop-loss: $109.00 (−8.0% downside)
- Risk per trade: 1.0%
- **Recommended position: 105 shares ($12,443)**
- **Position size: 12.4% of portfolio**

**Concentration Check:**
- Correlation with existing holdings: medium (0.55 avg with tech positions)
- Sector exposure: 35% post-trade (technology)
- Flags: ⚠️ Sector concentration > 30% — consider the Conservative size

**Alternative (Kelly Method):**
- Kelly fraction: 18%
- Half-Kelly: 9%
- **Kelly-based position: 95 shares ($11,258)**

**Risk Management:**
- Maximum loss at stop: $998 (1.0% of portfolio)
- Position beta: 1.65 (vs. S&P 500)
- Contribution to portfolio VaR: +0.6%

**Sizing Scenarios:**

| Scenario | Shares | Position Value | % Portfolio | Max Loss |
|----------|--------|----------------|-------------|----------|
| Conservative | 95 | $11,258 | 11.3% | $901 |
| Base (recommended) | 105 | $12,443 | 12.4% | $998 |
| Aggressive | 158 | $18,723 | 18.7% | $1,501 |

**Recommendation:** Use Conservative scenario due to sector concentration. Technology sector already at 35% — adding full position would bring to 48%, creating single-sector risk. If comfortable with concentration, Base scenario is acceptable in the bull regime.
```

---

## Notes

**When to use this skill:**
- User asks "how much should I buy of [TICKER]?"
- User is sizing a new position
- User wants to rebalance based on conviction levels
- User asks about portfolio allocation or risk management

**Do NOT use for:**
- Day trading position sizing (different risk profile)
- Options position sizing (requires separate Greeks-based calculation)
- Long-term buy-and-hold allocation (use DCF + fundamental conviction instead)

**Integration with other skills:**
- Combine with `swing-trade-setup` for entry-specific sizing
- Combine with `portfolio_risk` for full portfolio optimization
- Combine with `dcf-valuation` for fundamental conviction overlay
