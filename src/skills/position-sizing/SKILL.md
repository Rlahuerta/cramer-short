---
name: position-sizing
description: Calculate optimal position size based on portfolio risk, Markov prediction confidence, and individual position risk characteristics.
---

# Position Sizing Skill

Use this skill when the user asks about position sizing, portfolio allocation, or risk management for a specific ticker or portfolio.

## Workflow

### Step 1 — Gather Portfolio Context

Call `portfolio_risk` to understand current portfolio risk characteristics:

```
portfolio_risk({ tickers: [...] })  // user's watchlist or specified tickers
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

**Call `markov_distribution` for confidence-weighted sizing:**

```
markov_distribution({
  ticker: "[TICKER]",
  horizon: 14,
  historicalPrices: [...],
  polymarketMarkets: []
})
```

**Extract:**
- `predictionConfidence` — model decisiveness (0–1)
- `actionSignal.actionLevels.stopLoss` — downside risk level
- `actionSignal.sellProbability` — probability of downside
- `metadata.regimeState` — current regime (bull/bear/sideways)
- `scenarios.expectedReturn` — expected return over horizon

### Step 3 — Calculate Position Risk Parameters

**Downside risk (for position sizing):**
```
downsideRisk = (currentPrice - stopLoss) / currentPrice
```

**Confidence-weighted risk adjustment:**
```
baseConfidence = 0.30  // baseline confidence threshold
confidenceFactor = predictionConfidence / baseConfidence
// Clamp to 0.5× – 2.0× range
confidenceFactor = max(0.5, min(2.0, confidenceFactor))
```

**Interpretation:**
- `predictionConfidence ≥ 0.40` → confidenceFactor ≈ 1.3× (increase size)
- `predictionConfidence 0.25–0.40` → confidenceFactor ≈ 0.8–1.3× (standard size)
- `predictionConfidence < 0.25` → confidenceFactor ≈ 0.5–0.8× (reduce size)

**Regime adjustment:**
- Bear regime → reduce max position size by 20% (defensive stance)
- Sideways regime → reduce by 10% (lower expected follow-through)
- Bull regime → no adjustment

### Step 4 — Apply Position Sizing Method

Use the **Kelly-based fractional sizing** approach (conservative Kelly for swing trades):

**Method A: Risk-based sizing (recommended for most users)**

```
portfolioValue = user's total portfolio value
riskPerTrade = 0.01  // 1% portfolio risk per trade (standard swing trade rule)
downsideRisk = (entryPrice - stopLoss) / entryPrice

positionValue = (portfolioValue × riskPerTrade × confidenceFactor) / downsideRisk
shares = positionValue / entryPrice
positionSizePct = positionValue / portfolioValue × 100
```

**Example:**
- Portfolio: $100,000
- Entry: $100, Stop: $92 (8% downside)
- predictionConfidence: 0.42 → confidenceFactor: 1.4
- Position value = ($100,000 × 0.01 × 1.4) / 0.08 = $17,500
- Shares: 175
- Position size: 17.5% of portfolio

**Method B: Kelly criterion (aggressive, for experienced traders)**

```
winProbability = markov actionSignal.buyProbability (for long) or sellProbability (for short)
lossProbability = 1 - winProbability
winLossRatio = upside / downside  // from risk/reward calculation

kellyFraction = winProbability - (lossProbability / winLossRatio)
fractionalKelly = kellyFraction × 0.5  // half-Kelly for safety
fractionalKelly = max(0, min(0.25, fractionalKelly))  // cap at 25% portfolio

positionValue = portfolioValue × fractionalKelly × confidenceFactor
```

**Recommendation:** Use Method A for most users. Method B can produce aggressive sizes during high-conviction setups — only for experienced traders with strong risk tolerance.

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

### Step 6 — Apply Confidence Threshold Rules

**Hard rules based on Markov predictionConfidence:**

| Confidence | Max Position Size | Risk Per Trade | Notes |
|------------|------------------|----------------|-------|
| ≥ 0.40 (High) | 20% portfolio | 1.5% | High-conviction setups |
| 0.25–0.40 (Medium) | 15% portfolio | 1.0% | Standard setups |
| < 0.25 (Low) | 5% portfolio | 0.5% | Speculative, reduce exposure |

**If predictionConfidence < 0.25:**
- Explicitly warn: "⚠️ Low Markov confidence — accuracy drops to ~55% below 0.25 threshold"
- Recommend waiting for higher-confidence signal OR using minimum position size
- Suggest tighter stop-loss to limit downside

### Step 7 — Build Position Sizing Recommendation

Present in this format:

```
## [TICKER] Position Sizing Recommendation

**Portfolio Context:**
- Portfolio value: $X,XXX
- Current portfolio volatility: XX% (annualized)
- Portfolio VaR (95%): X.X%
- Portfolio Sharpe: X.XX

**Markov Confidence:**
- Prediction confidence: 0.XX ([High/Medium/Low])
- Regime state: [bull/bear/sideways]
- Buy probability: XX%
- Expected return: +X.X%

**Position Sizing (Risk-Based Method):**
- Entry price: $X.XX
- Stop-loss: $X.XX (−X.X% downside)
- Risk per trade: X% (adjusted for confidence)
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

### Step 8 — Provide Sizing Scenarios

Show conservative/base/aggressive options:

```
**Sizing Scenarios:**

| Scenario | Shares | Position Value | % Portfolio | Max Loss |
|----------|--------|----------------|-------------|----------|
| Conservative | XXX | $X,XXX | X% | $XXX |
| Base (recommended) | XXX | $X,XXX | X% | $XXX |
| Aggressive | XXX | $X,XXX | X% | $XXX |

**Recommendation:** Use Base scenario for standard swing trade. Conservative if portfolio VaR > 15% or predictionConfidence < 0.25. Aggressive only if high conviction (confidence ≥ 0.40) and low correlation with existing holdings.
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

**Markov Confidence:**
- Prediction confidence: 0.42 (High)
- Regime state: bull
- Buy probability: 64%
- Expected return: +8.2%

**Position Sizing (Risk-Based Method):**
- Entry price: $118.50
- Stop-loss: $109.00 (−8.0% downside)
- Risk per trade: 1.5% (high confidence → 1.5× standard)
- **Recommended position: 190 shares ($22,515)**
- **Position size: 22.5% of portfolio**

**Concentration Check:**
- Correlation with existing holdings: medium (0.55 avg with tech positions)
- Sector exposure: 35% post-trade (technology)
- Flags: ⚠️ Sector concentration > 30% — consider reducing to 18% position size

**Alternative (Kelly Method):**
- Kelly fraction: 18%
- Half-Kelly: 9%
- **Kelly-based position: 95 shares ($11,258)**

**Risk Management:**
- Maximum loss at stop: $1,805 (1.8% of portfolio)
- Position beta: 1.65 (vs. S&P 500)
- Contribution to portfolio VaR: +0.6%

**Sizing Scenarios:**

| Scenario | Shares | Position Value | % Portfolio | Max Loss |
|----------|--------|----------------|-------------|----------|
| Conservative | 95 | $11,258 | 11.3% | $901 |
| Base (recommended) | 140 | $16,590 | 16.6% | $1,327 |
| Aggressive | 190 | $22,515 | 22.5% | $1,805 |

**Recommendation:** Use Conservative scenario due to sector concentration. Technology sector already at 35% — adding full position would bring to 48%, creating single-sector risk. If comfortable with concentration, Base scenario is acceptable given high Markov confidence (0.42) and bull regime.
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
- Combine with `portfolio-risk` for full portfolio optimization
- Combine with `dcf` for fundamental conviction overlay
