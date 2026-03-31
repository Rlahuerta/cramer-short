# Markov Chain Probability Distribution for Asset Price Forecasting

> **Design approved:** 2026-03-31
> **Approach:** Polymarket-First Distribution with Markov Interpolation
> **Integration:** Enhancement to existing `probability-assessment` skill

---

## Overview

This design adds a Markov chain-based probability distribution generator to Dexter. It produces full probability distributions for stock/ETF prices at user-specified horizons (1-90 days) by combining:

1. **Polymarket threshold markets** — Real-money probability anchors at specific price levels
2. **Historical regime transitions** — Markov transition matrix estimated from 60-90 days of price history
3. **Sentiment adjustments** — Dynamic multipliers from social sentiment signals

---

## Architecture

```
User Query: "What's the probability distribution for NVDA in 30 days?"
                              |
                              v
              +-------------------------------+
              | Step 1: Asset Detection       |
              | Extract ticker + horizon       |
              +---------------+---------------+
                              |
                              v
              +-------------------------------+
              | Step 2: Polymarket Thresholds |
              | Fetch markets with $ prices    |
              | -> [{price: 900, prob: 0.72}]  |
              +---------------+---------------+
                              |
                              v
              +-------------------------------+
              | Step 3: Historical Regime      |
              | Fetch 60-90d price history     |
              | -> Classify daily states       |
              | -> Estimate P(stay|bull)=0.72 |
              +---------------+---------------+
                              |
                              v
              +-------------------------------+
              | Step 4: Sentiment Adjustment  |
              | social_sentiment(ticker)       |
              | -> Apply multiplier to P       |
              +---------------+---------------+
                              |
                              v
              +-------------------------------+
              | Step 5: Markov Interpolation  |
              | For each price level between   |
              | anchors: n-step transition     |
              +---------------+---------------+
                              |
                              v
              +-------------------------------+
              | Step 6: Distribution Chart     |
              | price_distribution_chart()    |
              +-------------------------------+
```

---

## Components

### 1. `extractPriceThresholds`

Parse Polymarket market questions to extract numeric price thresholds.

**Input:** Array of Polymarket market objects with `question` and `probability` fields.

**Output:** Array of `PriceThreshold` objects sorted by price ascending.

**Regex patterns:**
```typescript
const PATTERNS = {
  exceed: /(?:exceed|above|over|surpass)\s*\$(\d[\d,K]*)/i,
  below: /(?:below|under|drop\s*(?:below|to))\s*\$(\d[\d,K]*)/i,
  reach: /(?:reach|hit)\s*\$(\d[\d,K]*)/i,
};
```

**Edge cases:**
- Handle K/M suffixes (`$70K` → `70000`)
- Skip ambiguous questions without numeric threshold
- Deduplicate same price levels (keep highest probability)

---

### 2. `classifyRegimeState`

Classify a trading day into a regime state for transition matrix estimation.

**States:**
```typescript
type RegimeState = 'bull' | 'bear' | 'sideways' | 'high_vol';
```

**Classification rules:**
- `high_vol`: volatility > 2%
- `bull`: return > 1% (and not high_vol)
- `bear`: return < -1% (and not high_vol)
- `sideways`: |return| ≤ 1% (and not high_vol)

---

### 3. `estimateTransitionMatrix`

Compute transition probability matrix from historical state sequence.

**Algorithm:**
1. Count transitions from state[i] → state[i+1]
2. Normalize each row to sum to 1
3. Apply smoothing (add 0.01 epsilon to all cells, then renormalize) to avoid zeros

**Default for insufficient data:** Use identity-like matrix with 0.6 diagonal, 0.2 off-diagonal (persistence assumption).

**Output:** 4×4 transition matrix with rows summing to 1.

---

### 4. `adjustTransitionMatrix`

Apply sentiment-based adjustments to baseline transition probabilities.

**Formula:**
```typescript
const shift = sentiment.bullish - sentiment.bearish; // -1 to +1
const alpha = 0.15; // adjustment strength

adjusted.bull.to.bull = base.bull.to.bull * (1 + alpha * shift);
adjusted.bull.to.bear = base.bull.to.bear * (1 - alpha * shift);
adjusted.bear.to.bear = base.bear.to.bear * (1 - alpha * -shift);
// ... then renormalize rows
```

---

### 5. `interpolateDistribution`

Fill gaps between Polymarket anchor points using n-step Markov transitions.

**Algorithm:**
1. Generate 20 price levels at ~1.5% intervals from current price
2. For each level:
   - If Polymarket anchor exists nearby → use anchor probability
   - Otherwise → estimate from Markov chain using n-step transitions
3. Ensure monotonicity (higher price → lower prob above)

**Markov estimation:**
1. Compute P^n (n-step transition matrix)
2. Get probability of being in bull/bear state after n steps
3. Map regime probabilities to expected drift
4. Use log-normal survival function for final probability

---

### 6. `markovDistribution` Tool

**Schema:**
```typescript
z.object({
  ticker: z.string().describe('Stock/ETF ticker symbol'),
  horizon: z.number().min(1).max(90).describe('Forecast horizon in trading days'),
  currentPrice: z.number().optional().describe('Current price (fetched if not provided)'),
})
```

**Output:**
```typescript
interface MarkovDistributionResult {
  ticker: string;
  currentPrice: number;
  horizon: number;
  distribution: Array<{
    price: number;
    probability: number;      // P(price > this level)
    lowerBound: number;       // 90% CI lower
    upperBound: number;       // 90% CI upper
  }>;
  metadata: {
    polymarketAnchors: number;
    regimeState: RegimeState;
    sentimentAdjustment: number;
    historicalDays: number;
  };
}
```

---

### 7. Enhancement to `probability-assessment` Skill

Add Step 2c after the existing Step 2b:

```markdown
## Step 2c — Markov Distribution (asset price queries only)

**If the query is about an asset price probability distribution:**

1. After gathering Polymarket thresholds (Step 2b), if you have ≥2 price levels,
   you MUST call `markov_distribution` to generate the full distribution.

2. The tool will:
   - Fetch historical price data to estimate regime transitions
   - Apply sentiment adjustments from social_sentiment
   - Interpolate between Polymarket anchors using Markov chain
   - Return a complete {price, probability}[] array

3. Call `price_distribution_chart` with the Markov-enhanced distribution
   to visualize the full probability curve.

4. In the Signal Evidence section, report:
   - Polymarket anchor points (raw data)
   - Historical regime used (bull/bear/sideways)
   - Sentiment adjustment applied
   - Interpolation method note
```

---

## Edge Cases

| Case | Handling |
|------|----------|
| No Polymarket markets | Fall back to pure historical Markov forecast with larger uncertainty |
| Only one threshold | Use as calibration point, interpolate one direction |
| < 30 days history | Use sector ETF or SPY as proxy transition matrix |
| No sentiment data | Use unadjusted baseline matrix |
| Horizon > 90 days | Cap at 90, warn about reduced accuracy |
| Extreme price move (>10%) | Force regime to `high_vol` |

---

## Test Strategy (TDD)

Write tests in this order:

1. **`extractPriceThresholds`** — Unit tests for regex parsing
2. **`classifyRegimeState`** — Unit tests for state classification
3. **`estimateTransitionMatrix`** — Unit tests for matrix computation
4. **`adjustTransitionMatrix`** — Unit tests for sentiment adjustment
5. **`interpolateDistribution`** — Unit tests for interpolation logic
6. **Integration test** — Full `markov_distribution` tool with mocked APIs
7. **E2E test** — Update `probability-assessment/skill.e2e.test.ts`

---

## File Structure

```
src/tools/finance/
├── markov-distribution.ts              # New tool (~250 lines)
├── markov-distribution.test.ts         # Unit tests
├── markov-distribution.integration.test.ts  # Integration tests
├── price-distribution-chart.ts         # Existing (unchanged)
└── ...

src/skills/probability-assessment/
├── SKILL.md                            # Updated with Step 2c
└── skill.e2e.test.ts                   # Updated E2E test
```

---

## Dependencies

**Existing tools (no changes):**
- `polymarket_search` — fetch threshold markets
- `get_market_data` — current price
- `yahoo-finance.getHistoricalPrices` — historical data
- `social_sentiment` — sentiment adjustment
- `price_distribution_chart` — visualization

**New code:**
- `markov-distribution.ts` — all Markov logic (no external dependencies)

---

## Reference Documents

- `references/DISTILLATION-polymarket-forecasting.md` — Polymarket signal quality rules
- `references/markov-probability/` — Academic papers on Markov chains for stock prices
- `references/sentiment-analysis/` — Sentiment analysis techniques

---

## Implementation Estimate

| Component | Lines | Complexity |
|-----------|-------|------------|
| `extractPriceThresholds` | ~40 | Low |
| `classifyRegimeState` | ~20 | Low |
| `estimateTransitionMatrix` | ~50 | Medium |
| `adjustTransitionMatrix` | ~30 | Low |
| `interpolateDistribution` | ~80 | High |
| `markovDistribution` tool | ~50 | Medium |
| Tests | ~300 | — |
| **Total** | ~570 | — |

---

## Success Criteria

1. Given a ticker and horizon, produces a probability distribution across 15-25 price levels
2. Uses Polymarket thresholds as anchor points when available
3. Interpolates between anchors using Markov chain transitions
4. Adjusts transition probabilities based on current sentiment
5. Returns metadata about regime state, anchors used, and adjustment applied
6. All unit and integration tests pass
7. E2E test demonstrates full workflow with real data