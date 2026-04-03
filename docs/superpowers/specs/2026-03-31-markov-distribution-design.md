# Markov Chain Probability Distribution for Asset Price Forecasting

> **Design approved:** 2026-03-31
> **Approach:** Polymarket-First Distribution with Markov Interpolation
> **Integration:** Enhancement to existing `probability-assessment` skill

---

## Overview

This design adds a Markov chain-based probability distribution generator to Cramer-Short. It produces full probability distributions for stock/ETF prices at user-specified horizons (1-90 days) by combining:

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

**Input:** Array of Polymarket market objects with `question`, `probability`, and optional `volume`/`createdAt` fields.

**Output:** Array of `PriceThreshold` objects sorted by price ascending.

**YES-bias correction:** Apply `correctedProb = rawProb × 0.95` to all anchor probabilities.
Rationale: Reichenbach & Walther (2025) found systematic YES-overtrading across 124M Polymarket
trades. Raw YES probability is consistently overstated; a ~5% downward discount improves calibration.

**Liquidity guard:** Set `trustScore = 'low'` for markets that:
- Are less than 48 hours old (price set by first movers with incomplete information)
- Have volume below a meaningful threshold (thin order book = noise, not information)

Only markets with `trustScore = 'high'` are used as primary anchors. Low-trust markets are
included with reduced weight in the blending formula.

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
type RegimeState = 'bull' | 'bear' | 'high_vol_bull' | 'high_vol_bear' | 'sideways';
```

**Classification rules (joint encoding — volatility and direction are orthogonal):**
- `high_vol_bull`: volatility > 2% **and** return > 0
- `high_vol_bear`: volatility > 2% **and** return ≤ 0
- `bull`: return > 1% (and volatility ≤ 2%)
- `bear`: return < -1% (and volatility ≤ 2%)
- `sideways`: |return| ≤ 1% (and volatility ≤ 2%)

> **Why joint states?** In the original design, `high_vol` took priority as an override,
> destroying directional information on high-volatility days. A strongly bullish +4% day
> and a strongly bearish -4% day would both be classified identically as `high_vol`.
> Joint states preserve both dimensions. This is consistent with Nguyen (2018, *IJFS*),
> where states encode distinct combinations of mean return and variance.
>
> **V2 upgrade path:** Replace the 2% threshold with a VIX-percentile regime classifier.
> Davidovic & McCleary (2025) show VIX captures 45–50% of return variation vs <5% for
> raw volatility. The 2% threshold is a reasonable V1 simplification.

**State space is now 5×5** for the transition matrix.

---

### 3. `estimateTransitionMatrix`

Compute transition probability matrix from historical state sequence.

**Algorithm:**
1. Count transitions from state[i] → state[i+1]
2. Normalize each row to sum to 1
3. Apply smoothing (add 0.01 epsilon to all cells, then renormalize) to avoid zeros

**Default for insufficient data:** Use identity-like matrix with 0.6 diagonal and uniform off-diagonal.
Formula: `offDiag = (1 - 0.6) / (numStates - 1)`. For 5 states: `0.4 / 4 = 0.1`, rows sum to exactly 1.0.

> **Bug fix:** The prior design specified "0.2 off-diagonal" for a 4-state matrix, which
> yields row sums of 0.6 + 3×0.2 = 1.2 ≠ 1. The correct formula is `(1 − diagonal) / (numStates − 1)`.

**Smoothing:** Apply Dirichlet-style smoothing: add **α = 0.1** to all transition counts before
normalizing. This is the standard flat prior recommended by Welton & Ades (2005, *Med Decis Making*)
and provides meaningful regularization on 60-day windows. The former value of 0.01 was too
conservative (barely moves estimates when counts are in the tens).

**Output:** 5×5 transition matrix with rows summing to 1.

---

### 4. `adjustTransitionMatrix`

Apply sentiment-based adjustments to baseline transition probabilities.

**Formula:**
```typescript
const shift = sentiment.bullish - sentiment.bearish; // -1 to +1
const alpha = 0.07; // adjustment strength (reduced from 0.15 — Davidovic & McCleary 2025 show
                    // news sentiment captures <5% of return variation; FinBERT lacks robust
                    // standalone predictive power. Overly strong adjustments distort the
                    // empirically estimated transition matrix more than warranted.)

adjusted.bull.to.bull = base.bull.to.bull * (1 + alpha * shift);   // bullish → more bull persistence
adjusted.bull.to.bear = base.bull.to.bear * (1 - alpha * shift);   // bullish → less bull→bear
adjusted.bear.to.bear = base.bear.to.bear * (1 - alpha * shift);   // bullish → less bear persistence
adjusted.bear.to.bull = base.bear.to.bull * (1 + alpha * shift);   // bullish → more bear→bull
// ... then renormalize each row to sum to 1
```

> **Bug fix:** The prior design had `adjusted.bear.to.bear = base * (1 - alpha * -shift)`,
> which expands to `base * (1 + alpha * shift)`. This *increases* bear persistence under
> bullish sentiment — the opposite of the intended effect. The double-negative has been removed.

---

### 5. `interpolateDistribution`

Fill gaps between Polymarket anchor points using n-step Markov transitions.

**Algorithm:**
1. Generate 20 price levels at ~1.5% intervals from current price
2. For each level:
   - If Polymarket anchor exists nearby → apply **YES-bias correction**:
     `correctedProb = rawProb × 0.95`
     (Reichenbach & Walther 2025 found systematic YES-overtrading across 124M trades)
   - Otherwise → estimate from Markov chain using n-step transitions
3. Ensure monotonicity (higher price → lower prob above)

**Markov estimation:**
1. Compute `P^n` (n-step transition matrix via eigendecomposition for efficiency: `A^k = Q·D^k·Q⁻¹`)
2. Compute **mixing time weight** to decay Markov signal at long horizons:
   ```
   ρ = second-largest absolute eigenvalue of P
   markovWeight = exp(−ρ × n)     // approaches 0 as n → ∞ (stationarity)
   ```
   At short horizons (n ≤ 10), `markovWeight ≈ 1` and the regime state dominates.
   At long horizons (n ≥ 45), `markovWeight → 0` and Polymarket anchors dominate.
3. Compute **expected drift and volatility** from state-conditional parameters:
   ```
   μ_eff = Σᵢ P(state=i after n steps) × μᵢ      // weighted mean daily return per regime
   σ_eff = sqrt(Σᵢ P(state=i | n steps) × σᵢ²)   // weighted std dev per regime
   μ_n   = n × μ_eff                                // log-space drift for n-day horizon
   σ_n   = σ_eff × sqrt(n)                          // log-space vol (Brownian scaling)
   ```
   where `μᵢ`, `σᵢ` are estimated from empirical returns in each regime state.
4. Apply **log-normal survival function**:
   ```
   P(price > X) = 1 − Φ((ln(X / S₀) − μ_n) / σ_n)
   ```
   where `Φ` is the standard normal CDF, `S₀` is current price.
5. **Blend Markov estimate with Polymarket anchor** (when anchor exists):
   ```
   finalProb = markovWeight × markovEstimate + (1 − markovWeight) × correctedAnchor
   ```

**Confidence intervals (90% CI via Monte Carlo):**
1. Simulate `N = 1000` independent n-step random walks through the transition matrix
2. For each walk, apply the log-normal survival formula with the realised regime weights
3. Take the 5th and 95th percentiles of the resulting probability distribution
4. Expose as `lowerBound` / `upperBound` on each distribution point

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
    probability: number;      // P(price > this level), bias-corrected
    lowerBound: number;       // 90% CI lower (Monte Carlo, 5th percentile)
    upperBound: number;       // 90% CI upper (Monte Carlo, 95th percentile)
    source: 'polymarket' | 'markov' | 'blend';  // how this point was estimated
  }>;
  metadata: {
    polymarketAnchors: number;
    regimeState: RegimeState;
    sentimentAdjustment: number;
    historicalDays: number;
    mixingTimeWeight: number;      // exp(-ρ×n); near 1 = Markov-dominant, near 0 = anchor-dominant
    secondEigenvalue: number;      // ρ (spectral gap diagnostic)
    outOfSampleR2: number | null;  // R²_OS vs. historical-average baseline; null if no validation set
  };
}
```

**R²_OS metric:** When at least 20 days of held-out data exist after the training window,
compute out-of-sample R² vs. the historical-average baseline:
```
R²_OS = 1 − Σ(actual − predicted)² / Σ(actual − mean(actual))²
```
R²_OS > 0 means the Markov model adds value over the naive mean forecast.
Include this in metadata as a diagnostic for model reliability.

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

## Theoretical Notes & V2 Upgrade Paths

### Observable MC vs. Hidden Markov Model
This design uses an **observable Markov chain** (hard state classification per day) as a V1
simplification. The academic literature (Nguyen 2018, Kumar & Amer 2023, Voigt 2025) favours
**Hidden Markov Models (HMM)** where:
- Each day's observable return is an *emission* from a latent regime state
- States are inferred probabilistically via Baum-Welch EM estimation
- Soft state assignments reduce discretisation error at threshold boundaries

**V2:** Upgrade `classifyRegimeState` to a Gaussian-emission HMM. Use `AIC/BIC` to select the
optimal number of states (typically 3–4 for daily equity data per Nguyen 2018).

### Beta Distribution for Polymarket Probability Outputs
Polymarket prices live in `[0, 1]`. A **Beta(α, β) distribution** is theoretically superior to
log-normal for modeling bounded probabilities. Voigt (2025) shows Beta-HMM achieves 89.3%
classification accuracy on Polymarket data. Log-normal is used in V1 because it is simpler to
parameterise from empirical return statistics.

**V2:** Replace log-normal survival with Beta-distributed regime emissions when the primary
output is a Polymarket probability (not a price level).

### VIX-Percentile Regime Threshold
The `high_vol` 2% daily volatility threshold is a fixed heuristic. Davidovic & McCleary (2025)
show that **implied volatility (VIX) captures 45–50% of return variation** vs. <5% for realised
volatility measures. A regime threshold defined as "VIX above 80th historical percentile" would
be more empirically grounded and adaptive to market conditions.

**V2:** Add VIX data fetch and replace the `> 2% daily vol` rule with a VIX-percentile regime gate.

### Model Selection via AIC/BIC
The number of states (currently 5) is chosen heuristically. The standard approach
(Nguyen 2018) selects the optimal N via information criteria:
```
AIC = −2·ln(L) + 2k
BIC = −2·ln(L) + k·ln(T)
```
where `k = N² + 2N − 1` (parameters in an N-state HMM with Gaussian emissions), `T` = observations.

**V2:** Run state selection at model init, choosing N ∈ {2, 3, 4, 5} that minimises BIC.

---

Write tests in this order:

1. **`extractPriceThresholds`** — Regex parsing, YES-bias correction (×0.95), liquidity guard trustScore
2. **`classifyRegimeState`** — All 5 joint states: `high_vol_bull`, `high_vol_bear`, `bull`, `bear`, `sideways`
3. **`estimateTransitionMatrix`** — Row-sum invariant, Dirichlet α=0.1, correct default matrix (offDiag = 0.4/4 = 0.1 for 5 states)
4. **`adjustTransitionMatrix`** — Sign direction: bullish shift ↑ bull-persistence, ↓ bear-persistence (verify no double-negative)
5. **`interpolateDistribution`** — Log-normal survival formula, monotonicity, mixing-time decay at long horizons
6. **Monte Carlo CI** — 5th < point estimate < 95th percentile
7. **Integration test** — Full `markov_distribution` tool with mocked APIs
8. **E2E test** — Update `probability-assessment/skill.e2e.test.ts`

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