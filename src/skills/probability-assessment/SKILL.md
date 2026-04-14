---
name: probability_assessment
description: >
  Structured probability assessment that combines Polymarket crowd-implied
  odds, social sentiment, analyst consensus, and historical base rates into
  a single quantified estimate using weighted log-odds. Use when the user
  wants to know the probability of a future event (earnings beat, rate cut,
  regulatory outcome, recession, FDA approval, etc.) with explicit signal
  breakdown and uncertainty range. Also suitable for short-horizon asset
  price forecasts, especially BTC/crypto, when the user needs a full
  structured probability report.
---

# Probability Assessment Workflow

You are running the `probability_assessment` skill. Your job is to collect
multiple independent probability signals, synthesise them using the
**weighted log-odds framework**, and output a structured assessment.

---

## Step 1 — Identify the event and asset

Restate the binary or probabilistic question you are answering, e.g.:
- "Will NVDA beat Q2 2026 EPS consensus?"
- "Will the Fed cut rates before July 2026?"
- "Will PFE receive FDA approval for [drug] in 2026?"

Identify the **asset type** (tech_semiconductor, healthcare, financials, energy,
consumer, crypto, macro) so you know which signal categories to prioritise.

---

## Step 2 — Check pre-injected Polymarket context

Look in the system prompt for the **🎯 Prediction Markets** block. If it is
present, read the crowd-implied probabilities and their signal categories
directly — do **not** make redundant Polymarket API calls for the same queries.

If the block is absent or incomplete, call `polymarket_search` using the most
relevant signal-category search phrases (see default signal maps below).

**Default signal weights by asset type:**

| Asset Type       | Signal 1 (wt)         | Signal 2 (wt)      | Signal 3 (wt)          | Signal 4 (wt)       | Signal 5 (wt)     |
|------------------|-----------------------|--------------------|------------------------|---------------------|-------------------|
| Tech/Semi        | Earnings (0.35)       | Regulation (0.20)  | Fed rates (0.20)       | Recession (0.15)    |                   |
| Healthcare       | FDA Approval (0.40)   | Earnings (0.25)    | Drug policy (0.20)     | Fed rates (0.15)    |                   |
| Financials       | Fed rates (0.35)      | Earnings (0.30)    | Recession (0.25)       | Regulation (0.10)   |                   |
| Energy           | OPEC/Oil (0.35)       | Earnings (0.25)    | Geopolitical (0.25)    | Recession (0.15)    |                   |
| Consumer         | Earnings (0.35)       | Recession (0.30)   | Fed rates (0.20)       | Tariffs (0.15)      |                   |
| Crypto           | SEC/Regulation (0.30) | ETF/Product (0.25) | BTC Price Target (0.20) | Fed rates (0.15)    | Recession (0.10)  |
| Macro (general)  | Fed rates (0.35)      | Recession (0.35)   | Tariffs (0.20)         | Geopolitical (0.10) |                   |

---

## Step 2b — Extract price threshold markets for chart (asset price queries only)

**If the query is about an asset price** (e.g. "Will BTC be higher?", "Gold price forecast"):

After gathering Polymarket markets, identify all markets that mention a specific
**dollar price level** in their question text — patterns like:
- "Will BTC exceed $70,000?"
- "Will Gold (GC) settle at >$6,200 in June?"
- "Will BTC reach $80K by end of March?"

For each such market, record `{price, probability}` where:
- `price` = the dollar level as a number (e.g. 70000, 6200, 80000)
- `probability` = the YES probability (0–1) — treat these as **upper-tail**
  (P(asset price > this level))

**PRE-FLIGHT CHECK — you MUST complete this before writing any output:**

1. Count the distinct price levels you have from Polymarket.
2. If you have **2 or more** distinct price levels, **CALL `price_distribution_chart` NOW**
   — before writing the Signal Evidence section. This is **mandatory**, not optional.
   Pass the `{price, probability}[]` array and the current asset price.
3. Embed the chart output verbatim in the **Signal Evidence** section under
   Polymarket, immediately before the summary table.
4. If you have fewer than 2 price levels, note "No chart: fewer than 2 price
   thresholds available" in the Polymarket evidence section and continue.

> ⚠️ **Do NOT skip `price_distribution_chart` when you have ≥2 price levels.**
> The chart is the most informative visual in the entire output — it shows the
> crowd-implied price distribution at a glance. Missing it is a quality failure.

---

## Step 2c — Markov distribution (asset price queries only)

**If the query is about an asset price AND you have ≥2 Polymarket price thresholds:**

After calling `price_distribution_chart` (Step 2b), call `markov_distribution` to enrich
the distribution with regime-aware Markov interpolation and 90% Monte Carlo confidence intervals.

1. **Gather inputs** (you likely already have these from prior steps):
   - `ticker` — the asset symbol (e.g. NVDA, BTC-USD, SPY). For commodities, use the liquid ETF: gold → GLD, silver → SLV, oil → USO.
   - `horizon` — forecast horizon in trading days (convert calendar days: 30 calendar ≈ 21 trading)
   - `historicalPrices` — 60–90 days of daily close prices from `get_market_data` (oldest first)
   - `polymarketMarkets` — the raw Polymarket market objects (question + probability + volume)
   - `sentiment` (optional) — bullish/bearish from `social_sentiment` if already gathered

2. **Call `markov_distribution`** with these inputs. The tool will:
   - Classify each day into a regime state (bull/bear/high_vol_bull/high_vol_bear/sideways)
   - Estimate a 5×5 Markov transition matrix with Dirichlet smoothing
   - Detect structural breaks (regime shifts mid-window) and widen CI when detected
   - Blend Markov log-normal estimates with bias-corrected Polymarket anchors
   - Return P(price > X) for 20+ levels with [5th, 95th] percentile CI bounds

3. **Embed the output** in the Signal Evidence section after the `price_distribution_chart`.
   Highlight the regime state, mixing-time weight, and any warnings (sparse states, structural break,
   cross-platform divergence).

4. **Report in the Signal Evidence section:**
   - Current regime state (e.g. "bull", "high_vol_bear")
   - Mixing-time weight (e.g. "42% Markov / 58% Polymarket anchors")
   - R²_OS if available (positive = Markov adds predictive value over mean)
   - Any `⚠️` warnings from metadata

> ⚠️ **Do NOT skip `markov_distribution` when you have ≥2 price thresholds and historical prices.**
> It provides crucial uncertainty quantification (CI bounds) that `price_distribution_chart` alone
> cannot produce. Missing it is a quality failure for price distribution queries.

---

## Step 3 — Gather remaining signals

Collect as many of these as are relevant and available. Each becomes a
`LogOddsSignal` with a probability [0,1] and its category weight from
the table above.

### 3a. Social sentiment
Call `social_sentiment` or `x_search` with the asset ticker or event keyword.
Convert the bullish/bearish ratio to a probability:
- 70% bullish posts → probability ≈ 0.70
- Assign weight: **0.15** (social sentiment is a noisy signal)

### 3b. Analyst consensus
- Use `get_financials` to retrieve EPS estimates or analyst ratings.
- If analysts forecast a beat by ≥5%: probability ≈ 0.72
- If estimates are flat: probability ≈ 0.50
- If estimates revised down: probability ≈ 0.30
- Assign weight: **0.25**

### 3c. Historical base rate
- What fraction of similar events has happened historically? (e.g. NVDA beats
  EPS ~80% of recent quarters → probability = 0.80)
- Use `web_search` for company earnings history if not in memory.
- Assign weight: **0.20**

---

## Step 4 — Compute weighted log-odds combination

Apply the formula to each signal you have (drop absent signals, re-normalise
remaining weights to sum to 1.0):

```
log_odds(p) = ln(p / (1 − p))          [clamp p to 0.001–0.999]

combined_log_odds = Σ wᵢ × log_odds(pᵢ)

p_combined = 1 / (1 + exp(−combined_log_odds))

σ = √(Σ wᵢ × (log_odds(pᵢ) − combined_log_odds)²)
lower = 1 / (1 + exp(−(combined_log_odds − σ)))
upper = 1 / (1 + exp(−(combined_log_odds + σ)))
```

Flag divergence (⚠️) when σ > 0.3.

---

## Step 5 — Output structured assessment

The output must follow this exact order: **evidence first, then the summary table, then interpretation**.

### 5a — Signal Evidence (show the raw data, always first)

For **every** signal you used, list the exact data points you read. Do not
summarise or paraphrase — show the actual question text, percentage, and where
it came from. This is the most important part of the output.

```
**Signal Evidence**

Polymarket (crowd)
  • "Will BTC exceed $70K by March 30?" → 3.7% YES  ($350K volume)
  • "Will BTC stay above $60K through March?" → 99.7% YES  ($280K volume)
  • Implied range: $60–70K with ~4% chance of upside breakout

ETF / Product flows (weight 30%)
  • IBIT net flow last 7 days: −$120M (net outflow)
  • Spot BTC ETF combined AUM: −2.1% WoW
  • Interpretation: institutional flows are bearish short-term

Fed / Rates (weight 20%)
  • "Fed cuts before July 2026?" → 15% YES  (Polymarket)
  • 10Y yield: 4.32% (Reuters, Mar 28)
  • Interpretation: no rate tailwind expected in 30-day window

Recession risk (weight 15%)
  • "US recession by EOY 2026?" → 36% YES  (Polymarket)
  • Near-term 30-day recession probability: ~5% (baseline)
  • Interpretation: macro risk elevated but not imminent

Social sentiment (weight 15%)
  • Reddit r/Bitcoin: 42% bullish / 58% bearish  (150 posts, 24h)
  • Fear & Greed Index: 9/100  (Extreme Fear)
  • Interpretation: contrarian signal — Extreme Fear has historically marked local lows
```

### 5b — Summary table (after evidence)

```
📊 Probability Assessment: [Event question]

| Signal                  | Probability | Weight |
|-------------------------|-------------|--------|
| Polymarket (crowd)      |          4% |    35% |
| ETF/Product flows       |         25% |    30% |
| Fed rates               |         45% |    20% |
| Recession risk          |         40% |    15% |
|-------------------------|-------------|--------|
| **Combined (log-odds)** | **24% ±6pp**|        |

*Signals are [consistent / ⚠️ divergent — treat with caution].*
```

### 5c — Interpretation (after table)

One short paragraph:
- What the combined probability means in plain language
- The single most informative signal and why (reference the evidence above)
- Any data gaps (missing signals, thin Polymarket liquidity)
- Suggested action framing

---

## Step 6 — Bear case (required)

Apply Munger's inversion: state what would have to be true for the bull
scenario to fail. **Always include this block after the interpretation**:

```
**Bear case**: [1–2 sentences describing the primary scenario that invalidates
the thesis]. If [specific trigger — e.g. "Bitcoin breaks below $58K support"],
this assessment would revise down to ~X%. Watch: [one concrete indicator to
monitor — e.g. "weekly close below $60K", "SEC enforcement action", "Fed
surprise hike"].
```

Keep it concise (3 sentences max). Do not simply restate the low probability
as the bear case — identify the *mechanism* (catalyst + path) that makes it
play out.

---

## Constraints

- Never report a probability below 1% or above 99%.
- If only one signal is available, report it with a ±15pp uncertainty band and
  note that it is a single-source estimate.
- Do not fabricate Polymarket probabilities — only use values returned by
  `polymarket_search` or the pre-injected 🎯 block.
- Polymarket probabilities are market-implied, not guaranteed outcomes. Always
  include a disclaimer at the end.

---

## Trajectory Mode (Short-Horizon Forecasts)

When the user asks for day-by-day price movement (e.g., "What will AAPL do over
the next 7 days?"), use the `markov_distribution` tool with `trajectory: true`:

```
markov_distribution({
  ticker: "AAPL",
  horizon: 7,
  historicalPrices: [...],
  polymarketMarkets: [],
  trajectory: true,
  trajectoryDays: 7
})
```

This returns a day-by-day table with expected price, 90% CI, P(up), and cumulative
return for each day. Present the trajectory table alongside your probability
assessment for a complete short-term outlook.

**When to use trajectory mode:**
- User asks for day-by-day or multi-day outlook
- Short horizons (1–14 days) where the path matters, not just the endpoint
- Comparing how uncertainty grows over the forecast window

---

## Step 7 — Crypto-specific enrichment (BTC/crypto forecasts)

When the asset is BTC or another cryptocurrency, enrich the assessment with
these additional signals before computing log-odds:

| Tool                       | When to use                                         |
|----------------------------|-----------------------------------------------------|
| `get_onchain_crypto`       | Always — on-chain metrics, sentiment, developer activity |
| `get_fixed_income`         | Always — rate/yield-curve context affects crypto    |
| `get_options_chain`        | Optional — BTC ETF skew when options data is relevant |
| `geopolitics_search`        | Optional — only when policy/war/trade risk is a live driver |
| `trump_pressure_index`      | Optional — only when executive-order risk is material    |
| `markov_distribution`      | Short BTC horizons (≤14 days) — call with `trajectory=true` |

**How to integrate:**

1. **On-chain data** (`get_onchain_crypto`): Use the sentiment score as an
   additional signal. If sentiment > 70% bullish, weight as 0.60 probability
   of upside; if < 30% bullish, weight as 0.40. Include developer activity
   and community metrics in the evidence section.

2. **Rate context** (`get_fixed_income`): Use the 10Y–2Y spread and Fed
   funds rate as macro context. An inverted yield curve or high real rates
   are bearish for crypto; an easing cycle is bullish.

3. **Options skew** (`get_options_chain`, optional): If BTC ETF options are
   available, use put/call ratio and implied volatility to gauge skew.
   Heavy put skew → bearish signal (weight 0.15).

4. **Markov trajectory** (`markov_distribution` with `trajectory=true`):
   For short-horizon BTC forecasts (≤14 days), call markov_distribution
   after gathering Polymarket thresholds and on-chain data. The trajectory
   output provides day-by-day expected prices and confidence intervals,
   which you should present alongside the probability assessment summary.

---
