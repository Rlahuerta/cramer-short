# Polymarket & Prediction Tool — Improvement Research

**Date:** 2026-07-14  
**Scope:** Deep research brainstorm on improving prediction accuracy, Polymarket API integration, and forecast signal quality. Adjacent forecast/signal ideas also covered.  
**Authored by:** GitHub Copilot CLI (research + analysis session)

---

## TL;DR — Priority Index

| # | Area | Est. impact | Eng. cost | Priority |
|---|------|-------------|-----------|----------|
| A | CLOB API — bid-ask spread as quality signal | High | Low | **P1** |
| B | CLOB `/prices-history` — price velocity signal | High | Medium | **P1** |
| C | Longshot-bias calibration correction | High | Low | **P1** |
| D | Time-to-resolution weighting | Medium | Low | **P2** |
| E | WebSocket real-time anchor refresh | Medium | High | **P2** |
| F | `jumpDirection` annotation on extracted markets | High | Low | **P2** |
| G | Multi-platform fusion (Kalshi, Manifold, metaforecast) | Medium | Medium | **P2** |
| H | Orderbook imbalance / informed-flow signal | Medium | High | **P3** |
| I | Volatility regime-switching HMM (Idea 1 backlog) | High | High | **P3** |
| J | MSM — Markov-Switching Multifractal (Idea 3 backlog) | Very High | Very High | **Research** |

---

## 1. CLOB API — Untapped Polymarket Endpoint

### 1.1 Current State

`polymarket.ts` uses **only** the Gamma API (`gamma-api.polymarket.com`). This gives market metadata, probability snapshots, and volume. There is a completely separate **CLOB API** (`clob.polymarket.com`) that is not touched anywhere in the codebase.

```
Current:   https://gamma-api.polymarket.com   (metadata, discovery, yes/no price)
Missing:   https://clob.polymarket.com         (orderbook depth, price history, spread)
```

### 1.2 CLOB Endpoints Available (no auth required for read-only)

| Endpoint | Returns | Use for |
|----------|---------|---------|
| `GET /book?token_id=<tokenId>` | Full orderbook (all bids/asks) | L2 depth, spread |
| `GET /price?token_id=<tokenId>&side=buy` | Best bid or ask | BBO |
| `GET /midpoint?token_id=<tokenId>` | Arithmetic midpoint | Cleaner probability estimate |
| `GET /spread?token_id=<tokenId>` | Bid-ask spread | **Liquidity quality signal** |
| `GET /prices-history?market=<conditionId>` | Historical prices over a time range | **Price velocity / momentum** |

The `tokenId` is different from the market's `conditionId`. The Gamma API returns `clobTokenIds` per market as an array of `[yesTokenId, noTokenId]`.

### 1.3 Improvement A — Bid-Ask Spread as Quality Signal

**Problem:** `computeMarketQualityWeight` uses only `volume24h`, `ageDays`, tier, and spike flags. Low-liquidity markets with stale quotes can show high volume (from a single day's whale) but a wide spread that reveals thin underlying depth.

**Proposal:**

```typescript
// In ensemble.ts MarketInput
interface MarketInput {
  // ... existing fields ...
  bidAskSpread?: number;  // raw spread [0,1]; undefined → not fetched
}

// In computeMarketQualityWeight:
const wSpread = m.bidAskSpread !== undefined
  ? Math.max(0, 1 - m.bidAskSpread / 0.10)  // spread > 10 cents → quality 0
  : 1.0;  // if unknown, no penalty

const w = wAge * wLiq * tau * wSpread * (1 - deltaWhale * 0.5) * (1 - deltaTransitory * 0.3);
```

**Data flow:** When building `MarketInput` in `polymarket-forecast.ts`, make one CLOB `/spread?token_id=<yesTokenId>` call per market (batched, cached 5 min). Enriches `MarketInput.bidAskSpread`.

**Why it matters:** A YES price of 0.65 with a spread of 0.20 (bid 0.55 / ask 0.75) carries huge uncertainty. The midpoint 0.65 is not the "real" probability — the true value could be anywhere in [0.55, 0.75]. Discounting such markets prevents noise from dominating the ensemble.

### 1.4 Improvement B — Price Velocity via `/prices-history`

**Problem:** The snapshot-based spike detection (`evaluateMarketHistory` in `polymarket-forecast.ts`) only works after the tool has been called at least once 2–4 hours before. On the first run, spike detection is cold. More importantly, it cannot distinguish a *slow steady drift* from a *sudden manipulation spike*.

**Proposal:** On market acquisition, fetch the last 24h of CLOB price history and compute:
1. **Velocity** = `(p_now - p_24h_ago) / 24` (pp per hour of drift)
2. **Spike score** = max single-hour absolute jump in the window

```typescript
interface MarketInput {
  priceVelocityPpH?: number;   // pp/hour, positive = momentum, negative = fading
  maxHourlyJump?: number;      // max |Δp| in any 1h window over last 24h (spike proxy)
}
```

**Quality discount update:**
- `|priceVelocityPpH| > 2` → potential fast-moving event, reduce weight 20%
- `maxHourlyJump > 8pp` → spike signal regardless of local snapshot history

**Why it matters:** The CLOB `/prices-history` call replaces the dependency on prior `polymarket-snapshots.jsonl` history. The first run will still have full spike detection.

---

## 2. Calibration — Longshot Bias Correction

### 2.1 Current State

`adjustYesBias(p, beta=0.035)` applies a uniform −3.5pp discount only when `p > 0.5`. The `YES_BIAS_MULTIPLIER = 0.95` is a flat 5% haircut in the Markov pipeline.

This addresses the *favourite side* of the favourite-longshot bias but misses the *longshot side* entirely.

### 2.2 The Problem

Empirical calibration studies on Polymarket (Reichenbach & Walther 2025; l-marque calibration study on GitHub, 2024) confirm a U-shaped miscalibration curve:

- At `p ∈ [0.10, 0.90]`: market is roughly well-calibrated
- At `p < 0.10` (longshots): YES price is systematically **too high** — longshots resolve YES less often than their market price implies
- At `p > 0.90` (strong favourites): YES price is also slightly too high

The current code corrects only the region `p > 0.50` (favorites) with a flat shift. It does **nothing** for the fat tails — the very region most prone to manipulation and noise.

### 2.3 Proposed Asymmetric Calibration

Replace `adjustYesBias` with a shape that tracks the empirical longshot-bias curve:

```typescript
/**
 * Empirically calibrated YES-bias correction for Polymarket prices.
 *
 * Regime mapping based on Reichenbach & Walther (2025) and Brier-score
 * calibration curves from the l-marque Polymarket calibration study:
 *
 *   p < 0.05  → strong longshot discount (longshots are overpriced ~30-40%)
 *   p < 0.15  → moderate longshot discount (~15-20%)
 *   p ∈ [0.15, 0.85] → mild uniform discount from favourite-longshot bias
 *   p > 0.85  → mild favourite discount (favourites slightly overpriced)
 */
export function adjustYesBiasV2(p: number): number {
  if (p <= 0) return 0.001;
  if (p >= 1) return 0.999;

  let adjusted: number;
  if (p < 0.05) {
    // Longshot regime: discount by ~30% of p (pull toward 0)
    adjusted = p * 0.70;
  } else if (p < 0.15) {
    // Moderate longshot: linear interpolation from 30% to 5% discount
    const t = (p - 0.05) / 0.10;
    const discount = 0.70 + t * (0.95 - 0.70);  // 0.70 → 0.95 multiplier
    adjusted = p * discount;
  } else if (p <= 0.85) {
    // Mid-range: mild additive correction (current behavior)
    adjusted = p > 0.50 ? p - 0.035 : p;
  } else {
    // Strong favourite: mild haircut
    adjusted = p - 0.025;
  }

  return Math.max(0.001, Math.min(0.999, adjusted));
}
```

**Impact on jump-diffusion:** The Q→P transformation (`transformQToP` in `rnd-integration.ts`) should also apply this calibration *before* the Radon-Nikodym shift, so the corrected physical probability enters the MC step. The current pipeline applies `YES_BIAS_MULTIPLIER = 0.95` (flat) on the Polymarket anchor before `transformQToP` — replacing it with `adjustYesBiasV2` would give more nuanced correction for extreme-probability anchors.

**Testing plan:**  
Unit test: compare output of `adjustYesBiasV2` vs. `adjustYesBias` at `p ∈ {0.03, 0.08, 0.12, 0.30, 0.55, 0.70, 0.92}` to confirm longshot discounts and mid-range stability.

---

## 3. Time-to-Resolution Weighting

### 3.1 Theory

Prediction market probabilities are martingales: `E[P(T) | F_t] = P(t)`. As `t → T` (resolution), the distribution of `P(T)` collapses toward `{0, 1}`. A market with 2 days to expiry is a *much sharper* signal than one with 60 days. Yet the current `computeMarketQualityWeight` uses `ageDays` (how old the market is) not `daysToExpiry` (how close to resolution it is).

**Aging into the past ≠ converging to the future.** A 90-day-old market that expires tomorrow has near-certain information; a brand-new market that expires in 180 days is highly uncertain.

### 3.2 Proposed `wExpiry` Factor

```typescript
/**
 * Time-to-resolution quality boost.
 *
 * A market 1d from resolution gets weight ×1.5 (high certainty).
 * A market 30d from resolution gets weight ×1.0 (neutral).
 * A market 90d from resolution gets weight ×0.7 (penalty for uncertainty).
 */
function computeExpiryBoost(daysToExpiry: number): number {
  if (daysToExpiry <= 1) return 1.50;
  if (daysToExpiry <= 7) return 1.20;
  if (daysToExpiry <= 30) return 1.00;
  if (daysToExpiry <= 90) return 0.85;
  return 0.70;  // far-dated markets get a 30% discount
}
```

**Integration point:** `MarketInput` already has no `daysToExpiry` field. Add it as optional. `polymarket-forecast.ts` already computes `endDate` for each market; deriving `daysToExpiry` is trivial (`(Date.parse(endDate) - now) / 86400000`).

**Interaction with jump-diffusion:** `extractJumpEventMarkets` already computes `daysToSettlement` per market. Pipe it through to `MarketInput.daysToExpiry` so the ensemble quality weight benefits from the same data already in flight.

---

## 4. Jump-Direction Annotation on Extracted Markets

### 4.1 Current Gap

`extractJumpEventMarkets` (line 890 of `polymarket.ts`) returns `JumpEventMarket` objects that are passed to the Merton MC step via `jumpSpec`. The jump size defaults are hardcoded in `JUMP_DEFAULTS` per asset class. But there is no information about *direction* — is this a bullish jump (e.g., "ETF approved by Friday?") or a bearish jump (e.g., "Will US impose tariffs by Friday?")?

Without direction, every event market is assumed to be a **downside jump**. The compensation term `muJump` is always negative (−4% to −10%). Bullish Polymarket events (regulatory approvals, cease-fires, positive earnings surprises) are therefore mis-modelled.

### 4.2 Proposed Fix

Extend `JumpEventMarket` with a `jumpDirection` field:

```typescript
export interface JumpEventMarket {
  id: string;
  probability: number;
  daysToSettlement: number;
  question: string;
  jumpDirection?: 'up' | 'down' | 'unknown';  // ADD THIS
}
```

**Classification approach (lightweight, no LLM call needed for most cases):**

```typescript
// Rule-based keyword classification in extractJumpEventMarkets:
function classifyJumpDirection(question: string): 'up' | 'down' | 'unknown' {
  const q = question.toLowerCase();
  const bearish = /tariff|war|attack|invade|default|bankrupt|ban|sanction|collapse|crash|down|fall|below|cut/;
  const bullish = /approve|approve|pass|win|raise|above|exceed|surge|rally|peace|deal|merger|acqui/;
  if (bearish.test(q)) return 'down';
  if (bullish.test(q)) return 'up';
  return 'unknown';
}
```

**MC step update:** In `computeJumpDiffusionStep` (or equivalent), when `jumpDirection === 'up'`, flip `muJump` to positive:

```typescript
const effectiveMuJ = jumpSpec.jumpDirection === 'up'
  ? Math.abs(jumpSpec.muJ)    // positive drift on bullish events
  : -Math.abs(jumpSpec.muJ);  // negative drift on bearish events (default)
```

**Impact:** Avoids systematic bearish bias in the trajectory MC when Polymarket is tracking a bullish catalyst.

---

## 5. WebSocket Real-Time Anchor Refresh

### 5.1 What It Is

Polymarket exposes a public WebSocket at `wss://ws-subscriptions-clob.polymarket.com/ws/market`. After subscribing with one or more `token_id` values (YES token IDs), the server streams:
- Orderbook snapshots (full L2)
- Delta updates (price level changes)
- Best bid/ask updates
- New trade executions

No authentication is required for market data.

### 5.2 Use Case for Cramer-Short

Currently, `polymarket_forecast` is stateless between calls. The scheduled job runner (`bun start schedule run`) reruns full fetches on a cadence. For tracked geopolitical markets (Middle East escalation, Fed decision, tariff vote), there can be significant probability moves between scheduled checks that the agent only learns about on the next full run.

**Proposed flow:**
1. When a `polymarket_forecast` call sets an anchor, write the `token_id` list to a persistent watchlist (`.cramer-short/polymarket-ws-watchlist.json`).
2. A background WebSocket subscriber process monitors these markets.
3. On a `|Δp| > threshold` event (e.g., 5pp move in <10 min), write an alert to `.cramer-short/polymarket-alerts.jsonl`.
4. The agent's pre-context injection checks this file and prepends a warning if any tracked markets have moved since the last snapshot.

**Eng. cost:** Medium-High. Requires a persistent WebSocket process (runs alongside the gateway or TUI), alert format design, and pre-context injection hook.

---

## 6. Multi-Platform Prediction Market Fusion

### 6.1 Available Platforms

| Platform | API | Coverage | Strength |
|----------|-----|----------|----------|
| Polymarket | Gamma + CLOB | Crypto, geopolitics, macro | Real money, large volume |
| **Kalshi** | REST + WebSocket | US macro, Fed, economic indicators | CFTC-regulated, US policy focus |
| **Manifold** | REST API (public) | Broad, play-money and real-money | Diverse topics, fast question creation |
| **Metaculus** | REST API (public) | Science, geopolitics, long-horizon | Superforecaster community |
| **metaforecast.org** | Public API (Nuño Sempere) | Aggregates 15+ platforms | Cross-platform normalisation |

### 6.2 Why Cross-Platform Disagreement Matters

When Polymarket shows 40% but Metaculus superforecasters show 25% on the same event:
- The discrepancy is itself a signal — one market has an information advantage, or one has a liquidity distortion
- The *spread* in cross-platform estimates is a proxy for event uncertainty
- Weighting by inverse cross-platform variance would reduce the ensemble's sensitivity to any single platform's noise

### 6.3 Practical Entry Point — metaforecast.org

`metaforecast.org/api` (maintained by Nuño Sempere) provides a clean REST API:

```
GET https://metaforecast.org/api/v2/questions?query=<keyword>&limit=10
```

Response includes: question title, probability, platform, resolution date, n_forecasters, stars (quality score).

A lightweight integration in `signal-extractor.ts` could try a `metaforecast` fetch as a secondary lookup after the Gamma API fetch. If the same question appears on Metaculus at a significantly different probability (|Δp| > 10pp), add a warning to `MarketInput.warnings` and apply a 20% uncertainty-discount to the ensemble weight.

### 6.4 Kalshi Integration

Kalshi's CFTC-regulation and US-policy focus makes it the natural complement for macro signals:
- Fed funds rate decisions → Kalshi is likely more liquid than Polymarket for US policy events
- GDP contraction / recession contracts

Kalshi API (`api.elections.kalshi.com/trade-api/v2`) is public for read-only market data. No auth required for prices. A `kalshi_signal` tool could parallel-run alongside `polymarket_search`.

---

## 7. Calibration Backtesting Infrastructure

### 7.1 The Problem

The current calibration corrections (`adjustYesBias`, `YES_BIAS_MULTIPLIER`) were tuned on Reichenbach & Walther (2025) aggregate statistics. There is no project-local backtesting loop that validates our corrections against actual Polymarket market resolutions.

### 7.2 TheGraph On-Chain Data

Polymarket's on-chain history is queryable via The Graph:

```
https://api.thegraph.com/subgraphs/name/0xpolygon/polymarket
```

Available data: all historical trades (price, size, timestamp, side), market resolutions (YES/NO outcome). This allows:
1. Build a calibration curve: for markets where we observed price `p ∈ [k, k+0.05)`, what fraction resolved YES?
2. Compare the empirical calibration curve to the theoretical `adjustYesBiasV2` shape
3. Tune the correction multipliers based on Cramer-Short's specific market selection (rather than Polymarket-wide aggregate statistics)

### 7.3 Snapshot File as Local Proxy

More immediately, `.cramer-short/polymarket-snapshots.jsonl` already records every market fetch. When the tool is used regularly, resolved markets accumulate over time. A lightweight Python script (`research/tools/calibration_audit.py`) could:

1. Load all snapshot records
2. For resolved markets (those with `endDate` in the past), fetch the final resolution from the Gamma API
3. Build a Brier score and calibration curve
4. Emit per-probability-bucket overpricing estimates

This creates a data flywheel: the more the agent uses prediction markets, the more accurate its calibration becomes.

---

## 8. Orderbook Imbalance / Informed-Flow Signal

### 8.1 What It Is

The bid-ask spread (§1.3) captures liquidity. The *order imbalance* captures directional pressure:

```
imbalance = (totalBidVolume - totalAskVolume) / (totalBidVolume + totalAskVolume)
```

`imbalance > 0` → more buyers than sellers → upward pressure on probability.  
`imbalance < 0` → more sellers than buyers → downward pressure on probability.

In equity markets, high positive order imbalance is a strong short-term predictor of price moves (Chordia, Roll & Subrahmanyam 2002). In prediction markets, informed traders may position ahead of news releases by placing orders rather than taking the spread.

### 8.2 Integration Point

The CLOB `/book?token_id=<yesTokenId>` endpoint returns all bids/asks with sizes. Computing `imbalance` requires one REST call per market. It would be a new field in `MarketInput`:

```typescript
orderImbalance?: number;  // [-1, 1]; positive = buy pressure, negative = sell
```

When `orderImbalance < -0.5` (strong sell pressure), reduce the YES probability slightly before the ensemble step (`p_adjusted -= 0.02 * Math.abs(orderImbalance)`). This is a heuristic, not a theoretical result — treat it as a weak signal requiring empirical validation.

---

## 9. Volatility-Regime Switching (Deferred from Idea 1)

### 9.1 Status

This was Idea 1 from `docs/forecast-improvement-review-2026-04-28.md` (Section 3, P2). Not yet implemented. Key blocker: requires IV (implied volatility) history data, which is not in `data/prices.py`.

### 9.2 Practical Shortcut: VIX as Regime Proxy

Instead of running a secondary HMM on IV vs. spot correlation, use **VIX level** as a synchronous regime indicator:
- VIX < 15 → sticky-strike regime (moderate vol, no leverage effect penalty)
- VIX 15–25 → transitional regime (mild leverage effect)
- VIX > 25 → sticky-implied-tree regime (strong leverage effect, negative return-vol correlation)

VIX is already available via `get_market_data('VIX')`. This removes the data-source blocker at the cost of some precision (VIX is SP500-specific; does not reflect BTC or commodity vol directly).

**Implementation sketch:**

```typescript
function getVolatilityRegime(vix: number): 'sticky_strike' | 'transitional' | 'sticky_implied_tree' {
  if (vix < 15) return 'sticky_strike';
  if (vix < 25) return 'transitional';
  return 'sticky_implied_tree';
}

// In computeTrajectory, per-step:
const volMultiplier = regime === 'sticky_implied_tree' && z < 0
  ? 1.4   // leverage effect: vol spikes on down moves
  : regime === 'sticky_implied_tree' && z > 0
  ? 0.8   // muted vol on up moves in fear regime
  : 1.0;  // neutral
```

This is a non-breaking optional parameter: `trajectoryOptions.volRegime?: string`.

### 9.3 Asset-Class Gating

Leverage-effect vol regime is empirically confirmed for:
- **Equities** (strong negative return-vol correlation, ~−0.7)
- **Gold** (mild, ~−0.3)

It is **not** applicable to:
- **BTC** (positive return-vol correlation in bull markets — more like a commodity bubble)
- **Oil** (regime-dependent)

The `getAssetProfile` function in `markov-distribution.ts` already classifies assets. Gate the leverage-effect by `profile.assetClass === 'equity' || profile.assetClass === 'gold'`.

---

## 10. Markov-Switching Multifractal (Research Track)

### 10.1 Status

Idea 3 from `docs/forecast-improvement-review-2026-04-28.md` (Section 4, P3 / Research). Not implemented. Full implementation requires a Python research spike first (`research/spikes/msm/`) before TypeScript port.

### 10.2 Research Entry Points

- Calvet & Fisher (2001, 2004) papers are available at `references/markov-probability/` in the repo
- `research/models/trajectory.py` is the right starting point for the Python prototype
- Key parameter to calibrate: `k_components` (typically 3–5) and per-frequency transition probabilities `gamma`

### 10.3 Shortcut: GARCH(1,1) as Interim Improvement

MSM is complex to port. As an interim, replacing the constant `dailyVol` in `computeTrajectory` with a simple GARCH(1,1) update would already address the volatility clustering issue that Bloch critiques:

```
h_t = omega + alpha * z_{t-1}^2 * h_{t-1} + beta * h_{t-1}
vol_t = sqrt(h_t)
```

Parameters `(omega, alpha, beta)` can be estimated from the same 252-day price history already used for `dailyVol`. A GARCH(1,1) fit is 5 lines of Python (`arch` library) and the parameters are stable enough to pre-fit once at forecast time.

---

## 11. Polymarket Snapshot File Improvements

### 11.1 Current Issues

The `.cramer-short/polymarket-snapshots.jsonl` file grows indefinitely with no pruning. At 1 snapshot per market per fetch, and 4–8 markets per `polymarket_forecast` call, a daily user generates ~50–100 records/day. After a year, this is 20k–35k lines — still manageable, but worth addressing.

### 11.2 Proposed Improvements

**a) Pruning on write:** Keep only the 3 most recent snapshots per `marketId`. Since only the 2–4h and 24–48h windows matter, keeping 3 snapshots per market (current, 6h, 48h) is sufficient.

**b) Add bid/ask to snapshot:** Currently records only `probability` (midpoint). Recording `bid` and `ask` separately enables:
- Spread-trend detection (widening spread = deteriorating liquidity)
- Better spike detection: a spike in price without a spike in spread is more likely informational

**c) Index by marketId:** Currently `findSnapshotInWindow` does a linear scan over all records. A `Map<marketId, record[]>` in-memory index (built once on load) would make window lookups O(log n) per market instead of O(n·markets).

---

## 12. Signal Chain Gaps

### 12.1 Missing: Earnings-Calendar Polymarket Cross-Reference

When the agent runs a forecast for NVDA in the 14-day window, the Polymarket signal fetcher queries "NVIDIA" but does not check whether NVDA has an earnings release in that window. An earnings call is a scheduled event that *creates* a jump event even without a Polymarket market explicitly about earnings outcome.

**Proposed:** In `extractSignals()` (signal-extractor.ts), add an earnings-date check from `get_earnings_dates`. If earnings fall within the forecast horizon, automatically inject a synthetic jump event with the asset-class-appropriate default `muJ/sigmaJ` from `JUMP_DEFAULTS`, bypassing the Polymarket search requirement.

### 12.2 Missing: Sector-Wide Contagion in Impact Map

The `IMPACT_MAP` treats each event independently. But many geopolitical events have **correlated cross-sector effects**. For example, an oil embargo YES event affects:
- Energy (direct +20%)
- Airline (direct −15%)
- Materials (indirect +5%)
- Tech (indirect −3% via production cost)

The current map has these entries in separate rows and they are applied independently. There is no mechanism for the agent to recognise that "oil embargo YES" should simultaneously push up energy positions and push down airline positions when constructing an overall portfolio forecast.

This is a design gap requiring a more advanced "event graph" structure, and is flagged here as a known limitation.

### 12.3 Missing: Intraday Resolution Check for Expiring Markets

If a market expires today (0 days to resolution), its current price is not a probability — it is the traders' last-second bet before resolution. These markets should be flagged as **do not anchor** because they are no longer updating on information, they are updating on execution flows.

**Simple fix:** In `extractJumpEventMarkets` and in `polymarket-forecast.ts`, when `daysToSettlement < 1`, skip the market for anchoring and emit a warning.

---

## 13. Summary Recommendations

### Immediate Actions (P1 — code-ready, minimal risk)

1. **Implement `adjustYesBiasV2`** with proper longshot-bias shape. Replace `adjustYesBias` (backward-compat: same API, better calibration). Mirror in `research/models/ensemble.py`.
2. **Add `daysToExpiry` to `MarketInput`** and `computeExpiryBoost` to `computeMarketQualityWeight`. Pipe from `endDate` already available in Gamma API response.
3. **Skip markets with `daysToSettlement < 1`** in both `extractJumpEventMarkets` and `polymarket-forecast.ts` market acquisition.

### Near-Term Actions (P2 — medium engineering, high value)

4. **Add CLOB `/spread` fetch** as optional enrichment in `polymarket-forecast.ts`. Introduce `bidAskSpread` into `MarketInput` and quality weight.
5. **Add `jumpDirection` annotation** to `JumpEventMarket` via keyword-based classification. Update `computeJumpDiffusionStep` to respect direction.
6. **Snapshot file pruning** (top-3 per marketId) and in-memory index.

### Research Track

7. **Calibration backtest pipeline** using TheGraph on-chain resolution data + local snapshots.
8. **GARCH(1,1) interim volatility** in trajectory MC (single-file change in `markov-distribution.ts`).
9. **VIX-based regime proxy** for leverage-effect vol adjustment.
10. **MSM Python prototype** at `research/spikes/msm/` using Calvet & Fisher parameters.

---

## Appendix: References

| Source | Relevance |
|--------|-----------|
| Reichenbach & Walther (2025) | YES overpricing empirics, `references/prediction-markets/` |
| Tsang & Yang (2026) | Polymarket-Markov integration, `references/prediction-markets/` |
| El Hassan / Maddah / Taleb (2026) | Jump risk in short-tenor options, `references/derivatives-risk/` |
| Calvet & Fisher (2001, 2004) | MSM model, `references/markov-probability/` |
| l-marque calibration study (GitHub) | Longshot bias empirics, external |
| Metaforecast API (Sempere) | Cross-platform fusion, `metaforecast.org/api/v2` |
| Polymarket CLOB docs | Spread/book endpoints, `docs.polymarket.com` |
| Kalshi API docs | US policy markets, `api.elections.kalshi.com/trade-api/v2` |

---

*Generated by GitHub Copilot CLI — topic-branch `topic/opt-skills`, commit `af7f5ec`.*
