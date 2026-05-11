# Polymarket for Forecasting & Asset Valuation: Research Distillation

> **Synthesised from 8 peer-reviewed papers (2024–2026) on Polymarket mechanics,
> accuracy, price discovery, manipulation, and signal quality.**
> See `references/prediction-markets/` for source PDFs.

---

## Table of Contents

1. [What Polymarket Prices Actually Mean](#1-what-polymarket-prices-actually-mean)
2. [When Prices Are Trustworthy — and When They Aren't](#2-when-prices-are-trustworthy--and-when-they-arent)
3. [How to Read Price Movements as Signals](#3-how-to-read-price-movements-as-signals)
4. [Practical Strategies for Asset Forecasting](#4-practical-strategies-for-asset-forecasting)
5. [Red Flags: Manipulation, Whales, and Laundering](#5-red-flags-manipulation-whales-and-laundering)
6. [Cross-Market Arbitrage as a Quality Indicator](#6-cross-market-arbitrage-as-a-quality-indicator)
7. [Integrating Polymarket with Other Data Sources](#7-integrating-polymarket-with-other-data-sources)
8. [Reference Map](#8-reference-map)

---

## 1. What Polymarket Prices Actually Mean

### The Theoretical Anchor
A Polymarket price of **$0.63 = 63% implied probability** that the event resolves YES.
This is not an opinion poll — it's the market-clearing price from financially-incentivised
traders willing to risk capital on their information advantage.

### Reality Check: Aggregate Accuracy
| Study | Dataset | Accuracy |
|-------|---------|----------|
| Reichenbach & Walther (ssrn-5910522) | 124M trades, Nov 2022–Sep 2025 | Closely tracks realised probabilities; slightly outperforms bookmaker odds |
| Clinton & Huang (AccuracyEfficiency…) | 2,500+ markets, 2024 election, $2.4B | **67% accuracy on Polymarket** vs 93% PredictIt, 78% Kalshi |
| Cordoba Otalora & Themistocleous (futureinternet) | 11M transactions, 7 swing states | Led polling by **up to 14 days** in contested states (p < 0.01, r = 0.988) |

**Key insight:** Polymarket is accurate *in aggregate over many markets* but performs
significantly worse than competitors on highly-publicised, politically-charged markets
(elections) where expressive/strategic trading distorts prices.

### "Yes" Bias
Across 124M trades, there is a **systematic tendency to overtrade "Yes" contracts
and the default/lead option.** When using prices for valuation, this means:
- Raw YES probability is likely slightly **overstated** — apply a small downward discount
- For binary asset outcome bets, NO contracts may be systematically underpriced

---

## 2. When Prices Are Trustworthy — and When They Aren't

### ✅ High-trust conditions
- **Mature market** (contract has been trading for several weeks, not days or hours)
- **High liquidity** — wide order book depth, many counterparties
- **Well-defined resolution criteria** — objective trigger, not judgment call
- **Low political/expressive salience** — sports, economics, macro data releases
- **Price has been stable** for multiple days without large whale-driven spike

### ❌ Low-trust conditions

| Condition | Why prices mislead |
|-----------|-------------------|
| **Early in contract lifecycle** | Insufficient traders; price set by first movers with incomplete info |
| **Just before resolution** | Overreaction and noise trading spike; negative serial correlation observed |
| **Recent large single-trade spike** | Whale manipulation — proportional to whale's capital share |
| **Herding behaviour visible** (price tracking its own momentum) | Self-reinforcing dynamics divorced from fundamentals |
| **Cross-platform divergence** | Same-event contracts priced differently on Kalshi/PredictIt = someone is wrong |
| **Governance dispute likely** | Oracle system has known resolution ambiguity (bond challenges, Discord disputes) |
| **High political/social salience** | Expressive/partisan trading, "narrative shaping" motives |

---

## 3. How to Read Price Movements as Signals

### The Persistence Test — Most Important Rule

The **first price move after a shock is not informative on its own.**
What matters is whether it *persists*.

From Tsang & Yang (2603.03152v2), studying Biden-Trump debate, assassination attempt,
and Biden dropout at tick-level precision:

| Event | Initial Move | After 4 Hours | Interpretation |
|-------|-------------|--------------|---------------|
| Biden-Trump debate | +11¢ Trump | +2¢ net | **Transitory noise** — overreaction, mostly reversed |
| Trump assassination attempt | +11¢ Trump | persisted | **Durable information** — genuine belief shift |
| Biden dropout | –4¢ Trump trough | –2¢ net | **Disagreement** — heavy two-sided trading, unclear direction |

**For asset forecasting:** a price spike that reverses within hours = noise event, weak signal.
A spike that holds for 24–48h = regime change in the underlying probability.

### Distinguish These Three Post-Shock Patterns

1. **One-sided repricing** (persistent): small group of informed traders correcting misprice
   → use as input to conditional probability update for correlated assets

2. **Two-sided heavy trading, little net move**: genuine disagreement about how the shock
   maps to outcomes → *don't* trade on this signal; market hasn't decided

3. **Immediate spike + reversal**: expressive/momentum trading, thin liquidity
   → fade this signal; wait for stabilisation

### Lead-Lag Signal Extraction

In highly contested environments (AZ, NV, PA in 2024), Polymarket **preceded polling shifts
by up to 14 days** with exceptionally high correlation (up to r = 0.988, DTW similarity).

BUT in low-volatility environments (NC), the framework correctly identified a
**"low-signal environment"** — the market had nothing real to predict and produced noise.

**Practical rule:** The signal quality of Polymarket is *conditional on the degree of
genuine uncertainty*. In "obvious" outcomes (strong favourite), market prices carry
almost no forecasting value beyond confirming the prior. Maximum signal value exists
in **genuinely contested, uncertain, high-stakes** markets.

---

## 4. Practical Strategies for Asset Forecasting

### Strategy A: Binary Event as Conditional Probability Input

Use Polymarket probabilities as **p(event) inputs to conditional valuation models.**

Example: Fed rate decision market
```
p(cut) = 0.72   →   expected rate after = 0.72 × (current − 25bp) + 0.28 × current
                →   apply to bond duration / equity discount rate models
```

Best practices:
- Read price at market *maturity* (>3 weeks of active trading), not at launch
- Ignore the initial 48h of a new contract
- Use mid-price, not last trade (order book spread is meaningful)
- Validate against Kalshi/PredictIt for the same event — if same, confidence ↑

### Strategy B: Price *Velocity* as Regime Indicator

Don't just read the price — read the **rate of change:**
- Slow drift over days = gradual belief updating on new information = reliable
- Sharp spike (>10% in one hour) = either whale or genuine breaking news — wait 24h for persistence test
- Negative serial correlation in daily closes = overreaction cycle = ignore short-term moves

### Strategy C: Correlated Asset Positioning

When a Polymarket event resolves with a **persistent (not transitory) price shift:**

1. Identify the primary asset class exposure of the event
2. Map: election result → fiscal policy → sector impact → ticker
3. Take the delta of probability × estimated earnings/price impact

Example: "Trump tariffs on China" market moves from 55% → 78%
```
Δp = +23pp
Expected impact on XLI (industrials) = –3% per 20pp tariff probability increase
Position: hedge with put options sized for 23pp × (–3%/20pp) × portfolio exposure
```

### Strategy D: Cross-Market Spread Trade

From Saguillo et al. (2508.03474v1): **$40M in arbitrage extracted** from Polymarket
due to mismatched dependent markets.

When two logically related markets (e.g., "Candidate X wins state A" and "wins national")
are inconsistently priced:
- The spread is either a manipulation signal OR a genuine disagreement worth investigating
- Resolution: model the logical constraint (p(national win) ≥ p(state win if electoral math)
- Trade: buy underpriced, short overpriced (on platforms that allow it)

### Strategy E: Use Skilled Trader Proxy

Only ~30% of traders are profitable, and **profits persist over time** — skilled traders exist
and repeatedly outperform. Large sustained traders with on-chain visible consistent profit
are *informed participants.*

Practical implementation: monitor large-order flow on Polygon chain for:
- Sudden large limit orders (>$10k) in macro/geopolitical markets
- Consistent profitable addresses building positions — these are likely informed

---

## 5. Red Flags: Manipulation, Whales, and Laundering

### Whale Mechanics (Smart et al., 2601.20452v1)

A single well-capitalised trader can temporarily shift prices. The distortion:
- Is **proportional to their share of total market capital**
- Persists longer when other traders exhibit **herding behaviour**
- Decays based on non-whale learning rates

The "French Whale" case: a $45M bet by a French national temporarily pushed
Trump's odds up significantly and drove global media coverage — demonstrating how
market visibility amplifies individual influence beyond its informational content.

**Red flag signals:**
- Single large trade causes >5% price move with thin order book
- Price moves against all external fundamentals/polls simultaneously
- Same whale address building large positions across correlated markets
- Media coverage citing the market price AS NEWS (circular feedback loop)

### Prediction Laundering (Rohanifar et al., 2602.05181v1)

Four-stage process by which noisy bets become "authoritative probabilities":

1. **Structural Sanitisation**: platform decides WHICH futures are bet-able,
   pre-filtering reality through Polymarket's ontology
2. **Probabilistic Flattening**: strategic hedges, emotional bets, and informed
   positions all collapse into one percentage — heterogeneous motives invisible
3. **Architectural Masking**: capital concentration hidden behind "crowdsourced consensus"
4. **Epistemic Hardening**: governance disputes erased; messy Oracle process
   produces what looks like objective fact

**Practical defense:**
- Never use raw Polymarket probability as a single-source-of-truth for valuation
- Cross-validate with FRED/economic data, polling aggregates, option-implied vol
- Check the actual market resolution rules — many have ambiguous trigger conditions
- Governance disputes are resolved on Discord, not on-chain — check dispute history

---

## 6. Cross-Market Arbitrage as a Quality Indicator

Persistent arbitrage opportunities signal **price inefficiency** — use this to assess
whether a market is genuinely information-aggregating or just speculation:

| Signal | Interpretation |
|--------|---------------|
| Arbitrage opportunity closed within minutes | Market is well-arbitraged, price is informative |
| Arbitrage persists for hours | Thin liquidity, fragmented traders, noisy signal |
| Same event priced differently across Polymarket/Kalshi/PredictIt | At least one market is wrong; use the most liquid as anchor |
| Arbitrage peaks in final 2 weeks of contract | Pre-resolution noise, not information |

Clinton & Huang found that arbitrage opportunities on political markets **peaked in the
final two weeks before Election Day** — exactly when you'd expect the most information.
This paradox means **late-market prices are noisier than mid-period prices** despite
more information being available. Be most sceptical of pre-resolution spikes.

---

## 7. Integrating Polymarket with Other Data Sources

### Recommended Multi-Source Framework

```
Polymarket p(event)
    ↓ weight: 0.40
Superforecaster aggregate / Good Judgment Open
    ↓ weight: 0.25
Fundamental model (polls, economic indicators)
    ↓ weight: 0.25
Options market implied probability (for financial events)
    ↓ weight: 0.10
→ Ensemble forecast probability
→ Input to asset valuation / position sizing
```

### Signal Quality Tiers for Financial Research

**Tier 1 — High confidence inputs:**
- Macro data release outcomes (Fed decisions, CPI prints, payrolls)
- Corporate events with objective resolution (merger close, FDA approval)
- Sports and entertainment (minimal expressive trading)

**Tier 2 — Useful but discount by ~15%:**
- Geopolitical events (conflict escalation, sanctions)
- Legislative outcomes (bill passage, veto)
- Central bank chair appointments

**Tier 3 — Use only as directional signal, not quantitative input:**
- Electoral outcomes (demonstrated 67% accuracy, significant whale risk)
- "Will X say/do Y" speculative contracts
- Markets with <$100k total volume

### Lead-Time Exploitation

Polymarket's 14-day lead on polling makes it useful for **positioning ahead of consensus:**
- When Polymarket and polls diverge significantly → the *direction* of divergence is a signal
- Don't size based on the magnitude — that may be manipulated
- Size based on your own fundamental assessment of which direction is correct

---

## 8. Reference Map

| File | Paper | Key Contribution |
|------|-------|-----------------|
| `prediction-markets/ssrn-5910522.pdf` | Reichenbach & Walther (2025) | 124M-trade accuracy study; "Yes" bias; skilled trader persistence |
| `prediction-markets/AccuracyEfficiencyClintonHuangNov2025.pdf` | Clinton & Huang (2025) | 67% accuracy, price divergence across exchanges, arbitrage patterns |
| `prediction-markets/2603.03152v2.pdf` | Tsang & Yang (2026) | Tick-level price discovery after 3 political shocks; persistence test |
| `prediction-markets/futureinternet-17-00487-v2.pdf` | Cordoba & Themistocleous (2025) | 14-day polling lead; DTW signal validation; low-signal environments |
| `prediction-markets/2508.03474v1.pdf` | Saguillo et al. (2025) | $40M arbitrage extraction; combinatorial arbitrage mechanics |
| `prediction-markets/2601.20452v1.pdf` | Smart et al. (Oxford, 2026) | Whale manipulation model; herding amplification; $45M French Whale case |
| `prediction-markets/2602.05181v1.pdf` | Rohanifar et al. (2026) | Prediction laundering; epistemic stratification; Oracle governance risk |
| `prediction-markets/2603.03136v1.pdf` | Tsang & Yang (2026) | Anatomy of Polymarket; on-chain data structure; transaction ecology |

---

## Key Rules of Thumb (Quick Reference)

```
1. Price is trustworthy:   when mature (>3 weeks), high liquidity, no recent spike
2. Persistence test:       wait 24–48h after any price move before acting on it
3. "Yes" bias:             discount YES probabilities ~3–5% for systematic overtrade
4. Lead-lag:               Polymarket leads polls by up to 14 days in contested markets
5. Whale alert:            >5% move in <1 hour on thin order book → wait, don't act
6. Arbitrage signal:       persistent cross-platform spread = noisy, uninformative market
7. Never single-source:    always cross-validate with options vol, FRED, or fundamentals
8. Tier classification:    macro > geopolitical > electoral for quantitative reliability
9. Pre-resolution noise:   most sceptical of prices in final 2 weeks of contract lifecycle
10. Skilled traders exist: track on-chain consistent-profit addresses as informed signal
```
