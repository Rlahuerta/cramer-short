# Polymarket for Forecasting & Asset Valuation: Research Distillation

> **Synthesised from 14 local PDFs (2025–2026) on prediction markets and
> Polymarket mechanics, accuracy, price discovery, arbitrage, manipulation,
> trader profitability, and signal quality.**
> Sources live in this `prediction-markets/` folder and sibling `../polymarket/`.
>
> Note: the source set mixes arXiv papers, SSRN/working papers, one thesis, and
> journal/conference-style PDFs. This distillation does **not** assume all sources
> are peer reviewed.

---

## Table of Contents

1. [What Polymarket Prices Actually Mean](#1-what-polymarket-prices-actually-mean)
2. [When Prices Are Trustworthy — and When They Aren't](#2-when-prices-are-trustworthy--and-when-they-arent)
3. [How to Read Price Movements as Signals](#3-how-to-read-price-movements-as-signals)
4. [Practical Strategies for Asset Forecasting](#4-practical-strategies-for-asset-forecasting)
5. [Red Flags: Manipulation, Whales, Laundering, and Wash Risk](#5-red-flags-manipulation-whales-laundering-and-wash-risk)
6. [Arbitrage, Microstructure, and Trader Skill](#6-arbitrage-microstructure-and-trader-skill)
7. [Integrating Polymarket with Other Data Sources](#7-integrating-polymarket-with-other-data-sources)
8. [Reliability and Limitations](#reliability-and-limitations)
9. [Source Coverage / Reference Map](#9-source-coverage--reference-map)
10. [Verification Checklist](#10-verification-checklist)

---

## 1. What Polymarket Prices Actually Mean

### The Theoretical Anchor

A Polymarket price of **$0.63 = 63% implied probability** that the event resolves YES,
assuming negligible fees/frictions and a cleanly specified binary payoff. It is not an
opinion poll: it is the market-clearing price from traders risking capital, routed through
Polymarket's market design, order book, collateral, and oracle/resolution rules.

### Reality Check: Accuracy Is Conditional

| Study | Dataset / scope | Finding |
|-------|-----------------|---------|
| Reichenbach & Walther (`ssrn-5910522`) | >124M Polymarket trades, nearly 1M traders, 68k markets through Sep 2025 | Market predictions are accurate most of the time; documented default/YES overtrading; <30% of traders profitable; profits persist over time. |
| Akey, Grégoire, Harvie & Martineau (`../polymarket/ssrn-6443103`) | 2022–2026 Polymarket data: >2.4M users, $67B volume, 588M trades | Aggregate prices are well calibrated; gains concentrate heavily in sophisticated limit-order traders; top 1% capture 76.5% of trading gains. |
| Clinton & Huang (`AccuracyEfficiencyClintonHuangNov2025`) | >2,500 political markets across IEM, Kalshi, PredictIt, Polymarket during final five weeks of 2024 U.S. presidential campaign | PredictIt accuracy 93%, Kalshi 78%, Polymarket 67% among active markets; same-event prices diverged and arbitrage peaked in the final two weeks. |
| Cordoba Otalora & Themistocleous (`futureinternet-17-00487-v2`) | 11M Polymarket transactions vs polling in seven 2024 swing states | In contested states, Polymarket trends preceded polling shifts by up to 14 days with high correlation / DTW similarity; low-volatility states produced low signal. |

**Key synthesis:** Polymarket can be useful as an aggregate, real-money probability signal,
but signal quality varies by market type, liquidity, lifecycle, resolution clarity, and trader
composition. Broad calibration findings do not remove the documented inefficiency of
high-salience political markets.

### YES / Default-Option Bias

Reichenbach & Walther document a tendency to overtrade the default and YES option. The
source supports a **directional caution**, not a universal numeric adjustment. For valuation
work:

- Treat raw YES prices as potentially optimistic in thin or default-framed markets.
- Check whether the equivalent NO or complementary-outcome price implies the same view.
- Do not apply a fixed 3–5% discount mechanically; estimate any adjustment from market
  depth, spreads, comparable contracts, and external evidence.

---

## 2. When Prices Are Trustworthy — and When They Aren't

### Higher-trust conditions

- **Mature, actively traded market**: Reichenbach & Walther find inaccuracies primarily
early in a contract lifecycle; Tsang & Yang show liquidity and price impact improved as
election participation broadened.
- **High depth / narrow spreads / lower price impact**: Dubach's order-book study and
Tsang & Yang's Kyle's λ analysis both support using microstructure quality as a signal
quality filter.
- **Well-defined resolution criteria**: Rohanifar et al. show how ambiguous outcomes and
oracle disputes can be laundered into a clean-looking final YES/NO result.
- **Logical consistency across linked markets**: Saguillo et al. and Clinton & Huang both
show that arbitrage or cross-platform divergence is a warning sign.
- **Stable post-news repricing**: Tsang & Yang's shock study supports distinguishing
persistent repricing from reversed spikes.

### Lower-trust conditions

| Condition | Why prices may mislead |
|-----------|------------------------|
| **Early contract lifecycle** | Fewer traders and thinner books; Reichenbach & Walther associate inaccuracies with early lifecycle trading. |
| **Final pre-resolution window in political markets** | Clinton & Huang find arbitrage opportunities peaked in the final two weeks before Election Day and daily price changes were weakly or negatively autocorrelated. |
| **Large concentrated order flow / whale episodes** | Smart et al. model whale price distortion; Tsang & Yang document October whale episodes; Rohanifar et al. describe social tracking of whales. |
| **Herding / momentum without fundamentals** | Smart et al. find whale distortion lasts longer when non-whale traders herd or learn slowly. |
| **Cross-platform or within-platform inconsistency** | Clinton & Huang show same-event divergence across exchanges; Saguillo et al. estimate about $40M realized arbitrage extracted from Polymarket inconsistencies. |
| **Governance or oracle ambiguity** | Rohanifar et al. document off-platform Discord/social dispute labor and UMA bond/friction barriers before a clean final outcome appears. |
| **High political/social salience** | Clinton & Huang and Smart et al. both warn that election markets can shape narratives as well as aggregate information. |
| **Possible wash / self-counterparty activity** | Dubach reports self-counterparty wash-share metrics and category-specific incentives; Akey et al. run wash-trading robustness checks. |

---

## 3. How to Read Price Movements as Signals

### The Persistence Test

The first price move after a shock is not informative on its own. What matters is whether it
persists and whether trading is one-sided or two-sided.

From Tsang & Yang (`2603.03152v2`), using transaction-level Polymarket data for three 2024
U.S. presidential-election shocks:

| Event | Initial pattern | Later pattern | Interpretation |
|-------|-----------------|---------------|----------------|
| Biden-Trump debate | About +11¢ peak move for Trump YES | Largely reversed; end window only about +2¢ above pre-event | Transitory overreaction / noisy repricing. |
| Trump assassination attempt | About +11¢ peak move | Repricing persisted | Durable information or belief shift. |
| Biden dropout | Heavy trading and position flipping | Two-sided trading with little net price change | Genuine disagreement / substitution logic already partly priced. |

**Asset-forecasting rule:** treat a sharp spike as provisional until it survives a persistence
check. The PDFs support waiting for persistence; any 24–48h holding period is an analyst
heuristic rather than a paper-estimated threshold.

### Lead-Lag Signal Extraction

Cordoba Otalora & Themistocleous find that, in contested swing states, Polymarket price
trends preceded polling shifts by **up to 14 days** with high correlation / DTW similarity
(up to 0.988). They also identify low-volatility states such as North Carolina as low-signal
environments.

**Practical implication:** Polymarket is most useful when uncertainty is real and contested.
When outcomes are obvious or markets are thin, price movement can mostly confirm priors
or amplify noise.

### Regime Modeling

Voigt's Beta-HMM thesis (`BetaHMMpolymarket-18-1`) models bounded, mutually exclusive
Polymarket contract prices with a Groupwise Beta Hidden Markov Model. It reports 89.3%
classification accuracy and outperformance versus a naive benchmark and standard Beta-HMM.
Use this as evidence that latent regimes (speculative trading, convergence behavior, YES/NO
resolution asymmetry) can matter; do not treat a single raw price as the full state of the market.

---

## 4. Practical Strategies for Asset Forecasting

### Strategy A: Use Price as a Conditional Probability Input

Use Polymarket probabilities as one input to conditional valuation models, after checking
market quality.

Example: Fed decision market

```text
p(cut) = 0.72   →   expected rate after = 0.72 × (current − 25bp) + 0.28 × current
                →   apply to bond duration / equity discount-rate models
```

Best practices supported or qualified by the sources:

- Prefer mature, liquid markets; avoid early lifecycle prices unless modeling uncertainty explicitly.
- Use mid-price / order-book context, not only last trade; Dubach shows spreads and depth vary systematically.
- Validate logical consistency against related Polymarket contracts and, where available, Kalshi/PredictIt.
- Inspect resolution text and oracle/dispute mechanics before treating the price as an objective probability.

### Strategy B: Read Price Velocity and Microstructure

- Slow drift over days can indicate gradual belief updating.
- A sharp spike can be genuine news or whale/noise; apply the persistence test.
- Negative serial correlation / mean reversion in daily closes is an inefficiency warning in Clinton & Huang's political-market sample.
- Order-book depth, effective spreads, Kyle's λ, and self-counterparty / wash metrics are useful filters where available.

### Strategy C: Map Event Probability to Asset Exposure

When a Polymarket event shows a persistent repricing:

1. Identify the asset exposure of the event.
2. Estimate the conditional impact if the event resolves YES.
3. Size by `Δp × conditional impact`, then haircut for market-quality risk.

Example:

```text
Market: "Trump tariffs on China" moves 55% → 78%
Δp = +23pp
Estimated portfolio impact = -3% per +20pp tariff-probability increase
Hedge size ≈ 23pp × (-3% / 20pp) × exposed portfolio value
```

### Strategy D: Treat Arbitrage as a Quality Indicator

Saguillo et al. identify two Polymarket arbitrage forms: **Market Rebalancing Arbitrage**
within a market/condition and **Combinatorial Arbitrage** across related markets. They estimate
about **$40M** realized profit extracted across both.

When related markets are inconsistently priced:

- The spread may be a trading opportunity, but it is also a signal that the raw probability is noisy.
- Check whether the inconsistency is within a market, across Polymarket markets, or across venues.
- If a spread persists, lower confidence in using either side as a single clean valuation input.

### Strategy E: Use Skilled-Trader Evidence Carefully

Several PDFs support trader heterogeneity:

- Reichenbach & Walther: <30% of traders profitable; profits persist over time.
- Akey et al.: gains concentrate heavily; top 1% capture 76.5% of gains; limit-order makers fare better than liquidity takers.
- `ssrn-6624899`: among 273 heavily analyzed wallets, top decile of profitable wallets captures 52% of positive PnL; dominant pattern is resolution-edge holding rather than short-horizon scalping.
- `ssrn-6625018`: companion paper frames profitable regimes as probability-mispricing alpha sources.

Use on-chain or analytics-derived smart-money signals as **inputs**, not proof: wallet identity,
wash risk, survivorship/selection bias, and copied trades can distort interpretation.

---

## 5. Red Flags: Manipulation, Whales, Laundering, and Wash Risk

### Whale Mechanics

Smart et al. (`2601.20452v1`) use agent-based simulations and analytical price dynamics to
show that high-budget biased agents can temporarily shift prediction-market prices. Distortion
magnitude rises with the whale's share of total market capital, and distortion duration rises
when non-whale traders herd or learn slowly.

The 2024 "French Whale" episode is discussed in Clinton & Huang and Smart et al. as an
example of a well-financed Polymarket trader affecting odds and public narratives.

**Red flags:**

- Single large trade causes a material move in a thin book.
- Price moves sharply against all external fundamentals/polls.
- Same wallet or cluster accumulates correlated positions across related markets.
- Media cites the market price as news, creating circular feedback.

### Prediction Laundering

Rohanifar, Ahmed & Sultana (`2602.05181v1`) argue that Polymarket can turn messy,
subjective, high-uncertainty bets into authoritative-looking probabilities through four stages:

1. **Structural Sanitisation**: platform selection determines which futures become tradable.
2. **Probabilistic Flattening**: informed trades, hedges, emotional bets, and whale positions
   collapse into one percentage.
3. **Architectural Masking**: capital concentration and technical literacy gaps are obscured
   by a crowdsourced-consensus interface.
4. **Epistemic Hardening**: off-platform disputes and oracle governance are erased once the
   UI displays a final YES/NO result.

**Defense:** never use raw Polymarket probability as a single source of truth; check market
rules, dispute history where available, external data, and whether resolution depends on contested
human interpretation.

### Wash / Self-Counterparty Risk

Dubach (`../polymarket/2604.24366v1`) adds order-book microstructure evidence: longshot spread
premia, category-specific spread/depth patterns, archive latency, self-counterparty wash-share
metrics, and depth decay near resolution. Akey et al. also test robustness to potential wash trading.

For forecasting, wash risk mainly matters because volume alone can overstate information quality.
Prefer depth, spreads, unique counterparties, and persistence over headline volume.

---

## 6. Arbitrage, Microstructure, and Trader Skill

### Cross-Market Arbitrage as Quality Indicator

| Signal | Interpretation |
|--------|----------------|
| Arbitrage closes quickly | Market may be well-arbitraged; price more informative. |
| Arbitrage persists | Thin liquidity, execution constraints, or fragmented attention; lower signal confidence. |
| Same event diverges across Polymarket/Kalshi/PredictIt | At least one venue is wrong or markets differ in rules/liquidity; inspect before using. |
| Arbitrage peaks near resolution | In Clinton & Huang's political sample, late information did not eliminate inefficiency. |

### Anatomy Papers: Complementary, Not Duplicate

- Tsang & Yang (`2603.03136v1`) analyze transaction-level Polygon data for the 2024
  presidential market, decomposing activity into exchange-equivalent volume, net inflow,
  and gross market activity. They document Biden withdrawal, the September debate, October
  whale inflows, narrowing arbitrage deviations, and declining Kyle's λ as participation deepens.
- Dubach (`../polymarket/2604.24366v1`) focuses on the order book: spreads, depth profiles,
  trade-direction inference, wash-share metrics, and depth decay near resolution.

Together, they imply that **volume is not enough**. For market quality, inspect net inflow,
depth, spreads, price impact, and whether activity is directional or mechanical inventory/arbitrage flow.

### Trader Profitability Papers: Definitions Matter

There is a terminology conflict across the smart-money papers:

- `ssrn-6624899` says the dominant smart-money pattern is **not arbitrage** in the sense of
  short-horizon scalping or cross-market spread capture; it is patient resolution-edge holding.
- `ssrn-6625018` says every profitable wallet is executing **probability mispricing arbitrage**
  in a broader sense: buying probabilities that are wrong relative to eventual resolution.

Treat this as a definitional difference. For asset forecasting, the common point is that skilled
participants may be exploiting slow information assimilation, behavioral bias, microstructure
inefficiency, or cross-market inconsistency.

---

## 7. Integrating Polymarket with Other Data Sources

### Recommended Multi-Source Framework

```text
Polymarket p(event), quality-adjusted
    ↓
Comparable prediction markets / exchange odds where available
    ↓
Fundamental model (polls, macro data, company/event fundamentals)
    ↓
Options / rates / asset-market-implied probabilities where available
    ↓
Analyst ensemble probability → valuation / position sizing
```

Avoid fixed universal weights. The PDFs support conditional weighting: increase Polymarket
weight when liquidity, depth, resolution clarity, and cross-market consistency are strong; reduce
it for early lifecycle markets, contested political markets, whale episodes, late-election windows,
or oracle ambiguity.

### Signal Quality Tiers for Financial Research

**Higher confidence:** objective, liquid, well-specified contracts with stable order books and
consistent related-market pricing.

**Medium confidence:** contested markets where Polymarket has demonstrated lead-lag value but
must be checked for whale flow, arbitrage deviations, and external evidence.

**Lower confidence:** high-salience political/social markets near resolution, ambiguous governance
markets, low-depth markets, and markets whose headline volume may be contaminated by wash or
self-counterparty behavior.

The older rule “macro > geopolitical > electoral” should be treated as a practical prior, not a
claim directly established by all PDFs. The strongest direct evidence in this source set is that
2024 election markets were both useful and materially inefficient.

---

## Reliability and Limitations

Use Polymarket as a conditional signal, not a standalone probability feed:

- **Liquidity and maturity:** Reichenbach & Walther, Tsang & Yang, and Dubach all support
  down-weighting early, thin, wide-spread, high-impact, or near-resolution depth-decay markets.
- **Concentrated capital / manipulation:** Smart et al. and the Clinton & Huang "French Whale"
  discussion show that large traders can distort odds, especially when books are thin or other
  traders herd.
- **Resolution and oracle governance:** Rohanifar et al. show that ambiguous outcomes, platform
  selection, and dispute handling can make subjective settlement look clean; inspect rules and
  dispute history.
- **Cross-market divergence / arbitrage:** Saguillo et al. and Clinton & Huang show cross-market
  inconsistencies and late arbitrage; do not treat one venue's price as definitive when related
  contracts disagree.
- **Political and expressive trading:** The election-market papers and laundering critique show
  high-salience political markets can mix information, hedging, expressive bets, and media feedback.
- **Metadata / extraction uncertainty:** the source map lists several PDFs with noisy or missing
  `pdfinfo` metadata or unknown authors; treat those bibliographic fields as best-effort.

---

## 9. Source Coverage / Reference Map

| PDF | Inferred title | Year / authors | Extraction status | Key contribution |
|-----|----------------|----------------|-------------------|------------------|
| `./2508.03474v1.pdf` | *Unravelling the Probabilistic Forest: Arbitrage in Prediction Markets* | 2025; Oriol Saguillo, Vahid Ghafouri, Lucianna Kiffer, Guillermo Suarez-Tangil | pdfinfo title/authors and pdftotext abstract extracted | Defines/estimates Market Rebalancing and Combinatorial Arbitrage on Polymarket; about $40M realized arbitrage profit. |
| `./2601.20452v1.pdf` | *Manipulation in Prediction Markets: An Agent-based Modeling Experiment* | 2026; Bridget Smart, Ebba Mark, Anne Bastian, Josefina Waugh | pdfinfo title/authors and pdftotext abstract extracted | Models whale manipulation; distortion scales with capital share and persists with herding / slow learning. |
| `./2602.05181v1.pdf` | *Prediction Laundering: The Illusion of Neutrality, Transparency, and Governance in Polymarket* | 2026; Yasaman Rohanifar, Syed Ishtiaque Ahmed, Sharifa Sultana | pdfinfo title/authors and pdftotext body excerpts extracted | Sociotechnical critique of platform selection, probability flattening, capital opacity, and oracle/governance dispute erasure. |
| `./2603.03136v1.pdf` | *The Anatomy of Polymarket: Evidence from the 2024 Presidential Election* | 2026; Kwok Ping Tsang, Zichao Yang | pdfinfo title/authors and pdftotext abstract extracted | Transaction-level Polygon analysis; decomposes volume/net inflow/gross activity; studies election episodes, whale inflows, arbitrage deviations, Kyle's λ. |
| `./2603.03152v2.pdf` | *Political Shocks and Price Discovery in Prediction Markets: Evidence from the 2024 U.S. Presidential Election* | 2026; Kwok Ping Tsang, Zichao Yang | pdfinfo title/authors and pdftotext abstract/body excerpts extracted | Event-study of debate, assassination attempt, and Biden dropout; establishes persistence-vs-reversal shock patterns. |
| `./AccuracyEfficiencyClintonHuangNov2025.pdf` | *Prediction Markets? The Accuracy and Efficiency of $2.4 Billion in the 2024 Presidential Election* | 2025; Joshua D. Clinton, TzuFeng Huang | pdfinfo title/author blank or noisy; title/authors/date extracted from first page with pdftotext | Cross-exchange political-market accuracy/efficiency; Polymarket 67% active-market accuracy; arbitrage peaked in final two weeks; negative/weak autocorrelation. |
| `./BetaHMMpolymarket-18-1.pdf` | *Predicting Prediction Markets: A Beta-Hidden Markov Modeling Approach* | 2025; Conrad Oskar Voigt | pdfinfo title/author blank or noisy; title/authors/date extracted from first page with pdftotext | Thesis introducing Groupwise Beta-HMM for interdependent Polymarket contracts; reports 89.3% classification accuracy and regime asymmetries. |
| `./futureinternet-17-00487-v2.pdf` | *Beyond the Polls: Quantifying Early Signals in Decentralized Prediction Markets with Cross-Correlation and Dynamic Time Warping* | 2025; Francisco Cordoba Otalora, Marinos Themistocleous | pdfinfo title/authors and pdftotext abstract extracted | DPMVF framework; Polymarket led polling shifts by up to 14 days in contested swing states; low-volatility states were low signal. |
| `./ssrn-5910522.pdf` | *Exploring Decentralized Prediction Markets: Accuracy, Skill, and Bias on Polymarket* | 2025; Felix Reichenbach, Martin Walther | pdfinfo author field noisy; title/authors extracted from first page with pdftotext | >124M-trade Polymarket dataset; accuracy, default/YES overtrading, <30% profitable traders, persistent skill. |
| `../polymarket/2604.24366v1.pdf` | *The Anatomy of a Decentralized Prediction Market: Microstructure Evidence from the Polymarket Order Book* | 2026; Philipp D. Dubach | pdfinfo title/authors and pdftotext excerpts extracted | Order-book microstructure: spreads, depth, trade-direction inference, wash metrics, depth decay near resolution. |
| `../polymarket/ssrn-6161946.pdf` | *Can Polymarket Become a More Accurate “Survey”? A Replacement for Traditional Surveys That Are Easy to Game?* | 2026; Fahril Irkham | pdfinfo title/author blank or noisy; title/authors/date extracted from first page with pdftotext | Conceptual survey-style paper comparing Polymarket incentives with traditional surveys; summarizes binary contracts, USDC/Polygon/CLOB/oracle mechanics and risks. |
| `../polymarket/ssrn-6443103.pdf` | *Who Wins and Who Loses In Prediction Markets? Evidence from Polymarket* | 2026; Pat Akey, Vincent Grégoire, Nicolas Harvie, Charles Martineau | pdfinfo title/author blank or noisy; title/authors/date extracted from first page with pdftotext | Large wallet-level profitability study: 2.4M users, $67B volume, 588M trades; top 1% capture 76.5% of gains; aggregate calibration. |
| `../polymarket/ssrn-6624899.pdf` | *Smart Money on Polymarket: A Behavioral Anatomy of 273 Top Prediction-Market Traders* | 2026; authors Unknown in extracted text | pdfinfo title/author blank or noisy; title/date extracted from first page; no author-like lines found by pdftotext | Polydata.pro wallet panel; profit concentration, resolution-edge holding, automation/bot fingerprinting, politics as dominant profit pool. |
| `../polymarket/ssrn-6625018.pdf` | *What Is the Alpha on Polymarket? A Methodology of Probability Mispricing and a Taxonomy of Eleven Profitable Regimes* | 2026; authors Unknown in extracted text | pdfinfo title/author blank or noisy; title/date extracted from first page; no author-like lines found by pdftotext | Companion to Smart Money; maps eleven profitable regimes to slow information, behavioral bias, microstructure inefficiency, and cross-market inconsistency. |

---

## 10. Verification Checklist

All 14 current PDFs in scope were checked with local `pdfinfo` and/or `pdftotext` extraction:

- [x] `./2508.03474v1.pdf`
- [x] `./2601.20452v1.pdf`
- [x] `./2602.05181v1.pdf`
- [x] `./2603.03136v1.pdf`
- [x] `./2603.03152v2.pdf`
- [x] `./AccuracyEfficiencyClintonHuangNov2025.pdf`
- [x] `./BetaHMMpolymarket-18-1.pdf`
- [x] `./futureinternet-17-00487-v2.pdf`
- [x] `./ssrn-5910522.pdf`
- [x] `../polymarket/2604.24366v1.pdf`
- [x] `../polymarket/ssrn-6161946.pdf`
- [x] `../polymarket/ssrn-6443103.pdf`
- [x] `../polymarket/ssrn-6624899.pdf`
- [x] `../polymarket/ssrn-6625018.pdf`

---

## Key Rules of Thumb (Quick Reference)

```text
1. Price is most useful when market is mature, liquid, and resolution rules are clear.
2. After a shock, separate persistent repricing from temporary spike/reversal.
3. Treat YES/default prices as potentially biased; do not use a fixed universal discount.
4. Polymarket can lead polls in contested states, but low-volatility markets may be low signal.
5. Whale risk rises with concentrated capital, herding, and thin depth.
6. Persistent arbitrage or cross-venue divergence lowers confidence in raw probabilities.
7. Volume alone is insufficient; inspect depth, spreads, net inflow, price impact, and wash risk.
8. Late political-market prices can be noisy even when information is abundant.
9. Skilled traders exist, but smart-money labels can suffer from selection, copying, and wash issues.
10. Never single-source valuation: combine Polymarket with fundamentals, comparable markets, and asset-market signals.
```
