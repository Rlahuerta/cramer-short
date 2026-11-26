# When to Use the Markov Model

**Last Updated:** 2026-11-25
**Evidence:** 14 tickers × 6 horizons (5–30d), daily walk-forward over 2021–2025 with the live settings
(`bun run src/tools/finance/backtest/long-history-skill.ts`; see [Evidence Base](#evidence-base))

---

## Quick Answer

**The model has no measured directional edge.** Over 2021–2025 its BUY/SELL calls were right **56.5%** of the time,
**3.0pp less** than always calling "up" on the same days (59.5%, 95% CI of the gap [−3.6, −2.5]pp). No ticker and no
horizon from 5 to 30 days beat always calling "up", and none beat a coin flip on Brier score. Because the model calls
"up" most of the time, its hit rate tracks the market: about 60% in rising years, about 40% in the 2022 decline.

**Use it for:**
- ✅ A **probability distribution** of the horizon price — P(price > X) at many levels — and its **intervals**
- ✅ Scenario buckets and a sense of the range of outcomes, not a direction to trade
- ✅ Context next to other evidence (fundamentals, prediction markets), not as the deciding signal

**Do NOT use it for:**
- ❌ A directional edge on any asset or horizon (none was found; see [Measured Accuracy](#measured-accuracy))
- ❌ Sizing or filtering trades by `predictionConfidence` (the score does not rank accuracy)
- ❌ Forecasts beyond 30 days (at 45–60d `markovWeight` is near zero; beyond 60d the chain mixes away)
- ❌ A run that **abstains or returns no calibrated distribution** — treat that as a skip, not an edge

---

## Measured Accuracy

All tickers and horizons pooled, by period (traded = BUY/SELL calls; Brier against a coin flip's 0.25, lower is better):

| Period | Up-moves | Model hit rate | Always "up" | Gap [95% CI] | Brier − 0.25 [95% CI] |
|--------|----------|----------------|-------------|--------------|-----------------------|
| 2021 | 63% | 61.5% | 63.4% | −1.9pp [−2.6, −1.3] | −0.0047 [−0.0102, +0.0021] |
| 2022 (decline) | 43% | 39.6% | 41.7% | −2.0pp [−4.4, +0.1] | **+0.0133** [+0.0113, +0.0159] |
| 2023 | 63% | 58.1% | 62.7% | −4.6pp [−5.4, −3.9] | +0.0026 [−0.0018, +0.0070] |
| 2024–25 | 64% | 61.4% | 64.4% | −2.9pp [−3.5, −2.4] | **−0.0097** [−0.0127, −0.0063] |
| **All** | 59% | **56.5%** | **59.5%** | **−3.0pp** [−3.6, −2.5] | −0.0013 [−0.0034, +0.0010] |

The probabilities beat a coin flip only in the 2024–25 rise and were worse than one in the 2022 decline.

### Per asset (all periods, 5–30d)

| Asset | Gap vs always "up", across horizons | Brier − 0.25, across horizons | Cells significantly worse |
|-------|-------------------------------------|-------------------------------|---------------------------|
| SPY | −4.5 to −1.8pp | −0.019 to −0.004 | — |
| QQQ | −5.6 to −1.8pp | −0.013 to +0.000 | — |
| VOO | −4.8 to −1.6pp | −0.019 to −0.004 | — |
| VTI | −4.8 to −1.2pp | −0.016 to −0.002 | — |
| DIA | −5.6 to −1.4pp | +0.001 to +0.005 | — |
| GLD | −5.3 to −0.8pp | −0.011 to −0.002 | 30d (vs always "up") |
| IAU | −4.9 to −0.3pp | −0.010 to −0.002 | 14d, 30d (vs always "up") |
| MSFT | −3.9 to −2.2pp | −0.000 to +0.008 | — |
| NVDA | −3.3 to −0.2pp | −0.017 to −0.005 | 5d (vs always "up") |
| GOOGL | −5.0 to −1.5pp | −0.011 to +0.002 | — |
| AMZN | −8.1 to −1.1pp | +0.002 to +0.018 | 30d (vs always "up") |
| AAPL | −6.8 to −2.1pp | −0.003 to +0.015 | 30d (vs always "up") |
| TSLA | −7.3 to −1.4pp | +0.002 to +0.022 | 10d (vs a coin flip) |
| BTC-USD | −3.9 to +0.0pp | +0.005 to +0.010 | 5d (vs always "up"); 5d, 10d (vs a coin flip) |

No cell's CI cleared zero in the model's favour. Earlier versions of this guide listed "confirmed" assets and horizons
with 65–89% accuracy; those came from a 2024–25 audit scored against a coin flip, and they do not hold over 2021–2025
against always calling "up".

---

## The Confidence Score

The model outputs a **`predictionConfidence` score** (0–1) for each prediction. **Do not filter or size trades by it.**
On the live-config benchmark and 8 held-out tickers (2026-11-25) the score did not predict whether a BUY/SELL call was
right: the highest-scored fifth of calls was the least accurate, and no threshold separated better calls from worse
ones. The tool no longer warns below a confidence threshold.

**Abstain or no calibrated distribution** — The model declined to produce a calibrated result (typically because
trusted anchors are missing or the regime signal is too weak). Skip the trade rather than interpreting fallback text as
a signal.

---

## What the Output Means

### Key Metrics

| Metric | What It Is | What to Look For |
|--------|------------|------------------|
| **P(up)** | Probability price goes up | A lean, not an edge: it did not beat a coin flip over 2021–2025 |
| **Central 90% interval** | Range the horizon price should fall in 90% of the time | Covered 90.1% of outcomes over 2021–2025 and 89.9% in the 2022 decline (95%: 94.9%, 99%: 98.9%) |
| **Confidence** | Model-agreement score (0–1) | Not a reliability signal: it did not predict accuracy (2026-11-25) |
| **Brier Score** | Mean squared error of P(up) | 0.25 = a coin flip; lower is better |
| **Directional Accuracy** | % of correct BUY/SELL calls | 56.5% over 2021–2025, below always calling "up" (59.5%) |
| **Track record** | How often this call (BUY or SELL) was right for the asset type at this horizon, next to the share of all days that moved its way | Shown under every BUY/SELL; a hit rate no higher than the base rate means the call has not beaten simply following the market |

### How to Read the Decision Card

```
📊 Markov Distribution: SPY | Horizon: 14d
Current: $520.50 | Regime: bull
Anchors: 3 trusted | Anchor quality: HIGH
Mixing: 85% Markov / 15% Anchors

🎯 Recommendation: BUY
  P(up): 68% | Expected: $535.20 (+2.8%)
  90% CI: $510.30 – $560.10
  Confidence: 0.32

📏 Track record (etf at 14d, measured on DIA, QQQ, SPY, VOO, VTI, 2021-2025): BUY calls were right 64% of 4,663 daily calls; the price rose on 64% of all days.
```

**Interpretation:**
- **BUY** with a 68% modelled chance of upside — the model's lean, not a measured edge
- **Track record**: ETF BUY calls at 14 days were right 64% of the time, and the price rose on 64% of all days, so
  the call has done about as well as simply following the market. SELL calls show the same comparison against the
  share of days the price fell. The record covers 1–30 day horizons; for other horizons, for HOLD, and where fewer than
  30 such calls were measured, none is shown.
- **Expected return** +2.8% over 14 days
- **90% CI**: the range of outcomes ($510–$560), the most useful part of the output

---

## Failure Modes

| Scenario | Why | What to Do |
|----------|-----|------------|
| **Declining markets** | The model calls "up" most of the time; in 2022 its calls were right 39.6% of the time and its probabilities were worse than a coin flip | Do not read a BUY as protection against a downturn |
| **Earnings week** | No fundamental data; can't capture earnings surprises | Check the earnings calendar |
| **Macro shocks** (Fed, CPI) | Price history only; may miss regime shifts | Wait 1–2 days post-announcement for the regime to stabilize |
| **Abstain / no calibrated distribution** | Trusted anchors missing or regime signal too weak | Skip. Do not treat absence of output or fallback text as a signal |
| **45–60d horizons** | `markovWeight` near zero; no Markov signal | Do not use beyond 30d |

---

## Practical Examples

> "I want to know if SPY is likely to be higher in 20 days for a swing trade."

**Verdict:** Use it for the range, not the direction. SPY 20d calls were 4.5pp below always calling "up" over 2021–2025
(CI [−11.0, +0.7]pp); the distribution and its interval tell you how wide the outcomes are. If the model abstains,
skip.

> "I want BTC 14d direction for a crypto trade."

**Verdict:** ❌ Do not use for direction. BTC calls were within a few points of always calling "up" at every horizon,
and the probabilities were worse than a coin flip at 5d and 10d. Any broader trade commentary the agent provides for
BTC is fallback reasoning, not validated Markov output.

---

## How to Use It Well

1. **Read the distribution and interval, not the BUY/SELL** — the central 90% interval caught 89.9–90.9% of outcomes
   in every period from 2021 to 2025, the 2022 decline included; the direction has no measured edge
2. **Stay within 30 days** — beyond that the Markov signal mixes away
3. **Combine with fundamentals** — use Cramer-Short's other tools (SEC filings, financial metrics) for earnings and
   macro context
4. **Treat an abstain as a skip**

---

## Evidence Base

- `fixtures/backtest-prices-2021-2025.json` (Yahoo Finance daily closes; rebuilt by
  `scripts/build-backtest-prices-long.ts`)
- `src/tools/finance/backtest/long-history-skill.ts`: `run <ticker>` per ticker (a few minutes each; run them in
  parallel), then `report`. Daily steps, live history windows (83 days for equities, 174 for gold, 252 for BTC with
  its live policy), traded calls scored against always calling "up" on the same steps, P(up) against a coin flip,
  95% CIs from a block bootstrap.
- Confidence-score study: `docs/reviews/price-prediction-improvements-plan-2026-11-22.md`.
