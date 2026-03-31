---
name: trump-pressure
description: Analyze Trump political pressure levels and estimate the probability of a policy reversal (TACO event). Uses the Trump Pressure Index tool combining market data, Treasury yields, inflation expectations, approval ratings, gas prices, Polymarket predictions, and social sentiment.
---

# Trump Pressure Index Analysis

Perform a comprehensive analysis of political/market pressure on Trump and estimate TACO (Trump Always Chickens Out) probability.

## Steps

1. **Compute the Pressure Index** — Call the `trump_pressure_index` tool to get the current composite score, regime classification, and TACO probability.

2. **Interpret the regime** — Explain what the current pressure level means:
   - 🟢 **LOW** (<0.5σ): Markets calm, low political pressure. Policy likely stays on course.
   - 🟡 **MODERATE** (0.5–1.5σ): Growing unease. Watch for rhetoric shifts.
   - 🟠 **ELEVATED** (1.5–2.0σ): Significant stress. Historical precedent suggests policy adjustments incoming.
   - 🔴 **CRITICAL** (≥2.0σ): Extreme pressure. TACO event highly probable within 5–10 trading days based on historical patterns.

3. **Analyze component breakdown** — Identify which inputs are driving pressure:
   - S&P 500 decline (his "favorite report card")
   - Rising Treasury yields (government borrowing pain)
   - Inflation expectations surging (unhappy voters)
   - Falling approval ratings
   - Gas price spikes (consumer pain amplifier)
   - Polymarket policy-reversal markets (real-money bets)
   - Negative social sentiment

4. **Compare with historical TACO events** — Reference the nearest landmark:
   - Liberation Day (Apr 2025): 2.8σ → 90-day tariff pause, S&P +9.5% single day
   - US-China de-escalation (May 2025): 2.3σ → tariffs reduced 145% to 30%
   - UK trade deal (Jul 2025): 1.5σ → first bilateral deal
   - Auto tariff extension (Jan 2026): 1.8σ → sector exemption

5. **Assess TACO probability** — Explain the blended probability:
   - Markov component: regime transition probability from Monte Carlo simulation
   - Polymarket component: real-money policy-reversal market prices
   - Combined weight: 40% Markov + 60% Polymarket

6. **Provide actionable interpretation** — Based on the analysis:
   - **If CRITICAL**: "Position for a snap-back rally. Historical TACO events led to 5–10% single-day moves."
   - **If ELEVATED**: "Reduce tariff-sensitive exposure. Monitor for rhetoric shifts."
   - **If MODERATE**: "Maintain positions but increase hedges on policy-sensitive sectors."
   - **If LOW**: "No immediate policy reversal expected. Focus on fundamentals."

7. **Flag any warnings** — Highlight data quality issues, structural breaks, or missing components that affect reliability.
