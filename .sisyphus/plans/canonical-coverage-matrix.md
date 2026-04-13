# Canonical Coverage Matrix

**Generated:** 2026-04-04
**Commit:** `f3c63a9` — validate second-wave Markov asset candidates
**Harness:** `walkForward`, `WARMUP=120`, `STRIDE=10`, horizons `{5, 7, 10, 14, 20, 30}`

---

## Summary

This matrix captures all fixture-backed canonical validation results for the Markov-chain directional prediction model. Each asset × horizon cell is classified as:

| Classification | Criteria |
|----------------|----------|
| **Confirmed** | Directional accuracy ≥ 55%, Brier < 0.35, CI coverage ≥ 70%, step count ≥ 35 |
| **Exploratory** | Tested but did not clear all confirmation thresholds; includes marginal results |
| **Negative** | Directional accuracy < 50% or Brier > 0.42 across all tested horizons |

**45–60d horizons are explicitly excluded** from Markov-success claims because `markovWeight` falls near zero at those ranges.

---

## Coverage Matrix

| Asset | 5d | 7d | 10d | 14d | 20d | 30d | Notes |
|-------|----|----|-----|-----|-----|-----|-------|
| **SPY** | ✅ Confirmed | ✅ Confirmed | ✅ Confirmed | ✅ Confirmed | ✅ Confirmed | ✅ Confirmed | Broad-market index; strongest overall coverage |
| **QQQ** | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ✅ Confirmed | ✅ Confirmed | ✅ Confirmed | Tech-heavy index; mid-horizon strength |
| **GLD** | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ✅ Confirmed | ✅ Confirmed | Gold ETF; best at 20d+ |
| **VOO** | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ✅ Confirmed | ✅ Confirmed | ✅ Confirmed | S&P 500 ETF; mirrors SPY pattern |
| **DIA** | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ✅ Confirmed | Dow Jones ETF; only 30d clears bar |
| **VTI** | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ✅ Confirmed | ✅ Confirmed | ✅ Confirmed | Total market ETF; broad mid-horizon signal |
| **IAU** | ⚠️ Exploratory | ✅ Confirmed | ✅ Confirmed | ✅ Confirmed | ✅ Confirmed | ✅ Confirmed | Gold ETF; strongest commodity coverage |
| **MSFT** | ✅ Confirmed | ⚠️ Exploratory | ⚠️ Exploratory | ✅ Confirmed | ⚠️ Exploratory | ⚠️ Exploratory | Megacap tech; selective short/mid confirmation |
| **NVDA** | ✅ Confirmed | ⚠️ Exploratory | ⚠️ Exploratory | ✅ Confirmed | ✅ Confirmed | ⚠️ Exploratory | Semiconductor; strong 5d/14d/20d |
| **GOOGL** | ⚠️ Exploratory | ✅ Confirmed | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | Megacap tech; narrow 7d confirmation |
| **AMZN** | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ✅ Confirmed | ✅ Confirmed | Megacap tech; mid-horizon band only |
| **AAPL** | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | Single-name equity; never clears confirmation bar |
| **TSLA** | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | ⚠️ Exploratory | High volatility; marginal across all horizons |
| **BTC-USD** | ❌ Negative | ❌ Negative | ❌ Negative | ❌ Negative | ❌ Negative | ❌ Negative | Best observed: 49.2% at 30d; no good horizon |

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | **Confirmed** — clears all confirmation thresholds |
| ⚠️ | **Exploratory** — tested but marginal or incomplete |
| ❌ | **Negative** — no good tested horizon |

---

## Aggregate Statistics

### By Asset Class

| Class | Assets | Confirmed Cells | Exploratory Cells | Negative Cells |
|-------|--------|-----------------|-------------------|----------------|
| **Broad-market ETFs** | SPY, QQQ, VOO, VTI, DIA | 17 / 30 (57%) | 13 / 30 (43%) | 0 |
| **Commodity ETFs** | GLD, IAU | 7 / 12 (58%) | 5 / 12 (42%) | 0 |
| **Megacap Tech** | MSFT, NVDA, GOOGL, AMZN | 9 / 24 (38%) | 15 / 24 (62%) | 0 |
| **Single-name Equity** | AAPL, TSLA | 0 / 12 (0%) | 12 / 12 (100%) | 0 |
| **Crypto** | BTC-USD | 0 / 6 (0%) | 0 / 6 (0%) | 6 / 6 (100%) |

### By Horizon

| Horizon | Confirmed Assets | Exploratory Assets | Negative Assets |
|---------|-----------------|-------------------|-----------------|
| **5d** | SPY, MSFT, NVDA | QQQ, GLD, VOO, DIA, VTI, IAU, GOOGL, AMZN, AAPL, TSLA | BTC-USD |
| **7d** | SPY, IAU, GOOGL | QQQ, GLD, VOO, DIA, VTI, MSFT, NVDA, AMZN, AAPL, TSLA | BTC-USD |
| **10d** | SPY, IAU | QQQ, GLD, VOO, DIA, VTI, MSFT, NVDA, GOOGL, AMZN, AAPL, TSLA | BTC-USD |
| **14d** | SPY, QQQ, VOO, VTI, IAU, MSFT, NVDA | GLD, DIA, GOOGL, AMZN, AAPL, TSLA | BTC-USD |
| **20d** | SPY, QQQ, GLD, VOO, VTI, IAU, NVDA, AMZN | DIA, MSFT, GOOGL, AAPL, TSLA | BTC-USD |
| **30d** | SPY, QQQ, GLD, VOO, DIA, VTI, IAU, AMZN | MSFT, NVDA, GOOGL, AAPL, TSLA | BTC-USD |

---

## Key Findings

### Strongest Coverage

1. **SPY** — only asset confirmed at all six horizons (5–30d)
2. **IAU** — confirmed at 7d through 30d; strongest commodity exposure
3. **QQQ, VOO, VTI** — confirmed from 14d onward; reliable mid-horizon signals
4. **GLD** — confirmed at 20d/30d; gold exposure improves with horizon
5. **NVDA** — confirmed at 5d/14d/20d; strongest single-name tech coverage

### Weakest Coverage

1. **BTC-USD** — negative at all horizons; best result 49.2% at 30d
2. **AAPL** — exploratory at all horizons; never clears confirmation bar
3. **TSLA** — exploratory at all horizons; high volatility overwhelms signal
4. **DIA** — only confirmed at 30d; Dow Jones exposure is narrow
5. **GOOGL** — only confirmed at 7d; otherwise exploratory

### Horizon Patterns

- **5–10d band:** Only SPY, MSFT, NVDA, IAU, and GOOGL have any confirmed coverage. Most assets remain exploratory here.
- **14d band:** First broad confirmation wave — SPY, QQQ, VOO, VTI, IAU, MSFT, NVDA all clear the bar.
- **20–30d band:** Strongest overall — SPY, QQQ, GLD, VOO, VTI, IAU, AMZN all confirmed; DIA joins at 30d.

---

## Attribution Caveats

| Issue | Why it matters |
|-------|----------------|
| **45–60d excluded** | `markovWeight` is near zero at those horizons; wins cannot be attributed to Markov signal |
| **Single-name noisier** | Megacaps show selective confirmation but are less reliable than broad ETFs |
| **BTC structurally weak** | Even best-case 30d result (49.2%) is below random-skill threshold |
| **Sample size limits** | Each cell has n ≈ 36–38 steps; 95% CI on accuracy is ±8–10pp |

---

## Next-Stage Recommendations

### Do Next (Coverage-Complete Stage)

1. **Treat this as coverage-complete** for the current model version — no automatic expansion without a specific use case.
2. **Summarize the model** as "good for broad-market / gold exposures and selected megacaps at 14–30d, weak for BTC/AAPL/TSLA."
3. **Freeze fixture coverage** at the current 14 tickers unless a reviewer-justified gap is identified.

### Only If Needed (Gap-Filling Wave)

If broader asset-class coverage is still required, run one final wave focused on missing classes:

| Candidate | Class | Why |
|-----------|-------|-----|
| `IWM` | Small-cap ETF | Not yet canonical; exploratory only so far |
| `TLT` | Long-duration Treasuries | Bond exposure not yet validated |
| `EFA` | Developed ex-US | International equity gap |
| `EEM` | Emerging Markets | EM equity gap |

Do **not** add more overlap (e.g., another S&P 500 ETF) — the current set already covers that exposure.

### Do Not Do Next

- No model tuning or threshold retuning based on these results
- No BTC revisit without new data or structural changes
- No 45–60d Markov-success claims
- No expansion into single-name equities beyond the current four megacaps

---

## Evidence Trail

A reviewer should be able to trace every classification back to:

1. **Fixture diff** — `src/tools/finance/fixtures/backtest-prices.json` contains all 14 tickers with aligned dates and `count ≥ 501`
2. **Integration test output** — `RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts` printed per-ticker per-horizon metrics
3. **Docs update** — `docs/markov-prediction-guide.md` reflects only the fixture-backed confirmed horizons
4. **Commit history** — first wave (`af97e4d`) and second wave (`f3c63a9`) commits show the exact fixture and test expansion

---

## Version History

| Date | Commit | Change |
|------|--------|--------|
| 2026-04-04 | `f3c63a9` | Second-wave validation: MSFT, NVDA, GOOGL, AMZN |
| 2026-04-04 | `af97e4d` | First-wave validation: VOO, DIA, VTI, IAU |
| 2026-04-04 | `4c0f349` | Docs split: confirmed vs exploratory candidates |
| 2026-04-04 | `4fdd45d` | Horizon audit summary added to docs |

---

**Status:** Coverage-complete pending reviewer decision on gap-filling wave.
