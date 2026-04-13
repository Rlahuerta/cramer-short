# Coverage Milestone Summary

**Date:** 2026-04-04
**Commit:** `7742f8d`
**Stage:** Coverage-Complete (Pending Reviewer Decision)

---

## Executive Summary

The Markov-chain directional prediction model has completed canonical validation across **14 tickers** and **6 horizons** (5–30d), yielding **84 asset × horizon cells** with fixture-backed classifications.

**Recommendation:** Treat this as **coverage-complete** for the current model version. Further expansion should require explicit reviewer justification for missing asset classes rather than automatic breadth increases.

---

## Scope Validated

| Dimension | Coverage |
|-----------|----------|
| **Tickers** | 14 (SPY, QQQ, GLD, VOO, DIA, VTI, IAU, MSFT, NVDA, GOOGL, AMZN, AAPL, TSLA, BTC-USD) |
| **Horizons** | 6 (5d, 7d, 10d, 14d, 20d, 30d) |
| **Total Cells** | 84 |
| **Harness** | `walkForward`, `WARMUP=120`, `STRIDE=10` |
| **Step Count per Cell** | n ≈ 36–38 |
| **45–60d Status** | Explicitly excluded (markovWeight near zero) |

---

## Coverage Statistics

### Overall Classification

| Classification | Cells | Percentage |
|----------------|-------|------------|
| **Confirmed** | 33 / 84 | 39% |
| **Exploratory** | 45 / 84 | 54% |
| **Negative** | 6 / 84 | 7% |

### By Asset Class

| Class | Confirmed | Exploratory | Negative | Confirmation Rate |
|-------|-----------|-------------|----------|-------------------|
| **Broad-market ETFs** (SPY, QQQ, VOO, VTI, DIA) | 17 / 30 | 13 / 30 | 0 | 57% |
| **Commodity ETFs** (GLD, IAU) | 7 / 12 | 5 / 12 | 0 | 58% |
| **Megacap Tech** (MSFT, NVDA, GOOGL, AMZN) | 9 / 24 | 15 / 24 | 0 | 38% |
| **Single-name Equity** (AAPL, TSLA) | 0 / 12 | 12 / 12 | 0 | 0% |
| **Crypto** (BTC-USD) | 0 / 6 | 0 / 6 | 6 / 6 | 0% |

### By Horizon

| Horizon | Confirmed Assets | Confirmation Rate |
|---------|-----------------|-------------------|
| **5d** | SPY, MSFT, NVDA | 3 / 14 (21%) |
| **7d** | SPY, IAU, GOOGL | 3 / 14 (21%) |
| **10d** | SPY, IAU | 2 / 14 (14%) |
| **14d** | SPY, QQQ, VOO, VTI, IAU, MSFT, NVDA | 7 / 14 (50%) |
| **20d** | SPY, QQQ, GLD, VOO, VTI, IAU, NVDA, AMZN | 8 / 14 (57%) |
| **30d** | SPY, QQQ, GLD, VOO, DIA, VTI, IAU, AMZN | 8 / 14 (57%) |

---

## Model Characterization

**The model is:**
- ✅ **Strong** for broad-market ETFs (SPY confirmed at all 6 horizons)
- ✅ **Strong** for gold/commodity ETFs at 20d+ (IAU confirmed 7d–30d, GLD confirmed 20d/30d)
- ✅ **Useful** for selected megacap tech at 14–20d (MSFT, NVDA selective confirmation)
- ⚠️ **Marginal** for single-name equities (AAPL, TSLA never clear confirmation bar)
- ❌ **Not effective** for crypto (BTC-USD negative at all horizons; best result 49.2% at 30d)

**Practical sweet spot:** **14–30d** for non-crypto assets, with **SPY / QQQ already strong by 14d** and **gold ETFs strong from 20d onward**.

---

## Evidence Trail

A reviewer can trace every classification to:

1. **Fixture data** — `src/tools/finance/fixtures/backtest-prices.json` (14 tickers, aligned dates, count ≥ 501 each)
2. **Integration test output** — `RUN_INTEGRATION=1 bun test src/tools/finance/markov-backtest.integration.test.ts` (160 tests passed, 0 failures)
3. **Docs alignment** — `docs/markov-prediction-guide.md` reflects only fixture-backed confirmed horizons
4. **Coverage matrix** — `.sisyphus/plans/canonical-coverage-matrix.md` (cell-by-cell classification with aggregate stats)
5. **Commit history** — 5 commits (`4fdd45d` → `7742f8d`) showing sequential fixture expansion and docs updates

---

## Decision Points

### Option A: Coverage-Complete (Recommended)

**Treat the current 14-ticker, 6-horizon set as sufficient** for the current model version.

**Rationale:**
- 39% confirmed coverage across 84 cells
- Broad-market and commodity exposures well-covered
- Megacap tech has selective but real confirmation
- No evidence that more overlap (another S&P 500 ETF) would add value

**Next actions under this option:**
1. Summarize the model as "good for broad-market / gold exposures and selected megacaps at 14–30d, weak for BTC/AAPL/TSLA"
2. Freeze fixture coverage at 14 tickers unless a reviewer-justified gap is identified
3. Focus future work on model improvements (not breadth expansion)

---

### Option B: Gap-Filling Wave (Only If Justified)

**Run one final validation wave focused on missing asset classes**, not more overlap.

**Candidates:**

| Ticker | Class | Why |
|--------|-------|-----|
| `IWM` | Small-cap ETF | US small-cap exposure not yet validated |
| `TLT` | Long-duration Treasuries | Bond / rate-sensitive exposure gap |
| `EFA` | Developed ex-US | International equity gap |
| `EEM` | Emerging Markets | EM equity gap |

**Constraints if this option is chosen:**
- Maximum 4 additional tickers
- Same canonical harness (no threshold tuning)
- Same 6-horizon grid (no 45–60d expansion)
- Must clear the same confirmation bar to be promoted

**Do not add:** more S&P 500 ETFs, more megacap single names, or crypto beyond BTC.

---

## Out of Scope (Explicitly Not Recommended)

| Item | Why |
|------|-----|
| **BTC revisit** | Structurally weak; best result 49.2% at 30d is below random-skill threshold |
| **45–60d Markov-success claims** | `markovWeight` near zero; attribution not supportable |
| **More single-name overlap** | Current four megacaps already show selective confirmation; adding more would increase noise without clear benefit |
| **Model tuning based on these results** | This is a coverage milestone, not a tuning point; threshold retuning would invalidate the coverage baseline |

---

## Reviewer Questions

1. **Is 39% confirmed coverage across 84 cells sufficient** to treat the model as coverage-complete for its intended use case?
2. **Are the confirmed horizons (14–30d for non-crypto)** aligned with the practical use case, or is there a specific need for shorter-horizon (5–10d) reliability?
3. **If gap-filling is desired**, which of the four candidates (`IWM`, `TLT`, `EFA`, `EEM`) addresses a real user need rather than curiosity-driven expansion?
4. **Should BTC be explicitly excluded** from future model claims, or is there a credible path to improving crypto coverage?

---

## Recommended Next Step

**Approve Option A (Coverage-Complete)** and treat the current validation set as the baseline for this model version.

If a gap-filling wave is later justified, it should:
- Target missing asset classes only (small-cap, bonds, international)
- Use the same canonical harness and confirmation thresholds
- Be explicitly scoped as "gap-filling" rather than open-ended expansion

---

**Status:** Approved — Coverage-Complete (Option A)
**Decision Date:** 2026-04-04
**Rationale:** 39% confirmed coverage across 84 cells is sufficient for the current model version; broad-market and gold exposures well-covered; megacap tech has selective confirmation; no evidence that more overlap would add value.
