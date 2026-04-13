# Exploratory asset promotion review plan

**Status:** For review, not implementation
**Date:** 2026-04-04

## Goal

Turn the current exploratory asset sweep into an executable review stage that can promote genuinely strong candidates into canonical fixture coverage without overstating uncommitted results.

This plan is for fixture expansion, canonical walk-forward verification, and reporting alignment only. It does not authorize model changes.

## Verified Findings

### Current canonical positives remain limited

The committed fixture universe was audited with `walkForward`, `WARMUP=120`, and `STRIDE=10`.

| Asset | Review-safe good horizon(s) | Notes |
|---|---|---|
| SPY | 5d, 10d, 14d, 20d, 30d | confirmed positive across the full short-to-mid band |
| QQQ | 14d, 20d, 30d | strongest confirmed tech/index case |
| GLD | 20d, 30d | confirmed mid-horizon commodity-positive case |
| BTC-USD | none | no good tested horizon |
| AAPL | none | not good in canonical coverage |
| TSLA | none | not good in canonical coverage |

### Strong exploratory candidates exist, but they are not canonical yet

The broader 3-year liquid-asset sweep is exploratory only. It does not justify reviewer-visible claims until the same assets are added to committed fixture coverage and re-run on the canonical harness.

| Tier | Assets | Strict-good exploratory horizon(s) | Why they are next |
|---|---|---|---|
| Strong ETF / commodity-ETF candidates | VOO, DIA, VTI, IAU | VOO: 5d, 10d, 14d, 20d, 30d; DIA: 10d, 14d, 20d, 30d; VTI: 5d, 14d, 20d, 30d; IAU: 7d, 10d, 20d, 30d | structurally closest to already-confirmed winners |
| Promising single-name candidates | MSFT, NVDA, GOOGL, AMZN | MSFT: 5d, 7d; NVDA: 14d; GOOGL: 20d, 30d; AMZN: 30d | worth checking, but single names need stricter scrutiny |

### Some exploratory names should not be promoted first

These assets had interesting directional hits in some slices, but they did not clear the stricter review bar of strong directional accuracy **and** strong calibrated `P(up)` directional accuracy.

| Asset | Why it stays out of the first promotion wave |
|---|---|
| IWM | direction looked good at 5d, but calibrated `P(up)` stayed too weak |
| EFA | recommendation-only strength, not strict-safe |
| EEM | recommendation-only strength, not strict-safe |
| TLT | recommendation-only strength, not strict-safe |
| SLV | did not clear the strict bar |
| USO | did not clear the strict bar |
| META | did not clear the strict bar |

### 45-60d cannot be claimed as Markov-specific success

The exploratory sweep also showed very strong long-horizon direction for several assets, but average `markovWeight` falls near zero there.

That means 45-60d outcomes must not be used to claim Markov-specific success during this promotion stage.

## Decision

Use a two-tier review framing:

1. **Confirmed canonical coverage** stays limited to SPY, QQQ, and GLD on the horizons above, while BTC-USD, AAPL, and TSLA remain negative.
2. **Promotion candidates** for the next canonical wave are VOO, DIA, VTI, and IAU first, followed by MSFT, NVDA, GOOGL, and AMZN only if the first wave stays clean.

## Non-Goals

This stage does **not** include:

- model changes
- threshold retuning
- horizon expansion beyond the existing `{5, 7, 10, 14, 20, 30}` review grid
- reopening BTC work
- treating exploratory Yahoo sweeps as canonical proof

## Proposed Next Stage

**Name:** Exploratory-to-canonical asset validation

### Task 1 — Expand committed fixture coverage for first-wave candidates

**Objective:** Add first-wave exploratory candidates to committed fixture coverage using the same structure and date alignment as the current backtest fixture.

**First-wave assets:**

- `VOO`
- `DIA`
- `VTI`
- `IAU`

**Primary files to inspect/update:**

- `src/tools/finance/backtest/download-fixtures.ts`
- `src/tools/finance/fixtures/backtest-prices.json`

**Steps:**

1. Update the fixture-generation source to include the first-wave assets.
2. Regenerate or add their historical closes using the same format as existing fixture entries.
3. Verify the new entries include aligned `dates`, `closes`, `count`, and the correct `type` field.

**QA scenario:**

- Tooling: fixture generation plus file read
- Expected result: `backtest-prices.json` contains all first-wave assets in the same shape as SPY / QQQ / GLD

### Task 2 — Run canonical walk-forward on the first-wave assets

**Objective:** Evaluate first-wave candidates using the same canonical harness parameters as the existing confirmed assets.

**Primary files to inspect/update:**

- `src/tools/finance/markov-backtest.integration.test.ts`
- `src/tools/finance/backtest/walk-forward.ts`
- `src/tools/finance/backtest/metrics.ts`

**Steps:**

1. Run the canonical harness for each first-wave asset across `{5, 7, 10, 14, 20, 30}`.
2. Record, for every asset × horizon cell:
   - directional accuracy
   - calibrated `P(up)` directional accuracy
   - Brier score
   - average `markovWeight`
3. Capture the exact output in a reviewer-readable table.

**QA scenario:**

- Tooling: canonical integration harness and metrics output
- Expected result: every first-wave asset has a complete six-horizon result table with no missing cells

### Task 3 — Classify first-wave results against the strict review bar

**Objective:** Decide which first-wave assets truly join the confirmed set.

**Classification rule to preserve:**

- exploratory promise is not enough
- a review-safe “good” horizon requires strong directional accuracy **and** strong calibrated `P(up)` directional accuracy
- 45-60d results remain excluded from Markov-specific claims

**Steps:**

1. Compare each first-wave result table against the current confirmed set.
2. Mark each horizon as:
   - confirmed positive
   - weak / mixed
   - negative
3. Choose the practical best horizon band for each passing asset.
4. Explicitly document any failures; do not silently drop them.

**QA scenario:**

- Tooling: measured output review only
- Expected result: every classification is backed by quoted metrics and no asset is promoted on recommendation-only strength

### Task 4 — Decide whether to open the second wave

**Objective:** Only after the ETF / commodity-ETF wave is settled, decide whether single-name candidates deserve canonical fixture expansion.

**Second-wave candidates:**

- `MSFT`
- `NVDA`
- `GOOGL`
- `AMZN`

**Steps:**

1. Review whether the first-wave process stayed stable and reviewer-clear.
2. If yes, repeat Tasks 1-3 for the second-wave names.
3. Apply stricter skepticism to single-name promotions than to broad ETFs.

**QA scenario:**

- Tooling: same as Tasks 1-3
- Expected result: no single-name asset is promoted without clearing the same strict bar on the canonical harness

### Task 5 — Align docs after canonical validation completes

**Objective:** Update the Markov guide only after the canonical fixture stage settles the new assets.

**Primary files to inspect/update:**

- `docs/markov-prediction-guide.md`

**Steps:**

1. Move any newly confirmed assets from the exploratory table into the confirmed table.
2. Leave failed or mixed candidates in the exploratory section with explicit status.
3. Keep BTC-USD, AAPL, and TSLA negative unless new canonical evidence actually changes them.
4. Preserve the long-horizon attribution caveat.

**QA scenario:**

- Tooling: file read and diff review
- Expected result: no asset appears as both confirmed and exploratory, and no exploratory result is presented as already canonical

## Required Evidence

A reviewer should expect to see:

1. the fixture diff showing first-wave asset coverage was added
2. a complete canonical result table for every first-wave asset across `{5, 7, 10, 14, 20, 30}`
3. the classification sheet showing which horizons passed and which failed
4. the final doc diff, if any assets are promoted after canonical validation

## Risks

| Risk | Why it matters |
|---|---|
| Exploratory results get mistaken for proof | This would overstate uncommitted evidence. |
| Recommendation-only wins get promoted | Weak calibrated `P(up)` agreement is not review-safe. |
| Broad ETF candidates are assumed to match SPY/GLD automatically | Similarity is a reason to test, not a result. |
| Second-wave single names expand scope too early | Single names are noisier and easier to overclaim. |
| Long-horizon wins get reframed as Markov-specific | `markovWeight` is too low there to support that attribution. |

## Review Questions

1. Are the first-wave candidates scoped narrowly enough to keep the validation stage reviewer-friendly?
2. Is the distinction between canonical and exploratory evidence impossible to miss?
3. Does the plan force explicit documentation of failures, not just successes?
4. Is the second wave gated clearly enough behind first-wave results?
5. Does the plan preserve the 45-60d attribution caveat strongly enough?
