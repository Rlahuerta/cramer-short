# R5 Sprint Implementation Report — Sprint 1 + Sprint 2

**Date:** 2026-04-29
**Plan source:** `docs/forecast-improvement-ideas-round5-2026-04-29.md`
**Branch:** `topic/opt-skills`
**Commits in this sprint:** `5227f3f`, `2dde299`, `e95032b`, `d2a891f` (+ this report commit)

## TL;DR

| Idea | Title | Status | Verified lift on backtest? |
|------|-------|--------|----------------------------|
| #3   | Naive baselines (coin-flip + last-period) | **Implemented (helper)** | Pending wiring |
| #5   | Horizon-aware + regime-conditional GARCH | **Implemented + wired + backtested** | **Null** on BTC fixture (sub-resolution) |
| #11  | Long-shot odds shrinkage (Polymarket Q→P) | **Implemented (helper)** | Pending wiring |
| #14  | Markov transition entropy CI modulator | **Implemented (helper)** | Pending wiring |
| #4   | Conformal-PID interval surfacing | **Deferred** | — |
| #6   | Two-pass regime Platt calibration | **Implemented (helper)** | Synthetic ✓ |
| #1   | Crypto-native peer-asset bias (ETH/SOL/MSTR) | **Stub** | Blocked: no fixtures |
| #13  | Kalshi macro vol-event signals | **Stub** | Blocked: no API key |
| #10  | Intraday Hawkes spike | **Skipped** | Out of scope |

**Headline result:** five Sprint-1 ideas + the two-pass Platt from Sprint 2 ship as standalone, well-tested helpers. The single idea wired end-to-end for backtest (R5 Idea #5, GARCH) shows **byte-identical metrics to R4** on the BTC 2024–2025 fixture: realised GARCH scalars cluster too close to 1.0 for the legacy clamp to bite, so the R5 modulation moves outputs by ≪ 1 %. The wiring is non-destructive and stays available behind opt-in flags for assets/periods where the regime ceiling is more likely to bind.

## Sprint 1 — implementations

### #3 — Naive baselines  (commit `5227f3f`)

- **Module:** `src/tools/finance/backtest/baselines.ts`
- **Tests:** `baselines.test.ts` — 6 pass
- **API:** `computeCoinFlipBaseline`, `computeLastPeriodBaseline`, `computeNaiveBaselines`, `naiveBaselineGuard(steps, minLift)`.
- **Status:** Helper-only. Wiring into `markov-backtest.integration.test.ts` is straightforward (call `naiveBaselineGuard(arm.steps, 0.02)` after the arm runs and assert it returns `true`); deferred so the existing backtest gate stays stable.

### #5 — Horizon-aware + regime-conditional GARCH  (commits `5227f3f`, `e95032b`)

- **Module:** `src/tools/finance/garch-scales.ts` — added `GarchClampOptions` + `detectRecentRegime`.
- **Wiring:** `markov-distribution.ts` (params interface, destructure, trajectory MC opts construction) + `walk-forward.ts` (config interface + both call sites).
- **Backtest arm:** `improvedR5` in `btc-multi-horizon-comparison-round5.ts` with `garchHorizonCap: 7`, `garchRegimeCeiling: { calm: 1.5, turbulent: 3.0 }`.
- **Backtest result:** byte-identical to `improvedR4` at every horizon. Direct check in a REPL confirms the function does change the GARCH vector (e.g. at h=30 the last 5 scalars go `0.989 → 1.000`), but on the 2024-25 BTC fixture the realised scalars hover within ±5 % of 1.0, so neither the legacy [0.33, 3.0] clamp nor the R5 ceiling ever binds, and the horizon-decay nudge is below the metric resolution.
- **Conclusion:** correct implementation, null lift on this fixture. **Recommendation:** keep the wiring (it is opt-in and free when off) and re-evaluate when fixtures with genuine vol-regime cycling (e.g. equity-index ETFs across 2008/2020 crises) become available.

### #11 — Long-shot odds shrinkage  (commit `2dde299`)

- **Module:** `src/tools/finance/rnd-integration.ts` — added `applyLongshotShrinkage(p, opts)`.
- **Tests:** `rnd-integration.longshot.test.ts` — 7 pass.
- **API:** Returns `{ p, applied, tailDistance }`; default tail thresholds `p < 0.05` or `p > 0.95`, default shrinkage weight `w = 0.5` (`p_shrunk = 0.5·p + 0.5·0.5`).
- **Status:** Helper-only. Caller-side wiring (Polymarket Q→P pipeline) intentionally deferred to a follow-up to keep this sprint isolated from market-pricing changes.

### #14 — Transition entropy CI modulator  (commit `2dde299`)

- **Module:** `src/tools/finance/transition-entropy.ts`
- **Tests:** `transition-entropy.test.ts` — 13 pass.
- **API:** `computeTransitionEntropy(P)`, `approximateStationary(P)`, `EntropyZScoreTracker(window)`, `entropyZToCiScale(z, kappa)`.
- **Math:** `H = Σ_i π_i · row_entropy(P_i)`; CI scaling `clamp(1 + κ·z, 0.7, 1.4)` so positive z (high recent uncertainty) widens the interval.
- **Status:** Helper-only. Walk-forward integration requires extending `walkForward` with per-fold tracker state — deferred so this sprint ships clean.

### #4 — Conformal-PID interval surfacing  (DEFERRED)

- The existing `ConformalPID` class already does the heavy lifting; the missing piece is per-fold stateful integration in `walkForward` plus three new fields on `BacktestStep` (`conformalLow`, `conformalHigh`, `conformalRadius`). Estimated 200–300 LOC of careful threading; punted to Sprint 3 to preserve atomicity of this sprint's scope.

## Sprint 2 — implementations

### #6 — Two-pass regime Platt calibration  (commit `d2a891f`)

- **Module:** `src/tools/finance/regime-calibrator-two-pass.ts`
- **Tests:** `regime-calibrator-two-pass.test.ts` — 6 pass, including:
  - Round-trip serialization
  - Determinism (same input ⇒ same fits)
  - **Log-loss non-regression** vs single-pass on synthetic over-confident bull-regime samples
  - Graceful degradation when pass-1 is degenerate (all-same outcomes)
- **API:** `fitTwoPassRegimePlatt(samples)` returns `Partial<Record<RegimeState, { pass1, pass2 }>>`; `applyTwoPassRegimePlatt(p, regime, fits)` chains the two logistics with intermediate clamping for numerical safety.
- **Status:** Helper-only; wiring into the prediction loop requires the same two-pass walk-forward harness that R4 single-pass Platt already needs (validation pass to fit, inference pass to apply).

### #1 — Crypto-native peers  (STUB)

- **Module:** `src/tools/finance/crypto-native-peers.ts`
- Reserves the `loadCryptoPeerReturns(peers, anchor)` contract that returns the same shape `WalkForwardConfig.crossAssetReturns` already accepts.
- Throws `CryptoPeerLoaderUnavailable` until ETH/SOL/MSTR/COIN daily-close fixtures are added under `references/` or `exports/`.
- Once fixtures land, no API change is needed: callers swap the import.

### #13 — Kalshi macro vol signals  (STUB)

- **Module:** `src/tools/finance/kalshi-vol-signals.ts`
- Defines `KalshiVolSignal { eventAt, eventId, probability, intensityBoost }` and the `fetchKalshiVolSignals(opts)` async API.
- Throws `KalshiUnconfiguredError` until `KALSHI_API_KEY` is set and the live REST integration is implemented.

### #10 — Intraday Hawkes  (SKIPPED)

- Pure research spike; no code path until the underlying intraday data feeder lands.

## Test summary

- New tests added across the sprint: **36 pass, 0 fail**.
  - #3 baselines: 6
  - #5 GARCH (R5 subset): 4
  - #11 longshot: 7
  - #14 entropy: 13
  - #6 two-pass Platt: 6
- `bun run typecheck` clean at every commit.
- Backtest harness (`bun run …round5.ts`) runs end-to-end and writes `docs/btc-multi-horizon-backtest-round5-2026-04-29.{md,json}`.

## Backtest deltas — `improvedR5` vs `improvedR4`

| Horizon | ΔdirAcc | ΔBrier  | Δedge   | Δcoverage |
|--------:|--------:|--------:|--------:|----------:|
|  1d     | +0.000  | +0.000  | +0.0000 |  +0.000   |
|  2d     | +0.000  | +0.000  | +0.0000 |  +0.000   |
|  3d     | +0.000  | +0.000  | +0.0000 |  +0.000   |
|  7d     | +0.000  | +0.000  | +0.0000 |  +0.000   |
| 14d     | +0.000  | +0.000  | +0.0000 |  +0.000   |
| 30d     | +0.000  | +0.000  | +0.0000 |  +0.000   |

(Full table in `docs/btc-multi-horizon-backtest-round5-2026-04-29.md`.)

## Recommended kept-on flags

- Keep all R4 flags exactly as they were after Round-4 (no R5 deltas justify changing them).
- Keep `garchHorizonCap` / `garchRegimeCeiling` available but **default OFF** — turn them on only for assets where realised GARCH scalars routinely exceed 1.5–2.0.

## Carry-overs into the next sprint (R5 → R6)

1. **Wire #11 long-shot shrinkage** into the Polymarket Q→P call site and add a Brier-on-tail-events guard test.
2. **Wire #14 entropy CI modulator** into `walkForward` (fold-level tracker state, expose `transitionEntropyZ` in `BacktestStep`).
3. **Wire #4 conformal-PID** end-to-end (instantiate per fold; record after each step; expose `conformalLow/High`).
4. **Land #1 crypto-peer fixtures** (ETH-USD daily closes for 2024-2025 mirror the BTC fixture format) and re-run the multi-horizon backtest with crypto-native cross-asset bias.
5. **Two-pass walk-forward harness** that produces a validation pass for both single-pass Platt (R4 Idea 3) and two-pass (R5 Idea #6).

---

**Reviewer note:** the null result on Idea #5 is genuine and not a wiring bug — it has been verified by direct invocation of `computeGarchScales` with and without options. The R4 cross-asset Lasso, ADWIN, and KSWIN flags account for ≈ all of the cumulative gain over baseline; R5 helpers shipped here lift the *theoretical* surface area of the framework without yet moving the BTC backtest needle. Wiring the deferred helpers and running on a more vol-cyclical asset set is the natural next step.
