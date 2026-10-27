# PI-16: Backtest metrics read the conservative CI as a central-90 interval

- Severity: M
- Area: C1c
- Status: Partially fixed (by decision)
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/backtest/metrics.ts:533` — `const sigma = width / (2 * CENTRAL_90_Z);`, `src/tools/finance/backtest/metrics.ts:851` — `const lower = Math.min(step.ciLower, step.ciUpper);`, `src/tools/finance/backtest/comparison-harness.ts:129` — `const overallCoverage = ciCoverage(steps);`, `src/tools/finance/backtest/comparison-harness.ts:130` — `const breakCoverage = breakSteps.length > 0 ? ciCoverage(breakSteps) : null;`
- Repro kind: runtime

## Resolution

Decision: score the literal central-90 where the metric is about the distribution, keep the helper interval where round 6 compares arms. `deriveIntervalScale` (CRPS, scaled CRPS, tail weights) and `murphyWinklerDecomposition` now read the literal central-90 bounds, falling back to `ciLower`/`ciUpper` when absent. `summarizeCalibrationSteps` keeps its helper-interval `coverageError`/`breakCoverageError` and adds `central90Coverage`, `central90CoverageError`, `breakCentral90Coverage` and `breakCentral90CoverageError` beside them, because the adaptive-conformal path rewrites `ciLower`/`ciUpper` but not `central90*`; switching the existing columns would make baseline and conformal arms compare identically in round 6's adopt/reject rule. The stale `BacktestStep.ciLower` "90% CI" comment was corrected.

The repro's first two assertions (CRPS equal for an identical central-90; Murphy-Winkler `meanWidth = 10`) now pass. Its third (`coverageError = 0.9`) is intentionally unmet; that value is `central90CoverageError`. Python `BacktestStep` has no central-90 fields, so Python CRPS/Murphy-Winkler still score `ci_lower`/`ci_upper` (recorded in `research/PARITY.md`).

## Summary

The walk-forward and replay engines store a deliberately conservative interval (survival thresholds 0.995/0.005, roughly a 99% interval) in `ciLower`/`ciUpper`, and the literal central-90 interval in `central90CiLower`/`central90CiUpper`. Three metric paths assume `ciLower`/`ciUpper` is a central-90 interval: `deriveIntervalScale` (drives CRPS, scaled CRPS, tail weights), `murphyWinklerDecomposition`, and `summarizeCalibrationSteps.coverageError`/`breakCoverageError`. Their reported numbers are therefore computed against the wrong interval on every walk-forward step.

## Root cause

`walk-forward.ts:328` and `replay.ts:220` populate `ciLower`/`ciUpper` from `extractCI(...)`, which uses `extractSurvivalInterval(dist, currentPrice, 0.995, 0.005)` — the ~99% interval — while `central90CiLower`/`central90CiUpper` carry the literal 90% interval. `deriveIntervalScale` divides the conservative width by `2 * CENTRAL_90_Z`, treating it as a 90% interval; `murphyWinklerDecomposition(alpha = 0.10)` measures width/misses against the conservative interval while claiming a 90% nominal score; `summarizeCalibrationSteps` measures coverage with `ciCoverage()` against a `targetCiCoverage` of 0.9 in its only caller (`btc-multi-horizon-comparison-round6.ts:79,199`).

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-16-backtest-ci-treats-conservative-interval-as-central-90.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-16-backtest-ci-treats-conservative-interval-as-central-90.ts
```

Expected: CRPS equal for identical central-90; MW meanWidth=10; coverageError=0.9

Actual: CRPS changed when only the conservative helper interval changed (central-90 is identical)

OBSERVED:
    crps(ci=95..105, no central90) = 0.840452
    crps(ci=90..110, central90=95..105) = 1.486237
    murphyWinkler meanWidth = 20.000000
    summarizeCalibrationSteps coverageError = 0.100000
    centralIntervalCoverage(0.9) = 0.000000

## Impact

Every `generateReport` CRPS/scaledCrps/tailWeightedCrps and Murphy-Winkler number is computed against the ~99% interval instead of the 90% interval, and round6's `covErr`/`breakCovErr` columns are systematically understated (~0.1 instead of ~0.9 for a completely uncovered slice). Model comparisons that rely on these scores can prefer a wider interval or overlook real coverage failure.

## Suggested fix direction (not applied)

- `metrics.ts:531-535`: derive the scale from the literal central-90 bounds first: `const bounds = resolveCentralIntervalBounds(step, 0.9); const width = Math.abs(bounds.upper - bounds.lower); const sigma = width / (2 * CENTRAL_90_Z);`.
- `metrics.ts:850-852`: replace the `Math.min/max(step.ciLower, step.ciUpper)` pair with `const { lower, upper } = resolveCentralIntervalBounds(step, 0.9);` for the default `alpha = 0.10` (map 0.05 → 0.95 and 0.01 → 0.99 if the parameter must stay general).
- `comparison-harness.ts:129-130`: use the already-imported `centralIntervalCoverage`: `const overallCoverage = centralIntervalCoverage(steps, 0.9);` and `const breakCoverage = breakSteps.length > 0 ? centralIntervalCoverage(breakSteps, 0.9) : null;`. Keep `ciCoverage` in the returned `ciCoverage` field if callers want the conservative number, but do not use it for `coverageError`.

## Evidence

- Reproduction script: `issues/repro/PI-16-backtest-ci-treats-conservative-interval-as-central-90.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-4-C1c.txt`.
