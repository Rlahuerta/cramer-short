# PI-53: Empty Polymarket market lists are labelled 'terminal' semantics

- Severity: L
- Area: C1c
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/backtest/arbiter-replay-runner.ts:110` — `return (Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] as ForecastMarketSemantics | undefined) ?? 'unknown';`, `src/tools/finance/backtest/combined-short-horizon-benchmark.ts:81` — `const primary = (Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] as ForecastMarketSemantics | undefined) ?? 'unknown';`
- Repro kind: runtime

## Summary

Both `primarySemantics()` (arbiter replay runner) and `summarizeSemantics()` (combined short-horizon benchmark) pick the highest-count entry from a six-key semantics counter and rely on `?? 'unknown'` for the no-market case. That fallback is unreachable: when a replay bundle has no Polymarket markets every count is 0, the stable sort keeps `'terminal'` first, so the bundle is reported as `terminal` and a spurious `terminal` row is added to the semantic calibration report.

## Root cause

`counts` is always initialised with all six semantics keys (`terminal`, `barrier_touch`, `range`, `path_dependent`, `ambiguous`, `unknown`), so `Object.entries(counts).sort(...)[0]` is always defined; the `?? 'unknown'` fallback can never fire. With all counts equal to 0 the stable sort returns insertion order, i.e. `terminal`. The same expression is duplicated in `combined-short-horizon-benchmark.ts:81` inside `summarizeSemantics()`.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-53-empty-market-list-labeled-terminal-semantics.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-53-empty-market-list-labeled-terminal-semantics.ts
```

Expected: row.primarySemantics = "unknown"; buckets = ["unknown"]

Actual: row.primarySemantics = "terminal"; buckets = ["terminal"]

OBSERVED:
    row.primarySemantics = "terminal"
    semanticBucketCalibration keys = ["terminal"]

## Impact

Semantic-bucket calibration mixes market-less bundles into the `terminal` bucket, so the reported mean predicted probability and actual frequency for "terminal" markets are computed over rows that had no market at all. The combined benchmark's `semanticSummary.primaryPolymarketSemantics` is wrong in the same way.

## Suggested fix direction (not applied)

Guard the empty counter explicitly before the sort, e.g. `const total = Object.values(counts).reduce((sum, value) => sum + value, 0); if (total === 0) return 'unknown';` in both files. (Alternatively return `'unknown'` when the top count is 0.)

## Evidence

- Reproduction script: `issues/repro/PI-53-empty-market-list-labeled-terminal-semantics.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-4-C1c.txt`.
