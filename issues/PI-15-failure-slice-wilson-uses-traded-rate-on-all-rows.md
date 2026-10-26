# PI-15: Failure-slice Wilson CI multiplies a traded-only accuracy by every bucket row

- Severity: M
- Area: C1c
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/backtest/btc-failure-analysis.ts:206` — `return Math.round(row.directionalAccuracy * row.count);`, `src/tools/finance/backtest/btc-failure-analysis.ts:217` — `const accuracyCI = wilsonCI(toSuccessCount(row), row.count, ciAlpha);`
- Repro kind: runtime

## Summary

`metrics.directionalAccuracy()` is a hit rate over traded (BUY/SELL) rows only. `btc-failure-analysis.toSuccessCount()` treats it as a per-row success rate and multiplies it by the bucket's full row count, HOLD rows included. The Wilson interval is then computed from a fabricated sample: a bucket with 2 correct out of 5 traded rows inside 20 total rows reports `wilson(8, 20)` instead of `wilson(2, 5)`, which is materially narrower and can pass the `passesWilsonScreen` filter (`accuracyCI.upper < aggregateAccuracy`) that decides whether a slice is an actionable weakness.

## Root cause

`bucketRow` (`metrics.ts:913-942`) reports `count` = all steps and `directionalAccuracy` = `directionalAccuracy(traded steps)`; the traded denominator is not carried on `BucketedMetricRow`. `toSuccessCount(row)` reconstructs successes as `round(tradedRate * count)`, mixing the two denominators, and `wilsonCI(...)` is called with `row.count` as `total`. Any bucket that mixes HOLD and traded steps gets an incorrect interval; the effect grows as the HOLD share grows.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-15-failure-slice-wilson-uses-traded-rate-on-all-rows.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-15-failure-slice-wilson-uses-traded-rate-on-all-rows.ts
```

Expected: accuracyCI = [0.117621, 0.769276] from 2/5 traded rows

Actual: accuracyCI = [0.218807, 0.613418] from 8/20 rows (HOLDs counted as misses)

OBSERVED:
    bucket rows=20 traded-only accuracy=0.400000
    reported CI = [0.218807, 0.613418]
    wilson(2, 5) = [0.117621, 0.769276]
    wilson(round(0.4*20)=8, 20) = [0.218807, 0.613418]

## Impact

The failure-slice ranking reports confidence intervals that overstate the sample size and understate uncertainty, so buckets dominated by abstentions can clear the Wilson screen and be promoted to an "actionable" verdict even though almost none of their rows were traded. The same wrong interval is rendered in the BTC failure-slice integration-test report.

## Suggested fix direction (not applied)

Carry the traded denominator through the ranking. In `metrics.ts:913-942` add `tradedCount: bucketSteps.filter(step => step.recommendation !== 'HOLD').length` to `BucketedMetricRow` (initialise it to 0 in the empty branch), then in `btc-failure-analysis.ts` compute `accuracyCI = wilsonCI(correctTradedRows, tradedCount, ciAlpha)` — deriving successes from the traded subset rather than `round(rate * count)` — and skip (or mark undefined) buckets whose `tradedCount` is below `minBucketN`.

## Evidence

- Reproduction script: `issues/repro/PI-15-failure-slice-wilson-uses-traded-rate-on-all-rows.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-4-C1c.txt`.
