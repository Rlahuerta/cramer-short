# PI-17: Selective directional accuracy still scores HOLD abstentions as correct

- Severity: M
- Area: C1c
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/backtest/metrics.ts:1301` — `return Math.abs(s.actualReturn) < holdThreshold;`
- Repro kind: runtime

## Summary

`f40143d` redefined `directionalAccuracy()` to score only traded (BUY/SELL) steps and to treat HOLD as an abstention, because a fixed `|r| < 3%` HOLD band let always-HOLD score ~78%. `selectiveDirectionalAccuracy()` was not updated: inside the confidence-selected set it still counts a HOLD as correct whenever `|actualReturn| < holdThreshold`. The selective accuracy and the risk-coverage curve built on it therefore still reward abstention — a model that never trades reports 100% selective accuracy at full coverage while `directionalAccuracy()` reports 0.

## Root cause

`selectiveDirectionalAccuracy` (`metrics.ts:1288-1310`) builds `selected` from the confidence threshold, then `correct` counts BUY→`r>0`, SELL→`r<0`, and everything else (HOLD) as correct when `|r| < holdThreshold`. The denominator stays `selected.length`, so abstentions inflate the numerator. `computeRCCurve` (`metrics.ts:1316-1324`) maps that function across thresholds, so the RC curve's accuracy axis is inflated whenever HOLDs are flat. The sibling `directionalAccuracy` was already fixed to exclude HOLD; this function is the remaining instance.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-17-selective-accuracy-rewards-hold-abstention.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-17-selective-accuracy-rewards-hold-abstention.ts
```

Expected: mixed selective accuracy = 0.666667; never-trading selective accuracy = 0.000000

Actual: mixed = 0.900000; never-trading = 1.000000 (HOLD scored as correct)

OBSERVED:
    mixed selective accuracy = 0.900000 (selected=10)
    directionalAccuracy(mixed) = 0.666667
    never-trading selective accuracy = 1.000000 coverage=1.000000
    never-trading directionalAccuracy = 0.000000
    RC curve first point accuracy = 1.000000

## Impact

The calibration table in `btc-multi-horizon-comparison-round6` prints `selDir@0.45` / `selCov@0.45` from `summarizeCalibrationSteps`, and the RC curve is the documented selective-prediction diagnostic. Both report inflated accuracy for models that abstain, which can make a HOLD-heavy arm look better than a trading arm and defeats the purpose of the risk-coverage analysis.

## Suggested fix direction (not applied)

In `metrics.ts:1298-1302`, compute the accuracy numerator/denominator over traded selected rows only: `const traded = selected.filter(s => s.recommendation !== 'HOLD'); if (traded.length === 0) return { accuracy: 0, coverage: selected.length / steps.length, selected: selected.length, total: steps.length }; const correct = traded.filter(s => (s.recommendation === 'BUY' ? s.actualReturn > 0 : s.actualReturn < 0));` and return `accuracy: correct.length / traded.length`. Keep `coverage`/`selected` based on the confidence-selected set so the coverage axis is unchanged.

## Evidence

- Reproduction script: `issues/repro/PI-17-selective-accuracy-rewards-hold-abstention.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-4-C1c.txt`.
