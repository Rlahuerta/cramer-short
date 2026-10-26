# PI-52: Trajectory 90% CI can narrow as the horizon grows

- Severity: L
- Area: C1a
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/markov-distribution/confidence-intervals.ts:643` — `const p95Idx = Math.min(nSamples - 1, Math.ceil(nSamples * 0.95));`, `src/tools/finance/markov-distribution/confidence-intervals.ts:645` — `let upperBound = prices[p95Idx];`
- Repro kind: runtime

## Summary

`computeTrajectory` documents that its shared-path Monte Carlo "ensures CI widths monotonically increase with horizon", and `confidence-intervals.test.ts:475` asserts that property. It is not guaranteed for sample percentiles: with `randomState = 76` the 90% band narrows from 36.07 to 35.38 between day 11 and day 12 (lower bound rises from 84.02 to 83.97 is not enough to offset the upper bound falling from 120.09 to 119.35). The Python mirror clamps each day's band to the previous day's explicitly (`research/models/trajectory/simulation.py`, "Enforce monotonically widening CIs"); the TS implementation has no clamp.

## Root cause

The per-day bounds are raw order statistics of the shared cumulative paths: `prices[p5Idx]` / `prices[p95Idx]` at lines 642–645 are taken independently each day and never compared with the prior day. Because path ranks can cross as daily increments accumulate, the sample 5th/95th percentiles can move inward even though the theoretical quantiles widen. The uncertainty-scale block at lines 655–664 widens relative to the same day's raw bounds only; it does not enforce cross-day monotonicity. The reference implementation in `research/models/trajectory/simulation.py` adds `lower_bound = min(lower_bound, trajectory[-1].lower_bound)` and `upper_bound = max(upper_bound, trajectory[-1].upper_bound)` before appending each point, which is the missing guard.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-52-trajectory-ci-non-monotone.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-52-trajectory-ci-non-monotone.ts
```

Expected: each trajectory day's 90% CI width is at least the previous day's width (within one cent of rounding)

Actual: day 12 width 35.38 < day 11 width 36.07

OBSERVED:
    {
      "seed": 76,
      "days": 14,
      "nSamples": 1000,
      "worstViolation": {
        "day": 12,
        "previousWidth": 36.07000000000001,
        "width": 35.379999999999995
      },
      "violations": [
        {
          "day": 12,
          "previousWidth": 36.07000000000001,
          "width": 35.379999999999995
        }
      ],
      "days11to13": [
        {
          "day": 11,
          "expectedPrice": 100,
          "lowerBound": 84.02,
          "upperBound": 120.09,
          "pUp": 0.5,
          "cumulativeReturn": "+0.0%",
          "regime": "sideways"
        },
        {
          "day": 12,
          "expectedPrice": 100,
          "lowerBound": 83.97,
          "upperBound": 119.35,
          "pUp": 0.5,
          "cumulativeReturn": "+0.0%",
          "regime": "sideways"
        },
        {
          "day": 13,
          "expectedPrice": 100,
          "lowerBound": 84.14,
          "upperBound": 120.43,
          "pUp": 0.5,
          "cumulativeReturn": "+0.0%",
          "regime": "sideways"
        }
      ]
    }

## Impact

A user reading the trajectory table sees the "honest measure of uncertainty" shrink while the forecast horizon lengthens, contradicting the function's documented contract and the equivalent Python output. With a fixed seed, walk-forward/backtest comparisons that use trajectory bands can also depend on this sampling artifact. The magnitude is small (≈2% of the band width in this case), which is why this is rated L rather than M.

## Suggested fix direction (not applied)

Insert the same previous-day clamp the Python mirror uses, immediately after the uncertainty-scale block (after `src/tools/finance/markov-distribution/confidence-intervals.ts:664`, before the `// P(up)` comment):

```ts
if (trajectory.length > 0) {
  const previous = trajectory[trajectory.length - 1];
  lowerBound = Math.min(lowerBound, previous.lowerBound);
  upperBound = Math.max(upperBound, previous.upperBound);
}
```

The repro's one-cent tolerance covers the `Math.round(... * 100) / 100` applied when the point is pushed.

## Evidence

Reproduction script: `issues/repro/PI-52-trajectory-ci-non-monotone.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-2-C1a.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above; the counterexample is deterministic under `randomState = 76`.
