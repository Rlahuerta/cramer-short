# PI-18: quality score signal-diversity component exceeds its 15-point cap

- Severity: M
- Area: C1d
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/utils/finance/ensemble.ts:674` — `const s4 = 15 * (signalsWithData / 4);`, `src/utils/finance/ensemble.ts:881` — `(markets.length > 0 ? 1 : 0) +`
- Repro kind: runtime

## Summary

`computeQualityScore()` weights signal diversity as a 15-point component but divides the count by 4, while `runEnsemble()` can pass five signal sources (Polymarket + sentiment + fundamental + options + Markov). With all five present the diversity component contributes `15 × 5/4 = 18.75` points, pushing the composite quality score above the ceiling implied by its own component weights and inflating the displayed A/B/C/D grade.

## Root cause

The normaliser in `const s4 = 15 * (signalsWithData / 4);` assumes four signal sources, but the caller builds `signalsWithData` from five terms — the first is the Polymarket signal (`markets.length > 0 ? 1 : 0`), followed by the four auxiliary signals. When all five are available, `s4` reaches 18.75 instead of its documented 15-point allotment (the Python mirror in `research/models/ensemble/scoring.py` states "s4 (15 pts): auxiliary signal coverage (out of 4 possible)").

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-18-quality-score-signal-diversity-overflow.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-18-quality-score-signal-diversity-overflow.ts
```

Expected: signal-diversity contribution <= 15 points (score 77, grade B)

Actual: signal-diversity contribution 19 points (score 81, grade A)

OBSERVED:
    signalsWithData=0 -> score 62; signalsWithData=5 -> score 81 (grade A); signal-diversity contribution 19 points; a 15-point component must not contribute 19 points

  false !== true

## Impact

The composite quality score and letter grade shown for a Polymarket forecast can be one grade too high whenever all signal sources are present. In the reproducer the same forecast profile scores 77 (grade B) with the diversity component capped and 81 (grade A) at HEAD — a user-visible conviction-grade inflation. The score is not always clamped to 100, so the error propagates at the 60/80 grade thresholds.

## Suggested fix direction (not applied)

`src/utils/finance/ensemble.ts:674` — `const s4 = 15 * (signalsWithData / 4);` → `const s4 = 15 * (signalsWithData / 5);` (equivalently, cap the component with `Math.min(15, 15 * (signalsWithData / 4))`). Either keeps a fully-covered forecast at the documented 15-point maximum.

## Evidence

Reproduction script: `issues/repro/PI-18-quality-score-signal-diversity-overflow.ts`. Produced by evidence file `.omo/evidence/bug-hunt-reproducible-issues/task-5-C1d.txt`. Green-flip verified with the change above applied to a scratch copy of `src/utils/finance/ensemble.ts` (repro exits 0, prints `NO BUG`).
