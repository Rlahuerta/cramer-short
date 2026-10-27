# PI-22: named GOLD mutator comparison drops the gold- id prefix

- Severity: M
- Area: C2
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/forecast-lab-run.ts:533` — `query?.match(/\b(markov-[A-Za-z0-9][A-Za-z0-9_.-]*)\b/i)?.[1]`
- Repro kind: runtime

## Summary

`forecast_lab_run` action `compare-best-vs-shipped` accepts the mutator to compare either as the explicit `mutationId` field or, when that is absent, by extracting a `markov-...` token from `query`. Every GOLD catalog entry is named with a `gold-` prefix (`buildGoldCatalog` passes `idPrefix: 'gold'` in `src/experiments/forecast-lab/mutators/markov-parameters.ts:819`), so the query extractor captures `markov-shorter-reactive-window` while the shipped/ledger id is `gold-markov-shorter-reactive-window`. The kept-run lookup then finds nothing and the action returns `status: 'error'` instead of the comparison.

## Root cause

`extractComparisonMutationId` (`src/tools/forecast-lab-run.ts:532-534`) matches `/\b(markov-[A-Za-z0-9][A-Za-z0-9_.-]*)\b/i`. The `\b` before `markov` succeeds after the hyphen in `gold-markov-...`, so the capture starts at `markov` and silently discards the `gold-` prefix. `findLatestKeptStructuredMutationEntry` compares the extracted string against `entry.mutationId`, which is stored with the full prefixed id, so the lookup fails (`forecast-lab-run.ts:1546-1554`).

The sibling extractor in `src/experiments/forecast-lab/query-router.ts:179` is already gold-aware (`((?:gold-)?markov-...)`), which is the intended shape for this id family.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-22-gold-mutator-prefix.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-22-gold-mutator-prefix.ts
```

Expected: status=ok; mutationId=gold-markov-shorter-reactive-window; activeMutationId=gold-markov-longer-stability-window

Actual: status=error because the query extractor strips the "gold-" prefix and looks up markov-shorter-reactive-window

OBSERVED:
    status=error; mutationId=undefined; activeMutationId=undefined; sourceRunId=undefined; error=No kept structured run matched mutationId "markov-shorter-reactive-window" for profile "gold-markov-short-horizon".

## Impact

A gold mutator comparison requested through the natural `query` route fails outright, so the operator cannot compare a named GOLD candidate against the active live run without knowing to pass `mutationId` explicitly. The failure is reported as a missing run rather than a parse problem, which misdirects the user. The GOLD profile is the only catalog with a prefixed id family, so this is consistently broken for GOLD and works for the other profiles.

## Suggested fix direction (not applied)

At `src/tools/forecast-lab-run.ts:533`, allow the optional `gold-` prefix exactly as the sibling router extractor does: replace `/\b(markov-[A-Za-z0-9][A-Za-z0-9_.-]*)\b/i` with `/(?<![A-Za-z0-9_.-])((?:gold-)?markov-[A-Za-z0-9][A-Za-z0-9_.-]*)\b/i` (also update `extractComparisonMutationId` callers/tests). Add a `forecast-lab-run` test that compares a named `gold-markov-...` mutator through `query` only.

## Evidence

Produced by `issues/repro/PI-22-gold-mutator-prefix.ts` (exit 42 with `OBSERVED:`), captured in `.omo/evidence/bug-hunt-reproducible-issues/task-7-C2.txt`. The script imports `src/tools/forecast-lab-run.ts` after chdir + env scrub and injects every ledger/active-state/file dependency, so no real `.cramer-short` state, network, or child process is used.
