# PI-6: replay bundles with ambiguous market semantics are dropped on read

- Severity: H
- Area: C3
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/arbiter-replay.ts:235` — `function isReplaySemantics(value: unknown): value is ForecastMarketSemantics {`, `src/tools/finance/arbiter-replay.ts:840` — `semantics: classifyPolymarketQuestion(market.question),`
- Repro kind: runtime

## Summary

`freezePolymarketReplayBlock()` stores whatever `classifyPolymarketQuestion()` returns, and that classifier can return `'ambiguous'`. The reader's `isReplaySemantics()` guard does not include `'ambiguous'`, so any persisted bundle containing an ambiguous market fails validation as a whole and is skipped by `readArbiterReplayBundles()` / `parseArbiterReplayBundleLine()`. The writer and the reader disagree about the same exported union type.

## Root cause

`ForecastMarketSemantics` is `'terminal' | 'barrier_touch' | 'range' | 'path_dependent' | 'ambiguous' | 'unknown'` (`src/tools/finance/forecast-arbitrator.ts:10-16`), and `freezePolymarketReplayBlock()` writes `semantics: classifyPolymarketQuestion(market.question)` directly. `classifyPolymarketQuestion()` returns `'ambiguous'` for conditional or mixed-semantics wording. However, `isReplaySemantics()` at `src/tools/finance/arbiter-replay.ts:235-241` accepts only `terminal`, `barrier_touch`, `range`, `path_dependent`, and `unknown`. `isReplayPolymarketMarket()` therefore rejects the entire bundle line, and `readJsonlRecords()` treats it as malformed and drops it.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-6-replay-ambiguous-semantics-dropped.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-6-replay-ambiguous-semantics-dropped.ts
```

Expected: the written bundle round-trips through parseArbiterReplayBundleLine()

Actual: the whole bundle is rejected because isReplaySemantics() omits ambiguous

OBSERVED:
    written semantics=ambiguous; parseArbiterReplayBundleLine()=null (bundle dropped); an 'ambiguous' market is written by the same module but rejected on read

## Impact

A captured replay bundle whose selected market uses conditional or mixed barrier/terminal wording is written to disk but can never be read back. The replay-label pipeline never labels it, the labeled cache never receives it, and horizon-aware Brier calibration sees fewer bundles than were captured. The loss is silent apart from a generic "Skipping malformed line" warning.

## Suggested fix direction (not applied)

`src/tools/finance/arbiter-replay.ts:240`:

```diff
-     || value === 'unknown';
+     || value === 'ambiguous'
+     || value === 'unknown';
```

so `isReplaySemantics()` covers every member of `ForecastMarketSemantics`.

## Evidence

Reproduction script: `issues/repro/PI-6-replay-ambiguous-semantics-dropped.ts`. Produced by evidence file `.omo/evidence/bug-hunt-reproducible-issues/task-8-C3.txt`. Green-flip verified in a scratch copy with the `value === 'ambiguous'` guard added (repro exits 0 and prints `NO BUG`).
