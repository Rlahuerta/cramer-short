# PI-20: approval source-run-id parser matches an unanchored run substring

- Severity: M
- Area: C2
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/experiments/forecast-lab/query-router.ts:137` — `text.match(new RegExp`
- Repro kind: runtime

## Summary

`detectForecastLabPromotionApproval` extracts the run id named in an approval message with `extractForecastLabSourceRunId`. That function builds `/(?:source\s+run|run)\s+(id)/` without a leading word boundary and returns only the *first* regex match. Any earlier token ending in `run` therefore wins: "Rerun the check for run `<id>`" matches the `run` inside "Rerun", captures `the` (which is rejected for lacking a hyphen/dot), and returns `undefined`; "the overrun 2026-01-01 was fine, promote run `<id>`" captures the unrelated `2026-01-01`. The resulting approval hint either loses the explicitly named run or carries a bogus one, and `agent.ts:817` forwards it verbatim to `forecast_lab_run promote-approved`.

## Root cause

`extractForecastLabSourceRunId` (`src/experiments/forecast-lab/query-router.ts:136-146`) uses

```ts
text.match(new RegExp(`(?:source\s+run|run)\s+(${SAFE_FORECAST_LAB_RUN_ID.source})`, 'i'))
```

with no `\b` before the alternation and with `String.match` on a non-global regex, which only returns the first match. The `SAFE_FORECAST_LAB_RUN_ID` body (`query-router.ts:97`) is itself fine; the defect is the unanchored `run` prefix and the first-match-only search. A word boundary (`\b(?:source\s+run|run)\s+`) would stop `rerun`/`overrun` from matching at all, and a global scan (or a boundary-anchored match) would let the real `run <id>` be found even when it appears later in the message.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-20-approval-run-id-parse.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-20-approval-run-id-parse.ts
```

Expected: every approval hint carries sourceRunId=forecast-lab-sol-2026-01-01T00-00-00-000Z

Actual: "rerun" drops the id (undefined) and "overrun 2026-01-01" is captured as the id

OBSERVED:
    plain approval: sourceRunId=forecast-lab-sol-2026-01-01T00-00-00-000Z (profileId=sol-markov-short-horizon)
    approval that mentions rerun: sourceRunId=undefined (profileId=sol-markov-short-horizon)
    approval that mentions an overrun date: sourceRunId=2026-01-01 (profileId=sol-markov-short-horizon)

## Impact

A genuine, explicitly approved promotion can target the wrong run. When the id is dropped, `runForecastLabApprovalFlow` omits `sourceRunId` and `promoteForecastLab` falls back to `findLatestKeptLedgerEntry` — the *latest* kept structured run for the profile, not the one the user named. When the id is replaced by an unrelated date token, the promotion fails outright. Both outcomes are silent at the intent-detection stage; the operator only sees a later error or an unexpected activation. Workaround: avoid words ending in "run" in the approval message, or call `promote-approved` with an explicit `sourceRunId`.

## Suggested fix direction (not applied)

At `src/experiments/forecast-lab/query-router.ts:137`, anchor the alternative on a word boundary and stop at the first *real* run-id token: use `/\b(?:source\s+run|run)\s+([A-Za-z0-9][A-Za-z0-9_.-]*)/gi` and `matchAll`, keeping the first candidate that passes the existing `includes('-') || includes('.')` guard. Add `query-router` tests for the "rerun"/"overrun" phrasings (the module currently has no colocated test file).

## Evidence

Produced by `issues/repro/PI-20-approval-run-id-parse.ts` (exit 42 with `OBSERVED:`), captured in `.omo/evidence/bug-hunt-reproducible-issues/task-7-C2.txt`. The script imports `src/experiments/forecast-lab/query-router.ts` after chdir + env scrub; it performs no I/O, network call, or child-process spawn.
