# PI-21: substring improvement terms divert ordinary finance queries into forecast-lab

- Severity: M
- Area: C2
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/experiments/forecast-lab/router.ts:43` — `if (rawQuery.includes(rawTerm)) {`, `src/experiments/forecast-lab/router.ts:48` — `return normalizedTerm.length > 0 && normalizedQuery.includes(normalizedTerm);`
- Repro kind: runtime

## Summary

`routeForecastLabQuery` treats an improvement cue as matched whenever the query merely *contains* it. Because `IMPROVEMENT_TERMS` includes short, common fragments such as `fix`, `tune` and `repair`, ordinary finance questions match inside unrelated words: "fixed-income" and "fixture" both contain "fix", and "fortune" contains "tune". The query is then classified as `intent: 'improvement'`, and `getForecastLabRoutingHint` returns a non-null hint with `shouldInvokeSkill: true`. The agent injects that hint (`injectForecastLabRoutingHint`) and tells the model to treat the request as a bounded forecast-workflow improvement task and to call `skill("forecast-lab")` before ordinary forecast/data tools.

## Root cause

`includesTerm` (`src/experiments/forecast-lab/router.ts:37-49`) uses raw substring containment in both of its branches:

- `rawQuery.includes(rawTerm)` (line 43), and
- `normalizedQuery.includes(normalizedTerm)` (line 48).

No word-boundary or token check is applied, so a cue only has to appear as a character sequence anywhere inside the query. `IMPROVEMENT_TERMS` (`router.ts:16-29`) contains `fix`, `tune`, `repair`, `refine`, etc., which are substrings of ordinary finance vocabulary (`fixed-income`, `fixture`, `prefix`, `fortune`, `tuned`). `routeForecastLabQuery` only needs *one* matched cue to return `intent: 'improvement'` (`router.ts:99-105`); a profile is not required for `getForecastLabRoutingHint` to return a hint (`query-router.ts:54-73`).

This is the same defect documented as the unfixed §5 minor "Substring routing" in `docs/reviews/deep-code-review-2026-10-23.md`; the current code still shows it.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-21-substring-routing-divert.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-21-substring-routing-divert.ts
```

Expected: intent=none and routingHintInjected=false for every fixture (word-boundary improvement terms only)

Actual: intent=improvement and routingHintInjected=true because "fix"/"tune" match inside fixed-income/fixture/fortune

OBSERVED:
    query="What is the fixed-income outlook for the next year?" matched fix (from fixed-income); intent=improvement; preferredProfileId=null; routingHintInjected=true
    query="Compare the fixture data quality across providers." matched fix (from fixture); intent=improvement; preferredProfileId=null; routingHintInjected=true
    query="What is the fortune 500 earnings trend?" matched tune (from fortune); intent=improvement; preferredProfileId=null; routingHintInjected=true

## Impact

Any ordinary research request that uses a word containing an improvement fragment is silently diverted into the bounded forecast-lab path: the model is told to call the forecast-lab skill first and to treat the work as a workflow-improvement experiment instead of answering the finance question with the normal tools. Fixed-income questions are a routine asset-class query in this product, so the false positive is user-visible. Workaround: disable `forecasting.enableForecastLabAutoRoute` / `enableForecastLabSkillHint`, or rephrase.

## Suggested fix direction (not applied)

Require a word-boundary/token match in both branches of `includesTerm`. For example, at `src/experiments/forecast-lab/router.ts:43` replace `if (rawQuery.includes(rawTerm)) {` with a token-boundary test such as `if (new RegExp(\`\\b${rawTerm.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')}\\b\`).test(rawQuery)) {`, and apply the same boundary rule to line 48 (or normalise to space-separated tokens and compare whole tokens). Add `router.test.ts` cases asserting `intent: 'none'` for "fixed-income", "fixture" and "fortune" queries.

## Evidence

Produced by `issues/repro/PI-21-substring-routing-divert.ts` (exit 42 with `OBSERVED:`), captured in `.omo/evidence/bug-hunt-reproducible-issues/task-7-C2.txt`. The script imports `src/experiments/forecast-lab/router.ts` and `src/experiments/forecast-lab/query-router.ts` after chdir + env scrub, with no network or child process.
