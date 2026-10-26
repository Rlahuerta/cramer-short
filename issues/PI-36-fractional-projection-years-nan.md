# PI-36: Fractional projection years silently produce NaN valuations and a false converged reverse DCF

- Severity: M
- Area: C6b
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/dcf-valuation.ts:82` — `years: z`, `src/utils/finance/dcf.ts:150` — `const finalFcf = projectedFcf[years - 1];`, `src/tools/finance/reverse-dcf.ts:30` — `years: z`, `src/tools/finance/reverse-dcf.ts:209` — `const finalFcf = impliedFcfProjection[years - 1];`
- Repro kind: runtime

## Summary

Both valuation tools declare `years` as `z.number().min(1).max(20)` with no integer constraint. The DCF skill parameter is also `type: number`, so a value such as `5.5` can reach the tools. The projection loop in `projectFcf()` runs `i < years` (six values for 5.5), but `computeFairValuePerShare()` and `solveReverseDcf()` read `projectedFcf[years - 1]` / `impliedFcfProjection[years - 1]` — index 4.5, i.e. `undefined` — and every downstream value becomes `NaN`.

Observed at the pinned HEAD:

- `dcf_valuation` returns `fairValuePerShare: null`, `enterpriseValue: null`, a sensitivity grid whose every data cell is `"NaN"`, and `validation.errors` that never mention `years`.
- `reverse_dcf` returns `resultStatus: "converged"`, `impliedGrowthRate: 2` (the HIGH bracket endpoint) and `impliedTv: null`.

## Root cause

Non-integer horizons are accepted at the tool boundary while the valuation engine assumes `years` is an integer array length. The raw, unvalidated `years` value is used both as a loop bound and as a `years - 1` array index.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-36-fractional-projection-years-nan.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-36-fractional-projection-years-nan.ts
```

Expected: years=5.5 is rejected at the schema boundary, or a finite fair value is returned

Actual: dcf.fairValuePerShare=null with a NaN sensitivity grid, and reverse_dcf reports converged at impliedGrowth=2 with impliedTv=null

OBSERVED:
    years=5.5 dcf.fairValuePerShare=null dcf.validation.errors=["Reported enterprise value must be positive"] reverse.status=converged reverse.impliedGrowth=2 reverse.impliedTv=null; a fractional horizon silently yields a non-finite fair value

## Impact

A plausible input (an LLM emitting `years: 5.5`, or a user asking for a "5.5-year horizon") silently produces a null headline in `dcf_valuation` with no explanation about `years`, and `reverse_dcf` reports a converged 200% implied growth with no terminal value. The failure is silent because the JSON serializer converts `NaN` to `null`; nothing in the validation block identifies the horizon as the cause.

## Suggested fix direction (not applied)

Constrain the horizon at the trust boundary:

- `src/tools/finance/dcf-valuation.ts:83` — `.number()` → `.number()` followed by `.int()` (line 83).
- `src/tools/finance/reverse-dcf.ts:31` — `.number()` → `.number()` followed by `.int()` (line 31).

Optionally also assert integer horizons inside `computeFairValuePerShare()` / `solveReverseDcf()` so the pure functions cannot be driven into NaN by direct callers. Verified by green-flip: the repro exits 0 after the two `.int()` additions because both invocations are rejected by the schema.

## Evidence

- Reproduction script: `issues/repro/PI-36-fractional-projection-years-nan.ts` (exit 42 at the pinned HEAD).
- Producing evidence file: `.omo/evidence/bug-hunt-reproducible-issues/task-12-C6b.txt`.
