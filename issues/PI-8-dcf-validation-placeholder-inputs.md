# PI-8: DCF valuation fabricates an EV error and a value-destruction warning on every run

- Severity: H
- Area: C6b
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/dcf-valuation.ts:160` — `roic: 0,`, `src/tools/finance/dcf-valuation.ts:162` — `reportedEv: 0,`, `src/tools/finance/dcf-valuation.ts:156` — `riskFreeRate: 0.043,`, `src/utils/finance/dcf-validation.ts:143` — `if (reportedEv <= 0) {`
- Repro kind: runtime

## Summary

`dcf_valuation` runs the full validation suite with values it never receives: `roic: 0`, `reportedEv: 0`, and a hardcoded `riskFreeRate: 0.043`. Two checks therefore fire on every invocation regardless of the company:

- `validateEvSanity()` rejects the impossible reported EV of 0, so every result carries `errors: ["Reported enterprise value must be positive"]`.
- `validateWaccVsRoic()` compares the real WACC against a zero ROIC, so every result carries `WACC (...) exceeds ROIC (0.00%) -- potential value destruction`.

The DCF skill instructs the agent to surface every error verbatim and that "if any error fires, the headline number is not trustworthy" (`src/skills/dcf/SKILL.md` Step 8). A healthy DCF therefore reports a false hard failure plus a false value-destruction warning on each run. The existing QA transcript at `.omo/drafts/dcf-improvements-qa-evidence.md` lines 123–124 shows a run hand-waving both messages ("false alarm"; "data formatting issue") instead of fixing them.

## Root cause

`DcfValidationInput` has no representation for "data not supplied", so the tool fills the gaps with sentinel numbers. `runDcfValidation()` then treats those sentinels as real observations, and `validateEvSanity()` turns `reportedEv = 0` into a hard error while `validateWaccVsRoic()` turns `roic = 0` into a warning. The tool's own output schema does not accept a reported EV or an ROIC, so the values can never be correct at this call site.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-8-dcf-validation-placeholder-inputs.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-8-dcf-validation-placeholder-inputs.ts
```

Expected: errors=[] and no "value destruction" warning for a healthy DCF

Actual: errors=["Reported enterprise value must be positive"] and "potential value destruction" warning on every call

OBSERVED:
    errors=["Reported enterprise value must be positive"] warnings=["WACC (10.00%) exceeds ROIC (0.00%) -- potential value destruction"]; EV-sanity error fired although no reported EV was supplied

  1 !== 0

## Impact

Every DCF answer contains one hard validation error and one wrong solvency warning. Per the skill contract the agent must either declare the headline fair value untrustworthy or explain the messages away, and a model that surfaces them verbatim reports "potential value destruction" for profitable companies. The EV-sanity check also cannot do its job: it never compares the computed EV to a real reported EV.

## Suggested fix direction (not applied)

Make "not supplied" expressible and stop passing placeholders.

- `src/utils/finance/dcf-validation.ts:13` — `riskFreeRate: number;` → `riskFreeRate?: number;`
- `src/utils/finance/dcf-validation.ts:17` — `roic: number;` → `roic?: number;`
- `src/utils/finance/dcf-validation.ts:19` — `reportedEv: number;` → `reportedEv?: number;`
- `src/utils/finance/dcf-validation.ts:196-197` — wrap the `validateTerminalGrowthVsRfr` call in `if (checks.riskFreeRate !== undefined) { ... }`.
- `src/utils/finance/dcf-validation.ts:212-215` — wrap the `validateWaccVsRoic` block in `if (checks.roic !== undefined) { ... }`.
- `src/utils/finance/dcf-validation.ts:217-220` — wrap the `validateEvSanity` block in `if (checks.reportedEv !== undefined) { ... }`.
- `src/tools/finance/dcf-valuation.ts:156,160,162` — delete the `riskFreeRate: 0.043,`, `roic: 0,`, and `reportedEv: 0,` lines from the `runDcfValidation({ ... })` call.

With that change a default DCF returns `{ errors: [], warnings: [] }`, and the EV-vs-reported check remains available to callers that actually have the reported EV (the skill already performs it in prose per Step 8B).

## Evidence

- Reproduction script: `issues/repro/PI-8-dcf-validation-placeholder-inputs.ts` (exit 42 at the pinned HEAD).
- Producing evidence file: `.omo/evidence/bug-hunt-reproducible-issues/task-12-C6b.txt`.
- Prior QA transcript showing the same two false alarms: `.omo/drafts/dcf-improvements-qa-evidence.md:123-124`.
