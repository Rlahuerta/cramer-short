# PI-50: Coverage gate silently passes when COVERAGE_LINES_MIN is not numeric

- Severity: M
- Area: C9
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `scripts/check-coverage.ts:4` — `const minimumLineCoverage = Number(process.env.COVERAGE_LINES_MIN ?? '22');`, `scripts/check-coverage.ts:26` — `if (lineCoverage < minimumLineCoverage) {`
- Repro kind: runtime

## Summary

The CI coverage gate reads `COVERAGE_LINES_MIN` with `Number(...)` and never validates the result. A non-numeric value becomes `NaN` (and an empty value becomes `0`), so the comparison `lineCoverage < minimumLineCoverage` is always false and the gate reports success for any coverage — including 0%.

## Root cause

`minimumLineCoverage` is derived with `Number(process.env.COVERAGE_LINES_MIN ?? '22')`. `Number('not-a-number')` is `NaN`, and every `< NaN` comparison is false, so the only failing condition in the script can never fire. `Number('')` is `0`, so an unset-in-spirit CI variable (`COVERAGE_LINES_MIN: ${{ vars.MISSING }}` expands to an empty string) also disables the gate. There is no `Number.isFinite` check and no rejection of an empty string.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-50-coverage-gate-nan-threshold-passes.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-50-coverage-gate-nan-threshold-passes.ts
```

Expected: The gate throws for an invalid threshold before evaluating coverage.

Actual: The threshold becomes NaN, the comparison is always false, so 0% coverage passes the gate.

OBSERVED:
    COVERAGE_LINES_MIN="not-a-number"
    coverage/lcov.info: LF:100 LH:0 (0.00% line coverage)
    import resolved without error (rejected=false)

## Impact

A typo or an unresolved CI variable silently turns the coverage gate into a no-op: `bun run test:coverage` exits 0 and CI stays green while line coverage can regress to zero. The failure is silent (the script even prints "Line coverage: 0.00%"), so the control is only discovered when coverage has already drifted. Severity M (silent disable of a CI control with a trivial misconfiguration).

## Suggested fix direction (not applied)

`scripts/check-coverage.ts:4` — validate the raw value before use:

current:

```ts
const minimumLineCoverage = Number(process.env.COVERAGE_LINES_MIN ?? '22');
```

proposed:

```ts
const rawMinimum = process.env.COVERAGE_LINES_MIN ?? '22';
const minimumLineCoverage = Number(rawMinimum);
if (rawMinimum.trim() === '' || !Number.isFinite(minimumLineCoverage)) {
  throw new Error(`COVERAGE_LINES_MIN must be a number, got ${JSON.stringify(rawMinimum)}`);
}
```

## Evidence

- Repro: `issues/repro/PI-50-coverage-gate-nan-threshold-passes.ts` (exit 42 at HEAD, `OBSERVED:` block above).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-17-C9.txt`.
