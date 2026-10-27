# PI-37: Reverse DCF uses an absolute $0.01 price tolerance and returns wrong implied growth for low-priced shares

- Severity: M
- Area: C6b
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/reverse-dcf.ts:113` — `const TOLERANCE = 0.01;`, `src/tools/finance/reverse-dcf.ts:134` — `if (Math.abs(fLow) <= TOLERANCE) {`, `src/tools/finance/reverse-dcf.ts:147` — `if (Math.abs(fHigh) <= TOLERANCE) {`, `src/tools/finance/reverse-dcf.ts:195` — `if (Math.abs(fMid) <= TOLERANCE) {`
- Repro kind: runtime

## Summary

`solveReverseDcf()` accepts a solution when `|fairValue(g) - currentPrice| <= 0.01`, an absolute tolerance in dollars per share. For a low per-share value, one cent is a large fraction of the price, so the "exact hits at boundaries" branches at lines 134 and 147 fire and return the LOW or HIGH bracket endpoint as `converged` even when the true root is far from it. The bisection at line 195 uses the same tolerance and can stop at a materially wrong growth rate.

Synthetic demonstration: base FCF 100, 1,000,000 diluted shares, true growth 0.30 produce a fair value of $0.003604. Feeding that price back to `reverse_dcf` returns `impliedGrowthRate: -0.20` (the LOW boundary) with `resultStatus: "converged"` — a 50-percentage-point error that the status reports as a successful solve.

## Root cause

The tolerance compares a per-share dollar error against a constant dollar amount instead of scaling it to the price being matched. At `current_price = 0.003604`, `TOLERANCE = 0.01` is ~2.8× the price itself, so `|fairValue(LOW) - currentPrice| <= TOLERANCE` is true for an entire range of growth rates and the boundary shortcut wins.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-37-reverse-dcf-absolute-price-tolerance.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-37-reverse-dcf-absolute-price-tolerance.ts
```

Expected: impliedGrowthRate within 0.05 of 0.30 (the synthetic growth used to build the price)

Actual: impliedGrowthRate=-0.2 at the LOW boundary with resultStatus "converged"

OBSERVED:
    trueGrowth=0.30 price=0.003604 impliedGrowth=-0.2 status=converged; solver did not recover the synthetic growth rate

## Impact

Reverse DCF is Step 6 of the DCF skill and its implied growth is reported directly to the user. For low-priced shares (penny/OTC names, or any cap table large enough that fair value per share is a fraction of a dollar) the tool reports a boundary value such as −20% or +200% as a converged solution, and the `converged` status suppresses any hint that the solve failed. Normal-priced shares are unaffected because 1 cent is negligible relative to their price.

## Suggested fix direction (not applied)

Scale the acceptance tolerance to the price being matched:

- `src/tools/finance/reverse-dcf.ts:113` — `const TOLERANCE = 0.01;` → `const TOLERANCE = Math.max(1e-9, Math.abs(currentPrice) * 1e-6);`

The three comparisons (lines 134, 147, 195) then need no change: the bracket-endpoint shortcuts only trigger when the boundary value is within 1e-6 of the price, and the bisection converges to roughly machine precision. Verified by green-flip: the repro exits 0 after this one-line change.

## Evidence

- Reproduction script: `issues/repro/PI-37-reverse-dcf-absolute-price-tolerance.ts` (exit 42 at the pinned HEAD).
- Producing evidence file: `.omo/evidence/bug-hunt-reproducible-issues/task-12-C6b.txt`.
