# PI-55: Routing cache records FMP for Financial Datasets successes

- Severity: L
- Area: C6a
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/get-financials.ts:306` — `const apiHint = result.tool.includes('yahoo') ? 'yahoo' : 'fmp';`
- Repro kind: runtime

## Summary

After a successful routed sub-tool call, `get_financials` persists a provider preference for the ticker. The provider is derived with `result.tool.includes('yahoo') ? 'yahoo' : 'fmp'`, so every non-Yahoo tool — and all of them are Financial Datasets tools — is recorded as `fmp`, even though FMP was never contacted. The bogus preference lives for 30 days and is injected into the next session's router prompt as "FMP free tier works for this ticker."

## Root cause

`FINANCE_TOOLS` contains Financial Datasets tools (`get_income_statements`, `get_key_ratios`, `get_earnings`, …) and Yahoo tools, but no FMP tools — FMP is only invoked as an internal fallback inside `fundamentals.ts`. The two-way ternary treats "not Yahoo" as "FMP", so the actual provider (`financial-datasets`) is never recorded. The cache's own type (`ApiPreference`) includes `financial-datasets`, confirming that value is expected to be used.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-55-routing-cache-fmp-mislabel.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-55-routing-cache-fmp-mislabel.ts
```

Expected: cached preferredApi(AAPL) = financial-datasets

Actual: cached preferredApi(AAPL) = fmp

OBSERVED:
    served by financialdatasets = true, served by FMP = false
    result.data keys = ["get_income_statements_AAPL"]
    cached preferredApi(AAPL) = fmp

## Impact

The routing cache is corrupted for every ticker whose data comes from the primary provider. For 30 days the next session receives a false routing hint steering the router away from the working provider, and the cache cannot distinguish a genuine FMP discovery from a Financial Datasets success. The immediate user-visible effect is a misleading prompt hint rather than a wrong number, hence low severity.

## Suggested fix direction (not applied)

Map the serving tool to its actual provider: `src/tools/finance/get-financials.ts:306` — `const apiHint = result.tool.includes('yahoo') ? 'yahoo' : 'fmp';` → `const apiHint = result.tool.includes('yahoo') ? 'yahoo' : 'financial-datasets';`. (A per-tool provider map is equivalent and future-proof.) Verified green-flip: with this change the repro exits 0.

## Evidence

- Reproduction script: `issues/repro/PI-55-routing-cache-fmp-mislabel.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-11-C6a.txt`.
- Green-flip: applied the one-line provider mapping to a scratch copy of `src/`; the same script exits 0.
