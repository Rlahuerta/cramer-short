# PI-33: FMP fallback drops the report-period filters the tool advertises

- Severity: M
- Area: C6a
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/fundamentals.ts:104` — `fmpTool: { invoke: (input: { ticker: string; period: 'annual' | 'quarterly'; limit: number }) => Promise<string> },`, `src/tools/finance/fundamentals.ts:111` — `const raw = await fmpTool.invoke({ ticker, period: fmpPeriod(period), limit });`, `src/tools/finance/fundamentals.ts:184` — `const fmpResult = await tryFmp(getFmpIncomeStatements, ticker, input.period, input.limit);`
- Repro kind: runtime

## Summary

`get_income_statements`, `get_balance_sheets`, and `get_cash_flow_statements` accept `report_period_gt/gte/lt/lte` and forward them to the primary endpoint via `createParams()`. When the primary request fails, `tryFmp()` is invoked with only `{ticker, period, limit}` — its input type has no date fields — so the requested fiscal window is silently discarded and the latest FMP periods are returned as the answer.

## Root cause

`tryFmp`'s contract (`fundamentals.ts:103-108`) cannot carry the period bounds: it forwards a fixed object literal to the FMP tool (`:111`), and the call sites (`:184`, `:227`, `:261`, and the three `Promise.allSettled` calls at `:298-302`) pass only ticker/period/limit. The bounds exist only in `createParams()` for the primary provider, so the fallback path has no knowledge of them.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-33-fmp-fallback-drops-period-filters.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-33-fmp-fallback-drops-period-filters.ts
```

Expected: every returned statement date inside 2019-01-01..2019-12-31

Actual: fallback returned an unfiltered FMP period: ["2024-09-28","2019-09-28"]

OBSERVED:
    requested report_period_gte = 2019-01-01, report_period_lte = 2019-12-31
    FMP request = https://financialmodelingprep.com/stable/income-statement?apikey=dummy-fmp-key&symbol=AAPL&period=annual&limit=4
    returned statement dates = ["2024-09-28","2019-09-28"]

## Impact

A historical query that depends on the fallback (the documented path for international tickers and for users without a Financial Datasets plan) is answered with the wrong fiscal years. Revenue, margin, or cash-flow figures from FY2024 are presented as the FY2019 answer, which is a wrong financial number; the `date` field is present in each row, but the caller has no indication that its filter was ignored.

## Suggested fix direction (not applied)

Thread the bounds into `tryFmp` and apply them to the fallback result. Minimal change: extend the `tryFmp` signature (`src/tools/finance/fundamentals.ts:103-108`) with `bounds?: { gt?: string; gte?: string; lt?: string; lte?: string }`; after `const error = ...` / `if (error) ...` (`:114-116`), filter `parsed.data` on the `date` field against the bounds, returning `{ kind: 'error' }` when no row survives; then pass `{ gt: input.report_period_gt, gte: input.report_period_gte, lt: input.report_period_lt, lte: input.report_period_lte }` at the four fallback call sites (`:184`, `:227`, `:261`, `:298-302`). Passing `from`/`to` on the FMP request instead is equivalent if the endpoint honours them. Verified green-flip: with the client-side filter the repro exits 0.

## Evidence

- Reproduction script: `issues/repro/PI-33-fmp-fallback-drops-period-filters.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-11-C6a.txt`.
- Green-flip: applied the client-side bounds filter to a scratch copy of `src/`; the same script exits 0.
