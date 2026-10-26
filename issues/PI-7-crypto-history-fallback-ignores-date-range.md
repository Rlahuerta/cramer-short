# PI-7: Crypto history fallback returns the last N days instead of the requested range

- Severity: H
- Area: C6a
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/crypto.ts:85` — `const closes = await fetchBinanceDailyCloses(input.ticker, days);`, `src/tools/finance/binance.ts:77` — `const endTime = Date.now();`, `src/tools/finance/binance.ts:78` — `const startTime = endTime - days * MS_PER_DAY;`
- Repro kind: runtime

## Summary

`get_crypto_prices` promises historical prices for `start_date`..`end_date`. When the Financial Datasets request fails (missing key, 402, network), the fallback computes only the *length* of the requested window and calls `fetchBinanceDailyCloses(ticker, days)`, which anchors its klines window to `Date.now()`. A request for 2021-01-01 → 2021-01-10 therefore returns the most recent ten days. The fallback payload is `{close, index}` with no dates, so the caller cannot see that the window was substituted.

## Root cause

`fetchBinanceDailyCloses` has no start/end parameters: it derives `endTime = Date.now()` and `startTime = endTime - days * MS_PER_DAY` (`binance.ts:77-78`). `getCryptoPrices` receives the real bounds, converts them to a day count, and drops the bounds (`crypto.ts:83-85`). The dated history tool thus reuses the same "last N days" primitive that `markov-distribution.ts` legitimately uses for live forecasts.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-7-crypto-history-fallback-ignores-date-range.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-7-crypto-history-fallback-ignores-date-range.ts
```

Expected: klines bounded by the requested range; closes = [42300]

Actual: klines anchored to the run date; closes = [99999]

OBSERVED:
    requested range 2021-01-01 → 2021-01-10
    binance request starts at <TS> (requested <TS>)
    binance request ends at <TS> (requested <TS>)
    returned closes = [99999]

## Impact

For every caller without a working Financial Datasets key — the documented fallback path — `get_crypto_prices` returns prices from the wrong period. Any return, volatility or drawdown computed for the requested historical window is actually computed over the last N days, and the payload carries no dates that would let the agent detect the substitution. That is a wrong financial number presented as the answer.

## Suggested fix direction (not applied)

- `src/tools/finance/binance.ts:72-79` — add an optional range parameter and use it when present:
  `export async function fetchBinanceDailyCloses(ticker: string, days = 120, range?: { startMs: number; endMs: number }): Promise<number[]>` with
  `const endTime = range?.endMs ?? Date.now();` and `const startTime = range?.startMs ?? endTime - days * MS_PER_DAY;`.
- `src/tools/finance/crypto.ts:85` — `const closes = await fetchBinanceDailyCloses(input.ticker, days);` → `const closes = await fetchBinanceDailyCloses(input.ticker, days, { startMs: startDate.getTime(), endMs: endDate.getTime() + MS_PER_DAY - 1 });`.
- Optionally include the per-candle dates in the fallback rows so the payload is self-describing.

Verified green-flip: with exactly these changes the repro exits 0 and prints `NO BUG`.

## Evidence

- Reproduction script: `issues/repro/PI-7-crypto-history-fallback-ignores-date-range.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-11-C6a.txt`.
- Green-flip: applied the change above to a scratch copy of `src/`; the same script exits 0.
