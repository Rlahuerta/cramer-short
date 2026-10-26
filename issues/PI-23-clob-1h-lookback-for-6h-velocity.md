# PI-23: CLOB price history uses a one-hour lookback for six-hour velocity

- Severity: M
- Area: C3
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/polymarket.ts:663` — `fetchClobPriceHistory(tokenId, '1h'),`, `src/tools/finance/polymarket-clob.ts:200` — `prices-history?market=`
- Repro kind: runtime

## Summary

`enrichStructuredMarketMicrostructure()` requests the CLOB `/prices-history` endpoint with `interval=1h`, then computes `computePriceVelocityPpH(history)` (6-hour regression window) and `computeMaxHourlyJump(history)` (24-hour window). Polymarket's `interval` parameter is the lookback span, not the bar size, so both statistics see only the last hour of data. In the repro a six-hour +6pp/hour ramp followed by a flat hour yields `priceVelocityPpH=0` because only the flat hour was fetched.

## Root cause

`fetchClobPriceHistory()` forwards its `interval` argument verbatim as the API's lookback span (`src/tools/finance/polymarket-clob.ts:200`). Its only production caller passes `'1h'` (`src/tools/finance/polymarket.ts:663`) even though the downstream primitives are documented as 6-hour and 24-hour windows (`computePriceVelocityPpH` defaults to `lookbackHours = 6`, `computeMaxHourlyJump` defaults to `windowHours = 24`). The lookback is therefore shorter than the window it is supposed to support.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-23-clob-1h-lookback-for-6h-velocity.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-23-clob-1h-lookback-for-6h-velocity.ts
```

Expected: |priceVelocityPpH| > 1 for a six-hour +6pp/hour ramp

Actual: priceVelocityPpH=0 because only the last 1h of history was requested

OBSERVED:
    priceVelocityPpH=0; requested interval(s)=["1h"]; a 6-hour regression cannot see the six-hour ramp with a one-hour lookback

  false !== true

## Impact

Whale-print / velocity / hourly-jump features are computed from one hour of price history regardless of the requested window. In `src/utils/finance/ensemble.ts` these fields drive market-quality penalties, so a genuine multi-hour ramp can be scored as flat and a spike just outside the last hour is invisible. The effect is a mis-weighted Polymarket signal and a misleading microstructure diagnostic, not a crash.

## Suggested fix direction (not applied)

`src/tools/finance/polymarket.ts:663` — `fetchClobPriceHistory(tokenId, '1h')` → `fetchClobPriceHistory(tokenId, '1d')` so the returned history covers both the 6-hour velocity regression and the 24-hour jump window.

## Evidence

Reproduction script: `issues/repro/PI-23-clob-1h-lookback-for-6h-velocity.ts`. Produced by evidence file `.omo/evidence/bug-hunt-reproducible-issues/task-8-C3.txt`. This re-verifies the CLOB interval minor from `docs/reviews/deep-code-review-2026-10-23.md` §6; the caller still passes `'1h'` at HEAD. Green-flip verified in a scratch copy with the caller changed to `'1d'` (repro exits 0 and prints `NO BUG`).
