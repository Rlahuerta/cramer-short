# PI-5: date-only market end dates settle replay labels at midnight UTC

- Severity: H
- Area: C3
- Status: Not a bug (latent)
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/arbiter-replay-labeler.ts:38` — `const marketEndMs = market.endDate ? Date.parse(market.endDate) : Number.NaN;`, `src/tools/finance/arbiter-replay-labeler.ts:39` — `return Number.isFinite(marketEndMs) ? marketEndMs : forecastTargetTimeMs(bundle);`
- Repro kind: runtime

## Resolution

Not a bug for the data this repo produces. Every price-history loader that ships (`fixture` and `local:<path>`, `replay-price-history-adapter.ts`) builds daily closes dated `YYYY-MM-DD`, which normalise to `T00:00Z`, so a date-only `endDate` at 00:00Z lines up with that day's own close (the repro's expectation assumes intraday history that nothing here produces). Checked with daily bars (06-29=95k, 06-30=90k, 07-01=120k; "above 100k on June 30"): current code answers `no`, which is correct; a 16:00Z target would pick the July 1 bar and answer `yes`. The 16:00 UTC claim is documented only in an untracked review doc and in this issue.

Latent only for a custom `local:` fixture carrying intraday timestamps. If that is ever supported: for a date-only `endDate`, settle on the last point within that UTC day instead of the first point at or after 00:00Z. That works for daily and intraday data alike. Not applied.

## Summary

Polymarket's `endDateIso` is date-only for daily markets (`"2026-06-30"`). `Date.parse()` treats that as `2026-06-30T00:00:00Z`, so `settlementPoint()` picks the first intraday observation on the settlement day instead of the daily close (these markets resolve around 16:00 UTC). In the repro the 01:00 print is 120,000 (above the 100,000 strike, YES) while the 20:00 close is 90,000 (below, NO); the labeler records YES.

## Root cause

`marketTargetTimeMs()` at `src/tools/finance/arbiter-replay-labeler.ts:34-40` parses the market's `endDate` with `Date.parse()` without distinguishing a date-only value from a full ISO timestamp. For `"2026-06-30"` the target becomes midnight UTC at the start of the day. `settlementPoint()` then returns the first point at or after that instant (01:00 in the repro) instead of the day's close around 16:00 UTC. Daily markets settle on the day's close, so the wrong bar is used for the semantic label.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-5-date-only-end-date-settles-at-midnight.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-5-date-only-end-date-settles-at-midnight.ts
```

Expected: semantic outcome=no (settlement uses the June 30 end-of-day close)

Actual: semantic outcome=yes (date-only endDate parsed as 00:00 UTC)

OBSERVED:
    semantic outcome=yes; settlement-day prints were 01:00=120000 (above) and 20:00=90000 (below); the end-of-day close says NO but the midnight target picked the 01:00 print

  'yes' !== 'no'

## Impact

Every daily Polymarket market captured in a replay bundle is labeled from an intraday print near midnight rather than the settlement-day close. The resulting semantic labels, forecast labels, and the Brier calibration fitted from them can be inverted whenever the price crosses the strike during the settlement day. The replay-label benchmark's accuracy numbers are therefore unreliable for date-only markets.

## Suggested fix direction (not applied)

`src/tools/finance/arbiter-replay-labeler.ts:38`:

```diff
- const marketEndMs = market.endDate ? Date.parse(market.endDate) : Number.NaN;
+ const marketEndMs = market.endDate
+   ? Date.parse(/^\d{4}-\d{2}-\d{2}$/.test(market.endDate) ? `${market.endDate}T16:00:00Z` : market.endDate)
+   : Number.NaN;
```

16:00 UTC is the documented daily resolution time; any settlement hour strictly after the first intraday print would remove the inversion shown in the repro.

## Evidence

Reproduction script: `issues/repro/PI-5-date-only-end-date-settles-at-midnight.ts`. Produced by evidence file `.omo/evidence/bug-hunt-reproducible-issues/task-8-C3.txt`. This is a re-verification of the date-only settlement minor from `docs/reviews/deep-code-review-2026-10-23.md` §6; the cited code is unchanged at HEAD. Green-flip verified in a scratch copy with the end-of-day normalization applied (repro exits 0 and prints `NO BUG`).
