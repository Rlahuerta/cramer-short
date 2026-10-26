# PI-19: sentiment variance ignores the sentiment signal magnitude

- Severity: M
- Area: C1d
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/utils/finance/ensemble.ts:617` — `sentSignal: number,`, `src/utils/finance/ensemble.ts:638` — `const varianceSent = (sentWeight * 0.04) ** 2;`
- Repro kind: runtime

## Summary

`computeVariance()` accepts a `sentSignal` argument that is documented to "contribute to the variance term", but the parameter is never read. Weak, zero and full-strength sentiment therefore produce an identical forecast sigma, so the 95% confidence interval is over-wide for weak sentiment and the sentiment magnitude never reaches the uncertainty estimate.

## Root cause

The sentiment variance is computed as `const varianceSent = (sentWeight * 0.04) ** 2;`. The adjacent comment says "assume ±4% std for full bullish/bearish unit", which implies scaling the ±4% by the signal magnitude, but `sentSignal` is not referenced anywhere in the function body. `runEnsemble()` passes `others.sentimentScore ?? 0` at `src/utils/finance/ensemble.ts:865`, so the caller does supply the magnitude; the callee drops it.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-19-sentiment-variance-ignores-signal.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-19-sentiment-variance-ignores-signal.ts
```

Expected: sigma(sentSignal=0.01) < sigma(sentSignal=1.0)

Actual: sigma is identical for sentSignal=0, 0.01 and 1.0

OBSERVED:
    sigma(sentSignal=0)=0.025659805143, sigma(sentSignal=0.01)=0.025659805143, sigma(sentSignal=1.0)=0.025659805143; weak and strong sentiment produce identical sigma

  false !== true

## Impact

The 95% CI returned by `runEnsemble()` (and therefore by the `polymarket_forecast` tool) is insensitive to sentiment strength. A near-neutral sentiment (e.g. 0.01) still injects the full ±(weight × 4%) uncertainty term, over-widening the CI relative to a properly scaled estimate; conversely the uncertainty term cannot shrink to zero when there is no sentiment direction. This is a misleading uncertainty/confidence number rather than a crash.

## Suggested fix direction (not applied)

`src/utils/finance/ensemble.ts:638` — `const varianceSent = (sentWeight * 0.04) ** 2;` → `const varianceSent = (sentWeight * 0.04 * Math.abs(sentSignal)) ** 2;`. This keeps a full bullish/bearish unit at the documented ±4% standard deviation while scaling monotonically with the signal magnitude (zero signal → zero sentiment variance).

## Evidence

Reproduction script: `issues/repro/PI-19-sentiment-variance-ignores-signal.ts`. Produced by evidence file `.omo/evidence/bug-hunt-reproducible-issues/task-5-C1d.txt`. Green-flip verified with the change above applied to a scratch copy of `src/utils/finance/ensemble.ts` (repro exits 0, prints `NO BUG`).
