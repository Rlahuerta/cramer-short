# PI-4: Cross-platform anchor merge treats a below-market YES probability as P(above)

- Severity: H
- Area: C1a
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/markov-distribution.ts:1935` — `const divergence = Math.abs(poly.rawProbability - kalshi.probability);`, `src/tools/finance/markov-distribution.ts:1947` — `const averaged = (poly.rawProbability + kalshi.probability) / 2;`, `src/tools/finance/markov-distribution.ts:1951` — `probability: averaged * YES_BIAS_MULTIPLIER, // apply bias correction to blended probability`
- Repro kind: runtime

## Summary

`PriceThreshold.probability` is always P(price > strike), but `rawProbability` keeps the market's raw YES price. For a market parsed as "below $X", the two are complementary: `extractPriceThresholds` stores `rawProbability = P(below)` while `probability = 1 − P(below)·0.95`. `mergeAnchorsWithCrossPlatformValidation` compares and averages `rawProbability` against Kalshi's P(above), so a Polymarket and Kalshi quote that agree at 0.24 merge to 0.494 and raise a false 56pp divergence warning.

## Root cause

`inspectAnchorCandidate` inverts below markets only in the canonical field (`markov-distribution.ts:1105-1106`: `survivalProb = isBelow ? (1 - correctedRaw) : correctedRaw`), leaving `rawProbability` as raw YES in both directions. The merge at lines 1935/1947/1951 assumes every Polymarket anchor's raw YES is P(above), averages it with the Kalshi P(above), and then re-applies the yes-bias multiplier to the average. For a below market that mixes P(below) into a P(above) estimate; the result is roughly `average(1−p, p)·0.95`, which collapses toward 0.5 instead of the agreed level. `PriceThreshold` carries no direction flag, so the canonical `probability` field is the only direction-correct quantity available to the merge.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-4-below-anchor-raw-probability-merge.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-4-below-anchor-raw-probability-merge.ts
```

Expected: merged P(above) stays near the agreed 0.24 and no divergence warning is emitted

Actual: merged P(above)=0.494 with 1 spurious warning(s)

OBSERVED:
    {
      "polymarket": {
        "price": 90000,
        "rawYes": 0.8,
        "pAbove": 0.24
      },
      "kalshi": {
        "price": 90000,
        "probability": 0.24,
        "volume": 10000
      },
      "merged": {
        "price": 90000,
        "rawProbability": 0.52,
        "probability": 0.494,
        "trustScore": "high",
        "trustWeight": 1,
        "source": "averaged",
        "endDate": "<TS>"
      },
      "warnings": [
        {
          "price": 90000,
          "polymarketProb": 0.8,
          "kalshiProb": 0.24,
          "divergencePp": 56
        }
      ]
    }

## Impact

Whenever the tool receives `kalshiAnchors` (the schema exposes the field) and a Polymarket anchor came from a "below/under/less than $X" market, the merged survival probability at that strike is pulled toward 0.5 — here by ~0.25 absolute, roughly doubling the true probability. The interpolation grid then uses the wrong P(above) at that strike, shifting the blended distribution, scenario buckets and action signal. The same arithmetic also fabricates a cross-platform divergence warning, which is surfaced to the user as a manipulation/liquidity signal.

## Suggested fix direction (not applied)

Use the direction-correct, already bias-corrected `poly.probability` as the Polymarket side of the merge:

- `src/tools/finance/markov-distribution.ts:1935`: `poly.rawProbability` → `poly.probability`.
- `src/tools/finance/markov-distribution.ts:1947`: `poly.rawProbability` → `poly.probability`.
- `src/tools/finance/markov-distribution.ts:1951`: `probability: averaged * YES_BIAS_MULTIPLIER,` → `probability: averaged,` (the bias is already baked into `poly.probability`, and Kalshi needs none).

## Evidence

Reproduction script: `issues/repro/PI-4-below-anchor-raw-probability-merge.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-2-C1a.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above; the Polymarket anchor's canonical `probability` (0.24) is the value both platforms agree on.
