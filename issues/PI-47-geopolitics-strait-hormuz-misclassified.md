# PI-47: Strait of Hormuz news is classified as China-Taiwan risk

- Severity: M
- Area: C8a
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/osint/event-asset-map.ts:81` — `'taiwan', 'pla', 'strait'`, `src/tools/osint/event-asset-map.ts:64` — `'strait of hormuz'`
- Repro kind: runtime

## Summary

`detectCategories` does substring matching over each category's keyword list. The
China-Taiwan entry includes the bare token `'strait'` (intended as "Taiwan
Strait"), so any Middle East headline about the Strait of Hormuz also matches
China-Taiwan. `geopolitics_search` then returns Taiwan supply-chain assets —
TSMC, NVIDIA, AMD, ASML, Intel as `risk-down` — for an oil-supply question. The
table's own Middle East entry already contains the specific phrase
`'strait of hormuz'`, so the generic token is redundant and actively wrong.
The China-Taiwan category is not even in the returned "detected categories" the
user sees only the ticker list, which makes the wrong implication look deliberate.

## Root cause

`event-asset-map.ts:81` lists `'strait'` as a detection keyword for
`china-taiwan`, while `event-asset-map.ts:64` lists `'strait of hormuz'` for
`middle-east`. `detectCategories` (event-asset-map.ts:208-216) returns every
category whose keyword is a substring of the lowercased text, with no
disambiguation, so "Strait of Hormuz closure" matches both. `getAssetImplications`
then unions the two categories' assets, producing chip-supply tickers for an oil
story.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-47-geopolitics-strait-hormuz-misclassified.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-47-geopolitics-strait-hormuz-misclassified.ts
```

Expected: a Strait of Hormuz story yields only Middle East / Iran / energy categories, with no chip-maker tickers

Actual: categories=["middle-east","china-taiwan","iran"], chipTickers=["TSM","NVDA","AMD","ASML","INTC"]

OBSERVED:
    {
      "categories": [
        "middle-east",
        "china-taiwan",
        "iran"
      ],
      "impliedTickers": [
        "TSM",
        "USO",
        "NVDA",
        "GLD",
        "AMD",
        "XOM",
        "MAERSK",
        "AAPL",
        "COP",
        "ASML",
        "LMT",
        "FDX",
        "INTC"
      ],
      "chipTickersFromOilStory": [
        "TSM",
        "NVDA",
        "AMD",
        "ASML",
        "INTC"
      ],
      "taiwanStoryCategories": [
        "china-taiwan"
      ],
      "energySupplyTickers": [
        "LNG (Cheniere Energy)",
        "XOM (ExxonMobil)",
        "CVX (Chevron)",
        "AIRL (Airline ETF (JETS))",
        "SLB (SLB (Schlumberger))"
      ]
    }

## Impact

The tool's headline output is a structured "asset implications" table. For a
Hormuz/oil question it tells the agent that TSMC/NVIDIA/AMD are risk-down plays
because of Taiwan — a wrong causal claim derived from a substring, presented with
the table's per-asset confidence numbers (0.75-0.95). Downstream answers and
memory entries built from that table inherit the error.

## Suggested fix direction (not applied)

Make the Taiwan keyword specific in `src/tools/osint/event-asset-map.ts:81` —
current `'taiwan', 'pla', 'strait'` → proposed `'taiwan', 'pla', 'taiwan strait'`
(and keep `'strait of hormuz'` in the Middle East list). Optionally make
`detectCategories` prefer the longest matching keyword per text span, but the
one-word data fix is the minimal change.

## Evidence

Script: `issues/repro/PI-47-geopolitics-strait-hormuz-misclassified.ts`
Evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-15-C8a.txt` (rc=42).
