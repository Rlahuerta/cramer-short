# PI-12: Crypto model-only output reports 100% anchor mixing

- Severity: M
- Area: C1a
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/markov-distribution.ts:5178` — `commodityModelOnly,`, `src/tools/finance/markov-distribution.ts:5371` — `markovWeight: commodityModelOnly ? 1 : m.mixingTimeWeight,`, `src/tools/finance/markov-distribution.ts:5372` — `anchorWeight: commodityModelOnly ? 0 : 1 - m.mixingTimeWeight,`, `src/tools/finance/markov-distribution.ts:3007` — `return 'Calibration: model-only (commodity bypass, no anchors)';`
- Repro kind: runtime

## Summary

`cryptoModelOnly` is a first-class no-anchor emission bypass (declared at line 5013 and included in `canEmitCanonical`/`anchorBypassApplied` at lines 5079/5087), exactly like `commodityModelOnly`. The user-facing mixing line and the canonical `markovWeight`/`anchorWeight` diagnostics only special-case the commodity flag, so a BTC model-only run with zero trusted anchors reports "Mixing: 0.0% Markov / 100.0% Anchors" and `anchorWeight ≈ 1` while `anchorBypassApplied` is true and `calibrationMode` is `model_only`.

## Root cause

The model-only bypass is recognized when deciding whether to emit and when labeling `calibrationMode` (`commodityModelOnly || cryptoModelOnly`, line 5086), but not in the presentation paths:

- `formatMarkovMixingLine` is called with `commodityModelOnly` only (line 5178), so for crypto it falls through to the generic formula `ρ^horizon` / `1 − ρ^horizon`. With the default 0.4 second eigenvalue and h=7 that is ~0.0000114 → "0.0% Markov / 100.0% Anchors".
- The canonical diagnostics at lines 5371–5372 use the same commodity-only condition, so `markovWeight`/`anchorWeight` contradict `anchorBypassApplied`.
- Even if the flag were passed through, `formatMarkovMixingLine`'s model-only text hardcodes "commodity bypass" (line 3007), which is wrong for crypto.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-12-crypto-model-only-mixing-label.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-12-crypto-model-only-mixing-label.ts
```

Expected: a model-only run reports 100% Markov (markovWeight=1, anchorWeight=0) and a model-only calibration line

Actual: markovWeight=0.000011354772919909694, anchorWeight=0.9999886452270801, line="Mixing: 0.0% Markov / 100.0% Anchors"

OBSERVED:
    {
      "status": "ok",
      "calibrationMode": "model_only",
      "anchorBypassApplied": true,
      "trustedAnchors": 0,
      "markovWeight": 0.000011354772919909694,
      "anchorWeight": 0.9999886452270801,
      "mixingLine": "Mixing: 0.0% Markov / 100.0% Anchors"
    }

## Impact

Every crypto short-horizon forecast emitted through the model-only bypass (zero trusted anchors) tells the user and the canonical payload that its distribution is ~100% Polymarket-anchor driven when there are no anchors at all; the actual distribution is pure Markov. This misstates the model provenance on a path that exists precisely because anchors are often unavailable, and contradicts the adjacent `anchorBypassApplied: true` / `calibrationMode: "model_only"` diagnostics in the same payload.

## Suggested fix direction (not applied)

Treat crypto and commodity bypasses identically in the presentation:

- `src/tools/finance/markov-distribution.ts:5178`: `commodityModelOnly,` → `commodityModelOnly: commodityModelOnly || cryptoModelOnly,`
- `src/tools/finance/markov-distribution.ts:5371`: `markovWeight: commodityModelOnly ? 1 : m.mixingTimeWeight,` → `markovWeight: commodityModelOnly || cryptoModelOnly ? 1 : m.mixingTimeWeight,`
- `src/tools/finance/markov-distribution.ts:5372`: `anchorWeight: commodityModelOnly ? 0 : 1 - m.mixingTimeWeight,` → `anchorWeight: commodityModelOnly || cryptoModelOnly ? 0 : 1 - m.mixingTimeWeight,`
- `src/tools/finance/markov-distribution.ts:3007`: `return 'Calibration: model-only (commodity bypass, no anchors)';` → `return 'Calibration: model-only (anchor bypass, no anchors)';`

## Evidence

Reproduction script: `issues/repro/PI-12-crypto-model-only-mixing-label.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-2-C1a.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above; the same payload reports `anchorBypassApplied: true` and `trustedAnchors: 0`.
