# PI-11: BTC short-horizon abstain payload never emits its forecastHint

- Severity: M
- Area: C1a
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/markov-distribution.ts:5259` — `const forecastHint = fullCanonical`, `src/tools/finance/markov-distribution.ts:547` — `if (params.canEmitCanonical) return null;`
- Repro kind: runtime

## Summary

The markov-distribution tool can never emit `forecastHint`. The wrapper builds the hint only when `fullCanonical` is true, while `buildForecastHint` returns `null` whenever `canEmitCanonical` is true — and `fullCanonical` implies `canEmitCanonical`. The only consumer of the field reads it in the opposite branch (`status === 'abstain'`), where the wrapper never constructs it.

## Root cause

`markov-distribution.ts:5259` guards the call with `fullCanonical ? buildForecastHint({ canEmitCanonical, ... }) : null`. `fullCanonical` is defined as `canEmitCanonical && !contextOnlyCanonical` (line 5083), so the argument is always `true`, and the helper's own guard at line 547 (`if (params.canEmitCanonical) return null;`) always returns `null`. The intended input is the abstain branch (`canEmitCanonical === false`), which the ternary never reaches. The tool-output test at `markov-distribution.tool-output.test.ts:802` locks the helper's contract (`canEmitCanonical: false` → non-null hint), but no test exercises the wrapper wiring.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-11-forecast-hint-never-emitted.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-11-forecast-hint-never-emitted.ts
```

Expected: a forecastHint object for the BTC-USD h=7 abstain payload (canEmitCanonical=false, confidence >= 0.10)

Actual: forecastHint=null while buildForecastHint(contract) returns {"usage":"forecast_only","markovReturn":0.00023041219846021844,"confidenceScore":1,"calibratedDistribution":false}

OBSERVED:
    {
      "status": "abstain",
      "canEmitCanonical": false,
      "predictionConfidence": 1,
      "expectedReturn": 0.009556381860152771,
      "mixingTimeWeight": 0.04822163907471462,
      "sameRunAsDirectCompute": true,
      "contractHint": {
        "usage": "forecast_only",
        "markovReturn": 0.00023041219846021844,
        "confidenceScore": 1,
        "calibratedDistribution": false
      },
      "emittedForecastHint": null
    }

## Impact

`src/agent/query-router/tool-call-extractors.ts:292` reads `forecastHint` only when `data['status'] === 'abstain'` to populate `forecast_return`/`confidence` for the forecast arbiter. Because the field is always `null`, every BTC short-horizon abstain loses the model's fallback return and confidence, so downstream arbitration only sees the abstain reasons. The feature is silently dead while its helper and contract tests remain green.

## Suggested fix direction (not applied)

At `src/tools/finance/markov-distribution.ts:5259`, change

```ts
const forecastHint = fullCanonical
```

to

```ts
const forecastHint = !canEmitCanonical
```

leaving the rest of the ternary unchanged. In that branch `canEmitCanonical` is `false`, `fullCanonical` is `false`, and `contextOnlyCanonical` is `false` (context-only implies `canEmitCanonical === true`), so the hint is produced exactly for abstains.

## Evidence

Reproduction script: `issues/repro/PI-11-forecast-hint-never-emitted.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-2-C1a.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above; the observed `contractHint` matches the helper contract asserted by `markov-distribution.tool-output.test.ts:802`.
