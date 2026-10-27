# PI-30: Legacy model setting is not migrated before modelId reads

- Severity: M
- Area: C5
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/utils/config.ts:198` — `if (key === 'provider') {`, `src/utils/config.ts:174` — `function migrateModelToProvider(config: Config): Config {`
- Repro kind: runtime

## Summary

The legacy `model` → `provider`/`modelId` migration is deferred until `getSetting('provider')` is called. Readers that ask for `modelId` first — the gateway (`gateway.ts:225`) and the heartbeat runner (`heartbeat/runner.ts:160`) — see no `modelId` and fall back to the default model, silently ignoring the model the user had saved under the pre-migration key.

## Root cause

`src/utils/config.ts:194-201`:

```ts
export function getSetting<T>(key: string, defaultValue: T): T {
  let config = loadConfig();
  
  // Run migration if accessing provider setting
  if (key === 'provider') {
    config = migrateModelToProvider(config);
  }
```

`loadConfig` itself never calls `migrateModelToProvider`. The migration therefore depends on call order. Interactive mode reads `provider` first (`model-selection.ts:49`) and works; gateway and heartbeat read `modelId` first and get `null`.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-30-legacy-model-migration-not-run-on-load.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-30-legacy-model-migration-not-run-on-load.ts
```

Expected: the first modelId read returns the legacy model 'claude-opus-4-6'

Actual: the first modelId read returned null; the migration only ran on the later provider read

OBSERVED:
    {
      "firstModelIdRead": null,
      "providerRead": "anthropic",
      "modelIdAfterProviderRead": "claude-opus-4-6"
    }

## Impact

After an upgrade from a settings file that still uses the legacy `model` key, gateway and heartbeat runs use the default model instead of the user's saved model. The behaviour is order-dependent and silent: the same settings file resolves correctly in the TUI but not in the gateway.

## Suggested fix direction (not applied)

Run the migration during load, before any getter can observe the config. In `src/utils/config.ts`, after line 132 (`const config = validateAndSanitizeConfig(parsed);`) and before the deprecated-model upgrade block, insert:

```ts
    migrateModelToProvider(config);
```

`migrateModelToProvider` is a hoisted function declaration and calls `saveConfig` once when it actually migrates, so the on-disk file is still rewritten exactly as before. The `key === 'provider'` branch in `getSetting` becomes redundant but harmless.

## Evidence

Reproduction script: `issues/repro/PI-30-legacy-model-migration-not-run-on-load.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-10-C5.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above.
