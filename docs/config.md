# Configuration Guide

Cramer-Short stores runtime configuration in `.cramer-short/settings.json`. This document describes the configuration schema, validation behavior, and migration logic.

## Schema

### Core Settings

```jsonc
{
  "provider": "string",           // Provider ID: openai, anthropic, google, ollama
  "modelId": "string",            // Model identifier (e.g., "gpt-5.4", "claude-sonnet-4-5")
  "maxIterations": 5-100,         // Agent iteration limit (default: 10)
  "contextThreshold": 10000-500000, // Context window threshold for compaction (default: 100000)
  "keepToolUses": 2-20,           // Number of recent tool uses to keep during compaction (default: 5)
  "cacheTtlMs": 60000-86400000,   // Cache TTL in milliseconds (default: 3600000 / 1 hour)
  "parallelToolLimit": 0-10,      // Max parallel tool calls (default: 3, 0 = unlimited)
  "llmCallTimeoutMs": 30000-600000 // LLM call timeout in milliseconds (default: 120000 / 2 minutes)
}
```

### Memory Settings

```jsonc
{
  "memory": {
    "enabled": true,                        // Enable memory system (default: true)
    "embeddingProvider": "openai" | "gemini" | "ollama" | "auto", // Embedding provider
    "embeddingModel": "string",             // Custom embedding model
    "maxSessionContextTokens": 8000         // Optional max tokens for session context injection
  }
}
```

### Forecasting Settings

```jsonc
{
  "forecasting": {
    "enableJumpDiffusion": false,           // Enable Merton jump-diffusion (default: false)
    "qToPMprCap": 1.5,                      // Q→P transformation MPR cap (default: 1.5, range: 0.1-10)
    "enableMSM": false,                     // Enable Markov-Switching Multifractal (default: false)
    "enableForecastLabAutoRoute": true,     // Enable forecast-lab query routing (default: true)
    "enableForecastLabSkillHint": true,     // Inject forecast-lab skill hints (default: true)
    "enableForecastLabMutatorRanking": false // Rank mutators from ledger evidence (default: false)
  }
}
```

## Validation Behavior

### Unknown Key Preservation

**Unknown keys are preserved without validation.** This allows forward compatibility and experimentation with new settings without breaking existing configurations.

Example:
```json
{
  "provider": "openai",
  "myCustomSetting": "value"  // Preserved, not validated
}
```

### Invalid Known Field Handling

When a known field fails validation:
1. A warning is logged to stderr
2. The invalid field is **stripped** from the configuration
3. The remaining valid fields are used
4. **The process does not crash**

Example:
```json
{
  "provider": "openai",
  "maxIterations": 200  // Invalid (max is 100)
}
```

Result:
- Warning: `[dexter] config validation warning: { maxIterations: ['Number must be less than or equal to 100'] }`
- Loaded config: `{ "provider": "openai" }`
- The invalid `maxIterations` field is dropped after the warning

### Validation Rules

| Field | Type | Range/Constraints |
|-------|------|-------------------|
| `provider` | string | - |
| `modelId` | string | - |
| `model` | string | (deprecated, migrated to `provider`) |
| `maxIterations` | number | 5-100 |
| `contextThreshold` | number | 10000-500000 |
| `keepToolUses` | number | 2-20 |
| `cacheTtlMs` | number | 60000-86400000 (1 min - 1 day) |
| `parallelToolLimit` | number | 0-10 |
| `llmCallTimeoutMs` | number | 30000-600000 (30s - 10 min) |
| `memory.enabled` | boolean | - |
| `memory.embeddingProvider` | string | `"openai"` \| `"gemini"` \| `"ollama"` \| `"auto"` |
| `memory.embeddingModel` | string | - |
| `memory.maxSessionContextTokens` | number | - |
| `forecasting.enableJumpDiffusion` | boolean | - |
| `forecasting.qToPMprCap` | number | 0.1-10 |
| `forecasting.enableMSM` | boolean | - |
| `forecasting.enableForecastLabAutoRoute` | boolean | - |
| `forecasting.enableForecastLabSkillHint` | boolean | - |
| `forecasting.enableForecastLabMutatorRanking` | boolean | - |

## Migration Logic

### Deprecated Model ID Upgrades

When loading config, deprecated model IDs are automatically upgraded:

| Deprecated | Upgraded To |
|------------|-------------|
| `gpt-5.2` | `gpt-5.4` |

After upgrade, the config is automatically saved with the new value.

Example:
```json
// Before load
{ "modelId": "gpt-5.2" }

// After load (auto-saved)
{ "modelId": "gpt-5.4" }
```

### Legacy `model` → `provider` Migration

The legacy `model` field is migrated to `provider` when the application reads the
provider setting through `getSetting('provider', ...)`:

| Legacy `model` | Migrated `provider` |
|----------------|---------------------|
| `gpt-5.4` | `openai` |
| `gpt-5.2` | `openai` |
| `claude-sonnet-4-5` | `anthropic` |
| `gemini-3` | `google` |

After that migration:
- The `provider` field is set
- The legacy `model` field is **deleted**
- The config is automatically saved

Example:
```json
// Before load
{ "model": "gpt-5.4" }

// After getSetting('provider', ...) (auto-saved)
{ "provider": "openai" }
```

When setting `provider` programmatically, any existing legacy `model` field is automatically removed.

## Configuration Caching

Configuration is cached in-process with a 1-second TTL to avoid repeated file I/O. The cache is invalidated:
- After 1 second
- After calling `saveConfig()`

## JSON Parse Error Handling

If `settings.json` contains invalid JSON:
1. A warning is logged to stderr with the parse error
2. The default empty config `{}` is returned
3. The invalid file is **not overwritten**
4. You must manually fix the JSON syntax to restore settings

Example warning:
```
[cramer-short] settings.json contains invalid JSON (Unexpected token '}' at position 42).
Falling back to defaults. Fix syntax in .cramer-short/settings.json to restore your settings.
```

## File Structure

- **Path**: `.cramer-short/settings.json`
- **Format**: JSON (pretty-printed with 2-space indentation)
- **Permissions**: Created with default file permissions
- **Directory**: Auto-created if missing (recursive)

## API Reference

### `loadConfig(): Config`

Loads configuration from `.cramer-short/settings.json` with:
- Unknown key preservation
- Invalid field stripping with warnings
- Deprecated model ID upgrades
- 1-second in-process cache

### `saveConfig(config: Config): boolean`

Writes configuration to `.cramer-short/settings.json`:
- Creates parent directory if missing
- Pretty-prints JSON with 2-space indentation
- Invalidates cache
- Returns `true` on success, `false` on I/O error

### `getSetting<T>(key: string, defaultValue: T): T`

Reads a single setting with fallback:
- Loads current config
- Applies migrations if `key === 'provider'`
- Returns `config[key] ?? defaultValue`

### `setSetting(key: string, value: unknown): boolean`

Writes a single setting:
- Loads current config
- Sets `config[key] = value`
- Removes legacy `model` field if `key === 'provider'`
- Saves config
- Returns `true` on success, `false` on I/O error

### `validateConfigValue(key: string, value: unknown): { valid: boolean; error?: string }`

Validates a single config value:
- Known keys are validated against their schema
- Unknown keys always pass validation (`{ valid: true }`)
- Returns `{ valid: false, error: string }` on validation failure

## Examples

### Valid Configuration

```json
{
  "provider": "openai",
  "modelId": "gpt-5.4",
  "maxIterations": 15,
  "memory": {
    "enabled": true,
    "embeddingProvider": "auto"
  },
  "forecasting": {
    "enableJumpDiffusion": true,
    "qToPMprCap": 2.0
  }
}
```

### Configuration with Unknown Keys (Valid)

```json
{
  "provider": "anthropic",
  "experimentalFeature": true,
  "customTimeout": 5000
}
```
All fields are preserved. Unknown keys are not validated.

### Configuration with Invalid Field (Recovered)

```json
{
  "provider": "openai",
  "maxIterations": 999
}
```

Loaded as:
```json
{
  "provider": "openai"
}
```
Warning logged: `maxIterations` out of range (5-100).

### Legacy Configuration (Auto-Migrated)

```json
{
  "model": "claude-sonnet-4-5",
  "modelId": "claude-sonnet-4-5"
}
```

Migrated and saved as:
```json
{
  "provider": "anthropic",
  "modelId": "claude-sonnet-4-5"
}
```

## Implementation

Configuration logic is implemented in `src/utils/config.ts`:
- Schema defined with `zod` using `passthrough()` for unknown keys
- Validation uses `safeParse()` to avoid throwing on invalid input
- Invalid fields are surgically removed from the raw object
- Migrations run lazily on `loadConfig()` and `getSetting('provider')`

## See Also

- [Memory System Documentation](./memory.md)
- [Feature Documentation](./features.md)
