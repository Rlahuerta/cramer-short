import { describe, it, expect, spyOn, beforeEach, afterEach } from 'bun:test';
import { mkdirSync, mkdtempSync, readFileSync, readdirSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { getSetting, saveConfig, validateConfigValue, validateAndSanitizeConfig } from './config.js';
import * as atomicWrite from './atomic-write.js';

describe('validateConfigValue — maxIterations', () => {
  it('accepts a value within range', () => {
    const result = validateConfigValue('maxIterations', 30);
    expect(result.valid).toBe(true);
  });

  it('rejects a value below the minimum (2 < 5)', () => {
    const result = validateConfigValue('maxIterations', 2);
    expect(result.valid).toBe(false);
    expect(result.error).toBeDefined();
  });

  it('rejects a value above the maximum (200 > 100)', () => {
    const result = validateConfigValue('maxIterations', 200);
    expect(result.valid).toBe(false);
    expect(result.error).toBeDefined();
  });

  it('rejects a non-numeric value', () => {
    const result = validateConfigValue('maxIterations', 'abc');
    expect(result.valid).toBe(false);
    expect(result.error).toBeDefined();
  });

  it('accepts the boundary minimum (5)', () => {
    expect(validateConfigValue('maxIterations', 5).valid).toBe(true);
  });

  it('accepts the boundary maximum (100)', () => {
    expect(validateConfigValue('maxIterations', 100).valid).toBe(true);
  });
});

describe('validateConfigValue — contextThreshold', () => {
  it('accepts a value within range', () => {
    const result = validateConfigValue('contextThreshold', 50000);
    expect(result.valid).toBe(true);
  });

  it('rejects a value below minimum', () => {
    expect(validateConfigValue('contextThreshold', 5000).valid).toBe(false);
  });

  it('rejects a value above maximum', () => {
    expect(validateConfigValue('contextThreshold', 600000).valid).toBe(false);
  });
});

describe('validateConfigValue — keepToolUses', () => {
  it('accepts a value within range', () => {
    const result = validateConfigValue('keepToolUses', 10);
    expect(result.valid).toBe(true);
  });

  it('rejects a value below minimum', () => {
    expect(validateConfigValue('keepToolUses', 1).valid).toBe(false);
  });

  it('rejects a value above maximum', () => {
    expect(validateConfigValue('keepToolUses', 25).valid).toBe(false);
  });
});

describe('validateConfigValue — cacheTtlMs', () => {
  it('accepts a valid TTL', () => {
    expect(validateConfigValue('cacheTtlMs', 900000).valid).toBe(true);
  });

  it('rejects a TTL below minimum', () => {
    expect(validateConfigValue('cacheTtlMs', 30000).valid).toBe(false);
  });
});

describe('validateConfigValue — parallelToolLimit', () => {
  it('accepts 0 (unlimited)', () => {
    expect(validateConfigValue('parallelToolLimit', 0).valid).toBe(true);
  });

  it('accepts a positive limit', () => {
    expect(validateConfigValue('parallelToolLimit', 5).valid).toBe(true);
  });

  it('rejects a value above maximum', () => {
    expect(validateConfigValue('parallelToolLimit', 11).valid).toBe(false);
  });
});

describe('validateConfigValue — llmCallTimeoutMs', () => {
  it('accepts a value within range', () => {
    expect(validateConfigValue('llmCallTimeoutMs', 300000).valid).toBe(true);
  });

  it('rejects a value below minimum', () => {
    expect(validateConfigValue('llmCallTimeoutMs', 10000).valid).toBe(false);
  });

  it('rejects a value above maximum', () => {
    expect(validateConfigValue('llmCallTimeoutMs', 900000).valid).toBe(false);
  });
});

describe('validateConfigValue — unknown keys', () => {
  it('passes through without validation', () => {
    const result = validateConfigValue('unknownKey', 5);
    expect(result.valid).toBe(true);
  });

  it('passes through string values for unknown keys', () => {
    const result = validateConfigValue('unknownKey', 'some-string');
    expect(result.valid).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Config schema validation (Zod)
// ---------------------------------------------------------------------------

describe('Config schema validation (Zod)', () => {
  let warnSpy: ReturnType<typeof spyOn>;

  beforeEach(() => {
    warnSpy = spyOn(console, 'warn').mockImplementation(() => {});
  });

  afterEach(() => {
    warnSpy.mockRestore();
  });

  it('valid config parses without warnings', () => {
    const config = { provider: 'openai', modelId: 'gpt-5.4', maxIterations: 25, llmCallTimeoutMs: 300000 };
    const result = validateAndSanitizeConfig(config);
    expect(warnSpy).not.toHaveBeenCalled();
    expect(result.provider).toBe('openai');
    expect(result.maxIterations).toBe(25);
    expect(result.llmCallTimeoutMs).toBe(300000);
  });

  it('maxIterations: "abc" → warning logged, field stripped, rest returned', () => {
    const config = { provider: 'openai', maxIterations: 'abc' };
    const result = validateAndSanitizeConfig(config);
    expect(warnSpy).toHaveBeenCalledWith(
      '[dexter] config validation warning:',
      expect.objectContaining({ maxIterations: expect.any(Array) }),
    );
    expect(result.maxIterations).toBeUndefined();
    expect(result.provider).toBe('openai');
  });

  it('maxIterations: 2 (below min 5) → warning logged, field stripped', () => {
    const config = { maxIterations: 2, modelId: 'gpt-5.4' };
    const result = validateAndSanitizeConfig(config);
    expect(warnSpy).toHaveBeenCalled();
    expect(result.maxIterations).toBeUndefined();
    expect(result.modelId).toBe('gpt-5.4');
  });

  it('contextThreshold: 999999999 (above max) → warning logged, field stripped', () => {
    const config = { contextThreshold: 999999999, provider: 'anthropic' };
    const result = validateAndSanitizeConfig(config);
    expect(warnSpy).toHaveBeenCalled();
    expect(result.contextThreshold).toBeUndefined();
    expect(result.provider).toBe('anthropic');
  });

  it('llmCallTimeoutMs: 900000 (above max) → warning logged, field stripped', () => {
    const config = { llmCallTimeoutMs: 900000, provider: 'openai' };
    const result = validateAndSanitizeConfig(config);
    expect(warnSpy).toHaveBeenCalled();
    expect(result.llmCallTimeoutMs).toBeUndefined();
    expect(result.provider).toBe('openai');
  });

  it('unknown key myCustomKey: "value" → passes through without warning', () => {
    const config = { myCustomKey: 'value', provider: 'openai' };
    const result = validateAndSanitizeConfig(config);
    expect(warnSpy).not.toHaveBeenCalled();
    expect(result.myCustomKey).toBe('value');
  });

  it('completely invalid JSON structure (array []) → returns {}', () => {
    const result = validateAndSanitizeConfig([]);
    expect(warnSpy).toHaveBeenCalled();
    expect(result).toEqual({});
  });

  it('memory.embeddingProvider: "invalid" → warning logged, field stripped', () => {
    const config = { memory: { embeddingProvider: 'invalid', embeddingModel: 'text-embedding-3-small' } };
    const result = validateAndSanitizeConfig(config);
    expect(warnSpy).toHaveBeenCalled();
    expect((result.memory as Record<string, unknown> | undefined)?.embeddingProvider).toBeUndefined();
    expect((result.memory as Record<string, unknown> | undefined)?.embeddingModel).toBe('text-embedding-3-small');
  });

  it('preserves unknown nested memory keys when the rest of the block is valid', () => {
    const config = {
      memory: {
        enabled: true,
        embeddingModel: 'text-embedding-3-small',
        legacyMemoryFlag: 'keep-me',
      },
    };
    const result = validateAndSanitizeConfig(config);
    const memory = result.memory as Record<string, unknown> | undefined;
    expect(warnSpy).not.toHaveBeenCalled();
    expect(memory?.enabled).toBe(true);
    expect(memory?.embeddingModel).toBe('text-embedding-3-small');
    expect(memory?.legacyMemoryFlag).toBe('keep-me');
  });
});

// ---------------------------------------------------------------------------
// §5.4 — forecasting block in ConfigSchema
// ---------------------------------------------------------------------------
describe('ConfigSchema — forecasting block', () => {
  it('accepts valid forecasting block with all fields', () => {
    const raw = {
      forecasting: {
        enableJumpDiffusion: true,
        qToPMprCap: 2.0,
        enableMSM: false,
        enableForecastLabAutoRoute: true,
        enableForecastLabSkillHint: true,
        enableForecastLabMutatorRanking: false,
      },
    };
    const result = validateAndSanitizeConfig(raw);
    expect((result as Record<string, unknown>).forecasting).toBeDefined();
    const f = (result as Record<string, unknown>).forecasting as Record<string, unknown>;
    expect(f.enableJumpDiffusion).toBe(true);
    expect(f.qToPMprCap).toBe(2.0);
    expect(f.enableMSM).toBe(false);
    expect(f.enableForecastLabAutoRoute).toBe(true);
    expect(f.enableForecastLabSkillHint).toBe(true);
    expect(f.enableForecastLabMutatorRanking).toBe(false);
  });

  it('accepts forecasting with only enableJumpDiffusion', () => {
    const raw = { forecasting: { enableJumpDiffusion: false } };
    const result = validateAndSanitizeConfig(raw);
    const f = (result as Record<string, unknown>).forecasting as Record<string, unknown>;
    expect(f.enableJumpDiffusion).toBe(false);
  });

  it('strips invalid qToPMprCap (negative)', () => {
    const raw = { forecasting: { enableJumpDiffusion: false, qToPMprCap: -1 } };
    const result = validateAndSanitizeConfig(raw);
    const f = (result as Record<string, unknown>).forecasting as Record<string, unknown>;
    expect(f.qToPMprCap).toBeUndefined();
  });

  it('strips non-boolean enableJumpDiffusion', () => {
    const raw = { forecasting: { enableJumpDiffusion: 'yes' } };
    const result = validateAndSanitizeConfig(raw);
    const f = (result as Record<string, unknown>).forecasting as Record<string, unknown>;
    expect(f.enableJumpDiffusion).toBeUndefined();
  });

  it('strips invalid forecast-lab rollout flags while preserving valid forecasting fields', () => {
    const raw = {
      forecasting: {
        enableJumpDiffusion: false,
        enableForecastLabAutoRoute: true,
        enableForecastLabSkillHint: 'yes',
        enableForecastLabMutatorRanking: 1,
      },
    };
    const result = validateAndSanitizeConfig(raw);
    const f = (result as Record<string, unknown>).forecasting as Record<string, unknown>;
    expect(f.enableJumpDiffusion).toBe(false);
    expect(f.enableForecastLabAutoRoute).toBe(true);
    expect(f.enableForecastLabSkillHint).toBeUndefined();
    expect(f.enableForecastLabMutatorRanking).toBeUndefined();
  });

  it('preserves other top-level keys alongside forecasting', () => {
    const raw = { maxIterations: 30, forecasting: { enableJumpDiffusion: true } };
    const result = validateAndSanitizeConfig(raw);
    expect(result.maxIterations).toBe(30);
  });

  it('preserves unknown nested forecasting keys when the rest of the block is valid', () => {
    const raw = {
      forecasting: {
        enableJumpDiffusion: true,
        enableForecastLabAutoRoute: false,
        legacyForecastingFlag: 'keep-me',
      },
    };
    const result = validateAndSanitizeConfig(raw);
    const f = (result as Record<string, unknown>).forecasting as Record<string, unknown>;
    expect(f.enableJumpDiffusion).toBe(true);
    expect(f.enableForecastLabAutoRoute).toBe(false);
    expect(f.legacyForecastingFlag).toBe('keep-me');
  });
});

// ---------------------------------------------------------------------------
// Legacy model migration (getSetting) — lossless carry into provider/modelId
// ---------------------------------------------------------------------------

const ORIGINAL_CWD = process.cwd();
let testDir: string;

function writeSettings(settings: Record<string, unknown>): void {
  mkdirSync(join(testDir, '.cramer-short'), { recursive: true });
  writeFileSync(join(testDir, '.cramer-short', 'settings.json'), JSON.stringify(settings, null, 2));
}

function readSettings(): Record<string, unknown> {
  return JSON.parse(readFileSync(join(testDir, '.cramer-short', 'settings.json'), 'utf-8'));
}

function setupSettingsFixture(): void {
  testDir = mkdtempSync(join(tmpdir(), 'cramer-config-'));
  process.chdir(testDir);
  saveConfig({}); // reset the config cache and start from an empty settings file
}

function teardownSettingsFixture(): void {
  process.chdir(ORIGINAL_CWD);
  rmSync(testDir, { recursive: true, force: true });
}

describe('legacy model migration (getSetting)', () => {
  beforeEach(setupSettingsFixture);
  afterEach(teardownSettingsFixture);

  it('carries an unmapped legacy model into provider and modelId (prefix routing)', () => {
    writeSettings({ model: 'claude-opus-4-6' });

    expect(getSetting('provider', 'openai')).toBe('anthropic');

    const saved = readSettings();
    expect(saved.provider).toBe('anthropic');
    expect(saved.modelId).toBe('claude-opus-4-6');
    expect(saved.model).toBeUndefined();
  });

  it('keeps mapped legacy models working and preserves them in modelId', () => {
    writeSettings({ model: 'gpt-5.2' });

    expect(getSetting('provider', 'openai')).toBe('openai');

    const saved = readSettings();
    expect(saved.modelId).toBe('gpt-5.2');
    expect(saved.model).toBeUndefined();
  });

  it('never overrides a provider already set in modern keys', () => {
    writeSettings({ provider: 'openai', modelId: 'gpt-5.4', model: 'claude-opus-4-6' });

    expect(getSetting('provider', 'openai')).toBe('openai');
    expect(readSettings().modelId).toBe('gpt-5.4');
  });

  it('never overrides a modelId already set in modern keys while resolving the legacy provider', () => {
    writeSettings({ modelId: 'gpt-5.4', model: 'claude-opus-4-6' });

    expect(getSetting('provider', 'openai')).toBe('anthropic');

    const saved = readSettings();
    expect(saved.modelId).toBe('gpt-5.4');
    expect(saved.model).toBeUndefined();
  });
});

// ---------------------------------------------------------------------------
// Atomic settings writes
// ---------------------------------------------------------------------------

describe('atomic settings writes', () => {
  beforeEach(setupSettingsFixture);
  afterEach(teardownSettingsFixture);

  it('saveConfig writes through atomicWriteFileSync', () => {
    const spy = spyOn(atomicWrite, 'atomicWriteFileSync');

    expect(saveConfig({ provider: 'openai' })).toBe(true);

    expect(spy).toHaveBeenCalledTimes(1);
    const [filePath, contents] = spy.mock.calls[0] as [string, string];
    expect(filePath).toBe(join('.cramer-short', 'settings.json'));
    expect(JSON.parse(contents)).toEqual({ provider: 'openai' });
    spy.mockRestore();
  });

  it('atomicWriteFileSync writes the contents and leaves no temp file behind', () => {
    const target = join(testDir, 'nested', 'out.json');

    atomicWrite.atomicWriteFileSync(target, '{"a":1}');

    expect(readFileSync(target, 'utf-8')).toBe('{"a":1}');
    expect(readdirSync(join(testDir, 'nested')).filter((name) => name.endsWith('.tmp'))).toEqual([]);
  });
});
