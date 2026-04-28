import { describe, it, expect, spyOn, beforeEach, afterEach } from 'bun:test';
import { validateConfigValue, validateAndSanitizeConfig } from './config.js';

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
      },
    };
    const result = validateAndSanitizeConfig(raw);
    expect((result as Record<string, unknown>).forecasting).toBeDefined();
    const f = (result as Record<string, unknown>).forecasting as Record<string, unknown>;
    expect(f.enableJumpDiffusion).toBe(true);
    expect(f.qToPMprCap).toBe(2.0);
    expect(f.enableMSM).toBe(false);
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

  it('preserves other top-level keys alongside forecasting', () => {
    const raw = { maxIterations: 30, forecasting: { enableJumpDiffusion: true } };
    const result = validateAndSanitizeConfig(raw);
    expect(result.maxIterations).toBe(30);
  });
});
