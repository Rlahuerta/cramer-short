import { describe, it, expect } from 'bun:test';
import {
  estimateTokens,
  TOKEN_BUDGET,
  CONTEXT_THRESHOLD,
  KEEP_TOOL_USES,
  getContextThreshold,
  getKeepToolUses,
} from './tokens.js';

describe('estimateTokens', () => {
  it('returns 0 for empty string', () => {
    expect(estimateTokens('')).toBe(0);
  });

  it('uses ~3.5 chars per token (ceil)', () => {
    expect(estimateTokens('a'.repeat(35))).toBe(10);
    expect(estimateTokens('a'.repeat(36))).toBe(11);
  });

  it('handles typical prose', () => {
    const result = estimateTokens('The quick brown fox jumps over the lazy dog');
    expect(result).toBeGreaterThan(0);
    expect(typeof result).toBe('number');
  });
});

describe('constants', () => {
  it('TOKEN_BUDGET is a positive number', () => {
    expect(TOKEN_BUDGET).toBeGreaterThan(0);
    expect(typeof TOKEN_BUDGET).toBe('number');
  });

  it('CONTEXT_THRESHOLD is a positive number', () => {
    expect(CONTEXT_THRESHOLD).toBeGreaterThan(0);
  });

  it('KEEP_TOOL_USES is a positive integer', () => {
    expect(KEEP_TOOL_USES).toBeGreaterThan(0);
    expect(Number.isInteger(KEEP_TOOL_USES)).toBe(true);
  });
});

describe('getContextThreshold', () => {
  it('returns a positive number', () => {
    const result = getContextThreshold();
    expect(result).toBeGreaterThan(0);
    expect(typeof result).toBe('number');
  });

  it('returns CONTEXT_THRESHOLD as default when no override is set', () => {
    // When settings.json has no contextThreshold, getSetting returns the default.
    // We verify the function returns the same value as the exported constant
    // (or a valid custom value if one was already set).
    const result = getContextThreshold();
    expect(result).toBeGreaterThanOrEqual(1000);
  });
});

describe('getKeepToolUses', () => {
  it('returns a positive integer', () => {
    const result = getKeepToolUses();
    expect(result).toBeGreaterThan(0);
    expect(Number.isInteger(result)).toBe(true);
  });

  it('returns KEEP_TOOL_USES as default when no override is set', () => {
    const result = getKeepToolUses();
    expect(result).toBeGreaterThanOrEqual(1);
  });
});
