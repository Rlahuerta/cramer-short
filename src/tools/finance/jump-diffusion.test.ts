/**
 * Unit tests for jump-diffusion helpers (Idea 2).
 * Mirror: research/tests/test_jump_diffusion.py
 */
import { describe, expect, test } from 'bun:test';
import {
  JUMP_DEFAULTS,
  buildJumpEventSpec,
  jumpDriftCompensator,
  polymarketProbToHazard,
  type JumpEventSpec,
} from './jump-diffusion.js';

describe('polymarketProbToHazard', () => {
  test('p=0 ⇒ λ=0', () => {
    expect(polymarketProbToHazard(0, 30)).toBe(0);
  });
  test('p=1 ⇒ saturates at 0.95', () => {
    expect(polymarketProbToHazard(1, 30)).toBe(0.95);
  });
  test('p=0.5, days=10 ⇒ −ln(0.5)/10 ≈ 0.0693', () => {
    const lambda = polymarketProbToHazard(0.5, 10);
    expect(lambda).toBeCloseTo(-Math.log(0.5) / 10, 10);
  });
  test('p=0.1, days=30 ⇒ −ln(0.9)/30 ≈ 0.00351', () => {
    const lambda = polymarketProbToHazard(0.1, 30);
    expect(lambda).toBeCloseTo(-Math.log(0.9) / 30, 10);
  });
  test('hazard never exceeds 0.95 even with horizon=1', () => {
    expect(polymarketProbToHazard(0.999, 1)).toBeLessThanOrEqual(0.95);
  });
  test('horizon < 1 is treated as 1 day', () => {
    expect(polymarketProbToHazard(0.5, 0)).toBeCloseTo(-Math.log(0.5), 10);
  });
});

describe('JUMP_DEFAULTS', () => {
  test('all asset classes present with negative meanLogJump', () => {
    for (const cls of ['etf', 'equity', 'crypto', 'commodity', 'geopolitics'] as const) {
      expect(JUMP_DEFAULTS[cls].meanLogJump).toBeLessThan(0);
      expect(JUMP_DEFAULTS[cls].stdLogJump).toBeGreaterThan(0);
    }
  });
  test('crypto has the largest jump magnitude', () => {
    expect(Math.abs(JUMP_DEFAULTS.crypto.meanLogJump))
      .toBeGreaterThan(Math.abs(JUMP_DEFAULTS.etf.meanLogJump));
  });
  test('geopolitics has meanLogJump ≤ −0.10 (tail-risk spec)', () => {
    expect(JUMP_DEFAULTS.geopolitics.meanLogJump).toBeLessThanOrEqual(-0.10);
  });
  test('geopolitics stdLogJump is wider than equity (reflects uncertainty)', () => {
    expect(JUMP_DEFAULTS.geopolitics.stdLogJump)
      .toBeGreaterThanOrEqual(JUMP_DEFAULTS.equity.stdLogJump);
  });
});

describe('jumpDriftCompensator', () => {
  test('empty array ⇒ 0', () => {
    expect(jumpDriftCompensator([])).toBe(0);
  });
  test('single event matches κ formula', () => {
    const e: JumpEventSpec = { id: 'x', dailyIntensity: 0.01, meanLogJump: -0.05, stdLogJump: 0.03 };
    const expected = 0.01 * (Math.exp(-0.05 + 0.03 * 0.03 / 2) - 1);
    expect(jumpDriftCompensator([e])).toBeCloseTo(expected, 14);
  });
  test('compensator is additive across events', () => {
    const e1: JumpEventSpec = { id: 'a', dailyIntensity: 0.005, meanLogJump: -0.05, stdLogJump: 0.03 };
    const e2: JumpEventSpec = { id: 'b', dailyIntensity: 0.01, meanLogJump: -0.08, stdLogJump: 0.05 };
    expect(jumpDriftCompensator([e1, e2]))
      .toBeCloseTo(jumpDriftCompensator([e1]) + jumpDriftCompensator([e2]), 14);
  });
});

describe('buildJumpEventSpec', () => {
  test('raw=0.30, horizon=30, equity prior produces sensible spec', () => {
    const spec = buildJumpEventSpec(0.30, 30, 0.10, 0.05, 0.20, JUMP_DEFAULTS.equity, 'mkt-1');
    expect(spec.id).toBe('mkt-1');
    expect(spec.meanLogJump).toBe(JUMP_DEFAULTS.equity.meanLogJump);
    expect(spec.stdLogJump).toBe(JUMP_DEFAULTS.equity.stdLogJump);
    expect(spec.dailyIntensity).toBeGreaterThan(0);
    expect(spec.dailyIntensity).toBeLessThan(0.95);
  });
  test('raw=0 ⇒ dailyIntensity=0 (event impossible)', () => {
    const spec = buildJumpEventSpec(0, 30, 0.10, 0.05, 0.20, JUMP_DEFAULTS.equity, 'mkt-2');
    expect(spec.dailyIntensity).toBe(0);
  });
});
