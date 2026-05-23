import { describe, expect, test } from 'bun:test';
import {
  classifyJumpDirection,
  type JumpDirection,
} from './polymarket.js';
import {
  buildJumpEventSpec,
  effectiveJumpMean,
  JUMP_DEFAULTS,
} from './jump-diffusion.js';

describe('classifyJumpDirection', () => {
  // Down-jump triggers
  test.each<[string, JumpDirection]>([
    ['Will SPY crash by 10% before Friday?', 'down'],
    ['Will Iran attack US assets?', 'down'],
    ['Recession declared by Q3?', 'down'],
    ['Will BTC drop below $50k?', 'down'],
    ['Major bank fails before year-end?', 'down'],
  ])('"%s" → down', (q, expected) => {
    expect(classifyJumpDirection(q)).toBe(expected);
  });

  // Up-jump triggers
  test.each<[string, JumpDirection]>([
    ['Will the Fed cut rates 50bp?', 'up'],
    ['Will BTC reach $200k by year-end?', 'up'],
    ['Will Nvidia announce new AI breakthrough?', 'up'],
    ['Trade deal signed with China?', 'up'],
  ])('"%s" → up', (q, expected) => {
    expect(classifyJumpDirection(q)).toBe(expected);
  });

  // Ambiguous
  test('ambiguous question returns unknown', () => {
    expect(classifyJumpDirection('Will the meeting happen?')).toBe('unknown');
  });
});

describe('effectiveJumpMean — direction sign flip', () => {
  const prior = JUMP_DEFAULTS.geopolitics; // meanLogJump: -0.10

  test('default (no direction) preserves prior sign', () => {
    expect(effectiveJumpMean(prior.meanLogJump, undefined)).toBeCloseTo(-0.10);
  });

  test('"down" preserves prior sign (defaults are downside)', () => {
    expect(effectiveJumpMean(prior.meanLogJump, 'down')).toBeCloseTo(-0.10);
  });

  test('"up" flips sign of negative prior', () => {
    expect(effectiveJumpMean(prior.meanLogJump, 'up')).toBeCloseTo(+0.10);
  });

  test('"up" with already-positive prior stays positive', () => {
    expect(effectiveJumpMean(0.05, 'up')).toBeCloseTo(0.05);
  });

  test('"down" with positive prior flips negative', () => {
    expect(effectiveJumpMean(0.05, 'down')).toBeCloseTo(-0.05);
  });

  test('"unknown" preserves prior', () => {
    expect(effectiveJumpMean(-0.07, 'unknown')).toBeCloseTo(-0.07);
  });
});

describe('buildJumpEventSpec — propagates jumpDirection', () => {
  test('produces effective negative mean for "down"', () => {
    const spec = buildJumpEventSpec(
      0.20, // raw Q-prob
      30,
      0.10,
      0.05,
      0.30,
      JUMP_DEFAULTS.geopolitics,
      'mkt-1',
      'down',
    );
    expect(spec.meanLogJump).toBeLessThan(0);
    expect(spec.jumpDirection).toBe('down');
  });

  test('produces effective positive mean for "up"', () => {
    const spec = buildJumpEventSpec(
      0.20,
      30,
      0.10,
      0.05,
      0.30,
      JUMP_DEFAULTS.geopolitics,
      'mkt-2',
      'up',
    );
    expect(spec.meanLogJump).toBeGreaterThan(0);
    expect(spec.jumpDirection).toBe('up');
  });

  test('omitted direction defaults to "unknown" and preserves prior sign', () => {
    const spec = buildJumpEventSpec(
      0.20,
      30,
      0.10,
      0.05,
      0.30,
      JUMP_DEFAULTS.geopolitics,
      'mkt-3',
    );
    expect(spec.meanLogJump).toBeLessThan(0); // geopolitics default is negative
    expect(spec.jumpDirection).toBe('unknown');
  });
});
