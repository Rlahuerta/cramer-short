import { describe, expect, test } from 'bun:test';
import {
  convergenceTime,
  convergenceTimeFactor,
  type ConvergenceResult,
} from './convergence-time.js';

describe('convergenceTime', () => {
  test('returns null when never converges (epsilon never crossed)', () => {
    const prices = [0.5, 0.48, 0.52, 0.49, 0.51];
    const r = convergenceTime(prices, 0.05);
    expect(r.converged).toBe(false);
    expect(r.daysToConverge).toBeNull();
  });

  test('detects YES convergence (crossing > 1 - ε)', () => {
    const prices = [0.5, 0.7, 0.85, 0.96, 0.97];
    const r = convergenceTime(prices, 0.05);
    expect(r.converged).toBe(true);
    expect(r.direction).toBe('yes');
    expect(r.daysToConverge).toBe(3); // index 3 is first crossing of 0.95
  });

  test('detects NO convergence (crossing < ε)', () => {
    const prices = [0.5, 0.3, 0.1, 0.04, 0.02];
    const r = convergenceTime(prices, 0.05);
    expect(r.converged).toBe(true);
    expect(r.direction).toBe('no');
    expect(r.daysToConverge).toBe(3);
  });

  test('handles empty array', () => {
    const r = convergenceTime([], 0.05);
    expect(r.converged).toBe(false);
    expect(r.daysToConverge).toBeNull();
  });

  test('respects custom epsilon', () => {
    const prices = [0.5, 0.85, 0.92];
    expect(convergenceTime(prices, 0.10).converged).toBe(true); // 0.92 > 0.90
    expect(convergenceTime(prices, 0.05).converged).toBe(false); // 0.92 < 0.95
  });
});

describe('convergenceTimeFactor', () => {
  test('factor = 1.0 when not converged (uncertainty)', () => {
    const r: ConvergenceResult = { converged: false, daysToConverge: null, direction: null };
    expect(convergenceTimeFactor(r)).toBeCloseTo(1.0, 6);
  });

  test('fast convergence (≤7d) ⇒ boost ≈ +15%', () => {
    const r: ConvergenceResult = { converged: true, daysToConverge: 5, direction: 'yes' };
    const f = convergenceTimeFactor(r);
    expect(f).toBeGreaterThan(1.10);
    expect(f).toBeLessThan(1.20);
  });

  test('slow convergence (≥30d) ⇒ damp ≈ −10%', () => {
    const r: ConvergenceResult = { converged: true, daysToConverge: 35, direction: 'yes' };
    const f = convergenceTimeFactor(r);
    expect(f).toBeGreaterThan(0.85);
    expect(f).toBeLessThan(0.95);
  });

  test('intermediate (14d) ⇒ moderate factor between damp and boost', () => {
    const r: ConvergenceResult = { converged: true, daysToConverge: 14, direction: 'yes' };
    const f = convergenceTimeFactor(r);
    expect(f).toBeGreaterThan(0.95);
    expect(f).toBeLessThan(1.10);
  });

  test('monotone decreasing in daysToConverge', () => {
    const a = convergenceTimeFactor({ converged: true, daysToConverge: 3, direction: 'yes' });
    const b = convergenceTimeFactor({ converged: true, daysToConverge: 14, direction: 'yes' });
    const c = convergenceTimeFactor({ converged: true, daysToConverge: 35, direction: 'yes' });
    expect(a).toBeGreaterThan(b);
    expect(b).toBeGreaterThan(c);
  });

  test('NO convergence applies asymmetric speed-up (40% faster effective horizon)', () => {
    // Voigt: NO converges ~40% faster, so equivalent days-to-confidence is days/0.6.
    // For same daysToConverge, NO should be treated as MORE confident (higher factor)
    // than YES — because the effective convergence speed is normalised.
    const yesFactor = convergenceTimeFactor({ converged: true, daysToConverge: 14, direction: 'yes' });
    const noFactor = convergenceTimeFactor({ converged: true, daysToConverge: 14, direction: 'no' });
    expect(noFactor).toBeGreaterThan(yesFactor);
  });

  test('factor is bounded in [0.85, 1.20] across realistic inputs', () => {
    for (const d of [1, 3, 7, 14, 21, 30, 60, 120]) {
      for (const dir of ['yes', 'no'] as const) {
        const f = convergenceTimeFactor({ converged: true, daysToConverge: d, direction: dir });
        expect(f).toBeGreaterThanOrEqual(0.85);
        expect(f).toBeLessThanOrEqual(1.20);
      }
    }
  });
});
