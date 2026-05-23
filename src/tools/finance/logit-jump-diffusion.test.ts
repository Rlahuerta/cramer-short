/**
 * Tests for logit-jump-diffusion — prediction-market price dynamics in
 * log-odds space with a martingale-constrained drift.
 */

import { describe, expect, test } from 'bun:test';
import {
  logit,
  invLogit,
  itoMartingaleDrift,
  simulateLogitJumpDiffusion,
  type LogitJumpDiffusionParams,
} from './logit-jump-diffusion.js';

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

describe('logit / invLogit', () => {
  test('round-trip', () => {
    for (const p of [0.05, 0.1, 0.3, 0.5, 0.7, 0.95]) {
      expect(invLogit(logit(p))).toBeCloseTo(p, 12);
    }
  });

  test('logit clips boundary inputs to a finite z', () => {
    expect(Number.isFinite(logit(0))).toBe(true);
    expect(Number.isFinite(logit(1))).toBe(true);
  });

  test('invLogit always returns a value strictly in (0, 1)', () => {
    for (const z of [-50, -1, 0, 1, 50]) {
      const p = invLogit(z);
      expect(p).toBeGreaterThan(0);
      expect(p).toBeLessThan(1);
    }
  });
});

describe('itoMartingaleDrift', () => {
  test('zero at p=0.5 (symmetry)', () => {
    expect(itoMartingaleDrift(0.5, 0.1, 0, 0, 0)).toBeCloseTo(0, 12);
  });

  test('negative when p < 0.5 (counters convex-region upward Jensen bias)', () => {
    // sigmoid is convex for x<0 ⇒ E[σ(x+ε)] > σ(x) ⇒ need negative drift in x.
    expect(itoMartingaleDrift(0.2, 0.1, 0, 0, 0)).toBeLessThan(0);
  });

  test('positive when p > 0.5 (counters concave-region downward Jensen bias)', () => {
    expect(itoMartingaleDrift(0.8, 0.1, 0, 0, 0)).toBeGreaterThan(0);
  });

  test('grows with diffusion variance', () => {
    const a = Math.abs(itoMartingaleDrift(0.2, 0.1, 0, 0, 0));
    const b = Math.abs(itoMartingaleDrift(0.2, 0.5, 0, 0, 0));
    expect(b).toBeGreaterThan(a);
  });
});

describe('simulateLogitJumpDiffusion', () => {
  const baseParams = (overrides: Partial<LogitJumpDiffusionParams> = {}): LogitJumpDiffusionParams => ({
    initialPrice: 0.30,
    days: 30,
    sigmaPerDay: 0.05,
    jumpIntensityPerDay: 0,
    jumpLogitMean: 0,
    jumpLogitStd: 0,
    nPaths: 2000,
    rng: mulberry32(7),
    ...overrides,
  });

  test('all simulated prices are strictly in (0, 1)', () => {
    const out = simulateLogitJumpDiffusion(baseParams({ sigmaPerDay: 0.20 }));
    for (let i = 0; i < out.terminal.length; i++) {
      expect(out.terminal[i]).toBeGreaterThan(0);
      expect(out.terminal[i]).toBeLessThan(1);
    }
  });

  test('zero volatility + zero jumps ⇒ prices are constant', () => {
    const out = simulateLogitJumpDiffusion(baseParams({ sigmaPerDay: 0, nPaths: 500 }));
    for (let i = 0; i < out.terminal.length; i++) {
      expect(out.terminal[i]).toBeCloseTo(0.30, 8);
    }
  });

  test('martingale: E[p_T] ≈ p_0 (diffusion only, large ensemble)', () => {
    const out = simulateLogitJumpDiffusion(baseParams({
      initialPrice: 0.30,
      sigmaPerDay: 0.10,
      nPaths: 5000,
    }));
    const mean = out.terminal.reduce((s, x) => s + x, 0) / out.terminal.length;
    // Sampling SE for n=5000, std ≈ 0.1 → ~0.0014; allow 0.01 slack.
    expect(Math.abs(mean - 0.30)).toBeLessThan(0.01);
  });

  test('martingale: E[p_T] ≈ p_0 holds with jumps too', () => {
    const out = simulateLogitJumpDiffusion(baseParams({
      initialPrice: 0.50,
      sigmaPerDay: 0.05,
      jumpIntensityPerDay: 0.02,
      jumpLogitMean: 0.5,
      jumpLogitStd: 0.3,
      nPaths: 8000,
    }));
    const mean = out.terminal.reduce((s, x) => s + x, 0) / out.terminal.length;
    expect(Math.abs(mean - 0.50)).toBeLessThan(0.02);
  });

  test('higher σ ⇒ wider terminal dispersion', () => {
    const lo = simulateLogitJumpDiffusion(baseParams({ sigmaPerDay: 0.05, nPaths: 3000 }));
    const hi = simulateLogitJumpDiffusion(baseParams({ sigmaPerDay: 0.20, nPaths: 3000, rng: mulberry32(8) }));
    const std = (xs: number[]): number => {
      const m = xs.reduce((s, x) => s + x, 0) / xs.length;
      return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length);
    };
    expect(std(hi.terminal)).toBeGreaterThan(std(lo.terminal));
  });

  test('positive jump intensity raises terminal dispersion', () => {
    const noJump = simulateLogitJumpDiffusion(baseParams({
      initialPrice: 0.50, sigmaPerDay: 0.05, jumpIntensityPerDay: 0, nPaths: 3000,
    }));
    const withJump = simulateLogitJumpDiffusion(baseParams({
      initialPrice: 0.50, sigmaPerDay: 0.05,
      jumpIntensityPerDay: 0.10, jumpLogitMean: 0, jumpLogitStd: 1.0,
      nPaths: 3000, rng: mulberry32(9),
    }));
    const std = (xs: number[]): number => {
      const m = xs.reduce((s, x) => s + x, 0) / xs.length;
      return Math.sqrt(xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length);
    };
    expect(std(withJump.terminal)).toBeGreaterThan(std(noJump.terminal));
  });

  test('returns full per-day matrix when storePaths=true', () => {
    const out = simulateLogitJumpDiffusion(baseParams({ days: 5, nPaths: 100, storePaths: true }));
    expect(out.paths).toBeDefined();
    expect(out.paths!.length).toBe(100);
    expect(out.paths![0].length).toBe(5);
  });

  test('Polymarket-informed intensity scales with horizon-implied λ', () => {
    // polymarketJumpProb=0.30 over 30 days ⇒ daily λ = 0.30/30 = 0.01.
    const out = simulateLogitJumpDiffusion({
      initialPrice: 0.5,
      days: 30,
      sigmaPerDay: 0,
      polymarketJumpProb: 0.30,
      jumpLogitMean: 1.0,
      jumpLogitStd: 0.1,
      nPaths: 4000,
      rng: mulberry32(11),
    });
    // Empirical jump-event rate per path-day must be close to 0.01.
    const rate = out.totalJumps / (out.terminal.length * 30);
    expect(rate).toBeGreaterThan(0.005);
    expect(rate).toBeLessThan(0.02);
  });
});
