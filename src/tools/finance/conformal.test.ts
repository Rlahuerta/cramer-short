/**
 * W3 Idea 2 — Online conformal PID wrapper tests.
 *
 * Reference: Angelopoulos, Candès & Tibshirani (2023), "Conformal PID Control
 * for Time Series Prediction", arXiv 2307.16895.
 *
 * The wrapper takes a stream of (forecastCenter, actual) pairs and produces a
 * prediction radius `q` that adapts so the long-run miscoverage approaches a
 * target α (e.g. α=0.1 for 90% coverage). Pure post-processing: it never
 * touches the underlying forecast model.
 */
import { describe, it, expect } from 'bun:test';
import { ConformalPID } from './conformal.js';

// Deterministic Gaussian sampler (Box–Muller) for reproducible tests.
function makeGaussian(seed: number): () => number {
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
  return () => {
    const u1 = Math.max(rand(), 1e-12);
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  };
}

describe('ConformalPID — construction', () => {
  it('builds with the default α=0.1 → 90% target coverage', () => {
    const c = new ConformalPID();
    expect(c.targetCoverage).toBeCloseTo(0.9, 6);
    expect(c.alpha).toBeCloseTo(0.1, 6);
  });

  it('respects a custom alpha', () => {
    const c = new ConformalPID({ alpha: 0.05 });
    expect(c.alpha).toBeCloseTo(0.05, 6);
    expect(c.targetCoverage).toBeCloseTo(0.95, 6);
  });

  it('seeds the radius with the configured initialRadius', () => {
    const c = new ConformalPID({ initialRadius: 2.5 });
    expect(c.currentRadius()).toBeCloseTo(2.5, 6);
  });

  it('exposes wrap(center) → {low, high} as the bare-bones API', () => {
    const c = new ConformalPID({ initialRadius: 1.0 });
    const interval = c.wrap(100);
    expect(interval.low).toBeCloseTo(99, 6);
    expect(interval.high).toBeCloseTo(101, 6);
  });
});

describe('ConformalPID — quantile convergence on i.i.d. Gaussian residuals', () => {
  it('converges toward the 90% one-sided radius (≈1.645σ) on σ=1 noise', () => {
    const gauss = makeGaussian(42);
    const c = new ConformalPID({ alpha: 0.1, initialRadius: 0.5, learningRate: 0.05 });
    for (let i = 0; i < 5_000; i++) {
      c.record(0, gauss());
    }
    // True 90% absolute-residual quantile of N(0,1) ≈ 1.645.
    expect(c.currentRadius()).toBeGreaterThan(1.4);
    expect(c.currentRadius()).toBeLessThan(1.9);
  });

  it('achieves empirical coverage within ±2pp of the 90% target after warm-up', () => {
    const gauss = makeGaussian(123);
    const c = new ConformalPID({ alpha: 0.1, initialRadius: 1.0, learningRate: 0.05 });
    let covered = 0;
    let total = 0;
    for (let i = 0; i < 5_000; i++) {
      const actual = gauss();
      if (i >= 1_000) {
        const interval = c.wrap(0);
        if (actual >= interval.low && actual <= interval.high) covered++;
        total++;
      }
      c.record(0, actual);
    }
    const empirical = covered / total;
    expect(empirical).toBeGreaterThan(0.88);
    expect(empirical).toBeLessThan(0.92);
  });

  it('adapts upward after a volatility regime shift (σ=1 → σ=3)', () => {
    const lo = makeGaussian(7);
    const hi = makeGaussian(8);
    const c = new ConformalPID({ alpha: 0.1, initialRadius: 1.0, learningRate: 0.05 });
    for (let i = 0; i < 2_000; i++) c.record(0, lo());
    const radiusAfterLowVol = c.currentRadius();
    for (let i = 0; i < 2_000; i++) c.record(0, 3 * hi());
    const radiusAfterHighVol = c.currentRadius();
    expect(radiusAfterHighVol).toBeGreaterThan(radiusAfterLowVol * 1.5);
  });
});

describe('ConformalPID — non-negativity and stability', () => {
  it('never returns a negative radius even with extreme overcoverage pressure', () => {
    const c = new ConformalPID({ alpha: 0.1, initialRadius: 0.01, learningRate: 0.5 });
    // Feed perfectly-on-target predictions repeatedly — should never push q < 0.
    for (let i = 0; i < 1_000; i++) c.record(0, 0);
    expect(c.currentRadius()).toBeGreaterThanOrEqual(0);
  });

  it('integral term decays via gamma to avoid runaway accumulation', () => {
    // Provide an integralDecay parameter.
    const c = new ConformalPID({ alpha: 0.1, initialRadius: 1.0, learningRate: 0.05, integralDecay: 0.99 });
    for (let i = 0; i < 200; i++) c.record(0, 5); // permanent miscoverage shock
    const radiusAfterShock = c.currentRadius();
    // Should eventually grow to capture the residual, not blow up arbitrarily large.
    expect(radiusAfterShock).toBeGreaterThan(0.5);
    expect(radiusAfterShock).toBeLessThan(50);
  });
});

describe('ConformalPID — coverage helpers', () => {
  it('reports running empirical coverage that converges toward 1−α', () => {
    const c = new ConformalPID({ alpha: 0.1, initialRadius: 1.0, learningRate: 0.05 });
    // Constant ±0.5 residuals — the controller should drive q toward 0.5,
    // landing at exactly the target 90% empirical coverage.
    for (let i = 0; i < 200; i++) c.record(0, i % 2 === 0 ? 0.5 : -0.5);
    expect(c.empiricalCoverage()!).toBeGreaterThan(0.85);
    expect(c.empiricalCoverage()!).toBeLessThan(1.0);
    expect(c.sampleCount()).toBe(200);
  });

  it('returns 0 sample count and undefined coverage with no data', () => {
    const c = new ConformalPID();
    expect(c.sampleCount()).toBe(0);
    expect(c.empiricalCoverage()).toBeUndefined();
  });
});
