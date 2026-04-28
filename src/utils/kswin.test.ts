import { describe, expect, it } from 'bun:test';
import { detectKswinDrift, kolmogorovSmirnovD } from './kswin.js';

function seedRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function gauss(rng: () => number): number {
  const u = Math.max(1e-12, rng());
  const v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

describe('kolmogorovSmirnovD', () => {
  it('returns 0 for identical samples', () => {
    const a = [1, 2, 3, 4, 5];
    expect(kolmogorovSmirnovD(a, a)).toBe(0);
  });

  it('returns ~1 for fully disjoint sets', () => {
    expect(kolmogorovSmirnovD([1, 2, 3], [10, 20, 30])).toBe(1);
  });

  it('returns 0 on empty input', () => {
    expect(kolmogorovSmirnovD([], [1, 2, 3])).toBe(0);
    expect(kolmogorovSmirnovD([1, 2, 3], [])).toBe(0);
  });

  it('detects scale shift between two gaussian samples', () => {
    const rng = seedRng(11);
    const small = Array.from({ length: 100 }, () => Math.abs(gauss(rng)) * 0.005);
    const big = Array.from({ length: 100 }, () => Math.abs(gauss(rng)) * 0.04);
    const d = kolmogorovSmirnovD(small, big);
    // KS for ~8x scale ratio with n=m=100 → D should be very large.
    expect(d).toBeGreaterThan(0.5);
  });
});

describe('detectKswinDrift', () => {
  it('returns no drift for short input', () => {
    const r = detectKswinDrift([1, 2, 3, 4, 5]);
    expect(r.drift).toBe(false);
    expect(r.keepCount).toBe(5);
  });

  it('returns no drift for stationary i.i.d. series', () => {
    const rng = seedRng(42);
    const series = Array.from({ length: 300 }, () => Math.abs(gauss(rng)) * 0.01);
    const r = detectKswinDrift(series);
    expect(r.drift).toBe(false);
    expect(r.maxD).toBeLessThan(r.criticalD);
  });

  it('detects variance-step drift that ADWIN would miss', () => {
    const rng = seedRng(7);
    // 200 calm samples (σ≈0.005), then 80 burst samples (σ≈0.04).
    // Mean of |ret| jumps from ~0.004 to ~0.032 — KS will fire.
    const calm = Array.from({ length: 200 }, () => Math.abs(gauss(rng)) * 0.005);
    const burst = Array.from({ length: 80 }, () => Math.abs(gauss(rng)) * 0.04);
    const series = [...calm, ...burst];
    const r = detectKswinDrift(series);
    expect(r.drift).toBe(true);
    expect(r.maxD).toBeGreaterThan(r.criticalD);
    // Should keep substantially fewer than the full 280 samples.
    expect(r.keepCount).toBeLessThan(series.length);
    // But never below minKeep (default 60).
    expect(r.keepCount).toBeGreaterThanOrEqual(60);
  });

  it('respects minKeep floor', () => {
    const rng = seedRng(13);
    const calm = Array.from({ length: 200 }, () => Math.abs(gauss(rng)) * 0.005);
    const burst = Array.from({ length: 30 }, () => Math.abs(gauss(rng)) * 0.04);
    const series = [...calm, ...burst];
    const r = detectKswinDrift(series, { minKeep: 100 });
    if (r.drift) expect(r.keepCount).toBeGreaterThanOrEqual(100);
  });

  it('is deterministic', () => {
    const rng = seedRng(99);
    const series = Array.from({ length: 250 }, () => Math.abs(gauss(rng)) * 0.01);
    const a = detectKswinDrift(series);
    const b = detectKswinDrift(series);
    expect(a).toEqual(b);
  });

  it('with smaller windows triggers earlier on rapid drift', () => {
    const rng = seedRng(21);
    const calm = Array.from({ length: 100 }, () => Math.abs(gauss(rng)) * 0.005);
    const burst = Array.from({ length: 40 }, () => Math.abs(gauss(rng)) * 0.05);
    const series = [...calm, ...burst];
    const r = detectKswinDrift(series, { referenceWindow: 30, recentWindow: 30 });
    expect(r.drift).toBe(true);
  });
});
