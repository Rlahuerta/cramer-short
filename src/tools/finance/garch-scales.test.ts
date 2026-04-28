import { describe, expect, it } from 'bun:test';
import { computeGarchScales } from './garch-scales.js';

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

describe('computeGarchScales', () => {
  it('returns empty array on insufficient data', () => {
    expect(computeGarchScales([0.01, -0.01], 5)).toEqual([]);
    expect(computeGarchScales([], 5)).toEqual([]);
    expect(computeGarchScales([0.01, 0.02, 0.03, 0.04, 0.05], 0)).toEqual([]);
  });

  it('returns empty array on zero-variance series', () => {
    const flat = new Array(50).fill(0);
    expect(computeGarchScales(flat, 5)).toEqual([]);
  });

  it('produces horizon-length array of finite positive scalars', () => {
    const rng = seedRng(42);
    const r = Array.from({ length: 200 }, () => gauss(rng) * 0.02);
    const scales = computeGarchScales(r, 14);
    expect(scales).toHaveLength(14);
    for (const s of scales) {
      expect(Number.isFinite(s)).toBe(true);
      expect(s).toBeGreaterThan(0);
      expect(s).toBeLessThan(3.01);
    }
  });

  it('on recent vol-burst, near-term scales > 1 and decay toward unconditional', () => {
    const rng = seedRng(7);
    // 150 days of low vol (0.005), then 30 days of high vol (0.04).
    const calm = Array.from({ length: 150 }, () => gauss(rng) * 0.005);
    const burst = Array.from({ length: 30 }, () => gauss(rng) * 0.04);
    const series = [...calm, ...burst];
    const scales = computeGarchScales(series, 30);

    expect(scales).toHaveLength(30);
    // Day 1 should reflect the recent burst → significantly above 1.
    expect(scales[0]).toBeGreaterThan(1.1);
    // By day 29 it should have decayed toward (but not below) the
    // unconditional level.  Because persistence α+β = 0.95, decay is slow,
    // but day 29 must be lower than day 1.
    expect(scales[29]).toBeLessThan(scales[0]);
  });

  it('on calm-then-recent-calm series, scales near 1 (no clustering signal)', () => {
    const rng = seedRng(123);
    const r = Array.from({ length: 200 }, () => gauss(rng) * 0.01);
    const scales = computeGarchScales(r, 10);
    // All scales should be well within [0.7, 1.3] when the series is i.i.d. gaussian.
    for (const s of scales) {
      expect(s).toBeGreaterThan(0.5);
      expect(s).toBeLessThan(1.7);
    }
  });

  it('is deterministic for same input', () => {
    const rng = seedRng(99);
    const r = Array.from({ length: 100 }, () => gauss(rng) * 0.015);
    const a = computeGarchScales(r, 14);
    const b = computeGarchScales(r, 14);
    expect(a).toEqual(b);
  });
});
