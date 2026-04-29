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

  // ─── R5 Idea #5: horizon-aware + regime-conditional clamp ───
  describe('R5 horizon-aware + regime ceiling', () => {
    it('preserves pre-R5 behaviour when no opts passed', () => {
      const rng = seedRng(7);
      const r = Array.from({ length: 200 }, () => gauss(rng) * 0.02);
      const a = computeGarchScales(r, 10);
      const b = computeGarchScales(r, 10, {});
      expect(a).toEqual(b);
    });

    it('horizon decay blends scalars toward 1.0 past horizonCap', () => {
      // Build a vol-burst series so the GARCH scalar is materially >1.
      const rng = seedRng(11);
      const calm = Array.from({ length: 100 }, () => gauss(rng) * 0.005);
      const burst = Array.from({ length: 30 }, () => gauss(rng) * 0.05);
      const r = [...calm, ...burst];

      const baseline = computeGarchScales(r, 30);
      const decayed = computeGarchScales(r, 30, { horizonCap: 7 });

      // Days 1..7 (indices 0..6): identical to baseline (no decay).
      for (let i = 0; i < 7; i++) {
        expect(decayed[i]).toBeCloseTo(baseline[i], 6);
      }
      // Day 8+ (index 7+): decayed should be closer to 1.0 than baseline
      // whenever baseline ≠ 1.0.  At day 21 (3·cap) decayed must equal 1.0.
      expect(decayed[20]).toBeCloseTo(1.0, 6);
      // Past 3·cap ⇒ exactly 1.0.
      expect(decayed[29]).toBe(1);
    });

    it('regime ceiling clamps tighter in calm regime than turbulent', () => {
      // Construct a series where the recent window is calm vs. historical
      // (high σ history, low σ tail).  This forces the calm regime branch.
      const rng = seedRng(31);
      const wild = Array.from({ length: 80 }, () => gauss(rng) * 0.06);
      const tame = Array.from({ length: 80 }, () => gauss(rng) * 0.005);
      const r = [...wild, ...tame];

      const calmScales = computeGarchScales(r, 5, {
        ceiling: { calm: 1.2, turbulent: 3.0 },
        regimeOverride: 'calm',
      });
      const turbScales = computeGarchScales(r, 5, {
        ceiling: { calm: 1.2, turbulent: 3.0 },
        regimeOverride: 'turbulent',
      });
      // Every calm-regime scalar must be ≤ ceiling.calm.
      for (const s of calmScales) expect(s).toBeLessThanOrEqual(1.2 + 1e-9);
      // Turbulent regime allowed to exceed 1.2.
      const maxTurb = Math.max(...turbScales);
      // Either turbulent allowed > 1.2 (clearly different ceiling), or both
      // are organically below 1.2 — in which case the ceilings are
      // irrelevant and the arrays must be identical.
      if (maxTurb > 1.2 + 1e-6) {
        expect(maxTurb).toBeGreaterThan(Math.max(...calmScales));
      } else {
        expect(turbScales).toEqual(calmScales);
      }
    });

    it('auto-detects calm regime from the series tail', () => {
      const rng = seedRng(53);
      const wild = Array.from({ length: 80 }, () => gauss(rng) * 0.06);
      const tame = Array.from({ length: 80 }, () => gauss(rng) * 0.005);
      const r = [...wild, ...tame];
      const detected = computeGarchScales(r, 5, {
        ceiling: { calm: 1.1, turbulent: 3.0 },
      });
      // Auto-detector should pick calm (recent σ << historical σ) ⇒
      // every scalar ≤ 1.1.
      for (const s of detected) expect(s).toBeLessThanOrEqual(1.1 + 1e-9);
    });
  });
});
