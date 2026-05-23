/**
 * R5 Idea #11 — Longshot-bias shrinkage tests.
 */
import { describe, expect, it } from 'bun:test';
import { applyLongshotShrinkage } from './rnd-integration.js';

describe('R5 Idea #11 — applyLongshotShrinkage', () => {
  it('passes through probabilities in the safe zone', () => {
    for (const p of [0.10, 0.30, 0.50, 0.70, 0.90]) {
      const r = applyLongshotShrinkage(p);
      expect(r.applied).toBe(false);
      expect(r.p).toBe(p);
    }
  });

  it('shrinks longshots (p < 0.05) toward 0.5', () => {
    const r = applyLongshotShrinkage(0.02);
    expect(r.applied).toBe(true);
    // 0.5*0.5 + 0.5*0.02 = 0.26
    expect(r.p).toBeCloseTo(0.26, 6);
  });

  it('shrinks favorites (p > 0.95) toward 0.5', () => {
    const r = applyLongshotShrinkage(0.98);
    expect(r.applied).toBe(true);
    // 0.5*0.5 + 0.5*0.98 = 0.74
    expect(r.p).toBeCloseTo(0.74, 6);
  });

  it('respects custom weight (w=0.2 ⇒ lighter shrinkage)', () => {
    const r = applyLongshotShrinkage(0.02, { weight: 0.2 });
    // 0.2*0.5 + 0.8*0.02 = 0.116
    expect(r.p).toBeCloseTo(0.116, 6);
  });

  it('respects custom thresholds', () => {
    const r = applyLongshotShrinkage(0.08, { lowThreshold: 0.10 });
    expect(r.applied).toBe(true);
  });

  it('clamps result to [0,1]', () => {
    const r = applyLongshotShrinkage(0.0);
    expect(r.p).toBeGreaterThanOrEqual(0);
    expect(r.p).toBeLessThanOrEqual(1);
  });

  it('reports tail distance for diagnostics', () => {
    const r = applyLongshotShrinkage(0.02);
    expect(r.tailDistance).toBeCloseTo(0.48, 6);
  });
});
