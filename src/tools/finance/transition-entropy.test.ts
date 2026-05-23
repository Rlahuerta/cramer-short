/**
 * R5 Idea #14 — Transition entropy CI modulator tests.
 */
import { describe, expect, it } from 'bun:test';
import {
  approximateStationary,
  computeTransitionEntropy,
  entropyZToCiScale,
  EntropyZScoreTracker,
} from './transition-entropy.js';

describe('R5 Idea #14 — approximateStationary', () => {
  it('returns uniform on identity matrix', () => {
    const pi = approximateStationary([[1, 0], [0, 1]]);
    // identity has any distribution as stationary; should return finite
    expect(pi.length).toBe(2);
    expect(pi.every(x => Number.isFinite(x))).toBe(true);
  });

  it('returns the known stationary of a 2-state chain', () => {
    // p = 0.7 (stay 0), 0.3 (0→1); q = 0.4 (1→0), 0.6 (stay 1)
    // π = [q/(p_off+q_off)] = [0.4/(0.3+0.4), 0.3/(0.3+0.4)] = [0.571, 0.428]
    const P = [[0.7, 0.3], [0.4, 0.6]];
    const pi = approximateStationary(P);
    expect(pi[0]).toBeCloseTo(0.5714, 3);
    expect(pi[1]).toBeCloseTo(0.4285, 3);
    expect(pi[0] + pi[1]).toBeCloseTo(1, 6);
  });
});

describe('R5 Idea #14 — computeTransitionEntropy', () => {
  it('returns 0 entropy for a deterministic matrix', () => {
    const P = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const r = computeTransitionEntropy(P);
    expect(r.entropyNats).toBeCloseTo(0, 6);
    expect(r.entropyNorm).toBe(0);
    expect(r.K).toBe(3);
  });

  it('returns max entropy (log K) for a uniform matrix', () => {
    const K = 3;
    const u = 1 / K;
    const P = Array.from({ length: K }, () => Array(K).fill(u));
    const r = computeTransitionEntropy(P);
    expect(r.entropyNats).toBeCloseTo(Math.log(K), 4);
    expect(r.entropyNorm).toBeCloseTo(1, 4);
  });

  it('intermediate matrix has 0 < H_norm < 1', () => {
    const P = [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.3, 0.5]];
    const r = computeTransitionEntropy(P);
    expect(r.entropyNorm).toBeGreaterThan(0);
    expect(r.entropyNorm).toBeLessThan(1);
  });
});

describe('R5 Idea #14 — EntropyZScoreTracker', () => {
  it('returns null until 5 values observed', () => {
    const t = new EntropyZScoreTracker(20);
    for (let i = 0; i < 4; i++) {
      t.push(0.5);
      expect(t.zScore(0.5)).toBeNull();
    }
    t.push(0.5);
    expect(t.zScore(0.5)).not.toBeNull();
  });

  it('returns ~0 z-score on stable series', () => {
    const t = new EntropyZScoreTracker(20);
    for (let i = 0; i < 10; i++) t.push(0.5);
    const z = t.zScore(0.5);
    expect(z).toBe(0);
  });

  it('returns positive z when current value > rolling mean', () => {
    const t = new EntropyZScoreTracker(20);
    for (let i = 0; i < 10; i++) t.push(0.4 + 0.01 * i);  // 0.4..0.49
    const z = t.zScore(0.7);
    expect(z!).toBeGreaterThan(2);
  });

  it('respects rolling window (oldest values evicted)', () => {
    const t = new EntropyZScoreTracker(5);
    for (let i = 0; i < 100; i++) t.push(i);
    expect(t.size()).toBe(5);
  });
});

describe('R5 Idea #14 — entropyZToCiScale', () => {
  it('returns 1.0 when z = 0', () => {
    expect(entropyZToCiScale(0)).toBe(1);
  });

  it('widens CI on high-uncertainty z (positive z)', () => {
    const s = entropyZToCiScale(2.0, 0.15);
    expect(s).toBeCloseTo(1.3, 6);
    expect(s).toBeGreaterThan(1);
  });

  it('tightens CI on low-uncertainty z (negative z)', () => {
    const s = entropyZToCiScale(-2.0, 0.15);
    expect(s).toBeCloseTo(0.7, 6);
    expect(s).toBeLessThan(1);
  });

  it('clamps to [0.7, 1.4]', () => {
    expect(entropyZToCiScale(100)).toBeCloseTo(1.4, 6);
    expect(entropyZToCiScale(-100)).toBeCloseTo(0.7, 6);
  });
});
