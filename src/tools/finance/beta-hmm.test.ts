/**
 * Tests for Beta-HMM (Voigt 2025) — Hidden Markov Model with Beta emissions
 * for bounded [0,1] data such as Polymarket prices.
 */

import { describe, expect, test } from 'bun:test';
import {
  betaPdf,
  fitBetaMoM,
  initializeBetaHMM,
  forwardBeta,
  baumWelchBeta,
  viterbiBeta,
  type BetaHMMParams,
} from './beta-hmm.js';

function approx(a: number, b: number, tol = 1e-6): boolean {
  return Math.abs(a - b) <= tol;
}

describe('betaPdf', () => {
  test('returns 0 outside (0,1)', () => {
    expect(betaPdf(-0.1, { alpha: 2, beta: 5 })).toBe(0);
    expect(betaPdf(0, { alpha: 2, beta: 5 })).toBe(0);
    expect(betaPdf(1, { alpha: 2, beta: 5 })).toBe(0);
    expect(betaPdf(1.1, { alpha: 2, beta: 5 })).toBe(0);
  });

  test('uniform Beta(1,1) has pdf == 1 on (0,1)', () => {
    for (const x of [0.1, 0.3, 0.5, 0.7, 0.9]) {
      expect(approx(betaPdf(x, { alpha: 1, beta: 1 }), 1, 1e-9)).toBe(true);
    }
  });

  test('integrates (Riemann) to ~1 for several α,β', () => {
    const cases: Array<[number, number]> = [
      [2, 5],
      [5, 2],
      [3, 3],
      [0.7, 0.7], // U-shape
    ];
    for (const [a, b] of cases) {
      const N = 10_000;
      let sum = 0;
      for (let i = 1; i < N; i++) {
        const x = i / N;
        sum += betaPdf(x, { alpha: a, beta: b });
      }
      const integral = sum / N;
      expect(integral).toBeGreaterThan(0.99);
      expect(integral).toBeLessThan(1.01);
    }
  });

  test('peak of Beta(2,5) is near (α-1)/(α+β-2) = 1/5', () => {
    const peak = betaPdf(0.2, { alpha: 2, beta: 5 });
    const off1 = betaPdf(0.05, { alpha: 2, beta: 5 });
    const off2 = betaPdf(0.5, { alpha: 2, beta: 5 });
    expect(peak).toBeGreaterThan(off1);
    expect(peak).toBeGreaterThan(off2);
  });
});

describe('fitBetaMoM', () => {
  test('recovers α=2, β=5 from samples within tolerance', () => {
    // Generate a deterministic sample of Beta(2,5) using inverse CDF approximation.
    // Method-of-moments: α + β = m(1-m)/v - 1 ; α = (α+β) m
    const samples: number[] = [];
    const rng = mulberry32(42);
    for (let i = 0; i < 5000; i++) samples.push(sampleBeta(2, 5, rng));
    const weights = samples.map(() => 1);
    const fit = fitBetaMoM(weights, samples);
    expect(fit.alpha).toBeGreaterThan(1.6);
    expect(fit.alpha).toBeLessThan(2.5);
    expect(fit.beta).toBeGreaterThan(4.0);
    expect(fit.beta).toBeLessThan(6.5);
  });

  test('weighted MoM ignores zero-weighted samples', () => {
    const samples = [0.1, 0.9, 0.5, 0.5, 0.5];
    const w1 = [1, 1, 0, 0, 0]; // m=0.5 v=0.16 => α+β = 0.5*0.5/0.16 - 1 = 0.5625 ; α≈0.28
    const fit = fitBetaMoM(w1, samples);
    expect(approx((fit.alpha + fit.beta), 0.5625, 1e-3)).toBe(true);
  });

  test('returns α=β=1 when variance is zero', () => {
    const fit = fitBetaMoM([1, 1, 1], [0.4, 0.4, 0.4]);
    expect(fit.alpha).toBe(1);
    expect(fit.beta).toBe(1);
  });

  test('clips samples away from the boundary so MoM stays finite', () => {
    const fit = fitBetaMoM([1, 1, 1, 1], [0.0, 1.0, 0.5, 0.5]);
    expect(Number.isFinite(fit.alpha)).toBe(true);
    expect(Number.isFinite(fit.beta)).toBe(true);
    expect(fit.alpha).toBeGreaterThan(0);
    expect(fit.beta).toBeGreaterThan(0);
  });
});

describe('forwardBeta', () => {
  test('produces normalized scaled alphas (sum to 1 per timestep)', () => {
    const params: BetaHMMParams = {
      nStates: 2,
      pi: [0.5, 0.5],
      A: [[0.9, 0.1], [0.1, 0.9]],
      emissions: [{ alpha: 2, beta: 8 }, { alpha: 8, beta: 2 }],
    };
    const obs = [0.1, 0.15, 0.2, 0.85, 0.9];
    const { alpha } = forwardBeta(obs, params);
    for (let t = 0; t < obs.length; t++) {
      const s = alpha[t][0] + alpha[t][1];
      expect(approx(s, 1, 1e-9)).toBe(true);
    }
  });
});

describe('baumWelchBeta', () => {
  test('recovers regimes on a synthetic 2-state Beta sequence at >85% accuracy', () => {
    const rng = mulberry32(123);
    const n = 400;
    const trueStates: number[] = [];
    const obs: number[] = [];
    let state = 0;
    for (let t = 0; t < n; t++) {
      // Sticky 2-state chain: 90% stay, 10% switch
      if (rng() < 0.1) state = 1 - state;
      trueStates.push(state);
      // State 0: Beta(2, 8) (low-price regime, mean ~0.2)
      // State 1: Beta(8, 2) (high-price regime, mean ~0.8)
      const a = state === 0 ? 2 : 8;
      const b = state === 0 ? 8 : 2;
      obs.push(sampleBeta(a, b, rng));
    }
    const init = initializeBetaHMM(obs, 2);
    const fit = baumWelchBeta(obs, init, 30, 1e-4);
    expect(fit.converged || fit.iterations > 0).toBe(true);
    const path = viterbiBeta(obs, fit.params);

    // Two label permutations are valid; pick the one that scores higher.
    let agree = 0;
    for (let t = 0; t < n; t++) if (path[t] === trueStates[t]) agree++;
    let acc = agree / n;
    if (acc < 0.5) acc = 1 - acc;
    expect(acc).toBeGreaterThan(0.85);
  });
});

// ---------------------------------------------------------------------------
// Test helpers (deterministic RNG + Beta sampling via rejection)
// ---------------------------------------------------------------------------

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

// Marsaglia-Tsang gamma sampler → Beta via X / (X + Y).
function sampleGamma(shape: number, rng: () => number): number {
  if (shape < 1) {
    // Boost trick: Gamma(shape) = Gamma(shape+1) * U^(1/shape)
    return sampleGamma(shape + 1, rng) * Math.pow(rng(), 1 / shape);
  }
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  while (true) {
    let x = 0, v = 0;
    do {
      x = boxMuller(rng);
      v = 1 + c * x;
    } while (v <= 0);
    v = v * v * v;
    const u = rng();
    if (u < 1 - 0.0331 * x * x * x * x) return d * v;
    if (Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function boxMuller(rng: () => number): number {
  const u1 = Math.max(rng(), 1e-12);
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function sampleBeta(a: number, b: number, rng: () => number): number {
  const x = sampleGamma(a, rng);
  const y = sampleGamma(b, rng);
  const v = x / (x + y);
  return Math.min(0.999_999, Math.max(1e-6, v));
}
