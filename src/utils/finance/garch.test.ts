/**
 * P3a — GARCH(1,1) interim volatility helper (TDD).
 *
 * Source: docs/polymarket-prediction-improvements-research-2026-07.md §10.3
 *
 *   h_t  = ω + α · z_{t-1}² · h_{t-1} + β · h_{t-1}
 *   σ_t  = √h_t
 *
 * Pure helpers under test:
 *   - fitGarch11(returns) — moment-matching estimator returning {omega, alpha, beta, h0}
 *   - garchStep(prevH, prevZ, p) → next h
 *   - garchForecast(p, horizonDays) → array of σ_t for t = 1..horizonDays
 *
 * Note: this is not a full MLE GARCH fit. It uses a fixed-point initialisation
 * (alpha=0.10, beta=0.85) and matches the unconditional variance to the sample
 * variance — sufficient for trajectory MC use as Bloch's "interim" alternative
 * to MSM.
 */

import { describe, expect, it } from 'bun:test';
import { fitGarch11, garchForecast, garchStep, GARCH_DEFAULTS } from './garch.js';

function makeReturns(n: number, vol: number, seed: number = 1): number[] {
  // Deterministic pseudo-random Gaussian using Box-Muller.
  let s = seed;
  const rng = () => {
    s = (1103515245 * s + 12345) % 2 ** 31;
    return s / 2 ** 31;
  };
  const out: number[] = [];
  for (let i = 0; i < n; i += 2) {
    const u1 = Math.max(1e-12, rng());
    const u2 = rng();
    const r = Math.sqrt(-2 * Math.log(u1));
    out.push(r * Math.cos(2 * Math.PI * u2) * vol);
    if (i + 1 < n) out.push(r * Math.sin(2 * Math.PI * u2) * vol);
  }
  return out;
}

describe('GARCH_DEFAULTS', () => {
  it('uses standard equity GARCH(1,1) priors', () => {
    expect(GARCH_DEFAULTS.alpha).toBeCloseTo(0.10, 6);
    expect(GARCH_DEFAULTS.beta).toBeCloseTo(0.85, 6);
    // alpha + beta < 1 for stationarity
    expect(GARCH_DEFAULTS.alpha + GARCH_DEFAULTS.beta).toBeLessThan(1);
  });
});

describe('fitGarch11', () => {
  it('matches unconditional variance to sample variance', () => {
    const rets = makeReturns(252, 0.01);
    const params = fitGarch11(rets);
    const sampleVar = rets.reduce((s, r) => s + r * r, 0) / rets.length;
    // Unconditional variance: ω / (1 − α − β) = sample variance
    const uncondVar = params.omega / (1 - params.alpha - params.beta);
    expect(uncondVar).toBeCloseTo(sampleVar, 6);
  });

  it('returns alpha + beta < 1 (stationarity)', () => {
    const rets = makeReturns(100, 0.02);
    const params = fitGarch11(rets);
    expect(params.alpha + params.beta).toBeLessThan(1);
  });

  it('initial h0 equals sample variance', () => {
    const rets = makeReturns(100, 0.015);
    const params = fitGarch11(rets);
    const sampleVar = rets.reduce((s, r) => s + r * r, 0) / rets.length;
    expect(params.h0).toBeCloseTo(sampleVar, 6);
  });

  it('throws for empty input', () => {
    expect(() => fitGarch11([])).toThrow();
  });

  it('throws for input shorter than 5 observations', () => {
    expect(() => fitGarch11([0.01, 0.02])).toThrow();
  });
});

describe('garchStep', () => {
  it('applies the GARCH(1,1) recursion', () => {
    const params = { omega: 1e-6, alpha: 0.10, beta: 0.85, h0: 1e-4 };
    const prevH = 1e-4;
    const prevZ = 1.5;
    const expected = params.omega + params.alpha * prevZ * prevZ * prevH + params.beta * prevH;
    expect(garchStep(prevH, prevZ, params)).toBeCloseTo(expected, 12);
  });

  it('reverts toward β-only fixed point with z=0 over many steps', () => {
    // With z=0, recursion becomes h_{t+1} = ω + β·h_t (innovation drops out),
    // whose fixed point is ω/(1−β), NOT the unconditional ω/(1−α−β).
    const params = { omega: 1e-6, alpha: 0.10, beta: 0.85, h0: 1e-4 };
    const fixedPoint = params.omega / (1 - params.beta);
    let h = 5 * fixedPoint;
    for (let t = 0; t < 200; t++) h = garchStep(h, 0, params);
    expect(h).toBeCloseTo(fixedPoint, 5);
  });
});

describe('garchForecast', () => {
  it('returns an array of length horizonDays', () => {
    const params = { omega: 1e-6, alpha: 0.10, beta: 0.85, h0: 1e-4 };
    const sigmas = garchForecast(params, 30);
    expect(sigmas).toHaveLength(30);
  });

  it('all σ_t values are positive and finite', () => {
    const params = { omega: 1e-6, alpha: 0.10, beta: 0.85, h0: 1e-4 };
    const sigmas = garchForecast(params, 50);
    for (const s of sigmas) {
      expect(s).toBeGreaterThan(0);
      expect(Number.isFinite(s)).toBe(true);
    }
  });

  it('multi-step forecast (z=0 path) converges toward unconditional sigma', () => {
    const params = { omega: 1e-6, alpha: 0.10, beta: 0.85, h0: 1e-2 };
    const uncondSigma = Math.sqrt(params.omega / (1 - params.alpha - params.beta));
    const sigmas = garchForecast(params, 500);
    // After 500 steps, should be very close to unconditional σ
    expect(sigmas[sigmas.length - 1]).toBeCloseTo(uncondSigma, 4);
  });
});
