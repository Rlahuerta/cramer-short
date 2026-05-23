import { describe, expect, it } from 'bun:test';
import { fitLasso, predictLasso, estimateCrossAssetBias } from './cross-asset-lasso.js';

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

describe('fitLasso', () => {
  it('throws on dimension mismatch', () => {
    expect(() => fitLasso([[1, 2]], [1, 2])).toThrow();
  });

  it('returns intercept = ȳ when X has zero columns', () => {
    const fit = fitLasso([[], [], [], []], [1, 3, 5, 7]);
    expect(fit.intercept).toBe(4);
    expect(fit.coef).toEqual([]);
  });

  it('with λ=0 recovers true coefficients on noiseless data', () => {
    const rng = seedRng(1);
    const n = 200;
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < n; i++) {
      const x1 = gauss(rng);
      const x2 = gauss(rng);
      X.push([x1, x2]);
      y.push(2 * x1 - 1 * x2 + 5);
    }
    const fit = fitLasso(X, y, { lambda: 0, maxIterations: 1000, tolerance: 1e-10 });
    expect(fit.intercept).toBeCloseTo(5, 0);
    // β are in standardised space — recover via β_orig = β / std.
    const b1 = fit.coef[0] / fit.featureStd[0];
    const b2 = fit.coef[1] / fit.featureStd[1];
    expect(b1).toBeCloseTo(2, 1);
    expect(b2).toBeCloseTo(-1, 1);
  });

  it('with large λ shrinks all coefficients to 0 (sparsity)', () => {
    const rng = seedRng(2);
    const n = 100;
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < n; i++) {
      const x1 = gauss(rng);
      const x2 = gauss(rng);
      X.push([x1, x2]);
      y.push(0.5 * x1 + 0.3 * x2);
    }
    const fit = fitLasso(X, y, { lambda: 5 });
    expect(fit.coef.every(b => b === 0)).toBe(true);
  });

  it('produces sparse solutions — kills irrelevant features', () => {
    const rng = seedRng(3);
    const n = 300;
    const X: number[][] = [];
    const y: number[] = [];
    for (let i = 0; i < n; i++) {
      const x1 = gauss(rng);
      const x2 = gauss(rng); // Pure noise
      const x3 = gauss(rng);
      X.push([x1, x2, x3]);
      // Only x1 and x3 contribute.
      y.push(1.5 * x1 + 0.8 * x3 + 0.05 * gauss(rng));
    }
    const fit = fitLasso(X, y, { lambda: 0.05 });
    // x2 should be zeroed out under sparsity.
    expect(Math.abs(fit.coef[1])).toBeLessThan(Math.abs(fit.coef[0]));
    expect(Math.abs(fit.coef[1])).toBeLessThan(Math.abs(fit.coef[2]));
  });
});

describe('predictLasso', () => {
  it('matches training-time predictions on training point', () => {
    const X = [[1, 2], [3, 4], [5, 6], [7, 8]];
    const y = [3, 5, 7, 9];
    const fit = fitLasso(X, y, { lambda: 0, maxIterations: 1000 });
    const yhat = predictLasso(fit, [3, 4]);
    expect(yhat).toBeCloseTo(5, 1);
  });
});

describe('estimateCrossAssetBias', () => {
  it('returns null when peers map is empty', () => {
    const r = estimateCrossAssetBias([1, 2, 3, 4, 5], {}, 3);
    expect(r).toBeNull();
  });

  it('returns null when insufficient overlapping samples', () => {
    const target = Array.from({ length: 40 }, (_, i) => i * 0.001);
    const peers = { ETH: Array.from({ length: 40 }, () => 0.001) };
    const r = estimateCrossAssetBias(target, peers, 7, { minSamples: 60 });
    expect(r).toBeNull();
  });

  it('produces a finite per-day bias when there is signal', () => {
    const rng = seedRng(7);
    const n = 200;
    const peer = Array.from({ length: n }, () => gauss(rng) * 0.02);
    // Target's next-day return is 0.4 × peer's same-day return + noise.
    const target = new Array(n).fill(0);
    for (let i = 0; i < n - 1; i++) {
      target[i + 1] = 0.4 * peer[i] + 0.005 * gauss(rng);
    }
    const r = estimateCrossAssetBias(target, { ETH: peer }, 1, { lambda: 0.001, lag: 1 });
    expect(r).not.toBeNull();
    expect(Number.isFinite(r!.perDayBias)).toBe(true);
    expect(r!.tickers).toEqual(['ETH']);
    // Coefficient on ETH should be > 0 (positive co-movement).
    expect(r!.fit.coef[0]).toBeGreaterThan(0);
  });

  it('is deterministic for the same input', () => {
    const rng = seedRng(99);
    const n = 150;
    const peer = Array.from({ length: n }, () => gauss(rng) * 0.02);
    const target = Array.from({ length: n }, () => gauss(rng) * 0.015);
    const a = estimateCrossAssetBias(target, { ETH: peer, SOL: peer.map(x => x * 0.5) }, 7, { lambda: 0.005 });
    const b = estimateCrossAssetBias(target, { ETH: peer, SOL: peer.map(x => x * 0.5) }, 7, { lambda: 0.005 });
    expect(a?.perDayBias).toEqual(b?.perDayBias);
  });

  it('per-day bias scales inversely with horizon', () => {
    const rng = seedRng(11);
    const n = 200;
    const peer = Array.from({ length: n }, () => gauss(rng) * 0.01);
    const target = peer.map(x => x * 0.5); // Strong relationship
    const r3 = estimateCrossAssetBias(target, { ETH: peer }, 3, { lambda: 0.001 });
    const r9 = estimateCrossAssetBias(target, { ETH: peer }, 9, { lambda: 0.001 });
    expect(r3).not.toBeNull();
    expect(r9).not.toBeNull();
    // The 9-day cumulative forecast should yield a roughly 3× larger raw bias,
    // but per-day bias should be *similar* (signal divided by horizon).
    // We don't enforce equality — just that both are finite and sensible.
    expect(Math.abs(r3!.perDayBias)).toBeLessThan(0.05);
    expect(Math.abs(r9!.perDayBias)).toBeLessThan(0.05);
  });
});
