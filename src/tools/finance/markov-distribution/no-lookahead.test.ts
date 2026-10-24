import { describe, expect, it } from 'bun:test';
import { walkForward } from '../backtest/walk-forward.js';

const realRandom = Math.random;

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

async function withSeed<T>(seed: number, fn: () => Promise<T>): Promise<T> {
  Math.random = seedRng(seed);
  try {
    return await fn();
  } finally {
    Math.random = realRandom;
  }
}

function prices(n = 400, seed = 0): number[] {
  const rng = seedRng(seed);
  const out: number[] = [];
  let price = 100;
  for (let i = 0; i < n; i++) {
    price *= Math.exp(0.0005 + gauss(rng) * 0.02);
    out.push(price);
  }
  return out;
}

describe('Markov walk-forward no-look-ahead regression lock', () => {
  it('does not let far-future price corruption affect earlier forecasts or scores', async () => {
    const horizon = 7;
    const warmup = 120;
    const stride = 10;
    const clean = prices();
    const corruptFrom = 320;
    const corrupted = [...clean];
    for (let i = corruptFrom; i < corrupted.length; i++) {
      corrupted[i] *= 100;
    }

    const base = await withSeed(2024, () => walkForward({
      ticker: 'TEST',
      prices: clean,
      horizon,
      warmup,
      stride,
    }));
    const tainted = await withSeed(2024, () => walkForward({
      ticker: 'TEST',
      prices: corrupted,
      horizon,
      warmup,
      stride,
    }));

    expect(base.errors).toEqual([]);
    expect(tainted.errors).toEqual([]);

    const taintedByT = new Map(tainted.steps.map(step => [step.t, step]));
    const safeSteps = base.steps.filter(step => step.t + horizon < corruptFrom && taintedByT.has(step.t));
    expect(safeSteps.length).toBeGreaterThan(0);

    for (const b of safeSteps) {
      const t = taintedByT.get(b.t)!;
      expect(t.predictedProb).toBe(b.predictedProb);
      expect(t.predictedReturn).toBe(b.predictedReturn);
      expect(t.ciLower).toBe(b.ciLower);
      expect(t.ciUpper).toBe(b.ciUpper);
      expect(t.confidence).toBe(b.confidence);
      expect(t.actualReturn).toBe(b.actualReturn);
      expect(t.realizedPrice).toBe(b.realizedPrice);
      expect(t.actualBinary).toBe(b.actualBinary);
    }
  });
});
