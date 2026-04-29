import { describe, expect, it } from 'bun:test';
import { computeMarkovDistribution } from '../markov-distribution.js';
import { computeFailureDecomposition, sharpness } from './metrics.js';
import { walkForward } from './walk-forward.js';

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

function syntheticVolBurstPrices(): number[] {
  const rng = seedRng(123);
  const prices = [100];
  for (let i = 0; i < 180; i++) {
    prices.push(prices.at(-1)! * Math.exp(0.0002 + gauss(rng) * 0.006));
  }
  for (let i = 0; i < 70; i++) {
    prices.push(prices.at(-1)! * Math.exp(-0.0005 + gauss(rng) * 0.055));
  }
  return prices;
}

function comparableStepProjection(result: Awaited<ReturnType<typeof walkForward>>) {
  return result.steps.map(step => ({
    t: step.t,
    predictedProb: step.predictedProb,
    rawPredictedProb: step.rawPredictedProb,
    predictedReturn: step.predictedReturn,
    ciLower: step.ciLower,
    ciUpper: step.ciUpper,
    recommendation: step.recommendation,
    confidence: step.confidence,
  }));
}

describe('R5 walk-forward forecast wiring', () => {
  it('keeps the default path stable but lets enableGarchVol change synthetic high-vol CIs', async () => {
    const prices = syntheticVolBurstPrices();
    const baseConfig = {
      ticker: 'BTC-USD',
      prices,
      horizon: 30,
      warmup: 120,
      stride: 10,
    } as const;

    const baseline = await withSeed(77, () => walkForward(baseConfig));
    const explicitOff = await withSeed(77, () => walkForward({
      ...baseConfig,
      enableGarchVol: false,
      garchHorizonCap: 7,
      garchRegimeCeiling: { calm: 1.1, turbulent: 1.2 },
    }));
    const enabled = await withSeed(77, () => walkForward({
      ...baseConfig,
      enableGarchVol: true,
      garchHorizonCap: 7,
      garchRegimeCeiling: { calm: 1.1, turbulent: 1.2 },
    }));

    expect(baseline.errors).toHaveLength(0);
    expect(enabled.errors).toHaveLength(0);
    expect(comparableStepProjection(explicitOff)).toEqual(comparableStepProjection(baseline));
    expect(enabled.steps.some(step => step.garchVolApplied === true)).toBe(true);

    const ciChanged = enabled.steps.some((step, i) => {
      const base = baseline.steps[i];
      return Math.abs((step.ciUpper - step.ciLower) - (base.ciUpper - base.ciLower)) > 1e-6;
    });
    expect(ciChanged).toBe(true);
    expect(sharpness(enabled.steps)).not.toBeCloseTo(sharpness(baseline.steps), 10);
  });

  it('records transition-entropy metadata and exposes an entropy CI backtest slice', async () => {
    const prices = syntheticVolBurstPrices();
    const result = await withSeed(91, () => walkForward({
      ticker: 'BTC-USD',
      prices,
      horizon: 7,
      warmup: 80,
      stride: 5,
      enableEntropyCiModulation: true,
      entropyWindowSize: 5,
      entropyKappa: 0.5,
    }));

    expect(result.errors).toHaveLength(0);
    expect(result.steps.length).toBeGreaterThan(5);
    expect(result.steps.every(step => typeof step.transitionEntropyNorm === 'number')).toBe(true);
    expect(result.steps.some(step => step.transitionEntropyZ !== null && step.transitionEntropyZ !== undefined)).toBe(true);
    expect(result.steps.some(step => step.entropyCiScale !== undefined && Math.abs(step.entropyCiScale - 1) > 1e-6)).toBe(true);

    const entropySlice = computeFailureDecomposition(result.steps).slices.find(s => s.key === 'entropyCiScale');
    expect(entropySlice).toBeDefined();
    expect(entropySlice!.rows.map(r => r.label)).toEqual(['tightened', 'neutral', 'widened']);
  });
});

describe('R5 longshot shrinkage RND wiring', () => {
  it('applies longshot shrinkage after Q-to-P conversion when enabled', async () => {
    const prices = syntheticVolBurstPrices();
    const currentPrice = 100;
    const now = Date.UTC(2026, 3, 29);
    const markets = [
      { question: 'Will Bitcoin be above $95 on May 13?', probability: 0.99, volume: 100_000, createdAt: now - 5 * 86_400_000, endDate: new Date(now + 14 * 86_400_000).toISOString() },
      { question: 'Will Bitcoin be above $100 on May 13?', probability: 0.50, volume: 100_000, createdAt: now - 5 * 86_400_000, endDate: new Date(now + 14 * 86_400_000).toISOString() },
      { question: 'Will Bitcoin be above $105 on May 13?', probability: 0.01, volume: 100_000, createdAt: now - 5 * 86_400_000, endDate: new Date(now + 14 * 86_400_000).toISOString() },
    ];

    const disabled = await withSeed(33, () => computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: markets,
      referenceTimeMs: now,
    }));
    const enabled = await withSeed(33, () => computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: markets,
      referenceTimeMs: now,
      enableLongshotShrinkage: true,
    }));

    expect(disabled.metadata.rndIntegration?.longshotShrinkageApplied).toBeUndefined();
    expect(enabled.metadata.rndIntegration?.longshotShrinkageApplied).toBe(true);
    expect(enabled.metadata.rndIntegration?.longshotShrinkageCount).toBeGreaterThanOrEqual(1);
    expect(enabled.metadata.rndIntegration?.maxLongshotTailDistance).toBeGreaterThan(0.45);
  });
});
