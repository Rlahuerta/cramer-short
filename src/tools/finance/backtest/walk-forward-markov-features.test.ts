import { describe, expect, it } from 'bun:test';
import { computeMarkovDistribution } from '../markov-distribution.js';
import { walkForward, type WalkForwardConfig } from './walk-forward.js';

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

function syntheticMetaRegimePrices(): number[] {
  const rng = seedRng(987);
  const prices = [100];
  for (let cycle = 0; cycle < 4; cycle++) {
    for (let i = 0; i < 25; i++) {
      prices.push(prices.at(-1)! * Math.exp(0.0001 + gauss(rng) * 0.003));
    }
    for (let i = 0; i < 25; i++) {
      const drift = i % 2 === 0 ? 0.018 : -0.016;
      prices.push(prices.at(-1)! * Math.exp(drift + gauss(rng) * 0.012));
    }
  }
  return prices;
}

function syntheticTrajectoryPrices(): number[] {
  const rng = seedRng(246);
  const prices = [100];
  for (let i = 0; i < 160; i++) {
    const drift = i % 3 === 0 ? 0.002 : -0.0008;
    prices.push(prices.at(-1)! * Math.exp(drift + gauss(rng) * 0.01));
  }
  return prices;
}

describe('walk-forward Markov feature flags', () => {
  it('randomState makes core forecast trajectories reproducible', async () => {
    const historicalPrices = syntheticTrajectoryPrices();
    const currentPrice = historicalPrices.at(-1)!;

    const run = (randomState: number) => computeMarkovDistribution({
      ticker: 'SPY',
      currentPrice,
      historicalPrices,
      horizon: 7,
      polymarketMarkets: [],
      trajectory: true,
      trajectoryDays: 7,
      randomState,
    });

    const first = await run(1234);
    const second = await run(1234);
    const different = await run(5678);

    expect(first.trajectory).toEqual(second.trajectory);
    expect(first.trajectory).not.toEqual(different.trajectory);
  });

  it('meta-regime conditional transitions run through walkForward and can change forecasts', async () => {
    const prices = syntheticMetaRegimePrices();
    const baseConfig: WalkForwardConfig = {
      ticker: 'BTC-USD',
      prices,
      horizon: 5,
      warmup: 120,
      stride: 20,
      btcBreakDivergenceThreshold: 1.0,
    };

    const baseline = await withSeed(73, () => walkForward(baseConfig));
    const enabled = await withSeed(73, () => walkForward({
      ...baseConfig,
      useMetaRegime: true,
      useConditionalTransitions: true,
    }));

    expect(enabled.errors).toHaveLength(0);
    expect(enabled.steps.length).toBeGreaterThan(0);
    expect(enabled.steps).not.toEqual(baseline.steps);
  });

  it('all new flags off is identical to the default walkForward path', async () => {
    const prices = syntheticMetaRegimePrices();
    const currentPrice = prices.at(-1)!;
    const baseConfig: WalkForwardConfig = {
      ticker: 'BTC-USD',
      prices,
      horizon: 5,
      warmup: 120,
      stride: 20,
      btcBreakDivergenceThreshold: 1.0,
    };

    const implicitDefault = await withSeed(91, () => walkForward(baseConfig));
    const explicitOff = await withSeed(91, () => walkForward({
      ...baseConfig,
      useMetaRegime: false,
      useConditionalTransitions: false,
      usePathIntegratedDrift: false,
    }));

    expect(explicitOff).toEqual(implicitDefault);

    const implicitCore = await withSeed(92, () => computeMarkovDistribution({
      ticker: 'BTC-USD',
      currentPrice,
      historicalPrices: prices,
      horizon: 5,
      polymarketMarkets: [],
    }));
    const explicitOffCore = await withSeed(92, () => computeMarkovDistribution({
      ticker: 'BTC-USD',
      currentPrice,
      historicalPrices: prices,
      horizon: 5,
      polymarketMarkets: [],
      useMetaRegime: false,
      useConditionalTransitions: false,
      usePathIntegratedDrift: false,
    }));

    expect(explicitOffCore).toEqual(implicitCore);
  });
});
