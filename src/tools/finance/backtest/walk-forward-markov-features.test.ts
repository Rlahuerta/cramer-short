import { describe, expect, it } from 'bun:test';
import { computeMarkovDistribution } from '../markov-distribution.js';
import { sliceCrossAssetReturnsToOrigin, walkForward, type WalkForwardConfig } from './walk-forward.js';

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

describe('walk-forward look-ahead guards', () => {
  it('conformal calibration only uses outcomes realized by the forecast origin', async () => {
    const horizon = 5;
    const result = await withSeed(11, () => walkForward({
      ticker: 'SPY',
      prices: syntheticTrajectoryPrices(),
      horizon,
      warmup: 100,
      stride: 1,
      randomState: 11,
      enableAdaptiveConformal: true,
      conformalResetOnStructuralBreak: false,
    }));

    expect(result.steps.length).toBeGreaterThan(horizon * 2);
    for (const step of result.steps) {
      const matured = result.steps.filter((earlier) => earlier.t + horizon <= step.t).length;
      expect(step.conformalSampleCount).toBe(matured);
    }
  }, 60_000);

  it('cross-asset peer returns are cut at the forecast origin', () => {
    const peers = { QQQ: [0.1, 0.2, 0.3, 0.4], TLT: [-0.1, -0.2, -0.3, -0.4] };
    expect(sliceCrossAssetReturnsToOrigin(peers, 2)).toEqual({ QQQ: [0.1, 0.2], TLT: [-0.1, -0.2] });
    expect(peers.QQQ).toHaveLength(4);
    expect(sliceCrossAssetReturnsToOrigin(undefined, 2)).toBeUndefined();
  });
});

describe('walk-forward rolling history window', () => {
  // Same prices from index 60 on; only the first 60 differ.
  const pricesA = syntheticTrajectoryPrices();
  const pricesB = pricesA.map((p, i) => (i < 60 ? p * 1.3 : p));
  const run = (prices: number[], extra: Partial<WalkForwardConfig>) => walkForward({
    ticker: 'SPY', prices, horizon: 1, warmup: 120, stride: 10, randomState: 42, ...extra,
  });
  const probs = (steps: { predictedProb: number }[]) => steps.map((s) => s.predictedProb);

  it('forecasts each origin from at most maxHistoryPrices trailing prices, like the live tool', async () => {
    const [a, b] = await Promise.all([run(pricesA, { maxHistoryPrices: 50 }), run(pricesB, { maxHistoryPrices: 50 })]);
    expect(a.steps.length).toBeGreaterThan(0);
    expect(probs(b.steps)).toEqual(probs(a.steps));
  });

  it('still uses the whole expanding history when no cap is set', async () => {
    const [a, b] = await Promise.all([run(pricesA, {}), run(pricesB, {})]);
    expect(probs(b.steps)).not.toEqual(probs(a.steps));
  });
});

describe('walk-forward anchored replay hook', () => {
  it('forecasts each origin with the markets anchorsAt returns, observed at its reference time', async () => {
    const prices = syntheticTrajectoryPrices();
    const start = Date.parse('2025-01-01T23:59:59Z');
    const day = 86_400_000;
    const horizon = 5;
    const isoDate = (ms: number) => new Date(ms).toISOString().slice(0, 10);
    // One mature, horizon-matched, liquid market struck at each origin's price: a trusted anchor.
    const anchorsAt = (t: number) => {
      const referenceTimeMs = start + t * day;
      const strike = (Math.round(prices[t] * 100) / 100).toFixed(2);
      const endDate = isoDate(referenceTimeMs + horizon * day);
      return {
        referenceTimeMs,
        markets: [{
          question: 'Will SYN be above $' + strike + ' on ' + endDate + '?',
          probability: 0.8,
          volume: 50_000,
          createdAt: referenceTimeMs - 10 * day,
          endDate,
        }],
      };
    };
    const base = { ticker: 'SYN', prices, horizon, warmup: 120, stride: 20, randomState: 42 };

    const plain = await walkForward(base);
    const anchored = await walkForward({ ...base, anchorsAt });

    expect(plain.steps.every(s => s.trustedAnchors === 0)).toBe(true);
    expect(anchored.steps.length).toBe(plain.steps.length);
    expect(anchored.steps.every(s => s.trustedAnchors === 1)).toBe(true);
    expect(anchored.steps.some((s, i) => Math.abs(s.predictedProb - plain.steps[i].predictedProb) > 0.01)).toBe(true);
  });
});

describe('walk-forward interval experiments', () => {
  it('passes the forecast volatility multiplier and the t degrees of freedom through to the forecast distribution', async () => {
    const base = { ticker: 'BTC-USD', prices: syntheticTrajectoryPrices(), horizon: 3, warmup: 120, stride: 20, randomState: 42 };
    const [plain, wide, fat] = await Promise.all([
      walkForward(base),
      walkForward({ ...base, forecastVolMultiplier: 1.2 }),
      walkForward({ ...base, studentTNuOverride: 3 }),
    ]);
    const width = (s: { central90CiLower?: number; central90CiUpper?: number; central99CiLower?: number; central99CiUpper?: number }, level: 90 | 99) =>
      Math.log(level === 90 ? s.central90CiUpper! / s.central90CiLower! : s.central99CiUpper! / s.central99CiLower!);
    expect(plain.steps.length).toBeGreaterThan(0);
    plain.steps.forEach((s, i) => {
      expect(width(wide.steps[i], 90)).toBeGreaterThan(1.1 * width(s, 90));
      // nu 3 instead of 4 + h = 7: the variance-matched fatter tail widens the 99% interval and narrows the 90%.
      expect(width(fat.steps[i], 99)).toBeGreaterThan(width(s, 99));
      expect(width(fat.steps[i], 90)).toBeLessThan(width(s, 90));
    });
  });
});

describe('walk-forward anchor horizon experiments', () => {
  const prices = syntheticTrajectoryPrices();
  const start = Date.parse('2025-01-01T23:59:59Z');
  const day = 86_400_000;
  const horizon = 3;
  const isoDate = (ms: number) => new Date(ms).toISOString().slice(0, 10);
  // A mature, liquid crypto market struck at the origin's price that resolves 5 days after the target.
  const anchorsAt = (t: number) => {
    const referenceTimeMs = start + t * day;
    const strike = (Math.round(prices[t] * 100) / 100).toFixed(2);
    const endDate = isoDate(referenceTimeMs + (horizon + 5) * day);
    return {
      referenceTimeMs,
      markets: [{ question: 'Will BTC be above $' + strike + ' on ' + endDate + '?', probability: 0.8, volume: 50_000, createdAt: referenceTimeMs - 10 * day, endDate }],
    };
  };
  const base = { ticker: 'BTC-USD', prices, horizon, warmup: 120, stride: 20, randomState: 42 };

  it('drops, down-weights or ignores anchors that resolve away from the target', async () => {
    const [plain, anchored, dropped, lowTrust, capped] = await Promise.all([
      walkForward(base),
      walkForward({ ...base, anchorsAt }),
      walkForward({ ...base, anchorsAt, offHorizonAnchorDays: 2, offHorizonAnchorPolicy: 'drop' }),
      walkForward({ ...base, anchorsAt, offHorizonAnchorDays: 2, offHorizonAnchorPolicy: 'low-trust' }),
      walkForward({ ...base, anchorsAt, maxAnchorHorizonDays: horizon - 1 }),
    ]);
    const probs = (r: typeof plain) => r.steps.map(s => s.predictedProb);

    expect(anchored.steps.every(s => s.trustedAnchors === 1)).toBe(true); // today: trusted although 5 days off
    expect(probs(dropped)).toEqual(probs(plain));
    expect(probs(capped)).toEqual(probs(plain));
    expect(lowTrust.steps.every(s => s.trustedAnchors === 0)).toBe(true);
    expect(probs(lowTrust)).not.toEqual(probs(plain)); // still blended, at half weight
    expect(probs(lowTrust)).not.toEqual(probs(anchored));
  });

  it('rescales anchors more than the given days off the target to the horizon, still trusted', async () => {
    const [anchored, rescaled, withinTolerance] = await Promise.all([
      walkForward({ ...base, anchorsAt }),
      walkForward({ ...base, anchorsAt, anchorHorizonRescaleDays: 2 }),
      walkForward({ ...base, anchorsAt, anchorHorizonRescaleDays: 10 }),
    ]);
    const probs = (r: typeof anchored) => r.steps.map(s => s.predictedProb);
    expect(rescaled.steps.every(s => s.trustedAnchors === 1)).toBe(true);
    // The 8-day market says 0.8 above spot; read at 3 days its z grows by sqrt(8 / 3), so P(up) rises.
    rescaled.steps.forEach((s, i) => expect(s.predictedProb).toBeGreaterThan(anchored.steps[i]!.predictedProb));
    expect(probs(withinTolerance)).toEqual(probs(anchored));
  });
});
