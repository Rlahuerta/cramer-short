import { describe, expect, it } from 'bun:test';
import { computeMarkovDistribution } from '../markov-distribution.js';
import {
  computeFailureDecomposition,
  selectiveDirectionalAccuracy,
  sharpness,
  type BacktestStep,
} from './metrics.js';
import {
  walkForward,
  type WalkForwardConfig,
  type WalkForwardResult,
} from './walk-forward.js';

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

function syntheticCalmPrices(): number[] {
  const rng = seedRng(321);
  const prices = [100];
  for (let i = 0; i < 250; i++) {
    prices.push(prices.at(-1)! * Math.exp(0.0003 + gauss(rng) * 0.004));
  }
  return prices;
}

function syntheticChoppyPrices(): number[] {
  const rng = seedRng(777);
  const prices = [100];
  for (let i = 0; i < 250; i++) {
    const drift = i % 2 === 0 ? 0.0025 : -0.0023;
    prices.push(prices.at(-1)! * Math.exp(drift + gauss(rng) * 0.012));
  }
  return prices;
}

interface AdaptiveConformalWalkForwardOptions {
  enableAdaptiveConformal?: boolean;
  conformalAlpha?: number;
  conformalBreakSensitivity?: number;
  conformalFastLearningRate?: number;
  conformalCooloffWindow?: number;
}

interface AdaptiveConformalBacktestDiagnostics {
  conformalApplied?: boolean;
  conformalRadius?: number;
  conformalCoverageEstimate?: number;
  conformalMode?: 'normal' | 'break';
}

type ConformalBacktestStep = BacktestStep & AdaptiveConformalBacktestDiagnostics;

const adaptiveConformalConfig = {
  enableAdaptiveConformal: true,
  conformalAlpha: 0.1,
  conformalBreakSensitivity: 1.5,
  conformalFastLearningRate: 0.2,
  conformalCooloffWindow: 20,
} satisfies AdaptiveConformalWalkForwardOptions;

const adaptiveConformalDisabledConfig = {
  enableAdaptiveConformal: false,
  conformalAlpha: 0.1,
  conformalBreakSensitivity: 1.5,
  conformalFastLearningRate: 0.2,
  conformalCooloffWindow: 20,
} satisfies AdaptiveConformalWalkForwardOptions;

function withAdaptiveConformalConfig(
  base: WalkForwardConfig,
  extra: AdaptiveConformalWalkForwardOptions,
): WalkForwardConfig & AdaptiveConformalWalkForwardOptions {
  return { ...base, ...extra };
}

function conformalStepView(result: WalkForwardResult): ConformalBacktestStep[] {
  return result.steps as ConformalBacktestStep[];
}

function conformalDiagnosticsProjection(result: WalkForwardResult) {
  return conformalStepView(result).map(step => ({
    t: step.t,
    conformalApplied: step.conformalApplied,
    conformalRadius: step.conformalRadius,
    conformalCoverageEstimate: step.conformalCoverageEstimate,
    conformalMode: step.conformalMode,
  }));
}

function comparableStepProjection(result: WalkForwardResult) {
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

function intervalWidth(step: Pick<BacktestStep, 'ciLower' | 'ciUpper'>): number {
  return step.ciUpper - step.ciLower;
}

function coveredCount(steps: readonly BacktestStep[]): number {
  return steps.filter(step => step.realizedPrice >= step.ciLower && step.realizedPrice <= step.ciUpper).length;
}

function medianIntervalWidth(steps: readonly Pick<BacktestStep, 'ciLower' | 'ciUpper'>[]): number {
  const widths = steps.map(intervalWidth).sort((a, b) => a - b);
  const mid = Math.floor(widths.length / 2);
  return widths.length % 2 === 0
    ? (widths[mid - 1] + widths[mid]) / 2
    : widths[mid];
}

function shockSlice(result: WalkForwardResult): ConformalBacktestStep[] {
  return conformalStepView(result).filter(step => step.t >= 180);
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

describe('R6 adaptive conformal walk-forward wiring', () => {
  it('keeps adaptive conformal disabled by default and identical to an explicit false flag', async () => {
    const prices = syntheticVolBurstPrices();
    const baseConfig = {
      ticker: 'BTC-USD',
      prices,
      horizon: 14,
      warmup: 120,
      stride: 5,
    } as const;

    const baseline = await withSeed(141, () => walkForward(baseConfig));
    const explicitlyDisabled = await withSeed(141, () => walkForward(
      withAdaptiveConformalConfig(baseConfig, adaptiveConformalDisabledConfig),
    ));

    expect(baseline.errors).toHaveLength(0);
    expect(explicitlyDisabled.errors).toHaveLength(0);
    expect(comparableStepProjection(explicitlyDisabled)).toEqual(comparableStepProjection(baseline));
    expect(conformalDiagnosticsProjection(explicitlyDisabled)).toEqual(conformalDiagnosticsProjection(baseline));
  });

  it('records adaptive conformal break diagnostics and improves shock-slice coverage on a synthetic volatility-burst series', async () => {
    const prices = syntheticVolBurstPrices();
    const baseConfig = {
      ticker: 'BTC-USD',
      prices,
      horizon: 14,
      warmup: 120,
      stride: 5,
    } as const;

    const baseline = await withSeed(141, () => walkForward(baseConfig));
    const adaptive = await withSeed(141, () => walkForward(
      withAdaptiveConformalConfig(baseConfig, adaptiveConformalConfig),
    ));
    const baselineShockSlice = shockSlice(baseline);
    const adaptiveShockSlice = shockSlice(adaptive);

    expect(baseline.errors).toHaveLength(0);
    expect(adaptive.errors).toHaveLength(0);
    expect(adaptiveShockSlice.length).toBeGreaterThan(0);
    expect(adaptiveShockSlice.every(step => step.conformalApplied === true)).toBe(true);
    expect(adaptiveShockSlice.some(step => step.conformalMode === 'break')).toBe(true);
    expect(adaptiveShockSlice.every(step => typeof step.conformalRadius === 'number')).toBe(true);
    expect(adaptiveShockSlice.every(step => typeof step.conformalCoverageEstimate === 'number')).toBe(true);
    expect(coveredCount(adaptiveShockSlice)).toBeGreaterThan(coveredCount(baselineShockSlice));
    expect(medianIntervalWidth(adaptiveShockSlice)).toBeGreaterThan(medianIntervalWidth(baselineShockSlice));
  });

  it('does not materially degrade sharpness on a calm synthetic series', async () => {
    const prices = syntheticCalmPrices();
    const baseConfig = {
      ticker: 'BTC-USD',
      prices,
      horizon: 14,
      warmup: 120,
      stride: 5,
    } as const;

    const baseline = await withSeed(211, () => walkForward(baseConfig));
    const adaptive = await withSeed(211, () => walkForward(
      withAdaptiveConformalConfig(baseConfig, adaptiveConformalConfig),
    ));

    expect(baseline.errors).toHaveLength(0);
    expect(adaptive.errors).toHaveLength(0);
    expect(conformalStepView(adaptive).every(step => step.conformalApplied === true)).toBe(true);
    expect(conformalStepView(adaptive).every(step => step.conformalMode === 'normal' || step.conformalMode === 'break')).toBe(true);
    expect(sharpness(adaptive.steps)).toBeLessThanOrEqual(sharpness(baseline.steps) * 1.05);
  });

  it('restores non-zero selective coverage at 0.45 on a calm BTC synthetic series with rebalanced confidence', async () => {
    const prices = syntheticCalmPrices();
    const baseConfig = {
      ticker: 'BTC-USD',
      prices,
      horizon: 7,
      warmup: 80,
      stride: 5,
    } as const;

    const legacy = await withSeed(141, () => walkForward({
      ...baseConfig,
      predictionConfidenceMode: 'legacy',
    }));
    const rebalanced = await withSeed(141, () => walkForward({
      ...baseConfig,
      predictionConfidenceMode: 'rebalanced',
    }));

    expect(legacy.errors).toHaveLength(0);
    expect(rebalanced.errors).toHaveLength(0);

    const legacySelective = selectiveDirectionalAccuracy(legacy.steps, 0.45);
    const rebalancedSelective = selectiveDirectionalAccuracy(rebalanced.steps, 0.45);

    expect(legacySelective.coverage).toBe(0);
    expect(rebalancedSelective.coverage).toBeGreaterThan(0);
    expect(rebalancedSelective.selected).toBeGreaterThan(legacySelective.selected);
  });
});

describe('soft regime weighting walk-forward wiring', () => {
  it('keeps soft regime weighting disabled by default and identical to an explicit false flag', async () => {
    const prices = syntheticChoppyPrices();
    const baseConfig = {
      ticker: 'BTC-USD',
      prices,
      horizon: 7,
      warmup: 120,
      stride: 5,
      predictionConfidenceMode: 'rebalanced' as const,
    };

    const baseline = await withSeed(501, () => walkForward(baseConfig));
    const explicitlyDisabled = await withSeed(501, () => walkForward({
      ...baseConfig,
      enableSoftRegimeWeighting: false,
    }));

    expect(baseline.errors).toHaveLength(0);
    expect(explicitlyDisabled.errors).toHaveLength(0);
    expect(comparableStepProjection(explicitlyDisabled)).toEqual(comparableStepProjection(baseline));
  });

  it('widens uncertainty and lowers average confidence on a choppy synthetic series when enabled', async () => {
    const prices = syntheticChoppyPrices();
    const baseConfig = {
      ticker: 'BTC-USD',
      prices,
      horizon: 7,
      warmup: 120,
      stride: 5,
      predictionConfidenceMode: 'rebalanced' as const,
    };

    const baseline = await withSeed(502, () => walkForward({
      ...baseConfig,
      enableSoftRegimeWeighting: false,
    }));
    const enabled = await withSeed(502, () => walkForward({
      ...baseConfig,
      enableSoftRegimeWeighting: true,
    }));

    expect(baseline.errors).toHaveLength(0);
    expect(enabled.errors).toHaveLength(0);
    expect(sharpness(enabled.steps)).toBeGreaterThan(sharpness(baseline.steps));

    const avgConfidence = (steps: WalkForwardResult['steps']) =>
      steps.reduce((sum, step) => sum + step.confidence, 0) / steps.length;

    expect(avgConfidence(enabled.steps)).toBeLessThan(avgConfidence(baseline.steps));
  });
});
