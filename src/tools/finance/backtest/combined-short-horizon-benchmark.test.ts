import { describe, beforeAll, expect } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { integrationIt } from '@/utils/test-guards.js';
import type { ArbiterReplayBundle } from '../arbiter-replay.js';
import {
  formatCombinedShortHorizonBenchmarkReport,
  runCombinedShortHorizonBenchmark,
  type CombinedShortHorizonBenchmarkReport,
  type CombinedShortHorizonBenchmarkSlice,
} from './combined-short-horizon-benchmark.js';
import { walkForward } from './walk-forward.js';

const TIMEOUT = 480_000;
const GENERATED_AT = '2026-05-05T00:00:00.000Z';

interface FixtureData {
  tickers: Record<string, {
    type: string;
    closes: number[];
    dates: string[];
    count: number;
    synthetic?: boolean;
  }>;
}

let fixture: FixtureData;

const BTC_7D_ABSTAIN_REASON =
  'Evaluator abstained on every labeled 7d replay bundle, so directional accuracy is undefined.';
const BTC_14D_ABSTAIN_REASON =
  'Evaluator abstained on every labeled 14d replay bundle, so directional accuracy is undefined.';

// Deterministic lock for the current BTC replay + price fixtures, not a claim of broad live accuracy.
// Refreshed after the zero-anchor cryptoModelOnly break fix enabled the more
// conservative PR3F blend during structural breaks, which raises HOLD counts.
const BTC_FIXTURE_BASELINE = {
  '1d': {
    markovOnly: {
      observationCount: 96,
      tradedCount: null,
      abstainCount: 20,
      directionalAccuracy: 0.59375,
      brierScore: 0.2550127431393302,
      structuralBreakCount: 72,
      ready: true,
      pendingReasons: [],
    },
    polymarketOnly: {
      observationCount: 1,
      tradedCount: 1,
      abstainCount: 0,
      directionalAccuracy: 1,
      brierScore: 0.04839999999999999,
      structuralBreakCount: null,
      ready: true,
      pendingReasons: [],
    },
    combinedArbitrated: {
      observationCount: 1,
      tradedCount: 1,
      abstainCount: 0,
      directionalAccuracy: 1,
      brierScore: 0.04839999999999999,
      structuralBreakCount: 1,
      ready: true,
      pendingReasons: [],
    },
  },
  '2d': {
    markovOnly: {
      observationCount: 96,
      tradedCount: null,
      abstainCount: 16,
      directionalAccuracy: 0.5,
      brierScore: 0.25477609958328445,
      structuralBreakCount: 50,
      ready: true,
      pendingReasons: [],
    },
    polymarketOnly: {
      observationCount: 1,
      tradedCount: 1,
      abstainCount: 0,
      directionalAccuracy: 1,
      brierScore: 0.04839999999999999,
      structuralBreakCount: null,
      ready: true,
      pendingReasons: [],
    },
    combinedArbitrated: {
      observationCount: 1,
      tradedCount: 1,
      abstainCount: 0,
      directionalAccuracy: 1,
      brierScore: 0.04839999999999999,
      structuralBreakCount: 0,
      ready: true,
      pendingReasons: [],
    },
  },
  '3d': {
    markovOnly: {
      observationCount: 96,
      tradedCount: null,
      abstainCount: 13,
      directionalAccuracy: 0.5416666666666666,
      brierScore: 0.2637207940101452,
      structuralBreakCount: 50,
      ready: true,
      pendingReasons: [],
    },
    polymarketOnly: {
      observationCount: 1,
      tradedCount: 1,
      abstainCount: 0,
      directionalAccuracy: 1,
      brierScore: 0.10239999999999996,
      structuralBreakCount: null,
      ready: true,
      pendingReasons: [],
    },
    combinedArbitrated: {
      observationCount: 1,
      tradedCount: 1,
      abstainCount: 0,
      directionalAccuracy: 1,
      brierScore: 0.04839999999999999,
      structuralBreakCount: 1,
      ready: true,
      pendingReasons: [],
    },
  },
  '7d': {
    markovOnly: {
      observationCount: 95,
      tradedCount: null,
      abstainCount: 13,
      directionalAccuracy: 0.5157894736842106,
      brierScore: 0.256354982659173,
      structuralBreakCount: 49,
      ready: true,
      pendingReasons: [],
    },
    polymarketOnly: {
      observationCount: 1,
      tradedCount: 1,
      abstainCount: 0,
      directionalAccuracy: 1,
      brierScore: 0.04839999999999999,
      structuralBreakCount: null,
      ready: true,
      pendingReasons: [],
    },
    combinedArbitrated: {
      observationCount: 1,
      tradedCount: 0,
      abstainCount: 1,
      directionalAccuracy: null,
      brierScore: 0.25,
      structuralBreakCount: 0,
      ready: false,
      pendingReasons: [BTC_7D_ABSTAIN_REASON],
    },
  },
  '14d': {
    markovOnly: {
      observationCount: 93,
      tradedCount: null,
      abstainCount: 9,
      directionalAccuracy: 0.5161290322580645,
      brierScore: 0.26736459290288817,
      structuralBreakCount: 85,
      ready: true,
      pendingReasons: [],
    },
    polymarketOnly: {
      observationCount: 1,
      tradedCount: 0,
      abstainCount: 1,
      directionalAccuracy: null,
      brierScore: 0.25,
      structuralBreakCount: null,
      ready: false,
      pendingReasons: [BTC_14D_ABSTAIN_REASON],
    },
    combinedArbitrated: {
      observationCount: 1,
      tradedCount: 0,
      abstainCount: 1,
      directionalAccuracy: null,
      brierScore: 0.25,
      structuralBreakCount: 1,
      ready: false,
      pendingReasons: [BTC_14D_ABSTAIN_REASON],
    },
  },
};

function summarizeSliceBaseline(slice: CombinedShortHorizonBenchmarkSlice) {
  return {
    observationCount: slice.observationCount,
    tradedCount: slice.tradedCount,
    abstainCount: slice.abstainCount,
    directionalAccuracy: slice.directionalAccuracy,
    brierScore: slice.brierScore,
    structuralBreakCount: slice.structuralBreakCount,
    ready: slice.ready,
    pendingReasons: slice.pendingReasons,
  };
}

function summarizeBtcFixtureBaseline(report: CombinedShortHorizonBenchmarkReport) {
  return {
    '1d': {
      markovOnly: summarizeSliceBaseline(report.horizons['1d'].markovOnly),
      polymarketOnly: summarizeSliceBaseline(report.horizons['1d'].polymarketOnly),
      combinedArbitrated: summarizeSliceBaseline(report.horizons['1d'].combinedArbitrated),
    },
    '2d': {
      markovOnly: summarizeSliceBaseline(report.horizons['2d'].markovOnly),
      polymarketOnly: summarizeSliceBaseline(report.horizons['2d'].polymarketOnly),
      combinedArbitrated: summarizeSliceBaseline(report.horizons['2d'].combinedArbitrated),
    },
    '3d': {
      markovOnly: summarizeSliceBaseline(report.horizons['3d'].markovOnly),
      polymarketOnly: summarizeSliceBaseline(report.horizons['3d'].polymarketOnly),
      combinedArbitrated: summarizeSliceBaseline(report.horizons['3d'].combinedArbitrated),
    },
    '7d': {
      markovOnly: summarizeSliceBaseline(report.horizons['7d'].markovOnly),
      polymarketOnly: summarizeSliceBaseline(report.horizons['7d'].polymarketOnly),
      combinedArbitrated: summarizeSliceBaseline(report.horizons['7d'].combinedArbitrated),
    },
    '14d': {
      markovOnly: summarizeSliceBaseline(report.horizons['14d'].markovOnly),
      polymarketOnly: summarizeSliceBaseline(report.horizons['14d'].polymarketOnly),
      combinedArbitrated: summarizeSliceBaseline(report.horizons['14d'].combinedArbitrated),
    },
  };
}

function makeReplayBundle(params: {
  ticker: 'GLD' | 'BTC';
  horizonDays: 1 | 2 | 3 | 7 | 14;
  actualBinary: 0 | 1;
  currentPrice: number;
  realizedPrice: number;
  markovForecastReturn: number;
  markovConfidence: number;
  structuralBreak?: boolean;
  flatProbability?: number;
  polymarketForecastReturn?: number;
  polymarketConfidence?: number;
  qualityScore?: number;
}): ArbiterReplayBundle {
  const question = params.ticker === 'BTC'
    ? `Will Bitcoin be above $${Math.round(params.currentPrice)} in ${params.horizonDays} day${params.horizonDays === 1 ? '' : 's'}?`
    : `Will GLD be above $${Math.round(params.currentPrice)} in ${params.horizonDays} day${params.horizonDays === 1 ? '' : 's'}?`;

  return {
    capturedAt: `2026-05-${String(params.horizonDays).padStart(2, '0')}T00:00:00.000Z`,
    ticker: params.ticker,
    horizonDays: params.horizonDays,
    currentPrice: params.currentPrice,
    leverage: 1,
    markov: {
      forecast_return: params.markovForecastReturn,
      p_up: params.markovForecastReturn > 0 ? 0.62 : 0.38,
      confidence: params.markovConfidence,
      structural_break: params.structuralBreak ?? false,
      flat_probability: params.flatProbability ?? 0.25,
      ci_low: params.currentPrice * 0.96,
      ci_high: params.currentPrice * 1.04,
    },
    ...(params.polymarketForecastReturn !== undefined
      ? {
          polymarket: {
            querySet: [`${params.ticker.toLowerCase()} price`],
            selectedMarketIds: [`${params.ticker}-${params.horizonDays}d`],
            selectedMarkets: [
              {
                marketId: `${params.ticker}-${params.horizonDays}d`,
                assetId: `${params.ticker}-${params.horizonDays}d-yes`,
                question,
                probability: params.polymarketForecastReturn > 0 ? 0.64 : 0.36,
                volume24h: 250_000,
                endDate: '2026-05-20T00:00:00.000Z',
                semantics: 'terminal',
                extractedPriceLevels: [Math.round(params.currentPrice)],
              },
            ],
            forecastReturn: params.polymarketForecastReturn,
            confidence: params.polymarketConfidence,
            qualityScore: params.qualityScore ?? 80,
            qualityGrade: 'A',
            warnings: [],
          },
        }
      : {}),
    ...(params.polymarketForecastReturn !== undefined
      ? {
          whale: {
            source: 'whale-alert',
            direction: params.actualBinary === 1 ? 'long' : 'short',
            confidence: 0.68,
            summary: 'Fixture whale support aligns with the tradeable direction.',
            observationWindowStart: '2026-05-01T00:00:00.000Z',
            observationWindowEnd: '2026-05-01T06:00:00.000Z',
            txCount: 2,
            notionalUsd: 8_000_000,
            txHashes: ['0xabc'],
          },
        }
      : {}),
    warnings: [],
    labels: {
      forecast: {
        realizedPrice: params.realizedPrice,
        realizedReturn: (params.realizedPrice - params.currentPrice) / params.currentPrice,
        actualBinary: params.actualBinary,
        labeledAt: '2026-05-20T12:00:00.000Z',
      },
      semantic: params.polymarketForecastReturn !== undefined
        ? [
            {
              marketId: `${params.ticker}-${params.horizonDays}d`,
              semantics: 'terminal',
              outcome: params.actualBinary === 1 ? 'yes' : 'no',
              labeledAt: '2026-05-20T12:00:00.000Z',
            },
          ]
        : [],
    },
  };
}

function makeGoldBundles(): ArbiterReplayBundle[] {
  return [
    makeReplayBundle({
      ticker: 'GLD',
      horizonDays: 1,
      actualBinary: 1,
      currentPrice: 205,
      realizedPrice: 207,
      markovForecastReturn: 0.01,
      markovConfidence: 0.82,
      polymarketForecastReturn: 0.012,
      polymarketConfidence: 0.76,
    }),
    makeReplayBundle({
      ticker: 'GLD',
      horizonDays: 2,
      actualBinary: 0,
      currentPrice: 206,
      realizedPrice: 202,
      markovForecastReturn: -0.009,
      markovConfidence: 0.79,
      polymarketForecastReturn: -0.01,
      polymarketConfidence: 0.74,
    }),
    makeReplayBundle({
      ticker: 'GLD',
      horizonDays: 3,
      actualBinary: 1,
      currentPrice: 204,
      realizedPrice: 208,
      markovForecastReturn: 0.007,
      markovConfidence: 0.72,
      structuralBreak: true,
      polymarketForecastReturn: 0.008,
      polymarketConfidence: 0.73,
    }),
    makeReplayBundle({
      ticker: 'GLD',
      horizonDays: 7,
      actualBinary: 0,
      currentPrice: 203,
      realizedPrice: 198,
      markovForecastReturn: 0.011,
      markovConfidence: 0.58,
      flatProbability: 0.42,
      polymarketForecastReturn: -0.009,
      polymarketConfidence: 0.77,
      qualityScore: 83,
    }),
    makeReplayBundle({
      ticker: 'GLD',
      horizonDays: 14,
      actualBinary: 1,
      currentPrice: 202,
      realizedPrice: 205,
      markovForecastReturn: 0.001,
      markovConfidence: 0.17,
      structuralBreak: true,
      flatProbability: 0.86,
    }),
  ];
}

function makeBtcBundles(): ArbiterReplayBundle[] {
  return [
    makeReplayBundle({
      ticker: 'BTC',
      horizonDays: 1,
      actualBinary: 1,
      currentPrice: 75_500,
      realizedPrice: 76_300,
      markovForecastReturn: 0.008,
      markovConfidence: 0.77,
      structuralBreak: true,
      polymarketForecastReturn: 0.01,
      polymarketConfidence: 0.75,
    }),
    makeReplayBundle({
      ticker: 'BTC',
      horizonDays: 2,
      actualBinary: 0,
      currentPrice: 76_000,
      realizedPrice: 74_900,
      markovForecastReturn: -0.011,
      markovConfidence: 0.8,
      polymarketForecastReturn: -0.013,
      polymarketConfidence: 0.78,
    }),
    makeReplayBundle({
      ticker: 'BTC',
      horizonDays: 3,
      actualBinary: 1,
      currentPrice: 74_800,
      realizedPrice: 76_100,
      markovForecastReturn: 0.012,
      markovConfidence: 0.74,
      structuralBreak: true,
      polymarketForecastReturn: 0.009,
      polymarketConfidence: 0.74,
    }),
    makeReplayBundle({
      ticker: 'BTC',
      horizonDays: 7,
      actualBinary: 0,
      currentPrice: 77_000,
      realizedPrice: 73_500,
      markovForecastReturn: 0.012,
      markovConfidence: 0.58,
      flatProbability: 0.41,
      polymarketForecastReturn: -0.01,
      polymarketConfidence: 0.79,
      qualityScore: 84,
    }),
    makeReplayBundle({
      ticker: 'BTC',
      horizonDays: 14,
      actualBinary: 1,
      currentPrice: 76_500,
      realizedPrice: 78_000,
      markovForecastReturn: 0.001,
      markovConfidence: 0.17,
      structuralBreak: true,
      flatProbability: 0.86,
    }),
  ];
}

describe('combined short-horizon benchmark', () => {
  beforeAll(() => {
    const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
    fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'));
  });

  integrationIt('routes GOLD to GLD and exposes separate markov/polymarket/combined slices for 1d/2d/3d/7d/14d', async () => {
    const report = await runCombinedShortHorizonBenchmark({
      ticker: 'GOLD',
      prices: fixture.tickers.GLD.closes,
      replayBundles: makeGoldBundles(),
      generatedAt: GENERATED_AT,
      replaySourcePath: 'inline-gold-fixture',
    });

    expect(report).toMatchObject({
      formatVersion: 'combined-short-horizon-benchmark.v1',
      generatedAt: GENERATED_AT,
      ticker: 'GOLD',
      resolvedTicker: 'GLD',
      replaySourcePath: 'inline-gold-fixture',
    });
    expect(report.horizons['1d'].markovOnly.ready).toBe(true);
    expect(report.horizons['1d'].markovOnly.ciCoverage).not.toBeNull();
    expect(report.horizons['1d'].markovOnly.structuralBreakCount).not.toBeNull();
    expect(report.horizons['1d'].polymarketOnly).toMatchObject({
      slice: 'polymarket-only',
      observationCount: 1,
      labeledObservationCount: 1,
      tradedCount: 1,
      evaluatorName: 'polymarket-only',
      ciCoverage: null,
      structuralBreakCount: null,
      ready: true,
    });
    expect(report.horizons['3d'].combinedArbitrated.structuralBreakCount).toBe(1);
    expect(report.horizons['7d'].combinedArbitrated.abstainCount).toBeGreaterThan(0);
    expect(report.horizons['14d'].combinedArbitrated.pendingReasons).toContain(
      'Evaluator abstained on every labeled 14d replay bundle, so directional accuracy is undefined.',
    );
    expect(JSON.parse(formatCombinedShortHorizonBenchmarkReport(report))).toEqual(report);
  }, TIMEOUT);

  integrationIt('locks the BTC Phase 1 fixture baseline and proves the BTC live policy is applied', async () => {
    const report = await runCombinedShortHorizonBenchmark({
      ticker: 'BTC',
      prices: fixture.tickers['BTC-USD'].closes,
      replayBundles: makeBtcBundles(),
      generatedAt: GENERATED_AT,
      replaySourcePath: 'inline-btc-fixture',
    });

    expect(report.resolvedTicker).toBe('BTC');
    expect(summarizeBtcFixtureBaseline(report)).toEqual(BTC_FIXTURE_BASELINE);

    const naiveOneDay = await walkForward({
      ticker: 'BTC',
      prices: fixture.tickers['BTC-USD'].closes,
      horizon: 1,
      warmup: 120,
      stride: 5,
    });
    const naiveHoldCount = naiveOneDay.steps.filter((step) => step.recommendation === 'HOLD').length;
    const naiveBreakCount = naiveOneDay.steps.filter(
      (step) => (step.originalStructuralBreakDetected ?? step.structuralBreakDetected) === true,
    ).length;

    expect(naiveOneDay.steps.length).toBe(122);
    expect(naiveHoldCount).toBe(28);
    expect(naiveBreakCount).toBe(120);
    expect(report.horizons['1d'].markovOnly.observationCount).toBeLessThan(naiveOneDay.steps.length);
    expect(report.horizons['1d'].markovOnly.abstainCount).toBeLessThan(naiveHoldCount);
    expect(report.horizons['1d'].markovOnly.structuralBreakCount).toBeLessThan(naiveBreakCount);
    expect(report.horizons['7d'].combinedArbitrated.pendingReasons).toEqual([BTC_7D_ABSTAIN_REASON]);
    expect(report.horizons['14d'].combinedArbitrated.pendingReasons).toEqual([BTC_14D_ABSTAIN_REASON]);

    if (process.env.FORECAST_LAB_OUTPUT_METRICS === '1') {
      console.log(`FORECAST_LAB_METRICS ${JSON.stringify(report)}`);
    }
  }, TIMEOUT);
});
