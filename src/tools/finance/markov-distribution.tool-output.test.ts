/**
 * Tool-output envelope tests split from markov-distribution.test.ts.
 */

import { FIXED_TEST_DATE, FIXED_TEST_NOW_MS, deterministicRandom, nextTestId } from '@/utils/test-determinism.js';
import { describe, it, expect, mock, afterEach, beforeEach, setSystemTime } from 'bun:test';
import { integrationIt } from '../../utils/test-guards.js';
import {
  classifyRegimeState,
  computeAdaptiveThresholds,
  estimateTransitionMatrix,
  buildDefaultMatrix,
  normalizeRows,
  adjustTransitionMatrix,
  assessAnchorCoverage,
  extractPriceThresholds,
  interpolateDistribution,
  computeMarkovDistribution,
  secondLargestEigenvalue,
  computeMixingWeight,
  computeR2OS,
  calibrateProbabilities,
  computePredictionConfidenceBreakdown,
  computePredictionConfidence,
  getAssetProfile,
  computeRegimeUpRates,
  shouldApplyBtc14dBearishBreakSellGate,
  logNormalSurvival,
  estimateRegimeStats,
  matPow,
  matMul,
  normalCDF,
  transitionGoodnessOfFit,
  markovDistributionTool,
  computeEnsembleSignal,
  NUM_STATES,
  STATE_INDEX,
  REGIME_STATES,
  studentTCDF,
  studentTSurvival,
  computeTrajectory,
  computeStartStateMixture,
  computeHorizonDriftVol,
  winsorize,
  interpolateSurvival,
  computeScenarioProbabilities,
  normalizeAnchorPricesForETF,
  buildProfileFallbackMatrix,
  computeBlendWeight,
  applyBreakFallbackCandidate,
  sortMarketsByHorizonCloseness,
  filterMarketsToHorizon,
  inferPolymarketSearchPhrase,
  buildPolymarketAnchorQueryVariants,
  normalizeSentiment,
  buildForecastHint,
  RECOMMENDED_CONFIDENCE_THRESHOLD,
  buildRecommendationProvenanceNote,
  buildBtcShortHorizonThinAnchorWarning,
  capBtcShortHorizonConfidence,
  formatMarkovMixingLine,
  applyCryptoTerminalAnchorFallback,
  evaluateAnchorTrust,
  isCompositeValidationAcceptable,
  getBtcShortHorizonLivePolicy,
  getGoldShortHorizonLivePolicy,
  normalizeHistoricalPriceTicker,
  shouldEmitContextOnlyCanonical,
} from './markov-distribution.js';
import type { RegimeState, MarkovDistributionPoint, PriceThreshold, ScenarioProbabilities } from './markov-distribution.js';
import { MS_PER_DAY } from '../../utils/time.js';

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
});

afterEach(() => {
  setSystemTime();
});

const realPolymarketModule = { ...(await import('./polymarket.js')) };
const realHmmModule = { ...(await import('./hmm.js')) };
const realApiModule = { ...(await import('./api.js')) };

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Install default Polymarket mock for auto-fetch tests that fresh-import
 * markov-distribution.js and depend on default anchor data.
 */
function installDefaultPolymarketMock(): void {
  mock.module('./polymarket.js', () => ({
    ...realPolymarketModule,
    fetchPolymarketMarkets: async (_query: string, _limit: number) => [
      {
        question: 'Will the price of Bitcoin be above $64000 on April 9?',
        probability: 0.78,
        volume24h: 250000,
        ageDays: 5,
        endDate: new Date(FIXED_TEST_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will the price of Bitcoin be above $66000 on April 9?',
        probability: 0.54,
        volume24h: 220000,
        ageDays: 5,
        endDate: new Date(FIXED_TEST_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will the price of Bitcoin be above $68000 on April 9?',
        probability: 0.31,
        volume24h: 190000,
        ageDays: 5,
        endDate: new Date(FIXED_TEST_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will Bitcoin reach $70000 this week?',
        probability: 0.22,
        volume24h: 180000,
        ageDays: 5,
        endDate: new Date(FIXED_TEST_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
    ],
    fetchPolymarketAnchorMarkets: async (_query: string, _limit: number, _options: unknown) => [
      {
        question: 'Will the price of Bitcoin be above $62000 by end of week?',
        probability: 0.85,
        volume24h: 300000,
        ageDays: 7,
        endDate: new Date(FIXED_TEST_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will the price of Bitcoin be above $65000 by end of week?',
        probability: 0.62,
        volume24h: 260000,
        ageDays: 6,
        endDate: new Date(FIXED_TEST_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will the price of Bitcoin be above $68000 by end of week?',
        probability: 0.38,
        volume24h: 210000,
        ageDays: 5,
        endDate: new Date(FIXED_TEST_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will the price of Bitcoin fall below $63000 by end of week?',
        probability: 0.25,
        volume24h: 190000,
        ageDays: 5,
        endDate: new Date(FIXED_TEST_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
    ],
  }));
}

afterEach(() => {
  mock.module('./polymarket.js', () => realPolymarketModule);
  mock.module('./hmm.js', () => realHmmModule);
  mock.module('./api.js', () => realApiModule);
});

function rowSums(m: number[][]): number[] {
  return m.map(row => row.reduce((s, v) => s + v, 0));
}

function allClose(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) < tol;
}

/** Build a simple deterministic state sequence: n days of given repeating pattern. */
function repeatStates(pattern: ReturnType<typeof classifyRegimeState>[], n: number) {
  return Array.from({ length: n }, (_, i) => pattern[i % pattern.length]);
}

describe('markov_distribution tool output envelope', () => {
  integrationIt('auto-fetches candidate Polymarket anchors when polymarketMarkets are omitted', async () => {
    installDefaultPolymarketMock();
    const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }

    const parsedInput = freshTool.schema.parse({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      trajectory: false,
    });

    const result = await freshTool.func(parsedInput);

    const parsed = JSON.parse(result);
    expect(parsed.data._tool).toBe('markov_distribution');
    expect(parsed.data.canonical?.diagnostics?.totalAnchors).toBeGreaterThan(0);
    expect(parsed.data.canonical?.diagnostics?.trustedAnchors).toBeGreaterThan(0);
  });

  it('recovers earlier BTC terminal anchors when strict 14-day auto-fetch results are barrier-only', async () => {
    const now = FIXED_TEST_NOW_MS;
    const day = MS_PER_DAY;

    mock.module('./polymarket.js', () => ({
      ...realPolymarketModule,
      fetchPolymarketMarkets: async () => [],
      fetchPolymarketAnchorMarkets: async () => [
        { question: 'Will Bitcoin reach $80,000 in April?', probability: 0.30, volume24h: 50000, ageDays: 5, endDate: new Date(now + 14 * day).toISOString() },
        { question: 'Will Bitcoin reach $150,000 in April?', probability: 0.02, volume24h: 10000, ageDays: 5, endDate: new Date(now + 14 * day).toISOString() },
        { question: 'Will Bitcoin dip to $65,000 in April?', probability: 0.15, volume24h: 40000, ageDays: 5, endDate: new Date(now + 14 * day).toISOString() },
        { question: 'Will the price of Bitcoin be above $84,000 on April 17?', probability: 0.50, volume24h: 30000, ageDays: 5, endDate: new Date(now + 2 * day).toISOString() },
        { question: 'Will the price of Bitcoin be above $80,000 on April 18?', probability: 0.62, volume24h: 25000, ageDays: 5, endDate: new Date(now + 3 * day).toISOString() },
        { question: 'Will the price of Bitcoin be below $78,000 on April 19?', probability: 0.25, volume24h: 20000, ageDays: 5, endDate: new Date(now + 4 * day).toISOString() },
      ],
      fetchPolymarketAnchorMarketsWithQueries: async () => [],
    }));

    const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }

    const result = await freshTool.func({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    expect(parsed.data._tool).toBe('markov_distribution');
    expect(parsed.data.status).toBe('ok');
    expect(parsed.data.canonical?.diagnostics?.totalAnchors).toBeGreaterThan(0);
    expect(parsed.data.canonical?.diagnostics?.trustedAnchors).toBeGreaterThan(0);
    expect(parsed.data.canonical?.diagnostics?.anchorQuality).not.toBe('none');
    expect(parsed.data.canonical?.diagnostics?.canEmitCanonical).toBe(true);
    expect(parsed.data.distribution).not.toBeNull();
  });

  it('emits via sparse crypto anchor wrapper path when BTC 14-day has exactly one trusted anchor', async () => {
    const now = FIXED_TEST_NOW_MS;
    const day = MS_PER_DAY;

    mock.module('./polymarket.js', () => ({
      ...realPolymarketModule,
      fetchPolymarketMarkets: async () => [],
      fetchPolymarketAnchorMarkets: async () => [
        { question: 'Will Bitcoin reach $90,000 in April?', probability: 0.30, volume24h: 50000, ageDays: 5, endDate: new Date(now + 14 * day).toISOString() },
        { question: `Will the price of Bitcoin be above $84,000 on ${new Date(now + 14 * day).toISOString().slice(0, 10)}?`, probability: 0.50, volume24h: 30000, ageDays: 10, endDate: new Date(now + 14 * day).toISOString() },
      ],
    }));

    const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }

    const result = await freshTool.func({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    const diagnostics = parsed.data.canonical?.diagnostics;
    expect(parsed.data.status).toBe('ok');
    expect(parsed.data.manualSynthesisForbidden).toBe(false);
    expect(diagnostics?.anchorQuality).toBe('sparse');
    expect(diagnostics?.trustedAnchors).toBe(1);
    expect(diagnostics?.canEmitCanonical).toBe(true);
    expect(parsed.data.distribution).not.toBeNull();
  });

  it('retries later BTC 14-day query variants when front-slice anchors are all low-trust', async () => {
    const now = FIXED_TEST_NOW_MS;
    const day = MS_PER_DAY;
    const targetMonth = new Date(now + 14 * day)
      .toLocaleString('en-US', { month: 'long' });
    const frontQueries = new Set([
      'Bitcoin price',
      'Bitcoin',
      'Bitcoin above',
      'Bitcoin below',
      `Bitcoin ${targetMonth}`,
      `Bitcoin above ${targetMonth}`,
    ]);
    const retryCalls: string[][] = [];

    mock.module('./polymarket.js', () => ({
      ...realPolymarketModule,
      fetchPolymarketMarkets: async () => [],
      fetchPolymarketAnchorMarkets: async (query: string) => {
        if (!frontQueries.has(query)) return [];
        return [
          {
            question: 'Will the price of Bitcoin be above $74,000 on April 24?',
            probability: 0.45,
            volume24h: 20000,
            ageDays: 0,
            endDate: new Date(now + 5 * day).toISOString(),
          },
          {
            question: 'Will the price of Bitcoin be above $78,000 on April 25?',
            probability: 0.25,
            volume24h: 18000,
            ageDays: 0,
            endDate: new Date(now + 6 * day).toISOString(),
          },
        ];
      },
      fetchPolymarketAnchorMarketsWithQueries: async (queries: string[]) => {
        retryCalls.push([...queries]);
        return [
          {
            question: `Will the price of Bitcoin be above $76,000 on ${new Date(now + 14 * day).toISOString().slice(0, 10)}?`,
            probability: 0.48,
            volume24h: 50000,
            ageDays: 7,
            endDate: new Date(now + 14 * day).toISOString(),
          },
          {
            question: `Will the price of Bitcoin be above $80,000 on ${new Date(now + 14 * day).toISOString().slice(0, 10)}?`,
            probability: 0.22,
            volume24h: 42000,
            ageDays: 7,
            endDate: new Date(now + 14 * day).toISOString(),
          },
        ];
      },
    }));

    const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }

    const result = await freshTool.func({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    const diagnostics = parsed.data.canonical?.diagnostics;
    expect(retryCalls).toHaveLength(1);
    expect(retryCalls[0]).toEqual([
      'crypto regulation',
      'SEC crypto',
      'cryptocurrency regulation',
      'Bitcoin ETF',
      'crypto ETF',
      'Bitcoin price target',
      'Bitcoin reach',
      'Bitcoin exceed',
      'Bitcoin price level',
      'Fed rate cut',
      'Federal Reserve rate',
      'FOMC',
      'US recession',
      'recession',
      'economic recession',
    ]);
    expect(parsed.data.status).toBe('ok');
    expect(diagnostics?.trustedAnchors).toBeGreaterThan(0);
    expect(diagnostics?.anchorQuality).not.toBe('none');
    expect(diagnostics?.canEmitCanonical).toBe(true);
    expect(parsed.data.distribution).not.toBeNull();
  });

  it('uses a date-windowed search for BTC 14-day anchor auto-fetch before later-query retry', async () => {
    const frontCalls: Array<{ query: string; endDateFilter?: { end_date_min: string; end_date_max: string } }> = [];
    const retryCalls: Array<{ queries: string[]; endDateFilter?: { end_date_min: string; end_date_max: string } }> = [];
    const targetMonth = new Date(FIXED_TEST_NOW_MS + 14 * MS_PER_DAY)
      .toLocaleString('en-US', { month: 'long' });

    mock.module('./polymarket.js', () => ({
      ...realPolymarketModule,
      fetchPolymarketMarkets: async () => [],
      fetchPolymarketAnchorMarkets: async (
        query: string,
        _limit: number,
        options: { ticker: string; horizonDays?: number; endDateFilter?: { end_date_min: string; end_date_max: string } },
      ) => {
        frontCalls.push({ query, endDateFilter: options.endDateFilter });
        return [];
      },
      fetchPolymarketAnchorMarketsWithQueries: async (
        queries: string[],
        _limit: number,
        options: { ticker: string; horizonDays?: number; endDateFilter?: { end_date_min: string; end_date_max: string } },
      ) => {
        retryCalls.push({ queries: [...queries], endDateFilter: options.endDateFilter });
        return [];
      },
    }));

    const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }

    const result = await freshTool.func({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    const diagnostics = parsed.data.canonical?.diagnostics;
    expect(frontCalls).toHaveLength(6);
    expect(frontCalls.map((call) => call.query)).toEqual([
      'Bitcoin price',
      'Bitcoin',
      'Bitcoin above',
      'Bitcoin below',
      `Bitcoin ${targetMonth}`,
      `Bitcoin above ${targetMonth}`,
    ]);
    expect(frontCalls.every((call) => call.endDateFilter?.end_date_min && call.endDateFilter?.end_date_max)).toBe(true);
    expect(retryCalls.length).toBeGreaterThanOrEqual(1);
    const datedRetryCall = retryCalls.find((call) => call.endDateFilter !== undefined);
    expect(datedRetryCall).toBeDefined();
    expect(datedRetryCall!.queries).toEqual([
      'crypto regulation',
      'SEC crypto',
      'cryptocurrency regulation',
      'Bitcoin ETF',
      'crypto ETF',
      'Bitcoin price target',
      'Bitcoin reach',
      'Bitcoin exceed',
      'Bitcoin price level',
      'Fed rate cut',
      'Federal Reserve rate',
      'FOMC',
      'US recession',
      'recession',
      'economic recession',
    ]);
    expect(datedRetryCall!.endDateFilter).toEqual({
      end_date_min: expect.any(String),
      end_date_max: expect.any(String),
    });

    const undatedFallbackCall = retryCalls.find((call) => call.endDateFilter === undefined);
    expect(undatedFallbackCall).toBeDefined();
    expect(undatedFallbackCall!.queries).toEqual([
      'Bitcoin price',
      'Bitcoin',
      'Bitcoin above',
      'Bitcoin below',
      `Bitcoin ${targetMonth}`,
      `Bitcoin above ${targetMonth}`,
    ]);
    expect(parsed.data.status).toBe('ok');
    expect(diagnostics?.trustedAnchors).toBe(0);
    expect(diagnostics?.canEmitCanonical).toBe(true);
  });

  integrationIt('auto-fetches BTC 14-day Polymarket anchors and emits via model-only path when dated threshold inventory is unavailable', async () => {
    mock.module('./polymarket.js', () => ({
      ...realPolymarketModule,
      fetchPolymarketMarkets: async () => [],
      fetchPolymarketAnchorMarkets: async () => [],
      fetchPolymarketAnchorMarketsWithQueries: async () => [],
    }));
    const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }

    const parsedInput = freshTool.schema.parse({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      trajectory: false,
    });

    const result = await freshTool.func(parsedInput);
    const parsed = JSON.parse(result);
    expect(parsed.data._tool).toBe('markov_distribution');
    expect(parsed.data.status).toBe('ok');
    expect(parsed.data.canonical?.diagnostics?.trustedAnchors).toBe(0);
    expect(parsed.data.canonical?.diagnostics?.anchorQuality).toBe('none');
    expect(parsed.data.canonical?.diagnostics?.canEmitCanonical).toBe(true);
  });

  it('returns abstain payload when anchors or validation are insufficient', async () => {
    const result = await markovDistributionTool.func({
      ticker: 'SPY',
      horizon: 7,
      currentPrice: 100,
      historicalPrices: Array.from({ length: 40 }, (_, i) => 100 + i * 0.5),
      polymarketMarkets: [],
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    expect(parsed.data._tool).toBe('markov_distribution');
    expect(typeof parsed.data.report).toBe('string');
    expect(parsed.data.status).toBe('abstain');
    expect(parsed.data.manualSynthesisForbidden).toBe(true);
    expect(Array.isArray(parsed.data.abstainReasons)).toBe(true);
    expect(parsed.data.canonical).toBeDefined();
    expect(parsed.data.canonical.scenarios).toBeNull();
    expect(parsed.data.canonical.actionSignal).toBeNull();
    expect(parsed.data.canonical.diagnostics).toBeDefined();
    expect(parsed.data.canonical.diagnostics.canEmitCanonical).toBe(false);
    expect(parsed.data.distribution).toBeNull();
    expect(parsed.data.forecastHint).toBeNull();
  });

  it('explains low-trust BTC 14d anchors in abstain diagnostics', async () => {
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }

    const currentPrice = prices[prices.length - 1];
    const offHorizonEndDate = new Date(FIXED_TEST_NOW_MS + 6 * MS_PER_DAY);
    const offHorizonLabel = offHorizonEndDate.toLocaleString('en-US', {
      month: 'long',
      day: 'numeric',
      timeZone: 'UTC',
    });
    const offHorizonIso = offHorizonEndDate.toISOString().slice(0, 10);

    const result = await markovDistributionTool.func({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [
        {
          question: `Will the price of Bitcoin be above $${Math.round(currentPrice * 1.02)} on ${offHorizonLabel}?`,
          probability: 0.42,
          volume: 2000,
          createdAt: FIXED_TEST_NOW_MS,
          endDate: offHorizonIso,
        },
        {
          question: `Will the price of Bitcoin be below $${Math.round(currentPrice * 0.96)} on ${offHorizonLabel}?`,
          probability: 0.18,
          volume: 1200,
          createdAt: FIXED_TEST_NOW_MS,
          endDate: offHorizonIso,
        },
        {
          question: `Will the price of Bitcoin be between $${Math.round(currentPrice * 0.99)} and $${Math.round(currentPrice * 1.01)} on ${offHorizonLabel}?`,
          probability: 0.22,
          volume: 400,
          createdAt: FIXED_TEST_NOW_MS,
          endDate: offHorizonIso,
        },
      ],
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    const diagnostics = parsed.data.canonical?.diagnostics;
    const inspection = diagnostics?.anchorInspection;

    expect(parsed.data.status).toBe('ok');
    expect(inspection).toBeDefined();
    expect(inspection.candidateMarkets).toBe(3);
    expect(inspection.terminalThresholdMatches).toBe(2);
    expect(inspection.trustedAnchors).toBe(0);
    expect(inspection.lowTrustAnchors).toBeGreaterThan(0);
    expect(inspection.lowTrustReasonCounts.youngMarket).toBeGreaterThan(0);
    expect(inspection.lowTrustReasonCounts.resolutionMismatch).toBeGreaterThan(0);
    expect(inspection.excludedNonTerminalMarkets).toBe(1);
    expect(inspection.closestResolutionDate).toBe(offHorizonIso);
    expect(inspection.closestResolutionOffsetDays).toBeLessThan(0);
    expect(parsed.data.canonical.diagnostics.anchorBypassApplied).toBe(true);
    expect(parsed.data.report).toContain('Anchors: 0 trusted');
    expect(parsed.data.report).toContain('2 low-trust');
    expect(parsed.data.report).toContain('No trusted Polymarket anchors');
  });

  it('emits full canonical signal for BTC short-horizon with zero anchors via model-only path', async () => {
    mock.module('./polymarket.js', () => ({
      ...realPolymarketModule,
      fetchPolymarketMarkets: async () => [],
      fetchPolymarketAnchorMarkets: async () => [],
    }));

    const { markovDistributionTool: emitTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);

    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }

    const result = await emitTool.func({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    expect(parsed.data.status).toBe('ok');
    expect(parsed.data.canonical.actionSignal).not.toBeNull();
    expect(parsed.data.canonical.diagnostics.canEmitCanonical).toBe(true);
    expect(parsed.data.canonical.diagnostics.anchorBypassApplied).toBe(true);
    expect(typeof parsed.data.canonical.actionSignal.expectedReturn).toBe('number');
    expect(Number.isFinite(parsed.data.canonical.actionSignal.expectedReturn)).toBe(true);
    expect(parsed.data.canonical.actionSignal.expectedReturn).not.toBe(0);
  });

  it('emits full canonical signal for BTC short-horizon with structural break via model-only path', async () => {
    mock.module('./polymarket.js', () => ({
      ...realPolymarketModule,
      fetchPolymarketMarkets: async () => [],
      fetchPolymarketAnchorMarkets: async () => [],
    }));

    const { markovDistributionTool: emitTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 100; i++) {
      const shock = i > 85 ? 0.04 : Math.sin(i * 0.12) * 0.004;
      p *= 1 + shock;
      prices.push(Math.round(p * 100) / 100);
    }

    const result = await emitTool.func({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    expect(parsed.data.status).toBe('ok');
    expect(parsed.data.canonical.diagnostics.structuralBreakDetected).toBe(true);
    expect(parsed.data.canonical.actionSignal).not.toBeNull();
    expect(parsed.data.canonical.diagnostics.canEmitCanonical).toBe(true);
    expect(parsed.data.canonical.diagnostics.anchorBypassApplied).toBe(true);
  });

  it('BTC break-threshold override can suppress a detected structural break', async () => {
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 100; i++) {
      const shock = i > 85 ? 0.04 : Math.sin(i * 0.12) * 0.004;
      p *= 1 + shock;
      prices.push(Math.round(p * 100) / 100);
    }

    const baseline = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
    });

    const relaxed = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      btcBreakDivergenceThreshold: 1.0,
    });

    expect(baseline.metadata.structuralBreakDetected).toBe(true);
    expect(relaxed.metadata.structuralBreakDetected).toBe(false);
  });

  it('suppresses forecastHint when BTC short-horizon abstain confidence is too low', () => {
    const forecastHint = buildForecastHint({
      canEmitCanonical: false,
      ticker: 'BTC-USD',
      horizon: 7,
      expectedReturn: 0.04,
      mixingTimeWeight: 0.6,
      predictionConfidence: 0.08,
    });

    expect(forecastHint).toBeNull();
  });

  it('returns the promoted BTC short-horizon live policy by horizon', () => {
    expect(getBtcShortHorizonLivePolicy('ETH-USD', 1)).toBeNull();
    expect(getBtcShortHorizonLivePolicy('BTC-USD', 30)).toBeNull();
    expect(getBtcShortHorizonLivePolicy('BTC-USD', 1)).toEqual({
      historyDays: 252,
      breakDivergenceThreshold: 0.10,
      rerunOnBreak: true,
      rerunWindowDays: 60,
    });
    expect(getBtcShortHorizonLivePolicy('BTC', 3)).toEqual({
      historyDays: 252,
      breakDivergenceThreshold: 0.15,
      rerunOnBreak: true,
      rerunWindowDays: 45,
    });
    expect(getBtcShortHorizonLivePolicy('BTC-USD', 2)).toEqual({
      historyDays: 252,
      breakDivergenceThreshold: 0.15,
      rerunOnBreak: true,
      rerunWindowDays: 120,
    });
    expect(getBtcShortHorizonLivePolicy('BTC-USD', 14)).toEqual({
      historyDays: 252,
      breakDivergenceThreshold: 0.08,
      rerunOnBreak: false,
    });
  });

  it('returns a GOLD short-horizon live policy for GLD and commodity-gold aliases at 1/2/3/7/14d', () => {
    for (const ticker of ['GLD', 'XAUUSD']) {
      for (const horizon of [1, 2, 3]) {
        expect(getGoldShortHorizonLivePolicy(ticker, horizon)).toEqual({
          historyDays: 252,
          breakDivergenceThreshold: 0.12,
          rerunOnBreak: false,
        });
      }
      for (const horizon of [7, 14]) {
        expect(getGoldShortHorizonLivePolicy(ticker, horizon)).toEqual({
          historyDays: 252,
          breakDivergenceThreshold: 0.15,
          rerunOnBreak: false,
        });
      }
    }
  });

  it('returns null for unrelated assets from the GOLD short-horizon seam', () => {
    expect(getGoldShortHorizonLivePolicy('SPY', 7)).toBeNull();
    expect(getGoldShortHorizonLivePolicy('BTC-USD', 7)).toBeNull();
    expect(getGoldShortHorizonLivePolicy('GOLD', 7)).toBeNull();
    expect(getGoldShortHorizonLivePolicy('GLD', 30)).toBeNull();
  });

  it('keeps BTC on the existing BTC short-horizon helper unchanged', () => {
    expect(getGoldShortHorizonLivePolicy('BTC-USD', 2)).toBeNull();
    expect(getBtcShortHorizonLivePolicy('BTC-USD', 2)).toEqual({
      historyDays: 252,
      breakDivergenceThreshold: 0.15,
      rerunOnBreak: true,
      rerunWindowDays: 120,
    });
    expect(getBtcShortHorizonLivePolicy('BTC-USD', 1)).toEqual({
      historyDays: 252,
      breakDivergenceThreshold: 0.10,
      rerunOnBreak: true,
      rerunWindowDays: 60,
    });
  });

  it('buildForecastHint contract: BTC-only, horizon ≤ 14, and attenuation formula', () => {
    const base = {
      canEmitCanonical: false,
      ticker: 'BTC-USD',
      horizon: 7,
      expectedReturn: 0.04,
      mixingTimeWeight: 0.6,
      predictionConfidence: 0.20,
    };

    expect(buildForecastHint({ ...base, ticker: 'ETH-USD' })).toBeNull();
    expect(buildForecastHint({ ...base, ticker: 'SPY' })).toBeNull();

    expect(buildForecastHint({ ...base, horizon: 15 })).toBeNull();
    expect(buildForecastHint({ ...base, horizon: 30 })).toBeNull();

    const atBoundary = buildForecastHint({ ...base, horizon: 14 });
    expect(atBoundary).not.toBeNull();

    expect(buildForecastHint({ ...base, canEmitCanonical: true })).toBeNull();

    const hint = buildForecastHint(base);
    expect(hint).not.toBeNull();
    expect(hint!.usage).toBe('forecast_only');
    expect(hint!.calibratedDistribution).toBe(false);
    expect(hint!.confidenceScore).toBe(0.20);
    expect(hint!.markovReturn).toBeCloseTo(
      0.04 * 0.6 * 0.5 * Math.min(1, 0.20 / RECOMMENDED_CONFIDENCE_THRESHOLD),
      10,
    );

    const highConf = buildForecastHint({ ...base, predictionConfidence: 0.30 });
    expect(highConf).not.toBeNull();
    expect(highConf!.markovReturn).toBeCloseTo(0.012, 10);
  });

  describe('BTC short-horizon confidence cap', () => {
    it('caps HIGH confidence to MEDIUM for BTC short-horizon outputs with structural break and weak diagnostics', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC-USD',
        horizon: 7,
        structuralBreakDetected: true,
        outOfSampleR2: 0.02,
        predictionConfidence: 0.35,
        confidence: 'HIGH',
      })).toBe('MEDIUM');
    });

    it('does not change LOW confidence even when BTC short-horizon diagnostics are weak', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC-USD',
        horizon: 7,
        structuralBreakDetected: true,
        outOfSampleR2: 0.02,
        predictionConfidence: 0.35,
        confidence: 'LOW',
      })).toBe('LOW');
    });

    it('does not cap when structural break is absent', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC-USD',
        horizon: 7,
        structuralBreakDetected: false,
        outOfSampleR2: 0.02,
        predictionConfidence: 0.35,
        confidence: 'HIGH',
      })).toBe('HIGH');
    });

    it('does not cap BTC long-horizon confidence', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC-USD',
        horizon: 30,
        structuralBreakDetected: true,
        outOfSampleR2: 0.02,
        predictionConfidence: 0.35,
        confidence: 'HIGH',
      })).toBe('HIGH');
    });

    it('recognizes BTC ticker alias directly', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC',
        horizon: 14,
        structuralBreakDetected: true,
        outOfSampleR2: 0.03,
        predictionConfidence: 0.50,
        confidence: 'HIGH',
      })).toBe('MEDIUM');
    });

    it('does not cap non-BTC assets', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'AAPL',
        horizon: 7,
        structuralBreakDetected: true,
        outOfSampleR2: 0.02,
        predictionConfidence: 0.35,
        confidence: 'HIGH',
      })).toBe('HIGH');
    });

    it('does not treat null R² as weak by itself', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC-USD',
        horizon: 14,
        structuralBreakDetected: true,
        outOfSampleR2: null,
        predictionConfidence: 0.50,
        confidence: 'HIGH',
      })).toBe('HIGH');
    });

    it('does not cap when both diagnostics are above threshold', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC-USD',
        horizon: 14,
        structuralBreakDetected: true,
        outOfSampleR2: 0.10,
        predictionConfidence: 0.50,
        confidence: 'HIGH',
      })).toBe('HIGH');
    });

    it('does not cap at the exact R² and prediction-confidence boundaries', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC-USD',
        horizon: 14,
        structuralBreakDetected: true,
        outOfSampleR2: 0.05,
        predictionConfidence: 0.40,
        confidence: 'HIGH',
      })).toBe('HIGH');
    });

    it('caps at horizon 14 but not 15', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC-USD',
        horizon: 14,
        structuralBreakDetected: true,
        outOfSampleR2: 0.03,
        predictionConfidence: 0.50,
        confidence: 'HIGH',
      })).toBe('MEDIUM');

      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC-USD',
        horizon: 15,
        structuralBreakDetected: true,
        outOfSampleR2: 0.03,
        predictionConfidence: 0.50,
        confidence: 'HIGH',
      })).toBe('HIGH');
    });

    it('caps when prediction confidence alone is weak', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC-USD',
        horizon: 14,
        structuralBreakDetected: true,
        outOfSampleR2: 0.10,
        predictionConfidence: 0.38,
        confidence: 'HIGH',
      })).toBe('MEDIUM');
    });

    it('caps when validation alone is weak', () => {
      expect(capBtcShortHorizonConfidence({
        ticker: 'BTC-USD',
        horizon: 14,
        structuralBreakDetected: true,
        outOfSampleR2: 0.03,
        predictionConfidence: 0.50,
        confidence: 'HIGH',
      })).toBe('MEDIUM');
    });
  });

  describe('BTC short-horizon thin-anchor caveat', () => {
    it('emits a thin-anchor warning for BTC short-horizon good-quality coverage with 2 trusted anchors', () => {
      expect(buildBtcShortHorizonThinAnchorWarning({
        ticker: 'BTC-USD',
        horizonDays: 7,
        trustedAnchors: 2,
        quality: 'good',
      })).toContain('thin anchor coverage');
    });

    it('does not emit the caveat for BTC short-horizon when anchor count is above 3', () => {
      expect(buildBtcShortHorizonThinAnchorWarning({
        ticker: 'BTC-USD',
        horizonDays: 7,
        trustedAnchors: 4,
        quality: 'good',
      })).toBe('');
    });

    it('does not emit the caveat for BTC long-horizon coverage', () => {
      expect(buildBtcShortHorizonThinAnchorWarning({
        ticker: 'BTC-USD',
        horizonDays: 30,
        trustedAnchors: 3,
        quality: 'good',
      })).toBe('');
    });

    it('emits the caveat for the BTC ticker alias directly', () => {
      expect(buildBtcShortHorizonThinAnchorWarning({
        ticker: 'BTC',
        horizonDays: 14,
        trustedAnchors: 3,
        quality: 'good',
      })).toContain('thin anchor coverage');
    });

    it('does not emit the caveat for non-BTC assets', () => {
      expect(buildBtcShortHorizonThinAnchorWarning({
        ticker: 'ETH-USD',
        horizonDays: 7,
        trustedAnchors: 2,
        quality: 'good',
      })).toBe('');
    });

    it('does not emit the caveat when quality is not good', () => {
      expect(buildBtcShortHorizonThinAnchorWarning({
        ticker: 'BTC-USD',
        horizonDays: 14,
        trustedAnchors: 3,
        quality: 'sparse',
      })).toBe('');
    });

    it('preserves good quality while adding the BTC short-horizon caveat', () => {
      const coverage = assessAnchorCoverage([
        { price: 62000, probability: 0.72, rawProbability: 0.72, trustScore: 'high', source: 'polymarket', endDate: '2026-04-25' },
        { price: 66000, probability: 0.38, rawProbability: 0.38, trustScore: 'high', source: 'polymarket', endDate: '2026-04-25' },
      ], 64000, { ticker: 'BTC-USD', horizonDays: 14 });

      expect(coverage.quality).toBe('good');
      expect(coverage.trustedAnchors).toBe(2);
      expect(coverage.warning).toContain('thin anchor coverage');
    });

    it('propagates the caveat through markovDistributionTool while keeping good quality and canonical output', async () => {
      const prices: number[] = [];
      let p = 65000;
      for (let i = 0; i < 120; i++) {
        p *= 1 + Math.sin(i * 0.12) * 0.004;
        prices.push(Math.round(p * 100) / 100);
      }

      const currentPrice = prices[prices.length - 1];
      const result = await markovDistributionTool.func({
        ticker: 'BTC-USD',
        horizon: 7,
        currentPrice,
        historicalPrices: prices,
        polymarketMarkets: [
          { question: `Will the price of Bitcoin be above $${Math.round(currentPrice * 0.98)} on April 25?`, probability: 0.68, volume: 5000, createdAt: FIXED_TEST_NOW_MS - MS_PER_DAY * 4 },
          { question: `Will the price of Bitcoin be above $${Math.round(currentPrice * 1.02)} on April 25?`, probability: 0.34, volume: 5000, createdAt: FIXED_TEST_NOW_MS - MS_PER_DAY * 4 },
        ],
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.status).toBe('ok');
      expect(parsed.data.canonical.diagnostics.anchorQuality).toBe('good');
      expect(parsed.data.canonical.diagnostics.canEmitCanonical).toBe(true);
      expect(parsed.data.canonical.diagnostics.anchorWarning).toContain('thin anchor coverage');
      expect(parsed.data.report).toContain('thin anchor coverage');
    });
  });

  describe('markov mixing line formatter', () => {
    it('preserves anchor presence when anchored mixing rounds to near-pure Markov', () => {
      expect(formatMarkovMixingLine({
        commodityModelOnly: false,
        markovWeight: 0.9998,
        anchorWeight: 0.0002,
        trustedAnchors: 5,
      })).toBe('Mixing: >99.9% Markov / <0.1% Anchors (anchors present; final blend is nearly pure Markov)');
    });

    it('preserves markov presence when anchored mixing rounds to near-pure anchors', () => {
      expect(formatMarkovMixingLine({
        commodityModelOnly: false,
        markovWeight: 0.0002,
        anchorWeight: 0.9998,
        trustedAnchors: 5,
      })).toBe('Mixing: <0.1% Markov / >99.9% Anchors (anchors present; final blend is nearly pure Anchors)');
    });

    it('keeps model-only wording reserved for bypass runs', () => {
      expect(formatMarkovMixingLine({
        commodityModelOnly: true,
        markovWeight: 1,
        anchorWeight: 0,
        trustedAnchors: 0,
      })).toBe('Calibration: model-only (commodity bypass, no anchors)');
    });
  });

  describe('Phase 1: BTC zero-anchor crypto model-only path', () => {
    // Helper: synthetic BTC prices with a clean structural break mid-series
    function btcBreakPrices(length = 70): number[] {
      const prices: number[] = [];
      let p = 65000;
      // First half: mild uptrend
      for (let i = 0; i < Math.floor(length / 2); i++) {
        p *= 1 + 0.001 + Math.sin(i * 0.3) * 0.002;
        prices.push(Math.round(p * 100) / 100);
      }
      // Second half: sharp volatility regime change
      p = 55000;
      for (let i = Math.floor(length / 2); i < length; i++) {
        p *= 1 + (i % 5 === 0 ? -0.015 : 0.008) + Math.sin(i * 0.7) * 0.005;
        prices.push(Math.round(p * 100) / 100);
      }
      return prices;
    }

    it('emits via crypto model-only path when BTC 3d has zero anchors, acceptable R^2, and structural break', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
      const prices = btcBreakPrices(80);

      const result = await freshTool.func({
        ticker: 'BTC-USD',
        horizon: 3,
        currentPrice: prices[prices.length - 1],
        historicalPrices: prices,
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.status).toBe('ok');
      expect(parsed.data.manualSynthesisForbidden).toBe(false);
      expect(parsed.data.canonical.diagnostics.canEmitCanonical).toBe(true);
      expect(parsed.data.distribution).not.toBeNull();
    });

    it('abstains via crypto model-only path when R^2 is below -0.03', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
      // Deterministic alternating prices produce poor out-of-sample R^2 without
      // making this regression depend on deterministicRandom().
      const prices = Array.from({ length: 80 }, (_, i) => (
        Math.round((65000 + (i % 2 === 0 ? 2000 : -2000) + (i % 5) * 10) * 100) / 100
      ));

      const result = await freshTool.func({
        ticker: 'BTC-USD',
        horizon: 7,
        currentPrice: prices[prices.length - 1],
        historicalPrices: prices,
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      // Should abstain because R^2 is poor with noisy prices
      expect(parsed.data.status === 'abstain' || parsed.data.canonical.diagnostics.canEmitCanonical === false).toBe(true);
    });

    it('abstains via crypto model-only path when confidence is below 0.18', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
      // Very short price history (low confidence from sparse data)
      const prices: number[] = [];
      let p = 65000;
      for (let i = 0; i < 30; i++) {
        p *= 1 + Math.sin(i * 0.5) * 0.003;
        prices.push(Math.round(p * 100) / 100);
      }

      const result = await freshTool.func({
        ticker: 'BTC-USD',
        horizon: 3,
        currentPrice: prices[prices.length - 1],
        historicalPrices: prices,
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      // Short history should yield low confidence
      expect(parsed.data.status === 'abstain' || parsed.data.canonical.diagnostics.canEmitCanonical === false).toBe(true);
    });
  });


  it('emits undefined provenance flags when break-confidence flags are off', async () => {
    const simplePrices = Array.from({ length: 90 }, (_, i) => 100 + i * 0.2 + Math.sin(i) * 2);
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 7,
      currentPrice: 118,
      historicalPrices: simplePrices,
      polymarketMarkets: [],
      trendPenaltyOnlyBreakConfidence: false,
      divergenceWeightedBreakConfidence: false,
    });

    expect(result.metadata.trendPenaltyOnlyBreakConfidenceActive).toBeUndefined();
    expect(result.metadata.divergenceWeightedBreakConfidenceActive).toBeUndefined();
  });

  it('exposes conformal diagnostics only when adaptive conformal is enabled', async () => {
    interface AdaptiveConformalMarkovDistributionOptions {
      enableAdaptiveConformal?: boolean;
      conformalAlpha?: number;
      conformalBreakSensitivity?: number;
    }

    interface AdaptiveConformalMetadataContract {
      applied: true;
      radius: number;
      coverageEstimate: number | null;
      mode: 'normal' | 'break';
    }

    type PlannedAdaptiveConformalMetadata =
      Awaited<ReturnType<typeof computeMarkovDistribution>>['metadata']
      & { conformal?: AdaptiveConformalMetadataContract };

    function getConformalMetadata(
      metadata: Awaited<ReturnType<typeof computeMarkovDistribution>>['metadata'],
    ): AdaptiveConformalMetadataContract | undefined {
      return (metadata as PlannedAdaptiveConformalMetadata).conformal;
    }

    const prices = Array.from({ length: 140 }, (_, i) =>
      i < 90
        ? 100 + i * 0.18 + Math.sin(i * 0.25)
        : 116 - (i - 90) * 0.45 + Math.sin(i * 0.25) * 2,
    );

    const disabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const explicitlyDisabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      enableAdaptiveConformal: false,
      conformalAlpha: 0.1,
      conformalBreakSensitivity: 1.5,
    } as Parameters<typeof computeMarkovDistribution>[0] & AdaptiveConformalMarkovDistributionOptions);
    const enabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      enableAdaptiveConformal: true,
      conformalAlpha: 0.1,
      conformalBreakSensitivity: 1.5,
    } as Parameters<typeof computeMarkovDistribution>[0] & AdaptiveConformalMarkovDistributionOptions);

    const disabledConformal = getConformalMetadata(disabled.metadata);
    const explicitlyDisabledConformal = getConformalMetadata(explicitlyDisabled.metadata);
    const enabledConformal = getConformalMetadata(enabled.metadata);

    expect(disabledConformal).toBeUndefined();
    expect(explicitlyDisabledConformal).toBeUndefined();
    expect(Object.hasOwn(disabled.metadata, 'conformal')).toBe(false);
    expect(Object.hasOwn(explicitlyDisabled.metadata, 'conformal')).toBe(false);
    expect(enabledConformal).toStrictEqual({
      applied: true,
      radius: expect.any(Number),
      coverageEstimate: null,
      mode: expect.stringMatching(/^(normal|break)$/),
    });
    expect(Object.keys(enabledConformal ?? {}).sort()).toEqual([
      'applied',
      'coverageEstimate',
      'mode',
      'radius',
    ]);
  });

  it('surfaces confidence-breakdown metadata alongside the collapsed score', async () => {
    interface ConfidenceBreakdownContract {
      mode: string;
      total: number;
      components: Record<string, number>;
      multipliers: Record<string, number>;
    }

    type PlannedConfidenceMetadata =
      Awaited<ReturnType<typeof computeMarkovDistribution>>['metadata']
      & { confidence?: ConfidenceBreakdownContract };

    function getConfidenceMetadata(
      metadata: Awaited<ReturnType<typeof computeMarkovDistribution>>['metadata'],
    ): ConfidenceBreakdownContract | undefined {
      return (metadata as PlannedConfidenceMetadata).confidence;
    }

    const prices = Array.from({ length: 140 }, (_, i) =>
      100 + i * 0.15 + Math.sin(i * 0.18) * 1.5,
    );

    const result = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
    });

    const confidence = getConfidenceMetadata(result.metadata);
    expect(confidence).toStrictEqual({
      mode: expect.any(String),
      total: result.predictionConfidence,
      components: {
        decisiveness: expect.any(Number),
        ensembleConsensus: expect.any(Number),
        hmmConvergence: expect.any(Number),
        regimeStability: expect.any(Number),
        momentumAgreement: expect.any(Number),
        baseRateAlignment: expect.any(Number),
        nearZeroR2Bonus: expect.any(Number),
        anchorSupport: expect.any(Number),
      },
      multipliers: {
        structuralBreak: expect.any(Number),
        assetType: expect.any(Number),
        volatility: expect.any(Number),
        normalization: expect.any(Number),
        posteriorUncertainty: expect.any(Number),
      },
    });
  });

  it('surfaces posterior-entropy diagnostics only when soft regime weighting is enabled', async () => {
    interface SoftRegimeMetadataContract {
      posteriorEntropy: number;
      forecastEntropy: number;
      ciScale: number;
      confidenceMultiplier: number;
      confidenceFloor: number;
      confidenceEntropyWeight: number;
      ciEntropyWeight: number;
      hmmWeightFloor: number;
      hmmWeightEntropyWeight: number;
      dominantStateProbability: number;
      currentStateProbabilities: number[];
      forecastProbabilities: number[];
      currentRegimeMixture?: { bull: number; bear: number; sideways: number };
      forecastRegimeMixture?: { bull: number; bear: number; sideways: number };
      transitionBlendWeight?: number;
    }

    type PlannedSoftRegimeMetadata =
      Awaited<ReturnType<typeof computeMarkovDistribution>>['metadata']
      & { softRegime?: SoftRegimeMetadataContract };

    function getSoftRegimeMetadata(
      metadata: Awaited<ReturnType<typeof computeMarkovDistribution>>['metadata'],
    ): SoftRegimeMetadataContract | undefined {
      return (metadata as PlannedSoftRegimeMetadata).softRegime;
    }

    const choppyPrices: number[] = Array.from({ length: 180 }, (_, i) =>
      i === 0 ? 100 : 0,
    );
    for (let i = 1; i < choppyPrices.length; i++) {
      const prev = choppyPrices[i - 1];
      const shock = i % 2 === 0 ? 0.012 : -0.011;
      choppyPrices[i] = prev * Math.exp(shock + Math.sin(i * 0.45) * 0.004);
    }

    const trendPrices = Array.from({ length: 180 }, (_, i) =>
      100 * Math.exp(i * 0.0025 + Math.sin(i * 0.08) * 0.002),
    );

    const disabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: choppyPrices[choppyPrices.length - 1],
      historicalPrices: choppyPrices,
      polymarketMarkets: [],
      predictionConfidenceMode: 'rebalanced',
      enableSoftRegimeWeighting: false,
    });
    const enabledChoppy = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: choppyPrices[choppyPrices.length - 1],
      historicalPrices: choppyPrices,
      polymarketMarkets: [],
      predictionConfidenceMode: 'rebalanced',
      enableSoftRegimeWeighting: true,
    });
    const enabledTrend = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: trendPrices[trendPrices.length - 1],
      historicalPrices: trendPrices,
      polymarketMarkets: [],
      predictionConfidenceMode: 'rebalanced',
      enableSoftRegimeWeighting: true,
    });

    const softChoppy = getSoftRegimeMetadata(enabledChoppy.metadata);
    const softTrend = getSoftRegimeMetadata(enabledTrend.metadata);

    expect(getSoftRegimeMetadata(disabled.metadata)).toBeUndefined();
    expect(softChoppy).toStrictEqual({
      posteriorEntropy: expect.any(Number),
      forecastEntropy: expect.any(Number),
      ciScale: expect.any(Number),
      confidenceMultiplier: expect.any(Number),
      confidenceFloor: expect.any(Number),
      confidenceEntropyWeight: expect.any(Number),
      ciEntropyWeight: expect.any(Number),
      hmmWeightFloor: expect.any(Number),
      hmmWeightEntropyWeight: expect.any(Number),
      dominantStateProbability: expect.any(Number),
      currentStateProbabilities: expect.any(Array),
      forecastProbabilities: expect.any(Array),
      currentRegimeMixture: expect.any(Object),
      forecastRegimeMixture: expect.any(Object),
      transitionBlendWeight: expect.any(Number),
    });
    expect(softChoppy!.currentStateProbabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 6);
    expect(softChoppy!.forecastProbabilities.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 6);
    expect(Object.values(softChoppy!.currentRegimeMixture ?? {}).reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 6);
    expect(Object.values(softChoppy!.forecastRegimeMixture ?? {}).reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 6);
    expect(softChoppy!.transitionBlendWeight).toBeGreaterThan(0);
    expect(softChoppy!.ciScale).toBeGreaterThan(1);
    expect(softChoppy!.confidenceMultiplier).toBeLessThan(1);
    expect(enabledChoppy.predictionConfidence).toBeLessThan(disabled.predictionConfidence);
    expect(
      Math.abs(enabledChoppy.actionSignal.expectedReturn - disabled.actionSignal.expectedReturn),
    ).toBeGreaterThan(1e-4);
    expect(softTrend).toBeDefined();
    expect(softTrend!.posteriorEntropy).toBeLessThanOrEqual(softChoppy!.posteriorEntropy);
    expect(softTrend!.ciScale).toBeLessThanOrEqual(softChoppy!.ciScale);
  });

  it('lets soft regime entropy coefficients be tuned explicitly without changing explicit defaults', async () => {
    const choppyPrices: number[] = Array.from({ length: 180 }, (_, i) =>
      i === 0 ? 100 : 0,
    );
    for (let i = 1; i < choppyPrices.length; i++) {
      const prev = choppyPrices[i - 1];
      const shock = i % 2 === 0 ? 0.012 : -0.011;
      choppyPrices[i] = prev * Math.exp(shock + Math.sin(i * 0.45) * 0.004);
    }

    const implicitDefaults = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: choppyPrices[choppyPrices.length - 1],
      historicalPrices: choppyPrices,
      polymarketMarkets: [],
      predictionConfidenceMode: 'rebalanced',
      enableSoftRegimeWeighting: true,
    });
    const explicitDefaults = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: choppyPrices[choppyPrices.length - 1],
      historicalPrices: choppyPrices,
      polymarketMarkets: [],
      predictionConfidenceMode: 'rebalanced',
      enableSoftRegimeWeighting: true,
      softRegimeConfidenceFloor: 0.65,
      softRegimeConfidenceEntropyWeight: 0.35,
      softRegimeCiEntropyWeight: 0.35,
      softRegimeHmmWeightFloor: 0.5,
      softRegimeHmmWeightEntropyWeight: 0.4,
    });
    const aggressive = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: choppyPrices[choppyPrices.length - 1],
      historicalPrices: choppyPrices,
      polymarketMarkets: [],
      predictionConfidenceMode: 'rebalanced',
      enableSoftRegimeWeighting: true,
      softRegimeConfidenceFloor: 0.4,
      softRegimeConfidenceEntropyWeight: 0.6,
      softRegimeCiEntropyWeight: 0.6,
      softRegimeHmmWeightFloor: 0.3,
      softRegimeHmmWeightEntropyWeight: 0.7,
    });

    expect(explicitDefaults.predictionConfidence).toBeCloseTo(implicitDefaults.predictionConfidence, 10);
    expect(explicitDefaults.metadata.softRegime?.ciScale).toBeCloseTo(
      implicitDefaults.metadata.softRegime?.ciScale ?? 0,
      10,
    );
    expect(aggressive.metadata.softRegime?.ciScale).toBeGreaterThan(
      implicitDefaults.metadata.softRegime?.ciScale ?? 0,
    );
    expect(aggressive.metadata.softRegime?.confidenceMultiplier).toBeLessThan(
      implicitDefaults.metadata.softRegime?.confidenceMultiplier ?? 1,
    );
    expect(aggressive.metadata.softRegime?.confidenceFloor).toBeCloseTo(0.4, 10);
    expect(aggressive.metadata.softRegime?.hmmWeightEntropyWeight).toBeCloseTo(0.7, 10);
  });

  it('pushes the raw forecast path toward the soft forecast regime mixture only when enabled', async () => {
    mock.module('./hmm.js', () => ({
      ...realHmmModule,
      baumWelch: () => ({
        params: {
          nStates: 3,
          pi: [1 / 3, 1 / 3, 1 / 3],
          A: [
            [0.70, 0.20, 0.10],
            [0.20, 0.60, 0.20],
            [0.10, 0.20, 0.70],
          ],
          means: [-0.02, 0.0, 0.02],
          stds: [0.01, 0.01, 0.01],
        },
        logLikelihood: -123,
        iterations: 3,
        converged: false,
      }),
      predict: () => ({
        currentState: 1,
        stateProbabilities: [],
        currentStateProbabilities: [0, 1, 0],
        forecastProbabilities: [0.10, 0.20, 0.70],
        expectedReturn: 0,
        expectedVolatility: 0.01,
      }),
    }));

    const returns = Array.from({ length: 160 }, (_, i) => [0, 0.025, -0.025, 0][i % 4]);
    const prices = [100];
    for (const ret of returns) prices.push(prices[prices.length - 1] * Math.exp(ret));

    const currentPrice = prices[prices.length - 1];
    const upsideTarget = currentPrice * 1.05;

    const disabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      predictionConfidenceMode: 'rebalanced',
      enableSoftRegimeWeighting: false,
    });
    const enabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      predictionConfidenceMode: 'rebalanced',
      enableSoftRegimeWeighting: true,
    });

    expect(
      interpolateSurvival(enabled.rawDistribution, upsideTarget),
    ).toBeGreaterThan(interpolateSurvival(disabled.rawDistribution, upsideTarget) + 0.03);
    expect(enabled.actionSignal.expectedReturn).toBeGreaterThan(disabled.actionSignal.expectedReturn);
  });

  it('keeps Student-t HMM emissions default-off and changes the forecast path only when enabled', async () => {
    const predictSawStudentT: boolean[] = [];

    mock.module('./hmm.js', () => ({
      ...realHmmModule,
      baumWelch: () => ({
        params: {
          nStates: 3,
          pi: [0.2, 0.6, 0.2],
          A: [
            [0.85, 0.10, 0.05],
            [0.15, 0.70, 0.15],
            [0.05, 0.10, 0.85],
          ],
          means: [-0.03, 0.0, 0.03],
          stds: [0.012, 0.010, 0.012],
        },
        logLikelihood: -88,
        iterations: 4,
        converged: true,
      }),
      predict: (_observations: number[], params: { nStates: number; studentTEmissions?: unknown[] }) => {
        const enabled = Array.isArray(params.studentTEmissions);
        if (params.nStates === 3) predictSawStudentT.push(enabled);
        return {
          currentState: enabled ? 2 : 1,
          stateProbabilities: [],
          currentStateProbabilities: enabled ? [0.15, 0.35, 0.50] : [0.10, 0.80, 0.10],
          forecastProbabilities: enabled ? [0.10, 0.20, 0.70] : [0.20, 0.60, 0.20],
          expectedReturn: enabled ? 0.012 : 0.002,
          expectedVolatility: enabled ? 0.035 : 0.012,
        };
      },
    }));

    const returns = Array.from({ length: 120 }, (_, i) => [0.002, -0.003, 0.004, -0.002, 0.001][i % 5]);
    const prices = [100];
    for (const ret of returns) prices.push(prices[prices.length - 1] * Math.exp(ret));
    const currentPrice = prices[prices.length - 1];

    const implicitDefault = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const explicitFalse = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      enableStudentTEmission: false,
    });
    const enabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      enableStudentTEmission: true,
    });

    expect(predictSawStudentT[0]).toBe(false);
    expect(predictSawStudentT[1]).toBe(false);
    expect(predictSawStudentT).toContain(true);
    expect(implicitDefault.metadata.hmm?.emissionFamily).toBe('gaussian');
    expect(explicitFalse.metadata.hmm?.emissionFamily).toBe('gaussian');
    expect(enabled.metadata.hmm?.emissionFamily).toBe('student-t-predictive');
    expect(enabled.metadata.hmm?.studentTDegreesOfFreedom?.length).toBe(3);
    expect(enabled.actionSignal.expectedReturn).toBeGreaterThan(implicitDefault.actionSignal.expectedReturn);
  });

  it('maps soft regime mixtures with Student-t predictive means when enabled', async () => {
    mock.module('./hmm.js', () => ({
      ...realHmmModule,
      baumWelch: () => ({
        params: {
          nStates: 3,
          pi: [0.2, 0.6, 0.2],
          A: [
            [0.85, 0.10, 0.05],
            [0.15, 0.70, 0.15],
            [0.05, 0.10, 0.85],
          ],
          means: [-0.03, 0.0, 0.03],
          stds: [0.012, 0.010, 0.012],
        },
        logLikelihood: -88,
        iterations: 4,
        converged: true,
      }),
      attachStudentTPredictiveEmissions: (
        _observations: number[],
        params: {
          nStates: number;
          pi: number[];
          A: number[][];
          means: number[];
          stds: number[];
        },
      ) => ({
        ...params,
        means: [0.03, -0.03, 0.0],
        studentTEmissions: [
          { degreesOfFreedom: 6, location: 0.03, scale: 0.012 },
          { degreesOfFreedom: 6, location: -0.03, scale: 0.012 },
          { degreesOfFreedom: 6, location: 0.0, scale: 0.010 },
        ],
      }),
      predict: () => ({
        currentState: 0,
        stateProbabilities: [],
        currentStateProbabilities: [0.70, 0.20, 0.10],
        forecastProbabilities: [0.10, 0.70, 0.20],
        expectedReturn: 0.012,
        expectedVolatility: 0.035,
      }),
    }));

    const returns = Array.from({ length: 120 }, (_, i) => [0.002, -0.003, 0.004, -0.002, 0.001][i % 5]);
    const prices = [100];
    for (const ret of returns) prices.push(prices[prices.length - 1] * Math.exp(ret));
    const currentPrice = prices[prices.length - 1];

    const result = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      enableSoftRegimeWeighting: true,
      enableStudentTEmission: true,
    });

    expect(result.metadata.softRegime?.currentRegimeMixture?.bull).toBeCloseTo(0.70, 8);
    expect(result.metadata.softRegime?.currentRegimeMixture?.bear).toBeCloseTo(0.20, 8);
    expect(result.metadata.softRegime?.currentRegimeMixture?.sideways).toBeCloseTo(0.10, 8);
    expect(result.metadata.softRegime?.forecastRegimeMixture?.bull).toBeCloseTo(0.10, 8);
    expect(result.metadata.softRegime?.forecastRegimeMixture?.bear).toBeCloseTo(0.70, 8);
    expect(result.metadata.softRegime?.forecastRegimeMixture?.sideways).toBeCloseTo(0.20, 8);
  });

  it('exposes ADWIN trim metadata when the historical window is mean-shifted', async () => {
    function makeRng(seed: number): () => number {
      let s = seed >>> 0;
      return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
      };
    }

    function randn(rng: () => number): number {
      const u1 = Math.max(1e-12, rng());
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function makePriceSeries(returns: number[], start = 100): number[] {
      const out = new Array(returns.length + 1);
      out[0] = start;
      for (let i = 0; i < returns.length; i++) out[i + 1] = out[i] * Math.exp(returns[i]);
      return out;
    }

    const rng = makeRng(7);
    const calm = Array.from({ length: 200 }, () => randn(rng) * 0.001);
    const shifted = Array.from({ length: 200 }, () => randn(rng) * 0.001 + 0.5);
    const prices = makePriceSeries([...calm, ...shifted]);

    const disabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const enabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      enableAdwinTrim: true,
      adwinDelta: 0.1,
    });

    expect(disabled.metadata.adwinTrim).toBeUndefined();
    expect(enabled.metadata.adwinTrim).toEqual({
      droppedPrices: expect.any(Number),
      keptPrices: expect.any(Number),
    });
    expect(enabled.metadata.historicalDays).toBeLessThan(disabled.metadata.historicalDays);
  });

  it('surfaces coefficient-clustering diagnostics only when enabled', async () => {
    function makeArSeries(phi: number, sigma: number, length: number): number[] {
      const series: number[] = [];
      let prev = 0.004;
      for (let i = 0; i < length; i++) {
        const noise = sigma * Math.sin((i + 1) * 1.73);
        const next = phi * prev + noise;
        series.push(next);
        prev = next;
      }
      return series;
    }

    function makePriceSeries(returns: number[], start = 100): number[] {
      const out = new Array(returns.length + 1);
      out[0] = start;
      for (let i = 0; i < returns.length; i++) out[i + 1] = out[i] * Math.exp(returns[i]);
      return out;
    }

    const prices = makePriceSeries([
      ...makeArSeries(0.85, 0.002, 80),
      ...makeArSeries(-0.72, 0.0025, 80),
      ...makeArSeries(0.2, 0.006, 80),
    ]);

    const disabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const enabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      enableCoefficientClustering: true,
    });

    expect(disabled.metadata.coefficientClustering).toBeUndefined();
    expect(enabled.metadata.coefficientClustering).toEqual({
      segmentCount: expect.any(Number),
      occupiedClusters: expect.any(Number),
      currentCluster: expect.any(Number),
      currentProbabilities: expect.any(Array),
      currentCoefficients: expect.any(Array),
      centroids: expect.any(Array),
      assignments: expect.any(Array),
    });
    expect(enabled.metadata.coefficientClustering!.occupiedClusters).toBeGreaterThanOrEqual(2);
    expect(
      enabled.metadata.coefficientClustering!.currentProbabilities.reduce((sum, value) => sum + value, 0),
    ).toBeCloseTo(1, 6);
  });

  it('exposes KSWIN trim metadata when the historical window has a variance shift', async () => {
    function makeRng(seed: number): () => number {
      let s = seed >>> 0;
      return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
      };
    }

    function randn(rng: () => number): number {
      const u1 = Math.max(1e-12, rng());
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function makePriceSeries(returns: number[], start = 100): number[] {
      const out = new Array(returns.length + 1);
      out[0] = start;
      for (let i = 0; i < returns.length; i++) out[i + 1] = out[i] * Math.exp(returns[i]);
      return out;
    }

    const rng = makeRng(17);
    const wild = Array.from({ length: 160 }, () => randn(rng) * 0.04);
    const calm = Array.from({ length: 160 }, () => randn(rng) * 0.002);
    const prices = makePriceSeries([...wild, ...calm]);

    const disabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const enabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      enableKswinTrim: true,
    });

    expect(disabled.metadata.kswinTrim).toBeUndefined();
    expect(enabled.metadata.kswinTrim).toEqual({
      droppedPrices: expect.any(Number),
      keptPrices: expect.any(Number),
      maxD: expect.any(Number),
      criticalD: expect.any(Number),
    });
  });

  it('exposes Hawkes amplification metadata when clustered jumps are present', async () => {
    function makeRng(seed: number): () => number {
      let s = seed >>> 0;
      return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
      };
    }

    function randn(rng: () => number): number {
      const u1 = Math.max(1e-12, rng());
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function makePriceSeries(returns: number[], start = 100): number[] {
      const out = new Array(returns.length + 1);
      out[0] = start;
      for (let i = 0; i < returns.length; i++) out[i + 1] = out[i] * Math.exp(returns[i]);
      return out;
    }

    const rng = makeRng(99);
    const returns: number[] = [];
    for (let i = 0; i < 600; i++) returns.push(randn(rng) * 0.01);
    for (let k = 0; k < 14; k++) returns[100 + k] += 0.14 * (rng() < 0.5 ? -1 : 1);
    for (let k = 0; k < 12; k++) returns[250 + k] += 0.12 * (rng() < 0.5 ? -1 : 1);
    for (let k = 0; k < 12; k++) returns[380 + k] += 0.14 * (rng() < 0.5 ? -1 : 1);
    const prices = makePriceSeries(returns);

    const disabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      trajectory: true,
      trajectoryDays: 7,
    });
    const enabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      trajectory: true,
      trajectoryDays: 7,
      enableHawkesIntensity: true,
      hawkesSigmaThreshold: 1.5,
    });

    expect(disabled.metadata.hawkes).toBeUndefined();
    expect(enabled.metadata.hawkes).toEqual({
      intensityMultiplier: expect.any(Number),
      branchingRatio: expect.anything(),
      jumpCount: expect.any(Number),
      endogenousJumpInjected: expect.any(Boolean),
    });
  });

  it('exposes cross-asset bias metadata only when trajectory drift is adjusted by peers', async () => {
    function makeRng(seed: number): () => number {
      let s = seed >>> 0;
      return () => {
        s = (s * 1664525 + 1013904223) >>> 0;
        return s / 0x100000000;
      };
    }

    function randn(rng: () => number): number {
      const u1 = Math.max(1e-12, rng());
      const u2 = rng();
      return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    function makePriceSeries(returns: number[], start = 100): number[] {
      const out = new Array(returns.length + 1);
      out[0] = start;
      for (let i = 0; i < returns.length; i++) out[i + 1] = out[i] * Math.exp(returns[i]);
      return out;
    }

    const rng = makeRng(11);
    const n = 220;
    const ethReturns = Array.from({ length: n }, () => randn(rng) * 0.02);
    const btcReturns = new Array(n).fill(0);
    for (let i = 0; i < n - 1; i++) {
      btcReturns[i + 1] = 0.4 * ethReturns[i] + 0.005 * randn(rng);
    }
    const prices = makePriceSeries(btcReturns);

    const disabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      trajectory: true,
      trajectoryDays: 7,
    });
    const enabled = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      trajectory: true,
      trajectoryDays: 7,
      enableCrossAssetBias: true,
      crossAssetReturns: { ETH: ethReturns },
      crossAssetLassoLambda: 0.001,
    });

    expect(disabled.metadata.crossAssetBias).toBeUndefined();
    expect(enabled.metadata.crossAssetBias).toEqual({
      perDayBias: expect.any(Number),
      tickers: ['ETH'],
      nonZeroCoefCount: expect.any(Number),
    });
  });

  it('keeps BTC 30-day off-window fallback candidates from enabling canonical emission', async () => {
    const now = FIXED_TEST_NOW_MS;
    const day = MS_PER_DAY;

    mock.module('./polymarket.js', () => ({
      ...realPolymarketModule,
      fetchPolymarketMarkets: async () => [],
      fetchPolymarketAnchorMarkets: async () => [
        {
          question: 'Will the price of Bitcoin be above $84,000 on April 24?',
          probability: 0.50,
          volume24h: 30000,
          ageDays: 10,
          endDate: new Date(now + 6 * day).toISOString(),
        },
      ],
    }));

    const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }

    const result = await freshTool.func({
      ticker: 'BTC-USD',
      horizon: 30,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    const diagnostics = parsed.data.canonical?.diagnostics;
    expect(parsed.data.status).toBe('abstain');
    expect(diagnostics?.totalAnchors).toBeGreaterThan(0);
    expect(diagnostics?.trustedAnchors).toBe(0);
    expect(diagnostics?.anchorQuality).toBe('none');
    expect(diagnostics?.canEmitCanonical).toBe(false);
    expect(parsed.data.distribution).toBeNull();
  });

  it('uses undated fallback only after date-windowed front slice and retry queries are exhausted', async () => {
    const callOptions: Array<{ queries: string[]; endDateFilter?: { end_date_min: string; end_date_max: string } }> = [];
    mock.module('./polymarket.js', () => ({
      ...realPolymarketModule,
      fetchPolymarketMarkets: async () => [],
      fetchPolymarketAnchorMarkets: async () => [],
      fetchPolymarketAnchorMarketsWithQueries: async (
        queries: string[],
        _limit: number,
        options: { ticker: string; horizonDays?: number; endDateFilter?: { end_date_min: string; end_date_max: string } },
      ) => {
        callOptions.push({ queries: [...queries], endDateFilter: options.endDateFilter });
        if (options.endDateFilter) {
          return [];
        }
        expect(queries).toEqual([
          'Bitcoin price',
          'Bitcoin',
          'Bitcoin above',
          'Bitcoin below',
          'Bitcoin ETF',
          'crypto ETF',
        ]);
        return [
          {
            question: 'Will the price of Bitcoin be above $76,000 on April 24?',
            probability: 0.5,
            volume24h: 5000,
            ageDays: 0,
            endDate: '2026-04-24',
          },
        ];
      },
    }));

    const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }

    const result = await freshTool.func({
      ticker: 'BTC-USD',
      horizon: 30,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    const diagnostics = parsed.data.canonical?.diagnostics;
    expect(callOptions).toHaveLength(2);
    expect(callOptions[0].endDateFilter).toEqual({
      end_date_min: expect.any(String),
      end_date_max: expect.any(String),
    });
    expect(callOptions[1].endDateFilter).toBeUndefined();
    expect(parsed.data.status).toBe('abstain');
    expect(diagnostics?.totalAnchors).toBe(1);
    expect(diagnostics?.trustedAnchors).toBe(0);
  });

  it('BTC 14d uses undated fallback when date-windowed queries return empty', async () => {
    const frontCalls: Array<{ query: string; endDateFilter?: { end_date_min: string; end_date_max: string } }> = [];
    const fallbackCalls: Array<{ queries: string[]; endDateFilter?: { end_date_min: string; end_date_max: string } }> = [];
    const targetMonth = new Date(FIXED_TEST_NOW_MS + 14 * MS_PER_DAY)
      .toLocaleString('en-US', { month: 'long' });

    mock.module('./polymarket.js', () => ({
      ...realPolymarketModule,
      fetchPolymarketMarkets: async () => [],

      fetchPolymarketAnchorMarkets: async (
        query: string,
        _limit: number,
        options: { ticker: string; horizonDays?: number; endDateFilter?: { end_date_min: string; end_date_max: string } },
      ) => {
        frontCalls.push({ query, endDateFilter: options.endDateFilter });
        return [];
      },

      fetchPolymarketAnchorMarketsWithQueries: async (
        queries: string[],
        _limit: number,
        options: { ticker: string; horizonDays?: number; endDateFilter?: { end_date_min: string; end_date_max: string } },
      ) => {
        fallbackCalls.push({ queries: [...queries], endDateFilter: options.endDateFilter });
        if (options.endDateFilter) {
          return [];
        }
        return [
          {
            question: 'Will the price of Bitcoin be above $76,000 on June 1?',
            probability: 0.52,
            volume24h: 8000,
            ageDays: 0,
            endDate: '2026-06-01',
          },
        ];
      },
    }));

    const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }

    const result = await freshTool.func({
      ticker: 'BTC-USD',
      horizon: 14,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    const diagnostics = parsed.data.canonical?.diagnostics;

    expect(frontCalls).toHaveLength(6);
    expect(frontCalls.map((call) => call.query)).toEqual([
      'Bitcoin price',
      'Bitcoin',
      'Bitcoin above',
      'Bitcoin below',
      `Bitcoin ${targetMonth}`,
      `Bitcoin above ${targetMonth}`,
    ]);
    expect(frontCalls.every((call) => call.endDateFilter?.end_date_min && call.endDateFilter?.end_date_max)).toBe(true);

    expect(fallbackCalls.length).toBeGreaterThanOrEqual(1);
    const undatedFallbackCall = fallbackCalls.find((call) => call.endDateFilter === undefined);
    expect(undatedFallbackCall).toBeDefined();
    expect(undatedFallbackCall!.queries).toEqual([
      'Bitcoin price',
      'Bitcoin',
      'Bitcoin above',
      'Bitcoin below',
      `Bitcoin ${targetMonth}`,
      `Bitcoin above ${targetMonth}`,
    ]);
    expect(parsed.data.status).toBe('ok');
    expect(diagnostics?.totalAnchors).toBe(1);
    expect(diagnostics?.trustedAnchors).toBe(0);
  });

  it('returns canonical payload when trusted anchors and positive validation are present', async () => {
    const prices: number[] = [];
    let p = 100;
    for (let i = 0; i < 91; i++) {
      p *= 1 + Math.sin(i * 0.15) * 0.006;
      prices.push(Math.round(p * 100) / 100);
    }
    const currentPrice = prices[prices.length - 1];
    const result = await markovDistributionTool.func({
      ticker: 'TEST',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [
        { question: `Will TEST be above $${Math.round(currentPrice * 0.97)} on April 9?`, probability: 0.72, volume: 5000, createdAt: FIXED_TEST_NOW_MS - MS_PER_DAY * 5 },
        { question: `Will TEST be above $${Math.round(currentPrice)} on April 9?`, probability: 0.50, volume: 5000, createdAt: FIXED_TEST_NOW_MS - MS_PER_DAY * 5 },
        { question: `Will TEST be above $${Math.round(currentPrice * 1.03)} on April 9?`, probability: 0.28, volume: 5000, createdAt: FIXED_TEST_NOW_MS - MS_PER_DAY * 5 },
      ],
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    expect(parsed.data.status).toBe('ok');
    expect(parsed.data.manualSynthesisForbidden).toBe(false);
    expect(parsed.data.abstainReasons).toEqual([]);
    expect(parsed.data.canonical.scenarios).toBeDefined();
    expect(parsed.data.canonical.actionSignal).toBeDefined();
    expect(parsed.data.canonical.diagnostics.canEmitCanonical).toBe(true);
    expect(parsed.data.canonical.diagnostics).toHaveProperty('recommendationProvenance');
    expect(parsed.data.canonical.diagnostics.decisionSurface).toBe('calibrated');
    expect(parsed.data.canonical.diagnostics.trajectoryInterpretation).toBeNull();
    expect(parsed.data.report).toContain('Latent regime:');
    expect(Array.isArray(parsed.data.distribution)).toBe(true);
    expect(Array.isArray(parsed.data.decisionDistribution)).toBe(true);
  });

  it('labels trajectory output as path context while keeping canonical decision metadata explicit', async () => {
    const prices: number[] = [];
    let p = 100;
    for (let i = 0; i < 91; i++) {
      p *= 1 + Math.sin(i * 0.15) * 0.006;
      prices.push(Math.round(p * 100) / 100);
    }
    const currentPrice = prices[prices.length - 1];
    const result = await markovDistributionTool.func({
      ticker: 'TEST_TRAJECTORY',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [
        { question: `Will TEST_TRAJECTORY be above $${Math.round(currentPrice * 0.97)} on April 9?`, probability: 0.72, volume: 5000, createdAt: FIXED_TEST_NOW_MS - MS_PER_DAY * 5 },
        { question: `Will TEST_TRAJECTORY be above $${Math.round(currentPrice)} on April 9?`, probability: 0.50, volume: 5000, createdAt: FIXED_TEST_NOW_MS - MS_PER_DAY * 5 },
        { question: `Will TEST_TRAJECTORY be above $${Math.round(currentPrice * 1.03)} on April 9?`, probability: 0.28, volume: 5000, createdAt: FIXED_TEST_NOW_MS - MS_PER_DAY * 5 },
      ],
      trajectory: true,
      trajectoryDays: 7,
    });

    const parsed = JSON.parse(result);
    expect(parsed.data.report).toContain('DAY PATH CONTEXT TRAJECTORY');
    expect(parsed.data.report).toContain('Path context only');
    expect(parsed.data.canonical.diagnostics.decisionSurface).toBe('calibrated');
    expect(parsed.data.canonical.diagnostics.trajectoryInterpretation).toBe('path_context');
    expect(Array.isArray(parsed.data.decisionDistribution)).toBe(true);
    expect(Array.isArray(parsed.data.trajectory)).toBe(true);
  });

  it('emits context-only output when short-horizon BTC has good anchors but validation is unavailable', async () => {
    const now = Date.parse('2026-05-08T12:00:00Z');
    setSystemTime(new Date(now));
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 30; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }
    const currentPrice = prices[prices.length - 1];

    const result = await markovDistributionTool.func({
      ticker: 'BTC-USD',
      horizon: 3,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [
        { question: `Will the price of Bitcoin be above $${Math.round(currentPrice * 0.99)} on May 11?`, probability: 0.62, volume: 100_000, createdAt: now - 7 * MS_PER_DAY, endDate: new Date(now + 3 * MS_PER_DAY).toISOString() },
        { question: `Will the price of Bitcoin be above $${Math.round(currentPrice)} on May 11?`, probability: 0.50, volume: 100_000, createdAt: now - 7 * MS_PER_DAY, endDate: new Date(now + 3 * MS_PER_DAY).toISOString() },
        { question: `Will the price of Bitcoin be above $${Math.round(currentPrice * 1.01)} on May 11?`, probability: 0.39, volume: 100_000, createdAt: now - 7 * MS_PER_DAY, endDate: new Date(now + 3 * MS_PER_DAY).toISOString() },
      ],
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    expect(parsed.data.status).toBe('ok');
    expect(parsed.data.manualSynthesisForbidden).toBe(false);
    expect(parsed.data.abstainReasons).toEqual([]);
    expect(parsed.data.contextOnlyReasons).toEqual([
      'Out-of-sample Markov validation is unavailable, so this anchored crypto forecast is emitted as context only.',
    ]);
    expect(parsed.data.forecastHint).toBeNull();
    expect(parsed.data.canonical.scenarios).toBeNull();
    expect(parsed.data.canonical.actionSignal).toBeNull();
    expect(parsed.data.canonical.diagnostics.canEmitCanonical).toBe(true);
    expect(parsed.data.canonical.diagnostics.status).toBe('context_only');
    expect(parsed.data.canonical.diagnostics.calibrationMode).toBe('context_only');
    expect(parsed.data.canonical.diagnostics.contextOnlyReasons).toEqual(parsed.data.contextOnlyReasons);
    expect(Array.isArray(parsed.data.distribution)).toBe(true);
    expect(parsed.data.report).toContain('🟡 Context-only Markov output');
    expect(parsed.data.report).toContain('validation is unavailable');
    expect(parsed.data.report).not.toContain('┌─ Your Options');
  });

  it('uses horizon-return validation when short-horizon crypto has enough history', async () => {
    const prices: number[] = [];
    let p = 65000;
    for (let i = 0; i < 120; i++) {
      p *= 1 + Math.sin(i * 0.12) * 0.004;
      prices.push(Math.round(p * 100) / 100);
    }
    const currentPrice = prices[prices.length - 1];
    const result = await markovDistributionTool.func({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [
        { question: `Will the price of Bitcoin be above $${Math.round(currentPrice * 0.97)} on April 9?`, probability: 0.72, volume: 5000, createdAt: FIXED_TEST_NOW_MS - MS_PER_DAY * 3 },
        { question: `Will the price of Bitcoin be above $${Math.round(currentPrice)} on April 9?`, probability: 0.51, volume: 5000, createdAt: FIXED_TEST_NOW_MS - MS_PER_DAY * 3 },
        { question: `Will the price of Bitcoin be above $${Math.round(currentPrice * 1.03)} on April 9?`, probability: 0.29, volume: 5000, createdAt: FIXED_TEST_NOW_MS - MS_PER_DAY * 3 },
      ],
      trajectory: false,
    });

    const parsed = JSON.parse(result);
    expect(parsed.data.canonical.diagnostics.anchorQuality).toBe('good');
    expect(parsed.data.canonical.diagnostics.trustedAnchors).toBeGreaterThanOrEqual(2);
    expect(parsed.data.canonical.diagnostics.outOfSampleR2).not.toBeNull();
    expect(parsed.data.status).toBe('ok');
  });

  // -----------------------------------------------------------------------
  // Commodity model-only bypass tests
  // -----------------------------------------------------------------------
  describe('commodity model-only bypass', () => {
    it('emits canonical for commodity with zero anchors when thresholds pass', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);

      const prices: number[] = [];
      let p = 180;
      for (let i = 0; i < 120; i++) {
        p *= 1.001 + Math.sin(i * 0.1) * 0.001;
        prices.push(Math.round(p * 100) / 100);
      }

      const result = await freshTool.func({
        ticker: 'GLD',
        horizon: 7,
        currentPrice: prices[prices.length - 1],
        historicalPrices: prices,
        polymarketMarkets: [],
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.status).toBe('ok');
      expect(parsed.data.canonical.diagnostics.canEmitCanonical).toBe(true);
      expect(parsed.data.canonical.diagnostics.calibrationMode).toBe('model_only');
      expect(parsed.data.canonical.diagnostics.anchorBypassApplied).toBe(true);
      expect(parsed.data.canonical.diagnostics.totalAnchors).toBe(0);
      expect(parsed.data.canonical.diagnostics.trustedAnchors).toBe(0);
    });

    it('abstains for commodity when R² is too low', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);

      // Deterministic oscillating series that produces strongly negative R² without a structural break.
      const prices: number[] = [];
      let p = 200;
      for (let i = 0; i < 95; i++) {
        const shock = Math.sin(i * 0.15) * 0.004;
        p *= 1 + shock;
        prices.push(Math.round(p * 100) / 100);
      }

      const result = await freshTool.func({
        ticker: 'GLD',
        horizon: 7,
        currentPrice: prices[prices.length - 1],
        historicalPrices: prices,
        polymarketMarkets: [],
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.status).toBe('abstain');
      expect(parsed.data.canonical.diagnostics.canEmitCanonical).toBe(false);
    });

    it('abstains for commodity when confidence is too low', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);

      // Very short series (25 prices) to get low predictionConfidence
      const prices: number[] = [];
      let p = 200;
      for (let i = 0; i < 25; i++) {
        p *= 1 + Math.sin(i * 0.3) * 0.005;
        prices.push(Math.round(p * 100) / 100);
      }

      const result = await freshTool.func({
        ticker: 'GLD',
        horizon: 7,
        currentPrice: prices[prices.length - 1],
        historicalPrices: prices,
        polymarketMarkets: [],
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.status).toBe('abstain');
      expect(parsed.data.canonical.diagnostics.canEmitCanonical).toBe(false);
    });

    it('emits commodity model-only distribution despite structural break', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);

      // Generate prices with sharp break after index 85
      const prices: number[] = [];
      let p = 200;
      for (let i = 0; i < 100; i++) {
        const shock = i > 85 ? 0.05 : Math.sin(i * 0.12) * 0.004;
        p *= 1 + shock;
        prices.push(Math.round(p * 100) / 100);
      }

      const result = await freshTool.func({
        ticker: 'GLD',
        horizon: 7,
        currentPrice: prices[prices.length - 1],
        historicalPrices: prices,
        polymarketMarkets: [],
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.status).toBe('ok');
      expect(parsed.data.canonical.diagnostics.structuralBreakDetected).toBe(true);
    });

    it('exposes calibrationMode and anchorBypassApplied in diagnostics', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);

      const prices: number[] = [];
      let p = 180;
      for (let i = 0; i < 120; i++) {
        p *= 1.001 + Math.sin(i * 0.1) * 0.001;
        prices.push(Math.round(p * 100) / 100);
      }

      const result = await freshTool.func({
        ticker: 'GLD',
        horizon: 7,
        currentPrice: prices[prices.length - 1],
        historicalPrices: prices,
        polymarketMarkets: [],
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.status).toBe('ok');
      expect(parsed.data.canonical.diagnostics.calibrationMode).toBe('model_only');
      expect(parsed.data.canonical.diagnostics.anchorBypassApplied).toBe(true);
    });

    it('uses markovWeight=1 and anchorWeight=0 for model-only emission', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);

      const prices: number[] = [];
      let p = 180;
      for (let i = 0; i < 120; i++) {
        p *= 1.001 + Math.sin(i * 0.1) * 0.001;
        prices.push(Math.round(p * 100) / 100);
      }

      const result = await freshTool.func({
        ticker: 'GLD',
        horizon: 7,
        currentPrice: prices[prices.length - 1],
        historicalPrices: prices,
        polymarketMarkets: [],
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.status).toBe('ok');
      expect(parsed.data.canonical.diagnostics.markovWeight).toBe(1);
      expect(parsed.data.canonical.diagnostics.anchorWeight).toBe(0);
    });

    it('auto-fetches the 252d GOLD live-policy history for GLD short horizons', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      mock.module('./api.js', () => ({
        ...realApiModule,
        api: {
          ...realApiModule.api,
          get: async (_path: string, params: Record<string, string>) => {
            const start = new Date(params.start_date).getTime();
            const end = new Date(params.end_date).getTime();
            const requestedDays = Math.round((end - start) / MS_PER_DAY);
            const length = requestedDays >= 200 ? 252 : 120;
            let price = 220;
            const prices = Array.from({ length }, (_, i) => {
              price *= 1.0006 + Math.sin(i * 0.05) * 0.0008;
              return { close: Math.round(price * 100) / 100 };
            });
            return { data: { prices } };
          },
        },
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);

      const result = await freshTool.func({
        ticker: 'GLD',
        horizon: 7,
        polymarketMarkets: [],
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.canonical.diagnostics.historicalDays).toBe(251);
      expect(parsed.data.canonical.diagnostics.goldShortHorizonLivePolicy).toEqual({
        historyDays: 252,
        breakDivergenceThreshold: 0.15,
        rerunOnBreak: false,
      });
      expect(parsed.data.report).toContain('GOLD short-horizon live policy used 252d history with structural-break threshold 0.15');
    });

    it('detects lower-vol GLD structural breaks on 1d/2d/3d horizons behind the GOLD seam', async () => {
      function makeRng(seed: number): () => number {
        let s = seed >>> 0;
        return () => {
          s = (s * 1664525 + 1013904223) >>> 0;
          return s / 0x100000000;
        };
      }

      function randn(rng: () => number): number {
        const u1 = Math.max(1e-12, rng());
        const u2 = rng();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }

      const rng = makeRng(178);
      const returns: number[] = [];
      let prevA = 0;
      for (let i = 0; i < 126; i++) {
        const next = 0.00025 + 0.45 * prevA + randn(rng) * 0.0009;
        returns.push(next);
        prevA = next;
      }
      let prevB = 0;
      for (let i = 0; i < 126; i++) {
        const next = -0.00015 - 0.30 * prevB + randn(rng) * 0.0010;
        returns.push(next);
        prevB = next;
      }
      const historicalPrices = [250];
      for (const ret of returns) {
        historicalPrices.push(historicalPrices[historicalPrices.length - 1] * Math.exp(ret));
      }
      const currentPrice = historicalPrices[historicalPrices.length - 1];

      for (const horizon of [1, 2, 3]) {
        const result = await computeMarkovDistribution({
          ticker: 'GLD',
          horizon,
          currentPrice,
          historicalPrices,
          polymarketMarkets: [],
        });

        expect(result.metadata.structuralBreakDivergence).toBeGreaterThan(0.12);
        expect(result.metadata.structuralBreakDetected).toBe(true);
        expect(getGoldShortHorizonLivePolicy('GLD', horizon)?.breakDivergenceThreshold).toBeLessThan(0.15);
      }
    });

    it('uses divergence-weighted confidence for trending GLD 1d/2d/3d breaks without enabling fallback matrices', async () => {
      function makeRng(seed: number): () => number {
        let s = seed >>> 0;
        return () => {
          s = (s * 1664525 + 1013904223) >>> 0;
          return s / 0x100000000;
        };
      }

      function randn(rng: () => number): number {
        const u1 = Math.max(1e-12, rng());
        const u2 = rng();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }

      const rng = makeRng(28);
      const returns: number[] = [];
      let prevA = 0;
      for (let i = 0; i < 126; i++) {
        const next = 0.0002 + 0.5 * prevA + randn(rng) * 0.0008;
        returns.push(next);
        prevA = next;
      }
      let prevB = 0;
      for (let i = 0; i < 126; i++) {
        const next = -0.0001 + 0.2 * prevB + randn(rng) * 0.0009;
        returns.push(next);
        prevB = next;
      }
      const historicalPrices = [250];
      for (const ret of returns) {
        historicalPrices.push(historicalPrices[historicalPrices.length - 1] * Math.exp(ret));
      }
      const currentPrice = historicalPrices[historicalPrices.length - 1];

      for (const horizon of [1, 2, 3]) {
        const result = await computeMarkovDistribution({
          ticker: 'GLD',
          horizon,
          currentPrice,
          historicalPrices,
          polymarketMarkets: [],
        });

        expect(result.metadata.structuralBreakDetected).toBe(true);
        expect(result.metadata.divergenceWeightedBreakConfidenceActive).toBe(true);
        expect(result.metadata.confidence?.multipliers.structuralBreak).toBeCloseTo(0.7, 10);
        expect(result.metadata.breakFallbackCandidateId ?? null).toBeNull();
        expect(result.metadata.breakFallbackMode ?? null).toBeNull();
      }
    });

    it('keeps ultra-short GOLD behavior gated to canonical GLD tickers', async () => {
      function makeRng(seed: number): () => number {
        let s = seed >>> 0;
        return () => {
          s = (s * 1664525 + 1013904223) >>> 0;
          return s / 0x100000000;
        };
      }

      function randn(rng: () => number): number {
        const u1 = Math.max(1e-12, rng());
        const u2 = rng();
        return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
      }

      const rng = makeRng(912);
      const returns: number[] = [];
      let prevA = 0;
      for (let i = 0; i < 126; i++) {
        const next = 0.00025 + 0.45 * prevA + randn(rng) * 0.0009;
        returns.push(next);
        prevA = next;
      }
      let prevB = 0;
      for (let i = 0; i < 126; i++) {
        const next = -0.00015 - 0.30 * prevB + randn(rng) * 0.0010;
        returns.push(next);
        prevB = next;
      }
      const historicalPrices = [250];
      for (const ret of returns) {
        historicalPrices.push(historicalPrices[historicalPrices.length - 1] * Math.exp(ret));
      }
      const currentPrice = historicalPrices[historicalPrices.length - 1];

      for (const ticker of ['GLD', 'XAUUSD']) {
        const result = await computeMarkovDistribution({
          ticker,
          horizon: 2,
          currentPrice,
          historicalPrices,
          polymarketMarkets: [],
        });

        expect(result.metadata.structuralBreakDetected).toBe(true);
        expect(result.metadata.divergenceWeightedBreakConfidenceActive).toBe(true);
      }

      const btcResult = await computeMarkovDistribution({
        ticker: 'BTC-USD',
        horizon: 2,
        currentPrice,
        historicalPrices,
        polymarketMarkets: [],
      });

      expect(btcResult.metadata.divergenceWeightedBreakConfidenceActive).toBeUndefined();
    });

    it('auto-fetches the 252d BTC live-policy history for short horizons', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      mock.module('./api.js', () => ({
        ...realApiModule,
        api: {
          ...realApiModule.api,
          get: async (_path: string, params: Record<string, string>) => {
            const start = new Date(params.start_date).getTime();
            const end = new Date(params.end_date).getTime();
            const requestedDays = Math.round((end - start) / MS_PER_DAY);
            const length = requestedDays >= 200 ? 252 : 120;
            let price = 60000;
            const prices = Array.from({ length }, (_, i) => {
              price *= 1.001 + Math.sin(i * 0.07) * 0.0015;
              return { close: Math.round(price * 100) / 100 };
            });
            return { data: { prices } };
          },
        },
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);

      const result = await freshTool.func({
        ticker: 'BTC-USD',
        horizon: 2,
        polymarketMarkets: [],
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.canonical.diagnostics.historicalDays).toBe(251);
      expect(parsed.data.canonical.diagnostics.btcShortHorizonLivePolicy).toEqual({
        historyDays: 252,
        breakDivergenceThreshold: 0.15,
        rerunOnBreak: true,
        rerunWindowDays: 120,
      });
      expect(parsed.data.report).toContain('BTC short-horizon live policy used 252d history with structural-break threshold 0.15');
    });

    it('reruns BTC 1d live mode on the last 60d after a structural break and records provenance', async () => {
      mock.module('./polymarket.js', () => ({
        ...realPolymarketModule,
        fetchPolymarketMarkets: async () => [],
        fetchPolymarketAnchorMarkets: async () => [],
      }));

      const { markovDistributionTool: freshTool } = await import(`./markov-distribution.js?t=${nextTestId('module')}`);

      const prices: number[] = [];
      let p = 65000;
      prices.push(p);
      for (let i = 1; i < 252; i++) {
        const shock = i < 200 ? (i % 2 === 0 ? 0.02 : -0.02) : 0.02;
        p *= 1 + shock;
        prices.push(Math.round(p * 100) / 100);
      }

      const result = await freshTool.func({
        ticker: 'BTC-USD',
        horizon: 1,
        currentPrice: prices[prices.length - 1],
        historicalPrices: prices,
        polymarketMarkets: [],
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.canonical.diagnostics.historicalDays).toBe(59);
      expect(parsed.data.canonical.diagnostics.btcShortHorizonLivePolicy).toEqual({
        historyDays: 252,
        breakDivergenceThreshold: 0.10,
        rerunOnBreak: true,
        rerunWindowDays: 60,
      });
      expect(parsed.data.canonical.diagnostics.btcShortHorizonRerunTriggered).toBe(true);
      expect(parsed.data.canonical.diagnostics.originalHistoricalDays).toBe(251);
      expect(parsed.data.canonical.diagnostics.originalStructuralBreakDetected).toBe(true);
      expect(parsed.data.canonical.diagnostics.originalStructuralBreakDivergence).toBeGreaterThan(0.10);
      expect(parsed.data.report).toContain('reran on the last 60d');
    });

    it('keeps anchored BTC abstain reports free of GOLD proxy metadata and commodity bypass wording', async () => {
      const now = Date.parse('2026-05-08T12:00:00Z');
      setSystemTime(new Date(now));
      const prices: number[] = [];
      let p = 65000;
      for (let i = 0; i < 95; i++) {
        const shock = Math.sin(i * 0.15) * 0.004;
        p *= 1 + shock;
        prices.push(Math.round(p * 100) / 100);
      }

      const currentPrice = prices[prices.length - 1];
      const result = await markovDistributionTool.func({
        ticker: 'BTC-USD',
        horizon: 1,
        currentPrice,
        historicalPrices: prices,
        polymarketMarkets: [
          {
            question: `Will the price of Bitcoin be above $${Math.round(currentPrice * 0.99)} on May 9?`,
            probability: 0.61,
            volume: 120_000,
            createdAt: now - 7 * MS_PER_DAY,
            endDate: new Date(now + MS_PER_DAY).toISOString(),
          },
          {
            question: `Will the price of Bitcoin be above $${Math.round(currentPrice * 1.01)} on May 9?`,
            probability: 0.42,
            volume: 110_000,
            createdAt: now - 7 * MS_PER_DAY,
            endDate: new Date(now + MS_PER_DAY).toISOString(),
          },
          {
            question: `Will the price of Bitcoin fall below $${Math.round(currentPrice * 0.98)} on May 9?`,
            probability: 0.28,
            volume: 90_000,
            createdAt: now - 7 * MS_PER_DAY,
            endDate: new Date(now + MS_PER_DAY).toISOString(),
          },
        ],
        trajectory: false,
      });

      const parsed = JSON.parse(result);
      expect(parsed.data.status).toBe('abstain');
      expect(parsed.data.canonical.diagnostics.trustedAnchors).toBe(3);
      expect(parsed.data.canonical.diagnostics.anchorBypassApplied).toBe(false);
      expect(parsed.data.canonical.diagnostics.calibrationMode).toBe('anchored');
      expect(parsed.data.canonical.diagnostics.goldShortHorizonLivePolicy).toBeNull();
      expect(parsed.data.report).toContain('BTC short-horizon live policy used 252d history');
      expect(parsed.data.report).not.toContain('GOLD short-horizon live policy');
      expect(parsed.data.report).not.toContain('commodity bypass');
      expect(parsed.data.report).not.toContain('model-only commodity emission');
      expect(parsed.data.report).not.toContain('GLD');
      expect(parsed.data.report).not.toContain('gold');
    });
  });
});
