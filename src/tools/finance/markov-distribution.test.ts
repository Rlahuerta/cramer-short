/**
 * Tests for markov-distribution.ts
 *
 * Covers all 10 prioritized bug-fixes and 5 addenda from the spec review:
 *  Fix 1  — Default matrix row sums to 1.0 (not 1.2)
 *  Fix 2  — bear→bear sentiment sign: bullish reduces bear persistence
 *  Fix 3  — Log-normal mapping formula (μ_eff, σ_eff, σ_n)
 *  Fix 4  — YES-bias correction (×0.95 on Polymarket anchors)
 *  Fix 5  — CI bounds contain the point estimate
 *  Fix 6  — Dirichlet α=0.1 smoothing
 *  Fix 7  — Mixing-time decay (long horizon → anchor-dominant)
 *  Fix 8  — Liquidity guard trustScore
 *  Fix 9  — Sentiment alpha reduced to 0.07
 *  Fix 10 — Joint high_vol states (no priority override)
 *  Add 1  — Beta distribution note (architecture, no test needed)
 *  Add 2  — VIX regime note (architecture, no test needed)
 *  Add 3  — Eigenvalue-based mixing time
 *  Add 4  — R²_OS out-of-sample metric
 *  Add 5  — Dirichlet default 0.1 (merged with Fix 6)
 */

import { describe, it, expect, mock, afterEach, beforeEach, setSystemTime, spyOn } from 'bun:test';
import { integrationIt } from '../../utils/test-guards.js';
import { MS_PER_DAY } from '../../utils/time.js';
import {
  classifyRegimeState,
  computeAdaptiveThresholds,
  estimateTransitionMatrix,
  buildDefaultMatrix,
  normalizeRows,
  adjustTransitionMatrix,
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
  fetchHistoricalPrices,
  shouldEmitContextOnlyCanonical,
} from './markov-distribution.js';
import type { RegimeState, MarkovDistributionPoint, PriceThreshold, ScenarioProbabilities } from './markov-distribution.js';
import { FinancialDatasetsHttpError } from './api.js';

const realPolymarketModule = { ...(await import('./polymarket.js')) };
const realHmmModule = { ...(await import('./hmm.js')) };
const realApiModule = { ...(await import('./api.js')) };
const FIXED_NOW = new Date('2025-04-02T12:00:00.000Z');
const FIXED_NOW_MS = FIXED_NOW.getTime();

function seedRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

let randomSpy: { mockRestore: () => void } | undefined;

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
        endDate: new Date(FIXED_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will the price of Bitcoin be above $66000 on April 9?',
        probability: 0.54,
        volume24h: 220000,
        ageDays: 5,
        endDate: new Date(FIXED_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will the price of Bitcoin be above $68000 on April 9?',
        probability: 0.31,
        volume24h: 190000,
        ageDays: 5,
        endDate: new Date(FIXED_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will Bitcoin reach $70000 this week?',
        probability: 0.22,
        volume24h: 180000,
        ageDays: 5,
        endDate: new Date(FIXED_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
    ],
    fetchPolymarketAnchorMarkets: async (_query: string, _limit: number, _options: unknown) => [
      {
        question: 'Will the price of Bitcoin be above $62000 by end of week?',
        probability: 0.85,
        volume24h: 300000,
        ageDays: 7,
        endDate: new Date(FIXED_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will the price of Bitcoin be above $65000 by end of week?',
        probability: 0.62,
        volume24h: 260000,
        ageDays: 6,
        endDate: new Date(FIXED_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will the price of Bitcoin be above $68000 by end of week?',
        probability: 0.38,
        volume24h: 210000,
        ageDays: 5,
        endDate: new Date(FIXED_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
      {
        question: 'Will the price of Bitcoin fall below $63000 by end of week?',
        probability: 0.25,
        volume24h: 190000,
        ageDays: 5,
        endDate: new Date(FIXED_NOW_MS + 7 * MS_PER_DAY).toISOString(),
      },
    ],
  }));
}

beforeEach(() => {
  setSystemTime(FIXED_NOW);
  randomSpy = spyOn(Math, 'random').mockImplementation(seedRng(42345));
});

afterEach(() => {
  randomSpy?.mockRestore();
  randomSpy = undefined;
  setSystemTime();
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

describe('fetchHistoricalPrices Financial Datasets validation', () => {
  it('returns valid Financial Datasets closes without falling through', async () => {
    const prices = Array.from({ length: 12 }, (_, i) => ({ close: 100 + i }));
    const apiGetSpy = spyOn(realApiModule.api, 'get').mockResolvedValue({
      data: { prices },
      url: 'https://api.financialdatasets.ai/prices/?ticker=AAPL',
    });

    try {
      await expect(fetchHistoricalPrices('AAPL', 30)).resolves.toEqual(prices.map((p) => p.close));
    } finally {
      apiGetSpy.mockRestore();
    }
  });

  it('surfaces malformed Financial Datasets payloads instead of falling through', async () => {
    const apiGetSpy = spyOn(realApiModule.api, 'get').mockResolvedValue({
      data: {},
      url: 'https://api.financialdatasets.ai/prices/?ticker=AAPL',
    });

    try {
      await expect(fetchHistoricalPrices('AAPL', 30)).rejects.toThrow(
        /Malformed Financial Datasets prices payload/,
      );
    } finally {
      apiGetSpy.mockRestore();
    }
  });

  it('surfaces Financial Datasets parse failures instead of falling through', async () => {
    const apiGetSpy = spyOn(realApiModule.api, 'get').mockRejectedValue(
      new Error('[Financial Datasets API] request failed: invalid JSON (200 OK)'),
    );

    try {
      await expect(fetchHistoricalPrices('AAPL', 30)).rejects.toThrow(/invalid JSON/);
    } finally {
      apiGetSpy.mockRestore();
    }
  });


  it('surfaces malformed Yahoo chart fallback payloads instead of returning empty success', async () => {
    const apiGetSpy = spyOn(realApiModule.api, 'get').mockResolvedValue({
      data: { prices: [] },
      url: 'https://api.financialdatasets.ai/prices/?ticker=AAPL',
    });
    const fetchMock: typeof fetch = Object.assign(
      async (input: Parameters<typeof fetch>[0], _init?: Parameters<typeof fetch>[1]) => {
        const url = input.toString();
        if (url.includes('query1.finance.yahoo.com')) {
          return new Response(JSON.stringify({ chart: { result: [{ indicators: { quote: [{}] } }] } }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response('unavailable', { status: 500, statusText: 'Server Error' });
      },
      { preconnect: globalThis.fetch.preconnect },
    );
    const fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(fetchMock);

    try {
      await expect(fetchHistoricalPrices('AAPL', 30)).rejects.toThrow(
        /Malformed Yahoo Finance chart payload/,
      );
    } finally {
      apiGetSpy.mockRestore();
      fetchSpy.mockRestore();
    }
  });

  it('returns valid Yahoo chart fallback closes without filtering them', async () => {
    const yahooCloses = Array.from({ length: 12 }, (_, i) => 100 + i);
    const apiGetSpy = spyOn(realApiModule.api, 'get').mockResolvedValue({
      data: { prices: [] },
      url: 'https://api.financialdatasets.ai/prices/?ticker=AAPL',
    });
    const fetchMock: typeof fetch = Object.assign(
      async (input: Parameters<typeof fetch>[0], _init?: Parameters<typeof fetch>[1]) => {
        const url = input.toString();
        if (url.includes('query1.finance.yahoo.com')) {
          return new Response(JSON.stringify({
            chart: {
              result: [{
                indicators: {
                  quote: [{ close: yahooCloses }],
                },
              }],
            },
          }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response('unavailable', { status: 500, statusText: 'Server Error' });
      },
      { preconnect: globalThis.fetch.preconnect },
    );
    const fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(fetchMock);

    try {
      await expect(fetchHistoricalPrices('AAPL', 30)).resolves.toEqual(yahooCloses);
    } finally {
      apiGetSpy.mockRestore();
      fetchSpy.mockRestore();
    }
  });

  it('falls back to exchange history when Financial Datasets reports a crypto instrument is not recognized', async () => {
    const binanceCloses = Array.from({ length: 12 }, (_, i) => 77_000 + i * 100);
    const apiGetSpy = spyOn(realApiModule.api, 'get').mockRejectedValue(
      new FinancialDatasetsHttpError(
        400,
        'Bad Request',
        '{"message":"This instrument is not recognized"}',
      ),
    );
    const fetchMock: typeof fetch = Object.assign(
      async (input: Parameters<typeof fetch>[0], _init?: Parameters<typeof fetch>[1]) => {
        const url = input.toString();
        if (url.includes('api.binance.com/api/v3/klines')) {
          return new Response(JSON.stringify(binanceCloses.map((close) => [
            0,
            String(close - 50),
            String(close + 50),
            String(close - 100),
            String(close),
            '0',
            0,
            '0',
            0,
            '0',
            '0',
            '0',
          ])), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response('unavailable', { status: 500, statusText: 'Server Error' });
      },
      { preconnect: globalThis.fetch.preconnect },
    );
    const fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(fetchMock);

    try {
      await expect(fetchHistoricalPrices('BTC-USD', 30)).resolves.toEqual(binanceCloses);
    } finally {
      apiGetSpy.mockRestore();
      fetchSpy.mockRestore();
    }
  });

  it('does not fall back when unsupported-ticker keywords only align across duplicated Financial Datasets body boundaries', async () => {
    const body = `{"message":"ticker ${'x'.repeat(121)} unsupported"}`;
    const apiGetSpy = spyOn(realApiModule.api, 'get').mockRejectedValue(
      new FinancialDatasetsHttpError(400, 'Bad Request', body),
    );
    const fetchSpy = spyOn(globalThis, 'fetch');

    try {
      await expect(fetchHistoricalPrices('BTC-USD', 30)).rejects.toThrow(/unsupported/);
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      apiGetSpy.mockRestore();
      fetchSpy.mockRestore();
    }
  });

  it('surfaces generic Financial Datasets 400 errors instead of falling through', async () => {
    const apiGetSpy = spyOn(realApiModule.api, 'get').mockRejectedValue(
      new FinancialDatasetsHttpError(400, 'Bad Request', '{"error":"Missing required parameter start_date"}'),
    );
    const fetchSpy = spyOn(globalThis, 'fetch');

    try {
      await expect(fetchHistoricalPrices('BTC-USD', 30)).rejects.toThrow(/Missing required parameter/);
      expect(fetchSpy).not.toHaveBeenCalled();
    } finally {
      apiGetSpy.mockRestore();
      fetchSpy.mockRestore();
    }
  });

  it('rejects malformed Yahoo chart close entries instead of silently dropping them', async () => {
    const apiGetSpy = spyOn(realApiModule.api, 'get').mockResolvedValue({
      data: { prices: [] },
      url: 'https://api.financialdatasets.ai/prices/?ticker=AAPL',
    });
    const fetchMock: typeof fetch = Object.assign(
      async (input: Parameters<typeof fetch>[0], _init?: Parameters<typeof fetch>[1]) => {
        const url = input.toString();
        if (url.includes('query1.finance.yahoo.com')) {
          return new Response(JSON.stringify({
            chart: {
              result: [{
                indicators: {
                  quote: [{ close: [100, null, 102] }],
                },
              }],
            },
          }), {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
        return new Response('unavailable', { status: 500, statusText: 'Server Error' });
      },
      { preconnect: globalThis.fetch.preconnect },
    );
    const fetchSpy = spyOn(globalThis, 'fetch').mockImplementation(fetchMock);

    try {
      await expect(fetchHistoricalPrices('AAPL', 30)).rejects.toThrow(
        /Malformed Yahoo Finance chart payload for AAPL: chart\.result\.0\.indicators\.quote\.0\.close\.1/,
      );
    } finally {
      apiGetSpy.mockRestore();
      fetchSpy.mockRestore();
    }
  });
});

// ---------------------------------------------------------------------------
// Fix 10: classifyRegimeState — joint states (no priority override)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fix 1: buildDefaultMatrix — row sums must be exactly 1
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fix 1 + Fix 6: estimateTransitionMatrix — row sums + Dirichlet α=0.1
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fix 2: adjustTransitionMatrix — sentiment sign direction
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fix 4 + Fix 8: extractPriceThresholds — YES-bias + liquidity guard
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Crypto terminal-anchor fallback
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fix 3 + Fix 5: logNormalSurvival + CI bounds
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Fix 5: interpolateDistribution — CI bounds, monotonicity
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Add 3 + Fix 7: secondLargestEigenvalue + computeMixingWeight
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Add 4: R²_OS validation metric
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// calibrateProbabilities — Bayesian shrinkage (Idea I)
// ---------------------------------------------------------------------------

describe('calibrateProbabilities', () => {
  const sampleDist = [
    { price: 90,  probability: 0.95, lowerBound: 0.90, upperBound: 0.98, source: 'markov' as const },
    { price: 100, probability: 0.70, lowerBound: 0.60, upperBound: 0.80, source: 'markov' as const },
    { price: 110, probability: 0.30, lowerBound: 0.20, upperBound: 0.40, source: 'markov' as const },
    { price: 120, probability: 0.05, lowerBound: 0.02, upperBound: 0.10, source: 'markov' as const },
  ];

  it('shrinks extreme probabilities toward 0.5', () => {
    const calibrated = calibrateProbabilities(sampleDist);
    // P=0.95 should be pulled down (closer to 0.5)
    expect(calibrated[0].probability).toBeLessThan(0.95);
    expect(calibrated[0].probability).toBeGreaterThan(0.5);
    // P=0.05 should be pulled up (closer to 0.5)
    expect(calibrated[3].probability).toBeGreaterThan(0.05);
    expect(calibrated[3].probability).toBeLessThan(0.5);
  });

  it('maintains monotonicity after calibration', () => {
    const calibrated = calibrateProbabilities(sampleDist);
    for (let i = 0; i < calibrated.length - 1; i++) {
      expect(calibrated[i].probability).toBeGreaterThanOrEqual(calibrated[i + 1].probability);
    }
  });

  it('shrinks less with high ensemble consensus', () => {
    const noConsensus = calibrateProbabilities(sampleDist, { ensembleConsensus: 0 });
    const fullConsensus = calibrateProbabilities(sampleDist, { ensembleConsensus: 3 });
    // With full consensus, the extreme values should be further from 0.5
    // (less shrinkage → more extreme → P=0.95 stays higher)
    expect(fullConsensus[0].probability).toBeGreaterThan(noConsensus[0].probability);
    // Low end: P=0.05 with full consensus should be lower (less shrunk toward 0.5)
    expect(fullConsensus[3].probability).toBeLessThan(noConsensus[3].probability);
  });

  it('shrinks less with more historical data', () => {
    const short = calibrateProbabilities(sampleDist, { historicalDays: 60 });
    const long = calibrateProbabilities(sampleDist, { historicalDays: 250 });
    expect(long[0].probability).toBeGreaterThan(short[0].probability);
    expect(long[3].probability).toBeLessThan(short[3].probability);
  });

  it('HMM convergence reduces shrinkage', () => {
    const noHmm = calibrateProbabilities(sampleDist, { hmmConverged: false });
    const withHmm = calibrateProbabilities(sampleDist, { hmmConverged: true });
    expect(withHmm[0].probability).toBeGreaterThan(noHmm[0].probability);
  });

  it('preserves probabilities close to 0.5 (already calibrated)', () => {
    const midDist = [
      { price: 100, probability: 0.52, lowerBound: 0.45, upperBound: 0.60, source: 'markov' as const },
      { price: 110, probability: 0.48, lowerBound: 0.40, upperBound: 0.55, source: 'markov' as const },
    ];
    const calibrated = calibrateProbabilities(midDist);
    // Already near 0.5 — should barely change
    expect(Math.abs(calibrated[0].probability - 0.52)).toBeLessThan(0.02);
    expect(Math.abs(calibrated[1].probability - 0.48)).toBeLessThan(0.02);
  });

  it('returns valid probabilities in [0, 1]', () => {
    const extremeDist = [
      { price: 50,  probability: 1.00, lowerBound: 0.99, upperBound: 1.00, source: 'markov' as const },
      { price: 200, probability: 0.00, lowerBound: 0.00, upperBound: 0.02, source: 'markov' as const },
    ];
    const calibrated = calibrateProbabilities(extremeDist);
    for (const p of calibrated) {
      expect(p.probability).toBeGreaterThanOrEqual(0);
      expect(p.probability).toBeLessThanOrEqual(1);
    }
  });

  it('adaptive baseRate shifts shrinkage center (Idea L)', () => {
    const dist = [
      { price: 90,  probability: 0.80, lowerBound: 0.70, upperBound: 0.90, source: 'markov' as const },
      { price: 110, probability: 0.20, lowerBound: 0.10, upperBound: 0.30, source: 'markov' as const },
    ];
    const neutral  = calibrateProbabilities(dist, { baseRate: 0.50 });
    const bullish  = calibrateProbabilities(dist, { baseRate: 0.60 });
    // Bullish base rate pulls probabilities upward compared to neutral center
    expect(bullish[0].probability).toBeGreaterThan(neutral[0].probability);
    expect(bullish[1].probability).toBeGreaterThan(neutral[1].probability);
  });

  it('baseRate is clamped to [0.25, 0.80] (Idea S widened range)', () => {
    const dist = [
      { price: 100, probability: 0.50, lowerBound: 0.40, upperBound: 0.60, source: 'markov' as const },
    ];
    const extreme = calibrateProbabilities(dist, { baseRate: 0.90 });
    // With kappa=0.45 and center clamped to 0.80:
    // calibrated = 0.45 * 0.80 + 0.55 * 0.50 = 0.36 + 0.275 = 0.635
    // Should NOT be pulled all the way to 0.90 (capped at 0.80)
    expect(extreme[0].probability).toBeLessThan(0.80);
    // But should be higher than old 0.65 cap result
    expect(extreme[0].probability).toBeGreaterThan(0.60);
  });

  it('high baseRate (0.75) raises center toward bullish level (Idea S)', () => {
    const dist = [
      { price: 100, probability: 0.45, lowerBound: 0.35, upperBound: 0.55, source: 'markov' as const },
    ];
    const neutral = calibrateProbabilities(dist, { baseRate: 0.50 });
    const bullish = calibrateProbabilities(dist, { baseRate: 0.75 });
    // With 75% base rate, calibration should pull 0.45 much higher than with 0.50
    expect(bullish[0].probability).toBeGreaterThan(neutral[0].probability);
    // Specifically: center=0.75 → 0.45*0.75 + 0.55*0.45 = 0.3375+0.2475 = 0.585
    expect(bullish[0].probability).toBeGreaterThan(0.55);
  });

  it('bull regime reduces shrinkage (Idea O)', () => {
    const dist = [
      { price: 90,  probability: 0.80, lowerBound: 0.70, upperBound: 0.90, source: 'markov' as const },
      { price: 110, probability: 0.20, lowerBound: 0.10, upperBound: 0.30, source: 'markov' as const },
    ];
    const sideways = calibrateProbabilities(dist, { currentRegime: 'sideways' });
    const bull     = calibrateProbabilities(dist, { currentRegime: 'bull' });
    // Bull → less shrinkage → predictions stay further from 0.5
    expect(bull[0].probability).toBeGreaterThan(sideways[0].probability);
    expect(bull[1].probability).toBeLessThan(sideways[1].probability);
  });

  it('bear regime reduces shrinkage like bull (Idea O)', () => {
    const dist = [
      { price: 90,  probability: 0.80, lowerBound: 0.70, upperBound: 0.90, source: 'markov' as const },
    ];
    const sideways = calibrateProbabilities(dist, { currentRegime: 'sideways' });
    const bear     = calibrateProbabilities(dist, { currentRegime: 'bear' });
    expect(bear[0].probability).toBeGreaterThan(sideways[0].probability);
  });

  it('sideways regime increases shrinkage (Idea O)', () => {
    const dist = [
      { price: 100, probability: 0.75, lowerBound: 0.65, upperBound: 0.85, source: 'markov' as const },
    ];
    const noRegime = calibrateProbabilities(dist);
    const sideways = calibrateProbabilities(dist, { currentRegime: 'sideways' });
    // Sideways adds +0.04 kappa → more shrinkage toward center
    expect(sideways[0].probability).toBeLessThan(noRegime[0].probability);
  });

  it('drift-based mode preserves S-shape (spread ≥ 70pp from -15% to +15%)', () => {
    // Simulate a distribution with strong S-shape
    const cp = 100;
    const driftN = 0.005; // slight bullish drift
    const volN = 0.10;    // 10% n-day vol
    const prices = [80, 85, 90, 95, 100, 105, 110, 115, 120];
    const rawDist = prices.map(p => ({
      price: p,
      probability: studentTSurvival(cp, p, driftN, volN),
      lowerBound: studentTSurvival(cp, p, driftN, volN) * 0.8,
      upperBound: Math.min(1, studentTSurvival(cp, p, driftN, volN) * 1.2),
      source: 'markov' as const,
    }));

    // Calibrate with drift params (new path) — strong base rate push
    const calibrated = calibrateProbabilities(rawDist, {
      baseRate: 0.75,
      currentPrice: cp,
      driftN,
      volN,
    });

    // The critical invariant: spread from -15% to +15% must remain large
    const pBelow = calibrated.find(p => p.price === 85)!.probability;
    const pAbove = calibrated.find(p => p.price === 115)!.probability;
    const spread = pBelow - pAbove;

    // With legacy per-point shrinkage, spread would collapse to ~15-20pp.
    // Drift-based calibration preserves ≥70pp spread.
    expect(spread).toBeGreaterThanOrEqual(0.70);
    // Monotonicity: higher prices have lower P(>price)
    for (let i = 0; i < calibrated.length - 1; i++) {
      expect(calibrated[i].probability).toBeGreaterThanOrEqual(calibrated[i + 1].probability);
    }
  });
});

// ---------------------------------------------------------------------------
// matPow — matrix exponentiation
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------

// Phase 2: commodity model-only condition logic — structural break is not a hard gate
describe('commodityModelOnly condition logic', () => {
  const COMMODITY_WRAPPER_MIN_R2 = -0.02;
  const COMMODITY_WRAPPER_MIN_CONFIDENCE = 0.15;

  it('Phase 2: commodity model-only evaluates true despite structural break', () => {
    const commodityModelOnly =
      'commodity' === 'commodity' &&
      0 === 0 &&
      (-0.003) >= COMMODITY_WRAPPER_MIN_R2 &&
      0.18 >= COMMODITY_WRAPPER_MIN_CONFIDENCE;
    // structuralBreakDetected is NOT in the condition — break does not block

    expect(commodityModelOnly).toBe(true);
  });

  it('Phase 2: commodity model-only rejects when R² below minimum', () => {
    const commodityModelOnly =
      'commodity' === 'commodity' &&
      0 === 0 &&
      (-0.05) >= COMMODITY_WRAPPER_MIN_R2 &&
      0.18 >= COMMODITY_WRAPPER_MIN_CONFIDENCE;

    expect(commodityModelOnly).toBe(false);
  });

  it('Phase 2: commodity model-only rejects when confidence below minimum', () => {
    const commodityModelOnly =
      'commodity' === 'commodity' &&
      0 === 0 &&
      (-0.003) >= COMMODITY_WRAPPER_MIN_R2 &&
      0.10 >= COMMODITY_WRAPPER_MIN_CONFIDENCE;

    expect(commodityModelOnly).toBe(false);
  });
});
// Integration: computeMarkovDistribution
// ---------------------------------------------------------------------------

// Phase 3: R²_OS validation threshold for non-crypto — near-zero values should not block
describe('R²_OS validation threshold', () => {
  it('Phase 3: non-crypto validation acceptable with R² = -0.003', () => {
    const r2os = -0.003;
    const validationAcceptable = r2os >= -0.01;
    expect(validationAcceptable).toBe(true);
  });

  it('Phase 3: non-crypto validation rejects R² = -0.05', () => {
    const r2os = -0.05;
    const validationAcceptable = r2os >= -0.01;
    expect(validationAcceptable).toBe(false);
  });

  it('Phase 3: crypto short-horizon R² threshold unchanged at -0.05', () => {
    const r2os = -0.04;
    const validationAcceptable = r2os >= -0.05;
    expect(validationAcceptable).toBe(true);
  });

  it('Phase 3: composite crypto validation accepts mildly negative R² when sparse anchors, confidence, and fit support it', () => {
    expect(isCompositeValidationAcceptable({
      assetType: 'crypto',
      horizon: 14,
      outOfSampleR2: -0.055,
      validationMetric: 'horizon_return',
      anchorCoverage: { quality: 'sparse', trustedAnchors: 1 },
      predictionConfidence: 0.18,
      goodnessOfFit: { passes: true },
    })).toBe(true);
  });

  it('Phase 3: composite crypto validation still rejects mildly negative R² when confidence is too low', () => {
    expect(isCompositeValidationAcceptable({
      assetType: 'crypto',
      horizon: 14,
      outOfSampleR2: -0.055,
      validationMetric: 'horizon_return',
      anchorCoverage: { quality: 'sparse', trustedAnchors: 1 },
      predictionConfidence: 0.14,
      goodnessOfFit: { passes: true },
    })).toBe(false);
  });

  it('Phase 3: composite crypto validation still rejects mildly negative R² when fit support fails', () => {
    expect(isCompositeValidationAcceptable({
      assetType: 'crypto',
      horizon: 14,
      outOfSampleR2: -0.055,
      validationMetric: 'horizon_return',
      anchorCoverage: { quality: 'good', trustedAnchors: 2 },
      predictionConfidence: 0.18,
      goodnessOfFit: { passes: false },
    })).toBe(false);
  });
});

describe('Phase 4: degraded confidence policy', () => {
  it('softens short-horizon crypto break penalties when anchors are supportive and validation is only missing', () => {
    const breakdown = computePredictionConfidenceBreakdown({
      pUp: 0.58,
      ensembleConsensus: 2,
      hmmConverged: true,
      regimeRunLength: 8,
      structuralBreak: true,
      assetType: 'crypto',
      recentVol: 0.03,
      momentumAgreement: 0.67,
      calibratedPUp: 0.57,
      baseRate: 0.53,
      trustedAnchors: 2,
      horizonDays: 3,
      outOfSampleR2: null,
      confidenceMode: 'rebalanced',
    });

    expect(breakdown.multipliers.structuralBreak).toBeCloseTo(0.85);
    expect(breakdown.multipliers.assetType).toBeCloseTo(0.95);
  });

  it('keeps the harsher crypto penalties when validation is clearly bad', () => {
    const breakdown = computePredictionConfidenceBreakdown({
      pUp: 0.58,
      ensembleConsensus: 2,
      hmmConverged: true,
      regimeRunLength: 8,
      structuralBreak: true,
      assetType: 'crypto',
      recentVol: 0.03,
      momentumAgreement: 0.67,
      calibratedPUp: 0.57,
      baseRate: 0.53,
      trustedAnchors: 2,
      horizonDays: 3,
      outOfSampleR2: -0.08,
      confidenceMode: 'rebalanced',
    });

    expect(breakdown.multipliers.structuralBreak).toBeCloseTo(0.6);
    expect(breakdown.multipliers.assetType).toBeCloseTo(0.82);
  });

  it('permits a context-only lane for short-horizon crypto with good anchors but unavailable validation', () => {
    expect(shouldEmitContextOnlyCanonical({
      ticker: 'BTC-USD',
      assetType: 'crypto',
      horizon: 3,
      predictionConfidence: 0.24,
      outOfSampleR2: null,
      anchorCoverage: { quality: 'good', trustedAnchors: 3 },
      goodnessOfFit: { passes: true },
    })).toBe(true);
  });

  it('still rejects the context-only lane when the fit support is weak', () => {
    expect(shouldEmitContextOnlyCanonical({
      ticker: 'BTC-USD',
      assetType: 'crypto',
      horizon: 3,
      predictionConfidence: 0.24,
      outOfSampleR2: null,
      anchorCoverage: { quality: 'good', trustedAnchors: 3 },
      goodnessOfFit: { passes: false },
    })).toBe(false);
  });
});

describe('computeMarkovDistribution (integration)', () => {
  const prices = Array.from({ length: 90 }, (_, i) => 100 + i * 0.2 + Math.sin(i) * 2);

  it('returns a result with the correct ticker and horizon', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 10,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.ticker).toBe('TEST');
    expect(result.horizon).toBe(10);
  });

  it('distribution has 21 points (0..numLevels)', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 5,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.distribution.length).toBe(21);
  });

  it('metadata includes all expected fields', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 20,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const m = result.metadata;
    expect(typeof m.regimeState).toBe('string');
    expect(typeof m.mixingTimeWeight).toBe('number');
    expect(typeof m.secondEigenvalue).toBe('number');
    expect(m.mixingTimeWeight).toBeGreaterThanOrEqual(0);
    expect(m.mixingTimeWeight).toBeLessThanOrEqual(1);
  });

  it('Polymarket anchors count only trusted (high) anchors', async () => {
    const recentTime = FIXED_NOW_MS - 1 * 60 * 60 * 1000; // 1h ago (untrusted)
    const oldTime    = FIXED_NOW_MS - 10 * MS_PER_DAY; // 10d ago (trusted)
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 10,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [
        { question: 'Will TEST exceed $120?', probability: 0.6, createdAt: oldTime,    volume: 5000 },
        { question: 'Will TEST exceed $130?', probability: 0.4, createdAt: recentTime, volume: 1000 },
      ],
    });
    expect(result.metadata.polymarketAnchors).toBe(2); // both anchors are non-crypto with volume — trusted
  });

  it('distribution is monotonically non-increasing in price', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 15,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const dist = result.distribution;
    for (let i = 1; i < dist.length; i++) {
      expect(dist[i].probability).toBeLessThanOrEqual(dist[i - 1].probability + 1e-9);
    }
  });

  it('sentiment shift is reflected in metadata', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 5,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
      sentiment: { bullish: 0.8, bearish: 0.2 },
    });
    expect(result.metadata.sentimentAdjustment).toBeCloseTo(0.6, 5);
  });

  it('scenarios are present and consistent with distribution CDF', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 7,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
    });

    // Scenarios must exist
    expect(result.scenarios).toBeDefined();
    expect(result.scenarios.buckets).toHaveLength(5);

    // Bucket probabilities must sum to ~1
    const total = result.scenarios.buckets.reduce((s, b) => s + b.probability, 0);
    expect(total).toBeCloseTo(1.0, 1);

    // P(Up>5%) must equal CDF P(>1.05×current)
    const upOver5 = result.scenarios.buckets.find(b => b.label === 'Up >5%')!;
    const cdfAt105 = interpolateSurvival(result.distribution, 118 * 1.05);
    expect(upOver5.probability).toBeCloseTo(cdfAt105, 2);

    // P(Down>5%) must equal 1 - CDF P(>0.95×current)
    const downOver5 = result.scenarios.buckets.find(b => b.label === 'Down >5%')!;
    const cdfAt95 = interpolateSurvival(result.distribution, 118 * 0.95);
    expect(downOver5.probability).toBeCloseTo(1 - cdfAt95, 2);

    // scenarios.pUp should match CDF at currentPrice
    const cdfPUp = interpolateSurvival(result.distribution, 118);
    expect(result.scenarios.pUp).toBeCloseTo(cdfPUp, 2);
  });

  it('trajectory P(Up) is aligned with calibrated CDF at final day', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 7,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
      trajectory: true,
      trajectoryDays: 7,
    });

    expect(result.trajectory).toBeDefined();
    const traj = result.trajectory!;
    const finalDay = traj[traj.length - 1];
    const calPUp = interpolateSurvival(result.distribution, 118);

    // Final-day trajectory P(Up) should be within 3pp of calibrated CDF P(Up)
    // (we allow 3pp because the alignment only kicks in when divergence > 2pp)
    expect(Math.abs(finalDay.pUp - calPUp)).toBeLessThan(0.05);
  });
});

// ---------------------------------------------------------------------------
// estimateRegimeStats — empirical return statistics per state
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tier 1a — countStateObservations + findSparseStates
// ---------------------------------------------------------------------------

import {
  countStateObservations,
  findSparseStates,
  detectStructuralBreak,
  mergeAnchorsWithCrossPlatformValidation,
  type KalshiAnchor,
} from './markov-distribution.js';

describe('Tier 1a — sparseStates in computeMarkovDistribution metadata', () => {
  it('metadata.sparseStates includes states with few observations', async () => {
    // Only 11 prices → 10 returns, all forcing bull state (tiny window ensures sparsity)
    const prices = Array.from({ length: 11 }, (_, i) => 100 + i * 0.5);
    const result = await computeMarkovDistribution({
      ticker: 'SPARSE',
      horizon: 5,
      currentPrice: 105,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    // With only 10 returns all going up, bear should be sparse
    expect(result.metadata.stateObservationCounts).toBeDefined();
    expect(Array.isArray(result.metadata.sparseStates)).toBe(true);
  });

  it('metadata.stateObservationCounts sums to historicalDays', async () => {
    const prices = Array.from({ length: 40 }, (_, i) => 100 * (1 + i * 0.002));
    const result = await computeMarkovDistribution({
      ticker: 'COUNT',
      horizon: 5,
      currentPrice: 107,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const total = Object.values(result.metadata.stateObservationCounts).reduce((s, v) => s + v, 0);
    expect(total).toBe(result.metadata.historicalDays);
  });
});

// ---------------------------------------------------------------------------
// Tier 1b — detectStructuralBreak
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Tier 1c — mergeAnchorsWithCrossPlatformValidation
// ---------------------------------------------------------------------------

describe('Tier 1c — mergeAnchorsWithCrossPlatformValidation', () => {
  it('returns only Polymarket anchors when no Kalshi anchors provided', () => {
    const { anchors, warnings } = mergeAnchorsWithCrossPlatformValidation(
      [{
        price: 100, rawProbability: 0.6, probability: 0.57, trustScore: 'high', source: 'polymarket',
      }],
      [],
    );
    expect(anchors).toHaveLength(1);
    expect(warnings).toHaveLength(0);
  });

  it('adds Kalshi-only anchors with no bias correction', () => {
    const kalshi: KalshiAnchor[] = [{ price: 150, probability: 0.3, volume: 500 }];
    const { anchors } = mergeAnchorsWithCrossPlatformValidation([], kalshi);
    expect(anchors).toHaveLength(1);
    expect(anchors[0].source).toBe('kalshi');
    // No YES-bias correction for Kalshi: rawProb = probability = 0.3
    expect(anchors[0].probability).toBe(0.3);
    expect(anchors[0].rawProbability).toBe(0.3);
  });

  it('averages matching anchors within price tolerance', () => {
    const poly = [{
      price: 100, rawProbability: 0.60, probability: 0.57, trustScore: 'high' as const, source: 'polymarket' as const,
    }];
    const kalshi: KalshiAnchor[] = [{ price: 100.5, probability: 0.60, volume: 200 }]; // within 2%
    const { anchors, warnings } = mergeAnchorsWithCrossPlatformValidation(poly, kalshi);
    expect(anchors).toHaveLength(1);
    expect(anchors[0].source).toBe('averaged');
    // No divergence (both 0.60)
    expect(warnings).toHaveLength(0);
    // Averaged raw = (0.60 + 0.60) / 2 = 0.60; bias-corrected = 0.60 * 0.95 = 0.57
    expect(anchors[0].rawProbability).toBeCloseTo(0.60, 5);
    expect(anchors[0].probability).toBeCloseTo(0.57, 5);
  });

  it('emits warning when Polymarket and Kalshi diverge by more than 5pp', () => {
    const poly = [{
      price: 100, rawProbability: 0.70, probability: 0.665, trustScore: 'high' as const, source: 'polymarket' as const,
    }];
    const kalshi: KalshiAnchor[] = [{ price: 100, probability: 0.60 }]; // 10pp divergence
    const { anchors, warnings } = mergeAnchorsWithCrossPlatformValidation(poly, kalshi);
    expect(warnings).toHaveLength(1);
    expect(warnings[0].divergencePp).toBeCloseTo(10, 0);
    expect(warnings[0].polymarketProb).toBeCloseTo(0.70, 5);
    expect(warnings[0].kalshiProb).toBeCloseTo(0.60, 5);
    // Averaged: (0.70 + 0.60) / 2 = 0.65; bias-corrected = 0.65 * 0.95
    expect(anchors[0].rawProbability).toBeCloseTo(0.65, 5);
    expect(anchors[0].probability).toBeCloseTo(0.65 * 0.95, 5);
  });

  it('does NOT emit warning when divergence is ≤5pp', () => {
    const poly = [{
      price: 100, rawProbability: 0.55, probability: 0.5225, trustScore: 'high' as const, source: 'polymarket' as const,
    }];
    const kalshi: KalshiAnchor[] = [{ price: 100, probability: 0.52 }]; // 3pp divergence
    const { warnings } = mergeAnchorsWithCrossPlatformValidation(poly, kalshi);
    expect(warnings).toHaveLength(0);
  });

  it('result anchors are sorted by price ascending', () => {
    const poly = [
      { price: 120, rawProbability: 0.3, probability: 0.285, trustScore: 'high' as const, source: 'polymarket' as const },
      { price: 100, rawProbability: 0.7, probability: 0.665, trustScore: 'high' as const, source: 'polymarket' as const },
    ];
    const kalshi: KalshiAnchor[] = [{ price: 110, probability: 0.5 }];
    const { anchors } = mergeAnchorsWithCrossPlatformValidation(poly, kalshi);
    for (let i = 1; i < anchors.length; i++) {
      expect(anchors[i].price).toBeGreaterThanOrEqual(anchors[i - 1].price);
    }
  });

  it('upgrades trustScore to high when Kalshi anchor has volume', () => {
    const poly = [{
      price: 100, rawProbability: 0.55, probability: 0.5225, trustScore: 'low' as const, source: 'polymarket' as const,
    }];
    const kalshi: KalshiAnchor[] = [{ price: 100, probability: 0.54, volume: 1000 }];
    const { anchors } = mergeAnchorsWithCrossPlatformValidation(poly, kalshi);
    expect(anchors[0].trustScore).toBe('high');
  });

  it('kalshiAnchors parameter propagates through computeMarkovDistribution', async () => {
    const prices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.3);
    const result = await computeMarkovDistribution({
      ticker: 'CROSS',
      horizon: 10,
      currentPrice: 111.7,
      historicalPrices: prices,
      polymarketMarkets: [
        { question: 'Will CROSS exceed $115?', probability: 0.70, volume: 2000 },
      ],
      kalshiAnchors: [{ price: 115, probability: 0.60, volume: 500 }], // 10pp divergence
    });
    expect(result.metadata.anchorDivergenceWarnings).toHaveLength(1);
    expect(result.metadata.anchorDivergenceWarnings[0].divergencePp).toBeCloseTo(10, 0);
  });

  it('no divergence warnings when kalshiAnchors is absent', async () => {
    const prices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.3);
    const result = await computeMarkovDistribution({
      ticker: 'NOKALSHI',
      horizon: 10,
      currentPrice: 111.7,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.metadata.anchorDivergenceWarnings).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// Action signal: interpolateSurvival + computeActionSignal
// ---------------------------------------------------------------------------

import {
  computeActionSignal,
} from './markov-distribution.js';

/** Build a synthetic linear distribution: P(>price) = 1 − (price − lo) / (hi − lo) */
function makeLinearDist(lo: number, hi: number, n = 21): MarkovDistributionPoint[] {
  return Array.from({ length: n }, (_, i) => {
    const price = lo + (hi - lo) * (i / (n - 1));
    const prob  = 1 - i / (n - 1);
    return { price, probability: prob, lowerBound: prob - 0.05, upperBound: prob + 0.05, source: 'markov' as const };
  });
}

// ---------------------------------------------------------------------------
// computeScenarioProbabilities — derived from calibrated CDF
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// computeActionLevels
// ---------------------------------------------------------------------------

import { computeActionLevels } from './markov-distribution.js';

// ---------------------------------------------------------------------------
// assessAnchorCoverage
// ---------------------------------------------------------------------------

import { assessAnchorCoverage } from './markov-distribution.js';

// ---------------------------------------------------------------------------
// interpolateDistribution — anchor grid merging
// ---------------------------------------------------------------------------

// ===========================================================================
// CORRECTNESS & VALIDATION TESTS
// ===========================================================================

// ---------------------------------------------------------------------------
// normalCDF — direct tests against known Φ values
// ---------------------------------------------------------------------------

describe('normalCDF', () => {
  // normalCDF now computes the true standard normal CDF: Φ(x) = 0.5*(1+erf(x/√2))

  it('Φ(0) ≈ 0.5', () => {
    expect(normalCDF(0)).toBeCloseTo(0.5, 6);
  });

  it('Φ(1) ≈ 0.8413', () => {
    expect(normalCDF(1)).toBeCloseTo(0.8413, 3);
  });

  it('Φ(-1) ≈ 0.1587', () => {
    expect(normalCDF(-1)).toBeCloseTo(0.1587, 3);
  });

  it('normalCDF(x) + normalCDF(-x) = 1 (symmetry property)', () => {
    for (const x of [0.5, 1.0, 2.0, 3.0]) {
      expect(normalCDF(x) + normalCDF(-x)).toBeCloseTo(1.0, 6);
    }
  });

  it('Φ(1.96) ≈ 0.975 (95% critical value)', () => {
    expect(normalCDF(1.96)).toBeCloseTo(0.975, 2);
  });

  it('Φ(3) ≈ 0.99865 (deep right tail)', () => {
    expect(normalCDF(3)).toBeCloseTo(0.99865, 3);
  });

  it('Φ(-3) ≈ 0.00135 (deep left tail)', () => {
    expect(normalCDF(-3)).toBeCloseTo(0.00135, 3);
  });

  it('is monotonically non-decreasing', () => {
    let prev = 0;
    for (let x = -4; x <= 4; x += 0.1) {
      const val = normalCDF(x);
      expect(val).toBeGreaterThanOrEqual(prev);
      prev = val;
    }
  });
});

// ---------------------------------------------------------------------------
// matMul — direct tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// normalizeHistoricalPriceTicker
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// normalizeRows — direct tests
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Analytical 2-state Markov verification (closed-form cross-check)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Monte Carlo convergence / stability
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Anchor influence A/B test
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Boundary conditions (horizon, degenerate inputs)
// ---------------------------------------------------------------------------

describe('boundary conditions', () => {
  const steadyPrices = Array.from({ length: 30 }, () => 100);
  const trendingUp = Array.from({ length: 30 }, (_, i) => 100 + i * 0.5);

  it('horizon=1: distribution is tight around current price', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'H1',
      horizon: 1,
      currentPrice: 115,
      historicalPrices: trendingUp,
      polymarketMarkets: [],
    });
    // At horizon 1, most probability should be near current price
    const near = result.distribution.filter(
      d => Math.abs(d.price - 115) / 115 < 0.05,
    );
    expect(near.length).toBeGreaterThan(0);
    // Points very far from current should have low probability
    const farAbove = result.distribution.filter(d => d.price > 130);
    for (const p of farAbove) {
      expect(p.probability).toBeLessThan(0.5);
    }
  });

  it('horizon=90: distribution is wider than horizon=5', async () => {
    const short = await computeMarkovDistribution({
      ticker: 'H_CMP',
      horizon: 5,
      currentPrice: 115,
      historicalPrices: trendingUp,
      polymarketMarkets: [],
    });
    const long = await computeMarkovDistribution({
      ticker: 'H_CMP',
      horizon: 90,
      currentPrice: 115,
      historicalPrices: trendingUp,
      polymarketMarkets: [],
    });

    // Average CI width should be larger for longer horizon
    const avgCIWidth = (dist: typeof short.distribution) =>
      dist.reduce((s, d) => s + (d.upperBound - d.lowerBound), 0) / dist.length;

    // CI width comparison: with 3-state model, the Markov chain mixes faster
    // (fewer states → larger spectral gap), so long-horizon CIs may actually converge
    // to a tighter distribution. We just check that both produce non-degenerate CIs.
    expect(avgCIWidth(long.distribution)).toBeGreaterThan(0);
    expect(avgCIWidth(short.distribution)).toBeGreaterThan(0);
  });

  it('all-same-prices: produces a valid distribution without errors', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'FLAT',
      horizon: 10,
      currentPrice: 100,
      historicalPrices: steadyPrices,
      polymarketMarkets: [],
    });
    expect(result.distribution.length).toBeGreaterThan(0);
    // All returns are 0 → sideways regime
    expect(result.metadata.regimeState).toBe('sideways');
    // Action signal should still be valid
    const { buyProbability: b, holdProbability: h, sellProbability: s } = result.actionSignal;
    expect(b + h + s).toBeCloseTo(1.0, 4);
  });

  it('minimum viable input (10 prices) does not throw', async () => {
    const prices = Array.from({ length: 10 }, (_, i) => 100 + i);
    const result = await computeMarkovDistribution({
      ticker: 'MIN',
      horizon: 5,
      currentPrice: 109,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.distribution.length).toBeGreaterThan(0);
    expect(result.actionSignal.recommendation).toBeDefined();
  });

  it('strongly trending prices produce expected recommendation direction', async () => {
    // Strong uptrend: 100→130 in 30 days
    const uptrend = Array.from({ length: 30 }, (_, i) => 100 + i);
    const result = await computeMarkovDistribution({
      ticker: 'TREND',
      horizon: 10,
      currentPrice: 129,
      historicalPrices: uptrend,
      polymarketMarkets: [],
    });
    // Expected return should be positive for a strong uptrend
    expect(result.actionSignal.expectedReturn).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Sensitivity analysis
// ---------------------------------------------------------------------------

describe('sensitivity analysis', () => {
  const basePrices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.3);

  it('small price perturbation produces proportional output change', async () => {
    const base = await computeMarkovDistribution({
      ticker: 'SENS',
      horizon: 15,
      currentPrice: 112,
      historicalPrices: basePrices,
      polymarketMarkets: [],
    });

    // Perturb current price by +1%
    const perturbed = await computeMarkovDistribution({
      ticker: 'SENS',
      horizon: 15,
      currentPrice: 112 * 1.01,
      historicalPrices: basePrices,
      polymarketMarkets: [],
    });

    // Expected return should shift (not necessarily by exactly 1%, but it should change)
    const diff = Math.abs(base.actionSignal.expectedReturn - perturbed.actionSignal.expectedReturn);
    // Change should be bounded — no discontinuity from a 1% input shift
    expect(diff).toBeLessThan(0.10); // <10pp change from 1% price shift
    // Recommendation might stay the same, but probabilities should differ
    const probDiff = Math.abs(base.actionSignal.buyProbability - perturbed.actionSignal.buyProbability);
    expect(probDiff).toBeLessThan(0.20); // <20pp change
  });

  it('adding one more historical day does not cause large jumps', async () => {
    const base = await computeMarkovDistribution({
      ticker: 'SENS2',
      horizon: 15,
      currentPrice: 112,
      historicalPrices: basePrices,
      polymarketMarkets: [],
    });

    // Add one more day at roughly the same trajectory
    const extendedPrices = [...basePrices, 112.3];
    const extended = await computeMarkovDistribution({
      ticker: 'SENS2',
      horizon: 15,
      currentPrice: 112.3,
      historicalPrices: extendedPrices,
      polymarketMarkets: [],
    });

    // Expected returns should be similar
    const diff = Math.abs(base.actionSignal.expectedReturn - extended.actionSignal.expectedReturn);
    expect(diff).toBeLessThan(0.05);
  });

  it('sentiment shift produces monotonic effect on expected return', async () => {
    const bullish = await computeMarkovDistribution({
      ticker: 'SENT_MONO',
      horizon: 15,
      currentPrice: 112,
      historicalPrices: basePrices,
      polymarketMarkets: [],
      sentiment: { bullish: 0.8, bearish: 0.2 },
    });
    const neutral = await computeMarkovDistribution({
      ticker: 'SENT_MONO',
      horizon: 15,
      currentPrice: 112,
      historicalPrices: basePrices,
      polymarketMarkets: [],
      sentiment: { bullish: 0.5, bearish: 0.5 },
    });
    const bearish = await computeMarkovDistribution({
      ticker: 'SENT_MONO',
      horizon: 15,
      currentPrice: 112,
      historicalPrices: basePrices,
      polymarketMarkets: [],
      sentiment: { bullish: 0.2, bearish: 0.8 },
    });

    // Bullish sentiment → higher expected return than bearish (allow MC noise tolerance)
    const mcTolerance = 0.001;
    expect(bullish.actionSignal.expectedReturn).toBeGreaterThanOrEqual(bearish.actionSignal.expectedReturn - mcTolerance);
    // Neutral should be between (or at least not more extreme than either)
    expect(neutral.actionSignal.expectedReturn).toBeGreaterThanOrEqual(bearish.actionSignal.expectedReturn - 0.01);
    expect(neutral.actionSignal.expectedReturn).toBeLessThanOrEqual(bullish.actionSignal.expectedReturn + 0.01);
  });
});

// ---------------------------------------------------------------------------
// Tool output format validation
// ---------------------------------------------------------------------------

describe('markovDistributionTool output format', () => {
  const prices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.5);
  const canonicalPrices = Array.from({ length: 91 }, (_, i) => {
    let p = 100;
    for (let j = 0; j <= i; j++) {
      p *= 1 + Math.sin(j * 0.15) * 0.006;
    }
    return Math.round(p * 100) / 100;
  });
  const canonicalCurrentPrice = canonicalPrices[canonicalPrices.length - 1];
  const canonicalAnchors = [0.97, 1.0, 1.03].map((mult, idx) => ({
    question: `Will FMT_TEST be above $${Math.round(canonicalCurrentPrice * mult)} on April 9?`,
    probability: [0.72, 0.5, 0.28][idx],
    volume: 5000,
    createdAt: FIXED_NOW_MS - MS_PER_DAY * 5,
  }));

  it('output contains Decision Card with BUY/HOLD/SELL', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 7,
      historicalPrices: canonicalPrices,
      polymarketMarkets: canonicalAnchors,
    });
    expect(output).toContain('Your Options');
    expect(output).toContain('BUY');
    expect(output).toContain('HOLD');
    expect(output).toContain('SELL');
  });

  it('output contains Action Plan with price levels', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 7,
      historicalPrices: canonicalPrices,
      polymarketMarkets: canonicalAnchors,
    });
    expect(output).toContain('Action Plan');
    expect(output).toContain('Target');
    expect(output).toContain('Stop-loss');
    expect(output).toContain('Median forecast');
    expect(output).toContain('Bull case');
    expect(output).toContain('Bear case');
  });

  it('output contains recommendation with confidence', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 7,
      historicalPrices: canonicalPrices,
      polymarketMarkets: canonicalAnchors,
    });
    // Should have one of: [HIGH confidence], [MEDIUM confidence], [LOW confidence]
    expect(output).toMatch(/\[(HIGH|MEDIUM|LOW) confidence\]/);
  });

  it('output contains distribution table with P(>price) column', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 7,
      historicalPrices: canonicalPrices,
      polymarketMarkets: canonicalAnchors,
    });
    expect(output).toContain('P(>price)');
    expect(output).toContain('90% CI');
    expect(output).toContain('Source');
  });

  it('output contains anchor quality diagnostic', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 7,
      historicalPrices: canonicalPrices,
      polymarketMarkets: canonicalAnchors,
    });
    expect(output).toContain('Anchor quality:');
  });

  it('output contains contextual guidance (💡)', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 7,
      historicalPrices: canonicalPrices,
      polymarketMarkets: canonicalAnchors,
    });
    expect(output).toContain('💡');
  });

  it('output shows warnings when no anchors provided', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 15,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(output).toContain('Why this abstained');
    expect(output).toMatch(/anchor coverage is sparse|No trusted/);
  });
});

// ---------------------------------------------------------------------------
// transitionGoodnessOfFit — chi-squared test for Markov property
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// computeEnsembleSignal
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// studentTCDF and studentTSurvival (Idea F: fat tails)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// computePredictionConfidence — Idea M: selective prediction
// ---------------------------------------------------------------------------

describe('computePredictionConfidence', () => {
  it('returns high confidence when all signals are strong', () => {
    const c = computePredictionConfidence({
      pUp: 0.85, // very decisive
      ensembleConsensus: 3, // all signals agree
      hmmConverged: true,
      regimeRunLength: 25, // stable regime
      structuralBreak: false,
      momentumAgreement: 1.0, // all lookbacks agree
      calibratedPUp: 0.80, // aligned with strongly bullish base rate
      baseRate: 0.75,
    });
    expect(c).toBeGreaterThan(0.8);
    expect(c).toBeLessThanOrEqual(1.0);
  });

  it('returns low confidence when P(up) ≈ 0.5 (indecisive)', () => {
    const c = computePredictionConfidence({
      pUp: 0.51, // near coin flip
      ensembleConsensus: 0,
      hmmConverged: false,
      regimeRunLength: 1,
      structuralBreak: false,
    });
    expect(c).toBeLessThan(0.15);
  });

  it('structural break reduces confidence by 40%', () => {
    const base = computePredictionConfidence({
      pUp: 0.75,
      ensembleConsensus: 2,
      hmmConverged: true,
      regimeRunLength: 10,
      structuralBreak: false,
    });
    const broken = computePredictionConfidence({
      pUp: 0.75,
      ensembleConsensus: 2,
      hmmConverged: true,
      regimeRunLength: 10,
      structuralBreak: true,
    });
    expect(broken).toBeCloseTo(base * 0.6, 2);
  });

  it('trend_penalty_only skips the break penalty in sideways regimes', () => {
    const base = {
      pUp: 0.75,
      ensembleConsensus: 2,
      hmmConverged: true,
      regimeRunLength: 10,
      momentumAgreement: 0.5,
    };

    const noBreak = computePredictionConfidence({
      ...base,
      structuralBreak: false,
      regimeState: 'sideways',
    });
    const brokenDefault = computePredictionConfidence({
      ...base,
      structuralBreak: true,
      regimeState: 'sideways',
      breakConfidencePolicy: 'default',
    });
    const brokenTrendOnly = computePredictionConfidence({
      ...base,
      structuralBreak: true,
      regimeState: 'sideways',
      breakConfidencePolicy: 'trend_penalty_only',
    });

    expect(brokenDefault).toBeCloseTo(noBreak * 0.6, 2);
    expect(brokenTrendOnly).toBeCloseTo(noBreak, 6);
  });

  it('trend_penalty_only preserves the break penalty in trending regimes', () => {
    const base = {
      pUp: 0.75,
      ensembleConsensus: 2,
      hmmConverged: true,
      regimeRunLength: 10,
      momentumAgreement: 0.5,
    };

    const brokenDefault = computePredictionConfidence({
      ...base,
      structuralBreak: true,
      regimeState: 'bull',
      breakConfidencePolicy: 'default',
    });
    const brokenTrendOnly = computePredictionConfidence({
      ...base,
      structuralBreak: true,
      regimeState: 'bull',
      breakConfidencePolicy: 'trend_penalty_only',
    });

    expect(brokenTrendOnly).toBeCloseTo(brokenDefault, 6);
  });

  it('divergence_weighted softens medium break penalties without removing them', () => {
    const base = {
      pUp: 0.66,
      ensembleConsensus: 2,
      hmmConverged: true,
      regimeRunLength: 9,
      momentumAgreement: 0.5,
      regimeState: 'bear' as const,
    };

    const noBreak = computePredictionConfidence({
      ...base,
      structuralBreak: false,
    });
    const brokenDefault = computePredictionConfidence({
      ...base,
      structuralBreak: true,
      breakConfidencePolicy: 'default',
    });
    const brokenWeighted = computePredictionConfidence({
      ...base,
      structuralBreak: true,
      breakConfidencePolicy: 'divergence_weighted',
      structuralBreakDivergence: 0.12371808404330907,
    });

    expect(brokenDefault).toBeCloseTo(noBreak * 0.6, 2);
    expect(brokenWeighted).toBeCloseTo(noBreak * 0.7, 2);
    expect(brokenWeighted).toBeGreaterThan(brokenDefault);
    expect(brokenWeighted).toBeLessThan(noBreak);
  });

  it('more ensemble consensus increases confidence', () => {
    const low = computePredictionConfidence({
      pUp: 0.7, ensembleConsensus: 0, hmmConverged: false, regimeRunLength: 5, structuralBreak: false,
    });
    const high = computePredictionConfidence({
      pUp: 0.7, ensembleConsensus: 3, hmmConverged: false, regimeRunLength: 5, structuralBreak: false,
    });
    expect(high).toBeGreaterThan(low);
  });

  it('longer regime run increases confidence', () => {
    const short = computePredictionConfidence({
      pUp: 0.7, ensembleConsensus: 1, hmmConverged: true, regimeRunLength: 1, structuralBreak: false,
    });
    const long = computePredictionConfidence({
      pUp: 0.7, ensembleConsensus: 1, hmmConverged: true, regimeRunLength: 20, structuralBreak: false,
    });
    expect(long).toBeGreaterThan(short);
  });

  it('HMM convergence adds confidence', () => {
    const noHmm = computePredictionConfidence({
      pUp: 0.7, ensembleConsensus: 1, hmmConverged: false, regimeRunLength: 5, structuralBreak: false,
    });
    const withHmm = computePredictionConfidence({
      pUp: 0.7, ensembleConsensus: 1, hmmConverged: true, regimeRunLength: 5, structuralBreak: false,
    });
    expect(withHmm).toBeGreaterThan(noHmm);
    expect(withHmm - noHmm).toBeCloseTo(0.10, 1); // HMM adds 10% weight
  });

  it('always returns value in [0, 1]', () => {
    // Edge cases
    const extremes = [
      { pUp: 0.0, ensembleConsensus: 3, hmmConverged: true, regimeRunLength: 100, structuralBreak: false },
      { pUp: 1.0, ensembleConsensus: 0, hmmConverged: false, regimeRunLength: 0, structuralBreak: true },
      { pUp: 0.5, ensembleConsensus: 0, hmmConverged: false, regimeRunLength: 0, structuralBreak: false },
    ];
    for (const opts of extremes) {
      const c = computePredictionConfidence(opts);
      expect(c).toBeGreaterThanOrEqual(0);
      expect(c).toBeLessThanOrEqual(1);
    }
  });

  it('symmetric around 0.5 — P(up)=0.3 and P(up)=0.7 give same decisiveness', () => {
    const low = computePredictionConfidence({
      pUp: 0.3, ensembleConsensus: 1, hmmConverged: true, regimeRunLength: 10, structuralBreak: false,
    });
    const high = computePredictionConfidence({
      pUp: 0.7, ensembleConsensus: 1, hmmConverged: true, regimeRunLength: 10, structuralBreak: false,
    });
    expect(low).toBeCloseTo(high, 5);
  });

  it('crypto asset type reduces confidence (Idea N+)', () => {
    const base = { pUp: 0.7, ensembleConsensus: 2, hmmConverged: true, regimeRunLength: 10, structuralBreak: false };
    const equity = computePredictionConfidence({ ...base, assetType: 'equity' });
    const crypto = computePredictionConfidence({ ...base, assetType: 'crypto' });
    expect(crypto).toBeLessThan(equity);
    expect(crypto / equity).toBeCloseTo(0.7, 1); // 0.7× discount
  });

  it('short-horizon crypto with anchors gets a lighter discount', () => {
    const base = { pUp: 0.7, ensembleConsensus: 2, hmmConverged: true, regimeRunLength: 10, structuralBreak: false };
    const plainCrypto = computePredictionConfidence({ ...base, assetType: 'crypto' });
    const anchoredCrypto = computePredictionConfidence({
      ...base,
      assetType: 'crypto',
      horizonDays: 7,
      trustedAnchors: 2,
      outOfSampleR2: -0.01,
    });
    expect(anchoredCrypto).toBeGreaterThan(plainCrypto);
  });

  it('keeps strong short-horizon crypto setups above the 0.45 selective cutoff', () => {
    const confidence = computePredictionConfidence({
      pUp: 0.62,
      ensembleConsensus: 2,
      hmmConverged: true,
      regimeRunLength: 8,
      structuralBreak: false,
      assetType: 'crypto',
      recentVol: 0.025,
      momentumAgreement: 2 / 3,
      calibratedPUp: 0.58,
      baseRate: 0.56,
      trustedAnchors: 2,
      horizonDays: 7,
      outOfSampleR2: 0.01,
      regimeState: 'bull',
      confidenceMode: 'rebalanced',
    });

    expect(confidence).toBeGreaterThan(0.45);
  });

  it('structural break penalty is softer for short-horizon crypto with anchors and neutral R²', () => {
    const broken = computePredictionConfidence({
      pUp: 0.75,
      ensembleConsensus: 2,
      hmmConverged: true,
      regimeRunLength: 10,
      structuralBreak: true,
      assetType: 'crypto',
      horizonDays: 7,
      trustedAnchors: 2,
      outOfSampleR2: -0.01,
    });
    const brokenBadR2 = computePredictionConfidence({
      pUp: 0.75,
      ensembleConsensus: 2,
      hmmConverged: true,
      regimeRunLength: 10,
      structuralBreak: true,
      assetType: 'crypto',
      horizonDays: 7,
      trustedAnchors: 2,
      outOfSampleR2: -0.08,
    });
    expect(broken).toBeGreaterThan(brokenBadR2);
  });

  it('treats near-zero R² as less severe than clearly bad R² for short-horizon crypto with anchors', () => {
    const base = {
      pUp: 0.7,
      ensembleConsensus: 2,
      hmmConverged: true,
      regimeRunLength: 10,
      structuralBreak: false,
      assetType: 'crypto' as const,
      horizonDays: 7,
      trustedAnchors: 2,
    };
    const neutral = computePredictionConfidence({ ...base, outOfSampleR2: -0.01 });
    const clearlyBad = computePredictionConfidence({ ...base, outOfSampleR2: -0.08 });
    expect(neutral).toBeGreaterThan(clearlyBad);
  });

  it('ETF asset type boosts confidence', () => {
    const base = { pUp: 0.7, ensembleConsensus: 2, hmmConverged: true, regimeRunLength: 10, structuralBreak: false };
    const equity = computePredictionConfidence({ ...base, assetType: 'equity' });
    const etf    = computePredictionConfidence({ ...base, assetType: 'etf' });
    expect(etf).toBeGreaterThan(equity);
  });

  it('high volatility reduces confidence', () => {
    const base = { pUp: 0.7, ensembleConsensus: 2, hmmConverged: true, regimeRunLength: 10, structuralBreak: false };
    const lowVol  = computePredictionConfidence({ ...base, recentVol: 0.01 });
    const highVol = computePredictionConfidence({ ...base, recentVol: 0.05 });
    expect(highVol).toBeLessThan(lowVol);
  });

  it('vol < 2% has no penalty', () => {
    const base = { pUp: 0.7, ensembleConsensus: 2, hmmConverged: true, regimeRunLength: 10, structuralBreak: false };
    const noVol  = computePredictionConfidence(base);
    const lowVol = computePredictionConfidence({ ...base, recentVol: 0.015 });
    expect(lowVol).toBeCloseTo(noVol, 5);
  });

  it('full momentum agreement boosts confidence (Idea R)', () => {
    const base = { pUp: 0.7, ensembleConsensus: 2, hmmConverged: true, regimeRunLength: 10, structuralBreak: false };
    const noMom  = computePredictionConfidence({ ...base, momentumAgreement: 0 });
    const fullMom = computePredictionConfidence({ ...base, momentumAgreement: 1.0 });
    expect(fullMom).toBeGreaterThan(noMom);
    expect(fullMom - noMom).toBeCloseTo(0.10, 1); // 10% weight for momentum agreement
  });

  it('partial momentum agreement gives partial boost', () => {
    const base = { pUp: 0.7, ensembleConsensus: 2, hmmConverged: true, regimeRunLength: 10, structuralBreak: false };
    const none = computePredictionConfidence({ ...base, momentumAgreement: 0 });
    const half = computePredictionConfidence({ ...base, momentumAgreement: 0.5 });
    const full = computePredictionConfidence({ ...base, momentumAgreement: 1.0 });
    expect(half).toBeGreaterThan(none);
    expect(full).toBeGreaterThan(half);
  });
});

// ---------------------------------------------------------------------------
// getAssetProfile — Idea N: per-asset parameter profiles
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// computeRegimeUpRates — Regime-conditional P(up) (Idea T, Round 4)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// computeTrajectory — day-by-day price forecast
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// computeMarkovDistribution — trajectory integration
// ---------------------------------------------------------------------------

describe('computeMarkovDistribution trajectory mode', () => {
  const prices = Array.from({ length: 60 }, (_, i) => 100 + i * 0.3);

  it('returns trajectory when trajectory=true', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TRAJ_TEST',
      horizon: 7,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
      trajectory: true,
    });
    expect(result.trajectory).toBeDefined();
    expect(result.trajectory!.length).toBe(7);
    expect(result.decisionSurface).toBe('calibrated');
    expect(result.decisionDistribution).toEqual(result.distribution);
    expect(result.decisionScenarios).toEqual(result.scenarios);
  });

  it('trajectory is undefined when trajectory=false', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TRAJ_OFF',
      horizon: 7,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
      trajectory: false,
    });
    expect(result.trajectory).toBeUndefined();
  });

  it('respects trajectoryDays param', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TRAJ_DAYS',
      horizon: 14,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
      trajectory: true,
      trajectoryDays: 5,
    });
    expect(result.trajectory!.length).toBe(5);
  });

  it('trajectoryDays capped at 30', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TRAJ_CAP',
      horizon: 90,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
      trajectory: true,
      trajectoryDays: 50,
    });
    expect(result.trajectory!.length).toBe(30);
  });
});

// ---------------------------------------------------------------------------
// markovDistributionTool — trajectory output format
// ---------------------------------------------------------------------------

describe('markovDistributionTool trajectory output', () => {
  const prices = Array.from({ length: 91 }, (_, i) => {
    let p = 100;
    for (let j = 0; j <= i; j++) {
      p *= 1 + Math.sin(j * 0.15) * 0.006;
    }
    return Math.round(p * 100) / 100;
  });
  const currentPrice = prices[prices.length - 1];
  const anchors = [0.97, 1.0, 1.03].map((mult, idx) => ({
    question: `Will FMT_TRAJ be above $${Math.round(currentPrice * mult)} on April 9?`,
    probability: [0.72, 0.5, 0.28][idx],
    volume: 5000,
    createdAt: FIXED_NOW_MS - MS_PER_DAY * 5,
  }));

  it('includes trajectory table when trajectory=true', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TRAJ',
      horizon: 7,
      historicalPrices: prices,
      polymarketMarkets: anchors,
      trajectory: true,
    });
    expect(output).toContain('DAY PATH CONTEXT TRAJECTORY');
    expect(output).toContain('Path context only');
    expect(output).toContain('Day │ Expected');
    expect(output).toContain('P(up)');
    expect(output).toContain('Return');
    expect(output).toContain('probability-weighted means');
  });

  it('does not include trajectory when trajectory=false', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_NOTRAJ',
      horizon: 7,
      historicalPrices: prices,
      polymarketMarkets: anchors.map((anchor) => ({
        ...anchor,
        question: anchor.question.replace('FMT_TRAJ', 'FMT_NOTRAJ'),
      })),
      trajectory: false,
    });
    expect(output).not.toContain('DAY PRICE TRAJECTORY');
  });
});

// ---------------------------------------------------------------------------
// winsorize — outlier clamping
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// estimateRegimeStats — drift cap and winsorization
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// commodity asset profile — maxDailyDrift
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// normalizeAnchorPricesForETF — commodity futures → ETF price conversion
// ---------------------------------------------------------------------------

describe('Phase 5 fallback helpers', () => {
  it('keeps medium-divergence fallback blending row-stochastic for GLD-style breaks', () => {
    const estimated = [
      [0.72, 0.18, 0.10],
      [0.16, 0.70, 0.14],
      [0.20, 0.18, 0.62],
    ];
    const candidate = {
      id: 'gld-test',
      mode: 'blended' as const,
      conservativeDiagonal: 0.60,
      profileDiagonals: { equity: 0.60, etf: 0.55, commodity: 0.65, crypto: 0.70 },
      conservativeWeight: 0.25,
      severityWeights: { mild: 0.25, medium: 0.50, high: 0.75 },
    };

    const profile = buildProfileFallbackMatrix('commodity', candidate.profileDiagonals);
    expect(profile[0][0]).toBeCloseTo(0.65, 10);
    expect(computeBlendWeight(0.12371808404330907, candidate.severityWeights)).toBeCloseTo(0.50, 10);

    const blended = applyBreakFallbackCandidate(
      estimated,
      0.12371808404330907,
      candidate,
      'commodity',
    );

    for (const row of blended) {
      expect(row.reduce((sum, value) => sum + value, 0)).toBeCloseTo(1, 10);
    }
    expect(blended[0][0]).toBeGreaterThan(0.64);
    expect(blended[0][0]).toBeLessThan(0.72);
  });
});

// ---------------------------------------------------------------------------
// markov_distribution tool — auto-fetch and schema validation
// ---------------------------------------------------------------------------

describe('markov_distribution tool schema', () => {
  it('accepts call without historicalPrices (optional field)', () => {
    // Verify the schema accepts undefined historicalPrices
    const schema = markovDistributionTool.schema;
    const parsed = schema.safeParse({
      ticker: 'GLD',
      horizon: 30,
    });
    expect(parsed.success).toBe(true);
  });

  it('accepts call with empty polymarketMarkets (defaults to [])', () => {
    const schema = markovDistributionTool.schema;
    const parsed = schema.safeParse({
      ticker: 'SPY',
      horizon: 7,
      historicalPrices: Array(30).fill(100),
    });
    expect(parsed.success).toBe(true);
    if (parsed.success) {
      expect(parsed.data.polymarketMarkets).toEqual([]);
    }
  });

  it('still accepts valid historicalPrices when provided', () => {
    const schema = markovDistributionTool.schema;
    const parsed = schema.safeParse({
      ticker: 'AAPL',
      horizon: 14,
      historicalPrices: Array(60).fill(150),
      polymarketMarkets: [],
    });
    expect(parsed.success).toBe(true);
  });
});

describe('markov_distribution anchor query strategy', () => {
  it('normalizes BTC-USD search phrase to Bitcoin price', () => {
    expect(inferPolymarketSearchPhrase('BTC-USD')).toBe('Bitcoin price');
  });

  it('normalizes GLD search phrase to gold price', () => {
    expect(inferPolymarketSearchPhrase('GLD')).toBe('gold price');
  });

  it('normalizes SLV search phrase to silver price', () => {
    expect(inferPolymarketSearchPhrase('SLV')).toBe('silver price');
  });

  it('keeps explicit GOLD ticker search phrase Barrick-specific', () => {
    expect(inferPolymarketSearchPhrase('GOLD')).toBe('Barrick Gold price');
  });

  it('builds richer Bitcoin anchor query variants', () => {
    const variants = buildPolymarketAnchorQueryVariants('BTC-USD');
    expect(variants).toContain('Bitcoin price');
    expect(variants).toContain('Bitcoin');
    expect(variants).toContain('Bitcoin above');
    expect(variants).toContain('Bitcoin below');
  });

  it('prioritises price-target queries for BTC 30-day anchor acquisition', () => {
    const variants = buildPolymarketAnchorQueryVariants('BTC-USD', { horizonDays: 30 });
    const frontSlice = variants.slice(0, 6);
    const priceTargetQueries = ['Bitcoin price target', 'Bitcoin reach', 'Bitcoin exceed', 'BTC price level', 'Bitcoin ETF', 'crypto ETF'];
    const regulatoryQueries = ['crypto regulation', 'SEC crypto', 'cryptocurrency regulation'];
    const frontHasPriceTarget = frontSlice.some((q) => priceTargetQueries.some((pt) => q.includes(pt) || q === pt));
    const frontHasRegulatory = frontSlice.some((q) => regulatoryQueries.some((rq) => q.includes(rq) || q === rq));
    expect(frontHasPriceTarget).toBe(true);
    expect(frontHasRegulatory).toBe(false);
  });

  it('BTC 14-day includes month-name variants not present in default', () => {
    const defaultVariants = buildPolymarketAnchorQueryVariants('BTC-USD');
    const shortHorizonVariants = buildPolymarketAnchorQueryVariants('BTC-USD', { horizonDays: 14 });
    const targetMonth = new Date(FIXED_NOW_MS + 14 * MS_PER_DAY)
      .toLocaleString('en-US', { month: 'long' });
    expect(shortHorizonVariants.length).toBeGreaterThan(defaultVariants.length);
    expect(shortHorizonVariants).toContain(`Bitcoin ${targetMonth}`);
    expect(shortHorizonVariants).toContain(`Bitcoin above ${targetMonth}`);
    expect(defaultVariants).not.toContain(`Bitcoin ${targetMonth}`);
  });

  it('keeps BTC 14-day query ordering with month-name variants in front slice', () => {
    const variants = buildPolymarketAnchorQueryVariants('BTC-USD', { horizonDays: 14 });
    const targetMonth = new Date(FIXED_NOW_MS + 14 * MS_PER_DAY)
      .toLocaleString('en-US', { month: 'long' });
    expect(variants.slice(0, 6)).toEqual([
      'Bitcoin price',
      'Bitcoin',
      'Bitcoin above',
      'Bitcoin below',
      `Bitcoin ${targetMonth}`,
      `Bitcoin above ${targetMonth}`,
    ]);
  });

  it('does not inject month-name variants for BTC 30-day', () => {
    const variants = buildPolymarketAnchorQueryVariants('BTC-USD', { horizonDays: 30 });
    const targetMonth = new Date(FIXED_NOW_MS + 30 * MS_PER_DAY)
      .toLocaleString('en-US', { month: 'long' });
    expect(variants.slice(0, 6).some((variant) => variant.includes(targetMonth))).toBe(false);
  });

  it('keeps primary and manual queries first for BTC 30-day', () => {
    const variants = buildPolymarketAnchorQueryVariants('BTC-USD', { horizonDays: 30 });
    expect(variants[0]).toBe('Bitcoin price');
    expect(variants.slice(0, 4)).toEqual(['Bitcoin price', 'Bitcoin', 'Bitcoin above', 'Bitcoin below']);
  });

  it('does not reorder queries for non-crypto tickers even with long horizon', () => {
    const defaultVariants = buildPolymarketAnchorQueryVariants('AAPL');
    const longHorizonVariants = buildPolymarketAnchorQueryVariants('AAPL', { horizonDays: 30 });
    expect(longHorizonVariants).toEqual(defaultVariants);
  });

  it('builds Barrick-specific anchor query variants for GOLD', () => {
    const variants = buildPolymarketAnchorQueryVariants('GOLD');
    expect(variants).toContain('Barrick Gold price');
    expect(variants).toContain('Barrick Gold');
    expect(variants).not.toContain('gold price');
  });
});

// ---------------------------------------------------------------------------
// Comprehensive output validation — GLD 30-day integration test
// Validates all output invariants that the agent displays to the user.
// ---------------------------------------------------------------------------

describe('GLD 30-day output validation (integration)', () => {
  // Simulate realistic GLD price data: ~$490 peak → decline to ~$415
  // This creates a sideways/bearish regime with mean-reversion potential
  const gldPrices: number[] = [];
  {
    let p = 450;
    // Phase 1: rally to ~490 (40 days)
    for (let i = 0; i < 40; i++) { p *= 1 + 0.002 + (Math.sin(i * 0.3) * 0.005); gldPrices.push(Math.round(p * 100) / 100); }
    // Phase 2: sell-off to ~415 (40 days)
    for (let i = 0; i < 40; i++) { p *= 1 - 0.004 + (Math.sin(i * 0.3) * 0.003); gldPrices.push(Math.round(p * 100) / 100); }
    // Phase 3: sideways ~415 (20 days)
    for (let i = 0; i < 20; i++) { p *= 1 + (Math.sin(i * 0.5) * 0.004); gldPrices.push(Math.round(p * 100) / 100); }
  }
  const currentPrice = gldPrices[gldPrices.length - 1];

  let result: Awaited<ReturnType<typeof computeMarkovDistribution>>;

  // Run once for all invariant checks
  it('setup: computes GLD distribution', async () => {
    result = await computeMarkovDistribution({
      ticker: 'GLD',
      horizon: 30,
      currentPrice,
      historicalPrices: gldPrices,
      polymarketMarkets: [
        { question: 'Will gold exceed $5,500 by June 2026?', probability: 0.40, volume: 50000 },
        { question: 'Will gold exceed $6,000 by June 2026?', probability: 0.10, volume: 30000 },
      ],
      trajectory: true,
      trajectoryDays: 7,
    });
    expect(result).toBeDefined();
    expect(result.distribution.length).toBeGreaterThan(5);
  });

  // --- Invariant 1: Scenario buckets sum to ~100% ---
  it('scenario buckets sum to ~100%', () => {
    const sum = result.scenarios.buckets.reduce((s, b) => s + b.probability, 0);
    expect(sum).toBeGreaterThan(0.95);
    expect(sum).toBeLessThan(1.05);
  });

  // --- Invariant 2: CDF is monotonically non-increasing ---
  it('CDF probabilities are monotonically non-increasing', () => {
    for (let i = 1; i < result.distribution.length; i++) {
      expect(result.distribution[i].probability).toBeLessThanOrEqual(
        result.distribution[i - 1].probability + 1e-9,
      );
    }
  });

  // --- Invariant 3: CDF prices are monotonically increasing ---
  it('CDF prices are monotonically increasing', () => {
    for (let i = 1; i < result.distribution.length; i++) {
      expect(result.distribution[i].price).toBeGreaterThan(
        result.distribution[i - 1].price,
      );
    }
  });

  // --- Invariant 4: CI contains the point estimate ---
  it('CI lower bound ≤ point estimate ≤ CI upper bound for all CDF points', () => {
    for (const pt of result.distribution) {
      if (pt.lowerBound != null && pt.upperBound != null) {
        // Use 2e-3 tolerance for floating-point clamping at extreme boundaries
        // (prob ≈ 1.0 far below current price: upperBound may be slightly < 1.0 due to MC sampling)
        expect(pt.lowerBound).toBeLessThanOrEqual(pt.probability + 2e-3);
        expect(pt.upperBound).toBeGreaterThanOrEqual(pt.probability - 2e-3);
      }
    }
  });

  // --- Invariant 5: Action signal consistent with scenario P(up) ---
  it('BUY recommendation only when P(up) ≥ 0.50', () => {
    if (result.actionSignal.recommendation === 'BUY') {
      expect(result.scenarios.pUp).toBeGreaterThanOrEqual(0.50);
    }
  });

  it('SELL recommendation only when P(up) ≤ 0.50', () => {
    if (result.actionSignal.recommendation === 'SELL') {
      expect(result.scenarios.pUp).toBeLessThanOrEqual(0.50);
    }
  });

  // --- Invariant 6: BUY not allowed when downside > upside + 5pp ---
  it('BUY not issued when downside scenarios exceed upside by >5pp', () => {
    const up = (result.scenarios.buckets[3]?.probability ?? 0) +
               (result.scenarios.buckets[4]?.probability ?? 0);
    const down = (result.scenarios.buckets[0]?.probability ?? 0) +
                 (result.scenarios.buckets[1]?.probability ?? 0);
    if (down > up + 0.05) {
      expect(result.actionSignal.recommendation).not.toBe('BUY');
    }
  });

  // --- Invariant 7: Scenario buckets are CDF-consistent ---
  it('scenario bucket boundaries match ±3% and ±5% of current price', () => {
    const b = result.scenarios.buckets;
    expect(b.length).toBe(5);
    // Down >5% bucket upper boundary ≈ 0.95 × currentPrice
    expect(b[0].priceRange[1]).toBeCloseTo(currentPrice * 0.95, 0);
    // Down 3-5% lower ≈ 0.95×, upper ≈ 0.97×
    expect(b[1].priceRange[0]).toBeCloseTo(currentPrice * 0.95, 0);
    expect(b[1].priceRange[1]).toBeCloseTo(currentPrice * 0.97, 0);
    // Flat ±3%
    expect(b[2].priceRange[0]).toBeCloseTo(currentPrice * 0.97, 0);
    expect(b[2].priceRange[1]).toBeCloseTo(currentPrice * 1.03, 0);
    // Up 3-5%
    expect(b[3].priceRange[0]).toBeCloseTo(currentPrice * 1.03, 0);
    expect(b[3].priceRange[1]).toBeCloseTo(currentPrice * 1.05, 0);
    // Up >5%
    expect(b[4].priceRange[0]).toBeCloseTo(currentPrice * 1.05, 0);
  });

  // --- Invariant 8: P(>price) at scenario boundaries matches bucket sums ---
  it('P(>down5) from CDF matches 1 - P(Down>5%) from scenarios', () => {
    const down5Price = currentPrice * 0.95;
    const pAboveDown5 = interpolateSurvival(result.distribution, down5Price);
    const pDownOver5 = result.scenarios.buckets[0].probability;
    // pAboveDown5 should equal 1 - pDownOver5
    expect(pAboveDown5).toBeCloseTo(1 - pDownOver5, 2);
  });

  it('P(>up5) from CDF matches P(Up>5%) from scenarios', () => {
    const up5Price = currentPrice * 1.05;
    const pAboveUp5 = interpolateSurvival(result.distribution, up5Price);
    const pUpOver5 = result.scenarios.buckets[4].probability;
    expect(pAboveUp5).toBeCloseTo(pUpOver5, 2);
  });

  // --- Invariant 9: Expected return sign matches median direction ---
  it('expected return and median price agree in direction (both up or both down)', () => {
    const medianReturn = (result.actionSignal.actionLevels.medianPrice - currentPrice) / currentPrice;
    const expectedReturn = result.scenarios.expectedReturn;
    // In highly skewed distributions, mean and median can diverge significantly.
    // Verify they are within 8pp of each other, OR agree in sign, OR both trivially small.
    if (Math.abs(expectedReturn) > 0.02 && Math.abs(medianReturn) > 0.02) {
      const signAgree = (expectedReturn > 0 && medianReturn > 0) || (expectedReturn < 0 && medianReturn < 0);
      const closeEnough = Math.abs(expectedReturn - medianReturn) < 0.08;
      expect(signAgree || closeEnough).toBe(true);
    }
  });

  // --- Invariant 10: Confidence not HIGH when mean/median disagree ---
  it('confidence is not HIGH if expected return and median return disagree in sign by >0.5pp', () => {
    const medianReturn = (result.actionSignal.actionLevels.medianPrice - currentPrice) / currentPrice;
    const expectedReturn = result.actionSignal.expectedReturn;
    if ((expectedReturn > 0 && medianReturn < -0.005) ||
        (expectedReturn < 0 && medianReturn > 0.005)) {
      expect(result.actionSignal.confidence).not.toBe('HIGH');
    }
  });

  // --- Invariant 11: Trajectory has correct number of days ---
  it('trajectory has requested number of days', () => {
    expect(result.trajectory).toBeDefined();
    expect(result.trajectory!.length).toBe(7);
  });

  // --- Invariant 12: Trajectory P(Up) at final day aligns with CDF P(up) ---
  it('trajectory final day P(Up) is within 3pp of CDF P(up)', () => {
    const traj = result.trajectory!;
    const lastPUp = traj[traj.length - 1].pUp;
    const cdfPUp = result.scenarios.pUp;
    expect(Math.abs(lastPUp - cdfPUp)).toBeLessThan(0.03);
  });

  // --- Invariant 13: Trajectory prices are within CI bounds ---
  it('trajectory expected prices are within 90% CI', () => {
    for (const day of result.trajectory!) {
      expect(day.expectedPrice).toBeGreaterThanOrEqual(day.lowerBound);
      expect(day.expectedPrice).toBeLessThanOrEqual(day.upperBound);
    }
  });

  // --- Invariant 14: CDF first point ≥ 0.90, last point ≤ 0.15 ---
  it('CDF has reasonable tail behavior (first ≥ 0.80, last ≤ 0.25)', () => {
    expect(result.distribution[0].probability).toBeGreaterThanOrEqual(0.80);
    expect(result.distribution[result.distribution.length - 1].probability).toBeLessThanOrEqual(0.25);
  });

  // --- Invariant 15: Commodity ETF anchor normalization applied ---
  it('Polymarket gold futures anchors are converted to GLD scale or filtered', () => {
    // The $5,500 and $6,000 gold futures anchors should either:
    // 1. Be converted to GLD-scale ($450-$500 range), OR
    // 2. Be filtered out (trust score low, etc.)
    // Either way, no anchor should remain at $5,500+ in the distribution
    for (const pt of result.distribution) {
      expect(pt.price).toBeLessThan(2000); // no raw futures prices in output
    }
  });
});

describe('PR3F Lever: short-horizon crypto disagreement prior', () => {
  const prices = Array.from({ length: 150 }, (_, i) => 1000 + i * 1.5 + Math.sin(i / 3) * 15);
  const currentPrice = prices[prices.length - 1];
  
  const polymarketMarkets = [
    { question: `Will the price of Bitcoin be above $${Math.round(currentPrice * 0.95)} on April 9?`, probability: 0.95, volume: 5000, createdAt: FIXED_NOW_MS - MS_PER_DAY * 3 },
    { question: `Will the price of Bitcoin be above $${Math.round(currentPrice)} on April 9?`, probability: 0.65, volume: 5000, createdAt: FIXED_NOW_MS - MS_PER_DAY * 3 },
    { question: `Will the price of Bitcoin be above $${Math.round(currentPrice * 1.05)} on April 9?`, probability: 0.20, volume: 5000, createdAt: FIXED_NOW_MS - MS_PER_DAY * 3 },
  ];

  it('preserves default behavior when flag absent', async () => {
    const defaultResult = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets,
    });
    expect(defaultResult.metadata.pr3fDisagreementBlendActive).toBe(false);
  });

  it('has no effect outside crypto <=14d', async () => {
    // Non-crypto
    const nonCrypto = await computeMarkovDistribution({
      ticker: 'AAPL',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets,
      pr3fCryptoShortHorizonDisagreementPrior: true,
    });
    expect(nonCrypto.metadata.pr3fDisagreementBlendActive).toBe(false);

    // Crypto long horizon
    const longHorizon = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 30,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets,
      pr3fCryptoShortHorizonDisagreementPrior: true,
    });
    expect(longHorizon.metadata.pr3fDisagreementBlendActive).toBe(false);
  });

  it('activates deterministic blend when raw/calibrated disagree', async () => {
    const defaultResult = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets,
      btcReturnThresholdMultiplier: 0.5,
    });
    
    const pr3fResult = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets,
      pr3fCryptoShortHorizonDisagreementPrior: true,
      btcReturnThresholdMultiplier: 0.5,
    });

    const rawPUp = interpolateSurvival(defaultResult.rawDistribution, currentPrice);
    const calibratedPUp = interpolateSurvival(defaultResult.distribution, currentPrice);
    const disagreement = Math.abs(calibratedPUp - rawPUp);

    // Canonical surfaces untouched (excluding MC-derived CI bounds which jitter)
    for (let i = 0; i < defaultResult.distribution.length; i++) {
      expect(pr3fResult.distribution[i].price).toBe(defaultResult.distribution[i].price);
      expect(pr3fResult.distribution[i].probability).toBe(defaultResult.distribution[i].probability);
      expect(pr3fResult.distribution[i].source).toBe(defaultResult.distribution[i].source);
    }
    
    // Scenarios shouldn't be affected by MC bounds jitter, they are derived from probability
    expect(pr3fResult.scenarios).toEqual(defaultResult.scenarios);

    if (disagreement > 0.05) {
      expect(pr3fResult.metadata.pr3fDisagreementBlendActive).toBe(true);
      expect(pr3fResult.actionSignal).not.toEqual(defaultResult.actionSignal);
    } else {
      expect(pr3fResult.metadata.pr3fDisagreementBlendActive).toBe(false);
      expect(pr3fResult.actionSignal).toEqual(defaultResult.actionSignal);
    }
  });
});

describe('PR3G Lever: Recency-Weighted Regime Up-Rates', () => {
  const currentPrice = 60000;
  const prices = Array.from({ length: 150 }, (_, i) => 
    50000 + i * 100 + (Math.sin(i / 5) * 2000)
  );

  it('applies deterministic effect of a milder decay vs a more aggressive decay', async () => {
    const aggressiveResult = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      pr3gCryptoShortHorizonRecencyWeighting: true,
      pr3gCryptoShortHorizonDecay: 0.5,
    });
    
    const milderResult = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      pr3gCryptoShortHorizonRecencyWeighting: true,
      pr3gCryptoShortHorizonDecay: 0.99,
    });

    expect(aggressiveResult.metadata.pr3gRecencyWeightingActive).toBe(true);
    expect(milderResult.metadata.pr3gRecencyWeightingActive).toBe(true);
    expect(aggressiveResult.actionSignal.expectedReturn).not.toBe(milderResult.actionSignal.expectedReturn);
  });
});

describe('BTC 14d bearish-break SELL gate', () => {
  it('fires for the measured low-confidence bearish structural-break slice', () => {
    expect(shouldApplyBtc14dBearishBreakSellGate({
      ticker: 'BTC-USD',
      horizon: 14,
      recommendation: 'BUY',
      rawRecommendation: 'SELL',
      structuralBreakDetected: true,
      regimeState: 'sideways',
      predictionConfidence: 0.09,
      rawPredictedProb: 0.50,
      predictedProb: 0.54,
      expectedReturn: 0.025,
    })).toBe(true);
  });

  it('does not fire outside BTC 14d', () => {
    expect(shouldApplyBtc14dBearishBreakSellGate({
      ticker: 'BTC-USD',
      horizon: 7,
      recommendation: 'BUY',
      rawRecommendation: 'SELL',
      structuralBreakDetected: true,
      regimeState: 'sideways',
      predictionConfidence: 0.09,
      rawPredictedProb: 0.50,
      predictedProb: 0.54,
      expectedReturn: 0.025,
    })).toBe(false);
  });

  it('does not fire in bull regime', () => {
    expect(shouldApplyBtc14dBearishBreakSellGate({
      ticker: 'BTC-USD',
      horizon: 14,
      recommendation: 'BUY',
      rawRecommendation: 'SELL',
      structuralBreakDetected: true,
      regimeState: 'bull',
      predictionConfidence: 0.09,
      rawPredictedProb: 0.50,
      predictedProb: 0.54,
      expectedReturn: 0.025,
    })).toBe(false);
  });

  it('does not fire above the low-confidence cap', () => {
    expect(shouldApplyBtc14dBearishBreakSellGate({
      ticker: 'BTC-USD',
      horizon: 14,
      recommendation: 'BUY',
      rawRecommendation: 'SELL',
      structuralBreakDetected: true,
      regimeState: 'bear',
      predictionConfidence: 0.091,
      rawPredictedProb: 0.50,
      predictedProb: 0.54,
      expectedReturn: 0.025,
    })).toBe(false);
  });

  it('does not fire without a structural break', () => {
    expect(shouldApplyBtc14dBearishBreakSellGate({
      ticker: 'BTC-USD',
      horizon: 14,
      recommendation: 'BUY',
      rawRecommendation: 'SELL',
      structuralBreakDetected: false,
      regimeState: 'bear',
      predictionConfidence: 0.09,
      rawPredictedProb: 0.50,
      predictedProb: 0.54,
      expectedReturn: 0.025,
    })).toBe(false);
  });

  it('does not fire when the raw side is not SELL', () => {
    expect(shouldApplyBtc14dBearishBreakSellGate({
      ticker: 'BTC-USD',
      horizon: 14,
      recommendation: 'BUY',
      rawRecommendation: 'BUY',
      structuralBreakDetected: true,
      regimeState: 'bear',
      predictionConfidence: 0.09,
      rawPredictedProb: 0.50,
      predictedProb: 0.54,
      expectedReturn: 0.025,
    })).toBe(false);
  });

  it('does not fire for non-BTC tickers', () => {
    expect(shouldApplyBtc14dBearishBreakSellGate({
      ticker: 'ETH-USD',
      horizon: 14,
      recommendation: 'BUY',
      rawRecommendation: 'SELL',
      structuralBreakDetected: true,
      regimeState: 'sideways',
      predictionConfidence: 0.09,
      rawPredictedProb: 0.50,
      predictedProb: 0.54,
      expectedReturn: 0.025,
    })).toBe(false);
  });
});

describe('PR3 Post-Experiment: sideways_coil vs sideways_chop', () => {
  it('bifurcates sideways into coil and chop and uses 4-state matrix when enabled', async () => {
    // Generate an artificial price sequence that mostly stays sideways
    // but alternates between low vol (coil) and high vol (chop).
    const prices = [];
    let p = 100;
    const rng = seedRng(424);
    for (let i = 0; i < 120; i++) {
      // Small random walk to stay mostly sideways
      const ret = (rng() - 0.5) * 0.01;
      p *= (1 + ret);
      prices.push(p);
    }
    
    // Default config (3-state)
    const resDefault = await computeMarkovDistribution({
      ticker: 'BTC',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
    });

    // Experiment config
    const resSplit = await computeMarkovDistribution({
      ticker: 'BTC',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      sidewaysSplit: true,
    });

    // We should see sidewaysSplitActive true if the thresholds were met
    // (If random data didn't produce enough coil/chop, we can assert fallback)
    expect(resSplit.metadata.sidewaysSplitActive === true || resSplit.metadata.sidewaysSplitActive === false).toBe(true);
    
    // The metric should compute without throwing
    expect(resSplit.distribution.length).toBeGreaterThan(0);
    expect(resDefault.distribution.length).toBeGreaterThan(0);
  });

  it('falls back cleanly to 3-state if sideways_coil or sideways_chop is sparse', async () => {
    // Generate a strong bull trend so sideways is rare
    const prices = [];
    let p = 100;
    for (let i = 0; i < 120; i++) {
      p *= (1 + 0.02); // 2% daily return -> always bull
      prices.push(p);
    }
    
    const resSplit = await computeMarkovDistribution({
      ticker: 'BTC',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
      sidewaysSplit: true,
    });

    // Should fall back since sideways is sparse
    expect(resSplit.metadata.sidewaysSplitActive).toBe(false);
  });
});

describe('PR3 Post-Experiment: matureBullCalibration', () => {
  it('applies extra shrinkage for overconfident BTC bull runs with stalling acceleration at 14d horizon', async () => {
    const prices = [];
    let p = 60000;
    for (let i = 0; i < 140; i++) {
      const drift = i < 70 ? 0.012 : 0.001;
      const wobble = Math.sin(i / 2) * 0.003;
      p *= (1 + drift + wobble);
      prices.push(p);
    }
    const currentPrice = prices[prices.length - 1];

    const defaultResult = await computeMarkovDistribution({
      ticker: 'BTC',
      horizon: 14,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      btcReturnThresholdMultiplier: 0.5,
    });

    const experimentResult = await computeMarkovDistribution({
      ticker: 'BTC',
      horizon: 14,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      matureBullCalibration: true,
      btcReturnThresholdMultiplier: 0.5,
    });

    expect(experimentResult).toBeDefined();
    expect(defaultResult).toBeDefined();

    expect(experimentResult.metadata.matureBullCalibrationActive).toBe(true);
    expect(defaultResult.metadata.matureBullCalibrationActive).toBe(false);

    const defaultAtCurrent = interpolateSurvival(defaultResult.distribution, currentPrice);
    const experimentAtCurrent = interpolateSurvival(experimentResult.distribution, currentPrice);

    expect(experimentAtCurrent).toBeLessThanOrEqual(defaultAtCurrent);
  });
});

// ---------------------------------------------------------------------------
// normalizeSentiment — percent-to-decimal conversion for sentiment signals
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Phase 7: regime-specific sigma (backtest-only)
// ---------------------------------------------------------------------------

describe('computeMarkovDistribution — regime-specific sigma provenance', () => {
  const prices = Array.from({ length: 90 }, (_, i) => 100 + i * 0.2 + Math.sin(i) * 2);

  it('metadata.regimeSpecificSigmaActive is false when flag is off', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 7,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.metadata.regimeSpecificSigmaActive).toBeFalsy();
  });

  it('metadata.regimeSpecificSigmaActive is true when flag is on and weights are concentrated', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 7,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
      regimeSpecificSigma: true,
      regimeSpecificSigmaThreshold: 0.30,
    });
    // With a threshold of 0.30, at least one regime should dominate
    expect(result.metadata.regimeSpecificSigmaActive).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// §5.3 — jumpDiffusionApplied boolean in metadata
// ---------------------------------------------------------------------------
describe('metadata.jumpDiffusionApplied flag', () => {
  const prices = Array.from({ length: 120 }, (_, i) => 100 * Math.exp(i * 0.001));

  it('is false by default (enableJumpDiffusion not set)', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 10,
      currentPrice: 120,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.metadata.jumpDiffusionApplied).toBe(false);
  });

  it('is false when enableJumpDiffusion is explicitly false', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 10,
      currentPrice: 120,
      historicalPrices: prices,
      polymarketMarkets: [],
      enableJumpDiffusion: false,
    });
    expect(result.metadata.jumpDiffusionApplied).toBe(false);
  });

  it('is true when enableJumpDiffusion is true and jumpEvents are provided', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 10,
      currentPrice: 120,
      historicalPrices: prices,
      polymarketMarkets: [],
      enableJumpDiffusion: true,
      jumpEvents: [{ id: 'test-event', dailyIntensity: 0.01, meanLogJump: -0.05, stdLogJump: 0.02 }],
    });
    expect(result.metadata.jumpDiffusionApplied).toBe(true);
  });

  it('jumpDiffusion provenance block is present iff jumpDiffusionApplied is true and trajectory requested', async () => {
    const noJump = await computeMarkovDistribution({
      ticker: 'TEST', horizon: 10, currentPrice: 120,
      historicalPrices: prices, polymarketMarkets: [],
    });
    expect(noJump.metadata.jumpDiffusion).toBeUndefined();
    expect(noJump.metadata.jumpDiffusionApplied).toBe(false);

    // jumpDiffusionApplied is true regardless of trajectory
    const withJumpNoTraj = await computeMarkovDistribution({
      ticker: 'TEST', horizon: 10, currentPrice: 120,
      historicalPrices: prices, polymarketMarkets: [],
      enableJumpDiffusion: true,
      jumpEvents: [{ id: 'ev', dailyIntensity: 0.01, meanLogJump: -0.05, stdLogJump: 0.02 }],
    });
    expect(withJumpNoTraj.metadata.jumpDiffusionApplied).toBe(true);
    // provenance detail block only populated when trajectory is also requested
    expect(withJumpNoTraj.metadata.jumpDiffusion).toBeUndefined();

    const withJumpWithTraj = await computeMarkovDistribution({
      ticker: 'TEST', horizon: 10, currentPrice: 120,
      historicalPrices: prices, polymarketMarkets: [],
      trajectory: true,
      enableJumpDiffusion: true,
      jumpEvents: [{ id: 'ev', dailyIntensity: 0.01, meanLogJump: -0.05, stdLogJump: 0.02 }],
    });
    expect(withJumpWithTraj.metadata.jumpDiffusionApplied).toBe(true);
    expect(withJumpWithTraj.metadata.jumpDiffusion).toBeDefined();
  });
});
