import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import type { ArbiterReplayBundle } from '../arbiter-replay.js';
import type { ForecastArbiterInput, ForecastArbiterResult } from '../forecast-arbitrator.js';
import { compareReplayEvaluators, runArbiterReplay } from './arbiter-replay-runner.js';

function makeLabeledBundle(params: {
  capturedAt: string;
  ticker: string;
  actualBinary: 0 | 1;
  currentPrice?: number;
  semantics?: 'terminal' | 'barrier_touch' | 'range';
  withWhale?: boolean;
  evidence?: 'thin' | 'rich';
}): ArbiterReplayBundle {
  const selectedMarkets = params.semantics ? [{
    marketId: `${params.ticker}-market`,
    assetId: `${params.ticker}-asset`,
    question: params.semantics === 'range'
      ? 'Will Bitcoin be between $65,000 and $69,000 on May 8?'
      : params.semantics === 'barrier_touch'
        ? 'Will Bitcoin reach $70,000 by May 8?'
        : 'Will Bitcoin be above $70,000 on May 8?',
    probability: params.actualBinary === 1 ? 0.62 : 0.38,
    volume24h: 250000,
    endDate: '2026-05-08T00:00:00.000Z',
    semantics: params.semantics,
    extractedPriceLevels: params.semantics === 'range' ? [65000, 69000] : [70000],
  }] : [];

  return {
    capturedAt: params.capturedAt,
    ticker: params.ticker,
    horizonDays: 7,
    currentPrice: params.currentPrice ?? 68000,
    polymarket: {
      querySet: ['bitcoin price'],
      selectedMarketIds: selectedMarkets.map((market) => market.marketId),
      selectedMarkets,
      qualityScore: params.evidence === 'rich' ? 75 : 40,
      warnings: params.evidence === 'rich' ? [] : ['thin history'],
    },
    ...(params.withWhale ? {
      whale: {
        source: 'whale-alert',
        direction: 'long',
        confidence: 0.7,
        summary: 'Exchange outflows dominate.',
        observationWindowStart: '2026-05-01T00:00:00.000Z',
        observationWindowEnd: '2026-05-01T06:00:00.000Z',
        txCount: 2,
        notionalUsd: 8000000,
        txHashes: ['0xabc'],
      },
    } : {}),
    warnings: [],
    labels: {
      forecast: {
        realizedPrice: params.actualBinary === 1 ? 71000 : 66000,
        realizedReturn: params.actualBinary === 1 ? (71000 - 68000) / 68000 : (66000 - 68000) / 68000,
        actualBinary: params.actualBinary,
        labeledAt: '2026-05-08T12:00:00.000Z',
      },
      semantic: selectedMarkets.map((market) => ({
        marketId: market.marketId,
        semantics: market.semantics,
        outcome: params.actualBinary === 1 ? 'yes' : 'no',
        labeledAt: '2026-05-08T12:00:00.000Z',
      })),
    },
  };
}

function makeResult(params: {
  preferredDirection: ForecastArbiterResult['preferredDirection'];
  confidence: ForecastArbiterResult['confidence'];
  shouldEnterNow: boolean;
  divergent?: boolean;
}): ForecastArbiterResult {
  return {
    ticker: 'BTC',
    horizonDays: 7,
    currentPrice: 68000,
    leverage: 1,
    verdict: params.shouldEnterNow
      ? params.preferredDirection === 'long'
        ? 'LONG'
        : params.preferredDirection === 'short'
          ? 'SHORT'
          : 'NO_TRADE'
      : 'NO_TRADE',
    preferredDirection: params.preferredDirection,
    confidence: params.confidence,
    shouldEnterNow: params.shouldEnterNow,
    semanticSummary: {
      primaryPolymarketSemantics: 'terminal',
      counts: { terminal: 1, barrier_touch: 0, range: 0, path_dependent: 0, unknown: 0 },
      barrierPrices: [],
      reconciliation: 'test',
    },
    disagreement: {
      markovDirection: 'long',
      polymarketDirection: 'long',
      whaleDirection: 'neutral',
      isDivergent: params.divergent ?? false,
      summary: 'test disagreement',
    },
    leverageAssessment: {
      long: { directionalEdgePct: 0.01, riskAdjustedScore: 0.01, leveragePnlPct: 0.01, rr: 1.5, notes: [] },
      short: { directionalEdgePct: -0.01, riskAdjustedScore: -0.01, leveragePnlPct: -0.01, rr: 1.5, notes: [] },
      warning: null,
    },
    conditionalPlan: {
      longTrigger: null,
      shortTrigger: null,
      invalidation: null,
    },
    policy: {
      level: 'full',
      horizonEligible: true,
      tradeEligible: params.shouldEnterNow,
      reasons: [],
    },
    rationale: [],
    rawEvidence: {
      markov: null,
      polymarket: null,
      whale: null,
    },
  };
}

describe('arbiter replay runner', () => {
  let savedFetch: typeof globalThis.fetch;

  beforeEach(() => {
    savedFetch = globalThis.fetch;
  });

  afterEach(() => {
    globalThis.fetch = savedFetch;
  });

  it('feeds identical bundle rows into baseline and candidate evaluators', () => {
    const baselineInputs: ForecastArbiterInput[] = [];
    const candidateInputs: ForecastArbiterInput[] = [];
    const bundles = [
      makeLabeledBundle({ capturedAt: '2026-05-01T00:00:00.000Z', ticker: 'BTC', actualBinary: 1, semantics: 'terminal', withWhale: true, evidence: 'rich' }),
      makeLabeledBundle({ capturedAt: '2026-05-02T00:00:00.000Z', ticker: 'BTC', actualBinary: 0, semantics: 'barrier_touch', evidence: 'thin' }),
    ];

    const run = runArbiterReplay({
      bundles,
      baselineEvaluator: {
        name: 'baseline',
        evaluate(input) {
          baselineInputs.push(input);
          return makeResult({ preferredDirection: 'long', confidence: 'medium', shouldEnterNow: true });
        },
      },
      candidateEvaluator: {
        name: 'candidate',
        evaluate(input) {
          candidateInputs.push(input);
          return makeResult({ preferredDirection: 'short', confidence: 'medium', shouldEnterNow: true });
        },
      },
    });

    expect(run.labeledBundleCount).toBe(2);
    expect(baselineInputs).toEqual(candidateInputs);
    expect(run.baseline.totalRows).toBe(2);
    expect(run.candidate?.totalRows).toBe(2);
  });

  it('never calls live fetch while replaying fixed bundles', () => {
    globalThis.fetch = (() => {
      throw new Error('live fetch should not be called during replay');
    }) as unknown as typeof fetch;

    const run = runArbiterReplay({
      bundles: [makeLabeledBundle({ capturedAt: '2026-05-01T00:00:00.000Z', ticker: 'BTC', actualBinary: 1, semantics: 'terminal' })],
      baselineEvaluator: {
        name: 'baseline',
        evaluate() {
          return makeResult({ preferredDirection: 'long', confidence: 'high', shouldEnterNow: true, divergent: true });
        },
      },
    });

    expect(run.baseline.totalRows).toBe(1);
    expect(run.baseline.disagreementSlice.totalRows).toBe(1);
  });

  it('fails the acceptance gate when a candidate buys accuracy by abstaining too often', () => {
    const gate = compareReplayEvaluators(
      {
        name: 'baseline',
        totalRows: 10,
        tradedRows: 8,
        abstainRate: 0.2,
        directionalAccuracy: 0.625,
        brierScore: 0.24,
        semanticBucketCalibration: {},
        disagreementSlice: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
        whaleSupportSlices: {
          withSupport: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
          withoutSupport: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
        },
        polymarketEvidenceSlices: {
          none: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
          thin: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
          rich: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
        },
        rows: [],
      },
      {
        name: 'candidate',
        totalRows: 10,
        tradedRows: 3,
        abstainRate: 0.7,
        directionalAccuracy: 1,
        brierScore: 0.23,
        semanticBucketCalibration: {},
        disagreementSlice: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
        whaleSupportSlices: {
          withSupport: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
          withoutSupport: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
        },
        polymarketEvidenceSlices: {
          none: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
          thin: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
          rich: { totalRows: 0, tradedRows: 0, directionalAccuracy: null, brierScore: null, abstainRate: 0 },
        },
        rows: [],
      },
      { maxAbstainRateIncrease: 0.1 },
    );

    expect(gate.passed).toBe(false);
    expect(gate.reasons).toContain('Candidate abstain rate increased beyond the allowed tolerance.');
  });
});
