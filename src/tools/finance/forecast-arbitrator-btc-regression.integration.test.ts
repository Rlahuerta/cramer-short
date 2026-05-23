import { describe, expect } from 'bun:test';
import fixture from './fixtures/btc-combined-arbitrator-regression.json';
import { integrationIt } from '@/utils/test-guards.js';
import { createForecastArbitratorTool } from './forecast-arbitrator.js';
import type { ForecastArbiterInput, ForecastArbiterResult } from './forecast-arbitrator.js';

function parseResult(raw: unknown) {
  return JSON.parse(raw as string) as {
    data: {
      result: ForecastArbiterResult & {
        policy: {
          level: 'full' | 'context-only' | 'abstain';
          horizonEligible: boolean;
          tradeEligible: boolean;
          reasons: string[];
        };
      };
    };
  };
}

const forecastArbitratorTool = createForecastArbitratorTool({
  recordReplayBundleCapture: () => {},
});

const BTC_ALIGNED_LONG_FIXTURE: ForecastArbiterInput = {
  ticker: 'BTC',
  horizon_days: 7,
  current_price: 76_000,
  leverage: 2,
  markov: {
    forecast_return: 0.018,
    p_up: 0.68,
    confidence: 0.84,
    structural_break: false,
    flat_probability: 0.22,
    ci_low: 74_200,
    ci_high: 79_100,
    summary: 'Trusted Markov regime remains intact.',
    conformal: {
      applied: true,
      radius: 0.041,
      coverageEstimate: 0.93,
      mode: 'normal',
    },
  },
  polymarket: {
    forecast_return: 0.015,
    confidence: 0.78,
    quality_score: 82,
    markets: [
      {
        marketId: 'pm-test-1',
        assetId: 'asset-test-1',
        question: 'Will BTC be above $77,000 on May 7?',
        probability: 0.67,
        semantics: 'terminal',
      },
    ],
    summary: 'Terminal YES markets still lean higher into the requested horizon.',
  },
  whale: {
    direction: 'long',
    confidence: 0.7,
    summary: 'Whale desks have been net-accumulating on dips.',
  },
};

const BTC_BREAK_FULL_GUIDANCE_FIXTURE: ForecastArbiterInput = {
  ...BTC_ALIGNED_LONG_FIXTURE,
  markov: {
    ...BTC_ALIGNED_LONG_FIXTURE.markov!,
    confidence: 0.78,
    structural_break: true,
    flat_probability: 0.28,
    summary: 'Break detected, but confidence and terminal support remain strong.',
    conformal: {
      applied: true,
      radius: 0.039,
      coverageEstimate: 0.91,
      mode: 'normal',
    },
  },
  whale: {
    direction: 'long',
    confidence: 0.7,
    summary: 'Whale desks remain net long through the break.',
  },
};

describe('forecast_arbitrator BTC regression fixtures', () => {
  integrationIt('keeps the canonical BTC divergence fixture stable across decision semantics and preserved evidence', async () => {
    const raw = await forecastArbitratorTool.invoke(fixture.input);
    const result = parseResult(raw).data.result;

    expect(result).toMatchObject({
      ...fixture.expected,
      confidence: 'low',
      semanticSummary: {
        primaryPolymarketSemantics: 'barrier_touch',
        counts: {
          terminal: 0,
          barrier_touch: 1,
          range: 0,
          path_dependent: 0,
          unknown: 0,
        },
        barrierPrices: [75_000],
      },
      disagreement: {
        markovDirection: 'long',
        polymarketDirection: 'short',
        whaleDirection: 'neutral',
        isDivergent: true,
      },
      conditionalPlan: {
        longTrigger: 'Wait for a sweep/touch of $75,000 followed by reclaim above that level before considering LONG.',
        shortTrigger: 'Wait for an accepted break below $75,000 and failed retest before considering SHORT.',
        invalidation: 'If price chops around $75,000 without reclaim/rejection confirmation, keep the setup as no-trade.',
      },
      rawEvidence: {
        markov: fixture.input.markov,
        polymarket: fixture.input.polymarket,
        whale: {
          direction: 'neutral',
          summary: 'No whale transactions detected.',
        },
      },
    });
    expect(result.policy.reasons).toEqual([
      'Markov and Polymarket disagree on direction, so this horizon is context-only until the signals realign.',
      'Prediction-market support is barrier/path dependent rather than a clean terminal anchor.',
      'A structural-break flag is active, so regime trust is reduced.',
      'Flat-probability is elevated, which weakens immediate directional edge.',
      '10x leverage is too unforgiving for the current forecast quality.',
      'Markov prediction confidence is too weak to treat as a standalone trade trigger.',
    ]);
    expect(result.semanticSummary.reconciliation).toContain('Both can be true');
    expect(result.disagreement.summary).toBe('Divergence: Markov is long, Polymarket is short, whales are neutral.');
    expect(result.rationale).toEqual([
      'Markov and Polymarket point in opposite directions, so the arbiter rejects a one-model trade call.',
      'The leading Polymarket evidence appears path-dependent/barrier-like, which can be true even if terminal Markov drift is flat or positive.',
      'Markov assigns high probability to a flat/range outcome, which weakens both immediate LONG and SHORT setups.',
      '10x leverage makes a normal intraday move large enough to dominate the expected edge.',
      'Whale data is neutral, so it does not break the model tie.',
    ]);
    expect(result.leverageAssessment.warning).toBe(
      'At 10x, a 1% asset move is approximately 10% position P&L before fees/funding.',
    );
    expect(result.leverageAssessment.long.directionalEdgePct).toBeCloseTo(-0.335, 12);
    expect(result.leverageAssessment.short.directionalEdgePct).toBeCloseTo(0.335, 12);
  });

  integrationIt('adds a tradeable aligned BTC long fixture so full-guidance behavior is regression-locked', async () => {
    const raw = await forecastArbitratorTool.invoke(BTC_ALIGNED_LONG_FIXTURE);
    const result = parseResult(raw).data.result;

    expect(result).toMatchObject({
      verdict: 'LONG',
      preferredDirection: 'long',
      shouldEnterNow: true,
      confidence: 'high',
      semanticSummary: {
        primaryPolymarketSemantics: 'terminal',
        counts: {
          terminal: 1,
          barrier_touch: 0,
          range: 0,
          path_dependent: 0,
          unknown: 0,
        },
        barrierPrices: [],
        reconciliation: 'Signals are comparable as terminal directional forecasts.',
      },
      disagreement: {
        markovDirection: 'long',
        polymarketDirection: 'long',
        whaleDirection: 'long',
        isDivergent: false,
        summary: 'No strong directional conflict: Markov is long, Polymarket is long, whales are long.',
      },
      policy: {
        level: 'full',
        horizonEligible: true,
        tradeEligible: true,
      },
      conditionalPlan: {
        longTrigger: 'Wait for Markov and Polymarket to align bullish or for price to reclaim the nearest failed breakdown level.',
        shortTrigger: 'Wait for Markov and Polymarket to align bearish or for price to reject the nearest resistance level.',
        invalidation: 'If confirmation does not appear, preserve the raw forecasts but avoid a directional recommendation.',
      },
      rawEvidence: {
        markov: BTC_ALIGNED_LONG_FIXTURE.markov,
        polymarket: BTC_ALIGNED_LONG_FIXTURE.polymarket,
        whale: BTC_ALIGNED_LONG_FIXTURE.whale,
      },
    });
    expect(result.policy.reasons).toEqual([
      'Evidence is aligned and regime diagnostics are healthy enough for full guidance.',
    ]);
    expect(result.rationale).toEqual([
      'Evidence is sufficiently aligned after leverage/risk adjustment.',
    ]);
    expect(result.leverageAssessment.warning).toBeNull();
    expect(result.leverageAssessment.long.directionalEdgePct).toBeCloseTo(2.857, 12);
    expect(result.leverageAssessment.short.directionalEdgePct).toBeCloseTo(-2.857, 12);
  });

  integrationIt('keeps a structural-break BTC full-guidance path stable when terminal support stays strong', async () => {
    const raw = await forecastArbitratorTool.invoke(BTC_BREAK_FULL_GUIDANCE_FIXTURE);
    const result = parseResult(raw).data.result;

    expect(result).toMatchObject({
      verdict: 'LONG',
      preferredDirection: 'long',
      shouldEnterNow: true,
      confidence: 'high',
      semanticSummary: {
        primaryPolymarketSemantics: 'terminal',
        counts: {
          terminal: 1,
          barrier_touch: 0,
          range: 0,
          path_dependent: 0,
          unknown: 0,
        },
        barrierPrices: [],
        reconciliation: 'Signals are comparable as terminal directional forecasts.',
      },
      disagreement: {
        markovDirection: 'long',
        polymarketDirection: 'long',
        whaleDirection: 'long',
        isDivergent: false,
        summary: 'No strong directional conflict: Markov is long, Polymarket is long, whales are long.',
      },
      policy: {
        level: 'full',
        horizonEligible: true,
        tradeEligible: true,
      },
      rawEvidence: {
        markov: BTC_BREAK_FULL_GUIDANCE_FIXTURE.markov,
        polymarket: BTC_BREAK_FULL_GUIDANCE_FIXTURE.polymarket,
        whale: BTC_BREAK_FULL_GUIDANCE_FIXTURE.whale,
      },
    });
    expect(result.policy.reasons).toEqual([
      'A structural-break flag is active, so regime trust is reduced.',
    ]);
    expect(result.rationale).toEqual([
      'Evidence is sufficiently aligned after leverage/risk adjustment.',
    ]);
    expect(result.leverageAssessment.warning).toBeNull();
    expect(result.leverageAssessment.long.directionalEdgePct).toBeCloseTo(2.187, 12);
    expect(result.leverageAssessment.short.directionalEdgePct).toBeCloseTo(-2.187, 12);
  });
});
