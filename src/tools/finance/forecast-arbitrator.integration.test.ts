import { describe, expect } from 'bun:test';
import { integrationIt } from '../../utils/test-guards.js';
import { createForecastArbitratorTool } from './forecast-arbitrator.js';
import type { ForecastArbiterResult, ForecastMarketSemantics } from './forecast-arbitrator.js';

type PlannedPolicyLevel = 'full' | 'context-only' | 'abstain';

interface PlannedPolicy {
  level: PlannedPolicyLevel;
  horizonEligible: boolean;
  tradeEligible: boolean;
  reasons: string[];
}

type PlannedForecastArbiterResult = ForecastArbiterResult & {
  policy: PlannedPolicy;
  semanticSummary: ForecastArbiterResult['semanticSummary'];
  rawEvidence: {
    markov?: {
      forecast_return?: number;
      confidence?: number;
      structural_break?: boolean;
      flat_probability?: number;
      ci_low?: number;
      ci_high?: number;
      conformal?: {
        mode?: 'normal' | 'break';
      };
    };
  };
};

function parseResult(raw: unknown) {
  return JSON.parse(raw as string) as {
    data: {
      result: PlannedForecastArbiterResult;
    };
  };
}

function expectPlannedPolicy(
  result: PlannedForecastArbiterResult,
  expected: Pick<PlannedPolicy, 'level' | 'horizonEligible' | 'tradeEligible'>,
) {
  expect(result).toHaveProperty('policy');
  expect(result.policy).toMatchObject(expected);
  expect(Array.isArray(result.policy.reasons)).toBe(true);
}

const forecastArbitratorTool = createForecastArbitratorTool({
  recordReplayBundleCapture: () => {},
});

describe('forecast_arbitrator integration', () => {
  integrationIt('validates the LLM-style BTC 10x divergence payload through the real tool schema', async () => {
    const payload = {
      ticker: 'BTC-USD',
      horizon_days: '1',
      current_price: '75504.42',
      leverage: '10',
      markov: {
        forecast_return: '0.004062142875039587',
        p_up: '0.55',
        confidence: '0.274',
        structural_break: 'true',
        flat_probability: '0.828',
        ci_low: '72095',
        ci_high: '78116',
      },
      polymarket: {
        forecast_return: '-0.0121',
        quality_score: '83',
        markets: [
          {
            question: 'Will Bitcoin dip to $75,000 in April?',
            probability: '1',
          },
        ],
      },
      whale: {
        direction: 'NEUTRAL',
        confidence: null,
        summary: 'No whale transactions detected.',
      },
    };
    const raw = await forecastArbitratorTool.invoke(payload);

    const parsed = parseResult(raw);
    expect(parsed.data.result.verdict).toBe('NO_TRADE');
    expect(parsed.data.result.shouldEnterNow).toBe(false);
    expect(parsed.data.result.semanticSummary.primaryPolymarketSemantics).toBe('barrier_touch');
  });

  integrationIt('accepts break-aware conformal diagnostics and returns context-only policy metadata without stripping raw evidence', async () => {
    const payload = {
      ticker: 'BTC',
      horizon_days: '7',
      current_price: '76000',
      leverage: '5',
      markov: {
        forecast_return: '0.012',
        p_up: '0.61',
        confidence: '0.58',
        structural_break: 'true',
        flat_probability: '0.74',
        ci_low: '71500',
        ci_high: '79000',
        summary: 'Break regime detected; use drift only as context.',
        conformal: {
          applied: true,
          radius: '0.088',
          coverageEstimate: '0.61',
          mode: 'break',
        },
      },
      polymarket: {
        forecast_return: '-0.009',
        confidence: '0.76',
        quality_score: '81',
        markets: [
          {
            question: 'Will BTC finish below $75,000 on May 7?',
            probability: '0.59',
            semantics: 'terminal' as ForecastMarketSemantics,
          },
        ],
      },
      whale: {
        direction: 'neutral',
        confidence: '0.35',
        summary: 'Whale positioning is mixed and does not resolve the disagreement.',
      },
    };

    const raw = await forecastArbitratorTool.invoke(payload);
    const parsed = parseResult(raw);

    expect(parsed.data.result.shouldEnterNow).toBe(false);
    expectPlannedPolicy(parsed.data.result, {
      level: 'context-only',
      horizonEligible: true,
      tradeEligible: false,
    });
    expect(parsed.data.result).toMatchObject({
      rawEvidence: {
        markov: {
          forecast_return: 0.012,
          confidence: 0.58,
          structural_break: true,
          flat_probability: 0.74,
          ci_low: 71500,
          ci_high: 79000,
        },
      },
    });
  });

  integrationIt('keeps structural-break forecasts at full policy when terminal support and diagnostics stay strong', async () => {
    const payload = {
      ticker: 'BTC',
      horizon_days: '7',
      current_price: '76000',
      leverage: '2',
      markov: {
        forecast_return: '0.018',
        p_up: '0.68',
        confidence: '0.78',
        structural_break: 'true',
        flat_probability: '0.28',
        ci_low: '74200',
        ci_high: '79100',
        summary: 'Break detected, but confidence and terminal support remain strong.',
        conformal: {
          applied: true,
          radius: '0.039',
          coverageEstimate: '0.91',
          mode: 'normal',
        },
      },
      polymarket: {
        forecast_return: '0.015',
        confidence: '0.78',
        quality_score: '82',
        markets: [
          {
            question: 'Will BTC be above $77,000 on May 7?',
            probability: '0.67',
            semantics: 'terminal' as ForecastMarketSemantics,
          },
        ],
      },
      whale: {
        direction: 'long',
        confidence: '0.7',
        summary: 'Whale desks remain net long through the break.',
      },
    };

    const raw = await forecastArbitratorTool.invoke(payload);
    const parsed = parseResult(raw);

    expect(parsed.data.result.shouldEnterNow).toBe(true);
    expectPlannedPolicy(parsed.data.result, {
      level: 'full',
      horizonEligible: true,
      tradeEligible: true,
    });
  });

  integrationIt('drops malformed conformal mode strings instead of inventing a normal regime', async () => {
    const payload = {
      ticker: 'BTC',
      horizon_days: '7',
      current_price: '76000',
      leverage: '2',
      markov: {
        forecast_return: '0.018',
        p_up: '0.68',
        confidence: '0.84',
        structural_break: 'false',
        flat_probability: '0.22',
        ci_low: '74200',
        ci_high: '79100',
        conformal: {
          applied: 'true',
          radius: '0.041',
          coverageEstimate: '0.93',
          mode: 'regime_shift',
        },
      },
      polymarket: {
        forecast_return: '0.015',
        confidence: '0.78',
        quality_score: '82',
        markets: [
          {
            question: 'Will BTC be above $77,000 on May 7?',
            probability: '0.67',
          },
        ],
      },
      whale: {
        direction: 'long',
        confidence: '0.7',
      },
    };

    const raw = await forecastArbitratorTool.invoke(payload);
    const parsed = parseResult(raw);

    expect(parsed.data.result.rawEvidence.markov?.conformal?.mode).toBeUndefined();
    expect(parsed.data.result.policy.level).toBe('full');
  });
});
