import { describe, expect, it } from 'bun:test';
import {
  arbitrateForecast,
  classifyPolymarketQuestion,
  extractPriceLevels,
  forecastArbitratorTool,
} from './forecast-arbitrator.js';
import type {
  ForecastArbiterInput,
  ForecastArbiterResult,
  ForecastMarketEvidence,
  ForecastMarketSemantics,
} from './forecast-arbitrator.js';

type PlannedPolicyLevel = 'full' | 'context-only' | 'abstain';

interface PlannedConformalDiagnostics {
  applied: boolean;
  radius: number;
  coverageEstimate: number | null;
  mode: 'normal' | 'break';
}

interface PlannedPolicy {
  level: PlannedPolicyLevel;
  horizonEligible: boolean;
  tradeEligible: boolean;
  reasons: string[];
}

type PlannedMarkovInput = NonNullable<ForecastArbiterInput['markov']> & {
  conformal?: PlannedConformalDiagnostics;
};

type PlannedPolymarketInput = Omit<NonNullable<ForecastArbiterInput['polymarket']>, 'markets'> & {
  markets?: PlannedForecastMarketEvidence[];
};

type PlannedWhaleInput = NonNullable<ForecastArbiterInput['whale']>;

type PlannedForecastMarketEvidence = ForecastMarketEvidence & {
  semantics?: ForecastMarketSemantics;
};

type PlannedForecastArbiterInput = Omit<ForecastArbiterInput, 'markov' | 'polymarket' | 'whale'> & {
  markov: PlannedMarkovInput;
  polymarket?: PlannedPolymarketInput;
  whale?: PlannedWhaleInput;
};

type PlannedForecastArbiterResult = ForecastArbiterResult & {
  policy: PlannedPolicy;
  rawEvidence: {
    markov: PlannedMarkovInput | null;
    polymarket: PlannedPolymarketInput | null;
    whale: PlannedWhaleInput | null;
  };
};

function parseToolResult(raw: unknown) {
  return JSON.parse(raw as string) as { data: { result: PlannedForecastArbiterResult } };
}

function arbitratePlannedForecast(input: PlannedForecastArbiterInput): PlannedForecastArbiterResult {
  return arbitrateForecast(input) as PlannedForecastArbiterResult;
}

function expectPlannedPolicy(
  result: PlannedForecastArbiterResult,
  expected: Pick<PlannedPolicy, 'level' | 'horizonEligible' | 'tradeEligible'>,
) {
  expect(result).toHaveProperty('policy');
  expect(result.policy).toMatchObject(expected);
  expect(Array.isArray(result.policy.reasons)).toBe(true);
}

function buildTerminalMarket(
  question: string,
  probability: number,
  semantics: ForecastMarketSemantics = 'terminal',
): PlannedForecastMarketEvidence {
  return { question, probability, semantics };
}

function buildAlignedFullGuidanceFixture(): PlannedForecastArbiterInput {
  return {
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
        buildTerminalMarket('Will BTC be above $77,000 on May 7?', 0.67),
      ],
      summary: 'Terminal YES markets still lean higher into the requested horizon.',
    },
    whale: {
      direction: 'long',
      confidence: 0.7,
      summary: 'Whale desks have been net-accumulating on dips.',
    },
  };
}

function buildDirectionalDivergenceContextOnlyFixture(): PlannedForecastArbiterInput {
  const aligned = buildAlignedFullGuidanceFixture();
  return {
    ...aligned,
    polymarket: {
      forecast_return: -0.009,
      confidence: aligned.polymarket?.confidence,
      quality_score: aligned.polymarket?.quality_score,
      markets: [
        buildTerminalMarket('Will BTC finish below $75,000 on May 7?', 0.59),
      ],
      summary: 'Prediction markets lean lower while Markov still points modestly higher.',
    },
  };
}

function buildBreakModeConformalFixture(): PlannedForecastArbiterInput {
  const aligned = buildAlignedFullGuidanceFixture();
  return {
    ...aligned,
    markov: {
      ...aligned.markov,
      structural_break: true,
      summary: 'Break regime detected; treat drift as regime context only.',
      conformal: {
        applied: true,
        radius: 0.088,
        coverageEstimate: 0.61,
        mode: 'break',
      },
    },
  };
}

function buildMissingTrustedSupportAbstainFixture(): PlannedForecastArbiterInput {
  return {
    ticker: 'BTC',
    horizon_days: 14,
    current_price: 76_000,
    leverage: 3,
    markov: {
      forecast_return: 0.001,
      p_up: 0.51,
      confidence: 0.17,
      structural_break: true,
      flat_probability: 0.86,
      ci_low: 69_800,
      ci_high: 82_400,
      summary: 'No trusted terminal support remains after the break.',
      conformal: {
        applied: true,
        radius: 0.132,
        coverageEstimate: 0.54,
        mode: 'break',
      },
    },
    whale: {
      direction: 'neutral',
      confidence: 0.25,
      summary: 'No whale confirmation is available.',
    },
  };
}

describe('forecast arbitrator', () => {
  it('classifies touch/barrier markets separately from terminal forecast markets', () => {
    expect(classifyPolymarketQuestion('Will Bitcoin dip to $75,000 in April?')).toBe('barrier_touch');
    expect(classifyPolymarketQuestion('Will BTC be above $80,000 on May 1?')).toBe('terminal');
    expect(classifyPolymarketQuestion('Will BTC stay above $70K through April?')).toBe('path_dependent');
  });

  it('extracts BTC-style price levels from market questions', () => {
    expect(extractPriceLevels('Will Bitcoin dip to $75,000 or reach $80K?')).toEqual([75_000, 80_000]);
  });

  it('rejects an immediate 10x trade when Markov and Polymarket diverge in a flat-dominant market', () => {
    const result = arbitrateForecast({
      ticker: 'BTC',
      horizon_days: 1,
      current_price: 76_029.21,
      leverage: 10,
      markov: {
        forecast_return: 0.0041,
        p_up: 0.55,
        confidence: 0.274,
        structural_break: true,
        flat_probability: 0.828,
        ci_low: 72_095,
        ci_high: 78_116,
      },
      polymarket: {
        forecast_return: -0.0121,
        quality_score: 83,
        markets: [
          { question: 'Will Bitcoin dip to $75,000 in April?', probability: 1 },
        ],
      },
      whale: {
        direction: 'neutral',
        confidence: 0.35,
        summary: 'No whale transactions detected.',
      },
    });

    expect(result.verdict).toBe('NO_TRADE');
    expect(result.shouldEnterNow).toBe(false);
    expect(result.semanticSummary.primaryPolymarketSemantics).toBe('barrier_touch');
    expect(result.semanticSummary.reconciliation).toContain('Both can be true');
    expect(result.rationale.join(' ')).toContain('Markov and Polymarket point in opposite directions');
  });

  it('can produce an immediate long when evidence is aligned and leverage is modest', () => {
    const result = arbitrateForecast({
      ticker: 'BTC',
      horizon_days: 1,
      current_price: 76_000,
      leverage: 2,
      markov: {
        forecast_return: 0.012,
        p_up: 0.62,
        confidence: 0.7,
        flat_probability: 0.45,
        ci_low: 74_500,
        ci_high: 78_500,
      },
      polymarket: {
        forecast_return: 0.01,
        quality_score: 75,
        markets: [
          { question: 'Will BTC be above $76,500 on May 1?', probability: 0.63 },
        ],
      },
      whale: { direction: 'long', confidence: 0.6 },
    });

    expect(result.verdict).toBe('LONG');
    expect(result.shouldEnterNow).toBe(true);
    expect(result.preferredDirection).toBe('long');
  });

  it('promotes aligned high-confidence evidence to full guidance policy', () => {
    const result = arbitratePlannedForecast(buildAlignedFullGuidanceFixture());

    expect(result.verdict).toBe('LONG');
    expect(result.shouldEnterNow).toBe(true);
    expectPlannedPolicy(result, {
      level: 'full',
      horizonEligible: true,
      tradeEligible: true,
    });
  });

  it('keeps the horizon eligible but blocks trade entry when only the directional evidence diverges', () => {
    const result = arbitratePlannedForecast(buildDirectionalDivergenceContextOnlyFixture());

    expect(result.shouldEnterNow).toBe(false);
    expectPlannedPolicy(result, {
      level: 'context-only',
      horizonEligible: true,
      tradeEligible: false,
    });
  });

  it('abstains when trusted terminal support is missing rather than merely downgrading trade entry', () => {
    const result = arbitratePlannedForecast(buildMissingTrustedSupportAbstainFixture());

    expect(result.shouldEnterNow).toBe(false);
    expectPlannedPolicy(result, {
      level: 'abstain',
      horizonEligible: false,
      tradeEligible: false,
    });
  });

  it('threads break-aware conformal diagnostics into the preserved raw Markov evidence', () => {
    const fixture = buildBreakModeConformalFixture();
    const result = arbitratePlannedForecast(fixture);

    expectPlannedPolicy(result, {
      level: 'context-only',
      horizonEligible: true,
      tradeEligible: false,
    });
    expect(result).toMatchObject({
      rawEvidence: {
        markov: expect.objectContaining({
          forecast_return: fixture.markov.forecast_return,
          summary: fixture.markov.summary,
          conformal: fixture.markov.conformal,
        }),
      },
    });
  });

  it('does not coerce malformed conformal mode input into a normal regime', async () => {
    const fixture = buildAlignedFullGuidanceFixture();
    const raw = await forecastArbitratorTool.invoke({
      ...fixture,
      markov: {
        ...fixture.markov,
        conformal: {
          applied: true,
          radius: 0.041,
          coverageEstimate: 0.93,
          mode: 'regime_shift',
        },
      },
    });

    const parsed = parseToolResult(raw);
    expect(parsed.data.result.rawEvidence.markov?.conformal?.mode).toBeUndefined();
  });

  it('keeps raw Markov, Polymarket, and whale evidence visible even when policy downgrades to context-only', () => {
    const fixture = buildDirectionalDivergenceContextOnlyFixture();
    const result = arbitratePlannedForecast(fixture);

    expectPlannedPolicy(result, {
      level: 'context-only',
      horizonEligible: true,
      tradeEligible: false,
    });
    expect(result).toMatchObject({
      rawEvidence: {
        markov: expect.objectContaining({
          forecast_return: fixture.markov.forecast_return,
          summary: fixture.markov.summary,
        }),
        polymarket: expect.objectContaining({
          forecast_return: fixture.polymarket?.forecast_return,
          summary: fixture.polymarket?.summary,
        }),
        whale: expect.objectContaining({
          summary: fixture.whale?.summary,
        }),
      },
    });
  });

  it('tool output preserves raw Markov, Polymarket, and whale evidence', async () => {
    const raw = await forecastArbitratorTool.func({
      ticker: 'BTC',
      horizon_days: 1,
      current_price: 76_000,
      leverage: 10,
      markov: { forecast_return: 0.004, confidence: 0.27 },
      polymarket: {
        forecast_return: -0.012,
        quality_score: 83,
        markets: [{ question: 'Will Bitcoin dip to $75,000 in April?', probability: 1 }],
      },
      whale: { direction: 'neutral', summary: 'No confirmed whale signal.' },
    }, undefined);

    const parsed = parseToolResult(raw);
    expect(parsed.data.result.rawEvidence.markov?.forecast_return).toBe(0.004);
    expect(parsed.data.result.rawEvidence.polymarket?.forecast_return).toBe(-0.012);
    expect(parsed.data.result.rawEvidence.whale?.summary).toContain('No confirmed whale signal');
  });

  it('accepts LLM-shaped schema inputs with stringified numbers, nulls, and uppercase directions', async () => {
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
        summary: null,
      },
      polymarket: {
        forecast_return: '-0.0121',
        quality_score: '83',
        quality_grade: 'A',
        markets: [
          {
            question: 'Will Bitcoin dip to $75,000 in April?',
            probability: '1',
            semantics: null,
            price: null,
          },
        ],
      },
      whale: {
        direction: 'NEUTRAL',
        confidence: null,
        summary: { observed: 'No whale transactions detected.' },
      },
    };
    const raw = await forecastArbitratorTool.invoke(payload);

    const parsed = parseToolResult(raw);
    expect(parsed.data.result.ticker).toBe('BTC-USD');
    expect(parsed.data.result.leverage).toBe(10);
    expect(parsed.data.result.verdict).toBe('NO_TRADE');
    expect(parsed.data.result.rawEvidence.whale?.direction).toBe('neutral');
  });
});
