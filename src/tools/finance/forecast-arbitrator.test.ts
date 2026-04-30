import { describe, expect, it } from 'bun:test';
import {
  arbitrateForecast,
  classifyPolymarketQuestion,
  extractPriceLevels,
  forecastArbitratorTool,
} from './forecast-arbitrator.js';

function parseToolResult(raw: unknown) {
  return JSON.parse(raw as string) as { data: { result: ReturnType<typeof arbitrateForecast> } };
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
