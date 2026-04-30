import { describe, expect } from 'bun:test';
import { integrationIt } from '../../utils/test-guards.js';
import { forecastArbitratorTool } from './forecast-arbitrator.js';

function parseResult(raw: unknown) {
  return JSON.parse(raw as string) as {
    data: {
      result: {
        verdict: string;
        shouldEnterNow: boolean;
        semanticSummary: { primaryPolymarketSemantics: string };
      };
    };
  };
}

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
});
