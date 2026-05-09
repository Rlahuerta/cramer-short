import { describe, expect, it } from 'bun:test';
import type { ToolCallRecord } from './scratchpad.js';
import {
  buildForcedForecastArbiterArgs,
  shouldForceForecastArbitrator,
} from './agent.js';

describe('forecast arbitrator routing', () => {
  it('forces schema-safe arbitrator args for explicit BTC arbitrator-verdict trade-plan prompts', () => {
    const toolCalls: ToolCallRecord[] = [
      {
        tool: 'get_market_data',
        args: { query: 'Current crypto price snapshot for BTC' },
        result: JSON.stringify({
          data: {
            get_crypto_price_snapshot_BTC: {
              ticker: 'BTC',
              price: 80387.93,
            },
          },
        }),
      },
      {
        tool: 'markov_distribution',
        args: { ticker: 'BTC-USD', horizon: 1, trajectory: true, trajectoryDays: 1 },
        result: JSON.stringify({
          data: {
            _tool: 'markov_distribution',
            status: 'ok',
            canonical: {
              scenarios: {
                expectedReturn: 0.005,
                pUp: 0.56,
                buckets: [
                  { label: 'Flat +/-3%', probability: 0.4 },
                ],
              },
              actionSignal: { expectedReturn: 0.005, confidence: 'MEDIUM' },
              diagnostics: {
                predictionConfidence: 0.32,
                structuralBreakDetected: false,
              },
            },
            distribution: [
              { price: 79_000, probability: 0.95 },
              { price: 82_000, probability: 0.05 },
            ],
          },
        }),
      },
      {
        tool: 'polymarket_forecast',
        args: { ticker: 'BTC', horizon_days: 1, current_price: 80387.93, sentiment_score: 0 },
        result: JSON.stringify({
          data: {
            forecastReturn: 0.006,
            result: 'Polymarket Forecast: BTC | Horizon: 1 days | Grade: A (83/100)\nWill Bitcoin be above $81,000 tomorrow?: 64% YES',
          },
        }),
      },
      {
        tool: 'get_onchain_crypto',
        args: { ticker: 'BTC', metrics: ['market', 'sentiment'] },
        result: JSON.stringify({ data: { result: 'On-chain market context loaded.' } }),
      },
    ];

    const query = 'BTC-USD 24h forecast. Live BTC quote first, then sentiment, on-chain, Markov, Polymarket, rates, arbitrator. BTC only. No skill or forecast_lab_run. Return the same 10 blocks: Executive Trade Decision, Inputs Used, Markov Raw Forecast, Markov 9-Bucket Terminal Distribution, Polymarket Raw Forecast, Markov vs Polymarket Comparison, On-chain / Sentiment / Macro Check, Structural Break Diagnostic, Final Arbitrator Verdict, Final BTC Trade Plan.';

    expect(shouldForceForecastArbitrator(query, toolCalls)).toBe(true);
    expect(buildForcedForecastArbiterArgs(query, toolCalls)).toMatchObject({
      ticker: 'BTC',
      horizon_days: 1,
      current_price: 80387.93,
      markov: {
        forecast_return: 0.005,
        p_up: 0.56,
        confidence: 0.32,
        structural_break: false,
        flat_probability: 0.4,
        ci_low: 79_000,
        ci_high: 82_000,
      },
      polymarket: {
        forecast_return: 0.006,
        quality_score: 83,
        markets: [
          {
            question: 'Will Bitcoin be above $81,000 tomorrow?',
            probability: 0.64,
          },
        ],
      },
      whale: {
        direction: 'neutral',
        confidence: 0.35,
      },
    });
  });
});
