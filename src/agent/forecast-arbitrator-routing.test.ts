import { describe, expect, it } from 'bun:test';
import type { ToolCallRecord } from './scratchpad.js';
import {
  buildForcedForecastArbiterArgs,
  buildForcedCryptoForecastMarkovArgs,
  buildForcedPolymarketForecastArgs,
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

  it('forces deterministic schema-safe arbitrator args for briefing-style BTC prompts', () => {
    const toolCalls: ToolCallRecord[] = [
      {
        tool: 'get_market_data',
        args: { query: 'Current crypto price snapshot for BTC' },
        result: JSON.stringify({
          data: {
            get_crypto_price_snapshot_BTC: {
              ticker: 'BTC',
              price: 64_406.73,
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
                expectedReturn: 0.00408,
                pUp: 0.55,
                buckets: [
                  { label: 'Flat +/-3%', probability: 0.828 },
                ],
              },
              actionSignal: { expectedReturn: 0.006, confidence: 'MEDIUM' },
              diagnostics: {
                predictionConfidence: 0.274,
                structuralBreakDetected: true,
              },
            },
            distribution: [
              { price: 62_000, probability: 0.95 },
              { price: 68_000, probability: 0.05 },
            ],
          },
        }),
      },
      {
        tool: 'polymarket_forecast',
        args: { ticker: 'BTC', horizon_days: 1, current_price: 64_406.73, markov_return: 0.00408 },
        result: JSON.stringify({
          data: {
            forecastReturn: -0.0121,
            result: 'Polymarket Forecast: BTC | Horizon: 1 days | Grade: A (83/100)\nWill Bitcoin dip to $64,000 tomorrow?: 100% YES',
          },
        }),
      },
      {
        tool: 'get_onchain_crypto',
        args: { ticker: 'BTC', metrics: ['market', 'sentiment'] },
        result: JSON.stringify({ data: { result: 'No whale transactions detected.' } }),
      },
    ];

    const query = `BTC live trading briefing for the next 24 hours / 1 trading day.

BTC / BTC-USD only. No GOLD, GLD, commodities, ETH, SOL, or proxy context.

Gather live inputs first using:
- get_market_data
- social_sentiment
- get_onchain_crypto
- markov_distribution for BTC-USD with horizon=1, trajectory=true, trajectoryDays=1
- the Polymarket tool for BTC using the live BTC quote and 1-day horizon
- rates / macro context
- forecast arbitrator if available

Then return this 10-block format:
1. Executive Trade Decision
9. Final Arbitrator Verdict
10. Final BTC Trade Plan`;

    expect(shouldForceForecastArbitrator(query, toolCalls)).toBe(true);
    expect(buildForcedForecastArbiterArgs(query, toolCalls)).toMatchObject({
      ticker: 'BTC',
      horizon_days: 1,
      current_price: 64_406.73,
      markov: {
        forecast_return: 0.00408,
        p_up: 0.55,
        confidence: 0.274,
        structural_break: true,
        flat_probability: 0.828,
        ci_low: 62_000,
        ci_high: 68_000,
      },
      polymarket: {
        forecast_return: -0.0121,
        quality_score: 83,
        markets: [
          {
            question: 'Will Bitcoin dip to $64,000 tomorrow?',
            probability: 1,
          },
        ],
      },
      whale: {
        direction: 'neutral',
        confidence: 0.35,
      },
    });
  });

  it('does NOT force the arbitrator for a briefing prompt until Polymarket has been attempted', () => {
    // Markov has run, but Polymarket has NOT. The forced arbitrator must wait —
    // otherwise it fires with polymarket: null (the reported regression).
    const toolCalls: ToolCallRecord[] = [
      {
        tool: 'markov_distribution',
        args: { ticker: 'BTC-USD', horizon: 1, trajectory: true, trajectoryDays: 1 },
        result: JSON.stringify({
          data: {
            _tool: 'markov_distribution',
            status: 'ok',
            canonical: {
              scenarios: {
                expectedReturn: 0.00408,
                pUp: 0.55,
                buckets: [{ label: 'Flat +/-3%', probability: 0.75 }],
              },
              actionSignal: { expectedReturn: 0.004, confidence: 'MEDIUM' },
              diagnostics: { predictionConfidence: 0.3, structuralBreakDetected: false },
            },
            distribution: [
              { price: 62_000, probability: 0.95 },
              { price: 68_000, probability: 0.05 },
            ],
          },
        }),
      },
    ];

    const query = `BTC live trading briefing for the next 24 hours / 1 trading day.
BTC / BTC-USD only. No GOLD, GLD, commodities, ETH, SOL, or proxy context.
Gather live inputs first using markov_distribution, polymarket_forecast, get_onchain_crypto, forecast arbitrator if available.
Then return the 10-block format with Final Arbitrator Verdict and Final BTC Trade Plan.`;

    expect(shouldForceForecastArbitrator(query, toolCalls)).toBe(false);
  });

  it('forces the full deterministic pipeline (Markov + Polymarket) for briefing prompts', () => {
    const query = `BTC live trading briefing for the next 24 hours / 1 trading day.
BTC / BTC-USD only. No GOLD, GLD, commodities, ETH, SOL, or proxy context.
Gather live inputs first using markov_distribution, polymarket_forecast, get_onchain_crypto, forecast arbitrator if available.
Then return the 10-block format with Final Arbitrator Verdict and Final BTC Trade Plan.`;

    expect(buildForcedCryptoForecastMarkovArgs(query)).toMatchObject({
      ticker: 'BTC-USD',
      horizon: 1,
      trajectory: true,
    });
    expect(buildForcedPolymarketForecastArgs(query, [])).toMatchObject({
      ticker: 'BTC',
      horizon_days: 1,
    });
  });
});
