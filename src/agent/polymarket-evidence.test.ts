import { describe, expect, it } from 'bun:test';
import type { ToolCallRecord } from './scratchpad.js';
import {
  buildForcedCryptoForecastMarkovArgs,
  buildForcedForecastArbiterArgs,
  buildForcedGoldCombinedForecastArbiterArgs,
  buildForcedMarkovArgs,
  buildForcedNonCryptoPolymarketForecastArgs,
  buildForcedPolymarketForecastArgs,
  detectBtcShortHorizonDisagreement,
  shouldInjectBtcShortHorizonMixedEvidencePrompt,
} from './query-router.js';

// What polymarket_forecast actually returns with the live calibrator off (the default): no `forecastReturn`, a
// Polymarket-only `rawForecastReturn`, a `blendedForecastReturn` that already contains the Markov return, the
// printed "Raw Polymarket forecast" / "Blended forecast" lines, and a theme table whose questions are truncated.
const FULL_QUESTION = 'Will the price of Bitcoin be above $81,000 on November 27, 2026?';
const polymarketPayload = {
  data: {
    result: [
      'Polymarket Forecast: BTC | Horizon: 1 days | Grade: A (83/100)',
      'Raw Polymarket forecast: $79,415.00  (-1.21%)',
      'Blended forecast: $80,550.00  (+0.20%)',
      '│ crypto  │ ↓ │ Will the price of Bitcoin be above $81,000 on N… │ 61% │',
    ].join('\n'),
    rawForecastReturn: -0.0121,
    blendedForecastReturn: 0.002,
    qualityScore: 83,
    qualityGrade: 'A',
    markets: [{ question: FULL_QUESTION, probability: 0.64 }],
  },
};
const markovPayload = {
  data: {
    _tool: 'markov_distribution',
    status: 'ok',
    canonical: {
      scenarios: { expectedReturn: 0.02, pUp: 0.6 },
      actionSignal: { expectedReturn: 0.02, confidence: 'MEDIUM' },
      diagnostics: { predictionConfidence: 0.4, structuralBreakDetected: false, markovWeight: 1 },
    },
  },
};
const toolCalls: ToolCallRecord[] = [
  { tool: 'get_market_data', args: { query: 'Current crypto price snapshot for BTC' }, result: JSON.stringify({ data: { get_crypto_price_snapshot_BTC: { ticker: 'BTC', price: 80387.93 } } }) },
  { tool: 'markov_distribution', args: { ticker: 'BTC-USD', horizon: 1 }, result: JSON.stringify(markovPayload) },
  { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 1, current_price: 80387.93, markov_return: 0.02 }, result: JSON.stringify(polymarketPayload) },
  { tool: 'get_onchain_crypto', args: { ticker: 'BTC', metrics: ['market'] }, result: JSON.stringify({ data: { result: 'On-chain context.' } }) },
];
const query = 'BTC-USD 24h forecast with Markov, Polymarket and the forecast arbitrator verdict, then the trade plan.';

describe('Polymarket evidence from the real polymarket_forecast payload', () => {
  it('gives the arbitrator the Polymarket-only return, not the Markov-blended one', () => {
    expect(buildForcedForecastArbiterArgs(query, toolCalls)?.polymarket?.forecast_return).toBe(-0.0121);
  });

  it('takes the markets from the structured payload, with their full questions', () => {
    expect(buildForcedForecastArbiterArgs(query, toolCalls)?.polymarket?.markets).toEqual([
      { question: FULL_QUESTION, probability: 0.64 },
    ]);
  });

  it('detects a BTC short-horizon disagreement between a bullish Markov and a bearish Polymarket', () => {
    expect(detectBtcShortHorizonDisagreement('Provide a BTC forecast for the next 1 day', toolCalls)).toBe(true);
  });

  it('injects the mixed-evidence guard from the JSON results', () => {
    const fullToolResults = [
      '### markov_distribution(ticker=BTC-USD)', JSON.stringify(markovPayload),
      '### polymarket_forecast(ticker=BTC)', JSON.stringify(polymarketPayload),
    ].join('\n');
    expect(shouldInjectBtcShortHorizonMixedEvidencePrompt('Provide a BTC forecast for the next 7 days', fullToolResults)).toBe(true);
  });
});

describe('one horizon for Markov and Polymarket on a crypto query', () => {
  // Crypto trades every day, so its daily bars and Polymarket's resolution dates both count calendar days.
  for (const [query, days] of [
    ['Provide a BTC forecast for next week', 7],
    ['Provide a BTC forecast for the next 2 weeks', 14],
  ] as const) {
    it(`${query} -> ${days} days for both tools`, () => {
      expect(buildForcedCryptoForecastMarkovArgs(query)?.horizon).toBe(days);
      expect(buildForcedPolymarketForecastArgs(query, [])?.horizon_days).toBe(days);
    });
  }
});

describe('Polymarket counts calendar days on a non-crypto query', () => {
  // Markov steps through trading days; Polymarket matches markets by calendar days to resolution.
  it('Forecast GLD for the next 2 weeks -> Markov 10 trading days, Polymarket 14 calendar days', () => {
    const query = 'Forecast GLD for the next 2 weeks';
    expect(buildForcedMarkovArgs(query)?.horizon).toBe(10);
    expect(buildForcedNonCryptoPolymarketForecastArgs(query, [])?.horizon_days).toBe(14);
  });

  it('the gold arbitrator finds the Polymarket run made at the calendar-day horizon', () => {
    const query = 'Give me a gold forecast for the next 2 weeks using Markov and Polymarket';
    const toolCalls: ToolCallRecord[] = [
      {
        tool: 'markov_distribution',
        args: { ticker: 'GLD', horizon: 10 },
        result: JSON.stringify({
          data: {
            _tool: 'markov_distribution',
            status: 'abstain',
            abstainReasons: ['Prediction-market anchor coverage is low.'],
            canonical: {
              ticker: 'GLD', currentPrice: 430.7, horizon: 10,
              diagnostics: { trustedAnchors: 0, totalAnchors: 3, predictionConfidence: 0.34, structuralBreakDetected: true },
            },
          },
        }),
      },
      {
        tool: 'polymarket_forecast',
        args: { ticker: 'GLD', horizon_days: 14 },
        result: JSON.stringify({
          data: { result: 'Polymarket Forecast: Gold (GLD) | Horizon: 14 days | Grade: B+ (78/100)' },
        }),
      },
    ];

    expect(buildForcedGoldCombinedForecastArbiterArgs(query, toolCalls)?.polymarket?.quality_score).toBe(78);
  });
});
