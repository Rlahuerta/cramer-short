/**
 * TDD tests for normalizeForecastToolTickers.
 *
 * Reproduces the reported bug: on a query explicitly scoped to a single asset
 * (e.g. "BTC / BTC-USD only"), the model emitted
 * markov_distribution(ticker="PLAN"), which is never corrected and fails the
 * price fetch, sending the agent into a runaway loop. The normalizer rewrites
 * the ticker of the model's forecast tool calls (markov_distribution,
 * polymarket_forecast, forecast_arbitrator) to the query's resolved ticker
 * before execution.
 */
import { describe, it, expect } from 'bun:test';
import { AIMessage } from '@langchain/core/messages';
import { normalizeForecastToolTickers } from './planning-tool-calls.js';

const BTC_ONLY_QUERY = 'BTC live 1-day briefing. BTC / BTC-USD only. No GOLD, ETH, SOL.';

function aiMsg(name: string, args: Record<string, unknown>): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: [{ id: 'c1', name, args, type: 'tool_call' as const }],
    additional_kwargs: {},
  });
}

function tickerOf(response: AIMessage): unknown {
  return response.tool_calls?.[0]?.args?.ticker;
}

describe('normalizeForecastToolTickers', () => {
  it('rewrites a wrong markov_distribution ticker to the resolved BTC-USD ticker', () => {
    const response = aiMsg('markov_distribution', {
      ticker: 'PLAN',
      horizon: 1,
      trajectory: true,
      trajectoryDays: 1,
    });

    normalizeForecastToolTickers(response, BTC_ONLY_QUERY);

    expect(tickerOf(response)).toBe('BTC-USD');
    // other args preserved
    expect(response.tool_calls?.[0]?.args?.horizon).toBe(1);
    expect(response.tool_calls?.[0]?.args?.trajectory).toBe(true);
  });

  it('rewrites a wrong polymarket_forecast ticker to the bare BTC ticker', () => {
    const response = aiMsg('polymarket_forecast', { ticker: 'PLAN', horizon_days: 1 });

    normalizeForecastToolTickers(response, BTC_ONLY_QUERY);

    expect(tickerOf(response)).toBe('BTC');
  });

  it('rewrites a wrong forecast_arbitrator ticker to the bare BTC ticker', () => {
    const response = aiMsg('forecast_arbitrator', { ticker: 'PLAN', horizon_days: 1 });

    normalizeForecastToolTickers(response, BTC_ONLY_QUERY);

    expect(tickerOf(response)).toBe('BTC');
  });

  it('normalizes markov crypto ticker format (BTC -> BTC-USD)', () => {
    const response = aiMsg('markov_distribution', { ticker: 'BTC', horizon: 1 });

    normalizeForecastToolTickers(response, BTC_ONLY_QUERY);

    expect(tickerOf(response)).toBe('BTC-USD');
  });

  it('normalizes polymarket crypto ticker format (BTC-USD -> BTC)', () => {
    const response = aiMsg('polymarket_forecast', { ticker: 'BTC-USD', horizon_days: 1 });

    normalizeForecastToolTickers(response, BTC_ONLY_QUERY);

    expect(tickerOf(response)).toBe('BTC');
  });

  it('leaves an already-correct markov ticker untouched', () => {
    const response = aiMsg('markov_distribution', { ticker: 'BTC-USD', horizon: 1 });

    normalizeForecastToolTickers(response, BTC_ONLY_QUERY);

    expect(tickerOf(response)).toBe('BTC-USD');
  });

  it('rewrites the ticker for an equity "X only" query', () => {
    const query = 'NVDA only. 5 trading day distribution.';
    const response = aiMsg('markov_distribution', { ticker: 'PLAN', horizon: 5 });

    normalizeForecastToolTickers(response, query);

    expect(tickerOf(response)).toBe('NVDA');
  });

  it('rewrites a commodity-gold ticker to the GLD proxy for markov ("GOLD only")', () => {
    const response = aiMsg('markov_distribution', { ticker: 'PLAN', horizon: 5 });

    normalizeForecastToolTickers(response, 'GOLD only. 5 trading day distribution.');

    expect(tickerOf(response)).toBe('GLD');
  });

  it('rewrites a commodity-gold polymarket ticker to the GLD proxy ("GLD only")', () => {
    const response = aiMsg('polymarket_forecast', { ticker: 'PLAN', horizon_days: 7 });

    normalizeForecastToolTickers(response, 'GLD only. 7 trading day distribution.');

    expect(tickerOf(response)).toBe('GLD');
  });

  it('rewrites a commodity-silver ticker to the SLV proxy for markov ("SILVER only")', () => {
    const response = aiMsg('markov_distribution', { ticker: 'PLAN', horizon: 5 });

    normalizeForecastToolTickers(response, 'SILVER only. 5 day forecast.');

    expect(tickerOf(response)).toBe('SLV');
  });

  it('rewrites an ETH ticker to ETH-USD for markov ("ETH only")', () => {
    const response = aiMsg('markov_distribution', { ticker: 'PLAN', horizon: 1 });

    normalizeForecastToolTickers(response, 'ETH / ETH-USD only. 1 trading day trajectory forecast.');

    expect(tickerOf(response)).toBe('ETH-USD');
  });

  it('rewrites a wrong ticker on a single-asset forecast query WITHOUT an "only" override', () => {
    const response = aiMsg('markov_distribution', { ticker: 'PLAN', horizon: 1 });

    normalizeForecastToolTickers(response, 'Give me a BTC 24h trajectory briefing with trade plan.');

    expect(tickerOf(response)).toBe('BTC-USD');
  });

  it('rewrites a commodity ticker on a single-asset query WITHOUT an "only" override', () => {
    const response = aiMsg('markov_distribution', { ticker: 'PLAN', horizon: 5 });

    normalizeForecastToolTickers(response, 'GOLD trajectory forecast next 5 trading days with a trade plan.');

    expect(tickerOf(response)).toBe('GLD');
  });

  it('does not touch non-forecast tool calls', () => {
    const response = aiMsg('get_market_data', { ticker: 'PLAN' });

    normalizeForecastToolTickers(response, BTC_ONLY_QUERY);

    expect(tickerOf(response)).toBe('PLAN');
  });

  it('is a no-op when the query has no exclusive asset scope', () => {
    const response = aiMsg('markov_distribution', { ticker: 'PLAN', horizon: 1 });

    normalizeForecastToolTickers(response, 'How is the market doing today?');

    expect(tickerOf(response)).toBe('PLAN');
  });

  it('is a no-op for ambiguous multi-asset queries without an override', () => {
    const response = aiMsg('markov_distribution', { ticker: 'ETH-USD', horizon: 7 });

    normalizeForecastToolTickers(response, 'Compare BTC and ETH over the next 7 days.');

    expect(tickerOf(response)).toBe('ETH-USD');
  });

  it('is a no-op for explicit gold-combined requests (handled by the gold normalizer)', () => {
    const query = 'Combined markov and polymarket forecast for GOLD over the next week';
    const response = aiMsg('markov_distribution', { ticker: 'PLAN', horizon: 5 });

    normalizeForecastToolTickers(response, query);

    expect(tickerOf(response)).toBe('PLAN');
  });

  it('handles a response with no tool calls', () => {
    const response = new AIMessage({ content: 'final answer', additional_kwargs: {} });
    expect(() => normalizeForecastToolTickers(response, BTC_ONLY_QUERY)).not.toThrow();
  });
});
