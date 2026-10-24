import { describe, it, expect } from 'bun:test';
import { resolveForecastToolTicker } from './forecast-ticker.js';

describe('resolveForecastToolTicker', () => {
  it('returns the markov form for markov_distribution (crypto)', () => {
    expect(resolveForecastToolTicker('BTC / BTC-USD only.', 'markov_distribution')).toBe('BTC-USD');
  });

  it('returns the bare root for polymarket_forecast (crypto)', () => {
    expect(resolveForecastToolTicker('BTC / BTC-USD only.', 'polymarket_forecast')).toBe('BTC');
  });

  it('returns the bare root for forecast_arbitrator (crypto)', () => {
    expect(resolveForecastToolTicker('BTC / BTC-USD only.', 'forecast_arbitrator')).toBe('BTC');
  });

  it('returns the GLD proxy for commodity-gold markov', () => {
    expect(resolveForecastToolTicker('GOLD only. 5 day distribution.', 'markov_distribution')).toBe('GLD');
    expect(resolveForecastToolTicker('GOLD only. 5 day distribution.', 'polymarket_forecast')).toBe('GLD');
  });

  it('resolves single-asset forecast queries without an "only" override', () => {
    expect(resolveForecastToolTicker('Give me a BTC 24h trajectory briefing.', 'markov_distribution')).toBe('BTC-USD');
  });

  it('is not fooled by a titlecase "Plan" before "Only" (regression)', () => {
    const query = 'BTC / BTC-USD only. 10. Final BTC Trade Plan\n Only provide a plan if TRADE.';
    expect(resolveForecastToolTicker(query, 'markov_distribution')).toBe('BTC-USD');
  });

  it('returns null for non-forecast tools', () => {
    expect(resolveForecastToolTicker('BTC / BTC-USD only.', 'get_market_data')).toBeNull();
  });

  it('returns null for macro queries with no resolvable asset', () => {
    expect(resolveForecastToolTicker('How is the market doing today?', 'markov_distribution')).toBeNull();
  });

  it('returns null for ambiguous multi-asset queries without an override', () => {
    expect(resolveForecastToolTicker('Compare BTC and ETH over the next 7 days.', 'markov_distribution')).toBeNull();
  });

  it('returns null for explicit gold-combined requests (handled elsewhere)', () => {
    expect(resolveForecastToolTicker('Combined markov and polymarket forecast for GOLD next week', 'markov_distribution')).toBeNull();
  });
});
