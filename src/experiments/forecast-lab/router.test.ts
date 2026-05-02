import { describe, expect, it } from 'bun:test';
import { routeForecastLabQuery } from './router.js';

function expectReason(route: ReturnType<typeof routeForecastLabQuery>, needle: string) {
  expect(route.reasons.some((reason) => reason.toLowerCase().includes(needle.toLowerCase()))).toBe(true);
}

describe('forecast-lab router', () => {
  it('routes BTC 1d/2d/3d optimization requests to the BTC ultra-short-horizon profile', () => {
    const route = routeForecastLabQuery(
      'Optimize the BTC 1d/2d/3d forecast lab setup and improve the short-horizon Markov behavior.',
    );

    expect(route).toMatchObject({
      intent: 'improvement',
      preferredProfileId: 'btc-markov-ultra-short-horizon',
    });
    expect(route.reasons.length).toBeGreaterThan(1);
    expectReason(route, 'improvement');
    expectReason(route, 'btc-markov-ultra-short-horizon');
  });

  it('routes multi-asset short-horizon mechanics queries to the multi-asset Markov profile', () => {
    const route = routeForecastLabQuery(
      'Improve the multi-asset short-horizon mechanics and optimize the Markov forecast profile.',
    );

    expect(route).toMatchObject({
      intent: 'improvement',
      preferredProfileId: 'multi-asset-markov-short-horizon',
    });
    expectReason(route, 'multi-asset-markov-short-horizon');
    expectReason(route, 'multi-asset');
  });

  it('routes arbitrator replay improvement requests to the arbiter replay profile', () => {
    const route = routeForecastLabQuery(
      'Please improve the arbitrator replay workflow and optimize the replay thresholds.',
    );

    expect(route).toMatchObject({
      intent: 'improvement',
      preferredProfileId: 'btc-arbiter-replay',
    });
    expectReason(route, 'btc-arbiter-replay');
    expectReason(route, 'replay');
  });

  it('does not trigger experiment routing for ordinary forecast usage or market analysis questions', () => {
    const route = routeForecastLabQuery(
      'How should I use the BTC forecast for ordinary market analysis and read the result?',
    );

    expect(route).toMatchObject({
      intent: 'none',
      preferredProfileId: null,
    });
    expect(route.reasons.length).toBeGreaterThan(0);
  });

  it('does not treat plain BTC forecast questions as improvement intent', () => {
    const route = routeForecastLabQuery('What is your BTC forecast for tomorrow?');

    expect(route).toMatchObject({
      intent: 'none',
      preferredProfileId: null,
    });
    expect(route.reasons.length).toBeGreaterThan(0);
  });
});
