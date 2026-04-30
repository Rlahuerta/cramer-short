import { describe, expect, it } from 'bun:test';
import {
  buildKalshiVolatilityCovariate,
  extractKalshiVolSignalsFromPayload,
  fetchKalshiVolSignals,
  KalshiUnconfiguredError,
} from './kalshi-vol-signals.js';

describe('fetchKalshiVolSignals', () => {
  it('fails loudly when no API key is configured', async () => {
    await expect(fetchKalshiVolSignals({
      fromDate: '2026-06-01',
      toDate: '2026-06-30',
      apiKey: '',
    })).rejects.toThrow(KalshiUnconfiguredError);
  });

  it('fetches and filters whitelisted macro markets', async () => {
    const fetchImpl = async (input: string | URL | Request) => {
      expect(String(input)).toContain('/markets?status=open&limit=200');
      return new Response(JSON.stringify({
        markets: [
          {
            ticker: 'FOMC-25JUN18-HIKE',
            title: 'Will the Fed hike rates at the June FOMC meeting?',
            expiration_time: '2026-06-18T18:00:00Z',
            yes_ask: 63,
          },
          {
            ticker: 'CPI-26JUN',
            title: 'Will CPI come in above 0.3% MoM?',
            expiration_time: '2026-06-11T12:30:00Z',
            yes_ask: 0.58,
          },
          {
            ticker: 'SPORTS-IGNORE',
            title: 'Will the Knicks win game 7?',
            expiration_time: '2026-06-20T00:00:00Z',
            yes_ask: 0.51,
          },
        ],
      }), { status: 200 });
    };

    const signals = await fetchKalshiVolSignals({
      fromDate: '2026-06-01',
      toDate: '2026-06-30',
      apiKey: 'test-key',
      fetchImpl,
    });

    expect(signals).toHaveLength(2);
    expect(signals.map((signal) => signal.eventType)).toEqual(['fomc', 'cpi']);
    expect(signals.every((signal) => signal.intensityBoost > 0)).toBe(true);
  });
});

describe('extractKalshiVolSignalsFromPayload', () => {
  it('rejects malformed payloads', () => {
    expect(() => extractKalshiVolSignalsFromPayload({}, {
      fromDate: '2026-06-01',
      toDate: '2026-06-30',
    })).toThrow('Kalshi response missing markets array');
  });
});

describe('buildKalshiVolatilityCovariate', () => {
  it('builds a forward-looking covariate that peaks ahead of events', () => {
    const covariate = buildKalshiVolatilityCovariate(
      ['2026-06-10', '2026-06-11', '2026-06-12', '2026-06-13', '2026-06-14'],
      [
        {
          eventAt: '2026-06-13T12:30:00Z',
          eventId: 'CPI-2026-06',
          probability: 0.61,
          intensityBoost: 1.2,
          eventType: 'cpi',
          sourceTitle: 'Will CPI come in above 0.3% MoM?',
        },
      ],
      3,
    );

    expect(covariate.activeSignals).toBe(1);
    expect(covariate.values[0]).toBeLessThan(covariate.values[2]);
    expect(covariate.values[2]).toBeLessThan(covariate.values[3]);
    expect(covariate.peakValue).toBe(covariate.values[3]);
    expect(covariate.values[4]).toBe(0);
  });
});
