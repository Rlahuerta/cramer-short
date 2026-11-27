import { describe, expect, it } from 'bun:test';
import { captureDeribitOptions, toDeribitCaptureRow } from './capture-deribit-options.js';

const NO_WAIT = { baseDelayMs: 0, jitter: 0, maxAttempts: 3 };
const book = (instrument: string) => Response.json({ result: [{ instrument_name: instrument }] });

describe('captureDeribitOptions', () => {
  it('retries a transient Deribit error and captures both currencies', async () => {
    let btcCalls = 0;
    const fetchImpl = (async (url: string) => {
      if (url.includes('currency=BTC') && ++btcCalls === 1) return new Response('unavailable', { status: 503 });
      return book(url.includes('currency=BTC') ? 'BTC-X' : 'ETH-X');
    }) as unknown as typeof fetch;

    const { rows, failures } = await captureDeribitOptions(fetchImpl, NO_WAIT);

    expect(failures).toEqual([]);
    expect(rows.map((r) => r.currency)).toEqual(['BTC', 'ETH']);
    expect(btcCalls).toBe(2);
  });

  it('still captures ETH when BTC keeps failing, and reports BTC', async () => {
    const fetchImpl = (async (url: string) =>
      url.includes('currency=BTC') ? new Response('unavailable', { status: 503 }) : book('ETH-X')) as unknown as typeof fetch;

    const { rows, failures } = await captureDeribitOptions(fetchImpl, NO_WAIT);

    expect(rows.map((r) => r.currency)).toEqual(['ETH']);
    expect(failures).toEqual(['BTC: Deribit BTC options: HTTP 503']);
  });
});

describe('toDeribitCaptureRow', () => {
  it('keeps the fields an implied density needs and drops the rest', () => {
    const row = toDeribitCaptureRow('BTC', new Date('2026-11-26T09:30:00Z'), [
      {
        instrument_name: 'BTC-26MAR27-40000-C', mark_iv: 61.88, mark_price: 0.365, bid_price: 0.3635,
        ask_price: 0.3665, open_interest: 104.1, underlying_price: 61519.45, underlying_index: 'BTC-26MAR27',
        volume_usd: 0, interest_rate: 0, high: null, creation_timestamp: 1795718081580,
      },
    ]);

    expect(row).toEqual({
      capturedAt: '2026-11-26T09:30:00.000Z',
      currency: 'BTC',
      options: [{
        instrument: 'BTC-26MAR27-40000-C', markIv: 61.88, markPrice: 0.365, bid: 0.3635, ask: 0.3665,
        openInterest: 104.1, underlyingPrice: 61519.45, underlyingIndex: 'BTC-26MAR27', volumeUsd: 0,
      }],
    });
  });
});

describe('captureDeribitOptions network errors', () => {
  it('retries the connection errors Bun throws (plain Error with a code, not a TypeError)', async () => {
    let btcCalls = 0;
    const fetchImpl = (async (url: string) => {
      if (url.includes('currency=BTC') && ++btcCalls === 1) {
        throw Object.assign(new Error('Unable to connect. Is the computer able to access the url?'), { code: 'ConnectionRefused' });
      }
      return book(url.includes('currency=BTC') ? 'BTC-X' : 'ETH-X');
    }) as unknown as typeof fetch;

    const { rows, failures } = await captureDeribitOptions(fetchImpl, NO_WAIT);

    expect(failures).toEqual([]);
    expect(rows.map((r) => r.currency)).toEqual(['BTC', 'ETH']);
  });
});
