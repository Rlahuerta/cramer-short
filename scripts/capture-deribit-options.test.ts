import { describe, expect, it } from 'bun:test';
import { toDeribitCaptureRow } from './capture-deribit-options.js';

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
