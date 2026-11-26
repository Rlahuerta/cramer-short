import { describe, expect, it } from 'bun:test';
import { formatCallTrackRecord, lookupCallTrackRecord } from './call-track-record.js';

const data = {
  period: '2021-2025',
  horizons: [5, 7, 14],
  measuredOn: { etf: ['SPY', 'QQQ'] },
  records: {
    'etf|7|BUY': { calls: 1200, hitRate: 0.61, baseRate: 0.63 },
    'etf|14|SELL': { calls: 300, hitRate: 0.34, baseRate: 0.37 },
  },
};

describe('lookupCallTrackRecord', () => {
  it('returns the record for the call at the nearest measured horizon', () => {
    expect(lookupCallTrackRecord('etf', 7, 'BUY', data)).toEqual({
      assetType: 'etf', horizon: 7, measuredHorizon: 7, call: 'BUY', calls: 1200, hitRate: 0.61, baseRate: 0.63,
      period: '2021-2025', measuredOn: ['SPY', 'QQQ'],
    });
    expect(lookupCallTrackRecord('etf', 12, 'SELL', data)?.measuredHorizon).toBe(14);
  });

  it('has nothing for HOLD, outside the measured horizons, or for an unmeasured asset type', () => {
    expect(lookupCallTrackRecord('etf', 7, 'HOLD', data)).toBeNull();
    expect(lookupCallTrackRecord('etf', 3, 'BUY', data)).toBeNull();
    expect(lookupCallTrackRecord('etf', 45, 'BUY', data)).toBeNull();
    expect(lookupCallTrackRecord('crypto', 7, 'BUY', data)).toBeNull();
  });
});

describe('formatCallTrackRecord', () => {
  it('states the hit rate next to the share of all days that went the call\'s way', () => {
    const record = lookupCallTrackRecord('etf', 12, 'SELL', data)!;
    expect(formatCallTrackRecord(record)).toBe(
      '📏 Track record (etf at 14d, measured on SPY, QQQ, 2021-2025): SELL calls were right 34% of 300 daily calls; '
      + 'the price fell on 37% of all days.',
    );
  });
});
