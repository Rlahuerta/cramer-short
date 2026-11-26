import { describe, expect, it } from 'bun:test';
import { buildCallTrackRecord, periodOf, skillGaps, type SkillRow } from './long-history-skill.js';

const row = (rec: SkillRow['rec'], p: number, y: 0 | 1): SkillRow => ({ ticker: 'SPY', horizon: 7, t: 0, date: '2022-06-01', p, y, rec });

describe('skillGaps', () => {
  it('scores traded calls against always-up on the same steps and P(up) against a coin flip', () => {
    // Traded: BUY right, SELL wrong, SELL right -> 2/3; always-up on those steps: 2 ups of 3 -> 2/3. HOLD only enters Brier.
    const rows = [row('BUY', 0.6, 1), row('SELL', 0.4, 1), row('SELL', 0.3, 0), row('HOLD', 0.5, 1)];
    const gaps = skillGaps(rows);
    expect(gaps.tradedAccuracy).toBeCloseTo(2 / 3, 12);
    expect(gaps.alwaysUpAccuracy).toBeCloseTo(2 / 3, 12);
    expect(gaps.accuracyMinusAlwaysUp).toBeCloseTo(0, 12);
    expect(gaps.brierMinusCoinFlip).toBeCloseTo((0.16 + 0.36 + 0.09 + 0.25) / 4 - 0.25, 12);
  });
});

describe('periodOf', () => {
  it('splits 2021, the 2022 decline, 2023 and 2024-25', () => {
    expect(['2021-12-31', '2022-01-03', '2023-06-30', '2024-01-02', '2025-12-30'].map(periodOf))
      .toEqual(['2021', '2022', '2023', '2024-25', '2024-25']);
  });
});

describe('buildCallTrackRecord', () => {
  it('records, per asset type, horizon and call, how often the call was right against the share of all days that went its way', () => {
    const at = (ticker: string, horizon: number, rec: SkillRow['rec'], y: 0 | 1): SkillRow => ({ ticker, horizon, t: 0, date: '2023-01-02', p: 0.5, y, rec });
    const rows = [
      // BUY right 2 of 3, SELL right 0 of 1. QQQ shares the etf type and its HOLD step only enters the base rate:
      // 5 etf steps at 7d, 3 up and 2 down.
      at('SPY', 7, 'BUY', 1), at('SPY', 7, 'BUY', 1), at('SPY', 7, 'BUY', 0), at('SPY', 7, 'SELL', 1),
      at('QQQ', 7, 'HOLD', 0),
    ];
    const data = buildCallTrackRecord(rows, () => 'etf');

    expect(data.horizons).toEqual([7]);
    expect(data.measuredOn).toEqual({ etf: ['QQQ', 'SPY'] });
    expect(data.records['etf|7|BUY']).toEqual({ calls: 3, hitRate: 2 / 3, baseRate: 3 / 5 });
    expect(data.records['etf|7|SELL']).toEqual({ calls: 1, hitRate: 0, baseRate: 2 / 5 });
  });
});
