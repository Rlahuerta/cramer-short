import { describe, expect, it } from 'bun:test';
import { periodOf, skillGaps, type SkillRow } from './long-history-skill.js';

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
