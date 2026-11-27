import { describe, expect, it } from 'bun:test';
import { dayBlockCi, stitchCloses } from './crypto-interval-coverage.js';

describe('stitchCloses', () => {
  const early = { dates: ['2025-12-30', '2025-12-31', '2026-01-01'], closes: [10, 11, 12] };
  const late = { dates: ['2025-12-31', '2026-01-01', '2026-01-02'], closes: [11, 12.001, 13] };

  it('keeps the early series before the seam and the late one from it', () => {
    expect(stitchCloses(early, late, '2026-01-01')).toEqual({
      dates: ['2025-12-30', '2025-12-31', '2026-01-01', '2026-01-02'],
      closes: [10, 11, 12.001, 13],
    });
  });

  it('refuses series that disagree by more than 0.1% on a shared day', () => {
    expect(() => stitchCloses(early, { ...late, closes: [11.5, 12, 13] }, '2026-01-01')).toThrow(/2025-12-31/);
  });

  it('refuses a missing day at the seam', () => {
    expect(() => stitchCloses(early, { dates: ['2026-01-02'], closes: [13] }, '2026-01-01')).toThrow(/gap/);
  });
});

describe('dayBlockCi', () => {
  it('moves every row of a date together', () => {
    // Each date has one BTC and one ETH row, so any resample keeps the ratio at 1.
    const rows = Array.from({ length: 100 }, (_, i) => [
      { date: `d${String(i).padStart(3, '0')}`, asset: 'BTC' },
      { date: `d${String(i).padStart(3, '0')}`, asset: 'ETH' },
    ]).flat();
    const ratio = (rs: typeof rows) => rs.filter(r => r.asset === 'BTC').length / rs.filter(r => r.asset === 'ETH').length;
    expect(dayBlockCi(rows, ratio)).toEqual([1, 1]);
  });

  it('brackets the mean of a varying statistic and is reproducible', () => {
    const rows = Array.from({ length: 400 }, (_, i) => ({ date: `d${String(i).padStart(3, '0')}`, x: i % 10 === 0 ? 1 : 0 }));
    const share = (rs: typeof rows) => rs.reduce((a, r) => a + r.x, 0) / rs.length;
    const [lo, hi] = dayBlockCi(rows, share);
    expect(lo).toBeLessThan(0.1);
    expect(hi).toBeGreaterThan(0.1);
    expect(dayBlockCi(rows, share)).toEqual([lo, hi]);
  });
});
