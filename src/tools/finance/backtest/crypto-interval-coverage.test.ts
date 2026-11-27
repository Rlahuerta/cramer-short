import { describe, expect, it } from 'bun:test';
import { normPPF } from '../../../utils/stats.js';
import { dayBlockCi, fitCandidate, insideRescaled, stitchCloses, tailRatio, type CoverageRow } from './crypto-interval-coverage.js';

/** A row whose intervals are a zero-centred normal in log price with sd 1, realized at log return x. */
const normalRow = (x: number, horizon = 1): CoverageRow => {
  const [z90, z95, z99] = [1.6449, 1.96, 2.5758];
  return {
    ticker: 'BTC-USD', horizon, t: 0, date: '2024-01-01', p: 0.5, y: x > 0 ? 1 : 0, rec: 'HOLD', r2: 0,
    realized: Math.exp(x), lo90: Math.exp(-z90), hi90: Math.exp(z90), lo95: Math.exp(-z95), hi95: Math.exp(z95),
    lo99: Math.exp(-z99), hi99: Math.exp(z99),
  };
};

describe('insideRescaled', () => {
  it('scales the log half-width about the log midpoint', () => {
    const row = { ...normalRow(2.0), lo90: Math.exp(0.5), hi90: Math.exp(1.5) }; // midpoint 1, half-width 0.5
    expect(insideRescaled(row, '90', 1)).toBe(false);
    expect(insideRescaled(row, '90', 2)).toBe(true);
    expect(insideRescaled({ ...row, realized: Math.exp(1.49) }, '90', 1)).toBe(true);
  });
});

describe('tailRatio', () => {
  it('is 1 for an unchanged nu and moves the 90% and 99% bounds opposite ways for fatter tails', () => {
    expect(tailRatio(0.99, 5, 5)).toBeCloseTo(1, 12);
    expect(tailRatio(0.99, 8, 3)).toBeGreaterThan(1);
    expect(tailRatio(0.9, 8, 3)).toBeLessThan(1);
  });
});

describe('fitCandidate', () => {
  it('recovers the scale that brings normal intervals to nominal on outcomes 1.2 times wider', () => {
    const rows = Array.from({ length: 2000 }, (_, i) => normalRow(1.2 * normPPF((i + 0.5) / 2000)));
    const fit = fitCandidate(rows, [undefined]);
    expect(fit.nu).toBeUndefined();
    expect(fit.k).toBeGreaterThanOrEqual(1.18);
    expect(fit.k).toBeLessThanOrEqual(1.22);
    expect(fit.worstGap).toBeLessThan(0.3);
  });
});

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
