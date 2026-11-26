import { describe, expect, it } from 'bun:test';
import { scoreBundles, summarizeSkill, yahooSymbolFor, type DailyCloses } from './polymarket-forecast-skill.js';

const closes: DailyCloses = {
  'BTC-USD': { dates: ['2026-03-01', '2026-03-02', '2026-03-03', '2026-03-08'], closes: [100, 125, 99, 50] },
};
const bundle = (capturedAt: string, horizonDays: number, forecastReturn: number, currentPrice = 100) =>
  ({ capturedAt, ticker: 'BTC', horizonDays, currentPrice, polymarket: { forecastReturn, qualityGrade: 'B' } });

describe('yahooSymbolFor', () => {
  it('maps crypto tickers to their -USD pair and leaves others alone', () => {
    expect(['BTC', 'BTC-USD', 'SOLUSD', 'GLD'].map(yahooSymbolFor)).toEqual(['BTC-USD', 'BTC-USD', 'SOL-USD', 'GLD']);
  });
});

describe('scoreBundles', () => {
  it('scores each forecast against the close at capture + horizon, one per ticker, horizon and capture hour', () => {
    const { rows, excluded } = scoreBundles([
      bundle('2026-03-01T10:00:00Z', 1, -0.01),          // 2026-03-02 close 125: +25%, called down
      bundle('2026-03-01T10:30:00Z', 1, 0.02),           // same hour: replaces the one above
      bundle('2026-03-01T12:00:00Z', 7, -0.05),          // 2026-03-08 close 50: -50%, called down
      bundle('2026-03-01T13:00:00Z', 1, 0.01, 150),      // 50% off the day's close: a test artefact
      bundle('2026-01-15T12:00:00.000Z', 1, 0.01),       // the test clock
      bundle('2026-03-03T09:00:00Z', 30, 0.01),          // resolves after the last close
    ], closes);

    expect(rows).toEqual([
      { key: 'BTC-USD|1', capturedAt: '2026-03-01T10:30:00Z', forecastReturn: 0.02, realizedReturn: 0.25, grade: 'B' },
      { key: 'BTC-USD|7', capturedAt: '2026-03-01T12:00:00Z', forecastReturn: -0.05, realizedReturn: -0.5, grade: 'B' },
    ]);
    expect(excluded).toEqual({ testClock: 1, priceMismatch: 1, unresolved: 1, noForecast: 0, noPrices: 0, sameHour: 1 });
  });
});

describe('summarizeSkill', () => {
  it('compares the hit rate with always-up, always-down and the better of the two', () => {
    const rows = [
      { key: 'X|1', capturedAt: '2026-03-01T00:00:00Z', forecastReturn: -0.01, realizedReturn: -0.02, grade: 'B' },
      { key: 'X|1', capturedAt: '2026-03-02T00:00:00Z', forecastReturn: -0.005, realizedReturn: 0.01, grade: 'B' },
      { key: 'X|1', capturedAt: '2026-03-03T00:00:00Z', forecastReturn: 0.01, realizedReturn: 0.03, grade: 'B' },
      { key: 'X|1', capturedAt: '2026-03-04T00:00:00Z', forecastReturn: -0.02, realizedReturn: -0.01, grade: 'B' },
    ];
    const s = summarizeSkill(rows);
    expect(s.n).toBe(4);
    expect(s.days).toBe(4);
    expect(s.hitRate).toBe(0.75);        // 3 of 4 directions right
    expect(s.alwaysUp).toBe(0.5);
    expect(s.alwaysDown).toBe(0.5);
    expect(s.gapVsBetterConstant).toBe(0.25);
    expect(s.rankingAuc).toBe(1);        // the forecasts order the outcomes perfectly
  });
});
