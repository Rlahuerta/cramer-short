import { describe, expect, it } from 'bun:test';
import { join } from 'node:path';

type Fixture = { tickers: Record<string, { type: string; closes: number[]; dates: string[]; count: number }> };
const fixtures = join(import.meta.dir, '..', 'fixtures');

describe('backtest-prices-2021-2025 fixture', () => {
  it('matches backtest-prices.json day for day over 2024-25 and starts in 2021', async () => {
    const short = await Bun.file(join(fixtures, 'backtest-prices.json')).json() as Fixture;
    const long = await Bun.file(join(fixtures, 'backtest-prices-2021-2025.json')).json() as Fixture;

    expect(Object.keys(long.tickers).sort()).toEqual(Object.keys(short.tickers).sort());
    for (const [ticker, s] of Object.entries(short.tickers)) {
      const l = long.tickers[ticker];
      expect(l.type).toBe(s.type);
      expect(l.dates[0] < '2021-01-08').toBe(true);
      const from = l.dates.indexOf(s.dates[0]);
      expect(l.dates.slice(from, from + s.dates.length)).toEqual(s.dates);
      expect(l.closes.slice(from, from + s.closes.length)).toEqual(s.closes);
    }
  });
});
