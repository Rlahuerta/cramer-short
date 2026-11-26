import { describe, expect, it } from 'bun:test';
import { closedMarkets, dayWindows } from './build-polymarket-btc-anchor-history.js';

describe('Polymarket anchor history builder', () => {
  it('pages through Gamma 100 events at a time until a short page', async () => {
    const urls: string[] = [];
    const pageSizes = [100, 100, 37];
    const fakeFetch = (async (url: string) => {
      urls.push(url);
      const size = pageSizes[urls.length - 1] ?? 0;
      return new Response(JSON.stringify(Array.from({ length: size }, () => ({ markets: [{ question: 'q' }] }))));
    }) as unknown as typeof fetch;

    const markets = await closedMarkets('bitcoin', '2026-01-03', '2026-01-04', fakeFetch);

    expect(markets).toHaveLength(237);
    expect(urls.map(u => new URL(u).searchParams.get('offset'))).toEqual(['0', '100', '200']);
    expect(new URL(urls[0]!).searchParams.get('limit')).toBe('100');
  });

  it('queries one day per window, so no window reaches the offset cap', () => {
    expect(dayWindows('2026-02', '2026-02')).toHaveLength(28);
    expect(dayWindows('2026-02', '2026-03')[0]).toEqual(['2026-02-01', '2026-02-02']);
    expect(dayWindows('2026-02', '2026-03').at(-1)).toEqual(['2026-03-31', '2026-04-01']);
  });
});
