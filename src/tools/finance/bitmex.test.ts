import { afterEach, describe, expect, it, mock } from 'bun:test';
import {
  fetchBitmexDailyCloses,
  resolveBitmexHistoricalSymbol,
  toBitmexSymbolCandidates,
} from './bitmex.js';

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('toBitmexSymbolCandidates', () => {
  it('maps BTC tickers to BitMEX XBT contracts first', () => {
    expect(toBitmexSymbolCandidates('BTC-USD')).toEqual(['XBTUSD', 'XBTUSDT', 'XBT_USDT']);
  });

  it('builds USDT, USD, and spot-style candidates for altcoin roots', () => {
    expect(toBitmexSymbolCandidates('HYPE-USD')).toEqual(['HYPEUSDT', 'HYPEUSD', 'HYPE_USDT']);
  });
});

describe('resolveBitmexHistoricalSymbol', () => {
  it('selects the most liquid open matching instrument', async () => {
    globalThis.fetch = (mock(async () => Response.json([
      {
        symbol: 'HYPEUSD',
        rootSymbol: 'HYPE',
        state: 'Open',
        markPrice: 42,
        foreignNotional24h: 1000,
      },
      {
        symbol: 'HYPEUSDT',
        rootSymbol: 'HYPE',
        state: 'Open',
        markPrice: 42,
        foreignNotional24h: 5000,
      },
      {
        symbol: 'HYPEZ26',
        rootSymbol: 'HYPE',
        state: 'Unlisted',
        markPrice: 43,
        foreignNotional24h: 999999,
      },
    ])) as unknown) as typeof fetch;

    await expect(resolveBitmexHistoricalSymbol('HYPE-USD')).resolves.toBe('HYPEUSDT');
  });
});

describe('fetchBitmexDailyCloses', () => {
  it('returns BitMEX bucketed closes oldest-first', async () => {
    const urls: string[] = [];
    globalThis.fetch = (mock(async (input: URL | RequestInfo) => {
      const url = String(input);
      urls.push(url);
      if (url.includes('/instrument/active')) {
        return Response.json([
          {
            symbol: 'SOLUSD',
            rootSymbol: 'SOL',
            state: 'Open',
            markPrice: 93,
            foreignNotional24h: 3000000,
          },
        ]);
      }
      return Response.json([
        { close: 93.4 },
        { close: 92.8 },
        { close: 91.7 },
      ]);
    }) as unknown) as typeof fetch;

    const result = await fetchBitmexDailyCloses('SOL-USD', 3);

    expect(result).toEqual([91.7, 92.8, 93.4]);
    expect(urls[1]).toContain('symbol=SOLUSD');
    expect(urls[1]).toContain('binSize=1d');
  });
});
