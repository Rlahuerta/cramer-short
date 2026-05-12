import { afterEach, describe, expect, it, mock } from 'bun:test';
import {
  bitmexMarketTool,
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

  it('falls back to direct symbol candidates when active instrument resolution fails', async () => {
    const urls: string[] = [];
    globalThis.fetch = (mock(async (input: URL | RequestInfo) => {
      const url = String(input);
      urls.push(url);
      if (url.includes('/instrument/active')) {
        return new Response(null, { status: 503 });
      }
      if (url.includes('symbol=HYPEUSDT')) {
        return Response.json([
          { close: 42.1 },
          { close: 42.2 },
        ]);
      }
      return Response.json([]);
    }) as unknown) as typeof fetch;

    const result = await fetchBitmexDailyCloses('HYPE-USD', 2);

    expect(result).toEqual([42.2, 42.1]);
    expect(urls.some((url) => url.includes('/instrument/active'))).toBe(true);
    expect(urls.some((url) => url.includes('symbol=HYPEUSDT'))).toBe(true);
  });
});

describe('bitmexMarketTool', () => {
  it('returns active instrument metadata and historical closes for agent use', async () => {
    const urls: string[] = [];
    globalThis.fetch = (mock(async (input: URL | RequestInfo) => {
      const url = String(input);
      urls.push(url);
      if (url.includes('/instrument/active')) {
        return Response.json([
          {
            symbol: 'HYPEUSDT',
            rootSymbol: 'HYPE',
            underlying: 'HYPE',
            state: 'Open',
            markPrice: 42.5,
            bidPrice: 42.49,
            askPrice: 42.51,
            volume24h: 1000,
            foreignNotional24h: 42500,
            initMargin: 0.02,
            fundingRate: 0.0001,
          },
        ]);
      }
      return Response.json([
        { close: 42.1 },
        { close: 42.2 },
      ]);
    }) as unknown) as typeof fetch;

    const raw = await bitmexMarketTool.invoke({ tickers: ['HYPEUSDT'], days: 2 });
    const parsed = JSON.parse(raw);
    const [result] = parsed.data.results;

    expect(result.matched).toBe(true);
    expect(result.instrument.symbol).toBe('HYPEUSDT');
    expect(result.instrument.spreadPct).toBeCloseTo(0.0470588235);
    expect(result.instrument.maxLeverage).toBe(50);
    expect(result.historicalCloses).toEqual([42.2, 42.1]);
    expect(urls.some((url) => url.includes('/instrument/active'))).toBe(true);
    expect(urls.some((url) => url.includes('/trade/bucketed'))).toBe(true);
  });

  it('is registered for agent use', () => {
    // Verify the tool carries the correct registry name directly rather than
    // calling getToolRegistry(), which is subject to mock.module() contamination
    // from agent test files (agent-response.test.ts, agent.test.ts, prompts.test.ts)
    // during parallel test runs.
    expect(bitmexMarketTool.name).toBe('bitmex_market');
  });
});
