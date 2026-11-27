import { FIXED_TEST_DATE, FIXED_TEST_NOW_MS, deterministicRandom, nextTestId } from '@/utils/test-determinism.js';
import { afterEach, afterAll, describe, expect, it, mock, beforeEach, setSystemTime } from 'bun:test';

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
});

afterEach(() => {
  setSystemTime();
});

const realApi = { ...(await import('./api.js')) };

const mockGet = mock(() => Promise.reject(new Error('FD unavailable')));

mock.module('./api.js', () => ({
  api: { get: mockGet },
}));

afterAll(() => {
  mock.module('./api.js', () => realApi);
});

const t = FIXED_TEST_NOW_MS;
const { getCryptoPriceSnapshot, getCryptoPrices } = await import(`./crypto.js?t=${t}`) as typeof import('./crypto.js');

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
  mockGet.mockClear();
  mockGet.mockImplementation(() => Promise.reject(new Error('FD unavailable')));
});

describe('crypto tool Binance fallbacks', () => {
  it('uses Binance snapshot fallback when primary provider fails', async () => {
    globalThis.fetch = (mock(async () => Response.json({
      symbol: 'BTCUSDT',
      lastPrice: '66010.12',
      priceChange: '-123.45',
      priceChangePercent: '-0.19',
      volume: '15432.1',
    })) as unknown) as typeof fetch;

    const raw = await getCryptoPriceSnapshot.invoke({ ticker: 'BTC-USD' });
    const parsed = JSON.parse(raw);
    expect(parsed.data.source).toBe('binance');
    expect(parsed.data.ticker).toBe('BTCUSDT');
    expect(parsed.data.price).toBe(66010.12);
  });

  it('uses Binance daily closes fallback when crypto history provider fails', async () => {
    globalThis.fetch = (mock(async () => Response.json([
      [1704067200000, '42000', '42500', '41800', '42300', '10000', 1704153600000, '0', 0, '0', '0', '0'],
      [1704153600000, '42300', '42700', '42000', '42650', '9000', 1704240000000, '0', 0, '0', '0', '0'],
      [1704240000000, '42650', '43100', '42500', '43000', '9500', 1704326400000, '0', 0, '0', '0', '0'],
    ])) as unknown) as typeof fetch;

    const raw = await getCryptoPrices.invoke({
      ticker: 'BTC-USD',
      interval: 'day',
      interval_multiplier: 1,
      start_date: '2024-01-01',
      end_date: '2024-01-03',
    });
    const parsed = JSON.parse(raw);
    expect(Array.isArray(parsed.data)).toBe(true);
    expect(parsed.data).toHaveLength(3);
    expect(parsed.data[0].close).toBe(42300);
  });

  it('bounds the Binance fallback klines to the requested date range, not the last N days', async () => {
    let klinesUrl = '';
    globalThis.fetch = (mock(async (input: RequestInfo | URL) => {
      klinesUrl = String(input);
      return Response.json([
        [1609459200000, '42000', '42500', '41800', '42300', '10000', 1609545599999, '0', 0, '0', '0', '0'],
      ]);
    }) as unknown) as typeof fetch;

    await getCryptoPrices.invoke({
      ticker: 'BTC-USD',
      interval: 'day',
      interval_multiplier: 1,
      start_date: '2021-01-01',
      end_date: '2021-01-10',
    });

    const params = new URL(klinesUrl).searchParams;
    expect(Number(params.get('startTime'))).toBe(Date.UTC(2021, 0, 1));
    expect(Number(params.get('endTime'))).toBe(Date.UTC(2021, 0, 10) + 86_399_999);
  });

  /** Serves daily klines the way Binance does: oldest first from startTime, at most `limit` rows. */
  function stubDailyKlines(): void {
    globalThis.fetch = (mock(async (input: RequestInfo | URL) => {
      const params = new URL(String(input)).searchParams;
      const day = 86_400_000;
      const endTime = Number(params.get('endTime'));
      const rows = [];
      for (
        let open = Math.ceil(Number(params.get('startTime')) / day) * day;
        open <= endTime && rows.length < Number(params.get('limit'));
        open += day
      ) {
        rows.push([open, '1', '1', '1', String(open / day), '1', open + day - 1, '0', 0, '0', '0', '0']);
      }
      return Response.json(rows);
    }) as unknown) as typeof fetch;
  }

  it('returns dated rows that reach the requested end of a multi-year window', async () => {
    stubDailyKlines();

    const parsed = JSON.parse(await getCryptoPrices.invoke({
      ticker: 'BTC-USD',
      interval: 'day',
      interval_multiplier: 1,
      start_date: '2022-01-01',
      end_date: '2024-12-31',
    }));

    expect(parsed.data).toHaveLength(1096);
    expect(parsed.data[0]).toEqual({ date: '2022-01-01', close: Date.UTC(2022, 0, 1) / 86_400_000 });
    expect(parsed.data.at(-1)).toEqual({ date: '2024-12-31', close: Date.UTC(2024, 11, 31) / 86_400_000 });
  });

  it.each([
    ['minute', '1m'],
    ['day', '1d'],
    ['week', '1w'],
    ['month', '1M'],
  ] as const)('requests %s bars as Binance %s klines', async (interval, binanceInterval) => {
    let klinesUrl = '';
    globalThis.fetch = (mock(async (input: RequestInfo | URL) => {
      klinesUrl = String(input);
      return Response.json([
        [1609459200000, '42000', '42500', '41800', '42300', '10000', 1609545599999, '0', 0, '0', '0', '0'],
      ]);
    }) as unknown) as typeof fetch;

    await getCryptoPrices.invoke({
      ticker: 'BTC-USD',
      interval,
      interval_multiplier: 1,
      start_date: '2021-01-01',
      end_date: '2021-01-01',
    });

    expect(new URL(klinesUrl).searchParams.get('interval')).toBe(binanceInterval);
  });
});
