import { afterEach, describe, expect, it, mock } from 'bun:test';

const mockGet = mock(() => Promise.reject(new Error('FD unavailable')));

mock.module('./api.js', () => ({
  api: { get: mockGet },
}));

const t = Date.now();
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
});
