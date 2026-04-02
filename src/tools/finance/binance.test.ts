import { afterEach, describe, expect, it, mock } from 'bun:test';
import { fetchBinanceDailyCloses, fetchBinanceTicker24h, toBinanceSymbol } from './binance.js';

const originalFetch = globalThis.fetch;

afterEach(() => {
  globalThis.fetch = originalFetch;
});

describe('toBinanceSymbol', () => {
  it('normalizes BTC-USD to BTCUSDT', () => {
    expect(toBinanceSymbol('BTC-USD')).toBe('BTCUSDT');
    expect(toBinanceSymbol('btc-usd')).toBe('BTCUSDT');
  });

  it('maps common crypto tickers to USDT pairs', () => {
    expect(toBinanceSymbol('ETH')).toBe('ETHUSDT');
    expect(toBinanceSymbol('SOL')).toBe('SOLUSDT');
  });
});

describe('fetchBinanceTicker24h', () => {
  it('returns null on non-OK response', async () => {
    globalThis.fetch = (mock(async () => new Response('', { status: 404 })) as unknown) as typeof fetch;
    const result = await fetchBinanceTicker24h('BTC-USD');
    expect(result).toBeNull();
  });

  it('parses 24h ticker payload', async () => {
    globalThis.fetch = (mock(async () => Response.json({
      symbol: 'BTCUSDT',
      lastPrice: '66010.12',
      priceChange: '-123.45',
      priceChangePercent: '-0.19',
      volume: '15432.1',
    })) as unknown) as typeof fetch;
    const result = await fetchBinanceTicker24h('BTC-USD');
    expect(result).toEqual({
      symbol: 'BTCUSDT',
      price: 66010.12,
      change24h: -123.45,
      changePercent24h: -0.19,
      volume24h: 15432.1,
    });
  });
});

describe('fetchBinanceDailyCloses', () => {
  it('parses close prices from klines', async () => {
    globalThis.fetch = (mock(async () => Response.json([
      [1704067200000, '42000', '42500', '41800', '42300', '10000', 1704153600000, '0', 0, '0', '0', '0'],
      [1704153600000, '42300', '42700', '42000', '42650', '9000', 1704240000000, '0', 0, '0', '0', '0'],
    ])) as unknown) as typeof fetch;
    const result = await fetchBinanceDailyCloses('BTC-USD', 2);
    expect(result).toEqual([42300, 42650]);
  });
});
