import { MS_PER_DAY } from '../../utils/time.js';
const BINANCE_BASE = 'https://api.binance.com';

type BinanceTicker24h = {
  symbol: string;
  lastPrice: string;
  priceChange: string;
  priceChangePercent: string;
  volume: string;
};

type BinanceKlineRow = [
  number,
  string,
  string,
  string,
  string,
  string,
  number,
  string,
  number,
  string,
  string,
  string,
];

export function toBinanceSymbol(ticker: string): string {
  const normalized = ticker.trim().toUpperCase();
  const overrides: Record<string, string> = {
    BTC: 'BTCUSDT',
    ETH: 'ETHUSDT',
    SOL: 'SOLUSDT',
    DOGE: 'DOGEUSDT',
    XRP: 'XRPUSDT',
    ADA: 'ADAUSDT',
    DOT: 'DOTUSDT',
    LINK: 'LINKUSDT',
  };
  if (overrides[normalized]) return overrides[normalized];
  return normalized
    .replace('-USD', 'USDT')
    .replace('/', '');
}

export async function fetchBinanceTicker24h(ticker: string): Promise<{
  symbol: string;
  price: number;
  change24h: number;
  changePercent24h: number;
  volume24h: number;
} | null> {
  try {
    const symbol = toBinanceSymbol(ticker);
    const res = await fetch(`${BINANCE_BASE}/api/v3/ticker/24hr?symbol=${symbol}`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(8_000),
    });
    if (!res.ok) return null;
    const data = await res.json() as BinanceTicker24h;
    return {
      symbol,
      price: parseFloat(data.lastPrice),
      change24h: parseFloat(data.priceChange),
      changePercent24h: parseFloat(data.priceChangePercent),
      volume24h: parseFloat(data.volume),
    };
  } catch {
    return null;
  }
}

export type BinanceInterval = '1m' | '1d' | '1w' | '1M';

const KLINES_PAGE_LIMIT = 1000;
const KLINES_MAX_PAGES = 5;
// Upper bounds, so the clamped window below never holds more bars than the pages allow.
const INTERVAL_MS: Record<BinanceInterval, number> = {
  '1m': 60_000,
  '1d': MS_PER_DAY,
  '1w': 7 * MS_PER_DAY,
  '1M': 31 * MS_PER_DAY,
};

/**
 * Dated kline closes for [startMs, endMs], oldest first. Binance returns the oldest
 * `limit` bars after startTime, so this pages forward until the window's end.
 * ponytail: capped at 5 pages (5,000 bars); a longer window keeps its newest bars.
 * Returns [] on any failure.
 */
export async function fetchBinanceCloses(
  ticker: string,
  { startMs, endMs, interval = '1d' }: { startMs: number; endMs: number; interval?: BinanceInterval },
): Promise<{ date: string; close: number }[]> {
  const symbol = toBinanceSymbol(ticker);
  const rows: BinanceKlineRow[] = [];
  let from = Math.max(startMs, endMs - (KLINES_PAGE_LIMIT * KLINES_MAX_PAGES - 1) * INTERVAL_MS[interval]);

  try {
    for (let page = 0; page < KLINES_MAX_PAGES && from <= endMs; page++) {
      const params = new URLSearchParams({
        symbol,
        interval,
        limit: String(KLINES_PAGE_LIMIT),
        startTime: String(from),
        endTime: String(endMs),
      });
      const res = await fetch(`${BINANCE_BASE}/api/v3/klines?${params}`, {
        headers: { Accept: 'application/json' },
        signal: AbortSignal.timeout(10_000),
      });
      if (!res.ok) return [];
      const batch = await res.json() as BinanceKlineRow[];
      rows.push(...batch);
      if (batch.length < KLINES_PAGE_LIMIT) break;
      from = batch[batch.length - 1][0] + 1;
    }
  } catch {
    return [];
  }

  return rows
    .map((row) => {
      const iso = new Date(row[0]).toISOString();
      return { date: interval === '1m' ? iso : iso.slice(0, 10), close: parseFloat(row[4]) };
    })
    .filter((row) => Number.isFinite(row.close));
}

export async function fetchBinanceDailyCloses(ticker: string, days = 120): Promise<number[]> {
  const endMs = Date.now();
  const closes = await fetchBinanceCloses(ticker, { startMs: endMs - days * MS_PER_DAY, endMs });
  return closes.map((row) => row.close);
}
