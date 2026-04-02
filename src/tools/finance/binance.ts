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

export async function fetchBinanceDailyCloses(
  ticker: string,
  days = 120,
): Promise<number[]> {
  const symbol = toBinanceSymbol(ticker);
  const endTime = Date.now();
  const startTime = endTime - days * 86_400_000;
  const params = new URLSearchParams({
    symbol,
    interval: '1d',
    limit: String(Math.min(days, 1000)),
    startTime: String(startTime),
    endTime: String(endTime),
  });

  try {
    const res = await fetch(`${BINANCE_BASE}/api/v3/klines?${params}`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) return [];
    const rows = await res.json() as BinanceKlineRow[];
    return rows
      .map((row) => parseFloat(row[4]))
      .filter((price) => Number.isFinite(price));
  } catch {
    return [];
  }
}
