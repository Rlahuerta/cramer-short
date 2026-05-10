const BITMEX_BASE = 'https://www.bitmex.com';

type BitmexInstrument = {
  symbol: string;
  rootSymbol?: string;
  underlying?: string;
  state?: string;
  markPrice?: number;
  lastPrice?: number;
  volume24h?: number;
  foreignNotional24h?: number;
  homeNotional24h?: number;
  turnover24h?: number;
};

type BitmexBucketRow = {
  close?: number;
};

function toBitmexRoot(ticker: string): string {
  const normalized = ticker.trim().toUpperCase().replace('/', '');
  if (normalized === 'BTC' || normalized === 'BTCUSD' || normalized === 'BTCUSDT') return 'XBT';
  if (normalized === 'BTC-USD' || normalized === 'BTC-USDT') return 'XBT';
  if (normalized === 'XAUUSD' || normalized === 'XAU-USD') return 'XAU';
  if (normalized === 'XAGUSD' || normalized === 'XAG-USD') return 'XAG';
  return normalized
    .replace(/[-_]?USD[T]?$/, '')
    .replace(/[-_]?XBT$/, '');
}

function bitmexLiquidity(instrument: BitmexInstrument): number {
  for (const key of ['foreignNotional24h', 'homeNotional24h', 'turnover24h'] as const) {
    const value = Number(instrument[key]);
    if (Number.isFinite(value) && value > 0) return value;
  }
  const volume = Number(instrument.volume24h ?? 0);
  const price = Number(instrument.markPrice ?? instrument.lastPrice ?? 0);
  return Number.isFinite(volume * price) ? volume * price : 0;
}

export function toBitmexSymbolCandidates(ticker: string): string[] {
  const root = toBitmexRoot(ticker);
  const candidates = [
    `${root}USDT`,
    `${root}USD`,
    `${root}_USDT`,
  ];
  if (root === 'XBT') {
    candidates.unshift('XBTUSD', 'XBTUSDT');
  }
  return [...new Set(candidates)];
}

export async function resolveBitmexHistoricalSymbol(ticker: string): Promise<string | null> {
  const root = toBitmexRoot(ticker);
  const candidates = new Set(toBitmexSymbolCandidates(ticker));

  try {
    const res = await fetch(`${BITMEX_BASE}/api/v1/instrument/active`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) return null;
    const instruments = await res.json() as BitmexInstrument[];
    const matching = instruments
      .filter((instrument) => instrument.state === undefined || instrument.state === 'Open')
      .filter((instrument) =>
        candidates.has(instrument.symbol)
        || instrument.rootSymbol?.toUpperCase() === root
        || instrument.underlying?.toUpperCase() === root
      )
      .filter((instrument) => Number.isFinite(Number(instrument.markPrice ?? instrument.lastPrice)));

    const [best] = matching.sort((a, b) => bitmexLiquidity(b) - bitmexLiquidity(a));
    return best?.symbol ?? null;
  } catch {
    return null;
  }
}

export async function fetchBitmexDailyCloses(
  ticker: string,
  days = 120,
): Promise<number[]> {
  const symbol = await resolveBitmexHistoricalSymbol(ticker);
  if (!symbol) return [];

  const params = new URLSearchParams({
    binSize: '1d',
    partial: 'false',
    symbol,
    count: String(Math.min(Math.max(days, 1), 750)),
    reverse: 'true',
  });

  try {
    const res = await fetch(`${BITMEX_BASE}/api/v1/trade/bucketed?${params}`, {
      headers: { Accept: 'application/json' },
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) return [];
    const rows = await res.json() as BitmexBucketRow[];
    return rows
      .slice()
      .reverse()
      .map((row) => Number(row.close))
      .filter((price) => Number.isFinite(price) && price > 0);
  } catch {
    return [];
  }
}
