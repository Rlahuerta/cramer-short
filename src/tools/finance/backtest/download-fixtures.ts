/**
 * One-time script to download historical price data for backtest fixtures.
 *
 * Usage:
 *   FINANCIAL_DATASETS_API_KEY=xxx bun run src/tools/finance/backtest/download-fixtures.ts
 *
 * Downloads ~2 years of daily closes for each ticker and saves to fixtures/backtest-prices.json.
 * For BTC, uses the crypto endpoint with BTC-USD ticker.
 */

import { writeFileSync } from 'fs';
import { join } from 'path';

const API_BASE = 'https://api.financialdatasets.ai';
const API_KEY = process.env.FINANCIAL_DATASETS_API_KEY;

if (!API_KEY) {
  console.error('Error: FINANCIAL_DATASETS_API_KEY env var is required');
  process.exit(1);
}

interface PriceBar {
  time: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

interface FixtureData {
  generatedAt: string;
  startDate: string;
  endDate: string;
  tickers: Record<string, {
    type: 'stock' | 'etf' | 'crypto';
    closes: number[];
    dates: string[];
    count: number;
  }>;
}

const TICKERS = [
  { symbol: 'SPY',     type: 'etf'    as const, endpoint: 'prices' },
  { symbol: 'AAPL',    type: 'stock'  as const, endpoint: 'prices' },
  { symbol: 'TSLA',    type: 'stock'  as const, endpoint: 'prices' },
  { symbol: 'GLD',     type: 'etf'    as const, endpoint: 'prices' },
  { symbol: 'QQQ',     type: 'etf'    as const, endpoint: 'prices' },
  { symbol: 'VOO',     type: 'etf'    as const, endpoint: 'prices' },
  { symbol: 'DIA',     type: 'etf'    as const, endpoint: 'prices' },
  { symbol: 'VTI',     type: 'etf'    as const, endpoint: 'prices' },
  { symbol: 'IAU',     type: 'etf'    as const, endpoint: 'prices' },
  { symbol: 'BTC-USD', type: 'crypto' as const, endpoint: 'crypto/prices' },
];

const START_DATE = '2024-01-01';
const END_DATE   = '2025-12-31';

async function fetchPrices(
  symbol: string,
  endpoint: string,
): Promise<PriceBar[]> {
  const url = new URL(`${API_BASE}/${endpoint}/`);
  url.searchParams.set('ticker', symbol);
  url.searchParams.set('interval', 'day');
  url.searchParams.set('start_date', START_DATE);
  url.searchParams.set('end_date', END_DATE);

  console.log(`  Fetching ${symbol} from ${url.toString()}`);

  const res = await fetch(url.toString(), {
    headers: { 'X-API-KEY': API_KEY! },
  });

  if (!res.ok) {
    throw new Error(`HTTP ${res.status} for ${symbol}: ${await res.text()}`);
  }

  const json = await res.json() as { prices?: PriceBar[] };
  return json.prices ?? [];
}

async function main() {
  console.log('Downloading backtest fixture data...\n');

  const fixture: FixtureData = {
    generatedAt: new Date().toISOString(),
    startDate: START_DATE,
    endDate: END_DATE,
    tickers: {},
  };

  for (const { symbol, type, endpoint } of TICKERS) {
    try {
      const bars = await fetchPrices(symbol, endpoint);
      const closes = bars.map(b => b.close);
      const dates  = bars.map(b => b.time);

      fixture.tickers[symbol] = { type, closes, dates, count: closes.length };
      console.log(`  ✓ ${symbol}: ${closes.length} bars\n`);
    } catch (err) {
      console.error(`  ✗ ${symbol}: ${err}\n`);
      fixture.tickers[symbol] = { type, closes: [], dates: [], count: 0 };
    }
  }

  const outPath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
  writeFileSync(outPath, JSON.stringify(fixture, null, 2));
  console.log(`\nSaved to ${outPath}`);
  console.log('Tickers:', Object.entries(fixture.tickers).map(([k, v]) => `${k}(${v.count})`).join(', '));
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
