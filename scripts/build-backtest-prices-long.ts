/**
 * Build fixtures/backtest-prices-2021-2025.json: daily closes for the backtest tickers from 2021 through 2025, so the
 * benchmark can be measured across the 2022 decline and not only the 2024-25 rise that backtest-prices.json covers.
 * Same source and format as backtest-prices.json (Yahoo Finance daily chart, raw close: split-adjusted, not
 * dividend-adjusted, rounded to cents), kept as a separate file so tests pinned to the 2024-25 fixture do not move.
 *
 *   bun run scripts/build-backtest-prices-long.ts [out]
 */
import { join } from 'node:path';

const TICKERS: Record<string, 'etf' | 'stock' | 'crypto'> = {
  SPY: 'etf', AAPL: 'stock', TSLA: 'stock', GLD: 'etf', QQQ: 'etf', 'BTC-USD': 'crypto', VOO: 'etf',
  DIA: 'etf', VTI: 'etf', IAU: 'etf', MSFT: 'stock', NVDA: 'stock', GOOGL: 'stock', AMZN: 'stock',
};
const START = '2021-01-01';
const END = '2025-12-31';
const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';

interface ChartResult {
  meta: { gmtoffset: number };
  timestamp: number[];
  indicators: { quote: Array<{ close: Array<number | null> }> };
}

/** Exchange-local trading dates with their closes; days Yahoo reports without a close are skipped. */
export function chartToDailyCloses(result: ChartResult): { dates: string[]; closes: number[] } {
  const dates: string[] = [];
  const closes: number[] = [];
  result.timestamp.forEach((ts, i) => {
    const close = result.indicators.quote[0].close[i];
    if (close === null || !Number.isFinite(close)) return;
    dates.push(new Date((ts + result.meta.gmtoffset) * 1000).toISOString().slice(0, 10));
    closes.push(Math.round(close * 100) / 100);
  });
  return { dates, closes };
}

async function fetchDailyCloses(ticker: string): Promise<{ dates: string[]; closes: number[] }> {
  const period1 = Date.parse(`${START}T00:00:00Z`) / 1000;
  const period2 = Date.parse(`${END}T23:59:59Z`) / 1000;
  const res = await fetch(
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}?period1=${period1}&period2=${period2}&interval=1d`,
    { headers: { 'User-Agent': UA, Accept: 'application/json' }, signal: AbortSignal.timeout(20_000) },
  );
  if (!res.ok) throw new Error(`Yahoo chart ${res.status} for ${ticker}`);
  const json = await res.json() as { chart: { result: ChartResult[] } };
  return chartToDailyCloses(json.chart.result[0]);
}

if (import.meta.main) {
  const [out = join(import.meta.dir, '..', 'src/tools/finance/fixtures/backtest-prices-2021-2025.json')] = process.argv.slice(2);
  const tickers: Record<string, { type: string; closes: number[]; dates: string[]; count: number }> = {};
  for (const [ticker, type] of Object.entries(TICKERS)) {
    const { dates, closes } = await fetchDailyCloses(ticker);
    tickers[ticker] = { type, closes, dates, count: closes.length };
    console.log(`${ticker}: ${closes.length} closes, ${dates[0]}..${dates[dates.length - 1]}`);
  }
  await Bun.write(out, JSON.stringify({
    generatedAt: new Date().toISOString(),
    startDate: START,
    endDate: END,
    source: 'Yahoo Finance daily chart, raw close (split-adjusted, not dividend-adjusted), rounded to cents',
    tickers,
  }));
  console.log(`-> ${out}`);
}
