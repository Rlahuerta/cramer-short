/**
 * Skill over 2021-2025 (fixtures/backtest-prices-2021-2025.json, built by scripts/build-backtest-prices-long.ts):
 * every backtest ticker at 5/7/10/14/20/30 days with the live settings, scored against the two baselines a forecast
 * must beat. Traded calls (BUY/SELL) against always calling "up" on the same steps; P(up) against a coin flip
 * (Brier 0.25). 95% CIs from a circular block bootstrap (block = horizon). The source of the accuracy figures in
 * docs/markov-prediction-guide.md and docs/markov-when-to-use.md.
 *
 *   bun run src/tools/finance/backtest/long-history-skill.ts run <ticker> [dir]   # a few minutes per ticker; run tickers in parallel
 *   bun run src/tools/finance/backtest/long-history-skill.ts report [dir]
 */
import { join } from 'node:path';
import { btcLiveCell } from './live-config-benchmark.js';
import { walkForward } from './walk-forward.js';

export const LONG_HISTORY_TICKERS = ['SPY', 'QQQ', 'VOO', 'VTI', 'DIA', 'GLD', 'IAU', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'AAPL', 'TSLA', 'BTC-USD'];
const HORIZONS = [5, 7, 10, 14, 20, 30];
const PERIODS = ['2021', '2022', '2023', '2024-25'] as const;

export interface SkillRow {
  ticker: string;
  horizon: number;
  t: number;
  date: string;
  p: number;
  y: 0 | 1;
  rec: 'BUY' | 'HOLD' | 'SELL';
}

export const periodOf = (date: string) => date < '2022' ? '2021' : date < '2023' ? '2022' : date < '2024' ? '2023' : '2024-25';
const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

export function skillGaps(rows: readonly SkillRow[]) {
  const traded = rows.filter(r => r.rec !== 'HOLD');
  const tradedAccuracy = mean(traded.map(r => (r.rec === 'BUY') === (r.y === 1) ? 1 : 0));
  const alwaysUpAccuracy = mean(traded.map(r => r.y));
  return {
    tradedAccuracy,
    alwaysUpAccuracy,
    accuracyMinusAlwaysUp: tradedAccuracy - alwaysUpAccuracy,
    brierMinusCoinFlip: mean(rows.map(r => (r.p - r.y) ** 2)) - 0.25,
  };
}

/** 95% CI of a statistic over rows grouped by ticker|horizon, resampling circular blocks of `horizon` steps. */
function blockCi(rows: readonly SkillRow[], stat: (rs: SkillRow[]) => number, reps = 300): [number, number] {
  const groups = new Map<string, SkillRow[]>();
  for (const r of rows) {
    const key = `${r.ticker}|${r.horizon}`;
    const group = groups.get(key);
    if (group) group.push(r);
    else groups.set(key, [r]);
  }
  let seed = 7;
  const rand = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
  const draws: number[] = [];
  for (let i = 0; i < reps; i++) {
    const sample: SkillRow[] = [];
    for (const g of groups.values()) {
      const resampled: SkillRow[] = [];
      while (resampled.length < g.length) {
        const start = Math.floor(rand() * g.length);
        for (let k = 0; k < g[0].horizon && resampled.length < g.length; k++) resampled.push(g[(start + k) % g.length]);
      }
      sample.push(...resampled);
    }
    draws.push(stat(sample));
  }
  draws.sort((a, b) => a - b);
  return [draws[Math.floor(0.025 * (reps - 1))], draws[Math.floor(0.975 * (reps - 1))]];
}

async function runTicker(ticker: string, dir: string): Promise<void> {
  const fixture = await Bun.file(join(import.meta.dir, '..', 'fixtures', 'backtest-prices-2021-2025.json')).json() as {
    tickers: Record<string, { closes: number[]; dates: string[] }>;
  };
  const { closes, dates } = fixture.tickers[ticker];
  const rows: SkillRow[] = [];
  for (const horizon of HORIZONS) {
    const cell = ticker === 'BTC-USD'
      ? btcLiveCell(horizon)
      : { config: {}, maxHistoryPrices: ['GLD', 'IAU'].includes(ticker) ? 174 : 83 };
    const result = await walkForward({
      ...cell.config, ticker, prices: closes, horizon, warmup: cell.maxHistoryPrices, stride: 1,
      maxHistoryPrices: cell.maxHistoryPrices, randomState: 42,
    });
    for (const s of result.steps) {
      rows.push({ ticker, horizon, t: s.t, date: dates[s.t], p: s.predictedProb, y: s.actualBinary as 0 | 1, rec: s.recommendation });
    }
  }
  await Bun.write(join(dir, `${ticker}.json`), JSON.stringify(rows));
  console.log(`${ticker}: ${rows.length} steps`);
}

async function report(dir: string): Promise<void> {
  const rows = (await Promise.all(LONG_HISTORY_TICKERS.map(t => Bun.file(join(dir, `${t}.json`)).json() as Promise<SkillRow[]>))).flat();
  const pct = (x: number) => `${x >= 0 ? '+' : ''}${(100 * x).toFixed(1)}`;
  const br = (x: number) => `${x >= 0 ? '+' : ''}${x.toFixed(4)}`;
  console.log('All tickers and horizons, by period: traded accuracy vs always-up, Brier minus coin flip');
  for (const period of [...PERIODS, 'all']) {
    const rs = period === 'all' ? rows : rows.filter(r => periodOf(r.date) === period);
    const g = skillGaps(rs);
    const [aLo, aHi] = blockCi(rs, x => skillGaps(x).accuracyMinusAlwaysUp);
    const [bLo, bHi] = blockCi(rs, x => skillGaps(x).brierMinusCoinFlip);
    const ups = mean(rs.map(r => r.y));
    console.log(`  ${period.padEnd(8)} up-moves ${(100 * ups).toFixed(0)}% | model ${(100 * g.tradedAccuracy).toFixed(1)}% vs always-up ${(100 * g.alwaysUpAccuracy).toFixed(1)}%: ${pct(g.accuracyMinusAlwaysUp)}pp [${pct(aLo)}, ${pct(aHi)}] | Brier ${br(g.brierMinusCoinFlip)} [${br(bLo)}, ${br(bHi)}]`);
  }
  let beatUp = 0, worseUp = 0, beatCoin = 0, worseCoin = 0;
  console.log('\nPer ticker and horizon (all periods)');
  for (const ticker of LONG_HISTORY_TICKERS) {
    for (const horizon of HORIZONS) {
      const rs = rows.filter(r => r.ticker === ticker && r.horizon === horizon);
      const g = skillGaps(rs);
      const [aLo, aHi] = blockCi(rs, x => skillGaps(x).accuracyMinusAlwaysUp);
      const [bLo, bHi] = blockCi(rs, x => skillGaps(x).brierMinusCoinFlip);
      if (aLo > 0) beatUp++; if (aHi < 0) worseUp++; if (bHi < 0) beatCoin++; if (bLo > 0) worseCoin++;
      console.log(`  ${ticker.padEnd(7)} ${String(horizon).padStart(2)}d  ${pct(g.accuracyMinusAlwaysUp)}pp [${pct(aLo)}, ${pct(aHi)}]  Brier ${br(g.brierMinusCoinFlip)} [${br(bLo)}, ${br(bHi)}]`);
    }
  }
  console.log(`\n${LONG_HISTORY_TICKERS.length * HORIZONS.length} cells: beat always-up ${beatUp}, worse ${worseUp}; beat a coin flip on Brier ${beatCoin}, worse ${worseCoin}`);
}

if (import.meta.main) {
  const [mode, a, b] = process.argv.slice(2);
  const defaultDir = join(process.cwd(), '.cramer-short', 'long-history-skill');
  if (mode === 'run' && a) await runTicker(a, b ?? defaultDir);
  else if (mode === 'report') await report(a ?? defaultDir);
  else console.log('usage: long-history-skill.ts run <ticker> [dir] | report [dir]');
}
