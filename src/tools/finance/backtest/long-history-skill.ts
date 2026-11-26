/**
 * Skill over 2021-2025 (fixtures/backtest-prices-2021-2025.json, built by scripts/build-backtest-prices-long.ts):
 * every backtest ticker at 5/7/10/14/20/30 days with the live settings, scored against the two baselines a forecast
 * must beat. Traded calls (BUY/SELL) against always calling "up" on the same steps; P(up) against a coin flip
 * (Brier 0.25). 95% CIs from a circular block bootstrap (block = horizon). Also how often the realized price fell
 * inside the central 90/95/99% intervals. The source of the accuracy and coverage figures in
 * docs/markov-prediction-guide.md and docs/markov-when-to-use.md. `run` also covers 1/2/3 days, used only by the
 * call track record (call-track-record-data.ts); `report` scores 5-30 days.
 *
 *   bun run src/tools/finance/backtest/long-history-skill.ts run <ticker> [dir]   # a few minutes per ticker; run tickers in parallel
 *   bun run src/tools/finance/backtest/long-history-skill.ts report [dir]
 *   bun run src/tools/finance/backtest/long-history-skill.ts track-record [dir]   # regenerates call-track-record-data.ts
 */
import { join } from 'node:path';
import { getAssetProfile } from '../markov-distribution/asset-profile.js';
import { btcLiveCell } from './live-config-benchmark.js';
import { walkForward } from './walk-forward.js';

export const LONG_HISTORY_TICKERS = ['SPY', 'QQQ', 'VOO', 'VTI', 'DIA', 'GLD', 'IAU', 'MSFT', 'NVDA', 'GOOGL', 'AMZN', 'AAPL', 'TSLA', 'BTC-USD'];
const HORIZONS = [5, 7, 10, 14, 20, 30];
const RUN_HORIZONS = [1, 2, 3, ...HORIZONS];
const PERIODS = ['2021', '2022', '2023', '2024-25'] as const;

export interface SkillRow {
  ticker: string;
  horizon: number;
  t: number;
  date: string;
  p: number;
  y: 0 | 1;
  rec: 'BUY' | 'HOLD' | 'SELL';
  /** Realized price at the horizon and the central interval bounds (absent from rows written before they were recorded). */
  realized?: number;
  lo90?: number;
  hi90?: number;
  lo95?: number;
  hi95?: number;
  lo99?: number;
  hi99?: number;
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

/** Share of realized prices inside each central interval, over the rows that carry intervals. */
export function intervalCoverage(rows: readonly SkillRow[]) {
  const withIntervals = rows.filter(r => r.realized !== undefined && r.lo90 !== undefined);
  const inside = (r: SkillRow, lo?: number, hi?: number) => (r.realized! >= lo! && r.realized! <= hi! ? 1 : 0);
  return {
    n: withIntervals.length,
    c90: mean(withIntervals.map(r => inside(r, r.lo90, r.hi90))),
    c95: mean(withIntervals.map(r => inside(r, r.lo95, r.hi95))),
    c99: mean(withIntervals.map(r => inside(r, r.lo99, r.hi99))),
  };
}

export interface CallTrackRecordData {
  period: string;
  horizons: number[];
  measuredOn: Record<string, string[]>;
  /** "assetType|horizon|BUY|SELL" -> calls made, share right, share of all steps that moved the call's way. */
  records: Record<string, { calls: number; hitRate: number; baseRate: number }>;
}

/** The BUY/SELL track record by asset type and horizon: what markov_distribution quotes next to its call. */
export function buildCallTrackRecord(rows: readonly SkillRow[], typeOf: (ticker: string) => string): CallTrackRecordData {
  const groups = new Map<string, SkillRow[]>();
  const measuredOn: Record<string, string[]> = {};
  for (const r of rows) {
    const type = typeOf(r.ticker);
    const key = `${type}|${r.horizon}`;
    const group = groups.get(key);
    if (group) group.push(r);
    else groups.set(key, [r]);
    if (!(measuredOn[type] ??= []).includes(r.ticker)) measuredOn[type].push(r.ticker);
  }
  for (const tickers of Object.values(measuredOn)) tickers.sort();
  const records: CallTrackRecordData['records'] = {};
  for (const [key, group] of groups) {
    const upShare = mean(group.map(r => r.y));
    for (const call of ['BUY', 'SELL'] as const) {
      const calls = group.filter(r => r.rec === call);
      if (calls.length === 0) continue;
      const rightWhen = call === 'BUY' ? 1 : 0;
      records[`${key}|${call}`] = {
        calls: calls.length,
        hitRate: mean(calls.map(r => r.y === rightWhen ? 1 : 0)),
        baseRate: call === 'BUY' ? upShare : 1 - upShare,
      };
    }
  }
  const years = rows.map(r => r.date.slice(0, 4)).sort();
  return {
    period: `${years[0]}-${years[years.length - 1]}`,
    horizons: [...new Set(rows.map(r => r.horizon))].sort((a, b) => a - b),
    measuredOn,
    records,
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
  for (const horizon of RUN_HORIZONS) {
    const cell = ticker === 'BTC-USD'
      ? btcLiveCell(horizon)
      : { config: {}, maxHistoryPrices: ['GLD', 'IAU'].includes(ticker) ? 174 : 83 };
    const result = await walkForward({
      ...cell.config, ticker, prices: closes, horizon, warmup: cell.maxHistoryPrices, stride: 1,
      maxHistoryPrices: cell.maxHistoryPrices, randomState: 42,
    });
    for (const s of result.steps) {
      rows.push({
        ticker, horizon, t: s.t, date: dates[s.t], p: s.predictedProb, y: s.actualBinary as 0 | 1, rec: s.recommendation,
        realized: s.realizedPrice, lo90: s.central90CiLower, hi90: s.central90CiUpper, lo95: s.central95CiLower,
        hi95: s.central95CiUpper, lo99: s.central99CiLower, hi99: s.central99CiUpper,
      });
    }
  }
  await Bun.write(join(dir, `${ticker}.json`), JSON.stringify(rows));
  console.log(`${ticker}: ${rows.length} steps`);
}

async function report(dir: string): Promise<void> {
  const rows = (await Promise.all(LONG_HISTORY_TICKERS.map(t => Bun.file(join(dir, `${t}.json`)).json() as Promise<SkillRow[]>)))
    .flat().filter(r => HORIZONS.includes(r.horizon));
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

  const cov = (rs: SkillRow[]) => { const c = intervalCoverage(rs); return `90%: ${(100 * c.c90).toFixed(1)}%  95%: ${(100 * c.c95).toFixed(1)}%  99%: ${(100 * c.c99).toFixed(1)}%  (n=${c.n})`; };
  console.log('\nCentral interval coverage (share of realized prices inside), by period');
  for (const period of [...PERIODS, 'all']) console.log(`  ${period.padEnd(8)} ${cov(period === 'all' ? rows : rows.filter(r => periodOf(r.date) === period))}`);
  console.log('By asset type (all periods)');
  for (const type of ['etf', 'commodity', 'equity', 'crypto']) console.log(`  ${type.padEnd(9)} ${cov(rows.filter(r => getAssetProfile(r.ticker).type === type))}`);
}

if (import.meta.main) {
  const [mode, a, b] = process.argv.slice(2);
  const defaultDir = join(process.cwd(), '.cramer-short', 'long-history-skill');
  if (mode === 'run' && a) await runTicker(a, b ?? defaultDir);
  else if (mode === 'report') await report(a ?? defaultDir);
  else if (mode === 'track-record') {
    const dir = a ?? defaultDir;
    const rows = (await Promise.all(LONG_HISTORY_TICKERS.map(t => Bun.file(join(dir, `${t}.json`)).json() as Promise<SkillRow[]>))).flat();
    const data = buildCallTrackRecord(rows, ticker => getAssetProfile(ticker).type);
    const out = join(import.meta.dir, '..', 'markov-distribution', 'call-track-record-data.ts');
    await Bun.write(out, `// Generated by \`bun run src/tools/finance/backtest/long-history-skill.ts track-record\`; do not edit by hand.\n`
      + `export const CALL_TRACK_RECORD_DATA = ${JSON.stringify(data, null, 2)};\n`);
    console.log(`${Object.keys(data.records).length} records -> ${out}`);
  } else console.log('usage: long-history-skill.ts run <ticker> [dir] | report [dir] | track-record [dir]');
}
