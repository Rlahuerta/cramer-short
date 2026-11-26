/**
 * Skill of the live polymarket_forecast tool, from the forecasts the agent captures as it runs (the arbiter replay
 * bundles). Each captured forecastReturn is scored against the realized move from the captured price to the daily
 * close at capture + horizon: its direction against always calling "up", always calling "down" and the better of the
 * two (the fair baseline in a trending stretch), and its within ticker x horizon ranking (AUROC). 95% CIs resample
 * whole capture days, since forecasts captured the same day share outcomes. Rerun as captures accumulate.
 *
 *   bun run src/tools/finance/backtest/polymarket-forecast-skill.ts [bundles.jsonl]
 */
import { FIXED_TEST_NOW_ISO } from '../../../utils/test-determinism.js';
import { DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH } from '../arbiter-replay.js';

export type DailyCloses = Record<string, { dates: string[]; closes: number[] }>;

interface CapturedBundle {
  capturedAt: string;
  ticker: string;
  horizonDays: number;
  currentPrice: number;
  polymarket?: { forecastReturn?: number; qualityGrade?: string };
}

export interface ScoredForecast {
  /** "SYMBOL|horizonDays" */
  key: string;
  capturedAt: string;
  forecastReturn: number;
  realizedReturn: number;
  grade: string | undefined;
}

const CRYPTO_BASES = new Set(['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'BNB', 'AVAX', 'HYPE']);

export function yahooSymbolFor(ticker: string): string {
  const base = ticker.toUpperCase().replace(/-?USDT?$/, '');
  return CRYPTO_BASES.has(base) ? `${base}-USD` : ticker.toUpperCase();
}

/**
 * Scored forecasts, one per ticker, horizon and capture hour (the last), and the reasons the rest were left out:
 * the test suite's fixed clock, a captured price more than 10% from that day's close (test artefacts), a horizon
 * that has not resolved yet, no forecast, or no prices for the symbol.
 */
export function scoreBundles(bundles: readonly CapturedBundle[], closes: DailyCloses) {
  const excluded = { testClock: 0, priceMismatch: 0, unresolved: 0, noForecast: 0, noPrices: 0, sameHour: 0 };
  const byHour = new Map<string, ScoredForecast>();
  for (const b of bundles) {
    if (b.capturedAt === FIXED_TEST_NOW_ISO) { excluded.testClock++; continue; }
    const forecastReturn = b.polymarket?.forecastReturn;
    if (typeof forecastReturn !== 'number' || !Number.isFinite(forecastReturn)) { excluded.noForecast++; continue; }
    const symbol = yahooSymbolFor(b.ticker);
    const series = closes[symbol];
    if (!series) { excluded.noPrices++; continue; }
    const day = b.capturedAt.slice(0, 10);
    let reference: number | undefined;
    for (let i = 0; i < series.dates.length && series.dates[i] <= day; i++) reference = series.closes[i];
    if (reference === undefined || Math.abs(b.currentPrice / reference - 1) > 0.10) { excluded.priceMismatch++; continue; }
    const target = new Date(Date.parse(b.capturedAt) + b.horizonDays * 86_400_000).toISOString().slice(0, 10);
    const at = series.dates.findIndex(d => d >= target);
    if (at < 0) { excluded.unresolved++; continue; }
    const hour = `${symbol}|${b.horizonDays}|${b.capturedAt.slice(0, 13)}`;
    if (byHour.has(hour)) excluded.sameHour++;
    byHour.set(hour, {
      key: `${symbol}|${b.horizonDays}`, capturedAt: b.capturedAt, forecastReturn,
      realizedReturn: series.closes[at] / b.currentPrice - 1, grade: b.polymarket?.qualityGrade,
    });
  }
  return { rows: [...byHour.values()], excluded };
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / xs.length;

/** AUROC of the forecast return against "the price rose", within each key, pair-weighted (0.5 = no ranking). */
function rankingAuc(rows: readonly ScoredForecast[]): number {
  const groups = new Map<string, ScoredForecast[]>();
  for (const r of rows) {
    const g = groups.get(r.key);
    if (g) g.push(r);
    else groups.set(r.key, [r]);
  }
  let pairs = 0, score = 0;
  for (const g of groups.values()) {
    const ups = g.filter(r => r.realizedReturn > 0), downs = g.filter(r => r.realizedReturn <= 0);
    for (const u of ups) for (const d of downs) {
      pairs++;
      score += u.forecastReturn > d.forecastReturn ? 1 : u.forecastReturn === d.forecastReturn ? 0.5 : 0;
    }
  }
  return pairs > 0 ? score / pairs : NaN;
}

export function summarizeSkill(rows: readonly ScoredForecast[]) {
  const called = rows.filter(r => r.forecastReturn !== 0);
  const hitRate = mean(called.map(r => (r.forecastReturn > 0) === (r.realizedReturn > 0) ? 1 : 0));
  const alwaysUp = mean(called.map(r => r.realizedReturn > 0 ? 1 : 0));
  return {
    n: rows.length,
    days: new Set(rows.map(r => r.capturedAt.slice(0, 10))).size,
    hitRate,
    alwaysUp,
    alwaysDown: 1 - alwaysUp,
    gapVsBetterConstant: hitRate - Math.max(alwaysUp, 1 - alwaysUp),
    rankingAuc: rankingAuc(rows),
  };
}

/** 95% CI of a statistic, resampling whole capture days. */
function dayClusteredCi(rows: readonly ScoredForecast[], stat: (rs: ScoredForecast[]) => number): [number, number] {
  const byDay = new Map<string, ScoredForecast[]>();
  for (const r of rows) {
    const day = r.capturedAt.slice(0, 10);
    const g = byDay.get(day);
    if (g) g.push(r);
    else byDay.set(day, [r]);
  }
  const days = [...byDay.values()];
  let seed = 3;
  const rand = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
  const draws: number[] = [];
  for (let i = 0; i < 1000; i++) draws.push(stat(days.flatMap(() => days[Math.floor(rand() * days.length)])));
  draws.sort((a, b) => a - b);
  return [draws[25], draws[974]];
}

async function fetchDailyCloses(symbols: string[], fromDate: string): Promise<DailyCloses> {
  const UA = 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';
  const period1 = Date.parse(`${fromDate}T00:00:00Z`) / 1000 - 10 * 86_400;
  const period2 = Math.floor(Date.now() / 1000);
  const out: DailyCloses = {};
  for (const symbol of symbols) {
    const res = await fetch(
      `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(symbol)}?period1=${period1}&period2=${period2}&interval=1d`,
      { headers: { 'User-Agent': UA, Accept: 'application/json' }, signal: AbortSignal.timeout(20_000) },
    );
    if (!res.ok) continue;
    const result = (await res.json() as { chart: { result: Array<{ meta: { gmtoffset: number }; timestamp: number[]; indicators: { quote: Array<{ close: Array<number | null> }> } }> } }).chart.result[0];
    const dates: string[] = [], closes: number[] = [];
    result.timestamp.forEach((ts, i) => {
      const close = result.indicators.quote[0].close[i];
      if (close !== null) { dates.push(new Date((ts + result.meta.gmtoffset) * 1000).toISOString().slice(0, 10)); closes.push(close); }
    });
    out[symbol] = { dates, closes };
  }
  return out;
}

if (import.meta.main) {
  const path = process.argv[2] ?? DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH;
  const bundles = (await Bun.file(path).text()).trim().split('\n').filter(Boolean).map(l => JSON.parse(l) as CapturedBundle);
  const symbols = [...new Set(bundles.map(b => yahooSymbolFor(b.ticker)))];
  const firstCapture = bundles.map(b => b.capturedAt).filter(t => t !== FIXED_TEST_NOW_ISO).sort()[0] ?? new Date().toISOString();
  const { rows, excluded } = scoreBundles(bundles, await fetchDailyCloses(symbols, firstCapture.slice(0, 10)));
  const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
  console.log(`${bundles.length} captured forecasts from ${path}; scored ${rows.length}; left out ${JSON.stringify(excluded)}`);
  const groups = new Map<string, ScoredForecast[]>();
  for (const r of rows) {
    const g = groups.get(r.key);
    if (g) g.push(r);
    else groups.set(r.key, [r]);
  }
  console.log('\nticker|horizon      n  days  hit     always-up  always-down  AUROC');
  for (const [key, g] of [...groups].sort((a, b) => b[1].length - a[1].length)) {
    if (g.length < 15) continue;
    const s = summarizeSkill(g);
    console.log(`${key.padEnd(15)} ${String(s.n).padStart(4)} ${String(s.days).padStart(5)}  ${pct(s.hitRate).padStart(6)}  ${pct(s.alwaysUp).padStart(9)}  ${pct(s.alwaysDown).padStart(11)}  ${s.rankingAuc.toFixed(2)}`);
  }
  if (rows.length === 0) process.exit(0);
  const all = summarizeSkill(rows);
  const [gLo, gHi] = dayClusteredCi(rows, rs => summarizeSkill(rs).gapVsBetterConstant);
  const [aLo, aHi] = dayClusteredCi(rows, rs => summarizeSkill(rs).rankingAuc);
  console.log(`\nAll: ${all.n} forecasts on ${all.days} capture days; hit ${pct(all.hitRate)} vs always-up ${pct(all.alwaysUp)}, always-down ${pct(all.alwaysDown)}`);
  console.log(`  hit minus the better constant: ${(100 * all.gapVsBetterConstant).toFixed(1)}pp [${(100 * gLo).toFixed(1)}, ${(100 * gHi).toFixed(1)}]`);
  console.log(`  within ticker x horizon AUROC: ${all.rankingAuc.toFixed(3)} [${aLo.toFixed(3)}, ${aHi.toFixed(3)}]`);
  console.log(gLo > 0 || aLo > 0.5
    ? '  Verdict: a measurable edge (a CI clears its no-skill line).'
    : '  Verdict: no measurable edge yet (both CIs include no skill).');
}
