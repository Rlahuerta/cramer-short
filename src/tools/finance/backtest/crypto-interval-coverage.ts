/**
 * Crypto interval coverage (2026-11-27): do the production Markov forecast's central 90/95/99% intervals for BTC and
 * ETH at 1/2/3/7/14 days cover what they claim? If not, does a crypto-only widening pass a rule fixed before scoring?
 * Background: FORECAST_VOL_SCALE = 1.05 was set on 14 tickers at 5-30 days and never isolated crypto at short
 * horizons; in H3 (ladder-implied-vol-benchmark.ts) a trailing-sigma proxy under-covered BTC/ETH 1-7d at 95/99%.
 *
 * Production path. walkForward() calls computeMarkovDistribution as the live markov_distribution tool does, with the
 * live configuration: BTC-USD through btcLiveCell(h) (the xiphos live levers including the meta-regime matrices, the
 * per-horizon break threshold, the post-break rerun at 1-3 days, 252 prices); ETH-USD with the defaults and 120 prices
 * (the tool's default fetch; crypto trades every day). FORECAST_VOL_SCALE and the crypto R² thresholds inside
 * computeMarkovDistribution apply as in production. The intervals are read off the forecast distribution (the
 * walk-forward central*Ci fields), as in long-history-skill.ts and the check that set FORECAST_VOL_SCALE. No Polymarket
 * anchors, as in that check. Stride 1, seed 42.
 *
 * Data. BTC-USD: fixtures/backtest-prices-2021-2025.json, then fixtures/crypto-prices-2025-2026.json from 2026-01-01.
 * ETH-USD is not in the 2021-2025 fixture: fixtures/crypto-peer-prices.json (2024-2025), then the 2025-2026 fixture
 * from 2026-01-01. The stitch refuses a shared day that differs by more than 0.1% or a gap at the seam. Origins: every
 * day with a full live window and a close h days later. Period = calendar year of the origin date.
 *
 * Statistics. Coverage = share of realized closes inside [lo, hi]. 95% CIs: circular block bootstrap over origin
 * dates (blocks of 28 consecutive dates; all assets and horizons of a date move together), 1000 reps, percentile.
 * "Emitted" = the forecasts the tool shows a user (model-only crypto at <= 14 days: out-of-sample R² >= -0.03);
 * reported, not gated.
 *
 * Decision rule, fixed and committed before any scoring:
 * 1. Baseline. If pooled BTC+ETH coverage over the five horizons is within ±1pp of nominal at 90, 95 and 99% in every
 *    period, the intervals are calibrated: no change.
 * 2. Otherwise score three crypto-only candidates (crypto at horizons <= 14 days; every other asset and horizon
 *    unchanged), each fitted on the earlier half (origins before 2025-01-01) and scored on the later half (from
 *    2025-01-01: periods 2025 and 2026):
 *    A. one volatility scale k on the forecast sigma, on top of FORECAST_VOL_SCALE;
 *    B. one k per horizon bucket, 1d / 2-3d / 4-7d / 8-14d (fitted on 1 / 2,3 / 7 / 14);
 *    C. one k plus the Student-t degrees of freedom nu for crypto (today max(3, 4 + h)).
 *    Fit: k on 0.90..1.60 in steps of 0.01, nu in {3, 4, 5, 6, 8}; minimize the largest |coverage - nominal| over
 *    90/95/99% on the earlier half, pooled over BTC and ETH (B: per bucket). The fit rescales each recorded interval
 *    in log price about its log midpoint (by t quantiles for nu); the score reruns the production code with the
 *    fitted values.
 *    A candidate passes only if, on the later half: (a) pooled coverage at 90, 95 and 99% is within ±1pp of nominal in
 *    each period; (b) the paired Brier change of P(up) vs the baseline has a 95% CI upper bound <= +0.0005; (c) the
 *    HOLD share moves by at most 1pp.
 * 3. Ship the passing candidate with the fewest parameters (A, then B, then C). None passing: no change.
 *
 * Outputs go to .cramer-short/crypto-interval-coverage/<baseline|A|B|C>. Each run takes a few minutes; run them in parallel.
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts run <BTC-USD|ETH-USD> <horizon>   # baseline
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts report                           # step 1
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts fit                              # step 2 fit
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts run <BTC-USD|ETH-USD> <horizon> <A|B|C>
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts score                            # step 2 score
 */
import { join } from 'node:path';
import { getAssetProfile } from '../markov-distribution/asset-profile.js';
import { horizonStudentTNu, inverseStudentTCDF } from '../markov-distribution/confidence-intervals.js';
import { btcLiveCell } from './live-config-benchmark.js';
import { intervalCoverage, type SkillRow } from './long-history-skill.js';
import { walkForward } from './walk-forward.js';

export const ASSETS = ['BTC-USD', 'ETH-USD'] as const;
export const HORIZONS = [1, 2, 3, 7, 14];
const ETH_HISTORY_PRICES = 120;
const CRYPTO_MODEL_ONLY_MIN_R2 = -0.03; // markov-distribution.ts: below it the tool abstains on a model-only crypto forecast
const SEAM = '2026-01-01';
const ROOT = join(process.cwd(), '.cramer-short', 'crypto-interval-coverage');
const BLOCK_DAYS = 28;
const LEVELS = [['90', 0.9], ['95', 0.95], ['99', 0.99]] as const;

export type CoverageRow = SkillRow & { r2: number | null };
interface Closes { dates: string[]; closes: number[] }

/** `early` before `from`, `late` from it; shared days must agree within 0.1% and the seam must have no missing day. */
export function stitchCloses(early: Closes, late: Closes, from: string): Closes {
  const lateByDate = new Map(late.dates.map((d, i) => [d, late.closes[i]!]));
  early.dates.forEach((d, i) => {
    const other = lateByDate.get(d);
    if (other !== undefined && Math.abs(early.closes[i]! / other - 1) > 0.001) {
      throw new Error(`fixtures disagree on ${d}: ${early.closes[i]} vs ${other}`);
    }
  });
  const head = early.dates.flatMap((d, i) => (d < from ? [[d, early.closes[i]!] as const] : []));
  const tail = late.dates.flatMap((d, i) => (d >= from ? [[d, late.closes[i]!] as const] : []));
  const lastHead = head[head.length - 1]?.[0];
  if (lastHead && tail[0] && Date.parse(tail[0][0]) - Date.parse(lastHead) !== 86_400_000) {
    throw new Error(`gap at the seam: ${lastHead} -> ${tail[0][0]}`);
  }
  const all = [...head, ...tail];
  return { dates: all.map(([d]) => d), closes: all.map(([, c]) => c) };
}

/** 95% percentile CI of `stat`, resampling circular blocks of consecutive dates; every row of a date moves with it. */
export function dayBlockCi<R extends { date: string }>(
  rows: readonly R[],
  stat: (rs: R[]) => number,
  reps = 1000,
): [number, number] {
  const byDate = new Map<string, R[]>();
  for (const r of rows) (byDate.get(r.date) ?? byDate.set(r.date, []).get(r.date)!).push(r);
  const days = [...byDate.keys()].sort().map(d => byDate.get(d)!);
  let seed = 7;
  const rand = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
  const draws: number[] = [];
  for (let i = 0; i < reps; i++) {
    const sample: R[] = [];
    for (let n = 0; n < days.length;) {
      const start = Math.floor(rand() * days.length);
      for (let k = 0; k < BLOCK_DAYS && n < days.length; k++, n++) sample.push(...days[(start + k) % days.length]!);
    }
    draws.push(stat(sample));
  }
  draws.sort((a, b) => a - b);
  return [draws[Math.floor(0.025 * (reps - 1))]!, draws[Math.ceil(0.975 * (reps - 1))]!];
}

type Level = '90' | '95' | '99';
/** k on 0.90..1.60, tried nearest 1 first so a tie keeps the smaller change. */
const K_GRID = Array.from({ length: 71 }, (_, i) => (90 + i) / 100).sort((a, b) => Math.abs(a - 1) - Math.abs(b - 1));
export const NU_GRID = [3, 4, 5, 6, 8];

const tailRatios = new Map<string, number>();
/** Central half-width at `level` of a variance-matched t (studentTSurvival's scaling) with nuTo, over the same with nuFrom. */
export function tailRatio(level: number, nuFrom: number, nuTo: number): number {
  const key = `${level}|${nuFrom}|${nuTo}`;
  const q = (nu: number) => inverseStudentTCDF(1 - (1 - level) / 2, nu) * Math.sqrt((nu - 2) / nu);
  return tailRatios.get(key) ?? tailRatios.set(key, q(nuTo) / q(nuFrom)).get(key)!;
}

/** Inside the row's `level` interval once its log half-width is scaled by k and, given nu, its t tail moved to nu. */
export function insideRescaled(row: CoverageRow, level: Level, k: number, nu?: number): boolean {
  const lo = Math.log(row[`lo${level}`]!);
  const hi = Math.log(row[`hi${level}`]!);
  const ratio = nu === undefined
    ? 1
    : tailRatio(Number(level) / 100, horizonStudentTNu(getAssetProfile(row.ticker).studentTNu, row.horizon), nu);
  return Math.abs(Math.log(row.realized!) - (lo + hi) / 2) <= ((hi - lo) / 2) * k * ratio;
}

/** The (k, nu) on the grid with the smallest largest |coverage - nominal| (pp) over 90/95/99; `nus` in order of preference. */
export function fitCandidate(rows: readonly CoverageRow[], nus: ReadonlyArray<number | undefined>) {
  let best = { k: 1, nu: undefined as number | undefined, worstGap: Infinity };
  for (const nu of nus) {
    for (const k of K_GRID) {
      const worstGap = Math.max(...LEVELS.map(([level, nominal]) =>
        Math.abs(100 * (rows.filter(r => insideRescaled(r, level, k, nu)).length / rows.length - nominal))));
      if (worstGap < best.worstGap) best = { k, nu, worstGap };
    }
  }
  return best;
}

async function loadCloses(asset: (typeof ASSETS)[number]): Promise<Closes> {
  const fixture = async (name: string) =>
    (await Bun.file(join(import.meta.dir, '..', 'fixtures', name)).json() as { tickers: Record<string, Closes> }).tickers[asset]!;
  const early = await fixture(asset === 'BTC-USD' ? 'backtest-prices-2021-2025.json' : 'crypto-peer-prices.json');
  return stitchCloses(early, await fixture('crypto-prices-2025-2026.json'), SEAM);
}

/** Forecast every origin with the production path, plus a candidate's settings (fitted from the baseline rows). */
async function run(asset: (typeof ASSETS)[number], horizon: number, candidate: Candidate | 'baseline'): Promise<void> {
  const { dates, closes } = await loadCloses(asset);
  const cell = asset === 'BTC-USD' ? btcLiveCell(horizon) : { config: {}, maxHistoryPrices: ETH_HISTORY_PRICES };
  const experiment = candidate === 'baseline' ? {} : await candidateConfig(candidate, horizon);
  const result = await walkForward({
    ...cell.config, ...experiment, ticker: asset, prices: closes, horizon, warmup: cell.maxHistoryPrices, stride: 1,
    maxHistoryPrices: cell.maxHistoryPrices, randomState: 42,
  });
  const dir = join(ROOT, candidate);
  if (result.errors.length > 0) throw new Error(`${asset} ${horizon}d: ${result.errors.length} failed origins, first ${result.errors[0]!.error}`);
  const rows: CoverageRow[] = result.steps.map(s => ({
    ticker: asset, horizon, t: s.t, date: dates[s.t]!, p: s.predictedProb, y: s.actualBinary as 0 | 1, rec: s.recommendation,
    r2: s.outOfSampleR2 ?? null, realized: s.realizedPrice, lo90: s.central90CiLower, hi90: s.central90CiUpper,
    lo95: s.central95CiLower, hi95: s.central95CiUpper, lo99: s.central99CiLower, hi99: s.central99CiUpper,
  }));
  await Bun.write(join(dir, `${asset}-${horizon}.json`), JSON.stringify(rows));
  console.log(`${asset} ${horizon}d: ${rows.length} origins ${rows[0]?.date}..${rows[rows.length - 1]?.date}`);
}

const emitted = (r: CoverageRow) => r.r2 !== null && Number.isFinite(r.r2) && r.r2 >= CRYPTO_MODEL_ONLY_MIN_R2;
const yearOf = (r: CoverageRow) => r.date.slice(0, 4);

/** "90: 88.1 [86.9, 89.2]  95: ..." with day-block CIs, plus n. */
function coverageCell(rows: CoverageRow[]): string {
  const c = intervalCoverage(rows);
  const parts = LEVELS.map(([label]) => {
    const key = `c${label}` as 'c90' | 'c95' | 'c99';
    const [lo, hi] = dayBlockCi(rows, rs => intervalCoverage(rs)[key]);
    return `${label}: ${(100 * c[key]).toFixed(1)} [${(100 * lo).toFixed(1)}, ${(100 * hi).toFixed(1)}]`;
  });
  return `${parts.join('  ')}  n=${c.n}`;
}

/** Largest |coverage - nominal| over the three levels, in pp. */
function worstGap(rows: CoverageRow[]): number {
  const c = intervalCoverage(rows);
  return Math.max(...LEVELS.map(([label, nominal]) => Math.abs(100 * (c[`c${label}` as 'c90' | 'c95' | 'c99'] - nominal))));
}

async function loadRows(dir: string): Promise<CoverageRow[]> {
  const files = ASSETS.flatMap(a => HORIZONS.map(h => join(dir, `${a}-${h}.json`)));
  return (await Promise.all(files.map(f => Bun.file(f).json() as Promise<CoverageRow[]>))).flat();
}

async function report(dir: string): Promise<void> {
  const rows = await loadRows(dir);
  const years = [...new Set(rows.map(yearOf))].sort();
  console.log('Central interval coverage (%), 95% day-block CIs. Pooled BTC+ETH over 1/2/3/7/14d, by period');
  for (const y of [...years, 'all']) {
    const rs = y === 'all' ? rows : rows.filter(r => yearOf(r) === y);
    console.log(`  ${y.padEnd(5)} ${coverageCell(rs)}  worst gap ${worstGap(rs).toFixed(1)}pp`);
  }
  console.log('\nEmitted forecasts only (R² >= -0.03), by period');
  for (const y of [...years, 'all']) {
    const rs = rows.filter(r => emitted(r) && (y === 'all' || yearOf(r) === y));
    console.log(`  ${y.padEnd(5)} ${coverageCell(rs)}  (${(100 * rs.length / rows.filter(r => y === 'all' || yearOf(r) === y).length).toFixed(0)}% emitted)`);
  }
  for (const asset of ASSETS) {
    console.log(`\n${asset} by horizon and period`);
    for (const h of HORIZONS) {
      for (const y of [...years, 'all']) {
        const rs = rows.filter(r => r.ticker === asset && r.horizon === h && (y === 'all' || yearOf(r) === y));
        if (rs.length > 0) console.log(`  ${String(h).padStart(2)}d ${y.padEnd(5)} ${coverageCell(rs)}`);
      }
    }
  }
  const failing = years.filter(y => worstGap(rows.filter(r => yearOf(r) === y)) > 1);
  console.log(`\nStep 1: ${failing.length === 0 ? 'calibrated (every period within ±1pp at 90/95/99)' : `not calibrated: outside ±1pp in ${failing.join(', ')}`}`);
}

const SPLIT = '2025-01-01';
export const bucketOf = (h: number) => (h === 1 ? '1d' : h <= 3 ? '2-3d' : h <= 7 ? '4-7d' : '8-14d');

const BUCKETS = ['1d', '2-3d', '4-7d', '8-14d'];
type Candidate = 'A' | 'B' | 'C';

/** Step 2 fit on the earlier half of the baseline rows: A, B per bucket, C. */
async function fits() {
  const rows = (await loadRows(join(ROOT, 'baseline'))).filter(r => r.date < SPLIT);
  return {
    rows,
    A: fitCandidate(rows, [undefined]),
    B: Object.fromEntries(BUCKETS.map(b => [b, fitCandidate(rows.filter(r => bucketOf(r.horizon) === b), [undefined])])),
    C: fitCandidate(rows, NU_GRID),
  };
}

async function candidateConfig(candidate: Candidate, horizon: number) {
  const f = await fits();
  const chosen = candidate === 'B' ? f.B[bucketOf(horizon)]! : f[candidate];
  return { forecastVolMultiplier: chosen.k, studentTNuOverride: chosen.nu };
}

async function showFits(): Promise<void> {
  const f = await fits();
  const show = (label: string, x: ReturnType<typeof fitCandidate>) =>
    console.log(`  ${label.padEnd(9)} k=${x.k.toFixed(2)}${x.nu === undefined ? '' : ` nu=${x.nu}`}  worst gap ${x.worstGap.toFixed(2)}pp (approximate)`);
  console.log(`Fit on ${f.rows.length} origins before ${SPLIT}; baseline worst gap ${worstGap(f.rows).toFixed(2)}pp`);
  show('A', f.A);
  for (const b of BUCKETS) show(`B ${b}`, f.B[b]!);
  show('C', f.C);
}

/** Step 2 score: each candidate's rerun against the baseline on the later half, by the rule in the header. */
async function score(): Promise<void> {
  const key = (r: CoverageRow) => `${r.ticker}|${r.horizon}|${r.date}`;
  const base = await loadRows(join(ROOT, 'baseline'));
  const baseByKey = new Map(base.map(r => [key(r), r]));
  const holdShare = (rs: CoverageRow[]) => rs.filter(r => r.rec === 'HOLD').length / rs.length;
  const later = (rs: CoverageRow[]) => rs.filter(r => r.date >= SPLIT);
  const years = [...new Set(later(base).map(yearOf))].sort();
  const pct = (x: number) => (100 * x).toFixed(1);
  console.log(`Baseline, later half: ${years.map(y => `${y} worst gap ${worstGap(later(base).filter(r => yearOf(r) === y)).toFixed(1)}pp`).join(', ')}`);
  for (const candidate of ['A', 'B', 'C'] as const) {
    const rows = await loadRows(join(ROOT, candidate));
    const early = rows.filter(r => r.date < SPLIT);
    const c = intervalCoverage(early);
    console.log(`\nCandidate ${candidate}. Earlier half, exact rerun: 90 ${pct(c.c90)}  95 ${pct(c.c95)}  99 ${pct(c.c99)}`);
    const scored = later(rows);
    for (const y of years) {
      const rs = scored.filter(r => yearOf(r) === y);
      console.log(`  ${y}  ${coverageCell(rs)}  worst gap ${worstGap(rs).toFixed(1)}pp`);
    }
    for (const b of BUCKETS) console.log(`  ${b.padEnd(5)} (2025-26) ${coverageCell(scored.filter(r => bucketOf(r.horizon) === b))}`);
    const paired = scored.map(r => ({ date: r.date, d: (r.p - r.y) ** 2 - (baseByKey.get(key(r))!.p - r.y) ** 2 }));
    const meanD = (ps: typeof paired) => ps.reduce((a, p) => a + p.d, 0) / ps.length;
    const [, brierHi] = dayBlockCi(paired, meanD);
    const dHold = holdShare(scored) - holdShare(later(base));
    const coverageOk = years.every(y => worstGap(scored.filter(r => yearOf(r) === y)) <= 1);
    const brierOk = brierHi <= 0.0005;
    const holdOk = Math.abs(dHold) <= 0.01;
    console.log(`  paired Brier ${meanD(paired).toFixed(5)} (95% upper ${brierHi.toFixed(5)})  HOLD ${dHold >= 0 ? '+' : ''}${pct(dHold)}pp`);
    console.log(`  (a) coverage ${coverageOk ? 'pass' : 'FAIL'}  (b) Brier ${brierOk ? 'pass' : 'FAIL'}  (c) HOLD ${holdOk ? 'pass' : 'FAIL'}  => ${coverageOk && brierOk && holdOk ? 'PASSES' : 'fails'}`);
  }
}

if (import.meta.main) {
  const [mode, a, b, c] = process.argv.slice(2);
  const candidate = c === undefined ? 'baseline' : c;
  if (mode === 'run' && (a === 'BTC-USD' || a === 'ETH-USD') && b && ['baseline', 'A', 'B', 'C'].includes(candidate)) {
    await run(a, Number(b), candidate as Candidate | 'baseline');
  } else if (mode === 'report') await report(join(ROOT, 'baseline'));
  else if (mode === 'fit') await showFits();
  else if (mode === 'score') await score();
  else console.log('usage: crypto-interval-coverage.ts run <BTC-USD|ETH-USD> <horizon> [A|B|C] | report | fit | score');
}
