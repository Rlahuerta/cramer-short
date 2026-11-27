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
 * Anchored comparison (added 2026-11-27 and committed before scoring it). Do Polymarket anchors change the coverage?
 * The same origins and live configuration, with the markets open at each origin's daily close fed through walkForward's
 * anchorsAt hook by anchorsAtFromHistory, as in anchored-replay-benchmark.ts: BTC from
 * fixtures/polymarket-btc-anchor-history.json, ETH from fixtures/polymarket-eth-ladder-history-2025-2026.json. The
 * model-only arm is the baseline run. Window: origins from the first day the asset's history has a market price.
 * Horizons 1/2/3/7d, plus 14d for an asset only if at least 30 of its 14d window origins have a trusted anchor (counts
 * reported either way). Two populations: every origin in the window, and the origins whose anchored forecast had at
 * least one trusted anchor. Reported: anchored and model-only coverage per asset, horizon and year; the paired coverage
 * difference (anchored minus model-only) with day-block CIs; anchored-step counts; the paired Brier change of P(up).
 * Verdict statistic, per level (90/95/99), pooled over BTC+ETH and per asset, in each population: the paired change in
 * |coverage - nominal| from model-only to anchored, with its 95% day-block CI.
 * - Production problem: the CI lies above zero (anchors push coverage away from nominal) at any level, pooled or per
 *   asset, in either population.
 * - Anchors help coverage: no production problem, and the pooled CI over the whole window lies below zero at one or
 *   more levels.
 * - Otherwise no detectable effect. No production default changes here; a problem is reported with a proposed change.
 *
 * Anchor horizon test (added 2026-11-27 and committed before scoring any candidate). The anchored comparison found,
 * outside its rule, that anchors widen the 1d intervals and narrow the 14d ones. Diagnosis (`anchor-gaps`, no
 * outcomes): for crypto at <= 14 days a mature market with volume is high-trust whatever its resolution date (trust 0.9
 * within 2 days of the target, else 0.7; both clear 0.6), and a high-trust anchor replaces the model's P(>K) at its
 * strike with weight (1 - mixing weight) x distance weight, whatever its expiry. The date-gap discount only runs in the
 * fallback, when no anchor is high-trust. At 14d every blended anchor resolves more than 5 days before the target; at
 * 1d, 71% (BTC) / 77% (ETH) of the blend weight resolves within half a day of it.
 * Candidates, crypto at <= 14 days; an anchor is off-horizon when it resolves more than g days from the target:
 *   L. off-horizon anchors become low-trust (blended at half weight);
 *   D. off-horizon anchors are dropped;
 *      g for L and for D fitted from {0.5, 1, 2} on BTC origins from the window start to 2025-06-30 (ETH's window
 *      starts later), minimizing the mean over horizons of the largest |coverage - nominal| at 90/95/99; ties keep
 *      the larger g;
 *   N. no anchors at horizons above 7 days.
 * Scored against today's anchored forecast on the same window origins from 2025-07-01 (BTC and ETH, 1/2/3/7/14d). A
 * candidate passes only if:
 *   (a) at no horizon and level, pooled over BTC+ETH, does the paired change in |coverage - nominal| have a 95%
 *       day-block CI above zero;
 *   (b) at 1d and at 14d that change, averaged over the three levels, is <= 0 and is < 0 at one of them, and at 1d or
 *       14d some level's CI lies below zero;
 *   (c) the paired Brier change of P(up) has a 95% upper bound <= +0.0005, so the anchors' Brier gain is kept.
 * The HOLD-share move is reported. If several pass, the smallest paired Brier change ships, test-first and mirrored in
 * research/; none passing, no change.
 *
 * Anchor rescaling test (added 2026-11-27 and committed before scoring any R run). The horizon test showed that
 * off-horizon anchors carry the right location but the wrong spread: dropping them gave back their Brier gain at 7d and
 * 14d. Candidates, crypto at <= 14 days, no fitted parameter:
 *   R.  an anchor resolving more than 2 days from the target (outside the code's near-target tolerance) has its P(>K)
 *       moved to the horizon by rescaleAnchorToHorizon; its trust and blend weight stay as today;
 *   R1. every anchor is moved by its exact expiry / horizon ratio, the on-horizon ones included.
 * The transform reads the market as a driftless normal in log price and scales its z = PhiInv(1 - p) by
 * sqrt(days to the market's end / horizon days). Drift is ignored: the market's implied median stays put, since daily
 * crypto drift is small next to the 1-14 day spread. A market resolving before the target is stretched (z shrinks,
 * wider); one resolving after it is compressed. A market at or past its end (days <= 0, which the replay never offers
 * because it keeps only open markets) and a certainty (p = 0 or 1) are left unchanged. There is no floor on p: a very
 * short-dated market quoted at the price floor understates its |z|, which pulls its rescaled far-strike probability
 * toward 0.5. Scored exactly as the anchor horizon test (against today's anchored forecast from 2025-07-01, ETH from
 * 2025-08-25, BTC+ETH, 1/2/3/7/14d, criteria (a)-(c)), with the paired Brier per horizon and the HOLD move reported.
 * If both pass, the smaller paired Brier change ships, test-first and mirrored in research/; none passing, no change.
 * Result: R passed and ships as the production default for crypto at <= 14 days. Every anchored run here other than R
 * and R1 therefore pins anchorHorizonRescaleDays to Infinity, to reproduce the forecast it scored.
 *
 * Outputs go to .cramer-short/crypto-interval-coverage/<baseline|A|B|C|anchored>. Each run takes a few minutes; run
 * them in parallel.
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts run <BTC-USD|ETH-USD> <horizon>   # baseline
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts report                           # step 1
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts fit                              # step 2 fit
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts run <BTC-USD|ETH-USD> <horizon> <A|B|C>
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts score                            # step 2 score
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts run <BTC-USD|ETH-USD> <horizon> anchored
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts anchors                          # anchored comparison
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts anchor-gaps                      # horizon test diagnosis
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts run BTC-USD <horizon> anchor-<L|D>-<0.5|1|2>
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts anchor-fit                       # fitted g for L and D
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts run <BTC-USD|ETH-USD> <horizon> <anchor-L-g|anchor-D-g|anchor-N>
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts anchor-score                     # horizon test verdict
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts run <BTC-USD|ETH-USD> <horizon> <anchor-R|anchor-R1>
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts rescale-score                    # rescaling test verdict
 *   bun run src/tools/finance/backtest/crypto-interval-coverage.ts channels <BTC-USD|ETH-USD> <horizon>  # which anchor channel moves it
 */
import { join } from 'node:path';
import { getAssetProfile } from '../markov-distribution/asset-profile.js';
import { horizonStudentTNu, inverseStudentTCDF } from '../markov-distribution/confidence-intervals.js';
import { applyCryptoTerminalAnchorFallback, extractPriceThresholds, isInformativeAnchor } from '../markov-distribution.js';
import { btcLiveCell } from './live-config-benchmark.js';
import { intervalCoverage, type SkillRow } from './long-history-skill.js';
import { anchorsAtFromHistory, type AnchorHistoryFixture } from './polymarket-anchor-history.js';
import { walkForward } from './walk-forward.js';

export const ASSETS = ['BTC-USD', 'ETH-USD'] as const;
export const HORIZONS = [1, 2, 3, 7, 14];
const ETH_HISTORY_PRICES = 120;
const CRYPTO_MODEL_ONLY_MIN_R2 = -0.03; // markov-distribution.ts: below it the tool abstains on a model-only crypto forecast
const SEAM = '2026-01-01';
const ROOT = join(process.cwd(), '.cramer-short', 'crypto-interval-coverage');
const BLOCK_DAYS = 28;
const LEVELS = [['90', 0.9], ['95', 0.95], ['99', 0.99]] as const;

/** `trusted` / `offered`: trusted anchors and markets offered at the origin (anchored runs only). */
export type CoverageRow = SkillRow & { r2: number | null; trusted?: number; offered?: number };
const ANCHOR_HISTORY = { 'BTC-USD': 'polymarket-btc-anchor-history.json', 'ETH-USD': 'polymarket-eth-ladder-history-2025-2026.json' };
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

async function anchorMarkets(asset: (typeof ASSETS)[number]) {
  return (await Bun.file(join(import.meta.dir, '..', 'fixtures', ANCHOR_HISTORY[asset])).json() as AnchorHistoryFixture).markets;
}

/** Paired change (pp) in |coverage - nominal| from model-only to anchored: negative = anchors move coverage toward nominal. */
export function missChange(pairs: ReadonlyArray<{ plain: CoverageRow; anchored: CoverageRow }>, level: Level): number {
  const key = `c${level}` as 'c90' | 'c95' | 'c99';
  const miss = (rs: CoverageRow[]) => Math.abs(100 * intervalCoverage(rs)[key] - Number(level));
  return miss(pairs.map(p => p.anchored)) - miss(pairs.map(p => p.plain));
}

/**
 * The anchor candidates: `anchor-L-<g>` / `anchor-D-<g>` (off-horizon low-trust / dropped), `anchor-N`, and the
 * rescaling test's `anchor-R` (anchors more than 2 days off moved to the horizon) and `anchor-R1` (every anchor moved).
 */
function anchorVariantConfig(variant: string) {
  if (variant === 'anchor-N') return { maxAnchorHorizonDays: 7 };
  if (variant === 'anchor-R') return { anchorHorizonRescaleDays: 2 };
  if (variant === 'anchor-R1') return { anchorHorizonRescaleDays: 0 };
  const m = /^anchor-([LD])-(0\.5|1|2)$/.exec(variant);
  return m ? { offHorizonAnchorDays: Number(m[2]), offHorizonAnchorPolicy: m[1] === 'L' ? 'low-trust' as const : 'drop' as const } : null;
}

/** Forecast every origin with the production path, plus a candidate's settings or the Polymarket anchors. */
async function run(asset: (typeof ASSETS)[number], horizon: number, candidate: string): Promise<void> {
  const { dates, closes } = await loadCloses(asset);
  const cell = asset === 'BTC-USD' ? btcLiveCell(horizon) : { config: {}, maxHistoryPrices: ETH_HISTORY_PRICES };
  const anchorVariant = anchorVariantConfig(candidate);
  const anchorsAt = candidate === 'anchored' || anchorVariant ? anchorsAtFromHistory(dates, await anchorMarkets(asset)) : undefined;
  // Anchored runs pin the pre-R forecast (no rescaling) they were scored on; R and R1 set their own.
  const experiment = candidate === 'baseline'
    ? {}
    : anchorsAt ? { anchorsAt, anchorHorizonRescaleDays: Infinity, ...anchorVariant } : await candidateConfig(candidate as Candidate, horizon);
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
    ...(anchorsAt ? { trusted: s.trustedAnchors ?? 0, offered: anchorsAt(s.t).markets.length } : {}),
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

async function loadRows(dir: string, assets: readonly string[] = ASSETS): Promise<CoverageRow[]> {
  const files = assets.flatMap(a => HORIZONS.map(h => join(dir, `${a}-${h}.json`)));
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

/**
 * Diagnosis for the anchor horizon test: at each window origin, the anchors the forecast blends (the production
 * extraction, informativeness filter and crypto fallback), by resolution gap = days from the origin's close to the
 * market's end minus the horizon (negative: the market resolves before the target). Shares are weighted by each
 * anchor's blend weight before the mixing-time factor: exp(-5 d²) for distance d from spot, halved for low trust.
 */
async function anchorGaps(): Promise<void> {
  const buckets = [['< -5', -Infinity, -5], ['-5..-2', -5, -2], ['-2..-0.5', -2, -0.5], ['-0.5..0.5', -0.5, 0.5], ['0.5..2', 0.5, 2], ['2..5', 2, 5], ['> 5', 5, Infinity]] as const;
  for (const asset of ASSETS) {
    const { dates, closes } = await loadCloses(asset);
    const markets = await anchorMarkets(asset);
    const anchorsAt = anchorsAtFromHistory(dates, markets);
    const start = dates.indexOf(markets.flatMap(m => Object.keys(m.daily)).sort()[0]!);
    console.log(`${asset} from ${dates[start]}: gap buckets (days) as shares of blend weight; trust weights of high-trust anchors`);
    for (const h of HORIZONS) {
      const weightByBucket = buckets.map(() => 0);
      const trustWeights = new Map<number, number>();
      let origins = 0, withAnchor = 0, fallback = 0, nearHigh = 0, used = 0;
      for (let t = start; t + h < closes.length; t++) {
        const { markets: offered, referenceTimeMs } = anchorsAt(t);
        const price = closes[t]!;
        const opts = { ticker: asset, horizonDays: h, referenceTimeMs };
        const strict = extractPriceThresholds(offered, opts).filter(a => isInformativeAnchor(a, price));
        if (offered.length > 0 && !strict.some(a => a.trustScore === 'high')) fallback++;
        const blended = applyCryptoTerminalAnchorFallback(offered, strict, asset, h, referenceTimeMs)
          .filter(a => isInformativeAnchor(a, price) && (a.trustScore === 'high' || a.trustWeight !== 0));
        origins++;
        if (blended.length > 0) withAnchor++;
        used += blended.length;
        const gapOf = (a: (typeof blended)[number]) => (Date.parse(a.endDate!) - referenceTimeMs) / 86_400_000 - h;
        if (blended.some(a => a.trustScore === 'high' && Math.abs(gapOf(a)) <= 2)) nearHigh++;
        for (const a of blended) {
          const w = Math.exp(-5 * (a.price / price - 1) ** 2) * (a.trustScore === 'high' ? 1 : 0.5);
          const gap = gapOf(a);
          weightByBucket[buckets.findIndex(([, lo, hi]) => gap >= lo && gap < hi)]! += w;
          if (a.trustScore === 'high') {
            const tw = Math.round(100 * (a.trustWeight ?? 1)) / 100;
            trustWeights.set(tw, (trustWeights.get(tw) ?? 0) + 1);
          }
        }
      }
      const total = weightByBucket.reduce((a, b) => a + b, 0) || 1;
      const shares = buckets.map(([label], i) => `${label}: ${(100 * weightByBucket[i]! / total).toFixed(0)}%`).join('  ');
      const tws = [...trustWeights].sort((a, b) => b[0] - a[0]).map(([w, n]) => `${w}×${n}`).join(' ');
      console.log(`  ${String(h).padStart(2)}d origins ${origins}, with an anchor ${withAnchor}, fallback ${fallback}, near-target high-trust ${nearHigh}, anchors/origin ${(used / origins).toFixed(1)}`);
      console.log(`      ${shares}   trust ${tws}`);
    }
  }
}

const ANCHOR_FIT_END = '2025-07-01';
const windowStartOf = async (asset: (typeof ASSETS)[number]) => (await anchorMarkets(asset)).flatMap(m => Object.keys(m.daily)).sort()[0]!;
/** Mean over horizons of the largest |coverage - nominal| (pp) at 90/95/99. */
const meanWorstGap = (rows: CoverageRow[]) => HORIZONS.reduce((a, h) => a + worstGap(rows.filter(r => r.horizon === h)), 0) / HORIZONS.length;

/** The anchor horizon test's fit: g for L and for D on BTC window origins before 2025-07; ties keep the larger g. */
async function fitAnchorGap(verbose: boolean): Promise<{ L: number; D: number }> {
  const start = await windowStartOf('BTC-USD');
  const fitRows = async (dir: string) => (await loadRows(join(ROOT, dir), ['BTC-USD'])).filter(r => r.date >= start && r.date < ANCHOR_FIT_END);
  if (verbose) console.log(`Fit on BTC origins ${start}..${ANCHOR_FIT_END} (exclusive): mean over horizons of the worst gap; today's anchored ${meanWorstGap(await fitRows('anchored')).toFixed(2)}pp`);
  const fitted = { L: 2, D: 2 };
  for (const policy of ['L', 'D'] as const) {
    let best = Infinity;
    for (const g of [2, 1, 0.5]) {
      const objective = meanWorstGap(await fitRows(`anchor-${policy}-${g}`));
      if (verbose) console.log(`  ${policy} g=${g}: ${objective.toFixed(2)}pp`);
      if (objective < best) [best, fitted[policy]] = [objective, g];
    }
  }
  if (verbose) console.log(`Fitted: L g=${fitted.L}, D g=${fitted.D}`);
  return fitted;
}

/** The anchor horizon test's candidates L (fitted g), D (fitted g) and N. */
async function anchorScore(): Promise<void> {
  const fitted = await fitAnchorGap(false);
  console.log(`Fitted L g=${fitted.L}, D g=${fitted.D}`);
  await scoreAnchorCandidates([['L', `anchor-L-${fitted.L}`], ['D', `anchor-D-${fitted.D}`], ['N', 'anchor-N']]);
}

/** Each candidate against today's anchored forecast on window origins from 2025-07-01, by criteria (a)-(c). */
async function scoreAnchorCandidates(candidates: ReadonlyArray<readonly [string, string]>): Promise<void> {
  const from: Record<string, string> = {};
  for (const asset of ASSETS) from[asset] = [ANCHOR_FIT_END, await windowStartOf(asset)].sort()[1]!;
  const key = (r: CoverageRow) => `${r.ticker}|${r.horizon}|${r.date}`;
  const scored = (rs: CoverageRow[]) => rs.filter(r => r.date >= from[r.ticker]!);
  const current = new Map(scored(await loadRows(join(ROOT, 'anchored'))).map(r => [key(r), r]));
  const holdShare = (rs: CoverageRow[]) => rs.filter(r => r.rec === 'HOLD').length / rs.length;
  const passing: Array<[string, number]> = [];
  console.log(`Scored from ${from['BTC-USD']} (BTC) and ${from['ETH-USD']} (ETH)`);
  for (const [name, dir] of candidates) {
    // missChange reads `plain` as before (today's anchored forecast) and `anchored` as after (the candidate).
    const pairs = scored(await loadRows(join(ROOT, dir))).map(r => ({ date: r.date, horizon: r.horizon, plain: current.get(key(r))!, anchored: r }));
    console.log(`\nCandidate ${name} (${dir}), ${pairs.length} pairs: today -> candidate coverage, change in |miss| [CI] by horizon`);
    let worse = false;
    const meanChange: Record<number, number> = {};
    let significant = false;
    for (const h of HORIZONS) {
      const ps = pairs.filter(p => p.horizon === h);
      const cells = LEVELS.map(([level]) => {
        const [lo, hi] = dayBlockCi(ps, xs => missChange(xs, level));
        if (lo > 0) worse = true;
        if (hi < 0 && (h === 1 || h === 14)) significant = true;
        const cov = (rs: CoverageRow[]) => (100 * intervalCoverage(rs)[`c${level}` as 'c90' | 'c95' | 'c99']).toFixed(1);
        return `${level}: ${cov(ps.map(p => p.plain))} -> ${cov(ps.map(p => p.anchored))} ${missChange(ps, level).toFixed(1)} [${lo.toFixed(1)}, ${hi.toFixed(1)}]`;
      });
      meanChange[h] = LEVELS.reduce((a, [level]) => a + missChange(ps, level), 0) / LEVELS.length;
      console.log(`  ${String(h).padStart(2)}d n=${ps.length}  ${cells.join('  ')}`);
    }
    const fmtCi = (xs: typeof pairs, stat: (ys: typeof pairs) => number, digits: number) => {
      const [lo, hi] = dayBlockCi(xs, stat);
      return `${stat(xs).toFixed(digits)} [${lo.toFixed(digits)}, ${hi.toFixed(digits)}]`;
    };
    const brier = (xs: typeof pairs) => xs.reduce((a, p) => a + (p.anchored.p - p.anchored.y) ** 2 - (p.plain.p - p.plain.y) ** 2, 0) / xs.length;
    const [brierLo, brierHi] = dayBlockCi(pairs, brier);
    const dHold = holdShare(pairs.map(p => p.anchored)) - holdShare(pairs.map(p => p.plain));
    const improves = meanChange[1]! <= 0 && meanChange[14]! <= 0 && (meanChange[1]! < 0 || meanChange[14]! < 0) && significant;
    const pass = !worse && improves && brierHi <= 0.0005;
    console.log(`  paired Brier ${brier(pairs).toFixed(5)} [${brierLo.toFixed(5)}, ${brierHi.toFixed(5)}]  HOLD ${dHold >= 0 ? '+' : ''}${(100 * dHold).toFixed(1)}pp`);
    console.log(`  paired Brier by horizon: ${HORIZONS.map(h => `${h}d ${fmtCi(pairs.filter(p => p.horizon === h), brier, 4)}`).join('  ')}`);
    console.log(`  mean |miss| change 1d ${meanChange[1]!.toFixed(2)}, 14d ${meanChange[14]!.toFixed(2)}`);
    console.log(`  (a) no horizon worse ${worse ? 'FAIL' : 'pass'}  (b) improves at 1d/14d ${improves ? 'pass' : 'FAIL'}  (c) Brier ${brierHi <= 0.0005 ? 'pass' : 'FAIL'}  => ${pass ? 'PASSES' : 'fails'}`);
    if (pass) passing.push([name, brier(pairs)]);
  }
  const ship = passing.sort((a, b) => a[1] - b[1])[0];
  console.log(`\nVerdict: ${ship ? `ship candidate ${ship[0]}` : 'no candidate passes; no change'}`);
}

/**
 * Which anchor channel moves the forecast (no outcomes): on every 5th window origin of one cell, each arm against
 * today's anchored forecast (no rescale), origin by origin: the mean change in the 90% log-width ln(hi/lo) and the mean
 * |change| in P(up). Arms: model-only, the risk-neutral-density step off, the strike blend off, and the rescale (R).
 */
async function channels(asset: (typeof ASSETS)[number], horizon: number): Promise<void> {
  const { dates, closes } = await loadCloses(asset);
  const anchorsAt = anchorsAtFromHistory(dates, await anchorMarkets(asset));
  const cell = asset === 'BTC-USD' ? btcLiveCell(horizon) : { config: {}, maxHistoryPrices: ETH_HISTORY_PRICES };
  const base = {
    ...cell.config, ticker: asset, prices: closes, horizon, warmup: dates.indexOf(await windowStartOf(asset)), stride: 5,
    maxHistoryPrices: cell.maxHistoryPrices, randomState: 42,
  };
  const today = { anchorsAt, anchorHorizonRescaleDays: Infinity };
  const arms = {
    today, model: {}, noDensity: { ...today, disableAnchorRnd: true }, noBlend: { ...today, disableAnchorStrikeBlend: true },
    R: { ...today, anchorHorizonRescaleDays: 2 },
  };
  const byT: Record<string, Map<number, { w: number; p: number }>> = {};
  for (const [name, extra] of Object.entries(arms)) {
    const { steps } = await walkForward({ ...base, ...extra });
    byT[name] = new Map(steps.map(s => [s.t, { w: Math.log(s.central90CiUpper! / s.central90CiLower!), p: s.predictedProb }]));
  }
  const vsToday = (name: string) => {
    const pairs = [...byT[name]!].flatMap(([t, x]) => (byT.today!.has(t) ? [[x, byT.today!.get(t)!] as const] : []));
    const width = pairs.reduce((a, [x, y]) => a + x.w / y.w - 1, 0) / pairs.length;
    const dp = pairs.reduce((a, [x, y]) => a + Math.abs(x.p - y.p), 0) / pairs.length;
    return `${name} width ${(100 * width).toFixed(1)}% |dP(up)| ${(100 * dp).toFixed(2)}pp`;
  };
  console.log(`${asset} ${String(horizon).padStart(2)}d n=${byT.today!.size}  vs today: ${['model', 'noDensity', 'noBlend', 'R'].map(vsToday).join('  ')}`);
}

/** The anchored comparison in the header: anchored vs baseline rows on the same origins of each asset's market window. */
async function anchors(): Promise<void> {
  const key = (r: CoverageRow) => `${r.ticker}|${r.horizon}|${r.date}`;
  const plainByKey = new Map((await loadRows(join(ROOT, 'baseline'))).map(r => [key(r), r]));
  const windowStart: Record<string, string> = {};
  for (const asset of ASSETS) windowStart[asset] = (await anchorMarkets(asset)).flatMap(m => Object.keys(m.daily)).sort()[0]!;
  const inWindow = (await loadRows(join(ROOT, 'anchored'))).filter(r => r.date >= windowStart[r.ticker]!);
  const with14: Record<string, boolean> = {};
  console.log('Anchored-step counts: window origins / >= 1 market offered / >= 1 trusted anchor');
  for (const asset of ASSETS) {
    console.log(`  ${asset}, window from ${windowStart[asset]}`);
    for (const h of HORIZONS) {
      const rs = inWindow.filter(r => r.ticker === asset && r.horizon === h);
      const trusted = rs.filter(r => (r.trusted ?? 0) > 0).length;
      if (h === 14) with14[asset] = trusted >= 30;
      console.log(`    ${String(h).padStart(2)}d ${rs.length} / ${rs.filter(r => (r.offered ?? 0) > 0).length} / ${trusted}${h === 14 ? (with14[asset] ? '  (included)' : '  (excluded: < 30 trusted)') : ''}`);
    }
  }
  type Pair = { date: string; anchored: CoverageRow; plain: CoverageRow };
  const pairs: Pair[] = inWindow
    .filter(r => r.horizon !== 14 || with14[r.ticker])
    .map(r => ({ date: r.date, anchored: r, plain: plainByKey.get(key(r))! }));
  const fmt = (ps: Pair[], stat: (xs: Pair[]) => number, digits = 1) => {
    const [lo, hi] = dayBlockCi(ps, stat);
    return `${stat(ps).toFixed(digits)} [${lo.toFixed(digits)}, ${hi.toFixed(digits)}]`;
  };
  const cov = (rs: CoverageRow[], level: Level) => 100 * intervalCoverage(rs)[`c${level}` as 'c90' | 'c95' | 'c99'];
  const coverageDiff = (level: Level) => (xs: Pair[]) => cov(xs.map(p => p.anchored), level) - cov(xs.map(p => p.plain), level);
  const brierDiff = (xs: Pair[]) => xs.reduce((a, p) => a + (p.anchored.p - p.anchored.y) ** 2 - (p.plain.p - p.plain.y) ** 2, 0) / xs.length;
  let problem = false;
  let help = false;
  const populations: Array<[string, Pair[]]> = [['window', pairs], ['trusted', pairs.filter(p => (p.anchored.trusted ?? 0) > 0)]];
  for (const [population, ps] of populations) {
    console.log(`\n== ${population === 'window' ? 'Every origin in the window' : 'Origins with >= 1 trusted anchor'} (${ps.length} pairs)`);
    for (const [set, rs] of [['pooled', ps], ...ASSETS.map(a => [a, ps.filter(p => p.anchored.ticker === a)] as const)] as Array<[string, Pair[]]>) {
      console.log(`  ${set}: model-only -> anchored coverage, paired difference [CI], change in |miss| [CI]`);
      for (const [level] of LEVELS) {
        const [lo, hi] = dayBlockCi(rs, xs => missChange(xs, level));
        const label = lo > 0 ? 'PROBLEM' : hi < 0 ? 'helps' : 'no detectable effect';
        if (lo > 0) problem = true;
        if (hi < 0 && population === 'window' && set === 'pooled') help = true;
        console.log(`    ${level}: ${cov(rs.map(p => p.plain), level).toFixed(1)} -> ${cov(rs.map(p => p.anchored), level).toFixed(1)}  diff ${fmt(rs, coverageDiff(level))}  |miss| ${fmt(rs, xs => missChange(xs, level))}  ${label}`);
      }
      console.log(`    paired Brier ${fmt(rs, brierDiff, 5)}`);
    }
    console.log('  per asset, horizon and year: model-only -> anchored, paired difference [CI], at 90 / 95 / 99');
    for (const asset of ASSETS) {
      for (const h of HORIZONS) {
        for (const y of [...new Set(ps.map(p => p.date.slice(0, 4)))].sort().concat('all')) {
          const rs = ps.filter(p => p.anchored.ticker === asset && p.anchored.horizon === h && (y === 'all' || p.date.startsWith(y)));
          if (rs.length === 0) continue;
          const cells = LEVELS.map(([level]) => `${cov(rs.map(p => p.plain), level).toFixed(1)} -> ${cov(rs.map(p => p.anchored), level).toFixed(1)} ${fmt(rs, coverageDiff(level))}`);
          console.log(`    ${asset} ${String(h).padStart(2)}d ${y.padEnd(4)} n=${String(rs.length).padStart(4)}  ${cells.join('  |  ')}`);
        }
      }
    }
  }
  console.log(`\nVerdict: ${problem ? 'production problem (anchors push coverage away from nominal somewhere; see PROBLEM)' : help ? 'anchors help coverage' : 'no detectable effect of anchors on coverage'}`);
}

if (import.meta.main) {
  const [mode, a, b, c] = process.argv.slice(2);
  const candidate = c === undefined ? 'baseline' : c;
  if (mode === 'run' && (a === 'BTC-USD' || a === 'ETH-USD') && b
    && (['baseline', 'A', 'B', 'C', 'anchored'].includes(candidate) || anchorVariantConfig(candidate))) {
    await run(a, Number(b), candidate);
  } else if (mode === 'report') await report(join(ROOT, 'baseline'));
  else if (mode === 'fit') await showFits();
  else if (mode === 'score') await score();
  else if (mode === 'anchors') await anchors();
  else if (mode === 'anchor-gaps') await anchorGaps();
  else if (mode === 'anchor-fit') await fitAnchorGap(true);
  else if (mode === 'anchor-score') await anchorScore();
  else if (mode === 'rescale-score') await scoreAnchorCandidates([['R', 'anchor-R'], ['R1', 'anchor-R1']]);
  else if (mode === 'channels' && (a === 'BTC-USD' || a === 'ETH-USD') && b) await channels(a, Number(b));
  else console.log('usage: crypto-interval-coverage.ts run <BTC-USD|ETH-USD> <horizon> [A|B|C|anchored|anchor-<L|D>-<g>|anchor-N] | report | fit | score | anchors | anchor-gaps | anchor-fit | anchor-score');
}
