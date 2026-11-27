/**
 * H3 (docs/reviews/polymarket-markov-connection-review-2026-11-26.md): does the volatility implied by a Polymarket
 * strike ladder ("<asset> above $K on <date>", one expiry, several strikes) predict the realized move better than the
 * trailing volatility the forecast width uses today?
 *
 * For each observation day and open expiry (exact end time) with >= 3 informative strikes (0.03 < P(above) < 0.97)
 * whose P(above) falls strictly with the strike, and >= 12 h to resolution, z = PhiInv(1 - P) is linear in ln(K / spot)
 * with slope 1 / sigma. The ladder's sigma covers the time to resolution; it is rescaled to
 * the day-close-to-expiry-close horizon the daily closes can score. Control: trailing 30-day sigma x FORECAST_VOL_SCALE.
 * Both are zero-mean normals in log price (H3 is about width, not direction). Scores: log score and CRPS differences
 * (negative = ladder better), interval coverage, and a blend whose weight is fitted on the first half of the days and
 * scored on the second. 95% CIs resample whole observation days and, separately, whole expiries; the verdict uses the
 * wider. Pass rule (frozen in the review): >= 150 ladder-days, CI excluding zero, coverage within +-1pp of nominal.
 *
 *   bun run src/tools/finance/backtest/ladder-implied-vol-benchmark.ts [ladder fixtures...]
 */
import { join } from 'node:path';
import { normCDF, normPPF } from '../../../utils/stats.js';
import { inverseStudentTCDF, lgamma, studentTCDF } from '../markov-distribution/confidence-intervals.js';
import { FORECAST_VOL_SCALE } from '../markov-distribution/core.js';
import type { AnchorHistoryFixture, AnchorHistoryMarket } from './polymarket-anchor-history.js';

const DAY_MS = 86_400_000;
const TRAILING_DAYS = 30;
const LEVELS = [[0.9, 1.6449], [0.95, 1.96], [0.99, 2.5758]] as const;
const NU_GRID = [2.5, 3, 3.5, 4, 5, 6, 8, 10, 15, 20, 30, 50, Infinity];
const HORIZONS = [['1d', 1, 1], ['2-3d', 2, 3], ['4-7d', 4, 7]] as const;

export interface LadderObservation {
  day: string;
  expiry: string;
  /** Whole days from the observation day's close to the expiry day's close. */
  h: number;
  sigmaLadder: number;
  sigmaTrailing: number;
  realized: number;
}

export function parseStrike(question: string): { strike: number; below: boolean } | null {
  const m = question.match(/\b(above|below)\s+\$([\d,]+(?:\.\d+)?)\s*(k)?\b/i);
  if (!m) return null;
  const strike = Number(m[2]!.replace(/,/g, '')) * (m[3] ? 1000 : 1);
  return Number.isFinite(strike) && strike > 0 ? { strike, below: m[1]!.toLowerCase() === 'below' } : null;
}

/**
 * Sigma of ln(S_T / spot) implied by the ladder, or null with fewer than 3 informative strikes or when P(above) does
 * not fall strictly with the strike: a flat ladder is unquoted (every strike at the 0.5 listing price) and a rising one
 * implies negative probability, so neither is a forecast.
 */
export function fitLadderSigma(points: ReadonlyArray<{ strike: number; pAbove: number }>, spot: number): number | null {
  const informative = points.filter(p => p.pAbove > 0.03 && p.pAbove < 0.97).sort((a, b) => a.strike - b.strike);
  if (informative.length < 3 || informative.some((p, i) => i > 0 && p.pAbove >= informative[i - 1]!.pAbove)) return null;
  const xs = informative.map(p => Math.log(p.strike / spot));
  const zs = informative.map(p => normPPF(1 - p.pAbove));
  const mx = mean(xs);
  const mz = mean(zs);
  const sxx = xs.reduce((a, x) => a + (x - mx) ** 2, 0);
  const slope = xs.reduce((a, x, i) => a + (x - mx) * (zs[i]! - mz), 0) / sxx;
  return slope > 0 ? 1 / slope : null;
}

export function ladderObservations(
  markets: readonly AnchorHistoryMarket[],
  closes: { dates: string[]; closes: number[] },
): LadderObservation[] {
  const index = new Map(closes.dates.map((d, i) => [d, i]));
  const groups = new Map<string, Array<{ strike: number; pAbove: number; endMs: number }>>();
  for (const m of markets) {
    const parsed = parseStrike(m.question);
    if (!parsed) continue;
    const endMs = Date.parse(m.endDate);
    for (const [day, p] of Object.entries(m.daily)) {
      if (endMs <= Date.parse(`${day}T23:59:59Z`)) continue; // sampled after resolution
      const key = `${day}|${m.endDate}`; // one ladder per exact end time: timed intraday markets share the date
      const group = groups.get(key) ?? [];
      group.push({ strike: parsed.strike, pAbove: parsed.below ? 1 - p : p, endMs });
      groups.set(key, group);
    }
  }
  const rows: LadderObservation[] = [];
  for (const [key, group] of groups) {
    const [day, endDate] = key.split('|') as [string, string];
    const expiry = endDate.slice(0, 10);
    const t = index.get(day);
    const e = index.get(expiry);
    const tau = (group[0]!.endMs - Date.parse(`${day}T23:59:59Z`)) / DAY_MS;
    // Under 12 hours to resolution the sqrt(h / tau) rescale blows up the noise (x300 one second out); only the timed
    // intraday markets get that close to a day-close sample.
    if (t === undefined || e === undefined || t < TRAILING_DAYS || e <= t || tau < 0.5) continue;
    const spot = closes.closes[t]!;
    const sigma = fitLadderSigma(group, spot);
    if (sigma === null) continue;
    const h = e - t;
    const returns = [];
    for (let i = t - TRAILING_DAYS + 1; i <= t; i++) returns.push(Math.log(closes.closes[i]! / closes.closes[i - 1]!));
    rows.push({
      day,
      expiry,
      h,
      sigmaLadder: sigma * Math.sqrt(h / tau),
      sigmaTrailing: stdev(returns) * Math.sqrt(h) * FORECAST_VOL_SCALE,
      realized: Math.log(closes.closes[e]! / spot),
    });
  }
  return rows.sort((a, b) => a.day.localeCompare(b.day) || a.expiry.localeCompare(b.expiry));
}

function mean(xs: readonly number[]): number {
  return xs.reduce((a, b) => a + b, 0) / xs.length;
}

function stdev(xs: readonly number[]): number {
  const m = mean(xs);
  return Math.sqrt(xs.reduce((a, x) => a + (x - m) ** 2, 0) / (xs.length - 1));
}

/** Negative log density of a zero-mean normal, constant dropped. */
const nll = (x: number, s: number) => Math.log(s) + 0.5 * (x / s) ** 2;

/** CRPS of a zero-mean normal. */
function crps(x: number, s: number): number {
  const z = x / s;
  return s * (z * (2 * normCDF(z) - 1) + 2 * Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI) - 1 / Math.sqrt(Math.PI));
}

/** 95% CI of mean(values), resampling whole clusters. */
function clusteredCi(values: readonly number[], clusterOf: readonly string[]): [number, number] {
  const byCluster = new Map<string, number[]>();
  values.forEach((v, i) => {
    const g = byCluster.get(clusterOf[i]!) ?? [];
    g.push(v);
    byCluster.set(clusterOf[i]!, g);
  });
  const clusters = [...byCluster.values()];
  let seed = 7;
  const rand = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
  const draws: number[] = [];
  for (let b = 0; b < 2000; b++) draws.push(mean(clusters.flatMap(() => clusters[Math.floor(rand() * clusters.length)]!)));
  draws.sort((a, b) => a - b);
  return [draws[49]!, draws[1949]!];
}

/** The wider of the day- and expiry-clustered CIs. */
function ci(values: readonly number[], rows: readonly LadderObservation[]): [number, number] {
  const [dLo, dHi] = clusteredCi(values, rows.map(r => r.day));
  const [eLo, eHi] = clusteredCi(values, rows.map(r => r.expiry));
  return [Math.min(dLo, eLo), Math.max(dHi, eHi)];
}

/** Student-t scale for standard deviation s, matched the way studentTSurvival matches it; nu = Infinity is the normal. */
const tScale = (s: number, nu: number) => (Number.isFinite(nu) ? s * Math.sqrt((nu - 2) / nu) : s);

/** Log density of a zero-mean Student-t with standard deviation s, constants kept so different nu compare. */
function studentTLogDensity(x: number, s: number, nu: number): number {
  const z = x / tScale(s, nu);
  if (!Number.isFinite(nu)) return -Math.log(s) - 0.5 * Math.log(2 * Math.PI) - 0.5 * z * z;
  return lgamma((nu + 1) / 2) - lgamma(nu / 2) - 0.5 * Math.log(nu * Math.PI) - Math.log(tScale(s, nu))
    - ((nu + 1) / 2) * Math.log1p(z * z / nu);
}

/** CRPS of a zero-mean Student-t with standard deviation s (closed form, Jordan, Krueger & Lerch 2019). */
export function studentTCrps(x: number, s: number, nu: number): number {
  if (!Number.isFinite(nu)) return crps(x, s);
  const k = tScale(s, nu);
  const z = x / k;
  const lbeta = (a: number, b: number) => lgamma(a) + lgamma(b) - lgamma(a + b);
  const density = Math.exp(studentTLogDensity(z, Math.sqrt(nu / (nu - 2)), nu)); // the standard t
  const constant = (2 * Math.sqrt(nu) * Math.exp(lbeta(0.5, nu - 0.5) - 2 * lbeta(0.5, nu / 2))) / (nu - 1);
  return k * (z * (2 * studentTCDF(z, nu) - 1) + (2 * density * (nu + z * z)) / (nu - 1) - constant);
}

/** Degrees of freedom from NU_GRID with the best mean log score on these rows. */
export function fitTailDof(rows: readonly LadderObservation[], sigmaOf: (r: LadderObservation) => number): number {
  let best = -Infinity;
  let bestNu = Infinity;
  for (const nu of NU_GRID) {
    const score = mean(rows.map(r => studentTLogDensity(r.realized, sigmaOf(r), nu)));
    if (score > best) [best, bestNu] = [score, nu];
  }
  return bestNu;
}

function tCoverage(rows: readonly LadderObservation[], sigmaOf: (r: LadderObservation) => number, nu: number): number[] {
  return LEVELS.map(([level]) => {
    const p = (1 + level) / 2;
    const q = Number.isFinite(nu) ? inverseStudentTCDF(p, nu) : normPPF(p);
    return mean(rows.map(r => (Math.abs(r.realized) <= q * tScale(sigmaOf(r), nu) ? 1 : 0)));
  });
}

/**
 * Fatter-tailed intervals: each arm's Student-t degrees of freedom is fitted on the first half of the days (as the
 * blend weight is) and both arms are scored on the second half, overall and per horizon with the same nu.
 */
export function scoreStudentT(rows: readonly LadderObservation[]) {
  const days = [...new Set(rows.map(r => r.day))];
  const splitDay = days[Math.floor(days.length / 2)]!;
  const fitRows = rows.filter(r => r.day < splitDay);
  const testRows = rows.filter(r => r.day >= splitDay);
  const nuLadder = fitTailDof(fitRows, r => r.sigmaLadder);
  const nuTrailing = fitTailDof(fitRows, r => r.sigmaTrailing);
  const evaluate = (rs: readonly LadderObservation[]) => {
    const dLog = rs.map(r =>
      studentTLogDensity(r.realized, r.sigmaTrailing, nuTrailing) - studentTLogDensity(r.realized, r.sigmaLadder, nuLadder));
    const dCrps = rs.map(r => studentTCrps(r.realized, r.sigmaLadder, nuLadder) - studentTCrps(r.realized, r.sigmaTrailing, nuTrailing));
    return {
      n: rs.length,
      days: new Set(rs.map(r => r.day)).size,
      logScore: { mean: mean(dLog), ci: ci(dLog, rs) },
      crps: { mean: mean(dCrps), ci: ci(dCrps, rs) },
      coverage: { ladder: tCoverage(rs, r => r.sigmaLadder, nuLadder), trailing: tCoverage(rs, r => r.sigmaTrailing, nuTrailing) },
    };
  };
  return {
    splitDay,
    nuLadder,
    nuTrailing,
    test: evaluate(testRows),
    byHorizon: HORIZONS.map(([name, lo, hi]) => ({ name, ...evaluate(testRows.filter(r => r.h >= lo && r.h <= hi)) })),
  };
}

export function scoreLadders(rows: readonly LadderObservation[]) {
  const days = [...new Set(rows.map(r => r.day))];
  const splitDay = days[Math.floor(days.length / 2)]!;
  const fitRows = rows.filter(r => r.day < splitDay);
  const testRows = rows.filter(r => r.day >= splitDay);
  const blendSigma = (r: LadderObservation, w: number) => w * r.sigmaLadder + (1 - w) * r.sigmaTrailing;
  let weight = 0;
  let best = Infinity;
  for (let w = 0; w <= 1.0001; w += 0.05) {
    const score = mean(fitRows.map(r => nll(r.realized, blendSigma(r, w))));
    if (score < best) [best, weight] = [score, Math.round(w * 100) / 100];
  }
  const dLog = rows.map(r => nll(r.realized, r.sigmaLadder) - nll(r.realized, r.sigmaTrailing));
  const dCrps = rows.map(r => crps(r.realized, r.sigmaLadder) - crps(r.realized, r.sigmaTrailing));
  const dBlend = testRows.map(r => nll(r.realized, blendSigma(r, weight)) - nll(r.realized, r.sigmaTrailing));
  const coverage = (rs: readonly LadderObservation[], sigma: (r: LadderObservation) => number) =>
    LEVELS.map(([, q]) => mean(rs.map(r => Math.abs(r.realized) <= q * sigma(r) ? 1 : 0)));
  const median = (xs: number[]) => xs.sort((a, b) => a - b)[xs.length >> 1]!;
  return {
    n: rows.length,
    ladderDays: days.length,
    expiries: new Set(rows.map(r => r.expiry)).size,
    medianSigmaLadder: median(rows.map(r => r.sigmaLadder)),
    medianSigmaTrailing: median(rows.map(r => r.sigmaTrailing)),
    logScore: { mean: mean(dLog), ci: ci(dLog, rows) },
    crps: { mean: mean(dCrps), ci: ci(dCrps, rows) },
    coverage: { trailing: coverage(rows, r => r.sigmaTrailing), ladder: coverage(rows, r => r.sigmaLadder) },
    blend: {
      weight,
      splitDay,
      testN: testRows.length,
      logScore: { mean: mean(dBlend), ci: ci(dBlend, testRows) },
      coverage: coverage(testRows, r => blendSigma(r, weight)),
    },
  };
}

if (import.meta.main) {
  const fixtures = join(import.meta.dir, '..', 'fixtures');
  const ladderPaths = process.argv.length > 2 ? process.argv.slice(2) : [
    join(fixtures, 'polymarket-btc-anchor-history.json'),
    join(fixtures, 'polymarket-eth-ladder-history-2025-2026.json'),
  ];
  const prices = (await Bun.file(join(fixtures, 'crypto-prices-2025-2026.json')).json()) as { tickers: Record<string, { dates: string[]; closes: number[] }> };
  const pct = (x: number) => `${(100 * x).toFixed(1)}%`;
  const f4 = (x: number) => x.toFixed(4);
  const report = (label: string, rows: LadderObservation[]) => {
    if (rows.length === 0) return;
    const s = scoreLadders(rows);
    const covLine = (c: number[]) => c.map(pct).join(' / ');
    const coverageOk = (c: number[]) => c.every((v, i) => Math.abs(v - LEVELS[i]![0]) <= 0.01);
    console.log(`\n${label}: ${s.n} ladder observations on ${s.ladderDays} days, ${s.expiries} expiries`);
    console.log(`  median sigma: ladder ${pct(s.medianSigmaLadder)} vs trailing x${FORECAST_VOL_SCALE} ${pct(s.medianSigmaTrailing)}`);
    console.log(`  ladder minus trailing, log score ${f4(s.logScore.mean)} [${f4(s.logScore.ci[0])}, ${f4(s.logScore.ci[1])}], CRPS ${s.crps.mean.toExponential(2)} [${s.crps.ci[0].toExponential(2)}, ${s.crps.ci[1].toExponential(2)}]`);
    console.log(`  coverage 90/95/99: trailing ${covLine(s.coverage.trailing)}, ladder ${covLine(s.coverage.ladder)}`);
    console.log(`  blend w=${s.blend.weight} (fitted before ${s.blend.splitDay}), on the later ${s.blend.testN}: log score ${f4(s.blend.logScore.mean)} [${f4(s.blend.logScore.ci[0])}, ${f4(s.blend.logScore.ci[1])}], coverage ${covLine(s.blend.coverage)}`);
    const passes = (m: { ci: [number, number] }, cov: number[]) => s.ladderDays >= 150 && m.ci[1] < 0 && coverageOk(cov);
    console.log(`  pass rule (>=150 days, CI < 0, coverage within 1pp): ladder ${passes(s.logScore, s.coverage.ladder) ? 'PASS' : 'fail'}, blend ${passes(s.blend.logScore, s.blend.coverage) ? 'PASS' : 'fail'}`);
    for (const [name, lo, hi] of HORIZONS) {
      const part = rows.filter(r => r.h >= lo && r.h <= hi);
      if (part.length < 30) continue;
      const d = part.map(r => nll(r.realized, r.sigmaLadder) - nll(r.realized, r.sigmaTrailing));
      const [lo95, hi95] = ci(d, part);
      console.log(`    h ${name.padEnd(4)} n=${String(part.length).padStart(5)}: log score ${f4(mean(d))} [${f4(lo95)}, ${f4(hi95)}]`);
    }
    const t = scoreStudentT(rows);
    console.log(`  Student-t, nu fitted before ${t.splitDay}: ladder ${t.nuLadder}, trailing ${t.nuTrailing}; scored on the later days`);
    for (const [label, e] of [['all', t.test] as const, ...t.byHorizon.map(b => [`h ${b.name}`, b] as const)]) {
      const gate = e.days >= 150 && e.logScore.ci[1] < 0 && coverageOk(e.coverage.ladder);
      console.log(`    ${label.padEnd(6)} n=${String(e.n).padStart(5)} on ${e.days} days: log score ${f4(e.logScore.mean)} [${f4(e.logScore.ci[0])}, ${f4(e.logScore.ci[1])}], CRPS ${e.crps.mean.toExponential(2)} [${e.crps.ci[0].toExponential(2)}, ${e.crps.ci[1].toExponential(2)}]`);
      console.log(`           coverage trailing ${covLine(e.coverage.trailing)}, ladder ${covLine(e.coverage.ladder)}: gate ${gate ? 'PASS' : 'fail'}`);
    }
  };
  const all: LadderObservation[] = [];
  for (const path of ladderPaths) {
    const fixture = (await Bun.file(path).json()) as AnchorHistoryFixture;
    const asset = /ethereum/i.test(fixture.source) ? 'ETH-USD' : 'BTC-USD';
    const rows = ladderObservations(fixture.markets, prices.tickers[asset]!);
    report(`${asset} (${path.split('/').pop()})`, rows);
    all.push(...rows.map(r => ({ ...r, expiry: `${asset}|${r.expiry}` })));
  }
  if (ladderPaths.length > 1) report('All assets', all.sort((a, b) => a.day.localeCompare(b.day)));
}
