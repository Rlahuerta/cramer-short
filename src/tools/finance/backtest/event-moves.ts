/**
 * H4 of the Polymarket <-> Markov review (2026-11-26): do day-over-day moves in Polymarket probabilities for scheduled
 * macro and crypto events predict the next trading day's |return| or direction beyond trailing volatility? Helpers for
 * scripts/h4-event-moves.ts, which holds the pre-registered design, builds the history and scores it.
 */
import { MS_PER_DAY } from '../../../utils/time.js';
import { normalCDF } from '../markov-distribution/confidence-intervals.js';
import { mulberry32 } from './metrics.js';

export const EVENT_GROUPS = ['fed', 'cpi', 'jobs', 'crypto_etf'] as const;
export type EventGroup = (typeof EVENT_GROUPS)[number];

const MONTH = '(?:January|February|March|April|May|June|July|August|September|October|November|December)';
const TITLE_RULES: ReadonlyArray<[EventGroup, RegExp]> = [
  ['fed', new RegExp(`^Fed Interest Rates:? ${MONTH} \\d{4}$`)],
  ['fed', new RegExp(`^Fed decision in ${MONTH}\\??$`)],
  ['cpi', /^What will U\.?S\.? inflation be from /i],
  ['cpi', /^US inflation >/],
  ['cpi', /^Annual inflation (?:above|under) /],
  ['cpi', new RegExp(`^${MONTH} Inflation(?: - (?:Monthly|Annual))?$`)],
  ['jobs', new RegExp(`^How many jobs added in ${MONTH}\\??$`, 'i')],
  ['jobs', new RegExp(`^Unemployment rate over [\\d.]+% in ${MONTH}\\?$`)],
  ['jobs', new RegExp(`^${MONTH} Unemployment Rate$`)],
];
const CRYPTO = /\b(?:bitcoin|btc|ethereum|eth|ether|solana|sol|doge|dogecoin|litecoin|ltc|ripple|xrp|cardano|ada|pepe|crypto)\b/i;

/** The pre-registered event list: which Gamma event titles count, by group. Titles only, never outcomes. */
export function classifyEventTitle(title: string): EventGroup | undefined {
  const t = title.trim();
  for (const [group, re] of TITLE_RULES) if (re.test(t)) return group;
  const cryptoEtf = /\bETFs?\b/.test(t) && CRYPTO.test(t) && /approv|begins trading|delay| in \d{4}\?/i.test(t);
  return cryptoEtf && !/flow|aum|inflow|price|parlay|attract/i.test(t) ? 'crypto_etf' : undefined;
}

const THRESHOLD_UP = /greater than|more than|\babove\b|\bover\b|>/i;
const THRESHOLD_DOWN = /less than|\bunder\b|\bbelow\b|</i;
const BAND = /or less|or more|[≤≥]|between|\d\s*k?\s*-\s*\d/i;
const isThreshold = (q: string) => (THRESHOLD_UP.test(q) || THRESHOLD_DOWN.test(q)) && !BAND.test(q);

/**
 * Mean of the numbers in a question, years left out ("between 100-150k" -> 125, "the March 2026 rate be 4.0%" -> 4);
 * "lose jobs" (spelled "loose" on Polymarket) is the band below zero.
 */
function questionValue(q: string): number {
  if (/\blo+se jobs\b/i.test(q)) return 0;
  const nums = [...q.matchAll(/\d+(?:\.\d+)?/g)].map(m => Number(m[0])).filter(n => !(Number.isInteger(n) && n >= 1900 && n <= 2100));
  return nums.reduce((s, n) => s + n, 0) / nums.length;
}

function fedWeight(q: string): number {
  if (/\bno (?:change|increase|decrease|cut|hike)\b/i.test(q)) return 0;
  const dir = /decrease|\bcut|lower/i.test(q) ? -1 : /increase|raise|hike/i.test(q) ? 1 : 0;
  const bp = /(\d+)\+?\s*(?:bps?|basis points?)\b/i.exec(q);
  return dir * (bp ? Number(bp[1]) / 25 : 1) || 0;
}

/**
 * Orientation of each market of one event, from the question text alone: +1 when YES means a higher policy rate (in
 * 25 bp steps for the Fed), hotter inflation, a stronger labour market or a crypto ETF approval. A threshold set takes
 * the sign of each comparison; a band set signs each band by its value against the event's mean value. The regression
 * learns what each direction means for each asset.
 */
export function orientationWeights(group: EventGroup, questions: readonly string[]): number[] {
  if (group === 'crypto_etf') return questions.map(q => (/delay|reject|den(?:y|ies|ied)|\bnot\b/i.test(q) ? -1 : 1));
  if (group === 'fed') return questions.map(fedWeight);
  const labour = (q: string) => (group === 'jobs' && /unemployment/i.test(q) ? -1 : 1);
  if (questions.every(isThreshold)) return questions.map(q => (THRESHOLD_UP.test(q) ? 1 : -1) * labour(q));
  const values = questions.map(questionValue);
  const mid = values.reduce((s, v) => s + v, 0) / values.length;
  return values.map((v, i) => (Math.abs(v - mid) < 1e-9 ? 0 : Math.sign(v - mid) * labour(questions[i])) || 0);
}

export interface EventMarketHistory {
  group: EventGroup;
  event: string;
  question: string;
  weight: number;
  createdAt: string;
  closedAt: string;
  /** UTC date -> last YES price at or before CUTOFF_HOUR_UTC that day (see sampleAtCutoff). */
  daily: Record<string, number>;
}

/**
 * Daily sampling time. At or before every asset's close (US equities 20:00Z in summer, 21:00Z in winter; BTC-USD 24:00Z)
 * with an hour to spare, so the feature window ends before the next-day return window starts.
 */
export const CUTOFF_HOUR_UTC = 19;

const dayKey = (ms: number) => new Date(ms).toISOString().slice(0, 10);

/**
 * One YES price per UTC day at the cutoff, from the first trade after creation to the day after the last trade before
 * close. Prices after close are dropped: the CLOB history continues at 0.5 once the book is cleared.
 */
export function sampleAtCutoff(points: ReadonlyArray<{ t: number; p: number }>, createdAt: string, closedAt: string): Record<string, number> {
  const from = Date.parse(createdAt) / 1000;
  const to = Date.parse(closedAt) / 1000;
  const live = points.filter(x => x.t >= from && x.t <= to).sort((a, b) => a.t - b.t);
  if (live.length === 0) return {};
  const daily: Record<string, number> = {};
  const lastDay = dayKey(live[live.length - 1].t * 1000 + MS_PER_DAY);
  let i = 0;
  let price: number | undefined;
  for (let day = dayKey(live[0].t * 1000); day <= lastDay; day = dayKey(Date.parse(`${day}T00:00:00Z`) + MS_PER_DAY)) {
    const cutoff = Date.parse(`${day}T00:00:00Z`) / 1000 + CUTOFF_HOUR_UTC * 3600;
    while (i < live.length && live[i].t <= cutoff) price = live[i++].p;
    if (price !== undefined) daily[day] = price;
  }
  return daily;
}

export interface EventFeatures {
  /** Per EVENT_GROUPS entry: the largest |Δp| among the group's markets. */
  moves: number[];
  /** Per EVENT_GROUPS entry: the sum of orientation × Δp. */
  signed: number[];
  /** Some market was priced at both cutoffs. */
  active: boolean;
}

/** Event moves between the cutoffs of two dates, over markets priced at both (a market counts from its first trade). */
export function eventFeaturesBetween(markets: readonly EventMarketHistory[], prev: string, date: string): EventFeatures {
  const moves = EVENT_GROUPS.map(() => 0);
  const signed = EVENT_GROUPS.map(() => 0);
  let active = false;
  for (const m of markets) {
    const a = m.daily[prev];
    const b = m.daily[date];
    if (a === undefined || b === undefined) continue;
    const g = EVENT_GROUPS.indexOf(m.group);
    active = true;
    moves[g] = Math.max(moves[g], Math.abs(b - a));
    signed[g] += m.weight * (b - a);
  }
  return { moves, signed, active };
}

export interface H4Row extends EventFeatures {
  ticker: string;
  /** Feature date D_i: event moves over (D_{i-1}, D_i] cutoffs, trailing volatility through the D_i close. */
  date: string;
  /** D_{i+1}: the target is the log return from the D_i close to this close. */
  targetDate: string;
  absReturn: number;
  up: number;
  /** [1, ln|r_i|, ln rms(r_{i-4..i}), ln rms(r_{i-21..i})]: HAR-style trailing volatility. */
  base: number[];
}

const LOG_FLOOR = 1e-4;

/** One row per trading date from `from` that has 22 past returns and a next close. */
export function buildRows(ticker: string, dates: readonly string[], closes: readonly number[], markets: readonly EventMarketHistory[], from: string): H4Row[] {
  const lr = closes.map((c, i) => (i === 0 ? 0 : Math.log(c / closes[i - 1])));
  const lnRms = (a: number, b: number) => {
    let s = 0;
    for (let k = a; k <= b; k++) s += lr[k] * lr[k];
    return Math.log(Math.max(Math.sqrt(s / (b - a + 1)), LOG_FLOOR));
  };
  const rows: H4Row[] = [];
  for (let i = 22; i + 1 < dates.length; i++) {
    if (dates[i] < from) continue;
    const r = lr[i + 1];
    rows.push({
      ticker,
      date: dates[i],
      targetDate: dates[i + 1],
      absReturn: Math.abs(r),
      up: r > 0 ? 1 : 0,
      base: [1, Math.log(Math.max(Math.abs(lr[i]), LOG_FLOOR)), lnRms(i - 4, i), lnRms(i - 21, i)],
      ...eventFeaturesBetween(markets, dates[i - 1], dates[i]),
    });
  }
  return rows;
}

/** Laplace: r ~ Laplace(0, s), ln s = xβ (an exponential model of |r|). Logistic: P(up) = σ(xβ). */
export type GlmFamily = 'laplace' | 'logistic';

/** Row log-likelihood, its derivative in the linear predictor, and minus its second derivative. */
function rowTerms(family: GlmFamily, eta: number, y: number): [number, number, number] {
  if (family === 'laplace') {
    const w = y * Math.exp(-eta);
    return [-Math.LN2 - eta - w, w - 1, w];
  }
  const p = 1 / (1 + Math.exp(-eta));
  const softplus = eta > 0 ? eta + Math.log1p(Math.exp(-eta)) : Math.log1p(Math.exp(eta));
  return [y * eta - softplus, y - p, p * (1 - p)];
}

const dot = (x: readonly number[], b: readonly number[]) => x.reduce((s, v, k) => s + v * b[k], 0);

/** Solves A z = b by Gaussian elimination with partial pivoting (A is copied). */
function solve(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / M[c][c];
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const z = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) z[r] = (M[r][n] - dot(M[r].slice(r + 1, n), z.slice(r + 1))) / M[r][r];
  return z;
}

/** Maximum likelihood by Newton's method with step halving (both log-likelihoods are concave). */
export function fitGlm(X: readonly number[][], y: readonly number[], family: GlmFamily): { beta: number[]; logLik: (x: readonly number[], y: number) => number } {
  const k = X[0].length;
  const total = (b: number[]) => X.reduce((s, x, i) => s + rowTerms(family, dot(x, b), y[i])[0], 0);
  let beta = new Array<number>(k).fill(0);
  let current = total(beta);
  for (let iter = 0; iter < 200; iter++) {
    const grad = new Array<number>(k).fill(0);
    const H = Array.from({ length: k }, () => new Array<number>(k).fill(0));
    X.forEach((x, i) => {
      const [, g, h] = rowTerms(family, dot(x, beta), y[i]);
      for (let a = 0; a < k; a++) {
        grad[a] += g * x[a];
        for (let c = 0; c < k; c++) H[a][c] += h * x[a] * x[c];
      }
    });
    // A tiny ridge keeps the step defined when a column is all zeros; the optimum (gradient 0) is unchanged.
    for (let a = 0; a < k; a++) H[a][a] += 1e-9;
    const step = solve(H, grad);
    let t = 1;
    let candidate = beta;
    let next = Number.NaN;
    for (let half = 0; half < 60; half++) {
      candidate = beta.map((b, a) => b + t * step[a]);
      next = total(candidate);
      if (next >= current) break;
      t /= 2;
    }
    if (!(next >= current)) break;
    const gain = next - current;
    beta = candidate;
    current = next;
    if (gain < 1e-10) break;
  }
  return { beta, logLik: (x, yi) => rowTerms(family, dot(x, beta), yi)[0] };
}

/**
 * Percentile 95% CI of the row mean, resampling circular blocks of `block` consecutive dates; all rows of a date move
 * together, so same-day correlation across assets stays in the resample.
 */
export function blockBootstrapMeanCi(rows: ReadonlyArray<{ date: string; value: number }>, block: number, reps: number, seed: number): [number, number] {
  const sums = new Map<string, { sum: number; n: number }>();
  for (const r of rows) {
    const s = sums.get(r.date) ?? { sum: 0, n: 0 };
    s.sum += r.value;
    s.n++;
    sums.set(r.date, s);
  }
  const days = [...sums.keys()].sort().map(d => sums.get(d)!);
  const rand = mulberry32(seed);
  const means: number[] = [];
  for (let rep = 0; rep < reps; rep++) {
    let sum = 0;
    let n = 0;
    for (let taken = 0; taken < days.length;) {
      const start = Math.floor(rand() * days.length);
      for (let k = 0; k < block && taken < days.length; k++, taken++) {
        const d = days[(start + k) % days.length];
        sum += d.sum;
        n += d.n;
      }
    }
    means.push(sum / n);
  }
  means.sort((a, b) => a - b);
  return [means[Math.floor(0.025 * reps)], means[Math.ceil(0.975 * reps) - 1]];
}

/** P(χ²_df > x): exact for 1 df, Wilson-Hilferty otherwise. */
export function chiSquareSf(x: number, df: number): number {
  if (df === 1) return 2 * (1 - normalCDF(Math.sqrt(Math.max(x, 0))));
  const v = 2 / (9 * df);
  return 1 - normalCDF((Math.cbrt(Math.max(x, 0) / df) - (1 - v)) / Math.sqrt(v));
}
