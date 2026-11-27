import { describe, expect, it } from 'bun:test';
import { normCDF, normPPF } from '../../../utils/stats.js';
import { inverseStudentTCDF } from '../markov-distribution/confidence-intervals.js';
import {
  fitLadderSigma, fitTailDof, ladderObservations, parseStrike, scoreStudentT, studentTCrps, type LadderObservation,
} from './ladder-implied-vol-benchmark.js';

/** Unit-variance Student-t quantile (nu = Infinity is the normal). */
const tQuantile = (p: number, nu: number) =>
  Number.isFinite(nu) ? inverseStudentTCDF(p, nu) * Math.sqrt((nu - 2) / nu) : normPPF(p);

/** n draws spread evenly over the distribution's quantiles, shuffled so every stretch of days sees the whole range. */
const spread = (n: number, nu: number) => Array.from({ length: n }, (_, i) => tQuantile((((i * 7919) % n) + 0.5) / n, nu));

describe('ladder-implied volatility benchmark', () => {
  it('reads the strike and side of a threshold question', () => {
    expect(parseStrike('Will the price of Bitcoin be above $78,000 on January 1?')).toEqual({ strike: 78000, below: false });
    expect(parseStrike('Will the price of Ethereum be below $2,450.50 on May 3?')).toEqual({ strike: 2450.5, below: true });
    expect(parseStrike('Will Bitcoin reach $150k in December?')).toBeNull();
  });

  it('recovers sigma from a lognormal ladder and skips near-certain strikes', () => {
    const spot = 100;
    const sigma = 0.05;
    const points = [80, 95, 100, 105, 130].map(k => ({ strike: k, pAbove: 1 - normCDF(Math.log(k / spot) / sigma) }));
    expect(fitLadderSigma(points, spot)).toBeCloseTo(sigma, 6);
    expect(fitLadderSigma(points.slice(0, 2).concat(points.slice(4)), spot)).toBeNull(); // only 95 is informative
  });

  it('rejects an unquoted or non-monotone ladder', () => {
    const flat = [95, 100, 105].map(k => ({ strike: k, pAbove: 0.5 }));
    const rising = [{ strike: 95, pAbove: 0.6 }, { strike: 100, pAbove: 0.7 }, { strike: 105, pAbove: 0.2 }];
    expect(fitLadderSigma(flat, 100)).toBeNull();
    expect(fitLadderSigma(rising, 100)).toBeNull();
  });

  it('builds one observation per open expiry and day, from the day close to the expiry close', () => {
    const closes = { dates: [] as string[], closes: [] as number[] };
    for (let i = 0; i < 40; i++) {
      closes.dates.push(new Date(Date.UTC(2026, 0, 1 + i)).toISOString().slice(0, 10));
      closes.closes.push(100 * Math.exp(0.01 * Math.sin(i)));
    }
    const market = (k: number, p: number) => ({
      question: `Will the price of Bitcoin be above $${k} on February 5?`, createdAt: '2026-02-01T17:00:00Z',
      endDate: '2026-02-05T17:00:00Z', volume: 1000, daily: { '2026-02-03': p, '2026-02-05': 0.5 },
    });
    const rows = ladderObservations([market(95, 0.9), market(100, 0.5), market(105, 0.1)], closes);

    expect(rows).toHaveLength(1); // 2026-02-05 closes after resolution
    expect(rows[0]).toMatchObject({ day: '2026-02-03', expiry: '2026-02-05', h: 2 });
    expect(rows[0]!.realized).toBeCloseTo(Math.log(closes.closes[35]! / closes.closes[33]!), 12);
  });

  it('keeps ladders with different end times apart and skips one that resolves within hours of the sample', () => {
    const closes = { dates: [] as string[], closes: [] as number[] };
    for (let i = 0; i < 40; i++) {
      closes.dates.push(new Date(Date.UTC(2026, 0, 1 + i)).toISOString().slice(0, 10));
      closes.closes.push(100 * Math.exp(0.01 * Math.sin(i)));
    }
    const ladder = (endDate: string, ps: number[]) => [95, 100, 105].map((k, i) => ({
      question: `Will the price of Bitcoin be above $${k} on February 5?`, createdAt: '2026-02-01T17:00:00Z',
      endDate, volume: 1000, daily: { '2026-02-03': ps[i]! },
    }));
    const rows = ladderObservations([
      ...ladder('2026-02-05T17:00:00Z', [0.9, 0.5, 0.1]),
      ...ladder('2026-02-05T04:00:00Z', [0.95, 0.5, 0.05]),
      ...ladder('2026-02-04T00:00:00Z', [0.9, 0.5, 0.1]), // one second after the 2026-02-03 sample
    ], closes);

    expect(rows.map(r => r.expiry)).toEqual(['2026-02-05', '2026-02-05']);
    expect(rows[0]!.sigmaLadder).not.toBeCloseTo(rows[1]!.sigmaLadder, 3);
  });

  it('fits heavy tails to Student-t data and none to normal data', () => {
    const rows = (xs: number[]) => xs.map(x => ({ realized: 0.02 * x, sigmaLadder: 0.02 }) as LadderObservation);
    const nuT = fitTailDof(rows(spread(2000, 4)), r => r.sigmaLadder);
    expect(nuT).toBeGreaterThanOrEqual(3);
    expect(nuT).toBeLessThanOrEqual(5);
    expect(fitTailDof(rows(spread(2000, Infinity)), r => r.sigmaLadder)).toBeGreaterThanOrEqual(30);
  });

  it('matches the normal CRPS as the degrees of freedom grow', () => {
    const z = 0.7;
    const normal = z * (2 * normCDF(z) - 1) + 2 * Math.exp(-0.5 * z * z) / Math.sqrt(2 * Math.PI) - 1 / Math.sqrt(Math.PI);
    expect(studentTCrps(0.014, 0.02, Infinity)).toBeCloseTo(0.02 * normal, 10);
    expect(studentTCrps(0.014, 0.02, 1e5)).toBeCloseTo(0.02 * normal, 5);
  });

  it('fits the tail on the first half of the days and scores the interval on the second', () => {
    const draws = spread(4000, 4);
    const rows: LadderObservation[] = draws.map((x, i) => ({
      day: new Date(Date.UTC(2026, 0, 1 + Math.floor(i / 10))).toISOString().slice(0, 10), expiry: 'e', h: 1 + (i % 7),
      sigmaLadder: 0.02, sigmaTrailing: 0.03, realized: 0.02 * x,
    }));
    const s = scoreStudentT(rows);

    expect(s.nuLadder).toBeGreaterThanOrEqual(3);
    expect(s.nuLadder).toBeLessThanOrEqual(5);
    expect(s.test.days).toBe(200);
    expect(s.test.logScore.ci[1]).toBeLessThan(0);
    s.test.coverage.ladder.forEach((c, i) => expect(Math.abs(c - [0.9, 0.95, 0.99][i]!)).toBeLessThanOrEqual(0.01));
    expect(s.byHorizon.map(b => b.name)).toEqual(['1d', '2-3d', '4-7d']);
  });
});
