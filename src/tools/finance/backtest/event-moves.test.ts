import { describe, expect, it } from 'bun:test';
import { mulberry32 } from './metrics.js';
import {
  blockBootstrapMeanCi,
  buildRows,
  chiSquareSf,
  classifyEventTitle,
  eventFeaturesBetween,
  fitGlm,
  orientationWeights,
  sampleAtCutoff,
  type EventMarketHistory,
} from './event-moves.js';

const ts = (iso: string) => Date.parse(iso) / 1000;

describe('classifyEventTitle (the pre-registered event list)', () => {
  it('keeps the scheduled Fed, CPI, jobs and crypto ETF families', () => {
    expect(classifyEventTitle('Fed Interest Rates March 2023')).toBe('fed');
    expect(classifyEventTitle('Fed Interest Rates: November 2024')).toBe('fed');
    expect(classifyEventTitle('Fed decision in December?')).toBe('fed');
    expect(classifyEventTitle('What will U.S. inflation be from January to February 2023?')).toBe('cpi');
    expect(classifyEventTitle('What will US inflation be from December 2022 to January 2023?')).toBe('cpi');
    expect(classifyEventTitle('US inflation >0.2% from Nov to Dec 2023?')).toBe('cpi');
    expect(classifyEventTitle('Annual inflation under 3% in August?')).toBe('cpi');
    expect(classifyEventTitle('November Inflation')).toBe('cpi');
    expect(classifyEventTitle('March Inflation - Monthly')).toBe('cpi');
    expect(classifyEventTitle('How many Jobs added in February?')).toBe('jobs');
    expect(classifyEventTitle('Unemployment rate over 4.2% in August?')).toBe('jobs');
    expect(classifyEventTitle('March Unemployment Rate')).toBe('jobs');
    expect(classifyEventTitle('Bitcoin ETF approved by Jan 15?')).toBe('crypto_etf');
    expect(classifyEventTitle('When will Bitcoin ETF be approved?')).toBe('crypto_etf');
    expect(classifyEventTitle('Will SEC delay Bitcoin ETF decision?')).toBe('crypto_etf');
    expect(classifyEventTitle('Ethereum ETF begins trading by July 26?')).toBe('crypto_etf');
    expect(classifyEventTitle('Cardano ETF in 2025?')).toBe('crypto_etf');
  });

  it('drops mention markets, other countries, flows, prices and unscheduled questions', () => {
    for (const title of [
      'What will Powell say during March Press Conference?',
      'September Inflation Argentina - Monthly',
      'October Unemployment Rate - Eurozone',
      'ECB Interest Rates: June 2025',
      'How many Fed rate cuts this year?',
      'Fed emergency rate cut in August?',
      'Bitcoin ETF Flows on October 17?',
      'Bitcoin price 1hr after ETF approval?',
      'Bitcoin ETFs first week combined AUM',
      'Parlay: US Bitcoin Spot and ETH Futures ETF this year?',
      'Price of dozen eggs in March?',
    ]) expect(classifyEventTitle(title)).toBeUndefined();
  });
});

describe('orientationWeights (+1 = YES means a higher rate, hotter inflation, a stronger labour market, an approval)', () => {
  it('weights Fed outcomes by the rate change in 25 bp steps', () => {
    expect(orientationWeights('fed', [
      'Fed decreases interest rates by 75+ bps after November 2024 meeting?',
      'Fed decreases interest rates by 50 bps after November 2024 meeting?',
      'Fed decreases interest rates by 25 bps after November 2024 meeting?',
      'No change in Fed interest rates after 2024 November meeting?',
      'Fed increases interest rates by 25+ bps after November 2024 meeting?',
      'Will the FED change rates to another level after Nov meeting?',
    ])).toEqual([-3, -2, -1, 0, 1, 0]);
    expect(orientationWeights('fed', [
      'Will the Fed raise interest rates by 0 bps after its December meeting?',
      'Will the Fed raise interest rates by 25 bps after its December meeting?',
      'No increase after March meeting?',
    ])).toEqual([0, 1, 0]);
  });

  it('gives every market of a threshold set the direction of its comparison', () => {
    expect(orientationWeights('cpi', [
      'Will US inflation be greater than 0.4% from January to February 2023?',
      'Will U.S. inflation be greater than 0.3% from January to February 2023?',
    ])).toEqual([1, 1]);
    expect(orientationWeights('cpi', ['US inflation >0.2% from Nov to Dec 2023?'])).toEqual([1]);
    expect(orientationWeights('cpi', ['Annual inflation under 3% in August?'])).toEqual([-1]);
  });

  it('signs band markets by their value against the middle of the event', () => {
    expect(orientationWeights('cpi', [
      'Will monthly inflation increase by 0.1% or less in March?',
      'Will monthly inflation increase by 0.2% in March?',
      'Will monthly inflation increase by 0.3% in March?',
      'Will monthly inflation increase by 0.4% in March?',
      'Will monthly inflation increase by 0.5% or more in March?',
    ])).toEqual([-1, -1, 0, 1, 1]);
    expect(orientationWeights('jobs', [
      'Will the US add less than 50k jobs in March?',
      'Will the US add between 100-150k jobs in March?',
      'Will the US add between 150-200k jobs in March?',
      'Will the US add between 200-250k jobs in March?',
      'Will the US add between 50-100k jobs in March?',
      'Will the US add more than 250k jobs in March?',
    ])).toEqual([-1, -1, 1, 1, -1, 1]);
  });

  it('reverses unemployment, where a higher value is a weaker labour market', () => {
    expect(orientationWeights('jobs', [
      'Will the March 2026 unemployment rate be ≤3.9%?',
      'Will the March 2026 unemployment rate be 4.0%?',
      'Will the March 2026 unemployment rate be 4.1%?',
    ])).toEqual([1, 0, -1]);
    expect(orientationWeights('jobs', ['Unemployment rate over 4.2% in August?'])).toEqual([-1]);
  });

  it('counts an ETF approval as +1 and a delay as -1', () => {
    expect(orientationWeights('crypto_etf', ['Bitcoin ETF approved by Jan 15?'])).toEqual([1]);
    expect(orientationWeights('crypto_etf', ['Will SEC delay Bitcoin ETF decision?'])).toEqual([-1]);
  });
});

describe('sampleAtCutoff', () => {
  const points = [
    { t: ts('2025-03-01T10:00:00Z'), p: 0.1 }, // before creation: ignored
    { t: ts('2025-03-02T18:00:00Z'), p: 0.3 },
    { t: ts('2025-03-02T20:00:00Z'), p: 0.4 }, // after the 19:00Z cutoff: counts for the next day
    { t: ts('2025-03-05T12:00:00Z'), p: 0.9 },
    { t: ts('2025-03-06T12:00:00Z'), p: 0.5 }, // after close (cleared book): ignored
  ];

  it('takes the last price at or before 19:00Z, from the first trade to the day after the last one before close', () => {
    expect(sampleAtCutoff(points, '2025-03-02T00:00:00Z', '2025-03-05T20:00:00Z')).toEqual({
      '2025-03-02': 0.3,
      '2025-03-03': 0.4,
      '2025-03-04': 0.4,
      '2025-03-05': 0.9,
      '2025-03-06': 0.9,
    });
  });

  it('is empty without a price inside the market life', () => {
    expect(sampleAtCutoff(points, '2025-04-01T00:00:00Z', '2025-04-02T00:00:00Z')).toEqual({});
  });
});

const market = (over: Partial<EventMarketHistory>): EventMarketHistory => ({
  group: 'fed', event: 'e', question: 'q', weight: 1, createdAt: '2025-01-01T00:00:00Z', closedAt: '2025-02-01T00:00:00Z', daily: {}, ...over,
});

describe('eventFeaturesBetween', () => {
  it('takes the largest absolute move and the oriented sum per group over markets priced on both dates', () => {
    const markets = [
      market({ group: 'fed', weight: -1, daily: { '2025-01-02': 0.6, '2025-01-03': 0.3 } }),
      market({ group: 'fed', weight: 0, daily: { '2025-01-02': 0.4, '2025-01-03': 0.7 } }),
      market({ group: 'fed', weight: 1, daily: { '2025-01-02': 0.0, '2025-01-03': 0.1 } }),
      market({ group: 'cpi', weight: 1, daily: { '2025-01-03': 0.5 } }), // not priced the day before: no move
    ];
    const f = eventFeaturesBetween(markets, '2025-01-02', '2025-01-03');
    expect(f.moves[0]).toBeCloseTo(0.3);
    expect(f.signed[0]).toBeCloseTo(0.3 + 0.1);
    expect(f.moves[1]).toBe(0);
    expect(f.signed[1]).toBe(0);
    expect(f.active).toBe(true);
    expect(eventFeaturesBetween(markets.slice(3), '2025-01-02', '2025-01-03').active).toBe(false);
  });
});

describe('buildRows', () => {
  it('pairs the event move up to the feature date with the return to the next trading date', () => {
    const dates = Array.from({ length: 30 }, (_, i) => new Date(Date.UTC(2024, 11, 1 + i)).toISOString().slice(0, 10));
    const closes = dates.map((_, i) => 100 * Math.exp(0.01 * (i % 2 === 0 ? 1 : -1) * (i + 1) / 10));
    const markets = [market({ group: 'cpi', weight: 1, daily: { [dates[24]]: 0.2, [dates[25]]: 0.5 } })];

    const rows = buildRows('SPY', dates, closes, markets, dates[23]);

    expect(rows.map(r => r.date)).toEqual(dates.slice(23, 29));
    const r = rows.find(x => x.date === dates[25])!;
    expect(r.targetDate).toBe(dates[26]);
    expect(r.absReturn).toBeCloseTo(Math.abs(Math.log(closes[26] / closes[25])));
    expect(r.up).toBe(closes[26] > closes[25] ? 1 : 0);
    expect(r.moves[1]).toBeCloseTo(0.3);
    expect(rows.find(x => x.date === dates[24])!.moves[1]).toBe(0); // the market had no price on dates[23]
    const lr = dates.map((_, i) => (i === 0 ? 0 : Math.log(closes[i] / closes[i - 1])));
    const rms = (a: number[]) => Math.sqrt(a.reduce((s, x) => s + x * x, 0) / a.length);
    const expected = [1, Math.log(Math.abs(lr[25])), Math.log(rms(lr.slice(21, 26))), Math.log(rms(lr.slice(4, 26)))];
    expected.forEach((v, k) => expect(r.base[k]).toBeCloseTo(v, 10));
  });
});

describe('fitGlm', () => {
  const rand = mulberry32(7);
  const normal = () => Math.sqrt(-2 * Math.log(rand() + 1e-12)) * Math.cos(2 * Math.PI * rand());
  const X = Array.from({ length: 4000 }, () => [1, normal(), normal()]);

  it('recovers the scale coefficients of a Laplace model of |return|', () => {
    const beta = [-4, 0.5, -0.3];
    const y = X.map(x => -Math.exp(x[0] * beta[0] + x[1] * beta[1] + x[2] * beta[2]) * Math.log(rand() + 1e-12));
    const fit = fitGlm(X, y, 'laplace');
    fit.beta.forEach((b, i) => expect(b).toBeCloseTo(beta[i], 1));
  });

  it('recovers logistic coefficients', () => {
    const beta = [0.2, 1, -0.5];
    const y = X.map(x => (rand() < 1 / (1 + Math.exp(-(x[0] * beta[0] + x[1] * beta[1] + x[2] * beta[2])))) ? 1 : 0);
    const fit = fitGlm(X, y, 'logistic');
    fit.beta.forEach((b, i) => expect(b).toBeCloseTo(beta[i], 0));
    expect(fit.logLik(X[0], y[0])).toBeLessThan(0);
  });
});

describe('blockBootstrapMeanCi', () => {
  it('is degenerate for a constant and reproducible for a seed', () => {
    const constant = Array.from({ length: 50 }, (_, i) => ({ date: `d${String(i).padStart(2, '0')}`, value: 0.2 }));
    const [lo, hi] = blockBootstrapMeanCi(constant, 5, 200, 1);
    expect(lo).toBeCloseTo(0.2);
    expect(hi).toBeCloseTo(0.2);
    const noisy = constant.map((r, i) => ({ ...r, value: Math.sin(i) }));
    expect(blockBootstrapMeanCi(noisy, 5, 200, 1)).toEqual(blockBootstrapMeanCi(noisy, 5, 200, 1));
    const mean = noisy.reduce((s, r) => s + r.value, 0) / noisy.length;
    const [a, b] = blockBootstrapMeanCi(noisy, 5, 500, 1);
    expect(a).toBeLessThan(mean);
    expect(b).toBeGreaterThan(mean);
  });
});

describe('chiSquareSf', () => {
  it('matches the 5% critical values', () => {
    expect(chiSquareSf(3.841, 1)).toBeCloseTo(0.05, 3);
    expect(chiSquareSf(26.296, 16)).toBeCloseTo(0.05, 2);
  });
});
