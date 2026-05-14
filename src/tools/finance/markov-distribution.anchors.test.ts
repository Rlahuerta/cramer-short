import { describe, it, expect } from 'bun:test';
import { estimateRegimeStats } from './markov-distribution/confidence-intervals.js';
import { buildDefaultMatrix } from './markov-distribution/transition.js';
import { applyCryptoTerminalAnchorFallback, assessAnchorCoverage, computeMarkovDistribution, evaluateAnchorTrust, extractPriceThresholds, filterMarketsToHorizon, interpolateDistribution, sortMarketsByHorizonCloseness } from './markov-distribution.js';
import type { PriceThreshold } from './markov-distribution.js';
import { MS_PER_DAY } from '../../utils/time.js';

const FIXED_NOW_MS = Date.parse('2025-04-02T12:00:00.000Z');
const RECENT_CREATED_AT_MS = FIXED_NOW_MS - 10 * 60 * 60 * 1000;
const STALE_CREATED_AT_MS = FIXED_NOW_MS - 72 * 3600_000;
const WEEK_OLD_CREATED_AT_MS = FIXED_NOW_MS - 7 * MS_PER_DAY;

function allClose(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) < tol;
}

describe('extractPriceThresholds', () => {
  it('parses "exceed $900" pattern', () => {
    const result = extractPriceThresholds([
      { question: 'Will NVDA exceed $900?', probability: 0.6 },
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].price).toBe(900);
  });

  it('Fix 4: applies YES-bias correction (×0.95) to raw probability', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC exceed $70000?', probability: 0.8 },
    ]);
    expect(result[0].rawProbability).toBe(0.8);
    expect(allClose(result[0].probability, 0.76, 1e-10)).toBe(true); // 0.8 × 0.95
  });

  it('parses K suffix correctly ($70K → 70000)', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC close above $70K on Friday?', probability: 0.5 },
    ]);
    expect(result[0].price).toBe(70_000);
  });

  it('parses M suffix correctly ($1M → 1000000)', () => {
    const result = extractPriceThresholds([
      { question: 'Will BRK.A exceed $1M?', probability: 0.3 },
    ]);
    expect(result[0].price).toBe(1_000_000);
  });

  it('Fix 8: non-crypto market under 48h with volume → trustScore high (youth does not block non-crypto)', () => {
    const result = extractPriceThresholds([
      { question: 'Will AAPL exceed $200?', probability: 0.7, createdAt: RECENT_CREATED_AT_MS, volume: 1000 },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result[0].trustScore).toBe('high');
  });

  it('Fix 8: market >48h with volume → trustScore high', () => {
    const result = extractPriceThresholds([
      { question: 'Will AAPL exceed $200?', probability: 0.7, createdAt: WEEK_OLD_CREATED_AT_MS, volume: 5000 },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result[0].trustScore).toBe('high');
  });

  it('Fix 8: zero volume → trustScore low', () => {
    const result = extractPriceThresholds([
      { question: 'Will TSLA exceed $300?', probability: 0.5, volume: 0 },
    ]);
    expect(result[0].trustScore).toBe('low');
  });

  it('evaluates anchor trust with an explicit decision table for crypto resolution rules', () => {
    expect(evaluateAnchorTrust({
      hasVolume: true,
      isYoung: false,
      isShortHorizonCrypto: false,
      isLongHorizonCrypto: false,
      isNearTargetResolution: false,
    })).toEqual({
      trustScore: 'high',
      trustWeight: 1,
      lowTrustReasons: [],
    });

    expect(evaluateAnchorTrust({
      hasVolume: true,
      isYoung: true,
      isShortHorizonCrypto: true,
      isLongHorizonCrypto: false,
      isNearTargetResolution: true,
    })).toEqual({
      trustScore: 'high',
      trustWeight: 0.7,
      lowTrustReasons: ['young_market'],
    });

    expect(evaluateAnchorTrust({
      hasVolume: true,
      isYoung: true,
      isShortHorizonCrypto: true,
      isLongHorizonCrypto: false,
      isNearTargetResolution: false,
    })).toEqual({
      trustScore: 'low',
      trustWeight: 0.35,
      lowTrustReasons: ['young_market', 'resolution_mismatch'],
    });

    expect(evaluateAnchorTrust({
      hasVolume: true,
      isYoung: false,
      isShortHorizonCrypto: false,
      isLongHorizonCrypto: true,
      isNearTargetResolution: true,
    })).toEqual({
      trustScore: 'high',
      trustWeight: 0.9,
      lowTrustReasons: [],
    });

    expect(evaluateAnchorTrust({
      hasVolume: true,
      isYoung: false,
      isShortHorizonCrypto: false,
      isLongHorizonCrypto: true,
      isNearTargetResolution: false,
    })).toEqual({
      trustScore: 'low',
      trustWeight: 0.35,
      lowTrustReasons: ['resolution_mismatch'],
    });
  });


  // Phase 1: Non-crypto anchor trust — youth should not block trust for commodities
  it('Phase 1: non-crypto commodity anchor with volume → trustScore high regardless of age', () => {
    expect(evaluateAnchorTrust({
      hasVolume: true,
      isYoung: true,           // <48h old — should NOT matter for non-crypto
      isShortHorizonCrypto: false,
      isLongHorizonCrypto: false,
      isNearTargetResolution: false,
    })).toEqual({
      trustScore: 'high',
      trustWeight: 0.75,
      lowTrustReasons: ['young_market'],  // still reported, does not block trust
    });
  });

  it('Phase 1: non-crypto commodity anchor with zero volume → trustScore low', () => {
    expect(evaluateAnchorTrust({
      hasVolume: false,
      isYoung: false,
      isShortHorizonCrypto: false,
      isLongHorizonCrypto: false,
      isNearTargetResolution: false,
    })).toEqual({
      trustScore: 'low',
      trustWeight: 0,
      lowTrustReasons: ['missing_volume'],
    });
  });

  it('Phase 1: long-horizon crypto young + no resolution match → still low trust', () => {
    expect(evaluateAnchorTrust({
      hasVolume: true,
      isYoung: true,
      isShortHorizonCrypto: false,
      isLongHorizonCrypto: true,
      isNearTargetResolution: false,
    })).toEqual({
      trustScore: 'low',
      trustWeight: 0.2,
      lowTrustReasons: ['young_market', 'resolution_mismatch'],
    });
  });
  it('skips markets with no price in question', () => {
    const result = extractPriceThresholds([
      { question: 'Will NVDA split its stock?', probability: 0.4 },
    ]);
    expect(result).toHaveLength(0);
  });

  it('deduplicates same price level, keeping highest probability', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC exceed $70000?', probability: 0.6 },
      { question: 'Will BTC close above $70000 at expiry?', probability: 0.7 },
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].rawProbability).toBe(0.7);
  });

  it('prefers the nearest-expiry GLD anchor for the same threshold on 1d/2d/3d horizons', () => {
    const now = new Date('2026-05-06T12:00:00Z').getTime();
    const day = MS_PER_DAY;

    for (const horizon of [1, 2, 3]) {
      const nearEndDate = new Date(now + horizon * day).toISOString();
      const farEndDate = new Date(now + (horizon + 7) * day).toISOString();
      const result = extractPriceThresholds([
        {
          question: 'Will gold exceed $4,600 by end of day?',
          probability: 0.62,
          volume: 10_000,
          createdAt: now - 7 * day,
          endDate: nearEndDate,
        },
        {
          question: 'Will gold exceed $4,600 next week?',
          probability: 0.71,
          volume: 10_000,
          createdAt: now - 7 * day,
          endDate: farEndDate,
        },
      ], { ticker: 'GLD', horizonDays: horizon, referenceTimeMs: now });

      expect(result).toHaveLength(1);
      expect(result[0].rawProbability).toBeCloseTo(0.62, 10);
      expect(result[0].endDate).toBe(nearEndDate);
    }
  });

  it('rejects barrier-style "reach" markets', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC reach $70000 this week?', probability: 0.5, volume: 1000 },
    ]);
    expect(result).toHaveLength(0);
  });

  it('rejects barrier-style "hit" markets', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC hit $70000 this week?', probability: 0.5, volume: 1000 },
    ]);
    expect(result).toHaveLength(0);
  });

  it('rejects barrier-style "dip to" markets', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC dip to $64000 this week?', probability: 0.2, volume: 1000 },
    ]);
    expect(result).toHaveLength(0);
  });

  it('rejects barrier-style "go past" markets', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC go past $100000 this year?', probability: 0.3, volume: 1000 },
    ]);
    expect(result).toHaveLength(0);
  });

  it('accepts terminal "at $X on date" markets', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC be at $70000 on April 5?', probability: 0.4, volume: 1000, createdAt: STALE_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result).toHaveLength(1);
    expect(result[0].price).toBe(70_000);
  });

  it('accepts date-anchored "trade above/below/over/under ... on/at <date>" markets', () => {
    const above = extractPriceThresholds([
      { question: 'Will BTC trade above $70000 on April 17?', probability: 0.4, volume: 1000, createdAt: STALE_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    const below = extractPriceThresholds([
      { question: 'Will BTC trade below $65000 at April 17?', probability: 0.3, volume: 1000, createdAt: STALE_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(above).toHaveLength(1);
    expect(above[0].price).toBe(70_000);
    expect(below).toHaveLength(1);
    expect(below[0].price).toBe(65_000);
  });

  it('accepts ISO date anchors (YYYY-MM-DD)', () => {
    const result = extractPriceThresholds([
      { question: 'Will WTI trade above $70 on 2025-05-01?', probability: 0.75, volume: 100_000, createdAt: STALE_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result).toHaveLength(1);
    expect(result[0].price).toBe(70);
  });

  it('accepts day-first formats with month name', () => {
    const result = extractPriceThresholds([
      { question: 'Will Brent trade below $65 on 1 May 2025?', probability: 0.60, volume: 100_000, createdAt: STALE_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result).toHaveLength(1);
    expect(result[0].price).toBe(65);
  });

  it('accepts ISO date anchors with slashes', () => {
    const result = extractPriceThresholds([
      { question: 'Will oil trade above $80 on 2025/06/15?', probability: 0.80, volume: 100_000, createdAt: STALE_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result).toHaveLength(1);
    expect(result[0].price).toBe(80);
  });

  it('rejects "trade above/below ... at expiry" (non-date anchor)', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC trade above $70000 at expiry?', probability: 0.4, volume: 1000, createdAt: STALE_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result).toHaveLength(0);
  });

  it('rejects "trade above/below ... at close" (non-date anchor)', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC trade above $70000 at close?', probability: 0.4, volume: 1000, createdAt: STALE_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result).toHaveLength(0);
  });

  it('rejects undated "trade above/below" markets (no date)', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC trade above $70000?', probability: 0.4, volume: 1000 },
    ]);
    expect(result).toHaveLength(0);
  });

  it('rejects path-style "stay above ... through <date>" markets', () => {
    const result = extractPriceThresholds([
      { question: 'Will Bitcoin stay above $70,000 through April 17?', probability: 0.3, volume: 1000 },
    ]);
    expect(result).toHaveLength(0);
  });

  it('sorts results by price ascending', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC exceed $80000?', probability: 0.4 },
      { question: 'Will BTC exceed $60000?', probability: 0.8 },
      { question: 'Will BTC exceed $70000?', probability: 0.6 },
    ]);
    expect(result.map(r => r.price)).toEqual([60_000, 70_000, 80_000]);
  });

  it('inverts probability for "fall below" markets to P(>price)', () => {
    // "Will gold fall below $4,200?" at P=0.31 means P(<4200) = 0.31*0.95
    // So P(>4200) = 1 - 0.31*0.95 = 0.7055
    const result = extractPriceThresholds([
      { question: 'Will gold fall below $4,200 by end of June?', probability: 0.31, volume: 1000, createdAt: WEEK_OLD_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result).toHaveLength(1);
    expect(result[0].price).toBe(4200);
    expect(result[0].probability).toBeCloseTo(1 - 0.31 * 0.95, 4); // inverted
    expect(result[0].probability).toBeGreaterThan(0.50); // must be P(above), which is high
  });

  it('inverts probability for "drop below" markets', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC drop below $50,000?', probability: 0.20, volume: 500, createdAt: STALE_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result).toHaveLength(1);
    expect(result[0].price).toBe(50_000);
    // P(<50K)=0.20*0.95=0.19, so P(>50K)=1-0.19=0.81
    expect(result[0].probability).toBeCloseTo(1 - 0.20 * 0.95, 4);
    expect(result[0].probability).toBeGreaterThan(0.70);
  });

  it('does NOT invert probability for "exceed" markets', () => {
    const result = extractPriceThresholds([
      { question: 'Will gold exceed $5,500?', probability: 0.40, volume: 1000, createdAt: STALE_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result).toHaveLength(1);
    expect(result[0].probability).toBeCloseTo(0.40 * 0.95, 4); // NOT inverted
    expect(result[0].probability).toBeLessThan(0.50);
  });

  it('handles mixed above/below markets for the same asset', () => {
    const result = extractPriceThresholds([
      { question: 'Will gold exceed $5,500 by June?', probability: 0.40, volume: 1000, createdAt: STALE_CREATED_AT_MS },
      { question: 'Will gold fall below $4,200 by June?', probability: 0.30, volume: 800, createdAt: STALE_CREATED_AT_MS },
    ], { referenceTimeMs: FIXED_NOW_MS });
    expect(result).toHaveLength(2);
    // $4,200 (below market): P(>4200) = 1 - 0.30*0.95 = 0.715
    const low = result.find(r => r.price === 4200)!;
    expect(low.probability).toBeCloseTo(1 - 0.30 * 0.95, 4);
    // $5,500 (above market): P(>5500) = 0.40*0.95 = 0.38
    const high = result.find(r => r.price === 5500)!;
    expect(high.probability).toBeCloseTo(0.40 * 0.95, 4);
    // CDF monotonicity: P(>4200) > P(>5500)
    expect(low.probability).toBeGreaterThan(high.probability);
  });
});
describe('applyCryptoTerminalAnchorFallback', () => {
  const REF_TIME = new Date('2025-04-15T12:00:00Z').getTime();
  const DAY_MS = MS_PER_DAY;

  it('returns strict anchors unchanged for non-crypto tickers', () => {
    const strictAnchors: PriceThreshold[] = [
      { price: 200, rawProbability: 0.6, probability: 0.57, trustScore: 'high', source: 'polymarket', endDate: null },
    ];
    const result = applyCryptoTerminalAnchorFallback(
      [], strictAnchors, 'SPY', 14, REF_TIME,
    );
    expect(result).toBe(strictAnchors);
  });

  it('returns strict anchors unchanged when crypto already has a usable high-trust anchor', () => {
    const strictAnchors: PriceThreshold[] = [
      { price: 85000, rawProbability: 0.5, probability: 0.475, trustScore: 'high', trustWeight: 0.9, source: 'polymarket', endDate: `2025-04-29` },
    ];
    const result = applyCryptoTerminalAnchorFallback(
      [], strictAnchors, 'BTC-USD', 14, REF_TIME,
    );
    expect(result).toBe(strictAnchors);
  });

  it('merges fallback anchors when strict crypto anchors exist but all remain low trust', () => {
    const strictAnchors: PriceThreshold[] = [
      {
        price: 84000,
        rawProbability: 0.45,
        probability: 0.4275,
        trustScore: 'low',
        trustWeight: 0.35,
        source: 'polymarket',
        endDate: '2025-04-19',
      },
    ];
    const markets = [
      {
        question: 'Will the price of Bitcoin be above $85000 on April 29?',
        probability: 0.48,
        volume: 30000,
        createdAt: REF_TIME - 5 * DAY_MS,
        endDate: '2025-04-29',
      },
    ];

    const result = applyCryptoTerminalAnchorFallback(
      markets, strictAnchors, 'BTC-USD', 14, REF_TIME,
    );

    expect(result).toHaveLength(2);
    expect(result.find((anchor) => anchor.price === 84000)?.trustScore).toBe('low');
    const recovered = result.find((anchor) => anchor.price === 85000);
    expect(recovered?.trustScore).toBe('high');
    expect(recovered?.trustWeight).toBe(0.9);
  });

  it('returns empty anchors unchanged when allMarkets is empty', () => {
    const result = applyCryptoTerminalAnchorFallback(
      [], [], 'BTC-USD', 14, REF_TIME,
    );
    expect(result).toHaveLength(0);
  });

  it('recovers terminal anchors from earlier-dated markets when strict set is empty for BTC', () => {
    const markets = [
      // Barrier-style (near-horizon) — rejected by extractPriceThresholds
      { question: 'Will Bitcoin reach $80,000 in April?', probability: 0.3, volume: 50000, endDate: '2025-04-29' },
      { question: 'Will Bitcoin dip to $65,000 in April?', probability: 0.15, volume: 40000, endDate: '2025-04-29' },
      // Terminal-style (earlier-dated) — parseable by extractPriceThresholds
      { question: 'Will the price of Bitcoin be above $84000 on April 17?', probability: 0.45, volume: 30000, createdAt: REF_TIME - 5 * DAY_MS, endDate: '2025-04-17' },
      { question: 'Will the price of Bitcoin be above $80000 on April 18?', probability: 0.65, volume: 25000, createdAt: REF_TIME - 3 * DAY_MS, endDate: '2025-04-18' },
    ];

    const result = applyCryptoTerminalAnchorFallback(
      markets, [], 'BTC-USD', 14, REF_TIME,
    );
    // Should recover the 2 terminal anchors from earlier-dated markets
    expect(result.length).toBeGreaterThanOrEqual(2);
    // Check that prices are in ascending order
    for (let i = 1; i < result.length; i++) {
      expect(result[i].price).toBeGreaterThanOrEqual(result[i - 1].price);
    }
  });

  it('applies date-gap discount to off-horizon anchors', () => {
    // Horizon=14, reference date April 15 → target resolution April 29
    // Anchor with endDate April 17 is 12 days before the target → 10 days off-horizon beyond tolerance
    const markets = [
      { question: 'Will the price of Bitcoin be above $84000 on April 17?', probability: 0.50, volume: 30000, createdAt: REF_TIME - 5 * DAY_MS, endDate: '2025-04-17' },
    ];

    const result = applyCryptoTerminalAnchorFallback(
      markets, [], 'BTC-USD', 14, REF_TIME,
    );
    expect(result).toHaveLength(1);

    // April 17 endDate → daysUntil = (Apr 17 - Apr 15) = 2 days
    // horizon=14 → |2 - 14| = 12 days offset
    // Beyond tolerance of 2 → discount = 1 - 0.03*(12-2) = 1 - 0.30 = 0.70
    // adjusted probability = 0.50 * 0.95 * 0.70 ≈ 0.3325
    expect(result[0].probability).toBeLessThan(0.5);
    expect(result[0].probability).toBeGreaterThan(0);
    expect(result[0].trustScore).toBe('low');
  });

  it('keeps a mature short-horizon crypto fallback trusted when it is only one day beyond tolerance', () => {
    const markets = [
      {
        question: 'Will the price of Bitcoin be above $84000 on April 20?',
        probability: 0.50,
        volume: 30000,
        createdAt: REF_TIME - 5 * DAY_MS,
        endDate: '2025-04-20',
      },
    ];

    const result = applyCryptoTerminalAnchorFallback(
      markets, [], 'BTC-USD', 2, REF_TIME,
    );

    expect(result).toHaveLength(1);
    expect(result[0].probability).toBeLessThan(0.50);
    expect(result[0].trustScore).toBe('high');
    expect(result[0].trustWeight).toBeGreaterThanOrEqual(0.6);
  });

  it('does not discount anchors within horizon tolerance', () => {
    // Anchor endDate April 29 → exactly at 14-day horizon
    const markets = [
      { question: 'Will the price of Bitcoin be above $84000 on April 29?', probability: 0.50, volume: 30000, createdAt: REF_TIME - 5 * DAY_MS, endDate: '2025-04-29' },
    ];

    const result = applyCryptoTerminalAnchorFallback(
      markets, [], 'BTC-USD', 14, REF_TIME,
    );
    expect(result).toHaveLength(1);
    // No discount — within ±2 day tolerance
    expect(result[0].trustScore).toBe('high');
  });

  it('keeps mature near-target BTC 30-day anchors trusted in direct extraction', () => {
    const result = extractPriceThresholds([
      {
        question: 'Will the price of Bitcoin be above $84000 on May 15?',
        probability: 0.50,
        volume: 30000,
        createdAt: REF_TIME - 5 * DAY_MS,
        endDate: '2025-05-15',
      },
    ], { ticker: 'BTC-USD', horizonDays: 30, referenceTimeMs: REF_TIME });

    expect(result).toHaveLength(1);
    expect(result[0].trustScore).toBe('high');
  });

  it('keeps off-window BTC 30-day anchors at low trust even when mature', () => {
    const result = extractPriceThresholds([
      {
        question: 'Will the price of Bitcoin be above $84000 on April 19?',
        probability: 0.50,
        volume: 30000,
        createdAt: REF_TIME - 5 * DAY_MS,
        endDate: '2025-04-19',
      },
    ], { ticker: 'BTC-USD', horizonDays: 30, referenceTimeMs: REF_TIME });

    expect(result).toHaveLength(1);
    expect(result[0].trustScore).toBe('low');
  });

  it('keeps undated BTC 30-day anchors at low trust even when mature', () => {
    const result = extractPriceThresholds([
      {
        question: 'Will the price of Bitcoin be above $84000 in May?',
        probability: 0.50,
        volume: 30000,
        createdAt: REF_TIME - 5 * DAY_MS,
      },
    ], { ticker: 'BTC-USD', horizonDays: 30, referenceTimeMs: REF_TIME });

    expect(result).toHaveLength(1);
    expect(result[0].trustScore).toBe('low');
  });

  it('does not activate for non-crypto tickers even with zero anchors', () => {
    const markets = [
      { question: 'Will SPY be above $500 on April 17?', probability: 0.6, volume: 10000, endDate: '2025-04-17' },
    ];

    const result = applyCryptoTerminalAnchorFallback(
      markets, [], 'SPY', 14, REF_TIME,
    );
    expect(result).toHaveLength(0);
  });

  it('clamps date-gap discount to minimum 0.5', () => {
    // Anchor endDate far from horizon → extreme offset discount
    const markets = [
      { question: 'Will the price of Bitcoin be below $60000 on April 16?', probability: 0.75, volume: 20000, createdAt: REF_TIME - 10 * DAY_MS, endDate: '2025-04-16' },
    ];

    const result = applyCryptoTerminalAnchorFallback(
      markets, [], 'BTC-USD', 14, REF_TIME,
    );
    expect(result).toHaveLength(1);
    // Ensure probability is still > 0 (clamped)
    expect(result[0].probability).toBeGreaterThan(0);
  });
});
describe('market horizon helpers', () => {
  const REF_TIME = new Date('2026-05-06T12:00:00Z').getTime();
  const DAY_MS = MS_PER_DAY;

  it('sorts markets by horizon closeness relative to the supplied reference time', () => {
    const markets = [
      { question: 'near-a', endDate: new Date(REF_TIME + 5 * DAY_MS).toISOString(), volume: 100 },
      { question: 'near-b', endDate: new Date(REF_TIME + 8 * DAY_MS).toISOString(), volume: 90 },
      { question: 'far', endDate: new Date(REF_TIME + 12 * DAY_MS).toISOString(), volume: 200 },
    ];

    expect(
      sortMarketsByHorizonCloseness(markets, 5, REF_TIME).map((market) => market.question),
    ).toEqual(['near-a', 'near-b', 'far']);
    expect(
      sortMarketsByHorizonCloseness(markets, 5, REF_TIME + 3 * DAY_MS).map((market) => market.question),
    ).toEqual(['near-b', 'near-a', 'far']);
  });

  it('filters markets relative to the supplied reference time on the strict horizon path', () => {
    const markets = [
      { question: 'day-1', endDate: new Date(REF_TIME + 1 * DAY_MS).toISOString() },
      { question: 'day-4', endDate: new Date(REF_TIME + 4 * DAY_MS).toISOString() },
      { question: 'day-9', endDate: new Date(REF_TIME + 9 * DAY_MS).toISOString() },
    ];

    expect(filterMarketsToHorizon(markets, 1, REF_TIME).map((market) => market.question)).toEqual(['day-1']);
    expect(
      filterMarketsToHorizon(markets, 1, REF_TIME + 3 * DAY_MS).map((market) => market.question),
    ).toEqual(['day-4']);
  });

  it('uses the supplied reference time when falling back to closest markets', () => {
    const markets = [
      { question: 'closer-at-start', endDate: new Date(REF_TIME + 6 * DAY_MS).toISOString(), volume: 10 },
      { question: 'closer-later', endDate: new Date(REF_TIME + 9 * DAY_MS).toISOString(), volume: 20 },
    ];

    expect(filterMarketsToHorizon(markets, 1, REF_TIME).map((market) => market.question)).toEqual([
      'closer-at-start',
      'closer-later',
    ]);
    expect(
      filterMarketsToHorizon(markets, 1, REF_TIME + 20 * DAY_MS).map((market) => market.question),
    ).toEqual([
      'closer-later',
      'closer-at-start',
    ]);
  });
});
describe('assessAnchorCoverage', () => {
  it('returns quality=none with zero anchors', () => {
    const result = assessAnchorCoverage([], 100);
    expect(result.quality).toBe('none');
    expect(result.trustedAnchors).toBe(0);
    expect(result.warning).toContain('No trusted');
  });

  it('returns quality=sparse with few far-apart anchors', () => {
    const anchors = [
      { price: 105, rawProbability: 0.9, probability: 0.85, trustScore: 'high' as const, source: 'polymarket' as const },
      { price: 200, rawProbability: 0.1, probability: 0.09, trustScore: 'high' as const, source: 'polymarket' as const },
    ];
    const result = assessAnchorCoverage(anchors, 100);
    expect(result.quality).toBe('sparse');
    expect(result.maxGapPct).toBeGreaterThan(15);
    expect(result.warning).toContain('Sparse');
  });

  it('returns quality=good with ≥3 closely-spaced anchors', () => {
    const anchors = [
      { price: 95,  rawProbability: 0.8, probability: 0.76, trustScore: 'high' as const, source: 'polymarket' as const },
      { price: 100, rawProbability: 0.5, probability: 0.47, trustScore: 'high' as const, source: 'polymarket' as const },
      { price: 105, rawProbability: 0.3, probability: 0.28, trustScore: 'high' as const, source: 'polymarket' as const },
    ];
    const result = assessAnchorCoverage(anchors, 100);
    expect(result.quality).toBe('good');
    expect(result.warning).toBe('');
  });

  it('ignores low-trust anchors', () => {
    const anchors = [
      { price: 95,  rawProbability: 0.8, probability: 0.76, trustScore: 'low' as const, source: 'polymarket' as const },
      { price: 100, rawProbability: 0.5, probability: 0.47, trustScore: 'low' as const, source: 'polymarket' as const },
    ];
    const result = assessAnchorCoverage(anchors, 100);
    expect(result.quality).toBe('none');
    expect(result.totalAnchors).toBe(2);
    expect(result.trustedAnchors).toBe(0);
  });

  it('anchorCoverage is included in computeMarkovDistribution result', async () => {
    const prices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.5);
    const result = await computeMarkovDistribution({
      ticker: 'COVTEST',
      horizon: 10,
      currentPrice: 119.5,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.metadata.anchorCoverage).toBeDefined();
    expect(result.metadata.anchorCoverage.quality).toBe('none');
  });
});
describe('interpolateDistribution anchor grid merging', () => {
  it('includes far-away anchors in distribution grid', () => {
    // Oil scenario: current price $100, anchor at $200 (100% away)
    const P = buildDefaultMatrix();
    const regimeStats = estimateRegimeStats(
      Array.from({ length: 30 }, () => 0.001),
      Array.from({ length: 30 }, () => 'sideways' as const),
    );
    const farAnchor = {
      price: 200,
      rawProbability: 0.15,
      probability: 0.14,
      trustScore: 'high' as const,
      source: 'polymarket' as const,
    };
    const dist = interpolateDistribution(100, 30, P, regimeStats, 'sideways', [farAnchor], 0.5);

    // The $200 anchor should now be included in the grid
    const hasNearAnchor = dist.some(d => Math.abs(d.price - 200) / 200 < 0.06);
    expect(hasNearAnchor).toBe(true);
  });

  it('grid extends below when anchor is below default range', () => {
    const P = buildDefaultMatrix();
    const regimeStats = estimateRegimeStats(
      Array.from({ length: 30 }, () => 0.001),
      Array.from({ length: 30 }, () => 'sideways' as const),
    );
    const lowAnchor = {
      price: 50,
      rawProbability: 0.95,
      probability: 0.90,
      trustScore: 'high' as const,
      source: 'polymarket' as const,
    };
    const dist = interpolateDistribution(100, 30, P, regimeStats, 'sideways', [lowAnchor], 0.5);
    const minPrice = Math.min(...dist.map(d => d.price));
    expect(minPrice).toBeLessThan(55);
  });

  it('far-away anchors have dampened influence (distance decay)', () => {
    const P = buildDefaultMatrix();
    const regimeStats = estimateRegimeStats(
      Array.from({ length: 30 }, () => 0.001),
      Array.from({ length: 30 }, () => 'sideways' as const),
    );
    // Near anchor: 5% away, probability 0.70
    const nearAnchor = {
      price: 105,
      rawProbability: 0.70,
      probability: 0.70,
      trustScore: 'high' as const,
      source: 'polymarket' as const,
    };
    // Far anchor: 50% away, probability 0.70
    const farAnchor = {
      price: 150,
      rawProbability: 0.70,
      probability: 0.70,
      trustScore: 'high' as const,
      source: 'polymarket' as const,
    };

    // Get distributions with each anchor separately
    const distNear = interpolateDistribution(100, 14, P, regimeStats, 'sideways', [nearAnchor], 0.5);
    const distFar = interpolateDistribution(100, 14, P, regimeStats, 'sideways', [farAnchor], 0.5);

    // Near the near-anchor price, the anchor should strongly influence the result
    const nearPoint = distNear.find(d => Math.abs(d.price - 105) / 105 < 0.03);
    // Near the far-anchor price, the influence should be dampened
    const farPoint = distFar.find(d => Math.abs(d.price - 150) / 150 < 0.03);

    // Both should exist in the grid
    expect(nearPoint).toBeDefined();
    expect(farPoint).toBeDefined();

    if (nearPoint && farPoint) {
      // The near anchor should pull probability closer to 0.70
      // The far anchor should be more dampened (closer to pure Markov)
      // At 5% distance: distanceWeight ≈ 0.988 (nearly full influence)
      // At 50% distance: distanceWeight ≈ 0.287 (heavily dampened)
      // So the far point's probability should be further from 0.70 (closer to Markov)
      const nearDeviation = Math.abs(nearPoint.probability - 0.70);
      const farDeviation = Math.abs(farPoint.probability - 0.70);
      expect(farDeviation).toBeGreaterThan(nearDeviation);
    }
  });
});
describe('Polymarket anchor influence', () => {
  const basePrices = Array.from({ length: 60 }, (_, i) => 100 + Math.sin(i / 5) * 3);

  it('anchors shift distribution probabilities vs no-anchor baseline', async () => {
    // Without anchors
    const noAnchor = await computeMarkovDistribution({
      ticker: 'AB_TEST',
      horizon: 20,
      currentPrice: 100,
      historicalPrices: basePrices,
      polymarketMarkets: [],
    });

    // With a strong anchor saying P(>$105) = 0.90
    const withAnchor = await computeMarkovDistribution({
      ticker: 'AB_TEST',
      horizon: 20,
      currentPrice: 100,
      historicalPrices: basePrices,
      polymarketMarkets: [
        { question: 'Will AB_TEST exceed $105?', probability: 0.90, volume: 50000, createdAt: '2025-01-01' },
      ],
    });

    // Find the point nearest $105 in each distribution
    const findNear105 = (dist: typeof noAnchor.distribution) =>
      dist.reduce((best, d) => Math.abs(d.price - 105) < Math.abs(best.price - 105) ? d : best);

    const noAnchorProb = findNear105(noAnchor.distribution).probability;
    const withAnchorProb = findNear105(withAnchor.distribution).probability;

    // Anchor at 90% should pull the probability UPWARD relative to pure Markov
    expect(withAnchorProb).toBeGreaterThan(noAnchorProb * 0.8);
    // And anchor metadata should differ
    expect(withAnchor.metadata.polymarketAnchors).toBeGreaterThan(noAnchor.metadata.polymarketAnchors);
  });

  it('high-trust vs low-trust anchors have different influence', async () => {
    // Old market (high trust) vs very new market (low trust, <48h)
    const highTrust = await computeMarkovDistribution({
      ticker: 'TRUST_TEST',
      horizon: 15,
      currentPrice: 100,
      historicalPrices: basePrices,
      polymarketMarkets: [
        { question: 'Will it exceed $105?', probability: 0.80, volume: 100000, createdAt: '2024-01-01' },
      ],
      referenceTimeMs: FIXED_NOW_MS,
    });

    const lowTrust = await computeMarkovDistribution({
      ticker: 'TRUST_TEST',
      horizon: 15,
      currentPrice: 100,
      historicalPrices: basePrices,
      polymarketMarkets: [
        { question: 'Will it exceed $105?', probability: 0.80, volume: 100000, createdAt: new Date(FIXED_NOW_MS).toISOString() },
      ],
      referenceTimeMs: FIXED_NOW_MS,
    });

    // High-trust anchor count should be higher
    expect(highTrust.metadata.polymarketAnchors).toBeGreaterThanOrEqual(lowTrust.metadata.polymarketAnchors);
  });

  it('anchor coverage diagnostic reflects anchor presence', async () => {
    const noAnchor = await computeMarkovDistribution({
      ticker: 'COV_AB',
      horizon: 10,
      currentPrice: 100,
      historicalPrices: basePrices,
      polymarketMarkets: [],
    });
    expect(noAnchor.metadata.anchorCoverage.quality).toBe('none');

    const withAnchors = await computeMarkovDistribution({
      ticker: 'COV_AB',
      horizon: 10,
      currentPrice: 100,
      historicalPrices: basePrices,
      polymarketMarkets: [
        { question: 'Will it exceed $95?', probability: 0.85, volume: 50000, createdAt: '2024-01-01' },
        { question: 'Will it exceed $100?', probability: 0.50, volume: 50000, createdAt: '2024-01-01' },
        { question: 'Will it exceed $105?', probability: 0.25, volume: 50000, createdAt: '2024-01-01' },
      ],
    });
    expect(withAnchors.metadata.anchorCoverage.quality).toBe('good');
    expect(withAnchors.metadata.anchorCoverage.trustedAnchors).toBe(3);
  });
});
