import { describe, it, expect } from 'bun:test';
import {
  buildRecommendationProvenanceNote,
  computeActionLevels,
  computeActionSignal,
  computeMarkovDistribution,
  computeScenarioProbabilities,
  interpolateSurvival,
} from './markov-distribution.js';
import type { MarkovDistributionPoint, ScenarioProbabilities } from './markov-distribution.js';

/** Build a synthetic linear distribution: P(>price) = 1 − (price − lo) / (hi − lo) */
function makeLinearDist(lo: number, hi: number, n = 21): MarkovDistributionPoint[] {
  return Array.from({ length: n }, (_, i) => {
    const price = lo + (hi - lo) * (i / (n - 1));
    const prob  = 1 - i / (n - 1);
    return { price, probability: prob, lowerBound: prob - 0.05, upperBound: prob + 0.05, source: 'markov' as const };
  });
}

describe('interpolateSurvival', () => {
  const dist = makeLinearDist(80, 120);

  it('returns 1.0 for price below the distribution minimum', () => {
    expect(interpolateSurvival(dist, 50)).toBe(1.0);
  });

  it('returns 0.0 for price above the distribution maximum', () => {
    expect(interpolateSurvival(dist, 200)).toBe(0.0);
  });

  it('returns 0.5 for the exact midpoint', () => {
    // midpoint = 100, which is the exact center of [80, 120]
    const p = interpolateSurvival(dist, 100);
    expect(p).toBeCloseTo(0.5, 1);
  });

  it('returns 0.5 for empty distribution', () => {
    expect(interpolateSurvival([], 100)).toBe(0.5);
  });

  it('interpolates correctly between two known points', () => {
    // At price 90 (25% along [80, 120]) → P ≈ 0.75
    const p = interpolateSurvival(dist, 90);
    expect(p).toBeGreaterThan(0.7);
    expect(p).toBeLessThan(0.8);
  });
});
describe('computeScenarioProbabilities', () => {
  it('bucket probabilities sum to ~1.0', () => {
    const dist = makeLinearDist(80, 120);
    const result = computeScenarioProbabilities(dist, 100);
    const total = result.buckets.reduce((s, b) => s + b.probability, 0);
    expect(total).toBeCloseTo(1.0, 2);
  });

  it('returns 5 labeled buckets in the correct order', () => {
    const dist = makeLinearDist(80, 120);
    const result = computeScenarioProbabilities(dist, 100);
    expect(result.buckets).toHaveLength(5);
    expect(result.buckets.map(b => b.label)).toEqual([
      'Down >5%', 'Down 3–5%', 'Flat ±3%', 'Up 3–5%', 'Up >5%',
    ]);
  });

  it('all bucket probabilities are non-negative', () => {
    const dist = makeLinearDist(80, 120);
    const result = computeScenarioProbabilities(dist, 100);
    for (const b of result.buckets) {
      expect(b.probability).toBeGreaterThanOrEqual(0);
    }
  });

  it('P(Up>5%) is consistent with CDF P(>price) at 1.05×current', () => {
    const dist = makeLinearDist(80, 120);
    const current = 100;
    const result = computeScenarioProbabilities(dist, current);
    const upOver5 = result.buckets.find(b => b.label === 'Up >5%')!;
    const cdfAt105 = interpolateSurvival(dist, current * 1.05);
    // P(Up>5%) = P(price > 1.05*current) from CDF
    expect(upOver5.probability).toBeCloseTo(cdfAt105, 5);
  });

  it('P(Down>5%) is consistent with CDF P(<price) at 0.95×current', () => {
    const dist = makeLinearDist(80, 120);
    const current = 100;
    const result = computeScenarioProbabilities(dist, current);
    const downOver5 = result.buckets.find(b => b.label === 'Down >5%')!;
    const cdfAt95 = interpolateSurvival(dist, current * 0.95);
    // P(Down>5%) = 1 - P(price > 0.95*current)
    expect(downOver5.probability).toBeCloseTo(1 - cdfAt95, 5);
  });

  it('Flat bucket covers the correct probability mass for a uniform distribution', () => {
    // Linear dist [80, 120] ≈ uniform. ±3% of 100 → [97, 103] = 6/40 = 15% of range.
    // For uniform, P(Flat) ≈ 6/40 = 0.15, which is smaller than P(Down>5%) ≈ 15/40 = 0.375
    const dist = makeLinearDist(80, 120);
    const result = computeScenarioProbabilities(dist, 100);
    const flat = result.buckets.find(b => b.label === 'Flat ±3%')!;
    expect(flat.probability).toBeCloseTo(0.15, 1);
  });

  it('price ranges are contiguous and cover the full distribution', () => {
    const dist = makeLinearDist(80, 120);
    const result = computeScenarioProbabilities(dist, 100);
    // Down >5%: [null, 95] — Down 3-5%: [95, 97] — Flat: [97, 103] — Up 3-5%: [103, 105] — Up >5%: [105, null]
    expect(result.buckets[0].priceRange[0]).toBeNull();
    expect(result.buckets[0].priceRange[1]).toBe(result.buckets[1].priceRange[0]);
    expect(result.buckets[1].priceRange[1]).toBe(result.buckets[2].priceRange[0]);
    expect(result.buckets[2].priceRange[1]).toBe(result.buckets[3].priceRange[0]);
    expect(result.buckets[3].priceRange[1]).toBe(result.buckets[4].priceRange[0]);
    expect(result.buckets[4].priceRange[1]).toBeNull();
  });

  it('pUp is consistent with interpolateSurvival at currentPrice', () => {
    const dist = makeLinearDist(80, 120);
    const current = 100;
    const result = computeScenarioProbabilities(dist, current);
    const cdfPUp = interpolateSurvival(dist, current);
    expect(result.pUp).toBeCloseTo(cdfPUp, 2);
  });

  it('expectedReturn is reasonable for a symmetric distribution', () => {
    const dist = makeLinearDist(80, 120);
    const result = computeScenarioProbabilities(dist, 100);
    // Symmetric around 100 → expected return near 0
    expect(Math.abs(result.expectedReturn)).toBeLessThan(0.05);
  });

  it('scenarios are consistent with CDF: cannot have P(Up>5%) > P(>lowerPrice) from CDF', () => {
    // This is the exact bug we're fixing: scenario P(Up>5%) must ≤ CDF P(>any price below the 5% threshold)
    const dist = makeLinearDist(80, 120);
    const current = 100;
    const result = computeScenarioProbabilities(dist, current);
    const upOver5 = result.buckets.find(b => b.label === 'Up >5%')!;
    // P(>$105) from CDF should exactly equal the scenario probability
    const cdfAt105 = interpolateSurvival(dist, current * 1.05);
    expect(upOver5.probability).toBeCloseTo(cdfAt105, 10);
    // And P(>$102) must be > P(Up>5%) (monotonicity)
    const cdfAt102 = interpolateSurvival(dist, current * 1.02);
    expect(cdfAt102).toBeGreaterThan(upOver5.probability);
  });
});
describe('computeActionSignal', () => {
  it('returns probabilities that sum to 1 (within floating-point tolerance)', () => {
    const dist = makeLinearDist(90, 110); // currentPrice = 100
    const sig = computeActionSignal(dist, 100);
    const total = sig.buyProbability + sig.holdProbability + sig.sellProbability;
    expect(total).toBeCloseTo(1.0, 5);
  });

  it('recommends SELL when distribution is heavily skewed downward', () => {
    // All prices below current → distribution skewed down
    const dist = makeLinearDist(50, 98); // max < current price of 100
    const sig = computeActionSignal(dist, 100);
    expect(sig.sellProbability).toBeGreaterThan(sig.buyProbability);
    expect(sig.recommendation).toBe('SELL');
  });

  it('recommends BUY when distribution is heavily skewed upward', () => {
    // All prices above current → distribution skewed up
    const dist = makeLinearDist(102, 150); // min > current price of 100
    const sig = computeActionSignal(dist, 100);
    expect(sig.buyProbability).toBeGreaterThan(sig.sellProbability);
    expect(sig.recommendation).toBe('BUY');
  });

  it('recommends HOLD for a symmetric distribution centred at current price', () => {
    const dist = makeLinearDist(90, 110);
    const sig = computeActionSignal(dist, 100);
    expect(sig.recommendation).toBe('HOLD');
    expect(sig.holdProbability).toBeGreaterThan(0);
  });

  it('respects custom thresholds', () => {
    const dist = makeLinearDist(90, 110);
    // With very tight thresholds most mass sits in HOLD zone
    const sig = computeActionSignal(dist, 100, 0.01, 0.01);
    expect(sig.buyThreshold).toBe(0.01);
    expect(sig.sellThreshold).toBe(0.01);
  });

  it('riskRewardRatio > 1 when distribution is bullish', () => {
    const dist = makeLinearDist(95, 120); // skewed upward
    const sig = computeActionSignal(dist, 100);
    expect(sig.riskRewardRatio).toBeGreaterThan(1);
  });

  it('riskRewardRatio < 1 when distribution is bearish', () => {
    const dist = makeLinearDist(80, 105); // skewed downward
    const sig = computeActionSignal(dist, 100);
    expect(sig.riskRewardRatio).toBeLessThan(1);
  });

  it('expectedReturn is positive for an upward-skewed distribution', () => {
    const dist = makeLinearDist(95, 120);
    const sig = computeActionSignal(dist, 100);
    expect(sig.expectedReturn).toBeGreaterThan(0);
  });

  it('expectedReturn is negative for a downward-skewed distribution', () => {
    const dist = makeLinearDist(80, 105);
    const sig = computeActionSignal(dist, 100);
    expect(sig.expectedReturn).toBeLessThan(0);
  });

  // ---------------------------------------------------------------------------
  // Cross-validation: scenario-gated recommendation
  // ---------------------------------------------------------------------------

  it('downgrades BUY to HOLD when P(up) < 0.50 from scenarios', () => {
    // Right-skewed dist with fat tail: mean positive but median below current
    // Prices from 90 to 200, but most mass is near 90-100 (bearish median)
    // Create distribution where P(>100) ≈ 0.45 (more likely to go down)
    // but E[price] > 100 due to fat right tail above 150
    const dist: MarkovDistributionPoint[] = [
      { price: 90,  probability: 0.99, lowerBound: 0.95, upperBound: 1.0,  source: 'markov' },
      { price: 95,  probability: 0.85, lowerBound: 0.80, upperBound: 0.90, source: 'markov' },
      { price: 100, probability: 0.45, lowerBound: 0.40, upperBound: 0.50, source: 'markov' },
      { price: 105, probability: 0.20, lowerBound: 0.15, upperBound: 0.25, source: 'markov' },
      { price: 110, probability: 0.15, lowerBound: 0.10, upperBound: 0.20, source: 'markov' },
      { price: 150, probability: 0.12, lowerBound: 0.08, upperBound: 0.15, source: 'markov' },
      { price: 200, probability: 0.10, lowerBound: 0.05, upperBound: 0.12, source: 'markov' },
    ];

    // Without scenarios, the fat right tail makes expectedReturn positive → BUY
    const sigNoScenarios = computeActionSignal(dist, 100, 0.05, 0.03, 30);
    expect(sigNoScenarios.expectedReturn).toBeGreaterThan(0);

    // With scenarios reflecting the CDF truth (P(up)=0.45 < 0.50), BUY should be downgraded
    const scenarios: ScenarioProbabilities = {
      buckets: [
        { label: 'Down >5%',  probability: 0.15, priceRange: [null, 95] },
        { label: 'Down 3–5%', probability: 0.30, priceRange: [95, 97] },
        { label: 'Flat ±3%',  probability: 0.25, priceRange: [97, 103] },
        { label: 'Up 3–5%',   probability: 0.10, priceRange: [103, 105] },
        { label: 'Up >5%',    probability: 0.20, priceRange: [105, null] },
      ],
      expectedPrice: 103,
      expectedReturn: 0.03,
      pUp: 0.45, // < 0.50 → bearish
    };

    const sigWithScenarios = computeActionSignal(dist, 100, 0.05, 0.03, 30, undefined, scenarios);
    expect(sigWithScenarios.recommendation).toBe('HOLD');
  });

  it('downgrades BUY to HOLD when downside scenarios exceed upside by >5pp', () => {
    const dist = makeLinearDist(90, 115); // slightly bullish
    const scenarios: ScenarioProbabilities = {
      buckets: [
        { label: 'Down >5%',  probability: 0.20, priceRange: [null, 95] },
        { label: 'Down 3–5%', probability: 0.15, priceRange: [95, 97] },
        { label: 'Flat ±3%',  probability: 0.40, priceRange: [97, 103] },
        { label: 'Up 3–5%',   probability: 0.10, priceRange: [103, 105] },
        { label: 'Up >5%',    probability: 0.15, priceRange: [105, null] },
      ],
      expectedPrice: 101,
      expectedReturn: 0.01,
      pUp: 0.55, // slightly above 0.50
    };
    // downside = 0.20 + 0.15 = 0.35, upside = 0.10 + 0.15 = 0.25
    // gap = 0.35 - 0.25 = 0.10 > 0.05 → downgrade BUY to HOLD
    const sig = computeActionSignal(dist, 100, 0.05, 0.03, 30, undefined, scenarios);
    expect(sig.recommendation).not.toBe('BUY');
  });

  it('downgrades SELL to HOLD when P(up) > 0.50 from scenarios', () => {
    // Bearish mean but P(up) > 0.50 → median above current
    const dist = makeLinearDist(50, 102); // mean below current, most mass below
    const scenarios: ScenarioProbabilities = {
      buckets: [
        { label: 'Down >5%',  probability: 0.10, priceRange: [null, 95] },
        { label: 'Down 3–5%', probability: 0.10, priceRange: [95, 97] },
        { label: 'Flat ±3%',  probability: 0.40, priceRange: [97, 103] },
        { label: 'Up 3–5%',   probability: 0.20, priceRange: [103, 105] },
        { label: 'Up >5%',    probability: 0.20, priceRange: [105, null] },
      ],
      expectedPrice: 101,
      expectedReturn: 0.01,
      pUp: 0.55, // > 0.50 → cannot SELL
    };
    const sig = computeActionSignal(dist, 100, 0.05, 0.03, 30, undefined, scenarios);
    // Even if mean-based logic would say SELL, P(up) > 0.50 overrides
    expect(sig.recommendation).not.toBe('SELL');
  });

  it('preserves BUY when scenarios confirm bullish tilt', () => {
    const dist = makeLinearDist(95, 120); // clearly bullish
    const scenarios: ScenarioProbabilities = {
      buckets: [
        { label: 'Down >5%',  probability: 0.05, priceRange: [null, 95] },
        { label: 'Down 3–5%', probability: 0.05, priceRange: [95, 97] },
        { label: 'Flat ±3%',  probability: 0.30, priceRange: [97, 103] },
        { label: 'Up 3–5%',   probability: 0.25, priceRange: [103, 105] },
        { label: 'Up >5%',    probability: 0.35, priceRange: [105, null] },
      ],
      expectedPrice: 108,
      expectedReturn: 0.08,
      pUp: 0.65, // clearly bullish
    };
    const sig = computeActionSignal(dist, 100, 0.05, 0.03, 30, undefined, scenarios);
    expect(sig.recommendation).toBe('BUY');
  });

  it('caps confidence to MEDIUM when mean and median disagree in sign', () => {
    // Distribution where median < current (P(>100) = 0.45) but mean > current (fat upper tail)
    const dist: MarkovDistributionPoint[] = [
      { price: 90,  probability: 0.99, lowerBound: 0.95, upperBound: 1.0,  source: 'markov' },
      { price: 95,  probability: 0.65, lowerBound: 0.60, upperBound: 0.70, source: 'markov' },
      { price: 100, probability: 0.45, lowerBound: 0.40, upperBound: 0.50, source: 'markov' },
      { price: 105, probability: 0.30, lowerBound: 0.25, upperBound: 0.35, source: 'markov' },
      { price: 110, probability: 0.20, lowerBound: 0.15, upperBound: 0.25, source: 'markov' },
      { price: 150, probability: 0.15, lowerBound: 0.10, upperBound: 0.20, source: 'markov' },
      { price: 200, probability: 0.10, lowerBound: 0.05, upperBound: 0.15, source: 'markov' },
    ];
    const scenarios: ScenarioProbabilities = {
      buckets: [
        { label: 'Down >5%',  probability: 0.35, priceRange: [null, 95] },
        { label: 'Down 3–5%', probability: 0.10, priceRange: [95, 97] },
        { label: 'Flat ±3%',  probability: 0.25, priceRange: [97, 103] },
        { label: 'Up 3–5%',   probability: 0.10, priceRange: [103, 105] },
        { label: 'Up >5%',    probability: 0.20, priceRange: [105, null] },
      ],
      expectedPrice: 117, // mean > current (fat tail)
      expectedReturn: 0.17, // mean return positive
      pUp: 0.45,
    };
    // CDF median is ~$98.75 (below $100) → negative median return
    // Mean is ~$117 (fat tail) → positive expected return
    const sig = computeActionSignal(dist, 100, 0.05, 0.03, 30, undefined, scenarios);
    // Mean is positive, median is negative → confidence should not be HIGH
    if (sig.expectedReturn > 0) {
      expect(sig.confidence).not.toBe('HIGH');
    }
  });

  it('uses bullish short-horizon crypto direction when pUp is clear but expected return stays in HOLD band', () => {
    const dist: MarkovDistributionPoint[] = [
      { price: 97,  probability: 0.95, lowerBound: 0.90, upperBound: 1.00, source: 'markov' },
      { price: 99,  probability: 0.72, lowerBound: 0.68, upperBound: 0.76, source: 'markov' },
      { price: 100, probability: 0.56, lowerBound: 0.52, upperBound: 0.60, source: 'markov' },
      { price: 101, probability: 0.40, lowerBound: 0.36, upperBound: 0.44, source: 'markov' },
      { price: 103, probability: 0.18, lowerBound: 0.14, upperBound: 0.22, source: 'markov' },
      { price: 105, probability: 0.08, lowerBound: 0.05, upperBound: 0.11, source: 'markov' },
    ];
    const scenarios: ScenarioProbabilities = {
      buckets: [
        { label: 'Down >5%',  probability: 0.04, priceRange: [null, 95] },
        { label: 'Down 3–5%', probability: 0.16, priceRange: [95, 97] },
        { label: 'Flat ±3%',  probability: 0.28, priceRange: [97, 103] },
        { label: 'Up 3–5%',   probability: 0.22, priceRange: [103, 105] },
        { label: 'Up >5%',    probability: 0.30, priceRange: [105, null] },
      ],
      expectedPrice: 100.2,
      expectedReturn: 0.002,
      pUp: 0.56,
    };

    const sig = computeActionSignal(dist, 100, 0.05, 0.03, 7, 0.04, scenarios, 'crypto');

    expect(sig.expectedReturn).toBeGreaterThan(0);
    expect(sig.recommendation).toBe('BUY');
    expect(sig.baseRecommendation).toBe('HOLD');
    expect(sig.recommendationSource).toBe('short_horizon_scenario');
  });

  it('uses bearish short-horizon crypto direction when pUp is clear but expected return stays in HOLD band', () => {
    const dist: MarkovDistributionPoint[] = [
      { price: 95,  probability: 0.92, lowerBound: 0.88, upperBound: 0.96, source: 'markov' },
      { price: 97,  probability: 0.80, lowerBound: 0.76, upperBound: 0.84, source: 'markov' },
      { price: 99,  probability: 0.62, lowerBound: 0.58, upperBound: 0.66, source: 'markov' },
      { price: 100, probability: 0.44, lowerBound: 0.40, upperBound: 0.48, source: 'markov' },
      { price: 101, probability: 0.28, lowerBound: 0.24, upperBound: 0.32, source: 'markov' },
      { price: 103, probability: 0.12, lowerBound: 0.09, upperBound: 0.15, source: 'markov' },
    ];
    const scenarios: ScenarioProbabilities = {
      buckets: [
        { label: 'Down >5%',  probability: 0.22, priceRange: [null, 95] },
        { label: 'Down 3–5%', probability: 0.20, priceRange: [95, 97] },
        { label: 'Flat ±3%',  probability: 0.24, priceRange: [97, 103] },
        { label: 'Up 3–5%',   probability: 0.16, priceRange: [103, 105] },
        { label: 'Up >5%',    probability: 0.18, priceRange: [105, null] },
      ],
      expectedPrice: 99.7,
      expectedReturn: -0.003,
      pUp: 0.44,
    };

    const sig = computeActionSignal(dist, 100, 0.05, 0.03, 7, 0.04, scenarios, 'crypto');

    expect(sig.expectedReturn).toBeLessThan(0);
    expect(sig.recommendation).toBe('SELL');
    expect(sig.baseRecommendation).toBe('HOLD');
    expect(sig.recommendationSource).toBe('short_horizon_scenario');
  });

  it('keeps short-horizon crypto HOLD when weak bearish pUp conflicts with positive expected return', () => {
    const dist: MarkovDistributionPoint[] = [
      { price: 98,  probability: 1.00, lowerBound: 0.96, upperBound: 1.00, source: 'markov' },
      { price: 99,  probability: 0.60, lowerBound: 0.56, upperBound: 0.64, source: 'markov' },
      { price: 100, probability: 0.478, lowerBound: 0.44, upperBound: 0.52, source: 'markov' },
      { price: 101, probability: 0.38, lowerBound: 0.34, upperBound: 0.42, source: 'markov' },
      { price: 103, probability: 0.00, lowerBound: 0.00, upperBound: 0.04, source: 'markov' },
    ];
    const scenarios = computeScenarioProbabilities(dist, 100);

    const sig = computeActionSignal(dist, 100, 0.05, 0.03, 2, 0.02, scenarios, 'crypto');

    expect(scenarios.pUp).toBeCloseTo(0.478);
    expect(sig.expectedReturn).toBeGreaterThan(0);
    expect(sig.riskRewardRatio).toBeGreaterThan(1);
    expect(sig.recommendation).toBe('HOLD');
    expect(sig.baseRecommendation).toBe('HOLD');
    expect(sig.recommendationSource).toBe('expected_return');
  });

  it('keeps the non-target short-horizon equity path unchanged for the same bullish setup', () => {
    const dist: MarkovDistributionPoint[] = [
      { price: 97,  probability: 0.95, lowerBound: 0.90, upperBound: 1.00, source: 'markov' },
      { price: 99,  probability: 0.72, lowerBound: 0.68, upperBound: 0.76, source: 'markov' },
      { price: 100, probability: 0.56, lowerBound: 0.52, upperBound: 0.60, source: 'markov' },
      { price: 101, probability: 0.40, lowerBound: 0.36, upperBound: 0.44, source: 'markov' },
      { price: 103, probability: 0.18, lowerBound: 0.14, upperBound: 0.22, source: 'markov' },
      { price: 105, probability: 0.08, lowerBound: 0.05, upperBound: 0.11, source: 'markov' },
    ];
    const scenarios: ScenarioProbabilities = {
      buckets: [
        { label: 'Down >5%',  probability: 0.04, priceRange: [null, 95] },
        { label: 'Down 3–5%', probability: 0.16, priceRange: [95, 97] },
        { label: 'Flat ±3%',  probability: 0.28, priceRange: [97, 103] },
        { label: 'Up 3–5%',   probability: 0.22, priceRange: [103, 105] },
        { label: 'Up >5%',    probability: 0.30, priceRange: [105, null] },
      ],
      expectedPrice: 100.2,
      expectedReturn: 0.002,
      pUp: 0.56,
    };

    const sig = computeActionSignal(dist, 100, 0.05, 0.03, 7, 0.04, scenarios, 'equity');

    expect(sig.recommendation).toBe('HOLD');
  });

  it('builds a provenance note for short-horizon crypto scenario overrides', () => {
    const note = buildRecommendationProvenanceNote({
      ticker: 'BTC-USD',
      horizon: 7,
      actionSignal: {
        buyProbability: 0.21,
        holdProbability: 0.39,
        sellProbability: 0.40,
        recommendation: 'SELL',
        confidence: 'LOW',
        expectedReturn: -0.003,
        riskRewardRatio: 0.92,
        buyThreshold: 0.05,
        sellThreshold: 0.03,
        actionLevels: {
          targetPrice: 103,
          stopLoss: 95,
          medianPrice: 99.7,
          bullCase: 105,
          bearCase: 94,
        },
        baseRecommendation: 'HOLD',
        recommendationSource: 'short_horizon_scenario',
      },
      scenarios: {
        buckets: [
          { label: 'Down >5%', probability: 0.22, priceRange: [null, 95] },
          { label: 'Down 3–5%', probability: 0.20, priceRange: [95, 97] },
          { label: 'Flat ±3%', probability: 0.24, priceRange: [97, 103] },
          { label: 'Up 3–5%', probability: 0.16, priceRange: [103, 105] },
          { label: 'Up >5%', probability: 0.18, priceRange: [105, null] },
        ],
        expectedPrice: 99.7,
        expectedReturn: -0.003,
        pUp: 0.44,
      },
    });

    expect(note).toContain('converted a HOLD into SELL');
    expect(note).toContain('P(up) is 44.0%');
  });

  it('builds a provenance note for the BTC bearish-break SELL gate', () => {
    const note = buildRecommendationProvenanceNote({
      ticker: 'BTC-USD',
      horizon: 14,
      actionSignal: {
        buyProbability: 0.32,
        holdProbability: 0.41,
        sellProbability: 0.27,
        recommendation: 'SELL',
        confidence: 'LOW',
        expectedReturn: 0.012,
        riskRewardRatio: 1.21,
        buyThreshold: 0.05,
        sellThreshold: 0.03,
        actionLevels: {
          targetPrice: 110,
          stopLoss: 95,
          medianPrice: 102,
          bullCase: 114,
          bearCase: 93,
        },
        baseRecommendation: 'BUY',
        recommendationSource: 'expected_return',
      },
      bearishBreakRecommendationGateActive: true,
    });

    expect(note).toContain('bearish-break gate fired');
    expect(note).toContain('final SELL overrides the base BUY');
  });

  it('builds a provenance note when the latent regime and final trade side differ', () => {
    const note = buildRecommendationProvenanceNote({
      ticker: 'BTC-USD',
      horizon: 1,
      regimeState: 'bull',
      actionSignal: {
        buyProbability: 0.31,
        holdProbability: 0.38,
        sellProbability: 0.31,
        recommendation: 'SELL',
        confidence: 'LOW',
        expectedReturn: -0.004,
        riskRewardRatio: 0.97,
        buyThreshold: 0.05,
        sellThreshold: 0.03,
        actionLevels: {
          targetPrice: 80800,
          stopLoss: 77500,
          medianPrice: 79850,
          bullCase: 81100,
          bearCase: 78600,
        },
        baseRecommendation: 'SELL',
        recommendationSource: 'expected_return',
      },
      scenarios: {
        buckets: [
          { label: 'Down >5%', probability: 0.029, priceRange: [null, 75907] },
          { label: 'Down 3–5%', probability: 0.072, priceRange: [75907, 77505] },
          { label: 'Flat ±3%', probability: 0.822, priceRange: [77505, 82300] },
          { label: 'Up 3–5%', probability: 0.056, priceRange: [82300, 83898] },
          { label: 'Up >5%', probability: 0.022, priceRange: [83898, null] },
        ],
        expectedPrice: 79873.21,
        expectedReturn: -0.0004,
        pUp: 0.499,
      },
    });

    expect(note).toContain('latent HMM backdrop');
    expect(note).toContain('Final SELL');
    expect(note).toContain('P(up) is 49.9%');
  });

  it('raw-direction hybrid preserves calibrated output surfaces while enabling BTC short-horizon provenance', async () => {
    const prices = Array.from({ length: 140 }, (_, i) => 100 + i * 0.15 + Math.sin(i * 0.25) * 2.5);
    const currentPrice = prices[prices.length - 1]!;

    const baseline = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
    });

    const hybrid = await computeMarkovDistribution({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets: [],
      rawDirectionHybrid: true,
    });

    expect(hybrid.distribution).toHaveLength(baseline.distribution.length);
    expect(hybrid.scenarios.expectedPrice).toBeCloseTo(baseline.scenarios.expectedPrice, 8);
    expect(hybrid.scenarios.expectedReturn).toBeCloseTo(baseline.scenarios.expectedReturn, 8);
    expect(hybrid.scenarios.pUp).toBeCloseTo(baseline.scenarios.pUp, 8);
    expect(hybrid.actionSignal.buyProbability).toBeCloseTo(baseline.actionSignal.buyProbability, 10);
    expect(hybrid.actionSignal.sellProbability).toBeCloseTo(baseline.actionSignal.sellProbability, 10);
    expect(hybrid.actionSignal.actionLevels).toEqual(baseline.actionSignal.actionLevels);
    expect(hybrid.metadata.rawDirectionHybridActive).toBe(true);
  });

  it('computeMarkovDistribution result includes actionSignal field', async () => {
    const prices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.5);
    const result = await computeMarkovDistribution({
      ticker: 'SIGTEST',
      horizon: 10,
      currentPrice: 119.5,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.actionSignal).toBeDefined();
    expect(['BUY', 'HOLD', 'SELL']).toContain(result.actionSignal.recommendation);
    expect(['HIGH', 'MEDIUM', 'LOW']).toContain(result.actionSignal.confidence);
    const { buyProbability: b, holdProbability: h, sellProbability: s } = result.actionSignal;
    expect(b + h + s).toBeCloseTo(1.0, 4);
  });

  it('actionSignal includes actionLevels with valid price targets', async () => {
    const prices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.5);
    const result = await computeMarkovDistribution({
      ticker: 'LVLTEST',
      horizon: 10,
      currentPrice: 119.5,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const lvl = result.actionSignal.actionLevels;
    expect(lvl).toBeDefined();
    expect(lvl.medianPrice).toBeGreaterThan(0);
    expect(lvl.targetPrice).toBeGreaterThan(0);
    expect(lvl.stopLoss).toBeGreaterThan(0);
    expect(lvl.bullCase).toBeGreaterThan(0);
    expect(lvl.bearCase).toBeGreaterThan(0);
    // targetPrice > medianPrice (target is the optimistic case)
    expect(lvl.targetPrice).toBeGreaterThanOrEqual(lvl.medianPrice);
    // stopLoss < medianPrice (stop-loss is below expected)
    expect(lvl.stopLoss).toBeLessThanOrEqual(lvl.medianPrice);
    // bullCase > bearCase
    expect(lvl.bullCase).toBeGreaterThan(lvl.bearCase);
  });
});
describe('computeActionLevels', () => {
  it('median is at 50th percentile of linear distribution', () => {
    const dist = makeLinearDist(80, 120);
    const lvl = computeActionLevels(dist, 100);
    expect(lvl.medianPrice).toBeCloseTo(100, 0);
  });

  it('target > median > stopLoss for any distribution', () => {
    const dist = makeLinearDist(80, 120);
    const lvl = computeActionLevels(dist, 100);
    expect(lvl.targetPrice).toBeGreaterThanOrEqual(lvl.medianPrice);
    expect(lvl.medianPrice).toBeGreaterThanOrEqual(lvl.stopLoss);
  });

  it('bullCase > bearCase for any distribution', () => {
    const dist = makeLinearDist(80, 120);
    const lvl = computeActionLevels(dist, 100);
    expect(lvl.bullCase).toBeGreaterThan(lvl.bearCase);
  });

  it('handles bullish distribution (all prices above current)', () => {
    const dist = makeLinearDist(102, 150);
    const lvl = computeActionLevels(dist, 100);
    expect(lvl.stopLoss).toBeGreaterThan(100);
    expect(lvl.targetPrice).toBeGreaterThan(100);
  });

  it('handles bearish distribution (all prices below current)', () => {
    const dist = makeLinearDist(50, 98);
    const lvl = computeActionLevels(dist, 100);
    expect(lvl.targetPrice).toBeLessThan(100);
    expect(lvl.medianPrice).toBeLessThan(100);
  });

  it('returns currentPrice for empty distribution', () => {
    const lvl = computeActionLevels([], 100);
    expect(lvl.medianPrice).toBe(100);
  });
});
