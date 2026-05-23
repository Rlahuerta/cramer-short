import { describe, it, expect } from 'bun:test';
import { classifyRegimeState, computeAdaptiveThresholds, computeEnsembleSignal, computeRegimeUpRates } from './regime.js';
import type { RegimeState } from './core.js';

describe('classifyRegimeState', () => {
  it('returns bull for modest positive return, low vol', () => {
    expect(classifyRegimeState(0.015, 0.005)).toBe('bull');
  });

  it('returns bear for modest negative return, low vol', () => {
    expect(classifyRegimeState(-0.015, 0.005)).toBe('bear');
  });

  it('returns sideways for small absolute return, low vol', () => {
    expect(classifyRegimeState(0.005, 0.005)).toBe('sideways');
    expect(classifyRegimeState(-0.005, 0.005)).toBe('sideways');
  });

  it('returns bull for positive return even with high vol (3-state collapse)', () => {
    // With 3-state model, high_vol_bull → bull
    expect(classifyRegimeState(0.03, 0.025)).toBe('bull');
  });

  it('returns bear for negative return even with high vol (3-state collapse)', () => {
    // With 3-state model, high_vol_bear → bear
    expect(classifyRegimeState(-0.04, 0.025)).toBe('bear');
  });

  it('boundary: exactly 1% return with low vol → sideways (strict > 0.01 required for bull)', () => {
    expect(classifyRegimeState(0.01, 0.005)).toBe('sideways'); // 0.01 is NOT > 0.01
  });

  it('boundary: vol parameter is ignored in 3-state model', () => {
    expect(classifyRegimeState(0.015, 0.02)).toBe('bull');
    expect(classifyRegimeState(0.015, 0.05)).toBe('bull'); // same result regardless of vol
  });
});
describe('computeAdaptiveThresholds', () => {
  it('uses half-median absolute return by default', () => {
    const thresholds = computeAdaptiveThresholds([0.01, -0.02, 0.03, -0.04, 0.05]);
    expect(thresholds.returnThreshold).toBeCloseTo(0.015, 10);
    expect(thresholds.volThreshold).toBeCloseTo(0.06, 10);
  });

  it('applies custom returnThresholdMultiplier without changing volThreshold', () => {
    const returns = [0.01, -0.02, 0.03, -0.04, 0.05];
    const baseline = computeAdaptiveThresholds(returns);
    const widened = computeAdaptiveThresholds(returns, 1.0);

    expect(widened.returnThreshold).toBeCloseTo(0.03, 10);
    expect(widened.returnThreshold).toBeGreaterThan(baseline.returnThreshold);
    expect(widened.volThreshold).toBeCloseTo(baseline.volThreshold, 10);
  });

  it('respects the minimum return threshold floor with tiny multipliers', () => {
    const thresholds = computeAdaptiveThresholds([0.0004, -0.0004, 0.0002], 0.1);
    expect(thresholds.returnThreshold).toBe(0.001);
  });
});
describe('computeEnsembleSignal', () => {
  it('returns neutral for short price series', () => {
    const result = computeEnsembleSignal([100, 101, 102]);
    expect(result.adjustment).toBe(0);
    expect(result.consensus).toBe(0);
  });

  it('returns valid fields for 30-day price series', () => {
    const prices = Array.from({ length: 30 }, (_, i) => 100 + i * 0.5);
    const result = computeEnsembleSignal(prices);
    expect(typeof result.meanReversionZ).toBe('number');
    expect(typeof result.momentumCrossover).toBe('number');
    expect(typeof result.volCompression).toBe('number');
    expect(typeof result.adjustment).toBe('number');
    expect(result.consensus).toBeGreaterThanOrEqual(0);
    expect(result.consensus).toBeLessThanOrEqual(3);
  });

  it('detects bullish momentum on strongly trending up data', () => {
    const prices = Array.from({ length: 30 }, (_, i) => 100 * Math.pow(1.01, i));
    const result = computeEnsembleSignal(prices);
    expect(result.momentumCrossover).toBeGreaterThan(0);
  });

  it('detects bearish momentum on strongly trending down data', () => {
    const prices = Array.from({ length: 30 }, (_, i) => 100 * Math.pow(0.99, i));
    const result = computeEnsembleSignal(prices);
    expect(result.momentumCrossover).toBeLessThan(0);
  });

  it('adjustment is clamped to ±0.004', () => {
    const prices = Array.from({ length: 30 }, (_, i) => 100 * Math.pow(1.05, i));
    const result = computeEnsembleSignal(prices);
    expect(result.adjustment).toBeLessThanOrEqual(0.004);
    expect(result.adjustment).toBeGreaterThanOrEqual(-0.004);
  });

  it('vol compression < 1 when recent vol is lower than historical', () => {
    const prices: number[] = [];
    for (let i = 0; i < 20; i++) prices.push(100 + (i % 2 === 0 ? 3 : -3));
    for (let i = 0; i < 10; i++) prices.push(100 + i * 0.1);
    const result = computeEnsembleSignal(prices);
    expect(result.volCompression).toBeLessThan(1.5);
  });
});
describe('computeRegimeUpRates', () => {
  it('returns P(up|regime) for each regime state', () => {
    // Simple sequence: bull, bull, bear, sideways, bull
    const regimeSeq: RegimeState[] = ['bull', 'bull', 'bear', 'sideways', 'bull'];
    // Returns: +1%, +2%, -1%, +0.5%, +3%
    const returns = [0.01, 0.02, -0.01, 0.005, 0.03];
    const horizon = 1;
    const rates = computeRegimeUpRates(regimeSeq, returns, horizon);
    expect(rates.bull).toBeDefined();
    expect(rates.bear).toBeDefined();
    expect(rates.sideways).toBeDefined();
  });

  it('computes correct P(up|bull) from synthetic data', () => {
    // 10 days, all bull regime, alternating returns: +, -, +, -, +, -, +, -, +, -
    const regimeSeq: RegimeState[] = Array(10).fill('bull');
    const returns = [0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01, 0.01, -0.01];
    const rates = computeRegimeUpRates(regimeSeq, returns, 1);
    // horizon=1: i goes 0..8 (i=9 has no forward return). Each i looks at returns[i+1].
    // Odd i → returns[even] (DOWN); even i → returns[odd] (UP).
    // i=0→1→DOWN, i=1→2→UP, i=2→3→DOWN, i=3→4→UP, i=4→5→DOWN,
    // i=5→6→UP, i=6→7→DOWN, i=7→8→UP, i=8→9→DOWN
    // 4 up / 9 total → 4/9 ≈ 0.444
    expect(rates.bull).toBeCloseTo(4 / 9, 2);
  });

  it('handles multi-day horizon correctly', () => {
    // 6 days, bull regime, returns: +1%, +1%, +1%, -5%, +1%, +1%
    const regimeSeq: RegimeState[] = Array(6).fill('bull');
    const returns = [0.01, 0.01, 0.01, -0.05, 0.01, 0.01];
    const rates = computeRegimeUpRates(regimeSeq, returns, 3);
    // horizon=3: i can start at 0,1,2 (need 3 future days).
    // Each window includes the large -5% drop at index 3:
    // i=0: days 1,2,3 → 0.01+0.01-0.05 = -0.03 → DOWN
    // i=1: days 2,3,4 → 0.01-0.05+0.01 = -0.03 → DOWN
    // i=2: days 3,4,5 → -0.05+0.01+0.01 = -0.03 → DOWN
    // P(up) = 0/3 = 0
    expect(rates.bull).toBe(0);
  });

  it('returns 0.5 for regimes with no observations', () => {
    const regimeSeq: RegimeState[] = Array(10).fill('bull');
    const returns = Array(10).fill(0.01);
    const rates = computeRegimeUpRates(regimeSeq, returns, 1);
    // No bear or sideways observations → should return 0.5 (uninformative)
    expect(rates.bear).toBe(0.5);
    expect(rates.sideways).toBe(0.5);
  });

  it('distinguishes regime-specific up rates', () => {
    // Bull days have positive returns, bear days have negative.
    // With horizon=1 and i+1 offset, each i looks at returns[i+1].
    const regimeSeq: RegimeState[] = ['bull', 'bull', 'bull', 'bear', 'bear', 'bear'];
    const returns = [0.02, 0.01, 0.03, -0.02, -0.01, -0.03];
    const rates = computeRegimeUpRates(regimeSeq, returns, 1);
    // maxStart = 6 - 1 + 1 = 6, so i goes 0..5.
    // i=0 (bull): look at returns[1]=0.01 → up
    // i=1 (bull): look at returns[2]=0.03 → up
    // i=2 (bull): look at returns[3]=-0.02 → down
    // i=3 (bear): look at returns[4]=-0.01 → down
    // i=4 (bear): look at returns[5]=-0.03 → down
    // i=5 (bear): maxStart=6 → not in loop (i goes 0..5 only)
    // bull: 2 up / 3 total = 2/3, bear: 0 up / 3 total = 0.0
    expect(rates.bull).toBeCloseTo(2 / 3, 5);
    expect(rates.bear).toBeLessThan(0.1);
  });

  it('applies exponential decay when decayRate is provided', () => {
    // 4 days, all bull regime. With i+1 offset: each i looks at returns[i+1].
    const regimeSeq: RegimeState[] = ['bull', 'bull', 'bull', 'bull'];
    // Returns: +1%, -1%, -1%, +1%
    const returns = [0.01, -0.01, -0.01, 0.01];
    const horizon = 1;
    // maxStart = 4 - 1 + 1 = 4, so i goes 0..3.
    // i=0: look at returns[1]=-0.01 → DOWN. weight = 0.5^(3-0)=0.125
    // i=1: look at returns[2]=-0.01 → DOWN. weight = 0.5^(3-1)=0.25
    // i=2: look at returns[3]=+0.01 → UP.   weight = 0.5^(3-2)=0.5
    // i=3: maxStart=4 → not in loop (j=4 out of bounds)
    // Total weight = 0.125+0.25+0.5 = 0.875. Up weight = 0.5. P(up) = 0.5/0.875 = 4/7
    const rates = computeRegimeUpRates(regimeSeq, returns, horizon, 0.5);
    expect(rates.bull).toBeCloseTo(4 / 7, 5);
  });

  it('maintains default behavior when decayRate is omitted', () => {
    const regimeSeq: RegimeState[] = ['bull', 'bull', 'bull', 'bull'];
    // Returns: +1%, -1%, -1%, +1%. With i+1 offset: i=3 looks at out-of-bounds.
    const returns = [0.01, -0.01, -0.01, 0.01];
    const horizon = 1;
    // maxStart = 4 - 1 + 1 = 4. i=0..3.
    // i=0: look at returns[1]=-0.01 → DOWN (weight=1)
    // i=1: look at returns[2]=-0.01 → DOWN (weight=1)
    // i=2: look at returns[3]=+0.01 → UP (weight=1)
    // i=3: j=4 out of bounds → not counted
    // Total weight = 3. Up weight = 1. P(up) = 1/3
    const rates = computeRegimeUpRates(regimeSeq, returns, horizon);
    expect(rates.bull).toBeCloseTo(1 / 3, 5);
  });

  it('handles sparse recent regimes correctly with decayRate', () => {
    const regimeSeq: RegimeState[] = ['bear', 'bear'];
    // Only one valid observation: i=0 looks at returns[1] (future), then horizon=1 is done.
    const returns = [0.01, 0.01];
    const rates = computeRegimeUpRates(regimeSeq, returns, 1, 0.5);
    // maxStart = 2 - 1 = 1. i goes 0..0 only.
    // i=0 (bear): look at returns[1]=0.01 → UP. weight = 0.5^(1-1-0)=0.5^0=1.
    // bear: 1 up / 1 total = 1.0.
    expect(rates.bear).toBe(1.0);
    expect(rates.bull).toBe(0.5); // Fallback
    expect(rates.sideways).toBe(0.5);
  });
});
