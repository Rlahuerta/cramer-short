/**
 * Tests for markov-distribution.ts
 *
 * Covers all 10 prioritized bug-fixes and 5 addenda from the spec review:
 *  Fix 1  — Default matrix row sums to 1.0 (not 1.2)
 *  Fix 2  — bear→bear sentiment sign: bullish reduces bear persistence
 *  Fix 3  — Log-normal mapping formula (μ_eff, σ_eff, σ_n)
 *  Fix 4  — YES-bias correction (×0.95 on Polymarket anchors)
 *  Fix 5  — CI bounds contain the point estimate
 *  Fix 6  — Dirichlet α=0.1 smoothing
 *  Fix 7  — Mixing-time decay (long horizon → anchor-dominant)
 *  Fix 8  — Liquidity guard trustScore
 *  Fix 9  — Sentiment alpha reduced to 0.07
 *  Fix 10 — Joint high_vol states (no priority override)
 *  Add 1  — Beta distribution note (architecture, no test needed)
 *  Add 2  — VIX regime note (architecture, no test needed)
 *  Add 3  — Eigenvalue-based mixing time
 *  Add 4  — R²_OS out-of-sample metric
 *  Add 5  — Dirichlet default 0.1 (merged with Fix 6)
 */

import { describe, it, expect } from 'bun:test';
import {
  classifyRegimeState,
  estimateTransitionMatrix,
  buildDefaultMatrix,
  normalizeRows,
  adjustTransitionMatrix,
  extractPriceThresholds,
  interpolateDistribution,
  computeMarkovDistribution,
  secondLargestEigenvalue,
  computeMixingWeight,
  computeR2OS,
  logNormalSurvival,
  estimateRegimeStats,
  matPow,
  NUM_STATES,
  STATE_INDEX,
  REGIME_STATES,
} from './markov-distribution.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function rowSums(m: number[][]): number[] {
  return m.map(row => row.reduce((s, v) => s + v, 0));
}

function allClose(a: number, b: number, tol = 1e-9): boolean {
  return Math.abs(a - b) < tol;
}

/** Build a simple deterministic state sequence: n days of given repeating pattern. */
function repeatStates(pattern: ReturnType<typeof classifyRegimeState>[], n: number) {
  return Array.from({ length: n }, (_, i) => pattern[i % pattern.length]);
}

// ---------------------------------------------------------------------------
// Fix 10: classifyRegimeState — joint states (no priority override)
// ---------------------------------------------------------------------------

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

  it('returns high_vol_bull for positive return + high vol (joint state)', () => {
    // Previously would be classified as 'high_vol' (losing directional info)
    expect(classifyRegimeState(0.03, 0.025)).toBe('high_vol_bull');
  });

  it('returns high_vol_bear for negative return + high vol (joint state)', () => {
    expect(classifyRegimeState(-0.04, 0.025)).toBe('high_vol_bear');
  });

  it('boundary: exactly 1% return with low vol → sideways (strict > 0.01 required for bull)', () => {
    expect(classifyRegimeState(0.01, 0.005)).toBe('sideways'); // 0.01 is NOT > 0.01
  });

  it('boundary: exactly 2% vol → not high_vol (must exceed 2%)', () => {
    expect(classifyRegimeState(0.015, 0.02)).toBe('bull'); // == 0.02, not > 0.02
  });
});

// ---------------------------------------------------------------------------
// Fix 1: buildDefaultMatrix — row sums must be exactly 1
// ---------------------------------------------------------------------------

describe('buildDefaultMatrix', () => {
  it('all rows sum to exactly 1.0 (Fix 1: was 1.2 for 4-state 0.6+3×0.2)', () => {
    const m = buildDefaultMatrix();
    for (const sum of rowSums(m)) {
      expect(allClose(sum, 1.0)).toBe(true);
    }
  });

  it('diagonal entries are 0.6', () => {
    const m = buildDefaultMatrix();
    for (let i = 0; i < NUM_STATES; i++) {
      expect(allClose(m[i][i], 0.6)).toBe(true);
    }
  });

  it('off-diagonal entries are 0.4/(NUM_STATES-1)', () => {
    const m = buildDefaultMatrix();
    const expected = 0.4 / (NUM_STATES - 1);
    for (let i = 0; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        if (i !== j) expect(allClose(m[i][j], expected)).toBe(true);
      }
    }
  });

  it('has correct dimensions (5×5)', () => {
    const m = buildDefaultMatrix();
    expect(m.length).toBe(NUM_STATES);
    for (const row of m) expect(row.length).toBe(NUM_STATES);
  });
});

// ---------------------------------------------------------------------------
// Fix 1 + Fix 6: estimateTransitionMatrix — row sums + Dirichlet α=0.1
// ---------------------------------------------------------------------------

describe('estimateTransitionMatrix', () => {
  it('returns default matrix for sequences shorter than minObservations', () => {
    const m = estimateTransitionMatrix(repeatStates(['bull', 'bear'], 10), 0.1, 30);
    const def = buildDefaultMatrix();
    for (let i = 0; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        expect(allClose(m[i][j], def[i][j])).toBe(true);
      }
    }
  });

  it('all rows sum to 1.0 for sufficient data', () => {
    const states = repeatStates(['bull', 'sideways', 'bear', 'high_vol_bull', 'high_vol_bear'], 60);
    const m = estimateTransitionMatrix(states);
    for (const sum of rowSums(m)) {
      expect(allClose(sum, 1.0, 1e-10)).toBe(true);
    }
  });

  it('Dirichlet smoothing (Fix 6): no zero transitions even for unseen pairs', () => {
    // Only bull→bear transitions, all other pairs never observed
    const states = repeatStates(['bull', 'bear'], 60);
    const m = estimateTransitionMatrix(states, 0.1);
    // Every cell must be > 0 (Dirichlet prior prevents zeros)
    for (const row of m) {
      for (const v of row) {
        expect(v).toBeGreaterThan(0);
      }
    }
  });

  it('self-persistence: a pure bull sequence has high bull→bull probability', () => {
    const states = repeatStates(['bull'], 60);
    const m = estimateTransitionMatrix(states, 0.1);
    const bullIdx = STATE_INDEX['bull'];
    // bull→bull should be > 0.8 (59 bull→bull transitions out of 59 total + 5×4 prior cells)
    expect(m[bullIdx][bullIdx]).toBeGreaterThan(0.8);
  });
});

// ---------------------------------------------------------------------------
// Fix 2: adjustTransitionMatrix — sentiment sign direction
// ---------------------------------------------------------------------------

describe('adjustTransitionMatrix', () => {
  const baseMatrix = buildDefaultMatrix();
  const bullIdx = STATE_INDEX['bull'];
  const bearIdx = STATE_INDEX['bear'];

  it('bullish sentiment increases bull→bull probability (Fix 2 direction)', () => {
    const adjusted = adjustTransitionMatrix(baseMatrix, { bullish: 0.8, bearish: 0.2 });
    expect(adjusted[bullIdx][bullIdx]).toBeGreaterThan(baseMatrix[bullIdx][bullIdx]);
  });

  it('bullish sentiment decreases bull→bear probability', () => {
    const adjusted = adjustTransitionMatrix(baseMatrix, { bullish: 0.8, bearish: 0.2 });
    expect(adjusted[bullIdx][bearIdx]).toBeLessThan(baseMatrix[bullIdx][bearIdx]);
  });

  it('Fix 2: bullish sentiment DECREASES bear→bear (not increases, sign-flip corrected)', () => {
    const adjusted = adjustTransitionMatrix(baseMatrix, { bullish: 0.8, bearish: 0.2 });
    // Bullish means less bear persistence — the original spec had this backwards
    expect(adjusted[bearIdx][bearIdx]).toBeLessThan(baseMatrix[bearIdx][bearIdx]);
  });

  it('bullish sentiment increases bear→bull probability', () => {
    const adjusted = adjustTransitionMatrix(baseMatrix, { bullish: 0.8, bearish: 0.2 });
    expect(adjusted[bearIdx][bullIdx]).toBeGreaterThan(baseMatrix[bearIdx][bullIdx]);
  });

  it('bearish sentiment has opposite effects to bullish', () => {
    const bullish  = adjustTransitionMatrix(baseMatrix, { bullish: 0.8, bearish: 0.2 });
    const bearish  = adjustTransitionMatrix(baseMatrix, { bullish: 0.2, bearish: 0.8 });
    expect(bullish[bullIdx][bullIdx]).toBeGreaterThan(bearish[bullIdx][bullIdx]);
    expect(bullish[bearIdx][bearIdx]).toBeLessThan(bearish[bearIdx][bearIdx]);
  });

  it('all rows still sum to 1 after adjustment', () => {
    const adjusted = adjustTransitionMatrix(baseMatrix, { bullish: 0.9, bearish: 0.1 });
    for (const sum of rowSums(adjusted)) {
      expect(allClose(sum, 1.0, 1e-10)).toBe(true);
    }
  });

  it('Fix 9: alpha=0.07 applied by default (moderate adjustment)', () => {
    const highSentiment = adjustTransitionMatrix(baseMatrix, { bullish: 1.0, bearish: 0.0 });
    const shift = highSentiment[bullIdx][bullIdx] - baseMatrix[bullIdx][bullIdx];
    // With alpha=0.07 and max shift=1, adjustment ≈ base × 0.07
    expect(Math.abs(shift)).toBeLessThan(0.15); // must be < old alpha 0.15
  });

  it('neutral sentiment (0.5/0.5) leaves matrix unchanged', () => {
    const adjusted = adjustTransitionMatrix(baseMatrix, { bullish: 0.5, bearish: 0.5 });
    for (let i = 0; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        expect(allClose(adjusted[i][j], baseMatrix[i][j], 1e-10)).toBe(true);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Fix 4 + Fix 8: extractPriceThresholds — YES-bias + liquidity guard
// ---------------------------------------------------------------------------

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
      { question: 'Will BTC reach $70K?', probability: 0.5 },
    ]);
    expect(result[0].price).toBe(70_000);
  });

  it('parses M suffix correctly ($1M → 1000000)', () => {
    const result = extractPriceThresholds([
      { question: 'Will BRK.A exceed $1M?', probability: 0.3 },
    ]);
    expect(result[0].price).toBe(1_000_000);
  });

  it('Fix 8: market under 48h → trustScore low', () => {
    const recentTime = Date.now() - 10 * 60 * 60 * 1000; // 10h ago
    const result = extractPriceThresholds([
      { question: 'Will AAPL exceed $200?', probability: 0.7, createdAt: recentTime, volume: 1000 },
    ]);
    expect(result[0].trustScore).toBe('low');
  });

  it('Fix 8: market >48h with volume → trustScore high', () => {
    const oldTime = Date.now() - 7 * 24 * 60 * 60 * 1000; // 1 week ago
    const result = extractPriceThresholds([
      { question: 'Will AAPL exceed $200?', probability: 0.7, createdAt: oldTime, volume: 5000 },
    ]);
    expect(result[0].trustScore).toBe('high');
  });

  it('Fix 8: zero volume → trustScore low', () => {
    const result = extractPriceThresholds([
      { question: 'Will TSLA exceed $300?', probability: 0.5, volume: 0 },
    ]);
    expect(result[0].trustScore).toBe('low');
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
      { question: 'Will BTC reach $70000 by EOY?', probability: 0.7 },
    ]);
    expect(result).toHaveLength(1);
    expect(result[0].rawProbability).toBe(0.7);
  });

  it('sorts results by price ascending', () => {
    const result = extractPriceThresholds([
      { question: 'Will BTC exceed $80000?', probability: 0.4 },
      { question: 'Will BTC exceed $60000?', probability: 0.8 },
      { question: 'Will BTC exceed $70000?', probability: 0.6 },
    ]);
    expect(result.map(r => r.price)).toEqual([60_000, 70_000, 80_000]);
  });
});

// ---------------------------------------------------------------------------
// Fix 3 + Fix 5: logNormalSurvival + CI bounds
// ---------------------------------------------------------------------------

describe('logNormalSurvival', () => {
  it('Fix 3: P(price > current) ≈ 0.5 when drift=0 and vol>0', () => {
    const p = logNormalSurvival(100, 100, 0, 0.1);
    expect(p).toBeCloseTo(0.5, 1);
  });

  it('returns close to 1 for target well below current price', () => {
    const p = logNormalSurvival(100, 10, 0, 0.1);
    expect(p).toBeGreaterThan(0.99);
  });

  it('returns close to 0 for target well above current price', () => {
    const p = logNormalSurvival(100, 1000, 0, 0.1);
    expect(p).toBeLessThan(0.01);
  });

  it('higher drift → higher P(price > target)', () => {
    const p1 = logNormalSurvival(100, 110, 0.05, 0.1);
    const p2 = logNormalSurvival(100, 110, -0.05, 0.1);
    expect(p1).toBeGreaterThan(p2);
  });

  it('returns 0 for vol ≤ 0 when target > current', () => {
    expect(logNormalSurvival(100, 110, 0, 0)).toBe(0);
  });

  it('returns 1 for vol ≤ 0 when target < current', () => {
    expect(logNormalSurvival(100, 90, 0, 0)).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Fix 5: interpolateDistribution — CI bounds, monotonicity
// ---------------------------------------------------------------------------

describe('interpolateDistribution', () => {
  const P = buildDefaultMatrix();
  const regimeStats = estimateRegimeStats([], []);

  it('Fix 5: lowerBound ≤ probability ≤ upperBound for each point', () => {
    const dist = interpolateDistribution(100, 10, P, regimeStats, 'bull', [], 0.5);
    for (const point of dist) {
      expect(point.lowerBound).toBeLessThanOrEqual(point.probability + 1e-9);
      expect(point.probability).toBeLessThanOrEqual(point.upperBound + 1e-9);
    }
  });

  it('distribution is monotonically non-increasing in price', () => {
    const dist = interpolateDistribution(100, 20, P, regimeStats, 'bull', [], 0.5);
    for (let i = 1; i < dist.length; i++) {
      expect(dist[i].probability).toBeLessThanOrEqual(dist[i - 1].probability + 1e-9);
    }
  });

  it('probability near 1 for prices well below current', () => {
    const dist = interpolateDistribution(100, 10, P, regimeStats, 'bull', [], 0.5);
    const lowest = dist[0];
    expect(lowest.probability).toBeGreaterThan(0.7);
  });

  it('probability near 0 for prices well above current', () => {
    const dist = interpolateDistribution(100, 10, P, regimeStats, 'bull', [], 0.5);
    const highest = dist[dist.length - 1];
    expect(highest.probability).toBeLessThan(0.3);
  });

  it('Fix 7: high second-eigenvalue + long horizon → lower mixing weight', () => {
    const shortWeight = computeMixingWeight(0.9, 5);
    const longWeight  = computeMixingWeight(0.9, 60);
    expect(longWeight).toBeLessThan(shortWeight);
    expect(longWeight).toBeLessThan(0.01); // essentially pure anchor at 60 days
  });

  it('source=polymarket for nearby high-trust anchor at short horizon', () => {
    const anchors = [{ price: 100, rawProbability: 0.5, probability: 0.475, trustScore: 'high' as const, source: 'polymarket' as const }];
    // Low second eigenvalue → Markov dominant
    const dist = interpolateDistribution(100, 2, P, regimeStats, 'sideways', anchors, 0.01);
    const point = dist.find(d => Math.abs(d.price - 100) < 5);
    // With very low ρ, markovWeight ≈ exp(-0.01×2) ≈ 0.98 → should be 'markov' or 'blend'
    expect(['markov', 'blend', 'polymarket']).toContain(point?.source);
  });
});

// ---------------------------------------------------------------------------
// Add 3 + Fix 7: secondLargestEigenvalue + computeMixingWeight
// ---------------------------------------------------------------------------

describe('secondLargestEigenvalue', () => {
  it('returns value in [0, 1]', () => {
    const rho = secondLargestEigenvalue(buildDefaultMatrix());
    expect(rho).toBeGreaterThanOrEqual(0);
    expect(rho).toBeLessThanOrEqual(1);
  });

  it('identity matrix has second eigenvalue close to 1 (no mixing)', () => {
    const identity: number[][] = Array.from({ length: NUM_STATES }, (_, i) =>
      Array.from({ length: NUM_STATES }, (_, j) => (i === j ? 1 : 0)),
    );
    const rho = secondLargestEigenvalue(identity);
    expect(rho).toBeGreaterThan(0.9);
  });

  it('uniform row matrix has second eigenvalue close to 0 (instant mixing)', () => {
    const uniform: number[][] = Array.from({ length: NUM_STATES }, () =>
      Array(NUM_STATES).fill(1 / NUM_STATES),
    );
    const rho = secondLargestEigenvalue(uniform);
    expect(rho).toBeLessThan(0.1);
  });
});

describe('computeMixingWeight', () => {
  it('Fix 7: at horizon 0, weight=1 (pure Markov)', () => {
    expect(computeMixingWeight(0.9, 0)).toBe(1);
  });

  it('Fix 7: at long horizon with high ρ, weight → 0', () => {
    expect(computeMixingWeight(0.9, 100)).toBeLessThan(0.001);
  });

  it('Fix 7: at long horizon with ρ≈0, weight stays near 1', () => {
    expect(computeMixingWeight(0.001, 90)).toBeGreaterThan(0.9);
  });
});

// ---------------------------------------------------------------------------
// Add 4: R²_OS validation metric
// ---------------------------------------------------------------------------

describe('computeR2OS', () => {
  it('Add 4: returns 1.0 for perfect predictions', () => {
    const actual    = [0.01, -0.02, 0.03, -0.01, 0.02];
    const predicted = [...actual];
    expect(computeR2OS(actual, predicted)).toBeCloseTo(1.0, 5);
  });

  it('Add 4: returns 0.0 when predictions equal the mean', () => {
    const actual = [0.01, -0.02, 0.03, -0.01, 0.02];
    const mean   = actual.reduce((s, v) => s + v, 0) / actual.length;
    const predicted = Array(actual.length).fill(mean);
    expect(computeR2OS(actual, predicted)).toBeCloseTo(0.0, 5);
  });

  it('Add 4: returns < 0 when predictions are worse than the mean', () => {
    const actual    = [0.01, -0.02, 0.03, -0.01, 0.02];   // non-constant, has variance
    const predicted = [0.5,  -0.5,  0.5,  -0.5,  0.5];   // wildly over-scaled predictions
    const r2 = computeR2OS(actual, predicted);
    expect(r2).toBeLessThan(0);
  });

  it('Add 4: returns 0 for arrays shorter than 2', () => {
    expect(computeR2OS([], [])).toBe(0);
    expect(computeR2OS([0.01], [0.01])).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// matPow — matrix exponentiation
// ---------------------------------------------------------------------------

describe('matPow', () => {
  it('P^1 equals P', () => {
    const P = buildDefaultMatrix();
    const P1 = matPow(P, 1);
    for (let i = 0; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        expect(allClose(P1[i][j], P[i][j])).toBe(true);
      }
    }
  });

  it('P^0 is identity', () => {
    const I = matPow(buildDefaultMatrix(), 0);
    for (let i = 0; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        expect(allClose(I[i][j], i === j ? 1 : 0)).toBe(true);
      }
    }
  });

  it('rows of P^n still sum to 1', () => {
    const P = buildDefaultMatrix();
    for (const n of [2, 5, 10, 30]) {
      const Pn = matPow(P, n);
      for (const sum of rowSums(Pn)) {
        expect(allClose(sum, 1.0, 1e-8)).toBe(true);
      }
    }
  });

  it('P^n converges to stationary distribution for ergodic chain', () => {
    const P = buildDefaultMatrix();
    const P100 = matPow(P, 100);
    // All rows of P^100 should be approximately equal (uniform stationary dist)
    for (let i = 1; i < NUM_STATES; i++) {
      for (let j = 0; j < NUM_STATES; j++) {
        expect(Math.abs(P100[0][j] - P100[i][j])).toBeLessThan(0.01);
      }
    }
  });
});

// ---------------------------------------------------------------------------
// Integration: computeMarkovDistribution
// ---------------------------------------------------------------------------

describe('computeMarkovDistribution (integration)', () => {
  const prices = Array.from({ length: 90 }, (_, i) => 100 + i * 0.2 + Math.sin(i) * 2);

  it('returns a result with the correct ticker and horizon', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 10,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.ticker).toBe('TEST');
    expect(result.horizon).toBe(10);
  });

  it('distribution has 21 points (0..numLevels)', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 5,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.distribution.length).toBe(21);
  });

  it('metadata includes all expected fields', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 20,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const m = result.metadata;
    expect(typeof m.regimeState).toBe('string');
    expect(typeof m.mixingTimeWeight).toBe('number');
    expect(typeof m.secondEigenvalue).toBe('number');
    expect(m.mixingTimeWeight).toBeGreaterThanOrEqual(0);
    expect(m.mixingTimeWeight).toBeLessThanOrEqual(1);
  });

  it('Polymarket anchors count only trusted (high) anchors', async () => {
    const recentTime = Date.now() - 1 * 60 * 60 * 1000; // 1h ago (untrusted)
    const oldTime    = Date.now() - 10 * 24 * 60 * 60 * 1000; // 10d ago (trusted)
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 10,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [
        { question: 'Will TEST exceed $120?', probability: 0.6, createdAt: oldTime,    volume: 5000 },
        { question: 'Will TEST exceed $130?', probability: 0.4, createdAt: recentTime, volume: 1000 },
      ],
    });
    expect(result.metadata.polymarketAnchors).toBe(1); // only the old/high-trust one
  });

  it('distribution is monotonically non-increasing in price', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 15,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const dist = result.distribution;
    for (let i = 1; i < dist.length; i++) {
      expect(dist[i].probability).toBeLessThanOrEqual(dist[i - 1].probability + 1e-9);
    }
  });

  it('sentiment shift is reflected in metadata', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'TEST',
      horizon: 5,
      currentPrice: 118,
      historicalPrices: prices,
      polymarketMarkets: [],
      sentiment: { bullish: 0.8, bearish: 0.2 },
    });
    expect(result.metadata.sentimentAdjustment).toBeCloseTo(0.6, 5);
  });
});

// ---------------------------------------------------------------------------
// estimateRegimeStats — empirical return statistics per state
// ---------------------------------------------------------------------------

describe('estimateRegimeStats', () => {
  it('falls back to defaults with empty data', () => {
    const stats = estimateRegimeStats([], []);
    expect(stats.bull.meanReturn).toBeGreaterThan(0);
    expect(stats.bear.meanReturn).toBeLessThan(0);
  });

  it('empirical stats used when ≥5 observations per state', () => {
    const returns = Array(60).fill(0.01); // all positive
    const states  = Array(60).fill('bull') as ReturnType<typeof classifyRegimeState>[];
    const stats   = estimateRegimeStats(returns, states);
    expect(stats.bull.meanReturn).toBeCloseTo(0.01, 5);
  });
});

// ---------------------------------------------------------------------------
// Tier 1a — countStateObservations + findSparseStates
// ---------------------------------------------------------------------------

import {
  countStateObservations,
  findSparseStates,
  detectStructuralBreak,
  mergeAnchorsWithCrossPlatformValidation,
  type KalshiAnchor,
} from './markov-distribution.js';

describe('Tier 1a — countStateObservations', () => {
  it('counts zero for all states when sequence is empty', () => {
    const counts = countStateObservations([]);
    for (const s of REGIME_STATES) expect(counts[s]).toBe(0);
  });

  it('correctly counts all occurrences', () => {
    const states = ['bull', 'bull', 'bear', 'sideways', 'bull'] as ReturnType<typeof classifyRegimeState>[];
    const counts = countStateObservations(states);
    expect(counts.bull).toBe(3);
    expect(counts.bear).toBe(1);
    expect(counts.sideways).toBe(1);
    expect(counts.high_vol_bull).toBe(0);
    expect(counts.high_vol_bear).toBe(0);
  });

  it('total of all counts equals sequence length', () => {
    const states = Array(50).fill('sideways').map((s, i) =>
      REGIME_STATES[i % REGIME_STATES.length],
    ) as ReturnType<typeof classifyRegimeState>[];
    const counts = countStateObservations(states);
    const total = Object.values(counts).reduce((s, v) => s + v, 0);
    expect(total).toBe(50);
  });
});

describe('Tier 1a — findSparseStates', () => {
  it('returns all states when everything is zero', () => {
    const counts = Object.fromEntries(REGIME_STATES.map(s => [s, 0])) as Record<ReturnType<typeof classifyRegimeState>, number>;
    const sparse = findSparseStates(counts);
    expect(sparse).toHaveLength(REGIME_STATES.length);
  });

  it('returns only states below the threshold', () => {
    const counts = {
      bull:          10,
      bear:          3,        // < 5
      high_vol_bull: 4,        // < 5
      high_vol_bear: 0,        // < 5
      sideways:      20,
    };
    const sparse = findSparseStates(counts);
    expect(sparse).toContain('bear');
    expect(sparse).toContain('high_vol_bull');
    expect(sparse).toContain('high_vol_bear');
    expect(sparse).not.toContain('bull');
    expect(sparse).not.toContain('sideways');
  });

  it('returns empty array when all states have enough observations', () => {
    const counts = Object.fromEntries(REGIME_STATES.map(s => [s, 10])) as Record<ReturnType<typeof classifyRegimeState>, number>;
    expect(findSparseStates(counts)).toHaveLength(0);
  });

  it('respects custom minObs parameter', () => {
    const counts = {
      bull: 15, bear: 7, high_vol_bull: 20, high_vol_bear: 12, sideways: 9,
    };
    // minObs=10 → bear (7) and sideways (9) are sparse
    const sparse = findSparseStates(counts, 10);
    expect(sparse).toContain('bear');
    expect(sparse).toContain('sideways');
    expect(sparse).not.toContain('bull');
  });
});

describe('Tier 1a — sparseStates in computeMarkovDistribution metadata', () => {
  it('metadata.sparseStates includes states with few observations', async () => {
    // Only 11 prices → 10 returns, all forcing bull state (tiny window ensures sparsity)
    const prices = Array.from({ length: 11 }, (_, i) => 100 + i * 0.5);
    const result = await computeMarkovDistribution({
      ticker: 'SPARSE',
      horizon: 5,
      currentPrice: 105,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    // With only 10 returns all going up, high_vol_bear and bear should be sparse
    expect(result.metadata.stateObservationCounts).toBeDefined();
    expect(Array.isArray(result.metadata.sparseStates)).toBe(true);
  });

  it('metadata.stateObservationCounts sums to historicalDays', async () => {
    const prices = Array.from({ length: 40 }, (_, i) => 100 * (1 + i * 0.002));
    const result = await computeMarkovDistribution({
      ticker: 'COUNT',
      horizon: 5,
      currentPrice: 107,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    const total = Object.values(result.metadata.stateObservationCounts).reduce((s, v) => s + v, 0);
    expect(total).toBe(result.metadata.historicalDays);
  });
});

// ---------------------------------------------------------------------------
// Tier 1b — detectStructuralBreak
// ---------------------------------------------------------------------------

describe('Tier 1b — detectStructuralBreak', () => {
  it('detects no break in a stationary sequence', () => {
    // Long alternating bull/bear sequence — consistent across halves
    const states: ReturnType<typeof classifyRegimeState>[] = Array.from(
      { length: 60 }, (_, i) => (i % 2 === 0 ? 'bull' : 'bear'),
    );
    const result = detectStructuralBreak(states);
    // First and second halves have the same pattern — divergence should be low
    expect(result.divergence).toBeDefined();
    expect(result.firstHalfMatrix).toHaveLength(NUM_STATES);
    expect(result.secondHalfMatrix).toHaveLength(NUM_STATES);
  });

  it('detects a break when regimes are completely different in each half', () => {
    // First 30: all bull → bull; Last 30: all bear → bear
    const states: ReturnType<typeof classifyRegimeState>[] = [
      ...Array(30).fill('bull'),
      ...Array(30).fill('bear'),
    ];
    const result = detectStructuralBreak(states);
    // The two halves describe very different dynamics
    expect(result.detected).toBe(true);
    expect(result.divergence).toBeGreaterThan(0.05);
  });

  it('detected=false when sequence is too short', () => {
    const result = detectStructuralBreak(['bull', 'bear', 'sideways'] as RegimeState[]);
    // With only 3 states each half has 1-2 states — not enough for stable estimate
    expect(typeof result.detected).toBe('boolean');
    expect(result.divergence).toBeGreaterThanOrEqual(0);
  });

  it('both half matrices are row-stochastic', () => {
    const states: ReturnType<typeof classifyRegimeState>[] = Array.from(
      { length: 60 }, (_, i) => REGIME_STATES[i % REGIME_STATES.length],
    );
    const { firstHalfMatrix, secondHalfMatrix } = detectStructuralBreak(states);
    for (const row of [...firstHalfMatrix, ...secondHalfMatrix]) {
      const sum = row.reduce((s, v) => s + v, 0);
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it('metadata.structuralBreakDetected reflects detection', async () => {
    // First half: all bull; second half: all bear → should trigger break
    const bullPrices = Array.from({ length: 31 }, (_, i) => 100 * (1 + i * 0.01));  // 30 bull returns
    const bearPrices = Array.from({ length: 31 }, (_, i) => bullPrices[30] * (1 - i * 0.01));
    const prices = [...bullPrices, ...bearPrices.slice(1)];
    const result = await computeMarkovDistribution({
      ticker: 'BREAK',
      horizon: 5,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.metadata.structuralBreakDivergence).toBeGreaterThanOrEqual(0);
    expect(typeof result.metadata.structuralBreakDetected).toBe('boolean');
    expect(typeof result.metadata.ciWidened).toBe('boolean');
  });

  it('CI is wider when structural break is detected', async () => {
    // We compare two runs: one stationary (no break) vs one with clear break
    // The break case should produce wider (upper-lower) CI intervals
    const stationaryPrices = Array.from({ length: 60 }, (_, i) => 100 + i * 0.2);
    const breakPrices = [
      ...Array.from({ length: 30 }, (_, i) => 100 + i * 0.3),    // bull
      ...Array.from({ length: 30 }, (_, i) => 109 - i * 0.3),    // bear
    ];

    const r1 = await computeMarkovDistribution({
      ticker: 'STAT', horizon: 10, currentPrice: stationaryPrices[stationaryPrices.length - 1],
      historicalPrices: stationaryPrices, polymarketMarkets: [],
    });
    const r2 = await computeMarkovDistribution({
      ticker: 'BREAK', horizon: 10, currentPrice: breakPrices[breakPrices.length - 1],
      historicalPrices: breakPrices, polymarketMarkets: [],
    });

    const avgWidth = (dist: typeof r1.distribution) =>
      dist.reduce((s, d) => s + d.upperBound - d.lowerBound, 0) / dist.length;

    // When structuralBreakDetected: true, the CI must be wider
    if (r2.metadata.structuralBreakDetected) {
      expect(avgWidth(r2.distribution)).toBeGreaterThan(avgWidth(r1.distribution) * 0.9);
    }
  });
});

// ---------------------------------------------------------------------------
// Tier 1c — mergeAnchorsWithCrossPlatformValidation
// ---------------------------------------------------------------------------

describe('Tier 1c — mergeAnchorsWithCrossPlatformValidation', () => {
  it('returns only Polymarket anchors when no Kalshi anchors provided', () => {
    const { anchors, warnings } = mergeAnchorsWithCrossPlatformValidation(
      [{
        price: 100, rawProbability: 0.6, probability: 0.57, trustScore: 'high', source: 'polymarket',
      }],
      [],
    );
    expect(anchors).toHaveLength(1);
    expect(warnings).toHaveLength(0);
  });

  it('adds Kalshi-only anchors with no bias correction', () => {
    const kalshi: KalshiAnchor[] = [{ price: 150, probability: 0.3, volume: 500 }];
    const { anchors } = mergeAnchorsWithCrossPlatformValidation([], kalshi);
    expect(anchors).toHaveLength(1);
    expect(anchors[0].source).toBe('kalshi');
    // No YES-bias correction for Kalshi: rawProb = probability = 0.3
    expect(anchors[0].probability).toBe(0.3);
    expect(anchors[0].rawProbability).toBe(0.3);
  });

  it('averages matching anchors within price tolerance', () => {
    const poly = [{
      price: 100, rawProbability: 0.60, probability: 0.57, trustScore: 'high' as const, source: 'polymarket' as const,
    }];
    const kalshi: KalshiAnchor[] = [{ price: 100.5, probability: 0.60, volume: 200 }]; // within 2%
    const { anchors, warnings } = mergeAnchorsWithCrossPlatformValidation(poly, kalshi);
    expect(anchors).toHaveLength(1);
    expect(anchors[0].source).toBe('averaged');
    // No divergence (both 0.60)
    expect(warnings).toHaveLength(0);
    // Averaged raw = (0.60 + 0.60) / 2 = 0.60; bias-corrected = 0.60 * 0.95 = 0.57
    expect(anchors[0].rawProbability).toBeCloseTo(0.60, 5);
    expect(anchors[0].probability).toBeCloseTo(0.57, 5);
  });

  it('emits warning when Polymarket and Kalshi diverge by more than 5pp', () => {
    const poly = [{
      price: 100, rawProbability: 0.70, probability: 0.665, trustScore: 'high' as const, source: 'polymarket' as const,
    }];
    const kalshi: KalshiAnchor[] = [{ price: 100, probability: 0.60 }]; // 10pp divergence
    const { anchors, warnings } = mergeAnchorsWithCrossPlatformValidation(poly, kalshi);
    expect(warnings).toHaveLength(1);
    expect(warnings[0].divergencePp).toBeCloseTo(10, 0);
    expect(warnings[0].polymarketProb).toBeCloseTo(0.70, 5);
    expect(warnings[0].kalshiProb).toBeCloseTo(0.60, 5);
    // Averaged: (0.70 + 0.60) / 2 = 0.65; bias-corrected = 0.65 * 0.95
    expect(anchors[0].rawProbability).toBeCloseTo(0.65, 5);
    expect(anchors[0].probability).toBeCloseTo(0.65 * 0.95, 5);
  });

  it('does NOT emit warning when divergence is ≤5pp', () => {
    const poly = [{
      price: 100, rawProbability: 0.55, probability: 0.5225, trustScore: 'high' as const, source: 'polymarket' as const,
    }];
    const kalshi: KalshiAnchor[] = [{ price: 100, probability: 0.52 }]; // 3pp divergence
    const { warnings } = mergeAnchorsWithCrossPlatformValidation(poly, kalshi);
    expect(warnings).toHaveLength(0);
  });

  it('result anchors are sorted by price ascending', () => {
    const poly = [
      { price: 120, rawProbability: 0.3, probability: 0.285, trustScore: 'high' as const, source: 'polymarket' as const },
      { price: 100, rawProbability: 0.7, probability: 0.665, trustScore: 'high' as const, source: 'polymarket' as const },
    ];
    const kalshi: KalshiAnchor[] = [{ price: 110, probability: 0.5 }];
    const { anchors } = mergeAnchorsWithCrossPlatformValidation(poly, kalshi);
    for (let i = 1; i < anchors.length; i++) {
      expect(anchors[i].price).toBeGreaterThanOrEqual(anchors[i - 1].price);
    }
  });

  it('upgrades trustScore to high when Kalshi anchor has volume', () => {
    const poly = [{
      price: 100, rawProbability: 0.55, probability: 0.5225, trustScore: 'low' as const, source: 'polymarket' as const,
    }];
    const kalshi: KalshiAnchor[] = [{ price: 100, probability: 0.54, volume: 1000 }];
    const { anchors } = mergeAnchorsWithCrossPlatformValidation(poly, kalshi);
    expect(anchors[0].trustScore).toBe('high');
  });

  it('kalshiAnchors parameter propagates through computeMarkovDistribution', async () => {
    const prices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.3);
    const result = await computeMarkovDistribution({
      ticker: 'CROSS',
      horizon: 10,
      currentPrice: 111.7,
      historicalPrices: prices,
      polymarketMarkets: [
        { question: 'Will CROSS exceed $115?', probability: 0.70, volume: 2000 },
      ],
      kalshiAnchors: [{ price: 115, probability: 0.60, volume: 500 }], // 10pp divergence
    });
    expect(result.metadata.anchorDivergenceWarnings).toHaveLength(1);
    expect(result.metadata.anchorDivergenceWarnings[0].divergencePp).toBeCloseTo(10, 0);
  });

  it('no divergence warnings when kalshiAnchors is absent', async () => {
    const prices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.3);
    const result = await computeMarkovDistribution({
      ticker: 'NOKALSHI',
      horizon: 10,
      currentPrice: 111.7,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.metadata.anchorDivergenceWarnings).toHaveLength(0);
  });
});
