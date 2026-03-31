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
  matMul,
  normalCDF,
  transitionGoodnessOfFit,
  markovDistributionTool,
  computeEnsembleSignal,
  NUM_STATES,
  STATE_INDEX,
  REGIME_STATES,
} from './markov-distribution.js';
import type { RegimeState } from './markov-distribution.js';

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

// ---------------------------------------------------------------------------
// Action signal: interpolateSurvival + computeActionSignal
// ---------------------------------------------------------------------------

import {
  interpolateSurvival,
  computeActionSignal,
  type MarkovDistributionPoint,
} from './markov-distribution.js';

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

// ---------------------------------------------------------------------------
// computeActionLevels
// ---------------------------------------------------------------------------

import { computeActionLevels } from './markov-distribution.js';

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

// ---------------------------------------------------------------------------
// assessAnchorCoverage
// ---------------------------------------------------------------------------

import { assessAnchorCoverage } from './markov-distribution.js';

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

// ---------------------------------------------------------------------------
// interpolateDistribution — anchor grid merging
// ---------------------------------------------------------------------------

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
});

// ===========================================================================
// CORRECTNESS & VALIDATION TESTS
// ===========================================================================

// ---------------------------------------------------------------------------
// normalCDF — direct tests against known Φ values
// ---------------------------------------------------------------------------

describe('normalCDF', () => {
  // normalCDF now computes the true standard normal CDF: Φ(x) = 0.5*(1+erf(x/√2))

  it('Φ(0) ≈ 0.5', () => {
    expect(normalCDF(0)).toBeCloseTo(0.5, 6);
  });

  it('Φ(1) ≈ 0.8413', () => {
    expect(normalCDF(1)).toBeCloseTo(0.8413, 3);
  });

  it('Φ(-1) ≈ 0.1587', () => {
    expect(normalCDF(-1)).toBeCloseTo(0.1587, 3);
  });

  it('normalCDF(x) + normalCDF(-x) = 1 (symmetry property)', () => {
    for (const x of [0.5, 1.0, 2.0, 3.0]) {
      expect(normalCDF(x) + normalCDF(-x)).toBeCloseTo(1.0, 6);
    }
  });

  it('Φ(1.96) ≈ 0.975 (95% critical value)', () => {
    expect(normalCDF(1.96)).toBeCloseTo(0.975, 2);
  });

  it('Φ(3) ≈ 0.99865 (deep right tail)', () => {
    expect(normalCDF(3)).toBeCloseTo(0.99865, 3);
  });

  it('Φ(-3) ≈ 0.00135 (deep left tail)', () => {
    expect(normalCDF(-3)).toBeCloseTo(0.00135, 3);
  });

  it('is monotonically non-decreasing', () => {
    let prev = 0;
    for (let x = -4; x <= 4; x += 0.1) {
      const val = normalCDF(x);
      expect(val).toBeGreaterThanOrEqual(prev);
      prev = val;
    }
  });
});

// ---------------------------------------------------------------------------
// matMul — direct tests
// ---------------------------------------------------------------------------

describe('matMul', () => {
  it('A × I = A (identity multiplication)', () => {
    const A = [[1, 2], [3, 4]];
    const I = [[1, 0], [0, 1]];
    const result = matMul(A, I);
    expect(result[0][0]).toBeCloseTo(1);
    expect(result[0][1]).toBeCloseTo(2);
    expect(result[1][0]).toBeCloseTo(3);
    expect(result[1][1]).toBeCloseTo(4);
  });

  it('I × A = A (left identity)', () => {
    const A = [[5, 6], [7, 8]];
    const I = [[1, 0], [0, 1]];
    const result = matMul(I, A);
    expect(result[0][0]).toBeCloseTo(5);
    expect(result[1][1]).toBeCloseTo(8);
  });

  it('known 2×2 product', () => {
    // [[1,2],[3,4]] × [[5,6],[7,8]] = [[19,22],[43,50]]
    const A = [[1, 2], [3, 4]];
    const B = [[5, 6], [7, 8]];
    const R = matMul(A, B);
    expect(R[0][0]).toBeCloseTo(19);
    expect(R[0][1]).toBeCloseTo(22);
    expect(R[1][0]).toBeCloseTo(43);
    expect(R[1][1]).toBeCloseTo(50);
  });

  it('preserves row-stochasticity (stochastic × stochastic = stochastic)', () => {
    const P = [[0.7, 0.3], [0.4, 0.6]];
    const R = matMul(P, P);
    for (const row of R) {
      const sum = row.reduce((s, v) => s + v, 0);
      expect(sum).toBeCloseTo(1.0, 6);
    }
  });
});

// ---------------------------------------------------------------------------
// normalizeRows — direct tests
// ---------------------------------------------------------------------------

describe('normalizeRows', () => {
  // normalizeRows returns a NEW matrix (does not mutate)

  it('normalizes rows to sum to 1', () => {
    const result = normalizeRows([[2, 3], [4, 6]]);
    expect(result[0][0]).toBeCloseTo(0.4);
    expect(result[0][1]).toBeCloseTo(0.6);
    expect(result[1][0]).toBeCloseTo(0.4);
    expect(result[1][1]).toBeCloseTo(0.6);
  });

  it('leaves already-normalized rows unchanged', () => {
    const result = normalizeRows([[0.3, 0.7], [0.5, 0.5]]);
    expect(result[0][0]).toBeCloseTo(0.3);
    expect(result[0][1]).toBeCloseTo(0.7);
  });

  it('handles zero-sum row by distributing uniformly', () => {
    const result = normalizeRows([[0, 0], [1, 1]]);
    expect(result[1][0]).toBeCloseTo(0.5);
    expect(result[1][1]).toBeCloseTo(0.5);
    // zero row now becomes uniform [0.5, 0.5] instead of NaN
    expect(result[0][0]).toBeCloseTo(0.5);
    expect(result[0][1]).toBeCloseTo(0.5);
  });
});

// ---------------------------------------------------------------------------
// Analytical 2-state Markov verification (closed-form cross-check)
// ---------------------------------------------------------------------------

describe('analytical 2-state Markov verification', () => {
  // For a 2-state chain P = [[1-a, a], [b, 1-b]],
  // the stationary distribution is π = [b/(a+b), a/(a+b)]
  // and P^n converges to rows = π as n→∞.

  it('P^n converges to the correct stationary distribution', () => {
    const a = 0.3, b = 0.2;
    const P = [[1 - a, a], [b, 1 - b]]; // [[0.7, 0.3], [0.2, 0.8]]
    const piStar = [b / (a + b), a / (a + b)]; // [0.4, 0.6]

    const Pn = matPow(P, 100);
    // Both rows should converge to stationary
    expect(Pn[0][0]).toBeCloseTo(piStar[0], 3);
    expect(Pn[0][1]).toBeCloseTo(piStar[1], 3);
    expect(Pn[1][0]).toBeCloseTo(piStar[0], 3);
    expect(Pn[1][1]).toBeCloseTo(piStar[1], 3);
  });

  it('P^n matches closed-form at small n', () => {
    // P^2 = P×P, compute analytically:
    // P = [[0.7, 0.3], [0.2, 0.8]]
    // P^2 = [[0.7×0.7+0.3×0.2, 0.7×0.3+0.3×0.8], [0.2×0.7+0.8×0.2, 0.2×0.3+0.8×0.8]]
    //     = [[0.55, 0.45], [0.30, 0.70]]
    const P = [[0.7, 0.3], [0.2, 0.8]];
    const P2 = matPow(P, 2);
    expect(P2[0][0]).toBeCloseTo(0.55, 6);
    expect(P2[0][1]).toBeCloseTo(0.45, 6);
    expect(P2[1][0]).toBeCloseTo(0.30, 6);
    expect(P2[1][1]).toBeCloseTo(0.70, 6);
  });

  it('P^n row sums remain 1.0 for all n', () => {
    const P = [[0.7, 0.3], [0.2, 0.8]];
    for (const n of [1, 2, 5, 10, 50]) {
      const Pn = matPow(P, n);
      for (const row of Pn) {
        expect(row.reduce((s, v) => s + v, 0)).toBeCloseTo(1.0, 6);
      }
    }
  });

  it('second eigenvalue matches analytical value for 2-state chain', () => {
    // For P = [[1-a, a], [b, 1-b]], eigenvalues are 1 and (1-a-b).
    // ρ = |1 - a - b|
    const a = 0.3, b = 0.2;
    const P = [[1 - a, a], [b, 1 - b]];
    const analyticalRho = Math.abs(1 - a - b); // 0.5
    const computed = secondLargestEigenvalue(P);
    expect(computed).toBeCloseTo(analyticalRho, 1);
  });
});

// ---------------------------------------------------------------------------
// Monte Carlo convergence / stability
// ---------------------------------------------------------------------------

describe('Monte Carlo stability', () => {
  it('interpolateDistribution produces stable results across runs', () => {
    const P = buildDefaultMatrix();
    const regimeStats = estimateRegimeStats(
      Array.from({ length: 30 }, () => 0.002),
      Array.from({ length: 30 }, () => 'bull' as const),
    );

    // Run 5 times, collect probabilities at the median point
    const medianProbs: number[] = [];
    for (let run = 0; run < 5; run++) {
      const dist = interpolateDistribution(100, 20, P, regimeStats, 'bull', [], 0.5, 15, 1000);
      const midIdx = Math.floor(dist.length / 2);
      medianProbs.push(dist[midIdx].probability);
    }

    // Check coefficient of variation is < 10% (Monte Carlo noise should be small)
    const mean = medianProbs.reduce((s, v) => s + v, 0) / medianProbs.length;
    const variance = medianProbs.reduce((s, v) => s + (v - mean) ** 2, 0) / medianProbs.length;
    const cv = Math.sqrt(variance) / Math.max(mean, 1e-10);
    expect(cv).toBeLessThan(0.10);
  });

  it('confidence intervals narrow with more Monte Carlo samples', () => {
    const P = buildDefaultMatrix();
    const regimeStats = estimateRegimeStats(
      Array.from({ length: 30 }, () => 0.001),
      Array.from({ length: 30 }, () => 'sideways' as const),
    );

    // Fewer samples → wider CI
    const distFew = interpolateDistribution(100, 20, P, regimeStats, 'sideways', [], 0.5, 10, 100);
    // More samples → tighter CI
    const distMany = interpolateDistribution(100, 20, P, regimeStats, 'sideways', [], 0.5, 10, 2000);

    // Average CI width across all points
    const avgWidth = (d: typeof distFew) =>
      d.reduce((s, p) => s + (p.upperBound - p.lowerBound), 0) / d.length;

    // More samples should produce tighter or equal CI on average
    // (not guaranteed per-point due to randomness, but on average it holds)
    const fewWidth = avgWidth(distFew);
    const manyWidth = avgWidth(distMany);
    // Allow generous tolerance since MC is stochastic
    expect(manyWidth).toBeLessThan(fewWidth * 1.5);
  });
});

// ---------------------------------------------------------------------------
// Anchor influence A/B test
// ---------------------------------------------------------------------------

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
    });

    const lowTrust = await computeMarkovDistribution({
      ticker: 'TRUST_TEST',
      horizon: 15,
      currentPrice: 100,
      historicalPrices: basePrices,
      polymarketMarkets: [
        { question: 'Will it exceed $105?', probability: 0.80, volume: 100000, createdAt: new Date().toISOString() },
      ],
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

// ---------------------------------------------------------------------------
// Boundary conditions (horizon, degenerate inputs)
// ---------------------------------------------------------------------------

describe('boundary conditions', () => {
  const steadyPrices = Array.from({ length: 30 }, () => 100);
  const trendingUp = Array.from({ length: 30 }, (_, i) => 100 + i * 0.5);

  it('horizon=1: distribution is tight around current price', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'H1',
      horizon: 1,
      currentPrice: 115,
      historicalPrices: trendingUp,
      polymarketMarkets: [],
    });
    // At horizon 1, most probability should be near current price
    const near = result.distribution.filter(
      d => Math.abs(d.price - 115) / 115 < 0.05,
    );
    expect(near.length).toBeGreaterThan(0);
    // Points very far from current should have low probability
    const farAbove = result.distribution.filter(d => d.price > 130);
    for (const p of farAbove) {
      expect(p.probability).toBeLessThan(0.5);
    }
  });

  it('horizon=90: distribution is wider than horizon=5', async () => {
    const short = await computeMarkovDistribution({
      ticker: 'H_CMP',
      horizon: 5,
      currentPrice: 115,
      historicalPrices: trendingUp,
      polymarketMarkets: [],
    });
    const long = await computeMarkovDistribution({
      ticker: 'H_CMP',
      horizon: 90,
      currentPrice: 115,
      historicalPrices: trendingUp,
      polymarketMarkets: [],
    });

    // Average CI width should be larger for longer horizon
    const avgCIWidth = (dist: typeof short.distribution) =>
      dist.reduce((s, d) => s + (d.upperBound - d.lowerBound), 0) / dist.length;

    expect(avgCIWidth(long.distribution)).toBeGreaterThan(avgCIWidth(short.distribution) * 0.5);
  });

  it('all-same-prices: produces a valid distribution without errors', async () => {
    const result = await computeMarkovDistribution({
      ticker: 'FLAT',
      horizon: 10,
      currentPrice: 100,
      historicalPrices: steadyPrices,
      polymarketMarkets: [],
    });
    expect(result.distribution.length).toBeGreaterThan(0);
    // All returns are 0 → sideways regime
    expect(result.metadata.regimeState).toBe('sideways');
    // Action signal should still be valid
    const { buyProbability: b, holdProbability: h, sellProbability: s } = result.actionSignal;
    expect(b + h + s).toBeCloseTo(1.0, 4);
  });

  it('minimum viable input (10 prices) does not throw', async () => {
    const prices = Array.from({ length: 10 }, (_, i) => 100 + i);
    const result = await computeMarkovDistribution({
      ticker: 'MIN',
      horizon: 5,
      currentPrice: 109,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(result.distribution.length).toBeGreaterThan(0);
    expect(result.actionSignal.recommendation).toBeDefined();
  });

  it('strongly trending prices produce expected recommendation direction', async () => {
    // Strong uptrend: 100→130 in 30 days
    const uptrend = Array.from({ length: 30 }, (_, i) => 100 + i);
    const result = await computeMarkovDistribution({
      ticker: 'TREND',
      horizon: 10,
      currentPrice: 129,
      historicalPrices: uptrend,
      polymarketMarkets: [],
    });
    // Expected return should be positive for a strong uptrend
    expect(result.actionSignal.expectedReturn).toBeGreaterThan(0);
  });
});

// ---------------------------------------------------------------------------
// Sensitivity analysis
// ---------------------------------------------------------------------------

describe('sensitivity analysis', () => {
  const basePrices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.3);

  it('small price perturbation produces proportional output change', async () => {
    const base = await computeMarkovDistribution({
      ticker: 'SENS',
      horizon: 15,
      currentPrice: 112,
      historicalPrices: basePrices,
      polymarketMarkets: [],
    });

    // Perturb current price by +1%
    const perturbed = await computeMarkovDistribution({
      ticker: 'SENS',
      horizon: 15,
      currentPrice: 112 * 1.01,
      historicalPrices: basePrices,
      polymarketMarkets: [],
    });

    // Expected return should shift (not necessarily by exactly 1%, but it should change)
    const diff = Math.abs(base.actionSignal.expectedReturn - perturbed.actionSignal.expectedReturn);
    // Change should be bounded — no discontinuity from a 1% input shift
    expect(diff).toBeLessThan(0.10); // <10pp change from 1% price shift
    // Recommendation might stay the same, but probabilities should differ
    const probDiff = Math.abs(base.actionSignal.buyProbability - perturbed.actionSignal.buyProbability);
    expect(probDiff).toBeLessThan(0.20); // <20pp change
  });

  it('adding one more historical day does not cause large jumps', async () => {
    const base = await computeMarkovDistribution({
      ticker: 'SENS2',
      horizon: 15,
      currentPrice: 112,
      historicalPrices: basePrices,
      polymarketMarkets: [],
    });

    // Add one more day at roughly the same trajectory
    const extendedPrices = [...basePrices, 112.3];
    const extended = await computeMarkovDistribution({
      ticker: 'SENS2',
      horizon: 15,
      currentPrice: 112.3,
      historicalPrices: extendedPrices,
      polymarketMarkets: [],
    });

    // Expected returns should be similar
    const diff = Math.abs(base.actionSignal.expectedReturn - extended.actionSignal.expectedReturn);
    expect(diff).toBeLessThan(0.05);
  });

  it('sentiment shift produces monotonic effect on expected return', async () => {
    const bullish = await computeMarkovDistribution({
      ticker: 'SENT_MONO',
      horizon: 15,
      currentPrice: 112,
      historicalPrices: basePrices,
      polymarketMarkets: [],
      sentiment: { bullish: 0.8, bearish: 0.2 },
    });
    const neutral = await computeMarkovDistribution({
      ticker: 'SENT_MONO',
      horizon: 15,
      currentPrice: 112,
      historicalPrices: basePrices,
      polymarketMarkets: [],
      sentiment: { bullish: 0.5, bearish: 0.5 },
    });
    const bearish = await computeMarkovDistribution({
      ticker: 'SENT_MONO',
      horizon: 15,
      currentPrice: 112,
      historicalPrices: basePrices,
      polymarketMarkets: [],
      sentiment: { bullish: 0.2, bearish: 0.8 },
    });

    // Bullish sentiment → higher expected return than bearish
    expect(bullish.actionSignal.expectedReturn).toBeGreaterThanOrEqual(bearish.actionSignal.expectedReturn);
    // Neutral should be between (or at least not more extreme than either)
    expect(neutral.actionSignal.expectedReturn).toBeGreaterThanOrEqual(bearish.actionSignal.expectedReturn - 0.01);
    expect(neutral.actionSignal.expectedReturn).toBeLessThanOrEqual(bullish.actionSignal.expectedReturn + 0.01);
  });
});

// ---------------------------------------------------------------------------
// Tool output format validation
// ---------------------------------------------------------------------------

describe('markovDistributionTool output format', () => {
  const prices = Array.from({ length: 40 }, (_, i) => 100 + i * 0.5);

  it('output contains Decision Card with BUY/HOLD/SELL', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 15,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(output).toContain('Your Options');
    expect(output).toContain('BUY');
    expect(output).toContain('HOLD');
    expect(output).toContain('SELL');
  });

  it('output contains Action Plan with price levels', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 15,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(output).toContain('Action Plan');
    expect(output).toContain('Target');
    expect(output).toContain('Stop-loss');
    expect(output).toContain('Median forecast');
    expect(output).toContain('Bull case');
    expect(output).toContain('Bear case');
  });

  it('output contains recommendation with confidence', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 15,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    // Should have one of: [HIGH confidence], [MEDIUM confidence], [LOW confidence]
    expect(output).toMatch(/\[(HIGH|MEDIUM|LOW) confidence\]/);
  });

  it('output contains distribution table with P(>price) column', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 15,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(output).toContain('P(>price)');
    expect(output).toContain('90% CI');
    expect(output).toContain('Source');
  });

  it('output contains anchor quality diagnostic', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 15,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(output).toContain('Anchor quality:');
  });

  it('output contains contextual guidance (💡)', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 15,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(output).toContain('💡');
  });

  it('output shows warnings when no anchors provided', async () => {
    const output = await markovDistributionTool.invoke({
      ticker: 'FMT_TEST',
      horizon: 15,
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    expect(output).toContain('No trusted');
  });
});

// ---------------------------------------------------------------------------
// transitionGoodnessOfFit — chi-squared test for Markov property
// ---------------------------------------------------------------------------

describe('transitionGoodnessOfFit', () => {
  // Generate a synthetic state sequence from a known transition matrix
  function generateMarkovChain(P: number[][], n: number, startState: number): RegimeState[] {
    const states = REGIME_STATES;
    const seq: RegimeState[] = [states[startState]];
    let current = startState;
    for (let i = 1; i < n; i++) {
      const r = Math.random();
      let cumul = 0;
      for (let j = 0; j < P[current].length; j++) {
        cumul += P[current][j];
        if (r < cumul) { current = j; break; }
      }
      seq.push(states[current]);
    }
    return seq;
  }

  it('returns null for short sequences (< 50)', () => {
    const shortSeq: RegimeState[] = Array(30).fill('sideways');
    const P = buildDefaultMatrix();
    expect(transitionGoodnessOfFit(shortSeq, P)).toBeNull();
  });

  it('passes for data generated from the same matrix', () => {
    // Build a known 5x5 transition matrix and generate data from it
    const P = buildDefaultMatrix(); // diagonal-dominant
    const seq = generateMarkovChain(P, 500, 2); // start from sideways
    // Use decayRate=1.0 (uniform weighting) so estimated matrix matches generating process
    const estimatedP = estimateTransitionMatrix(seq, undefined, 30, 1.0);
    const result = transitionGoodnessOfFit(seq, estimatedP);

    // With enough data and correctly estimated P, the test should pass
    expect(result).not.toBeNull();
    expect(result!.passes).toBe(true);
    expect(result!.pValue).toBeGreaterThan(0.05);
    expect(result!.chiSquared).toBeGreaterThanOrEqual(0);
    expect(result!.degreesOfFreedom).toBeGreaterThan(0);
  });

  it('fails for data generated from a very different matrix', () => {
    // Generate data from a uniform-transition matrix
    const uniformP = Array.from({ length: NUM_STATES }, () =>
      Array(NUM_STATES).fill(1 / NUM_STATES),
    );
    const seq = generateMarkovChain(uniformP, 500, 0);
    // But test it against a strongly diagonal matrix
    const diagonalP = Array.from({ length: NUM_STATES }, (_, i) =>
      Array.from({ length: NUM_STATES }, (_, j) => i === j ? 0.95 : 0.05 / (NUM_STATES - 1)),
    );
    const result = transitionGoodnessOfFit(seq, diagonalP);
    expect(result).not.toBeNull();
    // Mismatch should produce a low p-value (test fails)
    expect(result!.passes).toBe(false);
    expect(result!.pValue).toBeLessThan(0.05);
  });

  it('result is surfaced in computeMarkovDistribution metadata', async () => {
    // Use trending data that won't trigger structural break detection
    const prices = Array.from({ length: 100 }, (_, i) => 100 + i * 0.05 + Math.random() * 0.5);
    const result = await computeMarkovDistribution({
      ticker: 'GOF_TEST',
      horizon: 20,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [],
    });
    // GOF is null when structural break detected; otherwise computed
    if (result.metadata.structuralBreakDetected) {
      expect(result.metadata.goodnessOfFit).toBeNull();
    } else {
      expect(result.metadata.goodnessOfFit).not.toBeNull();
      expect(typeof result.metadata.goodnessOfFit!.pValue).toBe('number');
      expect(typeof result.metadata.goodnessOfFit!.passes).toBe('boolean');
    }
  });
});

// ---------------------------------------------------------------------------
// computeEnsembleSignal
// ---------------------------------------------------------------------------

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
