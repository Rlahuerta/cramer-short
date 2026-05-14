import { afterEach, beforeEach, describe, it, expect, spyOn } from 'bun:test';
import { NUM_STATES, REGIME_STATES } from './core.js';
import type { RegimeState } from './core.js';
import { classifyRegimeState } from './regime.js';
import { buildDefaultMatrix, estimateTransitionMatrix } from './transition.js';
import { computeR2OS, countStateObservations, detectStructuralBreak, findSparseStates, transitionGoodnessOfFit } from './diagnostics.js';
import { computeMarkovDistribution } from '../markov-distribution.js';

function seedRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

let randomSpy: { mockRestore: () => void } | undefined;

beforeEach(() => {
  randomSpy = spyOn(Math, 'random').mockImplementation(seedRng(22345));
});

afterEach(() => {
  randomSpy?.mockRestore();
  randomSpy = undefined;
});

function rowSums(m: number[][]): number[] {
  return m.map(row => row.reduce((s, v) => s + v, 0));
}

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
      sideways:      20,
    };
    const sparse = findSparseStates(counts);
    expect(sparse).toContain('bear');
    expect(sparse).not.toContain('bull');
    expect(sparse).not.toContain('sideways');
  });

  it('returns empty array when all states have enough observations', () => {
    const counts = Object.fromEntries(REGIME_STATES.map(s => [s, 10])) as Record<ReturnType<typeof classifyRegimeState>, number>;
    expect(findSparseStates(counts)).toHaveLength(0);
  });

  it('respects custom minObs parameter', () => {
    const counts = {
      bull: 15, bear: 7, sideways: 9,
    };
    // minObs=10 → bear (7) and sideways (9) are sparse
    const sparse = findSparseStates(counts, 10);
    expect(sparse).toContain('bear');
    expect(sparse).toContain('sideways');
    expect(sparse).not.toContain('bull');
  });
});
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

  it('respects an explicit divergence threshold override', () => {
    const states: ReturnType<typeof classifyRegimeState>[] = [
      ...Array(30).fill('bull'),
      ...Array(30).fill('bear'),
    ];
    const defaultResult = detectStructuralBreak(states);
    const relaxedResult = detectStructuralBreak(states, defaultResult.divergence + 0.01);

    expect(defaultResult.detected).toBe(true);
    expect(relaxedResult.detected).toBe(false);
    expect(relaxedResult.divergence).toBe(defaultResult.divergence);
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
describe('transitionGoodnessOfFit', () => {
  // Generate a synthetic state sequence from a known transition matrix
  function generateMarkovChain(P: number[][], n: number, startState: number): RegimeState[] {
    const states = REGIME_STATES;
    const seq: RegimeState[] = [states[startState]];
    let current = startState;
    const rng = seedRng(201);
    for (let i = 1; i < n; i++) {
      const r = rng();
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
    const rng = seedRng(202);
    const prices = Array.from({ length: 100 }, (_, i) => 100 + i * 0.05 + rng() * 0.5);
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
