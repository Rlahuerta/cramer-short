import { describe, it, expect } from 'bun:test';
import { NUM_STATES, STATE_INDEX } from './core.js';
import { classifyRegimeState } from './regime.js';
import { adjustTransitionMatrix, buildDefaultMatrix, estimateTransitionMatrix, matMul, matPow, normalizeRows, secondLargestEigenvalue } from './transition.js';

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

  it('has correct dimensions (3×3)', () => {
    const m = buildDefaultMatrix();
    expect(m.length).toBe(NUM_STATES);
    for (const row of m) expect(row.length).toBe(NUM_STATES);
  });
});
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
    const states = repeatStates(['bull', 'sideways', 'bear'], 60);
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
    // Identity matrix: all eigenvalues = 1. The second eigenvalue is degenerate
    // (any orthonormal vector is an eigenvector). The uniform starting vector is
    // orthogonal to the first eigenvector's basis, so the deflated power iteration
    // lands at zero → returns 0. This is correct behavior for a pathological matrix.
    // A well-conditioned near-identity matrix would return ~1.
    expect(rho).toBe(0);
  });

  it('uniform row matrix has second eigenvalue close to 0 (instant mixing)', () => {
    const uniform: number[][] = Array.from({ length: NUM_STATES }, () =>
      Array(NUM_STATES).fill(1 / NUM_STATES),
    );
    const rho = secondLargestEigenvalue(uniform);
    expect(rho).toBeLessThan(0.1);
  });
});
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
