/**
 * Mirrors `research/models/markov.py` (estimate_transition_matrix).
 */

import {
  NUM_STATES,
  STATE_INDEX,
  resolveForecastLabMarkovParameterDefaults,
  type RegimeState,
  type SentimentSignal,
  type TransitionMatrix,
} from './core.js';

/**
 * Estimate a 3×3 Markov transition matrix from a sequence of regime states.
 *
 * Smoothing: Dirichlet α scales inversely with sample size (default: max(0.01, 5/N)).
 * Converges to ~0.1 (Jeffreys prior) at N=50, and shrinks for larger samples to let
 * data dominate. Welton & Ades (2005) recommends α=0.1 for sparse counts; the adaptive
 * formula reduces over-smoothing for longer windows while still regularizing short ones.
 *
 * Default matrix (insufficient data): 0.6 diagonal, uniform off-diagonal.
 * offDiag = (1 − 0.6) / (NUM_STATES − 1) = 0.4 / 2 = 0.2 per cell (rows sum to 1.0).
 *
 * Bug note: The original spec specified "0.2 off-diagonal" for a 4-state matrix,
 * yielding row sums of 0.6 + 3×0.2 = 1.2. Fixed here to use the correct formula.
 */
export function estimateTransitionMatrix(
  states: RegimeState[],
  alpha?: number,     // Dirichlet smoothing constant (auto-tuned if omitted)
  minObservations = resolveForecastLabMarkovParameterDefaults().transitionMinObservations,
  decayRate = resolveForecastLabMarkovParameterDefaults().transitionDecay,   // Exponential decay: recent transitions weighted more (1.0 = no decay)
  stickinessShrinkage = false,
): TransitionMatrix {
  if (states.length < minObservations) {
    return buildDefaultMatrix();
  }

  // Auto-tune: scale inversely with sample size
  const effectiveAlpha = alpha ?? Math.max(0.01, 5.0 / states.length);

  // Initialise count matrix with Dirichlet prior
  const counts: number[][] = Array.from({ length: NUM_STATES }, () =>
    Array(NUM_STATES).fill(effectiveAlpha),
  );

  // Exponentially-weighted transition counts: recent transitions matter more.
  // weight = decayRate^(distance_from_end). Last transition gets weight=1.
  const n = states.length - 1;
  for (let i = 0; i < n; i++) {
    const from = STATE_INDEX[states[i]];
    const to   = STATE_INDEX[states[i + 1]];
    const age  = n - 1 - i; // 0 = most recent, n-1 = oldest
    counts[from][to] += Math.pow(decayRate, age);
  }

  const result = normalizeRows(counts);

  if (stickinessShrinkage) {
    const maxDiag = Math.max(...result.map((row, i) => row[i]));
    const dStart = 0.8;
    const s = Math.max(0, (maxDiag - dStart) / (1.0 - dStart)) ** 2;
    const priorStrength = 10.0 * (1.0 + 9.0 * s);
    const def = buildDefaultMatrix();

    for (let i = 0; i < NUM_STATES; i++) {
      const rawRowSum = Math.max(
        0,
        counts[i].reduce((sum, v) => sum + v, 0) - NUM_STATES * effectiveAlpha,
      );
      const lambda = rawRowSum / (rawRowSum + priorStrength);
      result[i] = result[i].map((v, j) => lambda * v + (1.0 - lambda) * def[i][j]);
    }
  }

  for (const row of result) {
    const sum = row.reduce((acc, v) => acc + v, 0);
    if (Math.abs(sum - 1.0) > 1e-10) throw new Error('Transition matrix rows must sum to 1');
    if (row.some(v => v < -1e-12)) throw new Error('Transition matrix entries must be nonnegative');
  }

  return result;
}

export function estimateConditionalTransitionMatrices(
  regimeSequence: RegimeState[],
  environmentSequence: string[],
  alpha?: number,
  decayRate?: number,
): Record<string, TransitionMatrix> {
  if (regimeSequence.length !== environmentSequence.length) {
    throw new Error('Length mismatch');
  }

  const pooled = regimeSequence.length >= 2
    ? estimateTransitionMatrix(regimeSequence, alpha, undefined, decayRate)
    : buildDefaultMatrix();
  const defaults = resolveForecastLabMarkovParameterDefaults();
  const sparseObservationCount = Math.max(2, Math.trunc(defaults.transitionMinObservations));

  const pairsByEnv: Record<string, Array<[RegimeState, RegimeState]>> = {};
  for (const env of environmentSequence) {
    pairsByEnv[env] = [];
  }
  for (let i = 0; i < regimeSequence.length - 1; i++) {
    const env = environmentSequence[i];
    pairsByEnv[env] ??= [];
    pairsByEnv[env].push([regimeSequence[i], regimeSequence[i + 1]]);
  }

  const matrices: Record<string, TransitionMatrix> = {};
  for (const [env, envPairs] of Object.entries(pairsByEnv)) {
    // Sparse gate unchanged: a pair used to occupy two entries in the old
    // flattened state sequence.
    matrices[env] = envPairs.length * 2 < sparseObservationCount
      ? pooled
      : estimateTransitionMatrixFromPairs(envPairs, alpha, decayRate);
  }
  return matrices;
}

/**
 * Estimate a matrix from ordered (from, to) pair observations.
 *
 * Equivalent to `estimateTransitionMatrix` over the pair sequence itself:
 * Dirichlet prior in every cell, each pair weighted by
 * `decayRate^(distance from the end)` (most recent pair weight 1), then the
 * same row normalization. Counting pairs directly is what prevents spurious
 * transitions across pair boundaries.
 */
function estimateTransitionMatrixFromPairs(
  pairs: ReadonlyArray<readonly [RegimeState, RegimeState]>,
  alpha?: number,
  decayRate = resolveForecastLabMarkovParameterDefaults().transitionDecay,
): TransitionMatrix {
  const effectiveAlpha = alpha ?? Math.max(0.01, 5.0 / pairs.length);
  const counts: number[][] = Array.from({ length: NUM_STATES }, () =>
    Array(NUM_STATES).fill(effectiveAlpha),
  );

  const n = pairs.length;
  for (let k = 0; k < n; k++) {
    const from = STATE_INDEX[pairs[k][0]];
    const to = STATE_INDEX[pairs[k][1]];
    const age = n - 1 - k; // 0 = most recent, n-1 = oldest
    counts[from][to] += Math.pow(decayRate, age);
  }

  return normalizeRows(counts);
}

/** Identity-like default matrix with correct row sums. */
export function buildDefaultMatrix(): TransitionMatrix {
  const diagonal = 0.6;
  const offDiag  = (1 - diagonal) / (NUM_STATES - 1); // 0.2 for 3 states
  return Array.from({ length: NUM_STATES }, (_, i) =>
    Array.from({ length: NUM_STATES }, (_, j) => (i === j ? diagonal : offDiag)),
  );
}

/** Normalize each row of a matrix to sum to 1. Zero-sum rows become uniform. */
export function normalizeRows(matrix: number[][]): TransitionMatrix {
  return matrix.map(row => {
    const sum = row.reduce((a, b) => a + b, 0);
    if (sum < 1e-12) {
      // Degenerate row: distribute uniformly to avoid NaN
      const uniform = 1 / row.length;
      return row.map(() => uniform);
    }
    return row.map(v => v / sum);
  });
}
/**
 * Apply sentiment-based adjustments to the baseline transition matrix.
 *
 * Only bull↔bear rows are adjusted — volatile states are intentionally left
 * unmodified since sentiment doesn't reliably predict intraday vol.
 *
 * α = 0.07 (reduced from the original 0.15). Davidovic & McCleary (2025, JRFM)
 * show that news sentiment scores (TextBlob/VADER/FinBERT) capture <5% of return
 * variation. Overly strong adjustments would corrupt the empirically estimated matrix.
 *
 * Sign fix: The original spec had `bear.to.bear = base * (1 - alpha * -shift)`,
 * which equals `base * (1 + shift)` and INCREASES bear persistence under bullish
 * sentiment. Corrected here: bullish shift reduces bear persistence (1 - alpha*shift).
 */
export function adjustTransitionMatrix(
  base: TransitionMatrix,
  sentiment: SentimentSignal,
  alpha = 0.07,
): TransitionMatrix {
  const shift = sentiment.bullish - sentiment.bearish; // -1 to +1
  const adjusted = base.map(row => [...row]);

  const bull = STATE_INDEX['bull'];
  const bear = STATE_INDEX['bear'];

  // Bull row: bullish sentiment → more persistence in bull, less exit to bear
  adjusted[bull][bull] = base[bull][bull] * (1 + alpha * shift);
  adjusted[bull][bear] = base[bull][bear] * (1 - alpha * shift);

  // Bear row: bullish sentiment → less persistence in bear, more exit to bull
  // (double-negative removed from original spec: was `(1 - alpha * -shift)`)
  adjusted[bear][bear] = base[bear][bear] * (1 - alpha * shift);
  adjusted[bear][bull] = base[bear][bull] * (1 + alpha * shift);

  // Clamp negatives to 0 before normalizing
  for (let i = 0; i < NUM_STATES; i++) {
    for (let j = 0; j < NUM_STATES; j++) {
      adjusted[i][j] = Math.max(0, adjusted[i][j]);
    }
  }

  return normalizeRows(adjusted);
}
// ---------------------------------------------------------------------------
// 5. Matrix math utilities
// ---------------------------------------------------------------------------

/** Matrix multiplication A × B. */
export function matMul(A: number[][], B: number[][]): number[][] {
  const rows = A.length;
  const cols = B[0]?.length ?? 0;
  return Array.from({ length: rows }, (_, i) =>
    Array.from({ length: cols }, (_, j) =>
      A[i].reduce((s, _, k) => s + A[i][k] * B[k][j], 0),
    ),
  );
}

/** Compute P^n by repeated squaring (O(n² log n)). */
export function matPow(P: TransitionMatrix, n: number): TransitionMatrix {
  if (n === 0) return Array.from({ length: P.length }, (_, i) =>
    Array.from({ length: P.length }, (_, j) => (i === j ? 1 : 0)),
  );
  if (n === 1) return P.map(r => [...r]);
  if (n % 2 === 0) {
    const half = matPow(P, n / 2);
    return matMul(half, half);
  }
  return matMul(P, matPow(P, n - 1));
}

function validateTransitionMatrix(P: number[][]): void {
  const n = P.length;
  if (n === 0 || P.some(row => row.length !== n)) {
    throw new Error('Transition matrix must be square');
  }
  for (const row of P) {
    const sum = row.reduce((acc, v) => acc + v, 0);
    if (Math.abs(sum - 1.0) > 1e-10) throw new Error('Transition matrix rows must sum to 1');
    if (row.some(v => v < -1e-12)) throw new Error('Transition matrix entries must be nonnegative');
  }
}

export function isIrreducible(P: TransitionMatrix, tol = 1e-12): boolean {
  const n = P.length;
  if (n === 0 || P.some(row => row.length !== n)) {
    throw new Error('Transition matrix must be square');
  }

  const reachable = P.map((row, i) => row.map((v, j) => i === j || v > tol));
  for (let k = 0; k < n; k++) {
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        reachable[i][j] = reachable[i][j] || (reachable[i][k] && reachable[k][j]);
      }
    }
  }
  return reachable.every(row => row.every(Boolean));
}

export function stationaryDistribution(
  P: TransitionMatrix,
  maxIterations = 1000,
  tolerance = 1e-10,
): number[] {
  validateTransitionMatrix(P);
  if (!isIrreducible(P)) {
    throw new Error('Transition matrix must be irreducible');
  }

  const n = P.length;
  let pi = Array(n).fill(1 / n);

  for (let iter = 0; iter < maxIterations; iter++) {
    const nextPi = matMul([pi], P)[0];
    const delta = nextPi.reduce((sum, v, i) => sum + Math.abs(v - pi[i]), 0);
    if (delta < tolerance) return nextPi;
    pi = nextPi;
  }

  throw new Error(`Stationary distribution did not converge in ${maxIterations} iterations`);
}

/**
 * Compute the second-largest absolute eigenvalue (magnitude) of a
 * row-stochastic transition matrix.
 *
 * Matches the Python reference (`np.linalg.eigvals`, second-largest magnitude)
 * for 2×2 and 3×3 matrices:
 *  - 2×2: closed-form quadratic on trace/determinant.
 *  - 3×3: λ = 1 is known (rows sum to 1), so factor (λ − 1) out of the
 *    characteristic polynomial and solve the remaining quadratic exactly.
 * This replaces a deflated power iteration that returned 0 for every
 * doubly-stochastic matrix: its uniform start vector is already the stationary
 * eigenvector, so the projection cancelled the iterate immediately (the
 * default matrix's true ρ = 0.4 came back as 0).
 *
 * ρ determines mixing time: exp(−ρ×n) is how quickly the chain forgets its
 * initial state. Small ρ → fast mixing, Markov signal decays quickly.
 *
 * Returns a value in [0, 1].
 */
export function secondLargestEigenvalue(P: TransitionMatrix): number {
  const magnitudes = eigenvalueMagnitudes(P);
  if (magnitudes.length < 2) return 0;
  magnitudes.sort((a, b) => b - a);
  return Math.min(1, Math.max(0, magnitudes[1]));
}

function eigenvalueMagnitudes(P: TransitionMatrix): number[] {
  const n = P.length;
  if (n <= 1) return n === 1 ? [Math.abs(P[0][0])] : [];
  if (n === 2) {
    const tr = P[0][0] + P[1][1];
    const det = P[0][0] * P[1][1] - P[0][1] * P[1][0];
    const disc = tr * tr - 4 * det;
    if (disc < 0) {
      const mag = Math.sqrt(Math.max(0, det));
      return [mag, mag];
    }
    const root = Math.sqrt(disc);
    return [Math.abs((tr + root) / 2), Math.abs((tr - root) / 2)];
  }
  if (n === 3) {
    const tr = P[0][0] + P[1][1] + P[2][2];
    const det =
      P[0][0] * (P[1][1] * P[2][2] - P[1][2] * P[2][1]) -
      P[0][1] * (P[1][0] * P[2][2] - P[1][2] * P[2][0]) +
      P[0][2] * (P[1][0] * P[2][1] - P[1][1] * P[2][0]);
    const b = tr - 1;
    const c = det;
    const disc = b * b - 4 * c;
    if (disc < 0) {
      const mag = Math.sqrt(Math.max(0, c));
      return [1, mag, mag];
    }
    const root = Math.sqrt(disc);
    const mu1 = b >= 0 ? (b + root) / 2 : (b - root) / 2;
    const mu2 = mu1 !== 0 ? c / mu1 : 0;
    return [1, Math.abs(mu1), Math.abs(mu2)];
  }
  throw new Error('secondLargestEigenvalue supports matrices up to 3×3');
}
