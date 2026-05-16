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
  const n = A.length;
  return Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) =>
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

/**
 * Compute the second-largest absolute eigenvalue of the transition matrix
 * using the power iteration method with deflation.
 *
 * ρ determines mixing time: exp(−ρ×n) is how quickly the chain forgets its
 * initial state. Small ρ → fast mixing, Markov signal decays quickly.
 *
 * Returns a value in [0, 1].
 */
export function secondLargestEigenvalue(P: TransitionMatrix, iterations = 100): number {
  const n = P.length;

  // First eigenvector (stationary distribution) via power iteration
  let v = Array(n).fill(1 / n);
  for (let iter = 0; iter < iterations; iter++) {
    const next = Array(n).fill(0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) next[j] += v[i] * P[i][j];
    const norm = next.reduce((s, x) => s + x, 0);
    v = next.map(x => x / norm);
  }
  // L2-normalize v for use in deflation (required for correct orthogonal projection)
  const vL2 = v.reduce((s, x) => s + x * x, 0) ** 0.5;
  const vUnit = vL2 < 1e-12 ? v : v.map(x => x / vL2);

  // Deflate: remove first eigenvector component, find second via power iteration.
  // Use uniform starting vector to avoid biasing toward any particular state.
  let w: number[] = Array.from({ length: n }, () => 1 / n);
  const wNorm = w.reduce((s, x) => s + x * x, 0) ** 0.5;
  w = w.map(x => x / wNorm);

  for (let iter = 0; iter < iterations; iter++) {
    const next = Array(n).fill(0);
    for (let i = 0; i < n; i++)
      for (let j = 0; j < n; j++) next[j] += w[i] * P[i][j];
    // Deflate: subtract L2-normalized component along stationary eigenvector
    const dot = next.reduce((s, x, i) => s + x * vUnit[i], 0);
    const deflated = next.map((x, i) => x - dot * vUnit[i]);
    const norm = deflated.reduce((s, x) => s + x * x, 0) ** 0.5;
    // deflated≈0 means second eigenvalue is ≈0 (e.g. uniform matrix has instant mixing)
    if (norm < 1e-10) return 0;
    w = deflated.map(x => x / norm);
  }

  const Pw = Array(n).fill(0);
  for (let i = 0; i < n; i++)
    for (let j = 0; j < n; j++) Pw[j] += w[i] * P[i][j];

  const lambda2 = w.reduce((s, x, i) => s + x * Pw[i], 0);
  return Math.min(1, Math.max(0, Math.abs(lambda2)));
}
