import {
  NUM_STATES,
  REGIME_STATES,
  STATE_INDEX,
  resolveForecastLabMarkovParameterDefaults,
  type RegimeState,
  type TransitionMatrix,
} from './core.js';
import { normalCDF } from './confidence-intervals.js';
import { buildDefaultMatrix, estimateTransitionMatrix } from './transition.js';

/** Result of chi-squared goodness-of-fit test for the transition matrix. */
export interface GoodnessOfFitResult {
  /** Chi-squared statistic */
  chiSquared: number;
  /** Degrees of freedom */
  degreesOfFreedom: number;
  /** Approximate p-value (higher = better fit; < 0.05 suggests poor Markov fit) */
  pValue: number;
  /** Whether the model passes the test at α=0.05 */
  passes: boolean;
}
// ---------------------------------------------------------------------------
// Goodness-of-fit: Chi-squared test for Markov transition matrix
// ---------------------------------------------------------------------------

/**
 * Chi-squared goodness-of-fit test comparing observed transition counts
 * against expected counts from the estimated transition matrix.
 *
 * Tests H₀: the observed transitions are consistent with the estimated P.
 * A low p-value (< 0.05) means the Markov assumption is a poor fit.
 *
 * Uses the Wilson–Hilferty approximation for the chi-squared CDF.
 */
export function transitionGoodnessOfFit(
  states: RegimeState[],
  P: TransitionMatrix,
  alpha = 0.1,  // same Dirichlet prior used during estimation
): GoodnessOfFitResult | null {
  if (states.length < 50) return null; // not enough data for reliable test

  // Build observed count matrix
  const observed: number[][] = Array.from({ length: NUM_STATES }, () =>
    Array(NUM_STATES).fill(0),
  );
  for (let i = 0; i < states.length - 1; i++) {
    observed[STATE_INDEX[states[i]]][STATE_INDEX[states[i + 1]]] += 1;
  }

  // Row totals for expected counts
  const rowTotals = observed.map(row => row.reduce((a, b) => a + b, 0));

  let chiSq = 0;
  let df = 0;

  // Per-row df = (contributing_cells_in_row − 1) for each active row.
  // Each row's transition probabilities are constrained to sum to 1, so a row
  // that contributes k cells to chi-sq has only (k − 1) degrees of freedom
  // (one cell is determined by the others). Total df is the sum across active rows.
  // Previous formulation (df_total = num_terms − activeRows × (NUM_STATES − 1))
  // over-corrects when some cells are skipped due to expected < 1, since it
  // subtracts (NUM_STATES − 1) parameters per row even when fewer cells contributed.
  for (let i = 0; i < NUM_STATES; i++) {
    if (rowTotals[i] < 5) continue; // skip rows with too few observations
    let rowCells = 0;
    for (let j = 0; j < NUM_STATES; j++) {
      const expected = rowTotals[i] * P[i][j];
      if (expected < 1) continue; // skip tiny expected counts (chi-sq unreliable)
      chiSq += (observed[i][j] - expected) ** 2 / expected;
      rowCells += 1;
    }
    if (rowCells >= 2) df += rowCells - 1; // row contributes (k − 1) df
  }
  df = Math.max(1, df);

  // Wilson–Hilferty normal approximation for chi-squared CDF
  const z = Math.cbrt(chiSq / df) - (1 - 2 / (9 * df));
  const zNorm = z / Math.sqrt(2 / (9 * df));
  const pValue = 1 - normalCDF(zNorm);

  return {
    chiSquared: chiSq,
    degreesOfFreedom: df,
    pValue,
    passes: pValue >= 0.05,
  };
}

// ---------------------------------------------------------------------------
// Tier 1a: countStateObservations + sparseStates
// ---------------------------------------------------------------------------

/**
 * Count how many times each regime state appears in the sequence.
 * Used to identify states with too few observations for reliable transition estimation.
 */
export function countStateObservations(states: RegimeState[]): Record<RegimeState, number> {
  const counts = Object.fromEntries(REGIME_STATES.map(s => [s, 0])) as Record<RegimeState, number>;
  for (const s of states) counts[s]++;
  return counts;
}

/**
 * Return states with fewer than `minObs` observations.
 * These states have outgoing transitions dominated by the Dirichlet prior,
 * not by empirical data. Callers should treat their transition rows with lower confidence.
 */
export function findSparseStates(
  observationCounts: Record<RegimeState, number>,
  minObs = 5,
): RegimeState[] {
  return REGIME_STATES.filter(s => observationCounts[s] < minObs);
}

// ---------------------------------------------------------------------------
// Tier 1b: detectStructuralBreak
// ---------------------------------------------------------------------------

/**
 * Detect a structural break in the transition matrix between the first and second
 * halves of the state sequence.
 *
 * Uses a chi-square-like divergence statistic on the empirical transition counts:
 *   D = Σᵢⱼ |P_first[i][j] − P_second[i][j]|²  (Frobenius-style element divergence)
 *
 * When D > threshold (default 0.05 per cell = 0.05 × N² total), the two halves of
 * the training window describe meaningfully different dynamics. In that case:
 *   1. Fall back to the default (identity-like) transition matrix — the full-window
 *      estimate mixes two different regimes and is unreliable.
 *   2. Widen all CI bounds by 50% to reflect increased model uncertainty.
 *
 * This addresses the non-stationarity limitation noted in Mettle et al. (2014)
 * and Welton & Ades (2005): time-homogeneous Markov assumption is violated when
 * the market regime changes mid-window.
 */
export interface StructuralBreakResult {
  detected: boolean;
  /** Sum of squared element-wise differences between first/second half matrices */
  divergence: number;
  firstHalfMatrix: TransitionMatrix;
  secondHalfMatrix: TransitionMatrix;
}

export function detectStructuralBreak(
  states: RegimeState[],
  divergenceThreshold = 0.05,
  alpha = 0.1,
  decayRate = resolveForecastLabMarkovParameterDefaults().transitionDecay,
  minLength = resolveForecastLabMarkovParameterDefaults().structuralBreakMinLength,
): StructuralBreakResult {
  // Each half must have enough observations for a meaningful chi-square-like
  // comparison. With NUM_STATES² = 9 transition cells and the rule of thumb of
  // ≥5 expected counts per cell, each half needs ≥45 transitions; rounded up to
  // 60 to keep margins comfortable. Below this, the divergence statistic is
  // dominated by Dirichlet smoothing rather than empirical signal — return
  // detected=false rather than risk a false alarm fallback.
  if (states.length < minLength) {
    const fallback = buildDefaultMatrix();
    return {
      detected: false,
      divergence: 0,
      firstHalfMatrix: fallback,
      secondHalfMatrix: fallback,
    };
  }
  const mid = Math.floor(states.length / 2);
  const firstHalf  = states.slice(0, mid);
  const secondHalf = states.slice(mid);

  const firstHalfMatrix  = estimateTransitionMatrix(firstHalf,  alpha, 10, decayRate);
  const secondHalfMatrix = estimateTransitionMatrix(secondHalf, alpha, 10, decayRate);

  let divergence = 0;
  for (let i = 0; i < NUM_STATES; i++) {
    for (let j = 0; j < NUM_STATES; j++) {
      divergence += (firstHalfMatrix[i][j] - secondHalfMatrix[i][j]) ** 2;
    }
  }

  return {
    detected: divergence > divergenceThreshold,
    divergence,
    firstHalfMatrix,
    secondHalfMatrix,
  };
}
// ---------------------------------------------------------------------------
// 7. R²_OS out-of-sample validation
// ---------------------------------------------------------------------------

/**
 * Compute out-of-sample R² vs. historical-average baseline.
 * R²_OS > 0 means the Markov model adds value over naive mean forecast.
 *
 * R²_OS = 1 − Σ(actual − predicted)² / Σ(actual − mean(actual))²
 * (Nguyen 2018, IJFS; Campbell & Thompson 2008)
 */
export function computeR2OS(
  actualReturns: number[],
  predictedReturns: number[],
): number {
  if (actualReturns.length < 2) return 0;
  const mean = actualReturns.reduce((s, v) => s + v, 0) / actualReturns.length;
  let ssRes = 0, ssTot = 0;
  for (let i = 0; i < actualReturns.length; i++) {
    ssRes += (actualReturns[i] - predictedReturns[i]) ** 2;
    ssTot += (actualReturns[i] - mean) ** 2;
  }
  if (ssTot < 1e-14) return 0;
  return 1 - ssRes / ssTot;
}
