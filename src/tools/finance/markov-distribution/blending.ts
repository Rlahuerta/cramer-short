import { NUM_STATES, type TransitionMatrix } from './core.js';
import type { AssetProfile } from './asset-profile.js';

export type BreakConfidencePolicy = 'default' | 'trend_penalty_only' | 'divergence_weighted';

// ---------------------------------------------------------------------------
// Phase 6: Divergence-weighted break confidence (backtest-only)
//
// When a structural break is detected, the current Phase 4 behavior applies
// a flat 0.6 penalty to confidence regardless of divergence severity.
// Phase 6 uses structuralBreakDivergence to select a lighter penalty for
// mild breaks and the current full penalty for high-divergence breaks.
// ---------------------------------------------------------------------------

/**
 * Penalty schedule mapping divergence severity buckets to confidence
 * multipliers. Lower values = harsher penalty.
 *
 * The existing Phase 4 flat penalty is 0.6. Phase 6 schedules allow
 * lighter penalties for mild breaks while preserving 0.6 for severe ones.
 */
export interface DivergencePenaltySchedule {
  /** Penalty multiplier for mild breaks (divergence ∈ [0.05, 0.10)). Default: 0.80 */
  mild: number;
  /** Penalty multiplier for medium breaks (divergence ∈ [0.10, 0.20)). Default: 0.70 */
  medium: number;
  /** Penalty multiplier for high breaks (divergence ≥ 0.20). Default: 0.60 (= Phase 4 baseline) */
  high: number;
}

/** Default Phase 6 schedule: mildest break → 0.80, medium → 0.70, high → 0.60 */
export const DEFAULT_DIVERGENCE_PENALTY_SCHEDULE: DivergencePenaltySchedule = {
  mild: 0.80,
  medium: 0.70,
  high: 0.60,
};

/**
 * Map a structural break divergence value to a confidence penalty multiplier
 * using the severity bucket semantics already used elsewhere in the codebase
 * (same thresholds as computeBlendWeight in Phase 5).
 *
 * - divergence < 0.05: no break penalty (no structural break detected or trivial divergence)
 * - divergence ∈ [0.05, 0.10): mild → schedule.mild
 * - divergence ∈ [0.10, 0.20): medium → schedule.medium
 * - divergence ≥ 0.20: high → schedule.high
 */
export function computeDivergencePenalty(
  divergence: number,
  schedule: DivergencePenaltySchedule,
): number {
  if (divergence < 0.05) return 1.0; // no break → no penalty
  if (divergence < 0.10) return schedule.mild;
  if (divergence < 0.20) return schedule.medium;
  return schedule.high;
}

// ---------------------------------------------------------------------------
// Phase 5: Experimental hybrid structural-break fallback candidates
// ---------------------------------------------------------------------------

/**
 * Experimental fallback candidate for hybrid structural-break matrix blending.
 * Backtest-only: these candidates are NOT used in production defaults.
 *
 * When a structural break is detected, the current hard-replacement behavior
 * substitutes the default matrix (diagonal=0.6). A fallback candidate allows
 * blending between the estimated matrix and a conservative/profile-based matrix,
 * weighted by the break's divergence severity.
 */
export interface BreakFallbackCandidate {
  /** Unique identifier for this candidate (e.g., 'C55', 'P_BALANCED_LAM050_H025') */
  id: string;
  /** How to apply the fallback: hard replacement, blended, or blended with weight cap */
  mode: 'hard' | 'blended' | 'blended_capped';
  /** Diagonal value for the generic conservative fallback matrix (off-diag = (1-d)/2) */
  conservativeDiagonal: number;
  /** Per-asset-type diagonal values for the profile fallback matrix */
  profileDiagonals: {
    equity: number;
    etf: number;
    commodity: number;
    crypto: number;
  };
  /** Weight of the conservative matrix in the hybrid: hybrid = λ*conservative + (1-λ)*profile */
  conservativeWeight: number;
  /** Severity-dependent blend weights: how much fallback matrix to use, by divergence bucket */
  severityWeights: {
    mild: number;    // divergence in [0.05, 0.10)
    medium: number;  // divergence in [0.10, 0.20)
    high: number;    // divergence >= 0.20
  };
  /** Maximum blend weight cap for blended_capped mode (undefined for other modes) */
  maxBlendWeight?: number;
}

/** Build a 3×3 conservative fallback matrix from a diagonal value. */
export function buildConservativeFallbackMatrix(diagonal: number): TransitionMatrix {
  const offDiag = (1 - diagonal) / (NUM_STATES - 1);
  return Array.from({ length: NUM_STATES }, (_, i) =>
    Array.from({ length: NUM_STATES }, (_, j) => (i === j ? diagonal : offDiag)),
  );
}

/** Build a 3×3 profile fallback matrix for a given asset type. */
export function buildProfileFallbackMatrix(
  assetType: AssetProfile['type'],
  profileDiagonals: BreakFallbackCandidate['profileDiagonals'],
): TransitionMatrix {
  const diagonal = profileDiagonals[assetType];
  const offDiag = (1 - diagonal) / (NUM_STATES - 1);
  return Array.from({ length: NUM_STATES }, (_, i) =>
    Array.from({ length: NUM_STATES }, (_, j) => (i === j ? diagonal : offDiag)),
  );
}

/** Blend two transition matrices: result = λ*A + (1-λ)*B */
export function blendMatrices(
  lambda: number,
  A: TransitionMatrix,
  B: TransitionMatrix,
): TransitionMatrix {
  return A.map((row, i) =>
    row.map((_, j) => lambda * A[i][j] + (1 - lambda) * B[i][j]),
  );
}

/**
 * Compute the blend weight for a structural-break fallback matrix,
 * based on the divergence severity bucket.
 */
export function computeBlendWeight(
  divergence: number,
  severityWeights: BreakFallbackCandidate['severityWeights'],
): number {
  if (divergence < 0.05) return 0;
  if (divergence < 0.10) return severityWeights.mild;
  if (divergence < 0.20) return severityWeights.medium;
  return severityWeights.high;
}

/**
 * Apply a BreakFallbackCandidate to produce the final transition matrix
 * for a structural-break window. Returns null if no fallback should be
 * applied (i.e., the current hard-replacement behavior should be used).
 *
 * When a candidate is supplied and a break is detected:
 * - `hard` mode: replaces the estimated matrix with the hybrid fallback matrix
 * - `blended` mode: (1-w)*estimated + w*hybridFallback, where w depends on divergence
 * - `blended_capped` mode: same as blended, but w is capped by maxBlendWeight
 */
export function applyBreakFallbackCandidate(
  estimatedMatrix: TransitionMatrix,
  divergence: number,
  candidate: BreakFallbackCandidate,
  assetType: AssetProfile['type'],
): TransitionMatrix {
  const conservativeMatrix = buildConservativeFallbackMatrix(candidate.conservativeDiagonal);
  const profileMatrix = buildProfileFallbackMatrix(assetType, candidate.profileDiagonals);
  const hybridFallback = blendMatrices(
    candidate.conservativeWeight,
    conservativeMatrix,
    profileMatrix,
  );

  const blendWeight = computeBlendWeight(divergence, candidate.severityWeights);
  const cappedWeight = candidate.mode === 'blended_capped' && candidate.maxBlendWeight !== undefined
    ? Math.min(blendWeight, candidate.maxBlendWeight)
    : blendWeight;

  if (candidate.mode === 'hard') {
    // Hard replacement: use the hybrid fallback matrix entirely
    return hybridFallback;
  }

  // Blended or blended_capped: interpolate between estimated and hybrid fallback
  return blendMatrices(1 - cappedWeight, estimatedMatrix, hybridFallback);
}
