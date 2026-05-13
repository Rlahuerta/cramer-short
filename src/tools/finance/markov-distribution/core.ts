import {
  createForecastLabAssetScopedRuntimeDefaults,
  type ForecastLabRuntimeAssetScope,
} from '../forecast-lab-runtime-defaults.js';
import type { Domain } from '../calibration-offsets.js';

/**
 * Regime state: 3 states (bull / bear / sideways).
 *
 * Collapsed from the original 5-state model (which included high_vol_bull and
 * high_vol_bear). With a 120-day walk-forward window, the high_vol states had
 * only 2-4 observations each, making their transition rows dominated by the
 * Dirichlet prior (uniform ≈ 0.2/state) with zero directional information.
 * Merging to 3 states concentrates ~40 obs/state → 5-10× more reliable
 * transition estimates. (Nguyen 2018 also uses 3-4 states for S&P 500.)
 */
export type RegimeState = 'bull' | 'bear' | 'sideways';

export const REGIME_STATES: RegimeState[] = [
  'bull', 'bear', 'sideways',
];

export const NUM_STATES = REGIME_STATES.length; // 3

/** Maps state name → matrix row/column index */
export const STATE_INDEX: Record<RegimeState, number> = {
  bull:     0,
  bear:     1,
  sideways: 2,
};

export const FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS: {
  readonly recommendedConfidenceThreshold: number;
  readonly transitionMinObservations: number;
  readonly transitionDecay: number;
  readonly structuralBreakMinLength: number;
  readonly momentumLookback: number;
  readonly momentumAdjustmentScale: number;
  readonly momentumAdjustmentClamp: number;
  readonly trendPenaltyOnlyBreakConfidence: boolean;
  readonly divergenceWeightedBreakConfidence: boolean;
} = {
  recommendedConfidenceThreshold: 0.22,
  transitionMinObservations: 30,
  transitionDecay: 0.97,
  structuralBreakMinLength: 36,
  momentumLookback: 10,
  momentumAdjustmentScale: 0.25,
  momentumAdjustmentClamp: 0.003,
  trendPenaltyOnlyBreakConfidence: true,
  divergenceWeightedBreakConfidence: false,
};

const forecastLabMarkovRuntimeDefaults = createForecastLabAssetScopedRuntimeDefaults(
  FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS,
);

export const PROMOTED_SOL_MARKOV_RUNTIME_DEFAULTS: Partial<typeof FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS> = {
  transitionMinObservations: 31,
  structuralBreakMinLength: 28,
  momentumLookback: 9,
  momentumAdjustmentScale: 0.252,
  momentumAdjustmentClamp: 0.00305,
};

const PROMOTED_HYPE_MARKOV_RUNTIME_DEFAULTS: Partial<typeof FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS> = {
  recommendedConfidenceThreshold: 0.15,
  momentumAdjustmentScale: 0.48,
  momentumAdjustmentClamp: 0.0058,
};

export function resolveForecastLabMarkovParameterDefaults(
  assetScope?: ForecastLabRuntimeAssetScope,
): typeof FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS {
  return forecastLabMarkovRuntimeDefaults.resolve(assetScope);
}

export function getForecastLabMarkovRuntimeDefaults(
  assetScope: ForecastLabRuntimeAssetScope,
): Partial<typeof FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS> | undefined {
  return forecastLabMarkovRuntimeDefaults.get(assetScope);
}

export function setForecastLabMarkovRuntimeDefaults(
  assetScope: ForecastLabRuntimeAssetScope,
  overrides?: Partial<typeof FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS>,
): void {
  forecastLabMarkovRuntimeDefaults.set(assetScope, overrides);
}

setForecastLabMarkovRuntimeDefaults('sol', PROMOTED_SOL_MARKOV_RUNTIME_DEFAULTS);
setForecastLabMarkovRuntimeDefaults('hype', PROMOTED_HYPE_MARKOV_RUNTIME_DEFAULTS);

/**
 * Recommended confidence threshold for selective prediction.
 * Based on coverage-milestone analysis (2026-04-04, 14 tickers × 6 horizons):
 * - conf ≥ 0.25: ~66% accuracy at ~44% coverage (aggregate)
 * - conf ≥ 0.30: ~71% accuracy at ~30% coverage
 * - conf ≥ 0.40: ~75% accuracy at ~15% coverage
 *
 * Use this threshold to filter low-confidence predictions when accuracy matters more than coverage.
 * For full coverage (no filtering), ignore this threshold.
 */
export const RECOMMENDED_CONFIDENCE_THRESHOLD = FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold;
export interface SentimentSignal {
  bullish: number;  // 0–1
  bearish: number;  // 0–1
}

/**
 * Normalize a sentiment signal that may arrive as human-readable percentages
 * (e.g. {bullish: 71, bearish: 29} from social_sentiment) into the 0–1
 * decimal format required by the Markov distribution internals.
 *
 * Heuristic: if either bullish or bearish is > 1, assume the caller passed
 * percentages (0–100 scale) and divide by 100. Values already in [0, 1] are
 * passed through unchanged.
 *
 * Strict failure modes (returns undefined → caller should not use sentiment):
 *  - Negative values
 *  - Values > 100
 *  - Non-number types
 *  - Mixed decimal/percent scales (except exact zero on one side, e.g. 100/0)
 */
export function normalizeSentiment(raw: unknown): SentimentSignal | undefined {
  if (raw == null || typeof raw !== 'object' || Array.isArray(raw)) return undefined;
  const obj = raw as Record<string, unknown>;
  const bullish = obj.bullish;
  const bearish = obj.bearish;
  if (typeof bullish !== 'number' || typeof bearish !== 'number') return undefined;
  if (!Number.isFinite(bullish) || !Number.isFinite(bearish)) return undefined;
  if (bullish < 0 || bearish < 0) return undefined;
  if (bullish > 100 || bearish > 100) return undefined;

  // Percent-scale: if either side exceeds 1, treat the pair as percentages.
  // Reject mixed-scale inputs like 71/0.3, but allow exact zero on one side.
  const isPercentScale = bullish > 1 || bearish > 1;
  if (isPercentScale) {
    const hasMixedScale = (bullish > 0 && bullish <= 1) || (bearish > 0 && bearish <= 1);
    if (hasMixedScale) return undefined;

    return {
      bullish: bullish / 100,
      bearish: bearish / 100,
    };
  }

  return {
    bullish,
    bearish,
  };
}
export interface PriceThreshold {
  price: number;
  /** Raw YES probability from Polymarket (0–1) */
  rawProbability: number;
  /** Bias-corrected probability (rawProbability × YES_BIAS_MULTIPLIER). Reichenbach & Walther (2025)
   *  found systematic YES-overtrading across 124M trades. Uses multiplicative form
   *  (vs. additive in ensemble.ts) because survival interpolation is log-spaced. */
  probability: number;
  /** Whether this anchor is trusted based on liquidity/age heuristics */
  trustScore: 'high' | 'low';
  /** Internal graded trust weight used to prefer better crypto anchors without
   *  changing the outward high/low diagnostics vocabulary. */
  trustWeight?: number;
  /** Optional domain tag (politics/sports/crypto/macro). When provided, the raw
   *  price is recalibrated via {@link recalibratePolymarketPrice} *before* the
   *  Girsanov Q→P shift. Absent ⇒ identity (treated as 'unknown'). */
  domain?: Domain;
  source: 'polymarket' | 'kalshi' | 'averaged';
  endDate?: string | null;
}

/** 3×3 stochastic matrix (rows sum to 1) */
export type TransitionMatrix = number[][];

export interface MarkovDistributionPoint {
  price: number;
  /** P(asset_price > this level) at the given horizon */
  probability: number;
  /** 90% CI lower bound (Monte Carlo 5th percentile) */
  lowerBound: number;
  /** 90% CI upper bound (Monte Carlo 95th percentile) */
  upperBound: number;
  /** How this distribution point was estimated */
  source: 'polymarket' | 'markov' | 'blend';
}

/** A single day in a price trajectory forecast. */
export interface TrajectoryPoint {
  /** Day number (1, 2, ..., N) */
  day: number;
  /** Regime-weighted mean price at this horizon */
  expectedPrice: number;
  /** 5th percentile price (lower 90% CI bound) */
  lowerBound: number;
  /** 95th percentile price (upper 90% CI bound) */
  upperBound: number;
  /** P(price > currentPrice) at this horizon */
  pUp: number;
  /** Cumulative return from current price, e.g., "+1.3%" */
  cumulativeReturn: string;
  /** Most likely regime at this horizon */
  regime: RegimeState;
}
