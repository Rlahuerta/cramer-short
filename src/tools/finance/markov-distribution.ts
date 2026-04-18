/**
 * Markov Chain Probability Distribution for Asset Price Forecasting
 *
 * Produces full probability distributions for stock/ETF prices at user-specified
 * horizons by combining:
 *  1. Polymarket threshold markets — real-money anchors at specific price levels
 *  2. Historical regime transitions — Markov transition matrix from 60-90 days of history
 *  3. Sentiment adjustments — dynamic multipliers from social sentiment signals
 *
 * Academic grounding:
 *  - Nguyen (2018, IJFS): 4-state HMM for S&P 500; AIC/BIC model selection
 *  - Mettle et al. (2014, SpringerPlus): Markov chain methodology for share prices
 *  - Kumar & Amer (2023, UVic): LSTM + 3-state Markov for stock prediction
 *  - Welton & Ades (2005): Dirichlet priors for transition matrix estimation
 *  - Voigt (2025): Beta-HMM for Polymarket prediction markets
 *  - Reichenbach & Walther (2025): YES-bias in Polymarket (124M trades)
 *  - Davidovic & McCleary (2025, JRFM): Sentiment alpha ≪ VIX for return prediction
 */

import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { resolveTickerSearchIdentity } from './asset-resolver.js';
import { YES_BIAS_MULTIPLIER } from '../../utils/ensemble.js';
import { baumWelch, predict as hmmPredict, type HMMParams } from './hmm.js';
import { api } from './api.js';
import { fetchBinanceDailyCloses } from './binance.js';
import { fetchPolymarketAnchorMarkets, fetchPolymarketAnchorMarketsWithQueries } from './polymarket.js';
import { extractSignals, normalizeForPolymarket } from './signal-extractor.js';
import { formatToolResult } from '../types.js';

// ---------------------------------------------------------------------------
// Auto-fetch historical prices (used when LLM omits historicalPrices)
// ---------------------------------------------------------------------------

/**
 * Yahoo Finance chart API — free, supports stocks, ETFs, crypto, commodities.
 * Returns oldest-first array of daily close prices.
 */
async function fetchYahooChartPrices(
  ticker: string,
  days: number,
): Promise<number[]> {
  // Map days to Yahoo range parameter
  const range = days <= 30 ? '1mo' : days <= 90 ? '3mo' : days <= 180 ? '6mo' : '1y';
  const url =
    `https://query1.finance.yahoo.com/v8/finance/chart/${encodeURIComponent(ticker)}` +
    `?range=${range}&interval=1d&includePrePost=false`;
  const UA =
    'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 ' +
    '(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36';
  try {
    const res = await fetch(url, {
      headers: { 'User-Agent': UA, Accept: 'application/json' },
      signal: AbortSignal.timeout(10_000),
    });
    if (!res.ok) return [];
    const json = await res.json() as {
      chart?: { result?: Array<{
        indicators?: { quote?: Array<{ close?: (number | null)[] }> };
      }> };
    };
    const closes = json.chart?.result?.[0]?.indicators?.quote?.[0]?.close;
    if (!Array.isArray(closes)) return [];
    return closes.filter((v): v is number => typeof v === 'number' && !isNaN(v));
  } catch {
    return [];
  }
}

/**
 * Fetch daily close prices. Tries Financial Datasets API first (fast, high quality),
 * then falls back to Yahoo Finance chart API (free, works for ETFs/commodities).
 * Returns oldest-first array of close prices, or empty array on total failure.
 */
export async function fetchHistoricalPrices(
  ticker: string,
  days = 120,
): Promise<number[]> {
  const endDate = new Date().toISOString().slice(0, 10);
  const startDate = new Date(Date.now() - days * 24 * 60 * 60 * 1000)
    .toISOString()
    .slice(0, 10);

  // Try Financial Datasets API first
  try {
    const { data } = await api.get('/prices/', {
      ticker,
      interval: 'day',
      start_date: startDate,
      end_date: endDate,
    });
    const prices: Array<{ close: number }> =
      (data as any).prices ?? (data as any) ?? [];
    const closes = prices
      .map((p) => p.close)
      .filter((v) => typeof v === 'number' && !isNaN(v));
    if (closes.length >= 10) return closes;
  } catch {
    // Financial Datasets failed (premium required, rate limit, etc.) — fall through
  }

  const binanceCloses = await fetchBinanceDailyCloses(ticker, days);
  if (binanceCloses.length >= 10) return binanceCloses;

  // Fallback: Yahoo Finance chart API (works for ETFs, commodities, most tickers)
  return fetchYahooChartPrices(ticker, days);
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

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
export const RECOMMENDED_CONFIDENCE_THRESHOLD = 0.25;

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
  source: 'polymarket' | 'kalshi' | 'averaged';
  endDate?: string | null;
}

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
  /** Regime-weighted expected price at this horizon */
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

/** A single scenario bucket in the probability distribution (e.g., "Down >5%"). */
export interface ScenarioBucket {
  /** Human-readable label, e.g., "Down >5%", "Flat ±3%", "Up 3-5%" */
  label: string;
  /** Probability mass in this bucket (0–1), derived from the calibrated CDF */
  probability: number;
  /** Price range boundaries [low, high]. null means unbounded in that direction. */
  priceRange: [number | null, number | null];
}

/**
 * Pre-computed scenario probabilities derived FROM the calibrated CDF distribution.
 * These are guaranteed to be consistent with the P(>Price) table and sum to ~100%.
 */
export interface ScenarioProbabilities {
  /** Scenario buckets: Down >5%, Down 3-5%, Flat ±3%, Up 3-5%, Up >5% */
  buckets: ScenarioBucket[];
  /** Expected price at horizon (from calibrated distribution median) */
  expectedPrice: number;
  /** Expected return as decimal, e.g., 0.006 for +0.6% */
  expectedReturn: number;
  /** P(price > currentPrice) from the calibrated CDF */
  pUp: number;
}

export interface ForecastHint {
  usage: 'forecast_only';
  markovReturn: number;
  confidenceScore: number;
  calibratedDistribution: false;
}

export function buildForecastHint(params: {
  canEmitCanonical: boolean;
  ticker: string;
  horizon: number;
  expectedReturn: number;
  mixingTimeWeight: number;
  predictionConfidence: number;
}): ForecastHint | null {
  const FORECAST_HINT_ABSTAIN_ATTENUATION = 0.5;
  const FORECAST_HINT_MIN_CONFIDENCE = 0.10;

  if (params.canEmitCanonical) return null;
  if (params.ticker !== 'BTC-USD' || params.horizon > 14) return null;
  if (!Number.isFinite(params.expectedReturn) || !Number.isFinite(params.mixingTimeWeight)) return null;
  if (params.predictionConfidence < FORECAST_HINT_MIN_CONFIDENCE) return null;

  const forecastHintConfidenceScale = Math.min(1, params.predictionConfidence / RECOMMENDED_CONFIDENCE_THRESHOLD);
  return {
    usage: 'forecast_only',
    markovReturn: params.expectedReturn * params.mixingTimeWeight * FORECAST_HINT_ABSTAIN_ATTENUATION * forecastHintConfidenceScale,
    confidenceScore: params.predictionConfidence,
    calibratedDistribution: false,
  };
}

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

export interface MarkovDistributionResult {
  ticker: string;
  currentPrice: number;
  horizon: number;
  distribution: MarkovDistributionPoint[];
  /** Raw (pre-calibration) distribution — wider spread, used for CI extraction */
  rawDistribution: MarkovDistributionPoint[];
  /** Actionable Buy / Hold / Sell signal derived from the distribution. */
  actionSignal: ActionSignal;
  /**
   * Pre-computed scenario probabilities derived from the calibrated CDF.
   * Use these instead of computing scenarios independently — they are
   * guaranteed to be consistent with the distribution[] P(>Price) table.
   */
  scenarios: ScenarioProbabilities;
  /**
   * Prediction confidence score (0–1). Higher values indicate the model is more
   * decisive and historically more accurate. Combines:
   *  - Directional decisiveness: |P(up) − 0.5| (how far from coin-flip)
   *  - Ensemble consensus: agreement among momentum, mean-reversion, crossover signals
   *  - HMM convergence: whether the Gaussian HMM converged
   *  - Regime stability: consecutive days in the same regime state
   *
   * Use for selective prediction (sHMM — El-Yaniv & Pidan, NeurIPS 2011):
   * filter out predictions with confidence below a threshold to trade coverage for accuracy.
   */
  predictionConfidence: number;
  /** Day-by-day price trajectory (present when trajectory mode is enabled) */
  trajectory?: TrajectoryPoint[];
  metadata: {
    polymarketAnchors: number;
    regimeState: RegimeState;
    sentimentAdjustment: number;
    historicalDays: number;
    /** exp(−ρ×n): near 1 = Markov-dominant, near 0 = Polymarket-anchor-dominant */
    mixingTimeWeight: number;
    /** Second-largest absolute eigenvalue of the transition matrix (spectral gap) */
    secondEigenvalue: number;
    /** R²_OS vs. historical-average baseline; null if no held-out data */
    outOfSampleR2: number | null;
    validationMetric: 'daily_return' | 'horizon_return';
    /** Tier 1: Observations per regime state in the training window */
    stateObservationCounts: Record<RegimeState, number>;
    /** Tier 1: States with < 5 observations — transitions dominated by Dirichlet prior */
    sparseStates: RegimeState[];
    /** Tier 1: Whether a structural break was detected between the two half-windows */
    structuralBreakDetected: boolean;
    /** Tier 1: Chi-square divergence statistic between first/second half matrices */
    structuralBreakDivergence: number;
    /** Tier 1: CI was widened by 50% due to structural break */
    ciWidened: boolean;
    /** Tier 1: Cross-platform divergence warnings (price levels where anchors disagree >5pp) */
    anchorDivergenceWarnings: Array<{
      price: number;
      polymarketProb: number;
      kalshiProb: number;
      divergencePp: number;
    }>;
    /** Anchor coverage diagnostic — how well Polymarket anchors cover the price range */
    anchorCoverage: AnchorCoverageDiagnostic;
    /** Goodness-of-fit: chi-squared p-value comparing observed vs expected transitions.
     *  Low p-value (< 0.05) means the Markov assumption is a poor fit. null if insufficient data. */
    goodnessOfFit: GoodnessOfFitResult | null;
    /** HMM fitting metadata (null if HMM was not attempted) */
    hmm: { converged: boolean; iterations: number; states: number; logLikelihood: number; volRegimeConverged?: boolean } | null;
    /** Ensemble signal metadata */
    ensemble: { consensus: number; adjustment: number };
    pr3fDisagreementBlendActive: boolean;
    rawDirectionHybridActive?: boolean;
    pr3gRecencyWeightingActive: boolean;
    startStateMixtureActive?: boolean;
    sidewaysSplitActive?: boolean;
    matureBullCalibrationActive?: boolean;
    trendPenaltyOnlyBreakConfidenceActive?: boolean;
    /** Phase 6 provenance: whether divergence-weighted break confidence was active (backtest-only) */
    divergenceWeightedBreakConfidenceActive?: boolean;
    /** Phase 7 provenance: whether regime-specific sigma was active (backtest-only) */
    regimeSpecificSigmaActive?: boolean;
    /** Phase 5 provenance: which fallback candidate was used (backtest-only) */
    breakFallbackCandidateId?: string;
    /** Phase 5 provenance: which fallback mode was applied (backtest-only) */
    breakFallbackMode?: 'hard' | 'blended' | 'blended_capped';
  };
}

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

/** Actionable Buy / Hold / Sell summary derived from the Markov distribution. */
export interface ActionSignal {
  /** P(price rises ≥ buyThreshold above current price by horizon) */
  buyProbability: number;
  /** P(price stays within −sellThreshold..+buyThreshold of current price) */
  holdProbability: number;
  /** P(price falls ≥ sellThreshold below current price by horizon) */
  sellProbability: number;
  /** Primary recommendation based on distribution shape */
  recommendation: 'BUY' | 'HOLD' | 'SELL';
  /** Confidence level based on margin between leading and second probability */
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  /** Expected return over the horizon (e.g. 0.03 = +3%) */
  expectedReturn: number;
  /** Unconditional expected upside / expected downside (> 1 = favours upside) */
  riskRewardRatio: number;
  /** Upside threshold used (default 0.05 = +5%) */
  buyThreshold: number;
  /** Downside threshold used (default 0.03 = −3%) */
  sellThreshold: number;
  /** Key price levels for trade execution */
  actionLevels: ActionLevels;
}

/** Concrete price levels derived from the probability distribution for trade execution. */
export interface ActionLevels {
  /** Price where P(>price) ≈ 30% — profit-taking level (upside target) */
  targetPrice: number;
  /** Price where P(>price) ≈ 90% — strong support / stop-loss level */
  stopLoss: number;
  /** Price where P(>price) ≈ 50% — expected median price at horizon */
  medianPrice: number;
  /** 80th-percentile upside — optimistic but plausible target */
  bullCase: number;
  /** 20th-percentile downside — pessimistic but plausible outcome */
  bearCase: number;
}

/** Diagnostic for Polymarket anchor quality and coverage. */
export interface AnchorCoverageDiagnostic {
  totalAnchors: number;
  trustedAnchors: number;
  /** Largest gap between adjacent anchors as % of current price */
  maxGapPct: number;
  /** Whether anchor coverage is adequate for reliable blending */
  quality: 'good' | 'sparse' | 'none';
  /** Human-readable warning (empty if quality is 'good') */
  warning: string;
}

// ---------------------------------------------------------------------------
// Asset-class parameter profiles (Idea N)
// ---------------------------------------------------------------------------

/**
 * Per-asset-class parameter overrides. Different asset types have fundamentally
 * different volatility, mean-reversion, and regime-switching characteristics.
 * Using identical parameters for SPY (vol ~1.2%) and BTC (vol ~4%) is incorrect.
 */
export interface AssetProfile {
  /** Asset class identifier */
  type: 'etf' | 'equity' | 'crypto' | 'commodity';
  /** Calibration kappa multiplier (1.0 = default). Higher = more conservative. */
  kappaMultiplier: number;
  /** HMM weight multiplier (1.0 = default). Lower = trust HMM less. */
  hmmWeightMultiplier: number;
  /** Student-t degrees of freedom (lower = fatter tails) */
  studentTNu: number;
  /** Transition matrix decay rate */
  decayRate: number;
  /** Maximum absolute daily drift (caps regime meanReturn to prevent shock contamination) */
  maxDailyDrift?: number;
}

const ASSET_PROFILES: Record<AssetProfile['type'], AssetProfile> = {
  etf: {
    type: 'etf',
    kappaMultiplier: 0.85,     // ETFs are more predictable → trust model more
    hmmWeightMultiplier: 1.1,  // HMM works well on smoother series
    studentTNu: 5,
    decayRate: 0.97,
    maxDailyDrift: 0.008,      // ~2% annualized cap
  },
  equity: {
    type: 'equity',
    kappaMultiplier: 1.0,      // baseline
    hmmWeightMultiplier: 0.9,  // slightly less HMM trust (more idiosyncratic)
    studentTNu: 4,             // fatter tails than ETFs
    decayRate: 0.96,
    maxDailyDrift: 0.012,      // ~3% annualized cap
  },
  crypto: {
    type: 'crypto',
    kappaMultiplier: 1.3,      // crypto is noisier → more shrinkage toward base rate
    hmmWeightMultiplier: 0.5,  // HMM less reliable on crypto noise
    studentTNu: 3,             // fattest tails
    decayRate: 0.94,
    maxDailyDrift: 0.025,      // crypto can legitimately drift more
  },
  commodity: {
    type: 'commodity',
    kappaMultiplier: 1.1,      // commodities are driven by supply shocks → slightly more conservative
    hmmWeightMultiplier: 0.7,  // regime switching is real but noisy (geopolitics)
    studentTNu: 4,             // fat tails from supply shocks
    decayRate: 0.95,
    maxDailyDrift: 0.010,      // ~2.5% annualized; prevents geopolitical shock drift contamination
  },
};

/**
 * Map ticker symbol to asset profile. Falls back to 'equity' for unknown tickers.
 * Common ETFs and crypto tickers are recognized by pattern.
 */
export function getAssetProfile(ticker: string): AssetProfile {
  const t = ticker.toUpperCase();
  // Crypto detection
  if (t.includes('BTC') || t.includes('ETH') || t.includes('SOL') ||
      t.includes('DOGE') || t.includes('XRP') || t.endsWith('-USD') ||
      t.endsWith('USDT') || t.includes('CRYPTO')) {
    return ASSET_PROFILES.crypto;
  }
  // Commodity futures detection (CME/NYMEX/COMEX tickers + common names)
  const commodityTickers = new Set([
    'CL', 'NG', 'HO', 'RB',          // energy: crude, nat gas, heating oil, gasoline
    'GC', 'SI', 'HG', 'PL', 'PA',    // metals: gold, silver, copper, platinum, palladium
    'ZC', 'ZW', 'ZS', 'ZM', 'ZL',    // grains: corn, wheat, soybeans, soybean meal/oil
    'CT', 'KC', 'SB', 'CC', 'OJ',    // softs: cotton, coffee, sugar, cocoa, OJ
    'LE', 'HE', 'GF',                 // livestock: live cattle, lean hogs, feeder cattle
    'WTICOUSD', 'BRENTUSD',           // spot oil aliases
    'SILVER', 'COPPER',               // common names for precious/base metals
    'XAUUSD', 'XAGUSD',              // forex-style spot metal symbols
    'NATGAS', 'CRUDE', 'OIL',        // informal energy names
  ]);
  if (commodityTickers.has(t)) return ASSET_PROFILES.commodity;
  // Commodity ETFs — use commodity profile (they track commodity prices)
  const commodityEtfs = new Set([
    'USO', 'UNG', 'DBO', 'GSG', 'DJP', 'PDBC',  // broad/energy commodity ETFs
    'GLD', 'SLV', 'IAU', 'SGOL', 'PPLT',         // precious metal ETFs
    'CPER', 'JJC',                                  // copper ETFs
    'DBA', 'WEAT', 'CORN', 'SOYB',                // agriculture ETFs
  ]);
  if (commodityEtfs.has(t)) return ASSET_PROFILES.commodity;
  // ETF detection (common US ETFs and patterns)
  const etfTickers = new Set([
    'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'VXUS', 'EFA', 'EEM',
    'TLT', 'IEF', 'SHY', 'LQD', 'HYG',
    'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB',
    'ARKK', 'ARKG', 'ARKW', 'SOXL', 'TQQQ', 'SQQQ', 'SPXL', 'VGK',
    'IEMG', 'AGG', 'BND', 'SCHD', 'VYM', 'JEPI', 'VNQ', 'XLRE',
  ]);
  if (etfTickers.has(t)) return ASSET_PROFILES.etf;

  return ASSET_PROFILES.equity;
}

// ---------------------------------------------------------------------------
// 1. extractPriceThresholds
// ---------------------------------------------------------------------------

const ABOVE_PATTERNS = [
  /(?:exceed|above|over)\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
  /(?:settle|close)\s*(?:above|over|at\s*>?)\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
  /(?:be\s+)?at\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
  /\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)\s*(?:or\s*(?:higher|above|more))/i,
  /\$(\d[\d,]*(?:\.\d+)?)\s*(?:per\s*(?:barrel|bbl|ounce|oz|gallon|gal))/i,
  />\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
];

const BELOW_PATTERNS = [
  /(?:below|under|less\s*than)\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
  /(?:settle|close)\s*(?:below|under)\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
];

const BARRIER_PATTERNS = [
  /\breach\b/i,
  /\bhit\b/i,
  /\bgo\s+(?:above|below|over|under)\b/i,
  /\bremain\s+(?:above|below|over|under)\b/i,
  /\btrade\s+(?:above|below|over|under)\b/i,
  /\bsurpass\b/i,
  /\btouch\b/i,
  /\bstay\s+(?:above|below|over|under)\b/i,
  /\b(?:move|go)\s+to\b/i,
  /\b(?:dip|drop|fall|sink|decline|decrease)s?\s+to\b/i,
];

/** Parse a price string like "$70K" or "$1,234.56" into a number. */
function parsePrice(raw: string): number {
  const cleaned = raw.replace(/,/g, '');
  const multiplier = /[Kk]$/.test(cleaned) ? 1_000 : /[Mm]$/.test(cleaned) ? 1_000_000 : 1;
  return parseFloat(cleaned.replace(/[KkMm]$/, '')) * multiplier;
}

/**
 * Parse Polymarket market objects into sorted price thresholds with bias correction.
 *
 * YES-bias correction: multiply all raw probabilities by YES_BIAS_MULTIPLIER (Reichenbach & Walther 2025).
 * Liquidity guard: markets under 48h or with zero volume receive trustScore='low'.
 */
export function extractPriceThresholds(
  markets: Array<{
    question: string;
    probability: number;
    volume?: number;
    createdAt?: string | number;
    endDate?: string | null;
  }>,
  options?: { ticker?: string; horizonDays?: number; referenceTimeMs?: number },
): PriceThreshold[] {
  const seen = new Map<number, PriceThreshold>();

  const now = options?.referenceTimeMs ?? Date.now();

  for (const market of markets) {
    if (BARRIER_PATTERNS.some((pattern) => pattern.test(market.question))) {
      anchorTrace('extract_market', {
        ticker: options?.ticker ?? null,
        horizonDays: options?.horizonDays ?? null,
        question: market.question,
        endDate: market.endDate ?? null,
        volume: market.volume ?? 0,
        excluded: 'barrier_pattern',
      });
      continue;
    }

    let matched: number | null = null;
    let isBelow = false;

    // Try "below" patterns first (more specific)
    for (const pattern of BELOW_PATTERNS) {
      const m = market.question.match(pattern);
      if (m) {
        matched = parsePrice(m[1]);
        isBelow = true;
        break;
      }
    }
    // Then try "above" patterns
    if (matched === null) {
      for (const pattern of ABOVE_PATTERNS) {
        const m = market.question.match(pattern);
        if (m) {
          matched = parsePrice(m[1]);
          isBelow = false;
          break;
        }
      }
    }
    if (matched === null) {
      anchorTrace('extract_market', {
        ticker: options?.ticker ?? null,
        horizonDays: options?.horizonDays ?? null,
        question: market.question,
        endDate: market.endDate ?? null,
        volume: market.volume ?? 0,
        excluded: 'no_terminal_threshold_match',
      });
      continue;
    }
    if (isNaN(matched) || matched <= 0) {
      anchorTrace('extract_market', {
        ticker: options?.ticker ?? null,
        horizonDays: options?.horizonDays ?? null,
        question: market.question,
        endDate: market.endDate ?? null,
        volume: market.volume ?? 0,
        matchedPrice: matched,
        excluded: 'invalid_threshold',
      });
      continue;
    }

    const isYoung =
      market.createdAt != null &&
      now - (typeof market.createdAt === 'string'
        ? Date.parse(market.createdAt)
        : market.createdAt) < 48 * 60 * 60 * 1000;

    const hasVolume = (market.volume ?? 0) > 0;
    const isCrypto = options?.ticker != null
      && getAssetProfile(options.ticker).type === 'crypto';
    const isShortHorizonCrypto = isCrypto
      && options?.horizonDays != null
      && options.horizonDays <= 14;
    const isLongHorizonCrypto = isCrypto
      && options?.horizonDays != null
      && options.horizonDays > 14;
    const isNearTargetResolution = options?.horizonDays != null && market.endDate
      ? Math.abs((Date.parse(market.endDate) - now) / 86_400_000 - options.horizonDays) <= 2
      : false;
    const trustScore: 'high' | 'low' =
      hasVolume && (
        (isLongHorizonCrypto && !isYoung && isNearTargetResolution)
        || (!isLongHorizonCrypto && (!isYoung || (isShortHorizonCrypto && isNearTargetResolution)))
      ) ? 'high' : 'low';

    const rawProbability = market.probability;
    const correctedRaw = rawProbability * YES_BIAS_MULTIPLIER;

    // Convert to survival probability P(> price):
    // "exceed/above $X" at P=p → P(> X) = p
    // "fall below $X" at P=p → P(< X) = p → P(> X) = 1 - p
    const survivalProb = isBelow ? (1 - correctedRaw) : correctedRaw;

    const corrected: PriceThreshold = {
      price: matched,
      rawProbability,
      probability: Math.max(0, Math.min(1, survivalProb)),
      trustScore,
      source: 'polymarket',
      endDate: market.endDate ?? null,
    };

    anchorTrace('extract_market', {
      ticker: options?.ticker ?? null,
      horizonDays: options?.horizonDays ?? null,
      question: market.question,
      endDate: market.endDate ?? null,
      volume: market.volume ?? 0,
      matchedPrice: matched,
      isBelow,
      hasVolume,
      isYoung,
      isNearTargetResolution,
      trustScore,
      rawProbability,
      correctedProbability: corrected.probability,
    });

    const existing = seen.get(matched);
    if (!existing || rawProbability > existing.rawProbability) {
      seen.set(matched, corrected);
    }
  }

  return Array.from(seen.values()).sort((a, b) => a.price - b.price);
}

// ---------------------------------------------------------------------------
// 1b. Crypto terminal-anchor fallback
// ---------------------------------------------------------------------------

/**
 * Market object shape for fallback extraction (matches Polymarket candidate type).
 */
interface FallbackMarket {
  question: string;
  probability: number;
  volume?: number;
  createdAt?: string | number;
  endDate?: string | null;
}

/**
 * For crypto assets only: when strict-horizon candidate filtering yields
 * zero terminal anchors (because all near-horizon markets are barrier-style
 * like "reach $80K" or "dip to $65K"), fall back to the nearest earlier
 * usable terminal-threshold markets ("above $X on April 17").
 *
 * Each fallback anchor receives a date-gap discount that reduces its
 * probability proportional to how far its endDate is from the target horizon:
 *   discount = 1 − 0.03 × max(0, |daysOffset| − 2)
 * clamped to [0.5, 1.0].
 *
 * Gate: only activates for crypto assets (`getAssetProfile(ticker).type === 'crypto'`)
 *       and only when `strictAnchors` is empty.
 *
 * Preserves the distinction between terminal anchors and barrier/path questions —
 * barrier matches from BARRIER_PATTERNS are never included by `extractPriceThresholds`.
 */
export function applyCryptoTerminalAnchorFallback(
  allMarkets: FallbackMarket[],
  strictAnchors: PriceThreshold[],
  ticker: string,
  horizon: number,
  referenceTimeMs?: number,
): PriceThreshold[] {
  if (getAssetProfile(ticker).type !== 'crypto') return strictAnchors;
  if (strictAnchors.length > 0) return strictAnchors;
  if (allMarkets.length === 0) return strictAnchors;

  const fallbackRaw = extractPriceThresholds(allMarkets, { ticker, horizonDays: horizon, referenceTimeMs });
  if (fallbackRaw.length === 0) return strictAnchors;

  const now = referenceTimeMs ?? Date.now();
  const HORIZON_TOLERANCE_DAYS = 2;
  const DISCOUNT_PER_DAY = 0.03;
  const MIN_DISCOUNT = 0.5;

  const discounted: PriceThreshold[] = fallbackRaw.map((anchor) => {
    if (anchor.endDate == null) {
      return { ...anchor, trustScore: 'low' } satisfies PriceThreshold;
    }

    const endMs = Date.parse(anchor.endDate);
    if (Number.isNaN(endMs)) {
      return { ...anchor, trustScore: 'low' } satisfies PriceThreshold;
    }

    const daysUntilResolution = (endMs - now) / 86_400_000;
    const daysOffset = Math.abs(daysUntilResolution - horizon);

    if (daysOffset <= HORIZON_TOLERANCE_DAYS) {
      return anchor;
    }

    const discount = Math.max(
      MIN_DISCOUNT,
      1 - DISCOUNT_PER_DAY * (daysOffset - HORIZON_TOLERANCE_DAYS),
    );
    const adjustedProbability = Math.max(0, Math.min(1, anchor.probability * discount));

    return {
      ...anchor,
      probability: adjustedProbability,
      trustScore: 'low' as const,
    } satisfies PriceThreshold;
  });

  return discounted.sort((a, b) => a.price - b.price);
}

function normalizeSearchIdentityPhrase(ticker: string, phrase: string): string {
  const identity = resolveTickerSearchIdentity(ticker);
  const shouldReplaceTicker = identity.searchQuery === identity.canonicalTicker;
  return normalizeForPolymarket(phrase, shouldReplaceTicker ? identity.canonicalTicker : null);
}

const anchorTrace = process.env.DEBUG_ANCHORS
  ? (...args: unknown[]) => console.error('[ANCHOR_TRACE]', ...args)
  : (..._args: unknown[]) => {};

export function inferPolymarketSearchPhrase(ticker: string): string {
  const identity = resolveTickerSearchIdentity(ticker);
  return normalizeSearchIdentityPhrase(ticker, `${identity.searchQuery} price`);
}

export function buildPolymarketAnchorQueryVariants(
  ticker: string,
  options?: { horizonDays?: number },
): string[] {
  const identity = resolveTickerSearchIdentity(ticker);
  const primary = inferPolymarketSearchPhrase(ticker);
  const manual = [
    normalizeSearchIdentityPhrase(ticker, identity.searchQuery),
    normalizeSearchIdentityPhrase(ticker, `${identity.searchQuery} above`),
    normalizeSearchIdentityPhrase(ticker, `${identity.searchQuery} below`),
  ];

  const signals = extractSignals(identity.searchQuery);
  const isLongHorizonCrypto = options?.horizonDays != null
    && options.horizonDays > 14
    && getAssetProfile(ticker).type === 'crypto';

  // For longer-horizon crypto anchor acquisition, price-target signals must
  // outrank regulatory/macro signals so fetchCandidatePolymarketAnchors'
  // front-6 query slice isn't consumed by "crypto regulation" / "SEC crypto".
  const orderedSignals = isLongHorizonCrypto
    ? reorderCryptoSignalsForAnchorAcquisition(signals)
    : signals;

  const extracted = orderedSignals
    .flatMap((signal) => [signal.searchPhrase, ...(signal.queryVariants ?? [])]);

  const variants = Array.from(new Set([primary, ...manual, ...extracted].filter(Boolean)));
  anchorTrace('query_variants', {
    ticker,
    horizonDays: options?.horizonDays ?? null,
    variants,
    frontSlice: variants.slice(0, 6),
  });
  return variants;
}

/**
 * Reorders crypto signal categories to prioritize price-target and
 * threshold-oriented queries for anchor acquisition. Price-target signals
 * (btc_price_target, etf_product) are promoted ahead of regulatory and
 * macro signals which rarely produce price-anchor-compatible markets.
 */
function reorderCryptoSignalsForAnchorAcquisition(signals: import('./signal-extractor.js').SignalCategory[]): import('./signal-extractor.js').SignalCategory[] {
  const priorityCategories = new Set(['btc_price_target', 'etf_product']);
  const front: typeof signals = [];
  const back: typeof signals = [];
  for (const signal of signals) {
    if (priorityCategories.has(signal.category)) {
      front.push(signal);
    } else {
      back.push(signal);
    }
  }
  return [...front, ...back];
}

function sortMarketsByHorizonCloseness<T extends { endDate?: string | null; volume?: number }>(
  markets: T[],
  horizonDays: number,
): T[] {
  return [...markets].sort((a, b) => {
    const aDays = a.endDate ? (Date.parse(a.endDate) - Date.now()) / 86_400_000 : Number.POSITIVE_INFINITY;
    const bDays = b.endDate ? (Date.parse(b.endDate) - Date.now()) / 86_400_000 : Number.POSITIVE_INFINITY;
    const aDist = Number.isFinite(aDays) ? Math.abs(aDays - horizonDays) : Number.POSITIVE_INFINITY;
    const bDist = Number.isFinite(bDays) ? Math.abs(bDays - horizonDays) : Number.POSITIVE_INFINITY;
    if (aDist !== bDist) return aDist - bDist;
    return (b.volume ?? 0) - (a.volume ?? 0);
  });
}

function filterMarketsToHorizon<T extends { endDate?: string | null }>(
  markets: T[],
  horizonDays: number,
): T[] {
  const strict = markets.filter((market) => {
    if (!market.endDate) return true;
    const endMs = Date.parse(market.endDate);
    if (Number.isNaN(endMs)) return true;
    const daysUntilResolution = (endMs - Date.now()) / 86_400_000;
    return Math.abs(daysUntilResolution - horizonDays) <= Math.max(2, horizonDays * 0.5);
  });

  if (strict.length > 0) return strict;
  return sortMarketsByHorizonCloseness(markets, horizonDays).slice(0, 8);
}

async function fetchCandidatePolymarketAnchors(
  ticker: string,
  horizonDays: number,
): Promise<Array<{
  question: string;
  probability: number;
  volume?: number;
  createdAt?: string | number;
  endDate?: string | null;
}>> {
  const queries = buildPolymarketAnchorQueryVariants(ticker, { horizonDays });
  const isCrypto = getAssetProfile(ticker).type === 'crypto';
  const isLongHorizonCrypto = isCrypto && horizonDays > 14;

  // For long-horizon crypto, constrain API to markets resolving near the target horizon
  // via end_date_min/end_date_max so near-term markets don't crowd out 30-day anchors.
  let endDateFilter: { end_date_min: string; end_date_max: string } | undefined;
  if (isLongHorizonCrypto) {
    const now = Date.now();
    const toleranceDays = Math.max(5, horizonDays * 0.5);
    const minDate = new Date(now + (horizonDays - toleranceDays) * 86_400_000);
    const maxDate = new Date(now + (horizonDays + toleranceDays) * 86_400_000);
    endDateFilter = {
      end_date_min: minDate.toISOString().slice(0, 10),
      end_date_max: maxDate.toISOString().slice(0, 10),
    };
  }

  const frontQueries = queries.slice(0, 6);

  anchorTrace('fetch_candidates_start', {
    ticker,
    horizonDays,
    isCrypto,
    isLongHorizonCrypto,
    endDateFilter: endDateFilter ?? null,
    queries: frontQueries,
  });

  let settled = await Promise.allSettled(
    frontQueries.map((query) => fetchPolymarketAnchorMarkets(query, 40, { ticker, horizonDays, endDateFilter })),
  );

  settled.forEach((result, index) => {
    const query = frontQueries[index];
    if (result.status === 'fulfilled') {
      anchorTrace('fetch_candidates_query_result', {
        ticker,
        horizonDays,
        query,
        returnedCount: result.value.length,
        markets: result.value.map((market) => ({
          question: market.question,
          volume24h: market.volume24h,
          ageDays: market.ageDays ?? null,
          endDate: market.endDate ?? null,
        })),
      });
    } else {
      anchorTrace('fetch_candidates_query_result', {
        ticker,
        horizonDays,
        query,
        error: result.reason instanceof Error ? result.reason.message : String(result.reason),
      });
    }
  });

  if (
    isLongHorizonCrypto
    && endDateFilter
    && settled.every((result) => result.status !== 'fulfilled' || result.value.length === 0)
    && queries.length > frontQueries.length
  ) {
    const retryQueries = queries.slice(frontQueries.length);
    anchorTrace('fetch_candidates_date_retry', {
      ticker,
      horizonDays,
      endDateFilter,
      retryQueries,
    });
    const retryMarkets = await fetchPolymarketAnchorMarketsWithQueries(
      retryQueries,
      40,
      { ticker, horizonDays, endDateFilter },
    );
    anchorTrace('fetch_candidates_date_retry_result', {
      ticker,
      horizonDays,
      retryQueries,
      returnedCount: retryMarkets.length,
      markets: retryMarkets.map((market) => ({
        question: market.question,
        volume24h: market.volume24h,
        ageDays: market.ageDays ?? null,
        endDate: market.endDate ?? null,
      })),
    });
    settled = [
      ...settled,
      {
        status: 'fulfilled',
        value: retryMarkets,
      } satisfies PromiseFulfilledResult<Awaited<ReturnType<typeof fetchPolymarketAnchorMarketsWithQueries>>>,
    ];
  }

  if (
    isLongHorizonCrypto
    && endDateFilter
    && settled.every((result) => result.status !== 'fulfilled' || result.value.length === 0)
  ) {
    anchorTrace('fetch_candidates_undated_fallback', {
      ticker,
      horizonDays,
      queries: frontQueries,
    });
    const fallbackMarkets = await fetchPolymarketAnchorMarketsWithQueries(
      frontQueries,
      40,
      { ticker, horizonDays },
    );
    anchorTrace('fetch_candidates_undated_fallback_result', {
      ticker,
      horizonDays,
      returnedCount: fallbackMarkets.length,
      markets: fallbackMarkets.map((market) => ({
        question: market.question,
        volume24h: market.volume24h,
        ageDays: market.ageDays ?? null,
        endDate: market.endDate ?? null,
      })),
    });
    settled = [
      ...settled,
      {
        status: 'fulfilled',
        value: fallbackMarkets,
      } satisfies PromiseFulfilledResult<Awaited<ReturnType<typeof fetchPolymarketAnchorMarketsWithQueries>>>,
    ];
  }

  const seen = new Set<string>();
  const combined = settled
    .filter((result): result is PromiseFulfilledResult<Awaited<ReturnType<typeof fetchPolymarketAnchorMarkets>>> => result.status === 'fulfilled')
    .flatMap((result) => result.value)
    .map((market) => ({
      question: market.question,
      probability: market.probability,
      volume: market.volume24h,
      createdAt: market.ageDays != null ? Date.now() - market.ageDays * 86_400_000 : undefined,
      endDate: market.endDate ?? null,
    }))
    .filter((market) => {
      if (seen.has(market.question)) return false;
      seen.add(market.question);
      return true;
    });

  anchorTrace('fetch_candidates_combined', {
    ticker,
    horizonDays,
    combinedCount: combined.length,
    combinedMarkets: combined.map((market) => ({
      question: market.question,
      volume: market.volume ?? 0,
      createdAt: market.createdAt ?? null,
      endDate: market.endDate ?? null,
    })),
  });

  let filtered = filterMarketsToHorizon(combined, horizonDays);
  anchorTrace('fetch_candidates_horizon_filter', {
    ticker,
    horizonDays,
    filteredCount: filtered.length,
    filteredMarkets: filtered.map((market) => ({
      question: market.question,
      volume: market.volume ?? 0,
      endDate: market.endDate ?? null,
    })),
  });

  // Crypto fallback: when strict-horizon filter yields only barrier-style markets,
  // also include earlier-dated terminal-anchor-compatible markets so the fallback
  // in computeMarkovDistribution can recover usable anchors.
  if (isCrypto) {
    const strictAnchors = extractPriceThresholds(filtered, { ticker, horizonDays });
    anchorTrace('fetch_candidates_strict_anchors', {
      ticker,
      horizonDays,
      strictAnchorCount: strictAnchors.length,
      strictAnchors: strictAnchors.map((anchor) => ({
        price: anchor.price,
        probability: anchor.probability,
        trustScore: anchor.trustScore,
        endDate: anchor.endDate ?? null,
      })),
    });
    if (strictAnchors.length === 0) {
      const broader = combined.filter((m) => {
        if (!m.endDate) return true;
        const endMs = Date.parse(m.endDate);
        if (Number.isNaN(endMs)) return true;
        const daysUntil = (endMs - Date.now()) / 86_400_000;
        return daysUntil > 0 && daysUntil <= horizonDays * 2;
      });
      anchorTrace('fetch_candidates_broader_fallback', {
        ticker,
        horizonDays,
        broaderCount: broader.length,
        broaderMarkets: broader.map((market) => ({
          question: market.question,
          volume: market.volume ?? 0,
          endDate: market.endDate ?? null,
        })),
      });
      filtered = broader.length > filtered.length ? broader : filtered;
    }
  }

  const sorted = sortMarketsByHorizonCloseness(filtered, horizonDays);
  anchorTrace('fetch_candidates_final', {
    ticker,
    horizonDays,
    finalCount: sorted.length,
    finalMarkets: sorted.map((market) => ({
      question: market.question,
      volume: market.volume ?? 0,
      endDate: market.endDate ?? null,
    })),
  });

  return sorted;
}

/**
 * Normalize Polymarket anchor prices for commodity ETFs.
 *
 * Polymarket gold markets reference futures/per-oz prices (e.g., $5,500 for GC)
 * while GLD trades at ~1/10th of spot gold. This function detects the mismatch
 * and scales anchors into the ETF's price range using the current price as reference.
 *
 * Heuristic: if the median anchor price is >3× the current price, we estimate a
 * conversion factor = currentPrice / medianAnchor and apply it. This is conservative
 * — it only fires when the price scales are obviously different (futures vs ETF).
 */
export function normalizeAnchorPricesForETF(
  anchors: PriceThreshold[],
  currentPrice: number,
  ticker: string,
): PriceThreshold[] {
  if (anchors.length === 0) return anchors;

  // Only apply to known commodity ETFs
  const commodityETFs = new Set([
    'GLD', 'IAU', 'SGOL',   // gold
    'SLV', 'SIVR',           // silver
    'USO', 'BNO',            // oil
    'UNG',                    // natural gas
    'CPER',                   // copper
  ]);
  if (!commodityETFs.has(ticker.toUpperCase())) return anchors;

  // Compute median anchor price (proper average for even-length arrays)
  const sorted = [...anchors].sort((a, b) => a.price - b.price);
  const mid = Math.floor(sorted.length / 2);
  const medianAnchor = sorted.length % 2 === 1
    ? sorted[mid].price
    : (sorted[mid - 1].price + sorted[mid].price) / 2;

  // Only convert if anchors are clearly in a different price scale (>3× current)
  if (medianAnchor <= currentPrice * 3) return anchors;

  // Estimate conversion factor: ratio of current ETF price to anchor median
  // For GLD: ~$415 / ~$4,700 ≈ 0.088 (≈1/10 gold spot → GLD)
  const conversionFactor = currentPrice / medianAnchor;

  return anchors.map((a) => ({
    ...a,
    price: Math.round(a.price * conversionFactor * 100) / 100,
  }));
}

// ---------------------------------------------------------------------------
// 2. classifyRegimeState
// ---------------------------------------------------------------------------

/**
 * Classify a trading day into a directional regime state.
 *
 * Collapsed from the original 5-state model: high_vol_bull and high_vol_bear
 * are merged into bull and bear respectively, since with 120-day windows the
 * high_vol variants had only 2-4 observations and produced noisy transitions.
 *
 * @param dailyReturn - Return for the day (e.g. 0.012 = +1.2%)
 * @param _dailyVolatility - Ignored (kept for backward compatibility)
 * @param returnThreshold - Adaptive threshold for bull/bear classification (default 0.01)
 * @param _volThreshold - Ignored (kept for backward compatibility)
 */
export function classifyRegimeState(
  dailyReturn: number,
  _dailyVolatility: number,
  returnThreshold = 0.01,
  _volThreshold = 0.02,
): RegimeState {
  if (dailyReturn > returnThreshold)  return 'bull';
  if (dailyReturn < -returnThreshold) return 'bear';
  return 'sideways';
}

/**
 * Compute per-asset adaptive thresholds from the return series.
 * Uses half-median of absolute returns for regime classification and
 * 2× median for high-volatility detection. This ensures ~30-40% of days
 * are bull, ~30-40% bear, ~20-30% sideways regardless of asset volatility.
 */
export function computeAdaptiveThresholds(
  returns: number[],
  returnThresholdMultiplier = 0.5,
): {
  returnThreshold: number;
  volThreshold: number;
} {
  if (returns.length === 0) return { returnThreshold: 0.01, volThreshold: 0.02 };
  const absReturns = returns.map(r => Math.abs(r)).sort((a, b) => a - b);
  const medianAbsReturn = absReturns[Math.floor(absReturns.length / 2)];
  return {
    returnThreshold: Math.max(0.001, returnThresholdMultiplier * medianAbsReturn),
    volThreshold: Math.max(0.005, 2.0 * medianAbsReturn),
  };
}

// ---------------------------------------------------------------------------
// 2b. Momentum signal
// ---------------------------------------------------------------------------

export interface MomentumSignal {
  /** Annualized return over lookback window */
  velocity: number;
  /** Change in velocity (recent half vs older half): >0 = accelerating */
  acceleration: number;
  /** R² of OLS regression on log(prices) — measures trend linearity (0–1) */
  trendStrength: number;
  /** Daily drift adjustment to add to regime-weighted mu (clamped ±0.003) */
  adjustment: number;
}

/**
 * Compute a momentum signal from recent prices.
 * Returns a small drift adjustment that tilts the Markov distribution in the
 * direction of the recent trend, weighted by trend linearity (R²).
 *
 * @param prices  Historical prices (at least lookback+1 entries)
 * @param lookback  Number of days to look back (default 20)
 */
export function computeMomentumSignal(prices: number[], lookback = 20): MomentumSignal {
  const nil: MomentumSignal = { velocity: 0, acceleration: 0, trendStrength: 0, adjustment: 0 };
  if (prices.length < lookback + 1) return nil;

  const window = prices.slice(-lookback - 1);

  // Velocity: annualized return over lookback
  const totalReturn = window[window.length - 1] / window[0] - 1;
  const velocity = Math.sign(totalReturn) * (Math.pow(1 + Math.abs(totalReturn), 252 / lookback) - 1);

  // Acceleration: velocity of recent half minus velocity of older half
  const half = Math.floor(lookback / 2);
  const recentRet = window[window.length - 1] / window[window.length - 1 - half] - 1;
  const olderRet  = window[window.length - 1 - half] / window[0] - 1;
  const recentVel = Math.sign(recentRet) * (Math.pow(1 + Math.abs(recentRet), 252 / half) - 1);
  const olderVel  = Math.sign(olderRet) * (Math.pow(1 + Math.abs(olderRet), 252 / half) - 1);
  const acceleration = recentVel - olderVel;

  // Trend strength: R² of OLS on log(prices)
  const logPrices = window.map(p => Math.log(p));
  const n = logPrices.length;
  const xMean = (n - 1) / 2;
  const yMean = logPrices.reduce((s, v) => s + v, 0) / n;
  let sxy = 0, sxx = 0, syy = 0;
  for (let i = 0; i < n; i++) {
    const dx = i - xMean;
    const dy = logPrices[i] - yMean;
    sxy += dx * dy;
    sxx += dx * dx;
    syy += dy * dy;
  }
  const trendStrength = syy > 0 ? (sxy * sxy) / (sxx * syy) : 0;

  // Adjustment: daily drift tilt = velocity/252 × trendStrength × scaling factor
  // Clamped to ±0.003 per day (~±0.75 annualized) to prevent extreme adjustments
  const rawAdj = (velocity / 252) * trendStrength * 0.25;
  const adjustment = Math.max(-0.003, Math.min(0.003, rawAdj));

  return { velocity, acceleration, trendStrength, adjustment };
}

// ---------------------------------------------------------------------------
// 5b. Ensemble signal (Idea D)
// ---------------------------------------------------------------------------

export interface EnsembleSignal {
  /** Mean-reversion z-score: negative = overbought, positive = oversold */
  meanReversionZ: number;
  /** Momentum crossover: SMA5/SMA20 ratio minus 1 */
  momentumCrossover: number;
  /** Volatility compression ratio: vol_5d / vol_20d. <1 = compressed, >1 = expanding */
  volCompression: number;
  /** Combined drift adjustment (daily), clamped to ±0.004 */
  adjustment: number;
  /** Number of signals agreeing on direction (0-3, higher = more consensus) */
  consensus: number;
}

/**
 * Compute ensemble signal from 3 orthogonal indicators:
 * 1. Mean-reversion z-score — contrarian (overbought/oversold)
 * 2. Momentum crossover — trend-following (SMA5 vs SMA20)
 * 3. Volatility compression — breakout anticipation
 *
 * Only applies adjustment when ≥2 of 3 signals agree on direction.
 */
export function computeEnsembleSignal(prices: number[]): EnsembleSignal {
  const nil: EnsembleSignal = { meanReversionZ: 0, momentumCrossover: 0, volCompression: 1, adjustment: 0, consensus: 0 };
  if (prices.length < 25) return nil;

  const returns: number[] = [];
  for (let i = 1; i < prices.length; i++) {
    returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
  }

  // --- Mean-reversion z-score: (price - SMA20) / σ20 ---
  const last20 = prices.slice(-20);
  const sma20 = last20.reduce((s, v) => s + v, 0) / 20;
  const std20 = Math.sqrt(last20.reduce((s, v) => s + (v - sma20) ** 2, 0) / 20);
  const meanReversionZ = std20 > 0 ? (prices[prices.length - 1] - sma20) / std20 : 0;
  // Negative z → price below mean → oversold → bullish signal
  const mrSignal = -meanReversionZ;

  // --- Momentum crossover: SMA5 / SMA20 - 1 ---
  const last5 = prices.slice(-5);
  const sma5 = last5.reduce((s, v) => s + v, 0) / 5;
  const momentumCrossover = sma20 > 0 ? sma5 / sma20 - 1 : 0;
  // Positive crossover → SMA5 above SMA20 → bullish
  const momSignal = momentumCrossover;

  // --- Volatility compression: vol5 / vol20 ---
  const ret5 = returns.slice(-5);
  const ret20 = returns.slice(-20);
  const mean5 = ret5.reduce((s, v) => s + v, 0) / ret5.length;
  const mean20 = ret20.reduce((s, v) => s + v, 0) / ret20.length;
  const vol5 = Math.sqrt(ret5.reduce((s, v) => s + (v - mean5) ** 2, 0) / ret5.length);
  const vol20 = Math.sqrt(ret20.reduce((s, v) => s + (v - mean20) ** 2, 0) / ret20.length);
  const volCompression = vol20 > 0 ? vol5 / vol20 : 1;
  // Low vol compression (<0.7) + trending = breakout imminent → amplify directional signal
  const volAmplifier = volCompression < 0.7 ? 1.5 : volCompression > 1.5 ? 0.5 : 1.0;

  // --- Consensus: count how many signals agree on direction ---
  const bullSignals = [
    mrSignal > 0.3 ? 1 : mrSignal < -0.3 ? -1 : 0,
    momSignal > 0.005 ? 1 : momSignal < -0.005 ? -1 : 0,
    // Vol compression direction: use recent 5d return direction
    ret5.reduce((s, v) => s + v, 0) > 0 ? 1 : -1,
  ] as const;

  const bullCount = bullSignals.filter(s => s > 0).length;
  const bearCount = bullSignals.filter(s => s < 0).length;
  const consensus = Math.max(bullCount, bearCount);

  // --- Combined adjustment: weighted average, only when ≥2 agree ---
  let adjustment = 0;
  if (consensus >= 2) {
    const direction = bullCount >= bearCount ? 1 : -1;
    // Weight: MR 0.4, Momentum 0.4, Vol 0.2
    const rawAdj = direction * (
      0.4 * Math.min(Math.abs(mrSignal), 2) * 0.001 +
      0.4 * Math.min(Math.abs(momSignal), 0.05) * 0.04 +
      0.2 * 0.001
    ) * volAmplifier;
    adjustment = Math.max(-0.004, Math.min(0.004, rawAdj));
  }

  return { meanReversionZ, momentumCrossover, volCompression, adjustment, consensus };
}

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
  minObservations = 30,
  decayRate = 0.97,   // Exponential decay: recent transitions weighted more (1.0 = no decay)
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
  const offDiag  = (1 - diagonal) / (NUM_STATES - 1); // 0.1 for 5 states
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

// ---------------------------------------------------------------------------
// Regime-conditional up-rate (Idea T, Round 4)
// ---------------------------------------------------------------------------

/**
 * Compute empirical P(up | regime) for each regime state from historical data.
 *
 * For each day where regime=R, look forward `horizon` days and check if the
 * cumulative return was positive. This gives the ACTUAL frequency of "up"
 * outcomes following each regime — much more accurate than the lossy
 * regime → mean return → Student-t survival mapping.
 *
 * Returns a map from regime state to P(up|regime). When combined with the
 * n-step transition probabilities, this yields:
 *   P(up) = Σ P(regime_i at horizon) × P(up | regime_i)
 */
export function computeRegimeUpRates(
  regimeSeq: RegimeState[],
  returns: number[],
  horizon: number,
  decayRate?: number,
): Record<RegimeState, number> {
  const counts: Record<RegimeState, { up: number; total: number }> = {
    bull: { up: 0, total: 0 },
    bear: { up: 0, total: 0 },
    sideways: { up: 0, total: 0 },
  };

  // regimeSeq[i] corresponds to returns[i]. Look forward `horizon` days.
  const maxStart = Math.min(regimeSeq.length, returns.length) - horizon;
  for (let i = 0; i < maxStart; i++) {
    const regime = regimeSeq[i];
    // Cumulative return over next `horizon` days (starts at i+1 — i.e., future returns only)
    let cumReturn = 0;
    for (let j = i + 1; j <= i + horizon; j++) {
      cumReturn += returns[j];
    }
    
    // Bounded exponential weighting: recent observations get more weight
    const weight = decayRate !== undefined
      ? Math.pow(decayRate, maxStart - 1 - i)
      : 1;

    counts[regime].total += weight;
    if (cumReturn > 0) counts[regime].up += weight;
  }

  const result = {} as Record<RegimeState, number>;
  for (const state of REGIME_STATES) {
    result[state] = counts[state].total > 0
      ? counts[state].up / counts[state].total
      : 0.5; // no data → uninformative
  }
  return result;
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

  for (let i = 0; i < NUM_STATES; i++) {
    if (rowTotals[i] < 5) continue; // skip rows with too few observations
    for (let j = 0; j < NUM_STATES; j++) {
      const expected = rowTotals[i] * P[i][j];
      if (expected < 1) continue; // skip tiny expected counts (chi-sq unreliable)
      chiSq += (observed[i][j] - expected) ** 2 / expected;
      df += 1;
    }
  }

  // df correction: subtract estimated parameters
  // For each active row we estimated (NUM_STATES-1) free params from that row
  const activeRows = rowTotals.filter(t => t >= 5).length;
  df = Math.max(1, df - activeRows * (NUM_STATES - 1));

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
  decayRate = 0.97,
): StructuralBreakResult {
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
// Tier 1c: mergeAnchorsWithCrossPlatformValidation
// ---------------------------------------------------------------------------

export interface KalshiAnchor {
  price: number;
  probability: number;  // raw probability from Kalshi (no YES-bias correction needed; Kalshi is better calibrated)
  volume?: number;
}

export interface AnchorDivergenceWarning {
  price: number;
  polymarketProb: number;
  kalshiProb: number;
  divergencePp: number;  // absolute difference in percentage points
}

/**
 * Merge Polymarket anchors with Kalshi anchors for the same price levels.
 *
 * Strategy (from DISTILLATION and Saguillo et al. 2025, $40M arbitrage paper):
 * - When both platforms price the same level within 5pp: use the average (both agree)
 * - When divergence > 5pp: use the average AND flag a warning — divergence of this
 *   magnitude is a manipulation or liquidity signal
 * - Kalshi anchors without a Polymarket counterpart: included directly (well-regulated,
 *   better calibrated per Clinton & Huang 2025)
 * - Polymarket-only anchors: retained with their trustScore
 *
 * Note: Kalshi does NOT exhibit the same YES-bias as Polymarket (regulated exchange,
 * professional market-makers), so no 0.95 correction is applied to Kalshi probabilities.
 */
export function mergeAnchorsWithCrossPlatformValidation(
  polymarketAnchors: PriceThreshold[],
  kalshiAnchors: KalshiAnchor[],
  divergenceThresholdPp = 0.05,
): { anchors: PriceThreshold[]; warnings: AnchorDivergenceWarning[] } {
  const warnings: AnchorDivergenceWarning[] = [];
  const PRICE_TOLERANCE = 0.02; // 2% price tolerance for matching levels

  // Build a working copy of Polymarket anchors
  const merged: PriceThreshold[] = polymarketAnchors.map(a => ({ ...a }));

  for (const kalshi of kalshiAnchors) {
    const matchIdx = merged.findIndex(
      a => Math.abs(a.price - kalshi.price) / kalshi.price < PRICE_TOLERANCE,
    );

    if (matchIdx === -1) {
      // Kalshi-only anchor: add directly (Kalshi is well-calibrated, high trust)
      merged.push({
        price: kalshi.price,
        rawProbability: kalshi.probability,
        probability: kalshi.probability,  // no YES-bias correction for Kalshi
        trustScore: (kalshi.volume ?? 0) > 0 ? 'high' : 'low',
        source: 'kalshi',
      });
    } else {
      const poly = merged[matchIdx];
      const divergence = Math.abs(poly.rawProbability - kalshi.probability);

      if (divergence > divergenceThresholdPp) {
        warnings.push({
          price: kalshi.price,
          polymarketProb: poly.rawProbability,
          kalshiProb: kalshi.probability,
          divergencePp: Math.round(divergence * 10000) / 100, // to 2dp
        });
      }

      // Average both platforms' raw probabilities, then apply bias correction
      const averaged = (poly.rawProbability + kalshi.probability) / 2;
      merged[matchIdx] = {
        ...poly,
        rawProbability: averaged,
        probability: averaged * YES_BIAS_MULTIPLIER, // apply bias correction to blended probability
        source: 'averaged',
        // Upgrade to high trust if either source qualifies
        trustScore: poly.trustScore === 'high' || (kalshi.volume ?? 0) > 0 ? 'high' : 'low',
      };
    }
  }

  // Sort by price ascending
  merged.sort((a, b) => a.price - b.price);
  return { anchors: merged, warnings };
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

/**
 * Standard normal CDF Φ(x) via Abramowitz & Stegun erf approximation (eq 7.1.26).
 * The A&S formula computes erf(t) which equals Φ(t√2)*2−1, so we
 * rescale the input by 1/√2 to obtain the true standard normal CDF.
 */
export function normalCDF(x: number): number {
  // Rescale: Φ(x) = 0.5*(1 + erf(x/√2))
  const z = x / Math.SQRT2;
  const a1 = 0.254829592, a2 = -0.284496736, a3 = 1.421413741;
  const a4 = -1.453152027, a5 = 1.061405429, p = 0.3275911;
  const sign = z < 0 ? -1 : 1;
  const t = 1 / (1 + p * Math.abs(z));
  const y = 1 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * Math.exp(-z * z);
  return 0.5 * (1 + sign * y);
}

// ---------------------------------------------------------------------------
// 6. interpolateDistribution
// ---------------------------------------------------------------------------

interface RegimeStats {
  meanReturn: number;  // daily mean log-return in this regime
  stdReturn:  number;  // daily std of log-return in this regime
}

/**
 * Estimate per-regime empirical return statistics from historical data.
 * Falls back to literature-informed defaults when data is sparse.
 */
/**
 * Winsorize an array: clamp values beyond ±k standard deviations to the boundary.
 * Prevents extreme outliers (geopolitical shocks, flash crashes) from contaminating
 * regime statistics.
 */
export function winsorize(values: number[], k = 3.0): number[] {
  if (values.length < 3) return [...values];
  const mean = values.reduce((s, v) => s + v, 0) / values.length;
  const std = Math.sqrt(values.reduce((s, v) => s + (v - mean) ** 2, 0) / values.length);
  if (std < 1e-12) return [...values];
  const lo = mean - k * std;
  const hi = mean + k * std;
  return values.map(v => Math.max(lo, Math.min(hi, v)));
}

export function estimateRegimeStats(
  returns: number[],
  states: RegimeState[],
  maxDailyDrift?: number,
): Record<RegimeState, RegimeStats> {
  const defaults: Record<RegimeState, RegimeStats> = {
    bull:          { meanReturn:  0.005, stdReturn: 0.010 },
    bear:          { meanReturn: -0.005, stdReturn: 0.012 },
    sideways:      { meanReturn:  0.000, stdReturn: 0.006 },
  };

  const bins: Record<RegimeState, number[]> = {
    bull: [], bear: [], sideways: [],
  };

  for (let i = 0; i < Math.min(returns.length, states.length); i++) {
    bins[states[i]].push(returns[i]);
  }

  const result = { ...defaults };
  for (const [state, vals] of Object.entries(bins) as [RegimeState, number[]][]) {
    if (vals.length >= 5) {
      // Winsorize at 3σ to remove shock outliers before computing stats
      const cleaned = winsorize(vals);
      const mean = cleaned.reduce((s, v) => s + v, 0) / cleaned.length;
      const variance = cleaned.reduce((s, v) => s + (v - mean) ** 2, 0) / cleaned.length;
      let cappedMean = mean;
      // Cap daily drift to prevent shock-period contamination
      if (maxDailyDrift !== undefined && maxDailyDrift > 0) {
        cappedMean = Math.max(-maxDailyDrift, Math.min(maxDailyDrift, mean));
      }
      result[state] = { meanReturn: cappedMean, stdReturn: Math.sqrt(variance) };
    }
  }
  return result;
}

/**
 * Compute the mixing-time weight: how much to trust the Markov regime signal
 * vs. anchor-only at a given horizon.
 *
 * weight = exp(−ρ × n) where ρ = second-largest eigenvalue of P.
 * Near 1 at short horizons (Markov-dominant).
 * Approaches 0 at long horizons (Polymarket anchors dominate).
 */
export function computeMixingWeight(secondEigenvalue: number, horizon: number): number {
  return Math.exp(-secondEigenvalue * horizon);
}

/**
 * Log-normal survival function: P(price > X | current price S₀, drift μ_n, vol σ_n).
 * P(X > target) = 1 − Φ( (ln(target/S₀) − μ_n) / σ_n )
 */
export function logNormalSurvival(
  currentPrice: number,
  targetPrice: number,
  driftN: number,    // n-day log-space drift
  volN: number,      // n-day log-space vol
): number {
  if (volN <= 0) return targetPrice < currentPrice ? 1 : 0;
  const z = (Math.log(targetPrice / currentPrice) - driftN) / volN;
  return 1 - normalCDF(z);
}

/**
 * Regularized incomplete beta function I_x(a, b) via continued fraction.
 * Used to compute Student-t CDF. Lentz's method for convergence.
 */
function regularizedBeta(x: number, a: number, b: number): number {
  if (x <= 0) return 0;
  if (x >= 1) return 1;

  // Use symmetry relation when x > (a+1)/(a+b+2) for better convergence
  if (x > (a + 1) / (a + b + 2)) {
    return 1 - regularizedBeta(1 - x, b, a);
  }

  const lnBeta = lgamma(a) + lgamma(b) - lgamma(a + b);
  const front = Math.exp(Math.log(x) * a + Math.log(1 - x) * b - lnBeta) / a;

  // Lentz's continued fraction
  let f = 1, c = 1, d = 1 - (a + b) * x / (a + 1);
  if (Math.abs(d) < 1e-30) d = 1e-30;
  d = 1 / d;
  f = d;

  for (let m = 1; m <= 200; m++) {
    // Even step
    let numerator = m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m));
    d = 1 + numerator * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + numerator / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    f *= c * d;

    // Odd step
    numerator = -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1));
    d = 1 + numerator * d;
    if (Math.abs(d) < 1e-30) d = 1e-30;
    c = 1 + numerator / c;
    if (Math.abs(c) < 1e-30) c = 1e-30;
    d = 1 / d;
    const delta = c * d;
    f *= delta;

    if (Math.abs(delta - 1) < 1e-10) break;
  }

  return front * f;
}

/** Log-gamma via Stirling's approximation (Lanczos coefficients). */
function lgamma(x: number): number {
  const g = 7;
  const coef = [
    0.99999999999980993, 676.5203681218851, -1259.1392167224028,
    771.32342877765313, -176.61502916214059, 12.507343278686905,
    -0.13857109526572012, 9.9843695780195716e-6, 1.5056327351493116e-7,
  ];
  if (x < 0.5) {
    return Math.log(Math.PI / Math.sin(Math.PI * x)) - lgamma(1 - x);
  }
  x -= 1;
  let a = coef[0];
  for (let i = 1; i < g + 2; i++) {
    a += coef[i] / (x + i);
  }
  const t = x + g + 0.5;
  return 0.5 * Math.log(2 * Math.PI) + (x + 0.5) * Math.log(t) - t + Math.log(a);
}

/**
 * Student-t CDF: P(T ≤ x) for T ~ t(ν degrees of freedom).
 * Uses regularized incomplete beta function.
 */
export function studentTCDF(x: number, nu: number): number {
  if (nu <= 0) return normalCDF(x); // degenerate: fall back to normal
  const t2 = x * x;
  const betaArg = nu / (nu + t2);
  const ibeta = regularizedBeta(betaArg, nu / 2, 0.5);
  if (x >= 0) {
    return 1 - 0.5 * ibeta;
  } else {
    return 0.5 * ibeta;
  }
}

/**
 * Inverse Student-t CDF via bisection: find x such that CDF(x, nu) = p.
 * Used for drift-based calibration to convert a target P(up) into a drift value.
 */
export function inverseStudentTCDF(p: number, nu: number): number {
  if (p <= 0) return -50;
  if (p >= 1) return 50;
  if (Math.abs(p - 0.5) < 1e-12) return 0;
  let lo = -50, hi = 50;
  for (let iter = 0; iter < 100; iter++) {
    const mid = (lo + hi) / 2;
    const cdf = studentTCDF(mid, nu);
    if (cdf < p) lo = mid;
    else hi = mid;
    if (hi - lo < 1e-10) break;
  }
  return (lo + hi) / 2;
}

/**
 * Fat-tailed survival function: P(price > X) using Student-t distribution.
 * Same interface as logNormalSurvival but uses Student-t with `nu` degrees
 * of freedom (default 5, typical for daily equity returns).
 *
 * The scaling adjusts the t-distribution standard deviation to match
 * the Gaussian vol parameter: σ_t = σ_n × sqrt((ν-2)/ν) for ν>2.
 */
export function studentTSurvival(
  currentPrice: number,
  targetPrice: number,
  driftN: number,
  volN: number,
  nu = 5,
): number {
  if (volN <= 0) return targetPrice < currentPrice ? 1 : 0;
  if (targetPrice <= 0) return 1; // price can't go below 0; survival = 1
  // Scale vol to match t-distribution variance: Var(t_ν) = ν/(ν-2) for ν>2
  const scaledVol = nu > 2 ? volN * Math.sqrt((nu - 2) / nu) : volN;
  const z = (Math.log(targetPrice / currentPrice) - driftN) / scaledVol;
  // Guard extreme z-scores: regularizedBeta can diverge beyond |z|~50
  if (!Number.isFinite(z)) return z > 0 ? 0 : 1;
  const clamped = Math.max(-50, Math.min(50, z));
  const cdf = studentTCDF(clamped, nu);
  return Number.isFinite(cdf) ? 1 - cdf : (clamped > 0 ? 0 : 1);
}

/**
 * Run a single Monte Carlo random walk through the transition matrix for n steps,
 * starting from `initialStateIdx`. Returns the final n-step regime weight vector.
 */
function singleMarkovWalk(
  P: TransitionMatrix,
  initialStateIdx: number,
  n: number,
): number[] {
  // Use the n-step transition row for the initial state
  const Pn = matPow(P, n);
  return Pn[initialStateIdx];
}

/**
 * Compute the initial probability vector over the current regime using the last K observed states.
 * This replaces a hard start state (one-hot) with a smoothed mixture.
 */
export function computeStartStateMixture(
  recentStates: RegimeState[],
  alpha = 0.5
): Record<RegimeState, number> {
  const counts: Record<RegimeState, number> = { bull: alpha, bear: alpha, sideways: alpha } as Record<RegimeState, number>;
  for (const s of recentStates) {
    if (counts[s] !== undefined) counts[s]++;
  }
  const total = counts.bull + counts.bear + counts.sideways;
  return {
    bull: counts.bull / total,
    bear: counts.bear / total,
    sideways: counts.sideways / total,
  };
}

/**
 * Compute the effective n-step drift and volatility from the Markov chain.
 * Extracted so both interpolateDistribution and calibration logic can reuse it.
 */
export function computeHorizonDriftVol(
  horizon: number,
  P: TransitionMatrix,
  regimeStats: Record<RegimeState, RegimeStats>,
  initialState: RegimeState,
  momentumAdjustment = 0,
  hmmOverride?: { drift: number; vol: number; weight: number },
  startMixture?: Record<RegimeState, number>,
  regimeSpecificSigma?: boolean,
  regimeSpecificSigmaThreshold?: number,
): { mu_n: number; sigma_n: number } {
  const Pn = matPow(P, horizon);
  
  let stateWeights: number[];
  if (startMixture) {
    stateWeights = [0, 0, 0];
    for (const state of REGIME_STATES) {
      const w = startMixture[state];
      const idx = STATE_INDEX[state];
      for (let j = 0; j < 3; j++) {
        stateWeights[j] += w * Pn[idx][j];
      }
    }
  } else {
    stateWeights = Pn[STATE_INDEX[initialState]];
  }

  const mu_obs = REGIME_STATES.reduce(
    (s, state, i) => s + stateWeights[i] * regimeStats[state].meanReturn, 0,
  );
  // Variance of the mixture: E[σ²] + Var(μ).
  // E[σ²] captures within-regime volatility; Var(μ) captures between-regime
  // mean differences — critical when regime weights are mixed and means are well-separated.
  const varOfMeans = REGIME_STATES.reduce(
    (s, state, i) => s + stateWeights[i] * (regimeStats[state].meanReturn - mu_obs) ** 2, 0,
  );
  const mixtureSigmaObs = Math.sqrt(
    REGIME_STATES.reduce(
      (s, state, i) => s + stateWeights[i] * regimeStats[state].stdReturn ** 2, 0,
    ) + varOfMeans,
  );

  // Phase 7: when regime weights are concentrated and flag is enabled,
  // use the dominant regime's own sigma instead of the mixture sigma.
  // The mixture sigma inflates variance via Var(μ) when weights are mixed,
  // but when one regime dominates, that regime's own volatility is more appropriate.
  const maxWeight = Math.max(...stateWeights);
  const threshold = regimeSpecificSigmaThreshold ?? 0.60;
  const dominantIdx = stateWeights.indexOf(maxWeight);
  const dominantSigma = regimeStats[REGIME_STATES[dominantIdx]].stdReturn;
  const useRegimeSigma = regimeSpecificSigma === true && maxWeight > threshold;
  const sigma_obs = useRegimeSigma ? dominantSigma : mixtureSigmaObs;

  let mu_eff: number;
  let sigma_eff: number;
  if (hmmOverride) {
    const w = hmmOverride.weight;
    mu_eff = w * hmmOverride.drift + (1 - w) * mu_obs;
    sigma_eff = w * hmmOverride.vol + (1 - w) * sigma_obs;
  } else {
    mu_eff = mu_obs;
    sigma_eff = sigma_obs;
  }

  return {
    mu_n: horizon * (mu_eff + momentumAdjustment),
    sigma_n: sigma_eff * Math.sqrt(horizon),
  };
}

/**
 * Compute a day-by-day price trajectory for days 1..N.
 *
 * Uses a SINGLE set of Monte Carlo random walks and samples the path at each day,
 * rather than N independent simulations. This ensures:
 * 1. CI widths monotonically increase with horizon
 * 2. ~7× faster than N separate MC runs
 *
 * Returns one TrajectoryPoint per day with expected price, 90% CI, P(up), and regime.
 */
export function computeTrajectory(
  currentPrice: number,
  days: number,
  P: TransitionMatrix,
  regimeStats: Record<RegimeState, { meanReturn: number; stdReturn: number }>,
  initialState: RegimeState,
  momentumAdjustment: number,
  hmmOverride?: { drift: number; vol: number; weight: number },
  nSamples = 1000,
  nu = 5,
  empiricalDailyVol?: number,
  startMixture?: Record<RegimeState, number>,
): TrajectoryPoint[] {
  const initialIdx = STATE_INDEX[initialState];
  const trajectory: TrajectoryPoint[] = [];

  // Pre-compute regime weights at each day via matrix powers
  const regimeWeightsPerDay: number[][] = [];
  for (let d = 1; d <= days; d++) {
    const Pd = matPow(P, d);
    if (startMixture) {
      const weights = [0, 0, 0];
      for (const state of REGIME_STATES) {
        const w = startMixture[state];
        const idx = STATE_INDEX[state];
        for (let j = 0; j < 3; j++) {
          weights[j] += w * Pd[idx][j];
        }
      }
      regimeWeightsPerDay.push(weights);
    } else {
      regimeWeightsPerDay.push(Pd[initialIdx]);
    }
  }

  // Compute 1-day regime-weighted drift and vol for MC steps
  const { mu_n: drift1d, sigma_n: regimeVol1d } = computeHorizonDriftVol(
    1, P, regimeStats, initialState, momentumAdjustment, hmmOverride, startMixture
  );

  // Use max(regime-weighted vol, empirical vol) for MC paths.
  // Regime-weighted vol = E[Var(R|S)] misses between-state variance Var(E[R|S]),
  // causing systematically narrow CIs. Empirical vol captures total variance.
  const mcVol = empiricalDailyVol ? Math.max(regimeVol1d, empiricalDailyVol) : regimeVol1d;

  // Run shared Monte Carlo: generate nSamples random walks, each of length `days`
  // Using Student-t random variates for fat tails
  const paths: number[][] = []; // paths[sample][day] = log(price/currentPrice)
  for (let s = 0; s < nSamples; s++) {
    const path = new Array(days);
    let cumLogReturn = 0;
    for (let d = 0; d < days; d++) {
      // Student-t variate via inverse CDF of uniform
      const u = Math.random();
      const z = inverseStudentTCDF(u, nu);
      const scaledVol = nu > 2 ? mcVol * Math.sqrt((nu - 2) / nu) : mcVol;
      cumLogReturn += drift1d + z * scaledVol;
      path[d] = cumLogReturn;
    }
    paths.push(path);
  }

  for (let d = 1; d <= days; d++) {
    const dayIdx = d - 1;
    const stateWeights = regimeWeightsPerDay[dayIdx];

    // Drift and vol at this horizon
    const { mu_n, sigma_n } = computeHorizonDriftVol(
      d, P, regimeStats, initialState, momentumAdjustment, hmmOverride, startMixture,
    );

    // Expected price from analytical drift
    const analyticalExpected = currentPrice * Math.exp(mu_n);

    // CI bounds from Monte Carlo paths
    const prices = paths.map(path => currentPrice * Math.exp(path[dayIdx]));
    prices.sort((a, b) => a - b);
    const p5Idx = Math.max(0, Math.floor(nSamples * 0.05) - 1);
    const p50Idx = Math.floor(nSamples * 0.5);
    const p95Idx = Math.min(nSamples - 1, Math.ceil(nSamples * 0.95));
    const lowerBound = prices[p5Idx];
    const upperBound = prices[p95Idx];

    // Use MC median when empirical vol is used (more consistent with MC bounds),
    // otherwise use the analytical expected price
    const expectedPrice = empiricalDailyVol ? prices[p50Idx] : analyticalExpected;

    // P(up) from Student-t survival at currentPrice
    const pUp = studentTSurvival(currentPrice, currentPrice, mu_n, sigma_n, nu);

    // Cumulative return
    const ret = (expectedPrice - currentPrice) / currentPrice;
    const sign = ret >= 0 ? '+' : '';
    const cumulativeReturn = `${sign}${(ret * 100).toFixed(1)}%`;

    // Most likely regime at this horizon
    let maxWeight = -1;
    let regime: RegimeState = initialState;
    REGIME_STATES.forEach((state, i) => {
      if (stateWeights[i] > maxWeight) {
        maxWeight = stateWeights[i];
        regime = state;
      }
    });

    trajectory.push({
      day: d,
      expectedPrice: Math.round(expectedPrice * 100) / 100,
      lowerBound: Math.round(lowerBound * 100) / 100,
      upperBound: Math.round(upperBound * 100) / 100,
      pUp: Math.round(pUp * 1000) / 1000,
      cumulativeReturn,
      regime,
    });
  }

  return trajectory;
}

/**
 * Build probability distribution across price levels, blending Markov estimates
 * with Polymarket anchors using the mixing-time weight.
 *
 * Confidence intervals are computed via Monte Carlo simulation (N=1000 walks),
 * taking the 5th/95th percentile of the resulting probability distribution.
 */
export function interpolateDistribution(
  currentPrice: number,
  horizon: number,
  P: TransitionMatrix,
  regimeStats: Record<RegimeState, RegimeStats>,
  initialState: RegimeState,
  anchors: PriceThreshold[],
  secondEigenvalue: number,
  numLevels = 20,
  monteCarloSamples = 1000,
  ciWidthMultiplier = 1.0,
  momentumAdjustment = 0,
  hmmOverride?: { drift: number; vol: number; weight: number },
  dailyVol?: number,
  startMixture?: Record<RegimeState, number>,
  nu = 5,
  regimeSpecificSigma?: boolean,
  regimeSpecificSigmaThreshold?: number,
): MarkovDistributionPoint[] {
  // Adaptive grid: scale with volatility so CI covers ≥3σ for all assets.
  // Fixed 1.5%/step only covers ±14% total — fine for SPY (~1%/day) but
  // too narrow for TSLA (~3%/day) or BTC (~4%/day) where a 14-day 2σ move is 22-30%.
  const vol = dailyVol ?? 0.015;
  const volRange = 3.5 * vol * Math.sqrt(horizon);
  // Clamp to [0.15, 0.90] — minPrice must remain positive (>10% of currentPrice)
  const halfRange = Math.max(0.15, Math.min(0.90, volRange));
  let minPrice = currentPrice * (1 - halfRange);
  let maxPrice = currentPrice * (1 + halfRange);

  // Extend grid range to include all Polymarket anchors (fixes sparse-anchor bug)
  for (const a of anchors) {
    if (a.price < minPrice) minPrice = a.price * 0.95;
    if (a.price > maxPrice) maxPrice = a.price * 1.05;
  }

  const prices: number[] = [];
  for (let i = 0; i <= numLevels; i++) {
    prices.push(minPrice * Math.pow(maxPrice / minPrice, i / numLevels));
  }

  // Merge anchor prices into the grid so they are never missed
  for (const a of anchors) {
    const closestDist = prices.reduce(
      (best, p) => Math.min(best, Math.abs(p - a.price) / a.price), Infinity,
    );
    if (closestDist > 0.005) prices.push(a.price);
  }
  prices.sort((a, b) => a - b);

  const mixWeight = computeMixingWeight(secondEigenvalue, horizon);

  // Compute regime-weighted drift and vol via shared helper
  const { mu_n, sigma_n } = computeHorizonDriftVol(
    horizon, P, regimeStats, initialState, momentumAdjustment, hmmOverride, startMixture,
    regimeSpecificSigma, regimeSpecificSigmaThreshold,
  );

  // Nearest anchor lookup helper with distance-based dampening.
  // Anchors far from current price are less trustworthy (often illiquid/speculative).
  // Apply exponential decay: weight = exp(-k * distance²) where distance = |price - current| / current.
  // At 20% away: weight ≈ 0.67. At 40%: weight ≈ 0.20. At 60%+: weight ≈ 0.03.
  const findAnchor = (price: number) => {
    const TOLERANCE_PCT = 0.02;
    const raw = anchors.find(a => Math.abs(a.price - price) / price < TOLERANCE_PCT);
    if (!raw) return undefined;
    // Compute distance-decay factor
    const distFromCurrent = Math.abs(raw.price - currentPrice) / currentPrice;
    const DISTANCE_DECAY_K = 5.0; // controls how fast far anchors are dampened
    const distanceWeight = Math.exp(-DISTANCE_DECAY_K * distFromCurrent * distFromCurrent);
    return { ...raw, distanceWeight };
  };

  // Monte Carlo: vary initial state draw for CI
  const rng = (): number => Math.random();
  const ciSamples: Map<number, number[]> = new Map(prices.map(p => [p, []]));

  for (let s = 0; s < monteCarloSamples; s++) {
    // Perturb drift and vol within sampling uncertainty
    const perturbedMu  = mu_n    + (rng() - 0.5) * sigma_n * 0.2;
    const perturbedVol = sigma_n * (0.9 + rng() * 0.2);
    for (const price of prices) {
      const p = studentTSurvival(currentPrice, price, perturbedMu, perturbedVol, nu);
      ciSamples.get(price)!.push(p);
    }
  }

  // Build distribution points
  const rawPoints = prices.map(price => {
    const anchor = findAnchor(price);
    const markovEst = studentTSurvival(currentPrice, price, mu_n, sigma_n, nu);

    let probability: number;
    let source: 'polymarket' | 'markov' | 'blend';

    if (anchor && anchor.trustScore === 'high') {
      // Scale anchor influence by distance from current price
      const anchorW = (1 - mixWeight) * anchor.distanceWeight;
      probability = (1 - anchorW) * markovEst + anchorW * anchor.probability;
      source = anchorW < 0.05 ? 'markov' : anchorW > 0.5 ? 'polymarket' : 'blend';
    } else if (anchor && anchor.trustScore === 'low') {
      // Low-trust anchors: half nominal influence, further scaled by distance
      const anchorW = (1 - mixWeight) * 0.5 * anchor.distanceWeight;
      probability = (1 - anchorW) * markovEst + anchorW * anchor.probability;
      source = 'blend';
    } else {
      probability = markovEst;
      source = 'markov';
    }

    const samples = ciSamples.get(price)!.sort((a, b) => a - b);
    const lo = samples[Math.floor(0.05 * samples.length)];
    const hi = samples[Math.floor(0.95 * samples.length)];

    // Apply CI widening multiplier (used when structural break detected)
    const halfWidth = (hi - lo) / 2;
    const center = (hi + lo) / 2;
    const widenedLo = Math.max(0, center - halfWidth * ciWidthMultiplier);
    const widenedHi = Math.min(1, center + halfWidth * ciWidthMultiplier);

    return { price, probability, lowerBound: widenedLo, upperBound: widenedHi, source };
  });

  // Enforce monotonicity: P(price > X) must be non-increasing in X
  for (let i = rawPoints.length - 2; i >= 0; i--) {
    if (rawPoints[i].probability < rawPoints[i + 1].probability) {
      rawPoints[i].probability = rawPoints[i + 1].probability;
    }
  }

  return rawPoints;
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

function computeValidationR2OS(params: {
  assetType: AssetProfile['type'];
  horizon: number;
  regimeSeq: RegimeState[];
  logReturns: number[];
  assetProfile: AssetProfile;
  transitionDecayOverride?: number;
}): { r2os: number | null; validationMetric: 'daily_return' | 'horizon_return' } {
  const { assetType, horizon, regimeSeq, logReturns, assetProfile, transitionDecayOverride } = params;
  const effectiveDecayRate = transitionDecayOverride ?? 0.97;

  const useHorizonValidator = assetType === 'crypto' && horizon >= 7 && horizon <= 14;
  if (useHorizonValidator) {
    const heldoutDays = Math.min(logReturns.length - 30, Math.max(12 * horizon, 84));
    if (logReturns.length >= 30 + heldoutDays) {
      const actuals: number[] = [];
      const predicted: number[] = [];
      const baseline: number[] = [];
      const startIdx = logReturns.length - heldoutDays;

      for (let start = startIdx; start + horizon <= logReturns.length; start += horizon) {
        const trainStates = regimeSeq.slice(0, start);
        const trainLogReturns = logReturns.slice(0, start);
        if (trainStates.length < 20 || trainLogReturns.length < 20) continue;

        const trainP = estimateTransitionMatrix(trainStates, undefined, 30, effectiveDecayRate);
        const trainRegimeStats = estimateRegimeStats(trainLogReturns, trainStates, assetProfile.maxDailyDrift);
        const lastTrainState = trainStates[trainStates.length - 1];
        const trainMean = trainLogReturns.reduce((s, v) => s + v, 0) / trainLogReturns.length;
        const realizedHorizonReturn = logReturns.slice(start, start + horizon).reduce((s, v) => s + v, 0);
        const { mu_n } = computeHorizonDriftVol(
          horizon,
          trainP,
          trainRegimeStats,
          lastTrainState,
          0,
          undefined,
        );

        actuals.push(realizedHorizonReturn);
        predicted.push(mu_n);
        baseline.push(horizon * trainMean);
      }

      if (actuals.length >= 6) {
        return {
          r2os: computeR2OS(actuals, predicted) - computeR2OS(actuals, baseline),
          validationMetric: 'horizon_return',
        };
      }
    }
  }

  const minHeldOut = 20;
  if (regimeSeq.length >= minHeldOut + 30) {
    const trainStates  = regimeSeq.slice(0, -minHeldOut);
    const trainReturns = logReturns.slice(0, -minHeldOut);
    const testReturns  = logReturns.slice(-minHeldOut);

    const trainP    = estimateTransitionMatrix(trainStates, undefined, 30, effectiveDecayRate);
    const trainRegimeStats = estimateRegimeStats(trainReturns, trainStates, assetProfile.maxDailyDrift);
    const trainMean = trainReturns.reduce((s, v) => s + v, 0) / trainReturns.length;
    const predicted = testReturns.map((_, i) => {
      const Pn = matPow(trainP, i + 1);
      const stateIdx = STATE_INDEX[trainStates[trainStates.length - 1]];
      const weights  = Pn[stateIdx];
      return REGIME_STATES.reduce(
        (s, state, j) => s + weights[j] * trainRegimeStats[state].meanReturn, 0,
      );
    });
    const baseline = Array(minHeldOut).fill(trainMean);
    return {
      r2os: computeR2OS(testReturns, predicted) - computeR2OS(testReturns, baseline),
      validationMetric: 'daily_return',
    };
  }

  return { r2os: null, validationMetric: 'daily_return' };
}

// ---------------------------------------------------------------------------
// 7b. Bayesian probability calibration (Idea I)
// ---------------------------------------------------------------------------

/**
 * Calibrate raw survival probabilities via Bayesian shrinkage toward the base rate.
 *
 * The raw Markov model is overconfident at extremes: predicted P=0.04 actually resolves
 * at ~0.47, and predicted P=0.95 resolves at ~0.59 (per walk-forward backtest). This is
 * a classic calibration failure — the model rank-orders correctly but has too-wide spread.
 *
 * Fix: shrink raw probabilities toward 0.5 (uninformative prior).
 *   calibrated = prior_weight × 0.5 + (1 - prior_weight) × raw
 *
 * prior_weight is controlled by:
 *   - Base shrinkage: always pull extremes toward 0.5 somewhat (κ)
 *   - Ensemble consensus: higher consensus → less shrinkage (more confident)
 *   - Data sufficiency: more historical data → less shrinkage
 *
 * After shrinkage, monotonicity is re-enforced (P(>X) must be non-increasing in X).
 */
export function calibrateProbabilities(
  distribution: MarkovDistributionPoint[],
  options?: {
    ensembleConsensus?: number;   // 0-3: how many ensemble signals agree
    historicalDays?: number;      // number of daily returns available
    hmmConverged?: boolean;       // whether HMM converged (adds confidence)
    baseRate?: number;            // empirical P(up) from recent history (default 0.5)
    kappaMultiplier?: number;     // asset-profile kappa scaling (Idea N). >1 = more shrinkage.
    currentRegime?: string;       // Idea O: regime-gated kappa adjustment
    matureBullCalibrationActive?: boolean; // PR3: BTC 14d mature bull calibration
    // Drift-based calibration params (preserves distribution S-shape)
    currentPrice?: number;        // required for drift-based mode
    driftN?: number;              // n-step drift (mu_n) from computeHorizonDriftVol
    volN?: number;                // n-step volatility (sigma_n)
    nu?: number;                 // Student-t degrees of freedom (defaults to asset profile value)
  },
): MarkovDistributionPoint[] {
  const consensus = options?.ensembleConsensus ?? 0;
  const nDays = options?.historicalDays ?? 60;
  const hmmOk = options?.hmmConverged ?? false;
  const kappaScale = options?.kappaMultiplier ?? 1.0;
  // Adaptive center: shrink toward empirical base rate, not 0.5.
  // Idea S (Round 4): Raised center cap from [0.35, 0.65] to [0.25, 0.80].
  const center = Math.max(0.25, Math.min(0.80, options?.baseRate ?? 0.5));

  // Base shrinkage coefficient κ ∈ [0.15, 0.55]
  let kappa = 0.45;

  // Less shrinkage when ensemble signals agree (each consensus point → -0.07)
  kappa -= consensus * 0.07;

  // Less shrinkage with more data (logarithmic scaling — diminishing returns)
  if (nDays > 60) {
    kappa -= Math.min(0.08, 0.04 * Math.log2(nDays / 60));
  }

  // HMM convergence adds a small confidence boost
  if (hmmOk) kappa -= 0.03;

  // Asset-profile scaling (Idea N): multiply kappa by profile factor
  kappa *= kappaScale;

  // Regime-gated adjustment (Idea O)
  const regime = options?.currentRegime;
  if (options?.matureBullCalibrationActive) {
    // PR3: Mature bull calibration - stop lower-shrinkage bonus, apply extra shrinkage
    kappa += 0.10;
  } else if (regime === 'bull' || regime === 'bear') {
    kappa -= 0.03;
  } else if (regime === 'sideways') {
    kappa += 0.03;
  }

  // Clamp to valid range
  kappa = Math.max(0.15, Math.min(0.55, kappa));

  // --- Drift-based calibration (preserves S-shape) ---
  // When currentPrice, driftN, and volN are provided, calibrate by shifting
  // the distribution's drift instead of compressing each probability independently.
  // This preserves the survival curve shape: far-below prices stay ~99%,
  // far-above prices stay ~1%, while P(up) at currentPrice is calibrated toward center.
  const cp = options?.currentPrice;
  const driftN = options?.driftN;
  const volN = options?.volN;

  if (cp != null && driftN != null && volN != null && volN > 0) {
    const nu = options?.nu ?? 5; // Student-t degrees of freedom
    const rawPUp = studentTSurvival(cp, cp, driftN, volN, nu);
    const targetPUp = Math.max(0.01, Math.min(0.99, kappa * center + (1 - kappa) * rawPUp));

    // Find the drift that produces targetPUp via inverse CDF:
    //   targetPUp = 1 - CDF(-calibratedDrift / scaledVol)
    //   CDF(-calibratedDrift / scaledVol) = 1 - targetPUp
    //   -calibratedDrift / scaledVol = inverseStudentTCDF(1 - targetPUp, nu)
    const scaledVol = nu > 2 ? volN * Math.sqrt((nu - 2) / nu) : volN;
    const zTarget = inverseStudentTCDF(1 - targetPUp, nu);
    const calibratedDrift = -zTarget * scaledVol;

    const calibrated = distribution.map(point => {
      let newProb: number;
      if (point.source === 'markov') {
        // Pure Markov point: recompute survival with calibrated drift
        newProb = studentTSurvival(cp, point.price, calibratedDrift, volN, nu);
      } else {
        // Blended/polymarket point: apply additive delta to preserve anchor contribution
        const oldMarkov = studentTSurvival(cp, point.price, driftN, volN, nu);
        const newMarkov = studentTSurvival(cp, point.price, calibratedDrift, volN, nu);
        newProb = Math.max(0, Math.min(1, point.probability + (newMarkov - oldMarkov)));
      }
      // Shift CI bounds by the same delta so they remain consistent with the point estimate.
      // Without this, calibrated probability can fall outside the raw MC CI bounds.
      const delta = newProb - point.probability;
      const newLower = Math.max(0, Math.min(1, (point.lowerBound ?? newProb) + delta));
      const newUpper = Math.max(0, Math.min(1, (point.upperBound ?? newProb) + delta));
      return {
        ...point,
        probability: newProb,
        lowerBound: Math.min(newLower, newProb),
        upperBound: Math.max(newUpper, newProb),
      };
    });

    // Re-enforce monotonicity (adjust bounds to stay consistent)
    for (let i = calibrated.length - 2; i >= 0; i--) {
      if (calibrated[i].probability < calibrated[i + 1].probability) {
        calibrated[i].probability = calibrated[i + 1].probability;
        // Ensure bounds still bracket the probability after adjustment
        if (calibrated[i].upperBound != null && calibrated[i].upperBound < calibrated[i].probability) {
          calibrated[i].upperBound = calibrated[i].probability;
        }
        if (calibrated[i].lowerBound != null && calibrated[i].lowerBound > calibrated[i].probability) {
          calibrated[i].lowerBound = calibrated[i].probability;
        }
      }
    }
    return calibrated;
  }

  // --- Legacy fallback: probability-level calibration ---
  // Used when drift params are not available (e.g., unit tests with synthetic data)
  const calibrated = distribution.map(point => {
    const newProb = kappa * center + (1 - kappa) * point.probability;
    const delta = newProb - point.probability;
    const newLower = Math.max(0, Math.min(1, (point.lowerBound ?? newProb) + delta));
    const newUpper = Math.max(0, Math.min(1, (point.upperBound ?? newProb) + delta));
    return {
      ...point,
      probability: newProb,
      lowerBound: Math.min(newLower, newProb),
      upperBound: Math.max(newUpper, newProb),
    };
  });

  // Re-enforce monotonicity after shrinkage (P(>X) non-increasing in X)
  for (let i = calibrated.length - 2; i >= 0; i--) {
    if (calibrated[i].probability < calibrated[i + 1].probability) {
      calibrated[i].probability = calibrated[i + 1].probability;
    }
  }

  return calibrated;
}

// ---------------------------------------------------------------------------
// 7c. Prediction confidence scoring (Idea M — sHMM selective prediction)
// ---------------------------------------------------------------------------

/**
 * Compute a 0–1 confidence score for a Markov prediction. Higher = more reliable.
 *
 * Combines six orthogonal signals:
 *   1. **Decisiveness** (30%): |P(up) − 0.5| × 2 — how far from a coin flip.
 *   2. **Ensemble consensus** (15%): fraction of signals (momentum, mean-reversion,
 *      crossover) that agree. consensus=3 → all agree → max contribution.
 *   3. **HMM convergence** (10%): Baum-Welch converged = +0.10 confidence.
 *   4. **Regime stability** (15%): consecutive days in the same regime / 20.
 *   5. **Momentum agreement** (10%): fraction of lookbacks agreeing on direction.
 *   6. **Base-rate alignment** (up to +20%): bonus when prediction agrees with
 *      empirical P(up); mild penalty for contra-directional predictions.
 *
 * Inspired by El-Yaniv & Pidan (NeurIPS 2011): selective prediction trades
 * coverage for accuracy by abstaining when confidence is low.
 */
export function computePredictionConfidence(options: {
  /** P(price > currentPrice at horizon) — preferably raw (pre-calibration) */
  pUp: number;
  /** Ensemble consensus count (0–3) */
  ensembleConsensus: number;
  /** Whether the HMM converged */
  hmmConverged: boolean;
  /** Consecutive days in the current regime state */
  regimeRunLength: number;
  /** Whether a structural break was detected */
  structuralBreak: boolean;
  /** Asset type — crypto/commodity predictions get a confidence discount */
  assetType?: 'etf' | 'equity' | 'crypto' | 'commodity';
  /** Recent daily volatility — high vol → harder to predict → lower confidence */
  recentVol?: number;
  /** Fraction of momentum lookbacks that agree on direction (0-1). Idea R. */
  momentumAgreement?: number;
  /** Calibrated P(up) — used for base-rate alignment scoring */
  calibratedPUp?: number;
  /** Historical base rate P(up) from recent returns */
  baseRate?: number;
  trustedAnchors?: number;
  horizonDays?: number;
  outOfSampleR2?: number | null;
  breakConfidencePolicy?: BreakConfidencePolicy;
  /** Preserve the Phase 4 sideways carve-out even when another break policy is active. */
  skipSidewaysBreakPenalty?: boolean;
  regimeState?: RegimeState;
  /** Chi-square divergence between the first/second half transition matrices. Required for 'divergence_weighted' policy. */
  structuralBreakDivergence?: number;
  /** Penalty schedule for divergence-weighted mode. Defaults to DEFAULT_DIVERGENCE_PENALTY_SCHEDULE. */
  divergencePenaltySchedule?: DivergencePenaltySchedule;
}): number {
  const { pUp, ensembleConsensus, hmmConverged, regimeRunLength, structuralBreak } = options;
  const cryptoShortHorizon = options.assetType === 'crypto'
    && options.horizonDays != null
    && options.horizonDays <= 14;
  const anchorsHelpful = (options.trustedAnchors ?? 0) >= 2;
  const r2 = options.outOfSampleR2;
  const r2ClearlyBad = typeof r2 === 'number' && Number.isFinite(r2) && r2 < -0.05;
  const r2NearZero = typeof r2 === 'number' && Number.isFinite(r2) && r2 >= -0.02 && r2 <= 0.02;

  // 1. Decisiveness: |P(up) - 0.5| scaled to [0, 1]
  const decisiveness = Math.min(1.0, Math.abs(pUp - 0.5) * 2);

  // 2. Ensemble consensus: 0/3 = 0, 1/3 = 0.33, 2/3 = 0.67, 3/3 = 1
  const consensusScore = ensembleConsensus / 3;

  // 3. HMM convergence: binary
  const hmmScore = hmmConverged ? 1.0 : 0.0;

  // 4. Regime stability: saturates at 20 consecutive days
  const stabilityScore = Math.min(1.0, regimeRunLength / 20);

  // Weighted combination (total base weights = 1.0)
  let confidence = 0.30 * decisiveness
                 + 0.15 * consensusScore
                 + 0.10 * hmmScore
                 + 0.15 * stabilityScore;

  // 5. Multi-lookback momentum agreement (Idea R)
  const momentumAgr = options.momentumAgreement ?? 0;
  confidence += 0.10 * momentumAgr;

  // 6. Base-rate alignment (Round 5): predictions that agree with the empirical
  // base rate are much more reliable than predictions that go against it.
  // In a 74% up market, a BUY prediction is right ~74% of the time (aligned),
  // while a SELL prediction is right ~26% of the time (contra-directional).
  const calPUp = options.calibratedPUp;
  const bRate = options.baseRate;
  if (calPUp !== undefined && bRate !== undefined) {
    const predDirection = calPUp >= 0.5 ? 1 : -1;
    const baseDirection = bRate >= 0.5 ? 1 : -1;
    const baseStrength = Math.abs(bRate - 0.5) * 2; // 0 at bRate=0.5, 1 at bRate=0/1

    if (predDirection === baseDirection) {
      // Aligned: boost proportional to how strong the base rate signal is
      confidence += 0.20 * baseStrength;
    } else {
      // Contra-directional: mild penalty — don't over-punish since model might have
      // genuine information that diverges from the base rate
      confidence -= 0.08 * baseStrength;
    }
  }

  if (cryptoShortHorizon && r2NearZero) {
    confidence += 0.08;
  }

  // Penalty for structural break (regime change mid-window → unreliable)
  if (structuralBreak) {
    const breakConfidencePolicy = options.breakConfidencePolicy ?? 'default';

    const skipBreakPenalty = options.regimeState === 'sideways'
      && (
        breakConfidencePolicy === 'trend_penalty_only'
        || options.skipSidewaysBreakPenalty === true
      );

    if (!skipBreakPenalty) {
      let breakPenalty: number;
      if (breakConfidencePolicy === 'divergence_weighted') {
        const divergence = options.structuralBreakDivergence ?? 0.20;
        const schedule = options.divergencePenaltySchedule ?? DEFAULT_DIVERGENCE_PENALTY_SCHEDULE;
        breakPenalty = computeDivergencePenalty(divergence, schedule);
      } else {
        breakPenalty = cryptoShortHorizon && anchorsHelpful && !r2ClearlyBad ? 0.8 : 0.6;
      }
      confidence *= breakPenalty;
    }
  }

  // Asset-type discount: crypto is inherently noisier → scale confidence down.
  // Commodities are driven by supply shocks → moderate discount.
  // ETFs are the most predictable → small boost.
  const assetType = options.assetType;
  if (assetType === 'crypto') {
    confidence *= cryptoShortHorizon && anchorsHelpful ? 0.85 : 0.7;
  } else if (assetType === 'commodity') {
    confidence *= 0.85;
  } else if (assetType === 'etf') {
    confidence *= 1.1;
  }

  // Volatility penalty: daily vol > 3% → scale confidence down.
  const vol = options.recentVol;
  if (vol && vol > 0.02) {
    const volPenalty = Math.max(0.7, 1 - (vol - 0.02) * 5);
    confidence *= volPenalty;
  }

  return Math.max(0, Math.min(1, confidence));
}

// ---------------------------------------------------------------------------
// 8a. interpolateSurvival + computeActionSignal — Buy/Hold/Sell signal
// ---------------------------------------------------------------------------

/**
 * Linearly interpolate P(price > targetPrice) from a monotone survival distribution.
 * Distribution must be sorted ascending by price (P(>price) non-increasing).
 */
export function interpolateSurvival(
  distribution: MarkovDistributionPoint[],
  targetPrice: number,
): number {
  if (distribution.length === 0) return 0.5;
  if (targetPrice <= distribution[0].price) return 1.0;
  if (targetPrice >= distribution[distribution.length - 1].price) return 0.0;
  for (let i = 0; i < distribution.length - 1; i++) {
    const lo = distribution[i];
    const hi = distribution[i + 1];
    if (targetPrice >= lo.price && targetPrice <= hi.price) {
      const t = (targetPrice - lo.price) / (hi.price - lo.price);
      return lo.probability + t * (hi.probability - lo.probability);
    }
  }
  return 0.0;
}

/**
 * Compute scenario probability buckets from the calibrated CDF distribution.
 * Buckets: Down >5%, Down 3–5%, Flat ±3%, Up 3–5%, Up >5%.
 * All probabilities are derived from `interpolateSurvival()` to guarantee
 * consistency with the P(>Price) table. Bucket probabilities sum to ~1.0.
 */
export function computeScenarioProbabilities(
  distribution: MarkovDistributionPoint[],
  currentPrice: number,
): ScenarioProbabilities {
  // Threshold prices
  const down5  = currentPrice * 0.95;
  const down3  = currentPrice * 0.97;
  const up3    = currentPrice * 1.03;
  const up5    = currentPrice * 1.05;

  // Read survival probabilities from the CDF
  const pAboveDown5  = interpolateSurvival(distribution, down5);
  const pAboveDown3  = interpolateSurvival(distribution, down3);
  const pAboveUp3    = interpolateSurvival(distribution, up3);
  const pAboveUp5    = interpolateSurvival(distribution, up5);

  // Derive bucket probabilities (all from the same CDF, guaranteed consistent)
  const pDownOver5  = 1.0 - pAboveDown5;         // P(price < down5)
  const pDown3to5   = pAboveDown5 - pAboveDown3;  // P(down5 ≤ price < down3)
  const pFlat       = pAboveDown3 - pAboveUp3;    // P(down3 ≤ price ≤ up3)
  const pUp3to5     = pAboveUp3 - pAboveUp5;      // P(up3 < price ≤ up5)
  const pUpOver5    = pAboveUp5;                   // P(price > up5)

  // P(up) from calibrated CDF
  const pUp = interpolateSurvival(distribution, currentPrice);

  // Expected return via trapezoidal integration of the survival function
  // E[price] = currentPrice + ∫₀^∞ P(price > x) dx - ∫₋∞^0 P(price < x) dx
  // Approximated by summing over distribution grid points
  let expectedPrice = currentPrice;
  if (distribution.length >= 2) {
    let integral = 0;
    for (let i = 0; i < distribution.length - 1; i++) {
      const dx = distribution[i + 1].price - distribution[i].price;
      const avgP = (distribution[i].probability + distribution[i + 1].probability) / 2;
      integral += avgP * dx;
    }
    // E[price] = minPrice + integral (from survival function identity)
    expectedPrice = distribution[0].price + integral;
  }
  const expectedReturn = (expectedPrice - currentPrice) / currentPrice;

  return {
    buckets: [
      { label: 'Down >5%',  probability: Math.max(0, pDownOver5), priceRange: [null, Math.round(down5 * 100) / 100] },
      { label: 'Down 3–5%', probability: Math.max(0, pDown3to5),  priceRange: [Math.round(down5 * 100) / 100, Math.round(down3 * 100) / 100] },
      { label: 'Flat ±3%',  probability: Math.max(0, pFlat),      priceRange: [Math.round(down3 * 100) / 100, Math.round(up3 * 100) / 100] },
      { label: 'Up 3–5%',   probability: Math.max(0, pUp3to5),    priceRange: [Math.round(up3 * 100) / 100, Math.round(up5 * 100) / 100] },
      { label: 'Up >5%',    probability: Math.max(0, pUpOver5),   priceRange: [Math.round(up5 * 100) / 100, null] },
    ],
    expectedPrice: Math.round(expectedPrice * 100) / 100,
    expectedReturn: Math.round(expectedReturn * 10000) / 10000,
    pUp: Math.round(pUp * 1000) / 1000,
  };
}

/**
 * Derive a Buy / Hold / Sell action signal from the probability distribution.
 *
 * Zones (relative to currentPrice):
 *   BUY  — P(price >  currentPrice × (1 + buyThreshold))   default: P(>+5%)
 *   SELL — P(price <  currentPrice × (1 − sellThreshold))  default: P(<−3%)
 *   HOLD — remainder
 *
 * Expected return via discrete integration of the survival function:
 *   E[price] ≈ Σᵢ (S(xᵢ) − S(xᵢ₊₁)) × midᵢ  +  tail corrections
 *
 * @param distribution   Output of interpolateDistribution (ascending price, P non-increasing)
 * @param currentPrice   Current price (serves as the 50% reference)
 * @param buyThreshold   Min upside for BUY zone (default 0.05 = +5%)
 * @param sellThreshold  Min downside for SELL zone (default 0.03 = −3%)
 * @param horizon        Forecast horizon in trading days (used to scale action thresholds)
 * @param recentVol      Recent daily volatility (e.g. 20-day std of returns). Used to
 *                        set dynamic thresholds so BUY/SELL triggers match actual signal range.
 */
export function computeActionSignal(
  distribution: MarkovDistributionPoint[],
  currentPrice: number,
  buyThreshold = 0.05,
  sellThreshold = 0.03,
  horizon = 30,
  recentVol?: number,
  scenarios?: ScenarioProbabilities,
  assetType?: AssetProfile['type'],
): ActionSignal {
  const pAboveBuy  = interpolateSurvival(distribution, currentPrice * (1 + buyThreshold));
  const pAboveSell = interpolateSurvival(distribution, currentPrice * (1 - sellThreshold));
  const pBelowSell = 1 - pAboveSell;
  const pHold      = Math.max(0, 1 - pAboveBuy - pBelowSell);

  // E[price] via integration: Σ mass_i × midprice_i  (trapezoid rule)
  let ePrice = 0;
  for (let i = 0; i < distribution.length - 1; i++) {
    const mass = distribution[i].probability - distribution[i + 1].probability;
    const mid  = (distribution[i].price + distribution[i + 1].price) / 2;
    ePrice += mass * mid;
  }
  // Bottom tail: P(price ≤ minPrice) × minPrice
  ePrice += (1 - distribution[0].probability) * distribution[0].price;
  // Top tail: P(price > maxPrice) × maxPrice
  ePrice += distribution[distribution.length - 1].probability
          * distribution[distribution.length - 1].price;

  const expectedReturn = Number.isFinite(ePrice) && currentPrice > 0
    ? (ePrice - currentPrice) / currentPrice
    : 0;

  // Unconditional expected upside and downside (dollar terms, then ratio)
  let eUpside = 0, eDownside = 0;
  for (let i = 0; i < distribution.length - 1; i++) {
    const mass = distribution[i].probability - distribution[i + 1].probability;
    const mid  = (distribution[i].price + distribution[i + 1].price) / 2;
    eUpside   += mass * Math.max(0, mid - currentPrice);
    eDownside += mass * Math.max(0, currentPrice - mid);
  }
  // Tail contributions
  eDownside += (1 - distribution[0].probability)
             * Math.max(0, currentPrice - distribution[0].price);
  eUpside   += distribution[distribution.length - 1].probability
             * Math.max(0, distribution[distribution.length - 1].price - currentPrice);

  const rawRRR = eDownside > 0 ? eUpside / eDownside : 1.0;
  const riskRewardRatio = Number.isFinite(rawRRR) ? rawRRR : 1.0;

  // Dynamic action thresholds (Idea H): scale with asset volatility instead of fixed values.
  // Previous fixed thresholds (±2% for 14-30d) were wider than the model's signal range
  // (E[R] std ≈ 1.85%), causing 74% HOLD predictions. Dynamic thresholds use:
  //   threshold = scaleFactor × dailyVol × sqrt(horizon)
  // which adapts to each asset's actual volatility and the forecast horizon.
  let actionBuyThr: number;
  let actionSellThr: number;
  if (recentVol && recentVol > 0) {
    // Lower thresholds → more BUY/SELL signals (fewer HOLDs).
    // In bullish markets, HOLD is usually wrong (actual returns are +3-10%),
    // so converting uncertain HOLDs to BUYs improves directional accuracy.
    const volScaled = recentVol * Math.sqrt(horizon);
    actionBuyThr  = Math.max(0.001, 0.08 * volScaled);
    actionSellThr = Math.max(0.001, 0.06 * volScaled);
  } else {
    actionBuyThr  = horizon <= 7 ? 0.003 : horizon <= 30 ? 0.005 : 0.008;
    actionSellThr = horizon <= 7 ? 0.002 : horizon <= 30 ? 0.003 : 0.005;
  }

  // Recommendation derived from expectedReturn (not zone argmax)
  let recommendation: 'BUY' | 'HOLD' | 'SELL';
  if (expectedReturn > actionBuyThr) {
    recommendation = 'BUY';
  } else if (expectedReturn < -actionSellThr) {
    recommendation = 'SELL';
  } else {
    recommendation = 'HOLD';
  }

  const shortHorizonCrypto = assetType === 'crypto' && horizon <= 14;
  if (shortHorizonCrypto && recommendation === 'HOLD' && scenarios) {
    recommendation = scenarios.pUp >= 0.50 ? 'BUY' : 'SELL';
  }

  // Cross-validate recommendation against scenario probabilities.
  // The mean (expectedReturn) can be pulled positive by a fat right tail even when
  // the median is negative and downside scenarios dominate. In such cases, a BUY
  // signal is misleading — the most likely outcome is a loss.
  if (scenarios) {
    const pUp = scenarios.pUp;
    const upScenarios   = (scenarios.buckets[3]?.probability ?? 0) + (scenarios.buckets[4]?.probability ?? 0);
    const downScenarios = (scenarios.buckets[0]?.probability ?? 0) + (scenarios.buckets[1]?.probability ?? 0);

    if (recommendation === 'BUY') {
      // Gate 1: P(up) < 50% → CDF says more likely to go down than up → cannot be BUY
      if (pUp < 0.50) {
        recommendation = 'HOLD';
      }
      // Gate 2: downside scenarios exceed upside by >5pp → bearish tilt → downgrade
      else if (downScenarios > upScenarios + 0.05) {
        recommendation = 'HOLD';
      }
    } else if (recommendation === 'SELL') {
      // Mirror: P(up) > 50% → more likely up → cannot be SELL
      if (pUp > 0.50) {
        recommendation = 'HOLD';
      }
      // Mirror: upside scenarios exceed downside by >5pp → bullish tilt → downgrade
      else if (upScenarios > downScenarios + 0.05) {
        recommendation = 'HOLD';
      }
    }
  }

  // Confidence from conviction strength relative to threshold
  const activeThr = recommendation === 'BUY' ? actionBuyThr : actionSellThr;
  const conviction = Math.abs(expectedReturn);
  let confidence: 'HIGH' | 'MEDIUM' | 'LOW' =
    conviction >= 2 * activeThr ? 'HIGH' : conviction >= activeThr ? 'MEDIUM' : 'LOW';

  // Compute action levels (includes the true median price from CDF)
  const actionLevels = computeActionLevels(distribution, currentPrice);

  // Cap confidence when mean (expectedReturn) and median disagree in sign.
  // scenarios.expectedReturn is the mean, NOT the median — use the true
  // CDF median from actionLevels for this cross-check.
  const medianReturn = (actionLevels.medianPrice - currentPrice) / currentPrice;
  if ((expectedReturn > 0 && medianReturn < -0.005) || (expectedReturn < 0 && medianReturn > 0.005)) {
    if (confidence === 'HIGH') confidence = 'MEDIUM';
  }

  return {
    buyProbability:  pAboveBuy,
    holdProbability: pHold,
    sellProbability: pBelowSell,
    recommendation,
    confidence,
    expectedReturn,
    riskRewardRatio,
    buyThreshold,
    sellThreshold,
    actionLevels,
  };
}

/**
 * Extract key price levels from the probability distribution for trade execution.
 * Uses linear interpolation on the survival curve to find specific percentiles.
 */
export function computeActionLevels(
  distribution: MarkovDistributionPoint[],
  currentPrice: number,
): ActionLevels {
  // Helper: find the price where P(>price) ≈ targetProb via linear interpolation
  const findPriceAtProb = (targetProb: number): number => {
    if (distribution.length === 0) return currentPrice;
    // P(>price) is non-increasing; find the bracket
    for (let i = 0; i < distribution.length - 1; i++) {
      const hi = distribution[i];     // higher probability (lower price)
      const lo = distribution[i + 1]; // lower probability (higher price)
      if (hi.probability >= targetProb && lo.probability <= targetProb) {
        if (Math.abs(hi.probability - lo.probability) < 1e-10) return hi.price;
        const t = (hi.probability - targetProb) / (hi.probability - lo.probability);
        return hi.price + t * (lo.price - hi.price);
      }
    }
    // Edge: targetProb above all points → return lowest price
    if (targetProb >= distribution[0].probability) return distribution[0].price;
    // Edge: targetProb below all points → return highest price
    return distribution[distribution.length - 1].price;
  };

  return {
    medianPrice: findPriceAtProb(0.50),
    targetPrice: findPriceAtProb(0.30),
    stopLoss:    findPriceAtProb(0.90),
    bullCase:    findPriceAtProb(0.20),
    bearCase:    findPriceAtProb(0.80),
  };
}

/**
 * Assess Polymarket anchor quality: how well do anchors cover the relevant price range?
 * Sparse anchors = large gaps = less reliable blending.
 */
export function assessAnchorCoverage(
  anchors: PriceThreshold[],
  currentPrice: number,
  options?: { ticker?: string; horizonDays?: number },
): AnchorCoverageDiagnostic {
  const trusted = anchors.filter(a => a.trustScore === 'high');
  if (trusted.length === 0) {
    return {
      totalAnchors: anchors.length,
      trustedAnchors: 0,
      maxGapPct: 100,
      quality: 'none',
      warning: 'No trusted Polymarket anchors — distribution is 100% Markov-model driven',
    };
  }

  // Sort by price and find largest gap (including gap from current price to first anchor)
  const sorted = [...trusted].sort((a, b) => a.price - b.price);
  let maxGap = 0;
  // Gap from current price to nearest anchor
  const nearestDist = sorted.reduce(
    (best, a) => Math.min(best, Math.abs(a.price - currentPrice) / currentPrice), Infinity,
  );
  maxGap = nearestDist;
  // Gaps between adjacent anchors
  for (let i = 0; i < sorted.length - 1; i++) {
    const gap = (sorted[i + 1].price - sorted[i].price) / sorted[i].price;
    if (gap > maxGap) maxGap = gap;
  }

  const maxGapPct = maxGap * 100;
  const isShortHorizonCrypto = options?.ticker != null
    && getAssetProfile(options.ticker).type === 'crypto'
    && options?.horizonDays != null
    && options.horizonDays <= 14;

  const quality: 'good' | 'sparse' | 'none' =
    trusted.length >= (isShortHorizonCrypto ? 2 : 3) && maxGapPct < (isShortHorizonCrypto ? 22 : 15) ? 'good'
    : trusted.length >= 1 ? 'sparse'
    : 'none';

  const warning = quality === 'good' ? ''
    : quality === 'sparse'
      ? `Sparse anchors (${trusted.length} trusted, max gap ${maxGapPct.toFixed(0)}%) — interpolation between anchors is model-driven`
      : 'No trusted anchors';

  return { totalAnchors: anchors.length, trustedAnchors: trusted.length, maxGapPct, quality, warning };
}

// ---------------------------------------------------------------------------
// 8c. Base-rate floor helper (PR3 Stage 2 ablation)
// ---------------------------------------------------------------------------

export function computeBaseRateFloor(
  baseRate: number,
  calibrationCenter: number,
  isCrypto: boolean,
  horizon: number,
  minPUpFloor?: number,
  bearMarginMultiplier?: number
): { bearMargin: number; pUpFloor: number } {
  let clampBase = 0.05;
  let clampCap = 0.10;
  let scale = 0.2;
  let floorMin = 0.35;

  if (isCrypto && horizon <= 14) {
    if (bearMarginMultiplier !== undefined) {
      clampBase *= bearMarginMultiplier;
      clampCap *= bearMarginMultiplier;
      scale *= bearMarginMultiplier;
    }
    if (minPUpFloor !== undefined) {
      floorMin = minPUpFloor;
    }
  }

  const bearMargin = Math.max(clampBase, Math.min(clampCap, clampBase + (baseRate - 0.5) * scale));
  const pUpFloor = Math.max(floorMin, calibrationCenter - bearMargin);

  return { bearMargin, pUpFloor };
}

// ---------------------------------------------------------------------------
// 8b. markovDistribution — main orchestration function
// ---------------------------------------------------------------------------

/**
 * Core Markov distribution computation. Separated from the LangChain tool wrapper
 * so it can be unit-tested directly without mocking the tool framework.
 */
export async function computeMarkovDistribution(params: {
  ticker: string;
  horizon: number;
  currentPrice: number;
  historicalPrices: number[];            // daily close prices, oldest first
  polymarketMarkets: Array<{
    question: string;
    probability: number;
    volume?: number;
    createdAt?: string | number;
  }>;
  sentiment?: SentimentSignal;
  /** Optional Kalshi anchors for cross-platform validation (Tier 1c) */
  kalshiAnchors?: KalshiAnchor[];
  /** Return day-by-day price trajectory instead of single-horizon snapshot */
  trajectory?: boolean;
  /** Number of days for trajectory (default: horizon, max 30) */
  trajectoryDays?: number;
  /** @deprecated experimental ablation — backtests only */
  cryptoShortHorizonConditionalWeight?: number;
  /** @deprecated experimental ablation — backtests only */
  sidewaysSplit?: boolean;
  /** @deprecated experimental ablation — backtests only */
  cryptoShortHorizonKappaMultiplier?: number;
  /** @deprecated experimental ablation — backtests only */
  cryptoShortHorizonRawDecisionAblation?: boolean;
  /** @deprecated experimental ablation — backtests only */
  rawDirectionHybrid?: boolean;
  /** @deprecated experimental ablation — backtests only */
  pr3fCryptoShortHorizonDisagreementPrior?: boolean;
  /** @deprecated experimental ablation — backtests only */
  cryptoShortHorizonPUpFloor?: number;
  /** @deprecated experimental ablation — backtests only */
  cryptoShortHorizonBearMarginMultiplier?: number;
  /** @deprecated experimental ablation — backtests only */
  pr3gCryptoShortHorizonRecencyWeighting?: boolean;
  /** @deprecated experimental ablation — backtests only */
  pr3gCryptoShortHorizonDecay?: number;
  /** @deprecated experimental ablation — backtests only */
  referenceTimeMs?: number;
  /** BTC short-horizon default-on: explicit false disables the promoted start-state mixture; explicit true preserves opt-in for other crypto short horizons (backtests only). */
  startStateMixture?: boolean;
  /** @deprecated experimental ablation — backtests only */
  matureBullCalibration?: boolean;
  /** @deprecated experimental ablation — backtests only */
  transitionDecayOverride?: number;
  /** @deprecated experimental ablation — backtests only */
  trendPenaltyOnlyBreakConfidence?: boolean;
  /** @internal Phase 5 experimental: hybrid structural-break fallback candidate (backtest-only) */
  breakFallbackCandidate?: BreakFallbackCandidate;
  /** Phase 6 experimental: use divergence-weighted break confidence penalties (backtest-only) */
  divergenceWeightedBreakConfidence?: boolean;
  /** Phase 6 experimental: penalty schedule for divergence-weighted mode. Defaults to DEFAULT_DIVERGENCE_PENALTY_SCHEDULE. */
  divergencePenaltySchedule?: DivergencePenaltySchedule;
  /** Phase 7 experimental: use dominant regime's own sigma instead of mixture sigma when regime weights are concentrated (backtest-only) */
  regimeSpecificSigma?: boolean;
  /** Phase 7 experimental: minimum max(stateWeight) to activate regime-specific sigma. Defaults to 0.60. */
  regimeSpecificSigmaThreshold?: number;
  /** Phase D experimental: BTC-only override for regime classification return threshold multiplier (backtest-only). */
  btcReturnThresholdMultiplier?: number;
  /** Phase C experimental: BTC-only override for structural break divergence threshold (backtest-only). */
  btcBreakDivergenceThreshold?: number;
}): Promise<MarkovDistributionResult> {
  const {
    ticker,
    horizon,
    currentPrice,
    historicalPrices,
    polymarketMarkets,
    sentiment,
    kalshiAnchors,
    trajectory,
    trajectoryDays,
    cryptoShortHorizonConditionalWeight,
    sidewaysSplit,
    cryptoShortHorizonKappaMultiplier,
    cryptoShortHorizonRawDecisionAblation,
    rawDirectionHybrid,
    pr3fCryptoShortHorizonDisagreementPrior,
    cryptoShortHorizonPUpFloor,
    cryptoShortHorizonBearMarginMultiplier,
    pr3gCryptoShortHorizonRecencyWeighting,
    pr3gCryptoShortHorizonDecay,
    startStateMixture,
    matureBullCalibration,
    transitionDecayOverride,
    trendPenaltyOnlyBreakConfidence,
    breakFallbackCandidate,
    divergenceWeightedBreakConfidence,
    divergencePenaltySchedule,
    regimeSpecificSigma,
    regimeSpecificSigmaThreshold,
    btcReturnThresholdMultiplier,
    btcBreakDivergenceThreshold,
  } = params;

  const normalizedSentiment = sentiment === undefined ? undefined : normalizeSentiment(sentiment);
  if (sentiment !== undefined && normalizedSentiment === undefined) {
    throw new Error('Invalid sentiment input: expected bullish/bearish as decimals in [0,1] or percentages in [0,100].');
  }

  // --- Asset profile (Idea N): per-asset-class parameter tuning ---
  const assetProfile = getAssetProfile(ticker);
  const isBtcTicker = ticker === 'BTC' || ticker === 'BTC-USD';
  const isBtcShortHorizonThresholdDefault = isBtcTicker && assetProfile.type === 'crypto' && horizon <= 14;

  // --- Daily returns and volatility ---
  const returns: number[] = [];
  const vols: number[]    = [];
  for (let i = 1; i < historicalPrices.length; i++) {
    const ret = (historicalPrices[i] - historicalPrices[i - 1]) / historicalPrices[i - 1];
    returns.push(ret);
    // Approximate daily vol as |return| (proxy; real impl should use (H-L)/O)
    vols.push(Math.abs(ret));
  }

  // --- Classify regime states with adaptive thresholds ---
  const effectiveReturnThresholdMultiplier = btcReturnThresholdMultiplier !== undefined && isBtcTicker
    ? btcReturnThresholdMultiplier
    : isBtcShortHorizonThresholdDefault
    ? 0.65
    : 0.5;
  const { returnThreshold, volThreshold } = computeAdaptiveThresholds(
    returns,
    effectiveReturnThresholdMultiplier,
  );
  const regimeSeq: RegimeState[] = returns.map((r, i) =>
    classifyRegimeState(r, vols[i], returnThreshold, volThreshold),
  );
  const currentRegime = regimeSeq.length > 0 ? regimeSeq[regimeSeq.length - 1] : 'sideways';

  // --- Tier 1a: State observation counts and sparse state detection ---
  const stateObservationCounts = countStateObservations(regimeSeq);
  const sparseStates = findSparseStates(stateObservationCounts);

  // --- Tier 1b: Structural break detection ---
  const effectiveTransitionDecay = transitionDecayOverride ?? 0.97;
  const effectiveBreakDivergenceThreshold = btcBreakDivergenceThreshold !== undefined
    && isBtcTicker
    ? btcBreakDivergenceThreshold
    : 0.05;
  const breakResult = regimeSeq.length >= 20
    ? detectStructuralBreak(regimeSeq, effectiveBreakDivergenceThreshold, 0.1, effectiveTransitionDecay)
    : { detected: false, divergence: 0, firstHalfMatrix: buildDefaultMatrix(), secondHalfMatrix: buildDefaultMatrix() };

  // --- Estimate transition matrix (fall back to default when break detected) ---
  let P: TransitionMatrix;
  if (breakResult.detected) {
    // Structural break detected: replace or blend the estimated matrix
    if (breakFallbackCandidate) {
      // Phase 5: apply hybrid fallback candidate (backtest-only)
      const estimatedMatrix = estimateTransitionMatrix(regimeSeq, undefined, 30, effectiveTransitionDecay);
      P = applyBreakFallbackCandidate(estimatedMatrix, breakResult.divergence, breakFallbackCandidate, assetProfile.type);
    } else {
      P = buildDefaultMatrix();
    }
  } else {
    P = estimateTransitionMatrix(regimeSeq, undefined, 30, effectiveTransitionDecay);
  }

  // --- Goodness-of-fit test (before sentiment adjustment, which is intentional) ---
  const gofResult = breakResult.detected ? null : transitionGoodnessOfFit(regimeSeq, P);

  // --- Sentiment adjustment ---
  const sentimentSignal = normalizedSentiment ?? { bullish: 0.5, bearish: 0.5 };
  const sentimentShift = sentimentSignal.bullish - sentimentSignal.bearish;
  P = adjustTransitionMatrix(P, sentimentSignal);

  // --- Regime statistics (winsorized + drift-capped per asset profile) ---
  const logReturns = returns.map(r => Math.log(1 + r));
  const regimeStats = estimateRegimeStats(logReturns, regimeSeq, assetProfile.maxDailyDrift);

  // --- Tier 1c: Polymarket anchors with optional cross-platform validation ---
  let polymarketAnchors = extractPriceThresholds(polymarketMarkets, { ticker, horizonDays: horizon, referenceTimeMs: params.referenceTimeMs });
  let anchorDivergenceWarnings: AnchorDivergenceWarning[] = [];

  // Normalize commodity ETF anchors: convert futures/per-oz prices to ETF scale
  polymarketAnchors = normalizeAnchorPricesForETF(
    polymarketAnchors, currentPrice, ticker,
  );

  // Filter uninformative anchors:
  // 1. Trivially-true anchors: price well below current with P≈1 (e.g., "gold stays above $3,000")
  // 2. Trivially-false anchors: price well above current with P≈0
  // These carry no predictive signal — they just confirm what the current price already implies.
  polymarketAnchors = polymarketAnchors.filter(a => {
    const dist = (a.price - currentPrice) / currentPrice;
    if (dist < -0.05 && a.probability > 0.90) return false;
    if (dist > 0.50 && a.probability < 0.05) return false;
    return true;
  });

  // Crypto fallback: when strict-horizon markets are all barrier-style and yield
  // zero terminal anchors, re-extract from the full unfiltered market list with
  // a date-gap discount on off-horizon anchors.
  polymarketAnchors = applyCryptoTerminalAnchorFallback(
    polymarketMarkets, polymarketAnchors, ticker, horizon, params.referenceTimeMs,
  );
  // Re-apply triviality filter on fallback anchors (e.g., off-horizon anchors may
  // now be trivially true/false relative to current price)
  polymarketAnchors = polymarketAnchors.filter(a => {
    const dist = (a.price - currentPrice) / currentPrice;
    if (dist < -0.05 && a.probability > 0.90) return false;
    if (dist > 0.50 && a.probability < 0.05) return false;
    return true;
  });

  if (params.kalshiAnchors && params.kalshiAnchors.length > 0) {
    const merged = mergeAnchorsWithCrossPlatformValidation(polymarketAnchors, params.kalshiAnchors);
    polymarketAnchors = merged.anchors;
    anchorDivergenceWarnings = merged.warnings;
  }

  // --- Spectral gap ---
  const rho = secondLargestEigenvalue(P);
  const mixWeight = computeMixingWeight(rho, horizon);

  // --- Tier 1b: Widen CI when structural break detected ---
  const ciWidthMultiplier = breakResult.detected ? 1.5 : 1.0;

  // --- Momentum signal + multi-lookback confirmation (Idea R) ---
  const momentum = computeMomentumSignal(historicalPrices);
  // Check if momentum direction agrees across multiple lookbacks (10d, 20d, 40d)
  const m10 = computeMomentumSignal(historicalPrices, 10);
  const m40 = historicalPrices.length > 41
    ? computeMomentumSignal(historicalPrices, 40)
    : null;
  // Count lookbacks agreeing with the primary signal direction
  const primaryDir = Math.sign(momentum.velocity);
  let lookbackAgreement = 1; // primary always agrees
  if (Math.sign(m10.velocity) === primaryDir && primaryDir !== 0) lookbackAgreement++;
  if (m40 && Math.sign(m40.velocity) === primaryDir && primaryDir !== 0) lookbackAgreement++;
  const totalLookbacks = m40 ? 3 : 2;
  // Halve momentum weight when structural break detected (trend may have broken)
  const momentumAdj = breakResult.detected ? momentum.adjustment * 0.5 : momentum.adjustment;

  // --- Ensemble signal (Idea D: combine mean-reversion + crossover + vol compression) ---
  const ensemble = computeEnsembleSignal(historicalPrices);
  // Only apply ensemble adjustment when consensus is high (≥2 of 3 signals agree)
  const ensembleAdj = ensemble.consensus >= 2 ? ensemble.adjustment : 0;
  // Combine momentum + ensemble into a single drift modifier
  const combinedDriftAdj = momentumAdj + ensembleAdj;

  // --- HMM forecast (Idea B: Hidden Markov Model + Idea C: Multi-feature) ---
  // Fit a Gaussian HMM on daily returns when we have enough data.
  // Also fit a volatility HMM on rolling vol for an independent vol-regime signal.
  let hmmOverride: { drift: number; vol: number; weight: number } | undefined;
  let hmmMeta: { converged: boolean; iterations: number; states: number; logLikelihood: number; volRegimeConverged?: boolean } | undefined;
  const HMM_MIN_OBS = 60; // need at least 60 returns for stable HMM
  if (returns.length >= HMM_MIN_OBS) {
    try {
      // Primary: return HMM (directional signal)
      const hmmResult = baumWelch(returns, 3, 50, 1e-3);
      const hmmForecast = hmmPredict(returns, hmmResult.params, horizon);

      // Secondary: volatility regime HMM (Idea C — orthogonal vol signal)
      // 5-day rolling realized volatility as independent feature
      const rollingVol: number[] = [];
      const VOL_WINDOW = 5;
      for (let i = VOL_WINDOW; i < returns.length; i++) {
        const window = returns.slice(i - VOL_WINDOW, i);
        const mean = window.reduce((s, v) => s + v, 0) / VOL_WINDOW;
        const variance = window.reduce((s, v) => s + (v - mean) ** 2, 0) / VOL_WINDOW;
        rollingVol.push(Math.sqrt(variance));
      }

      let volRegimeConverged = false;
      let volScaleFactor = 1.0; // neutral default
      if (rollingVol.length >= HMM_MIN_OBS) {
        try {
          const volHmm = baumWelch(rollingVol, 2, 30, 1e-3);
          if (volHmm.converged) {
            volRegimeConverged = true;
            const volForecast = hmmPredict(rollingVol, volHmm.params, Math.min(horizon, 20));
            // Current vol regime: high-vol state → widen uncertainty, low-vol → narrow it
            const avgVol = rollingVol.reduce((s, v) => s + v, 0) / rollingVol.length;
            if (avgVol > 0) {
              // Scale factor: >1 means currently in high-vol regime, <1 means low-vol
              volScaleFactor = volForecast.expectedReturn / avgVol;
              volScaleFactor = Math.max(0.5, Math.min(2.0, volScaleFactor)); // clamp
            }
          }
        } catch {
          // Vol HMM failed — use neutral factor
        }
      }

      hmmMeta = {
        converged: hmmResult.converged,
        iterations: hmmResult.iterations,
        states: hmmResult.params.nStates,
        logLikelihood: hmmResult.logLikelihood,
        volRegimeConverged,
      };
      if (hmmResult.converged && Number.isFinite(hmmForecast.expectedReturn)) {
        // Weight HMM based on data length, scaled by asset profile
        const baseHmmWeight = returns.length >= 120 ? 0.5 : 0.25;
        const hmmWeight = Math.min(0.7, baseHmmWeight * assetProfile.hmmWeightMultiplier);
        hmmOverride = {
          drift: hmmForecast.expectedReturn,
          vol: hmmForecast.expectedVolatility * volScaleFactor,
          weight: hmmWeight,
        };
      }
    } catch {
      // HMM fitting can fail on degenerate data — fall back to observable Markov
    }
  }

  
  // --- PR3 Post-Experiment: Sideways Split ---
  const isBtcShortHorizon = sidewaysSplit && assetProfile.type === 'crypto' && horizon <= 14 && (ticker === 'BTC' || ticker === 'BTC-USD');
  let sidewaysSplitActive = false;
  let weights3_experiment: number[] | undefined;
  let conditionalPUp_experiment: number | undefined;

  if (isBtcShortHorizon) {
    const seq4 = regimeSeq.map((r, i) => {
      if (r !== 'sideways') return r;
      const histSlice = historicalPrices.slice(0, i + 2);
      if (histSlice.length >= 25) {
        const sig = computeEnsembleSignal(histSlice);
        return sig.volCompression < 1.0 ? 'sideways_coil' : 'sideways_chop';
      }
      return 'sideways_chop'; // default for early days
    });

    const c4 = { bull: 0, bear: 0, sideways_coil: 0, sideways_chop: 0 };
    for (const s of seq4) c4[s as keyof typeof c4]++;

    // fallback threshold: at least 5 obs
    if (c4.sideways_coil >= 5 && c4.sideways_chop >= 5) {
      sidewaysSplitActive = true;
      
      const alpha4 = 5.0 / seq4.length;
      let P4 = Array.from({ length: 4 }, () => Array(4).fill(alpha4));
      const idx4: Record<string, number> = { bull: 0, bear: 1, sideways_coil: 2, sideways_chop: 3 };
      const decay4 = 0.97;
      const nSteps = seq4.length - 1;
      
      for (let i = 0; i < nSteps; i++) {
        P4[idx4[seq4[i]]][idx4[seq4[i+1]]] += Math.pow(decay4, nSteps - 1 - i);
      }
      
      // normalize
      P4 = P4.map(row => {
        const sum = row.reduce((a,b)=>a+b,0);
        return row.map(v => v/sum);
      });
      
      // sentiment adjustment (replicate adjustTransitionMatrix logic exactly)
      const shift = sentimentShift;
      const alpha_sent = 0.07;
      P4[0][0] *= (1 + alpha_sent * shift);
      P4[0][1] *= (1 - alpha_sent * shift);
      P4[1][1] *= (1 - alpha_sent * shift);
      P4[1][0] *= (1 + alpha_sent * shift);
      
      P4 = P4.map(row => {
        const sum = row.reduce((a,b)=>a+b,0);
        return row.map(v => Math.max(0, v/sum));
      });

      const Pn4 = matPow(P4, horizon);
      const w4 = Pn4[idx4[seq4[seq4.length - 1]]];
      
      weights3_experiment = [w4[0], w4[1], w4[2] + w4[3]];

      // Compute conditional up rates using the 4-state sequence
      const pr3gDecayRate = (pr3gCryptoShortHorizonRecencyWeighting && assetProfile.type === 'crypto' && horizon <= 14)
        ? (pr3gCryptoShortHorizonDecay ?? assetProfile.decayRate)
        : undefined;

      const upHits = { bull: 0, bear: 0, sideways_coil: 0, sideways_chop: 0 };
      const upTotals = { bull: 0, bear: 0, sideways_coil: 0, sideways_chop: 0 };
      const decay = pr3gDecayRate ?? 1.0;
      const nHorizonReturns = seq4.length - horizon;
      for (let i = 0; i < nHorizonReturns; i++) {
        const state = seq4[i] as keyof typeof upTotals;
        let ret = 0;
        for (let j = 1; j <= horizon; j++) ret += returns[i + j];
        const weight = Math.pow(decay, nHorizonReturns - 1 - i);
        upTotals[state] += weight;
        if (ret > 0) upHits[state] += weight;
      }
      
      const upRates4 = {
        bull: upTotals.bull > 0 ? upHits.bull / upTotals.bull : 0.5,
        bear: upTotals.bear > 0 ? upHits.bear / upTotals.bear : 0.5,
        sideways_coil: upTotals.sideways_coil > 0 ? upHits.sideways_coil / upTotals.sideways_coil : 0.5,
        sideways_chop: upTotals.sideways_chop > 0 ? upHits.sideways_chop / upTotals.sideways_chop : 0.5,
      };

      conditionalPUp_experiment = 
        w4[0] * upRates4.bull + 
        w4[1] * upRates4.bear + 
        w4[2] * upRates4.sideways_coil + 
        w4[3] * upRates4.sideways_chop;
    }
  }

  // Apply P_fake override to inject the 3-state horizon weights smoothly into the rest of the pipeline.
  // Guard: only activate if the experimental matrix is non-degenerate (all self-loops ≥ 0.333).
  // This prevents the override from producing a matrix that violates basic ergodicity assumptions.
  const sidewaysSplitSafe = sidewaysSplitActive
    && weights3_experiment !== undefined
    && Math.min(weights3_experiment[0], weights3_experiment[1], weights3_experiment[2]) >= 1 / 3;
  if (sidewaysSplitSafe) {
    const w = weights3_experiment!;
    P = [w, w, w];
  }

  // --- Distribution ---
  const useStartStateMixture = assetProfile.type === 'crypto' && horizon <= 14
    && (startStateMixture === true || (startStateMixture !== false && isBtcTicker));
  const mixture = useStartStateMixture && regimeSeq.length >= 5
    ? computeStartStateMixture(regimeSeq.slice(-5))
    : undefined;

  // Compute daily volatility for adaptive grid sizing
  const gridDailyVol = returns.length >= 20
    ? Math.sqrt(returns.slice(-20).reduce((s, v) => s + v * v, 0) / 20)
    : undefined;
  const rawDistribution = interpolateDistribution(
    currentPrice, horizon, P, regimeStats, currentRegime, polymarketAnchors, rho,
    20, 1000, ciWidthMultiplier, combinedDriftAdj, hmmOverride, gridDailyVol, mixture,
    assetProfile.studentTNu, regimeSpecificSigma, regimeSpecificSigmaThreshold,
  );

  // Compute drift/vol for drift-based calibration (same values used by interpolateDistribution)
  const { mu_n: calDriftN, sigma_n: calVolN } = computeHorizonDriftVol(
    horizon, P, regimeStats, currentRegime, combinedDriftAdj, hmmOverride, mixture,
    regimeSpecificSigma, regimeSpecificSigmaThreshold,
  );

  // --- Bayesian calibration (Idea I): shrink extreme probabilities toward base rate ---
  // Compute empirical base rate: P(up) from recent returns (Idea L: adaptive center)
  const recentReturns = returns.slice(-Math.min(120, returns.length));
  const baseRate = recentReturns.length > 0
    ? recentReturns.filter(r => r > 0).length / recentReturns.length
    : 0.5;

  // --- Regime-conditional P(up) (Idea T, Round 4) ---
  const pr3gDecayRate = (pr3gCryptoShortHorizonRecencyWeighting && assetProfile.type === 'crypto' && horizon <= 14)
    ? (pr3gCryptoShortHorizonDecay ?? assetProfile.decayRate)
    : undefined;
  const regimeUpRates = computeRegimeUpRates(regimeSeq, returns, horizon, pr3gDecayRate);
  const Pn = matPow(P, horizon);
  
  let stateWeightsForUp: number[];
  if (mixture) {
    stateWeightsForUp = [0, 0, 0];
    for (const state of REGIME_STATES) {
      const w = mixture[state];
      const idx = STATE_INDEX[state];
      for (let j = 0; j < 3; j++) {
        stateWeightsForUp[j] += w * Pn[idx][j];
      }
    }
  } else {
    stateWeightsForUp = Pn[STATE_INDEX[currentRegime]];
  }

  const conditionalPUp = (sidewaysSplitSafe && conditionalPUp_experiment !== undefined)
    ? conditionalPUp_experiment
    : REGIME_STATES.reduce(
        (s, state, i) => s + stateWeightsForUp[i] * regimeUpRates[state], 0,
      );
  let conditionalWeight = assetProfile.type === 'etf' ? 0.80
    : assetProfile.type === 'commodity' ? 0.60
    : assetProfile.type === 'equity' ? 0.65
    : 0.40; // crypto

  // Apply PR3B short-horizon crypto override if provided
  if (cryptoShortHorizonConditionalWeight !== undefined && assetProfile.type === 'crypto' && horizon <= 14) {
    conditionalWeight = cryptoShortHorizonConditionalWeight;
  }

  const calibrationCenter = conditionalWeight * conditionalPUp + (1 - conditionalWeight) * baseRate;

  // Drift-based calibration: shifts the distribution center (P(up)) toward
  // calibrationCenter while preserving the survival curve's S-shape.
  // Unlike the old probability-level shrinkage which compressed ALL probabilities
  // toward center (destroying tails), this only adjusts the drift parameter.
  let activeKappaMultiplier = assetProfile.kappaMultiplier;
  
  // Apply PR3B short-horizon crypto override if provided
  if (cryptoShortHorizonKappaMultiplier !== undefined && assetProfile.type === 'crypto' && horizon <= 14) {
    activeKappaMultiplier = cryptoShortHorizonKappaMultiplier;
  }

  // --- Regime run length: consecutive days ending in the current regime ---
  let regimeRunLength = 1;
  for (let i = regimeSeq.length - 2; i >= 0; i--) {
    if (regimeSeq[i] === currentRegime) regimeRunLength++;
    else break;
  }

  const rawPUp = interpolateSurvival(rawDistribution, currentPrice);
  const lookbackAgreementRatio = lookbackAgreement / totalLookbacks;

  const matureBullCalibrationActive = !!matureBullCalibration &&
    (ticker === 'BTC' || ticker === 'BTC-USD') &&
    assetProfile.type === 'crypto' &&
    horizon === 14 &&
    currentRegime === 'bull' &&
    rawPUp > 0.60 &&
    momentum.acceleration <= 0;

  const distribution = calibrateProbabilities(rawDistribution, {
    ensembleConsensus: ensemble.consensus,
    historicalDays: returns.length,
    hmmConverged: hmmMeta?.converged ?? false,
    baseRate: calibrationCenter,
    kappaMultiplier: activeKappaMultiplier,
    currentRegime,
    currentPrice,
    driftN: calDriftN,
    volN: calVolN,
    nu: assetProfile.studentTNu,
    matureBullCalibrationActive,
  });

  // --- Base-rate floor (Idea S, Round 4) ---
  // Adaptive bear margin: scales with how bullish the base rate is.
  // In strongly bullish markets, require very strong evidence to predict down.
  // Near 50%, the floor is tighter because even small over-prediction of bear hurts.
  const calPUpPreFloor = interpolateSurvival(distribution, currentPrice);
  // bearMargin scales linearly: from 0.05 (at baseRate=0.5) to 0.10 (at baseRate=0.75+)
  // This ensures the model defaults to UP for assets near 50-55% base rate
  const { bearMargin, pUpFloor } = computeBaseRateFloor(
    baseRate,
    calibrationCenter,
    assetProfile.type === 'crypto',
    horizon,
    cryptoShortHorizonPUpFloor,
    cryptoShortHorizonBearMarginMultiplier
  );
  if (calPUpPreFloor < pUpFloor) {
    const deficit = pUpFloor - calPUpPreFloor;
    for (const pt of distribution) {
      pt.probability = Math.min(1.0, pt.probability + deficit);
    }
  }

  // --- P(up) from both raw and calibrated distributions for confidence scoring ---
  // Raw P(up) reflects model's actual signal strength BEFORE calibration compresses it.
  // This gives a much more discriminative decisiveness score.
  const calPUp = interpolateSurvival(distribution, currentPrice);

  // Recent daily volatility (20-day std of daily returns)
  const recentDailyVol = returns.length >= 20
    ? Math.sqrt(returns.slice(-20).reduce((s, v) => s + v * v, 0) / 20)
    : undefined;

  const anchorCoverage = assessAnchorCoverage(polymarketAnchors, currentPrice, { ticker, horizonDays: horizon });

  // --- Optional R²_OS (leave-one-out on training tail) ---
  const validation = computeValidationR2OS({
    assetType: assetProfile.type,
    horizon,
    regimeSeq,
    logReturns,
    assetProfile,
    transitionDecayOverride,
  });
  const r2os = validation.r2os;

  // --- Prediction confidence (Idea M: selective prediction) ---
  // Use raw P(up) for decisiveness (how far from coin flip) since calibrated P(up)
  // is often compressed to 0.45-0.55, making decisiveness uniformly low.
  // Include base-rate alignment (Round 5): predictions aligned with the empirical
  // base rate are much more reliable than contra-directional ones.
  const predictionConfidence = computePredictionConfidence({
    pUp: rawPUp,
    ensembleConsensus: ensemble.consensus,
    hmmConverged: hmmMeta?.converged ?? false,
    regimeRunLength,
    structuralBreak: breakResult.detected,
    assetType: assetProfile.type,
    recentVol: recentDailyVol,
    momentumAgreement: lookbackAgreement / totalLookbacks,
    calibratedPUp: calPUp,
    baseRate,
    trustedAnchors: anchorCoverage.trustedAnchors,
    horizonDays: horizon,
    outOfSampleR2: r2os,
    breakConfidencePolicy: divergenceWeightedBreakConfidence
      ? 'divergence_weighted'
      : trendPenaltyOnlyBreakConfidence
        ? 'trend_penalty_only'
        : 'default',
    skipSidewaysBreakPenalty: trendPenaltyOnlyBreakConfidence === true,
    regimeState: currentRegime,
    structuralBreakDivergence: breakResult.divergence,
    divergencePenaltySchedule,
  });

  // --- Trajectory computation (optional day-by-day forecast) ---
  let trajectoryResult: TrajectoryPoint[] | undefined;
  if (params.trajectory) {
    const trajDays = Math.min(30, Math.max(1, params.trajectoryDays ?? horizon));
    trajectoryResult = computeTrajectory(
      currentPrice, trajDays, P, regimeStats, currentRegime,
      combinedDriftAdj, hmmOverride, 1000, assetProfile.studentTNu,
      recentDailyVol, mixture,
    );

    // Align trajectory P(Up) with the calibrated CDF at the final day.
    // The trajectory computes P(Up) independently (uncalibrated), which can
    // diverge from the calibrated distribution. Override the final day and
    // interpolate intermediate days to maintain monotone consistency.
    if (trajectoryResult.length > 0) {
      const calibratedPUpFinal = calPUp;
      const lastIdx = trajectoryResult.length - 1;
      const rawFinalPUp = trajectoryResult[lastIdx].pUp;
      // Only override if they diverge by more than 2pp (avoid unnecessary perturbation)
      if (Math.abs(rawFinalPUp - calibratedPUpFinal) > 0.02) {
        for (let i = 0; i <= lastIdx; i++) {
          // Linear interpolation from 0.5 (day 0) toward calibrated P(Up) at final day
          const t = (i + 1) / (lastIdx + 1);
          trajectoryResult[i].pUp = Math.round(
            (0.5 + t * (calibratedPUpFinal - 0.5)) * 1000,
          ) / 1000;
        }
      }
    }
  }

  // --- Final bounds enforcement ---
  // Anchor blending, calibration, and monotonicity passes can push probabilities
  // outside MC-derived CI bounds. Enforce lowerBound ≤ probability ≤ upperBound.
  for (const pt of distribution) {
    if (pt.upperBound != null && pt.upperBound < pt.probability) {
      pt.upperBound = pt.probability;
    }
    if (pt.lowerBound != null && pt.lowerBound > pt.probability) {
      pt.lowerBound = pt.probability;
    }
  }

  // --- Scenario probabilities (derived from calibrated CDF) ---
  const scenarios = computeScenarioProbabilities(distribution, currentPrice);

  const COMMODITY_MODEL_ONLY_MIN_R2 = -0.02;
  const COMMODITY_MODEL_ONLY_MIN_CONFIDENCE = 0.15;

  const validationAcceptable = assetProfile.type === 'crypto' && horizon <= 14 && (r2os ?? 0) >= -0.05
    ? true
    : (r2os ?? 0) >= 0;

  // Commodity model-only path: allow emission when no anchors exist but model meets minimum quality bar.
  const commodityModelOnlyAllowed =
    assetProfile.type === 'commodity' &&
    anchorCoverage.trustedAnchors === 0 &&
    typeof r2os === 'number' &&
    Number.isFinite(r2os) &&
    r2os >= COMMODITY_MODEL_ONLY_MIN_R2 &&
    predictionConfidence >= COMMODITY_MODEL_ONLY_MIN_CONFIDENCE;
  
  const anchorBasedDiag =
    anchorCoverage.trustedAnchors > 0 &&
    anchorCoverage.quality === 'good' &&
    validationAcceptable &&
    !breakResult.detected;

  // Allows PR3F on price-only BTC harness: crypto + short-horizon + zero anchors + acceptable validation + no break.
  const priceOnlyCryptoDiag =
    assetProfile.type === 'crypto' &&
    horizon <= 14 &&
    anchorCoverage.trustedAnchors === 0 &&
    validationAcceptable &&
    !breakResult.detected;

  // Commodity model-only diagnostics: allow when commodity model-only conditions are met and no structural break.
  const commodityModelOnlyDiag =
    commodityModelOnlyAllowed &&
    !breakResult.detected;

  const diagnosticsPass = anchorBasedDiag || priceOnlyCryptoDiag || commodityModelOnlyDiag;

  const rawPUpForBlend = interpolateSurvival(rawDistribution, currentPrice);
  const calPUpForBlend = interpolateSurvival(distribution, currentPrice);
  
  const usePr3fBlend = Boolean(
    pr3fCryptoShortHorizonDisagreementPrior &&
    assetProfile.type === 'crypto' &&
    horizon <= 14 &&
    Math.abs(calPUpForBlend - rawPUpForBlend) > 0.05 &&
    diagnosticsPass
  );

  const useRawDecision = Boolean(
    cryptoShortHorizonRawDecisionAblation && 
    assetProfile.type === 'crypto' && 
    horizon <= 14 &&
    !usePr3fBlend
  );

  const useRawDirectionHybrid = Boolean(
    rawDirectionHybrid &&
    assetProfile.type === 'crypto' &&
    horizon <= 14 &&
    !usePr3fBlend
  );

  let decisionDistribution = distribution;
  let decisionScenarios = scenarios;
  let rawActionSignal: ActionSignal | null = null;

  if (usePr3fBlend) {
    decisionDistribution = distribution.map((d, i) => ({
      price: d.price,
      probability: (d.probability + rawDistribution[i].probability) / 2,
      lowerBound: d.lowerBound,
      upperBound: d.upperBound,
      source: 'blend',
    }));
    decisionScenarios = computeScenarioProbabilities(decisionDistribution, currentPrice);
  } else if (useRawDecision) {
    decisionDistribution = rawDistribution;
    decisionScenarios = computeScenarioProbabilities(rawDistribution, currentPrice);
  } else if (useRawDirectionHybrid) {
    rawActionSignal = computeActionSignal(
      rawDistribution,
      currentPrice,
      undefined,
      undefined,
      horizon,
      recentDailyVol,
      computeScenarioProbabilities(rawDistribution, currentPrice),
      assetProfile.type,
    );
  }

  const actionSignal = computeActionSignal(
    decisionDistribution, currentPrice, undefined, undefined, horizon,
    recentDailyVol, decisionScenarios, assetProfile.type,
  );

  if (useRawDirectionHybrid && rawActionSignal) {
    actionSignal.recommendation = rawActionSignal.recommendation;
  }

  return {
    ticker,
    currentPrice,
    horizon,
    distribution,
    rawDistribution,
    actionSignal,
    scenarios,
    predictionConfidence,
    trajectory: trajectoryResult,
    metadata: {
      polymarketAnchors: polymarketAnchors.filter(a => a.trustScore === 'high').length,
      regimeState: currentRegime,
      sentimentAdjustment: sentimentShift,
      historicalDays: returns.length,
      mixingTimeWeight: mixWeight,
      secondEigenvalue: rho,
      outOfSampleR2: r2os,
      validationMetric: validation.validationMetric,
      stateObservationCounts,
      sparseStates,
      structuralBreakDetected: breakResult.detected,
      structuralBreakDivergence: breakResult.divergence,
      ciWidened: breakResult.detected,
      anchorDivergenceWarnings,
      anchorCoverage,
      goodnessOfFit: gofResult,
      hmm: hmmMeta ?? null,
      ensemble,
      pr3fDisagreementBlendActive: usePr3fBlend,
      rawDirectionHybridActive: useRawDirectionHybrid,
      pr3gRecencyWeightingActive: pr3gDecayRate !== undefined,
      startStateMixtureActive: useStartStateMixture,
      sidewaysSplitActive,
      matureBullCalibrationActive,
      trendPenaltyOnlyBreakConfidenceActive: trendPenaltyOnlyBreakConfidence ? true : undefined,
      divergenceWeightedBreakConfidenceActive: divergenceWeightedBreakConfidence ? true : undefined,
      regimeSpecificSigmaActive: regimeSpecificSigma === true &&
        Math.max(...stateWeightsForUp) > (regimeSpecificSigmaThreshold ?? 0.60),
      breakFallbackCandidateId: breakFallbackCandidate?.id ?? undefined,
      breakFallbackMode: (breakResult.detected && breakFallbackCandidate) ? breakFallbackCandidate.mode : undefined,
    }
  };
}
// 9. LangChain tool wrapper
// ---------------------------------------------------------------------------

export const MARKOV_DISTRIBUTION_DESCRIPTION = `
**markov_distribution** — Full probability distribution for a stock/ETF/commodity price at a specified horizon.

**Use when:**
- The user asks for a probability distribution of future prices (not a point estimate)
- The query includes a specific price target and horizon (e.g. "Will NVDA hit $1000 in 30 days?")
- You already have ≥2 Polymarket price threshold markets available as anchors
- You want to interpolate between Polymarket anchors using regime-aware Markov transitions

**Commodity tickers:** For commodities, always use the liquid ETF ticker:
- Gold/XAUUSD → use **GLD** (ETF that tracks gold spot price)
- Silver/SILVER/XAGUSD → use **SLV**
- Oil/Crude/WTICOUSD → use **USO**
- Natural Gas → use **UNG**
 - Barrick Gold / GOLD stock / $GOLD → use **GOLD** (Barrick equity)
GLD, SLV, USO etc. are ETFs that replicate the underlying commodity price.

**What it does:**
- Combines Polymarket real-money anchors + historical regime transitions + sentiment
- Returns P(price > X) for 20+ price levels with 90% Monte Carlo confidence intervals
- Flags structural breaks (regime change mid-window), sparse states, and cross-platform divergence
- Automatically applies YES-bias correction (×${YES_BIAS_MULTIPLIER}) to raw Polymarket probabilities
- Accepts only **terminal threshold** anchors (e.g. "be above $60,000 on April 2", "settle below $4,200")
- Rejects touch / barrier markets like "reach $70K", "hit $70K", "dip to $64K", or "stay above $60K through March"

**Do NOT use when:**
- The query is a simple binary probability (use probability_assessment skill instead)
- You have fewer than 10 historical prices (not enough for regime estimation)
- The horizon is > 90 trading days (model accuracy degrades at long horizons)
`.trim();

export const markovDistributionTool = new DynamicStructuredTool({
  name: 'markov_distribution',
  description: `
Generate a full probability distribution for a stock/ETF/commodity price at a specified horizon.
Combines Polymarket threshold markets (real-money anchors) with historical Markov regime
transitions to produce P(price > X) for each price level in the distribution.

Use when the query asks for a probability distribution of future prices, not just a point estimate.
Requires: ticker symbol and horizon in trading days (1–90). Historical prices are auto-fetched if not provided.

IMPORTANT — Anchor semantics: only pass **terminal threshold** markets as anchors
(e.g. "be above $60,000 on April 2", "settle below $4,200"). Do NOT pass
touch/barrier markets like "reach $70K", "hit $70K", "dip to $64K", "drop below
$50K", or "stay above $60K through March" — those are path-dependent, not
P(price > X at horizon).

IMPORTANT — Commodity tickers: For commodities, use the liquid ETF ticker for best data availability:
  Gold/XAUUSD → GLD, Silver/SILVER/XAGUSD → SLV, Oil → USO, Natural Gas → UNG, Copper → CPER.
  Use GOLD only for Barrick Gold equity; GLD is the commodity-gold proxy.

Set trajectory=true for a day-by-day price forecast with expected price, 90% CI, and P(up) at each day.
Use trajectoryDays to control the number of days (1–30, default=horizon).
`.trim(),
  schema: z.object({
    ticker: z.string().describe('Stock/ETF/commodity ticker symbol, e.g. NVDA, SPY, BTC-USD, GLD. For commodities, prefer the liquid ETF (GLD for gold, SLV for silver, USO for oil) over futures tickers.'),
    horizon: z.number().int().min(1).max(90).describe('Forecast horizon in trading days'),
    currentPrice: z.number().optional().describe('Current price (fetched automatically if omitted)'),
    historicalPrices: z.array(z.number()).optional().describe(
      'Daily close prices oldest-first. Auto-fetched if omitted or empty. Minimum 10 required, 30+ recommended.',
    ),
    polymarketMarkets: z.array(z.object({
      question:    z.string(),
      probability: z.number().min(0).max(1),
      volume:      z.number().optional(),
      createdAt:   z.union([z.string(), z.number()]).optional(),
    })).optional().default([]).describe('Polymarket markets with dollar price thresholds (optional, defaults to empty)'),
    sentiment: z.object({
      bullish: z.number().min(0).max(100),
      bearish: z.number().min(0).max(100),
    }).optional().describe('Sentiment signal from social_sentiment tool. Accepts decimals (0–1) or percentages (0–100, auto-normalized).'),
    kalshiAnchors: z.array(z.object({
      price: z.number(),
      probability: z.number().min(0).max(1),
      volume: z.number().optional(),
    })).optional().describe('Kalshi prediction market anchors for cross-platform validation (optional)'),
    trajectory: z.boolean().optional().default(false)
      .describe('Return day-by-day price trajectory instead of single-horizon snapshot'),
    trajectoryDays: z.number().int().min(1).max(30).optional()
      .describe('Number of days for trajectory (default: horizon, max 30)'),
  }),
  func: async (input) => {
    // Auto-fetch historical prices if not provided or too few
    let historicalPrices = input.historicalPrices ?? [];
    if (historicalPrices.length < 10) {
      const fetched = await fetchHistoricalPrices(input.ticker, 120);
      if (fetched.length >= 10) {
        historicalPrices = fetched;
      } else {
        return formatToolResult({
          _tool: 'markov_distribution',
          error: `Could not fetch enough historical prices for ${input.ticker}. Got ${fetched.length} prices (minimum 10 required). Try providing historicalPrices manually via get_market_data first.`,
        });
      }
    }

    const price = input.currentPrice
      ?? historicalPrices[historicalPrices.length - 1];

    const polymarketMarkets = (input.polymarketMarkets && input.polymarketMarkets.length > 0)
      ? input.polymarketMarkets
      : await fetchCandidatePolymarketAnchors(input.ticker, input.horizon);

    const result = await computeMarkovDistribution({
      ticker:            input.ticker,
      horizon:           input.horizon,
      currentPrice:      price,
      historicalPrices:  historicalPrices,
      polymarketMarkets,
      sentiment:         input.sentiment,
      kalshiAnchors:     input.kalshiAnchors,
      trajectory:        input.trajectory,
      trajectoryDays:    input.trajectoryDays,
    });

    const { metadata: m, actionSignal: sig } = result;
    const lvl = sig.actionLevels;

    // Format helpers
    const fmt = (n: number) => n.toFixed(2);
    const pct = (n: number) => (n * 100).toFixed(1);
    const buyPct  = pct(sig.buyProbability);
    const holdPct = pct(sig.holdProbability);
    const sellPct = pct(sig.sellProbability);
    const buyThr  = (sig.buyThreshold  * 100).toFixed(0);
    const sellThr = (sig.sellThreshold * 100).toFixed(0);
    const retSign = sig.expectedReturn >= 0 ? '+' : '';
    const retPct  = pct(sig.expectedReturn);
    const rrLabel = sig.riskRewardRatio.toFixed(2);
    const recEmoji = sig.recommendation === 'BUY' ? '📈' : sig.recommendation === 'SELL' ? '📉' : '➡️ ';
    const cryptoShortHorizon = getAssetProfile(result.ticker).type === 'crypto' && result.horizon <= 14;
    const r2NeutralForCrypto = cryptoShortHorizon
      && m.validationMetric === 'horizon_return'
      && m.anchorCoverage.quality === 'good'
      && m.anchorCoverage.trustedAnchors >= 2
      && typeof m.outOfSampleR2 === 'number'
      && Number.isFinite(m.outOfSampleR2)
      && m.outOfSampleR2 >= -0.04;
    const sparseCryptoAnchorAllowed = cryptoShortHorizon
      && m.validationMetric === 'horizon_return'
      && m.anchorCoverage.quality === 'sparse'
      && m.anchorCoverage.trustedAnchors >= 1
      && typeof m.outOfSampleR2 === 'number'
      && Number.isFinite(m.outOfSampleR2)
      && m.outOfSampleR2 >= -0.05;
    const hasPositiveR2 = typeof m.outOfSampleR2 === 'number' && Number.isFinite(m.outOfSampleR2) && m.outOfSampleR2 > 0;
    const validationAcceptable = hasPositiveR2 || r2NeutralForCrypto || sparseCryptoAnchorAllowed;

    const COMMODITY_WRAPPER_MIN_R2 = -0.02;
    const COMMODITY_WRAPPER_MIN_CONFIDENCE = 0.15;
    const assetProfile = getAssetProfile(result.ticker);
    const commodityModelOnly =
      assetProfile.type === 'commodity' &&
      m.anchorCoverage.trustedAnchors === 0 &&
      typeof m.outOfSampleR2 === 'number' &&
      Number.isFinite(m.outOfSampleR2) &&
      m.outOfSampleR2 >= COMMODITY_WRAPPER_MIN_R2 &&
      result.predictionConfidence >= COMMODITY_WRAPPER_MIN_CONFIDENCE &&
      !m.structuralBreakDetected;

    const abstainReasons: string[] = [];

    if (m.anchorCoverage.trustedAnchors === 0 && !commodityModelOnly) {
      abstainReasons.push('No trusted terminal prediction-market anchors are available for this horizon.');
    } else if (m.anchorCoverage.trustedAnchors === 0 && commodityModelOnly) {
      // No abstain reason for anchors — commodity model-only bypass applies.
    } else if (m.anchorCoverage.quality !== 'good' && !sparseCryptoAnchorAllowed) {
      abstainReasons.push(`Prediction-market anchor coverage is ${m.anchorCoverage.quality}, so calibrated scenario buckets would be overly model-driven.`);
    }

    if (!validationAcceptable && !commodityModelOnly) {
      if (m.outOfSampleR2 === null || !Number.isFinite(m.outOfSampleR2)) {
        abstainReasons.push('Out-of-sample Markov validation is unavailable for this run.');
      } else {
        abstainReasons.push(`Out-of-sample Markov R² is ${m.outOfSampleR2.toFixed(3)}, so the model does not beat a historical-mean baseline.`);
      }
    }

    const canEmitCanonical =
      (m.anchorCoverage.trustedAnchors > 0
        && m.anchorCoverage.quality === 'good'
        && validationAcceptable)
      || sparseCryptoAnchorAllowed
      || commodityModelOnly;

    const calibrationMode: 'anchored' | 'model_only' = commodityModelOnly ? 'model_only' : 'anchored';
    const anchorBypassApplied = commodityModelOnly;

    // --- Section 1: Decision Card (FIRST — most important) ---
    const decisionCard = [
      `${recEmoji} ${sig.recommendation}  [${sig.confidence} confidence]  |  Expected return: ${retSign}${retPct}%  |  Risk/reward: ${rrLabel}×`,
      '',
      '┌─ Your Options ─────────────────────────────────────────┐',
      `│  📈 BUY   ${buyPct.padStart(5)}% chance price rises  >${buyThr}% above current   │`,
      `│  ➡️  HOLD  ${holdPct.padStart(5)}% chance price stays within ±${sellThr}%-${buyThr}%    │`,
      `│  📉 SELL  ${sellPct.padStart(5)}% chance price falls >${sellThr}% below current   │`,
      '└────────────────────────────────────────────────────────┘',
    ];

    // --- Section 2: Action Plan with price levels ---
    const actionPlan = [
      '',
      '🎯 Action Plan',
      '─'.repeat(60),
      `   Bull case (20% prob):  $${fmt(lvl.bullCase)}   (+${pct((lvl.bullCase - result.currentPrice) / result.currentPrice)}%)`,
      `   Target (30% prob):     $${fmt(lvl.targetPrice)}   (+${pct((lvl.targetPrice - result.currentPrice) / result.currentPrice)}%)`,
      `   Median forecast:       $${fmt(lvl.medianPrice)}   (${(lvl.medianPrice >= result.currentPrice ? '+' : '')}${pct((lvl.medianPrice - result.currentPrice) / result.currentPrice)}%)`,
      `   Bear case (80% prob):  $${fmt(lvl.bearCase)}   (${(lvl.bearCase >= result.currentPrice ? '+' : '')}${pct((lvl.bearCase - result.currentPrice) / result.currentPrice)}%)`,
      `   Stop-loss (90% prob):  $${fmt(lvl.stopLoss)}   (${(lvl.stopLoss >= result.currentPrice ? '+' : '')}${pct((lvl.stopLoss - result.currentPrice) / result.currentPrice)}%)`,
    ];

    // Generate contextual "what to do" based on recommendation
    const whatToDo: string[] = [''];
    if (sig.recommendation === 'BUY') {
      whatToDo.push(`💡 If buying: Enter near $${fmt(result.currentPrice)}, target $${fmt(lvl.targetPrice)}, stop-loss at $${fmt(lvl.stopLoss)}`);
      whatToDo.push(`   Max gain: +${pct((lvl.bullCase - result.currentPrice) / result.currentPrice)}%  |  Max loss to stop: ${pct((lvl.stopLoss - result.currentPrice) / result.currentPrice)}%`);
    } else if (sig.recommendation === 'SELL') {
      whatToDo.push(`💡 If selling/shorting: Exit at $${fmt(result.currentPrice)}, re-enter below $${fmt(lvl.bearCase)}`);
      whatToDo.push(`   Expected downside: ${pct((lvl.medianPrice - result.currentPrice) / result.currentPrice)}% to median`);
    } else {
      whatToDo.push(`💡 Range-bound: No strong edge. Wait for $${fmt(lvl.targetPrice)} (bullish break) or $${fmt(lvl.stopLoss)} (bearish break)`);
      whatToDo.push(`   Consider selling puts at $${fmt(lvl.stopLoss)} or calls at $${fmt(lvl.targetPrice)} to capture premium`);
    }

    // --- Section 3: Scenario Probability Table (derived from CDF) ---
    const scenarioSection: string[] = [];
    if (result.scenarios) {
      const sc = result.scenarios;
      scenarioSection.push('');
      scenarioSection.push('📊 Scenario Probabilities (from calibrated distribution)');
      scenarioSection.push('─'.repeat(60));
      for (const b of sc.buckets) {
        const pctStr = (b.probability * 100).toFixed(1).padStart(5);
        const lo = b.priceRange[0] !== null ? `$${fmt(b.priceRange[0])}` : '     —';
        const hi = b.priceRange[1] !== null ? `$${fmt(b.priceRange[1])}` : '—     ';
        scenarioSection.push(`  ${b.label.padEnd(10)} ${pctStr}%   ${lo} – ${hi}`);
      }
      const expRetSign = sc.expectedReturn >= 0 ? '+' : '';
      scenarioSection.push('');
      scenarioSection.push(`  Expected: $${fmt(sc.expectedPrice)} (${expRetSign}${(sc.expectedReturn * 100).toFixed(1)}%)  |  P(up): ${(sc.pUp * 100).toFixed(1)}%`);
    }

    // --- Section 3b: Methodology note ---
    const anchorCount = m.polymarketAnchors;
    const methodNote = anchorCount > 0
      ? `ℹ️  Method: 3-state Markov regime model (${m.historicalDays}d history) blended with ${anchorCount} Polymarket anchor(s). Probabilities from calibrated survival function with Monte Carlo confidence intervals.`
      : commodityModelOnly
        ? `ℹ️  Method: 3-state Markov regime model (${m.historicalDays}d history), model-only commodity emission. No prediction market anchors — distribution is 100% Markov-model driven.`
        : `ℹ️  Method: 3-state Markov regime model (${m.historicalDays}d history). Probabilities from calibrated survival function with Monte Carlo confidence intervals. No prediction market anchors available.`;

    // --- Section 4: Header and metadata ---
    const mixingLine = commodityModelOnly
      ? 'Calibration: model-only (commodity bypass, no anchors)'
      : `Mixing: ${pct(m.mixingTimeWeight)}% Markov / ${pct(1 - m.mixingTimeWeight)}% Anchors`;
    const header = [
      '',
      `📊 Markov Distribution: ${result.ticker} | Horizon: ${result.horizon}d`,
      `Current: $${fmt(result.currentPrice)} | Regime: ${m.regimeState}`,
      `Anchors: ${m.polymarketAnchors} trusted | Anchor quality: ${m.anchorCoverage.quality.toUpperCase()}`,
      mixingLine,
    ];

    // --- Section 4: Warnings ---
    const warnings: string[] = [];
    if (m.anchorCoverage.warning) warnings.push(`⚠️ ${m.anchorCoverage.warning}`);
    if (m.structuralBreakDetected)
      warnings.push(`⚠️ Structural break detected (divergence=${m.structuralBreakDivergence.toFixed(3)}); CI widened 50%`);
    if (m.sparseStates.length > 0)
      warnings.push(`⚠️ Sparse states (<5 obs): ${m.sparseStates.join(', ')} — transitions prior-dominated`);
    if (m.anchorDivergenceWarnings.length > 0)
      warnings.push(`⚠️ Cross-platform divergence: ${m.anchorDivergenceWarnings.map(w => `$${w.price} (${w.divergencePp.toFixed(1)}pp)`).join(', ')}`);
    if (m.outOfSampleR2 !== null)
      warnings.push(`R²_OS: ${m.outOfSampleR2.toFixed(3)} (>0 = Markov adds value over mean)`);
    if (m.goodnessOfFit && !m.goodnessOfFit.passes)
      warnings.push(`⚠️ Markov fit test failed (χ²=${m.goodnessOfFit.chiSquared.toFixed(1)}, p=${m.goodnessOfFit.pValue.toFixed(3)}) — transitions may not follow Markov property`);
    if (m.goodnessOfFit?.passes)
      warnings.push(`✓ Markov fit test passed (p=${m.goodnessOfFit.pValue.toFixed(3)})`);
    if (result.predictionConfidence < RECOMMENDED_CONFIDENCE_THRESHOLD)
      warnings.push(`⚠️ Low confidence (${result.predictionConfidence.toFixed(2)} < ${RECOMMENDED_CONFIDENCE_THRESHOLD.toFixed(2)}): accuracy drops to ~55% below this threshold; consider waiting for higher-confidence signal`);

    // --- Section 5: Full distribution table ---
    const table = [
      '',
      'Price         P(>price)    90% CI                 Source',
      '─'.repeat(60),
      ...result.distribution.map(d =>
        `$${d.price.toFixed(2).padStart(9)}   ${(d.probability * 100).toFixed(1).padStart(5)}%   `
        + `[${(d.lowerBound * 100).toFixed(1)}%–${(d.upperBound * 100).toFixed(1)}%]   ${d.source}`,
      ),
    ];

    // --- Section 6: Trajectory table (if requested) ---
    const trajectorySection: string[] = [];
    if (result.trajectory && result.trajectory.length > 0) {
      const trajDays = result.trajectory.length;
      trajectorySection.push('');
      trajectorySection.push(`═══ ${trajDays}-DAY PRICE TRAJECTORY: ${result.ticker} ═══`);
      trajectorySection.push(`Current: $${fmt(result.currentPrice)} | Regime: ${m.regimeState} | Confidence: ${result.predictionConfidence.toFixed(2)}`);
      trajectorySection.push('');
      trajectorySection.push('Day │ Expected │     90% CI Range    │ P(up) │ Return');
      trajectorySection.push('────┼──────────┼─────────────────────┼───────┼────────');
      for (const pt of result.trajectory) {
        const dayStr = String(pt.day).padStart(3);
        const expStr = `$${fmt(pt.expectedPrice)}`.padStart(8);
        const loStr = `$${fmt(pt.lowerBound)}`;
        const hiStr = `$${fmt(pt.upperBound)}`;
        const ciStr = `${loStr} – ${hiStr}`.padStart(19);
        const pUpStr = `${(pt.pUp * 100).toFixed(0)}%`.padStart(5);
        trajectorySection.push(`${dayStr} │ ${expStr} │ ${ciStr} │ ${pUpStr} │ ${pt.cumulativeReturn}`);
      }
      // Trend summary
      const first = result.trajectory[0];
      const last = result.trajectory[result.trajectory.length - 1];
      const ciWidenPerDay = ((last.upperBound - last.lowerBound) - (first.upperBound - first.lowerBound)) / (last.day - first.day || 1);
      const trendDir = last.pUp > 0.55 ? '📈 Trend: Bullish drift' : last.pUp < 0.45 ? '📉 Trend: Bearish drift' : '➡️  Trend: Sideways';
      trajectorySection.push('');
      trajectorySection.push(`${trendDir}, CI widens ~$${ciWidenPerDay.toFixed(2)}/day`);
      trajectorySection.push('⚠️  Point estimates are probability-weighted means, not forecasts.');
      trajectorySection.push('    The CI range is the honest measure of uncertainty.');
    }

    const forecastHint = buildForecastHint({
      canEmitCanonical,
      ticker: result.ticker,
      horizon: result.horizon,
      expectedReturn: result.actionSignal.expectedReturn,
      mixingTimeWeight: m.mixingTimeWeight,
      predictionConfidence: result.predictionConfidence,
    });

    const report = canEmitCanonical
      ? [
        ...decisionCard,
        ...actionPlan,
        ...whatToDo,
        ...scenarioSection,
        '',
        methodNote,
        ...header,
        ...(warnings.length > 0 ? ['', ...warnings] : []),
        ...table,
        ...trajectorySection,
      ].filter(l => l !== undefined).join('\n')
      : [
        '🚫 Diagnostics-only Markov output',
        '',
        `No calibrated scenario distribution was emitted for ${result.ticker} at the ${result.horizon}-day horizon.`,
        '',
        'Why this abstained:',
        ...abstainReasons.map((reason) => `- ${reason}`),
        '',
        ...header,
        ...(warnings.length > 0 ? ['', ...warnings] : []),
      ].filter(l => l !== undefined).join('\n');

    return formatToolResult({
      _tool: 'markov_distribution',
      status: canEmitCanonical ? 'ok' : 'abstain',
      manualSynthesisForbidden: !canEmitCanonical,
      abstainReasons,
      report,
      forecastHint,
      canonical: {
        ticker: result.ticker,
        currentPrice: result.currentPrice,
        horizon: result.horizon,
        scenarios: canEmitCanonical ? result.scenarios : null,
        actionSignal: canEmitCanonical ? result.actionSignal : null,
        diagnostics: {
          totalAnchors: m.anchorCoverage.totalAnchors,
          trustedAnchors: m.anchorCoverage.trustedAnchors,
          anchorQuality: m.anchorCoverage.quality,
          anchorWarning: m.anchorCoverage.warning || null,
          regimeState: m.regimeState,
          mixingTimeWeight: m.mixingTimeWeight,
          markovWeight: commodityModelOnly ? 1 : m.mixingTimeWeight,
          anchorWeight: commodityModelOnly ? 0 : 1 - m.mixingTimeWeight,
          outOfSampleR2: m.outOfSampleR2,
          structuralBreakDetected: m.structuralBreakDetected,
          structuralBreakDivergence: m.structuralBreakDivergence,
          ciWidened: m.ciWidened,
          predictionConfidence: result.predictionConfidence,
          calibrationMode,
          anchorBypassApplied,
          status: canEmitCanonical ? 'ok' : 'abstain',
          canEmitCanonical,
          manualSynthesisForbidden: !canEmitCanonical,
          abstainReasons,
          warnings,
        },
      },
      distribution: canEmitCanonical ? result.distribution : null,
      trajectory: canEmitCanonical ? (result.trajectory ?? null) : null,
    });
  },
});
