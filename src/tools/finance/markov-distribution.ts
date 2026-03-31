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
import { YES_BIAS_MULTIPLIER } from '../../utils/ensemble.js';
import { baumWelch, predict as hmmPredict, type HMMParams } from './hmm.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

/**
 * Joint regime state encoding both direction and volatility.
 *
 * Note: The original spec used a priority-override `high_vol` state that destroyed
 * directional information on volatile days. Joint states preserve both dimensions
 * (consistent with Nguyen 2018, where states encode distinct (mean, variance) pairs).
 */
export type RegimeState = 'bull' | 'bear' | 'high_vol_bull' | 'high_vol_bear' | 'sideways';

export const REGIME_STATES: RegimeState[] = [
  'bull', 'bear', 'high_vol_bull', 'high_vol_bear', 'sideways',
];

export const NUM_STATES = REGIME_STATES.length; // 5

/** Maps state name → matrix row/column index */
export const STATE_INDEX: Record<RegimeState, number> = {
  bull:          0,
  bear:          1,
  high_vol_bull: 2,
  high_vol_bear: 3,
  sideways:      4,
};

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
}

export interface SentimentSignal {
  bullish: number;  // 0–1
  bearish: number;  // 0–1
}

/** 5×5 stochastic matrix (rows sum to 1) */
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

export interface MarkovDistributionResult {
  ticker: string;
  currentPrice: number;
  horizon: number;
  distribution: MarkovDistributionPoint[];
  /** Actionable Buy / Hold / Sell signal derived from the distribution. */
  actionSignal: ActionSignal;
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
// 1. extractPriceThresholds
// ---------------------------------------------------------------------------

const PRICE_PATTERNS = [
  /(?:exceed|above|over|surpass)\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
  /(?:below|under|drop\s*(?:below|to))\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
  /(?:reach|hit)\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
  /\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)\s*(?:or\s*(?:higher|above|more))/i,
  // Commodity-specific patterns (per barrel, per ounce, etc.)
  /\$(\d[\d,]*(?:\.\d+)?)\s*(?:per\s*(?:barrel|bbl|ounce|oz|gallon|gal))/i,
  /(?:at|to|past)\s*\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
  // Plain price reference: "oil $150" or "gold $3000"
  /(?:oil|crude|gold|silver|bitcoin|btc|eth)\s+\$(\d[\d,]*(?:\.\d+)?(?:[KkMm])?)/i,
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
  }>,
): PriceThreshold[] {
  const seen = new Map<number, PriceThreshold>();

  for (const market of markets) {
    let matched: number | null = null;
    for (const pattern of PRICE_PATTERNS) {
      const m = market.question.match(pattern);
      if (m) {
        matched = parsePrice(m[1]);
        break;
      }
    }
    if (matched === null) continue;
    if (isNaN(matched) || matched <= 0) continue;

    const isYoung =
      market.createdAt != null &&
      Date.now() - (typeof market.createdAt === 'string'
        ? Date.parse(market.createdAt)
        : market.createdAt) < 48 * 60 * 60 * 1000;

    const hasVolume = (market.volume ?? 0) > 0;
    const trustScore: 'high' | 'low' = isYoung || !hasVolume ? 'low' : 'high';

    const rawProbability = market.probability;
    const corrected: PriceThreshold = {
      price: matched,
      rawProbability,
      probability: rawProbability * YES_BIAS_MULTIPLIER,
      trustScore,
      source: 'polymarket',
    };

    const existing = seen.get(matched);
    if (!existing || rawProbability > existing.rawProbability) {
      seen.set(matched, corrected);
    }
  }

  return Array.from(seen.values()).sort((a, b) => a.price - b.price);
}

// ---------------------------------------------------------------------------
// 2. classifyRegimeState
// ---------------------------------------------------------------------------

/**
 * Classify a trading day into a joint (direction × volatility) regime state.
 *
 * Volatility and directional return are orthogonal dimensions — a high-volatility
 * day can be strongly bullish or strongly bearish. Using a priority override (as in
 * the original spec) discards directional information. Joint states preserve both.
 *
 * @param dailyReturn - Return for the day (e.g. 0.012 = +1.2%)
 * @param dailyVolatility - Intraday vol proxy (e.g. (high - low) / open)
 * @param returnThreshold - Adaptive threshold for bull/bear classification (default 0.01)
 * @param volThreshold - Adaptive threshold for high-vol classification (default 0.02)
 */
export function classifyRegimeState(
  dailyReturn: number,
  dailyVolatility: number,
  returnThreshold = 0.01,
  volThreshold = 0.02,
): RegimeState {
  const highVol = dailyVolatility > volThreshold;
  if (highVol) {
    return dailyReturn > 0 ? 'high_vol_bull' : 'high_vol_bear';
  }
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
export function computeAdaptiveThresholds(returns: number[]): {
  returnThreshold: number;
  volThreshold: number;
} {
  if (returns.length === 0) return { returnThreshold: 0.01, volThreshold: 0.02 };
  const absReturns = returns.map(r => Math.abs(r)).sort((a, b) => a - b);
  const medianAbsReturn = absReturns[Math.floor(absReturns.length / 2)];
  return {
    returnThreshold: Math.max(0.001, 0.5 * medianAbsReturn),
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

/**
 * Estimate a 5×5 Markov transition matrix from a sequence of regime states.
 *
 * Smoothing: Dirichlet α scales inversely with sample size (default: max(0.01, 5/N)).
 * Converges to ~0.1 (Jeffreys prior) at N=50, and shrinks for larger samples to let
 * data dominate. Welton & Ades (2005) recommends α=0.1 for sparse counts; the adaptive
 * formula reduces over-smoothing for longer windows while still regularizing short ones.
 *
 * Default matrix (insufficient data): 0.6 diagonal, uniform off-diagonal.
 * offDiag = (1 − 0.6) / (NUM_STATES − 1) = 0.4 / 4 = 0.1 per cell (rows sum to 1.0).
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
): StructuralBreakResult {
  const mid = Math.floor(states.length / 2);
  const firstHalf  = states.slice(0, mid);
  const secondHalf = states.slice(mid);

  const firstHalfMatrix  = estimateTransitionMatrix(firstHalf,  alpha, 10);
  const secondHalfMatrix = estimateTransitionMatrix(secondHalf, alpha, 10);

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

  // Deflate: remove first eigenvector component, find second via power iteration
  let w = Array.from({ length: n }, (_, i) => (i === 0 ? 0.6 : 0.1));
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
export function estimateRegimeStats(
  returns: number[],
  states: RegimeState[],
): Record<RegimeState, RegimeStats> {
  const defaults: Record<RegimeState, RegimeStats> = {
    bull:          { meanReturn:  0.005, stdReturn: 0.010 },
    bear:          { meanReturn: -0.005, stdReturn: 0.012 },
    high_vol_bull: { meanReturn:  0.006, stdReturn: 0.025 },
    high_vol_bear: { meanReturn: -0.006, stdReturn: 0.025 },
    sideways:      { meanReturn:  0.000, stdReturn: 0.006 },
  };

  const bins: Record<RegimeState, number[]> = {
    bull: [], bear: [], high_vol_bull: [], high_vol_bear: [], sideways: [],
  };

  for (let i = 0; i < Math.min(returns.length, states.length); i++) {
    bins[states[i]].push(returns[i]);
  }

  const result = { ...defaults };
  for (const [state, vals] of Object.entries(bins) as [RegimeState, number[]][]) {
    if (vals.length >= 5) {
      const mean = vals.reduce((s, v) => s + v, 0) / vals.length;
      const variance = vals.reduce((s, v) => s + (v - mean) ** 2, 0) / vals.length;
      result[state] = { meanReturn: mean, stdReturn: Math.sqrt(variance) };
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
): MarkovDistributionPoint[] {
  const stepPct  = 0.015;
  let minPrice = currentPrice * Math.pow(1 - stepPct, numLevels / 2);
  let maxPrice = currentPrice * Math.pow(1 + stepPct, numLevels / 2);

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
  const initialIdx = STATE_INDEX[initialState];

  // n-step transition row for initial state
  const Pn = matPow(P, horizon);
  const stateWeights = Pn[initialIdx];

  // Compute regime-weighted drift and vol (observable Markov)
  const mu_obs = REGIME_STATES.reduce(
    (s, state, i) => s + stateWeights[i] * regimeStats[state].meanReturn, 0,
  );
  const sigma_obs = Math.sqrt(
    REGIME_STATES.reduce(
      (s, state, i) => s + stateWeights[i] * regimeStats[state].stdReturn ** 2, 0,
    ),
  );

  // Blend HMM forecast with observable Markov when available
  let mu_eff: number;
  let sigma_eff: number;
  if (hmmOverride) {
    const w = hmmOverride.weight; // HMM weight (0..1)
    mu_eff = w * hmmOverride.drift + (1 - w) * mu_obs;
    sigma_eff = w * hmmOverride.vol + (1 - w) * sigma_obs;
  } else {
    mu_eff = mu_obs;
    sigma_eff = sigma_obs;
  }

  const mu_n    = horizon * (mu_eff + momentumAdjustment);
  const sigma_n = sigma_eff * Math.sqrt(horizon);

  // Nearest anchor lookup helper
  const findAnchor = (price: number) => {
    const TOLERANCE_PCT = 0.02;
    return anchors.find(a => Math.abs(a.price - price) / price < TOLERANCE_PCT);
  };

  // Monte Carlo: vary initial state draw for CI
  const rng = (): number => Math.random();
  const ciSamples: Map<number, number[]> = new Map(prices.map(p => [p, []]));

  for (let s = 0; s < monteCarloSamples; s++) {
    // Perturb drift and vol within sampling uncertainty
    const perturbedMu  = mu_n    + (rng() - 0.5) * sigma_n * 0.2;
    const perturbedVol = sigma_n * (0.9 + rng() * 0.2);
    for (const price of prices) {
      const p = logNormalSurvival(currentPrice, price, perturbedMu, perturbedVol);
      ciSamples.get(price)!.push(p);
    }
  }

  // Build distribution points
  const rawPoints = prices.map(price => {
    const anchor = findAnchor(price);
    const markovEst = logNormalSurvival(currentPrice, price, mu_n, sigma_n);

    let probability: number;
    let source: 'polymarket' | 'markov' | 'blend';

    if (anchor && anchor.trustScore === 'high') {
      probability = mixWeight * markovEst + (1 - mixWeight) * anchor.probability;
      source = mixWeight > 0.9 ? 'markov' : mixWeight < 0.1 ? 'polymarket' : 'blend';
    } else if (anchor && anchor.trustScore === 'low') {
      // Low-trust anchors: weight them at half their nominal influence
      probability = mixWeight * markovEst + (1 - mixWeight) * 0.5 * anchor.probability
                  + (1 - mixWeight) * 0.5 * markovEst;
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
 */
export function computeActionSignal(
  distribution: MarkovDistributionPoint[],
  currentPrice: number,
  buyThreshold = 0.05,
  sellThreshold = 0.03,
  horizon = 30,
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

  const expectedReturn = (ePrice - currentPrice) / currentPrice;

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

  const riskRewardRatio = eDownside > 0 ? eUpside / eDownside : 1.0;

  // Horizon-scaled action thresholds for recommendation (calibrated via backtest grid search)
  const actionBuyThr  = horizon <= 7 ? 0.01 : horizon <= 30 ? 0.02 : 0.03;
  const actionSellThr = horizon <= 7 ? 0.008 : horizon <= 30 ? 0.015 : 0.02;

  // Recommendation derived from expectedReturn (not zone argmax)
  let recommendation: 'BUY' | 'HOLD' | 'SELL';
  if (expectedReturn > actionBuyThr) {
    recommendation = 'BUY';
  } else if (expectedReturn < -actionSellThr) {
    recommendation = 'SELL';
  } else {
    recommendation = 'HOLD';
  }

  // Confidence from conviction strength relative to threshold
  const activeThr = recommendation === 'BUY' ? actionBuyThr : actionSellThr;
  const conviction = Math.abs(expectedReturn);
  const confidence: 'HIGH' | 'MEDIUM' | 'LOW' =
    conviction >= 2 * activeThr ? 'HIGH' : conviction >= activeThr ? 'MEDIUM' : 'LOW';

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
    actionLevels: computeActionLevels(distribution, currentPrice),
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
  const quality: 'good' | 'sparse' | 'none' =
    trusted.length >= 3 && maxGapPct < 15 ? 'good'
    : trusted.length >= 1 ? 'sparse'
    : 'none';

  const warning = quality === 'good' ? ''
    : quality === 'sparse'
      ? `Sparse anchors (${trusted.length} trusted, max gap ${maxGapPct.toFixed(0)}%) — interpolation between anchors is model-driven`
      : 'No trusted anchors';

  return { totalAnchors: anchors.length, trustedAnchors: trusted.length, maxGapPct, quality, warning };
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
}): Promise<MarkovDistributionResult> {
  const { ticker, horizon, currentPrice, historicalPrices, polymarketMarkets, sentiment } = params;

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
  const { returnThreshold, volThreshold } = computeAdaptiveThresholds(returns);
  const regimeSeq: RegimeState[] = returns.map((r, i) =>
    classifyRegimeState(r, vols[i], returnThreshold, volThreshold),
  );
  const currentRegime = regimeSeq.length > 0 ? regimeSeq[regimeSeq.length - 1] : 'sideways';

  // --- Tier 1a: State observation counts and sparse state detection ---
  const stateObservationCounts = countStateObservations(regimeSeq);
  const sparseStates = findSparseStates(stateObservationCounts);

  // --- Tier 1b: Structural break detection ---
  const breakResult = regimeSeq.length >= 20
    ? detectStructuralBreak(regimeSeq)
    : { detected: false, divergence: 0, firstHalfMatrix: buildDefaultMatrix(), secondHalfMatrix: buildDefaultMatrix() };

  // --- Estimate transition matrix (fall back to default when break detected) ---
  let P = breakResult.detected
    ? buildDefaultMatrix()
    : estimateTransitionMatrix(regimeSeq);

  // --- Goodness-of-fit test (before sentiment adjustment, which is intentional) ---
  const gofResult = breakResult.detected ? null : transitionGoodnessOfFit(regimeSeq, P);

  // --- Sentiment adjustment ---
  const sentimentSignal = sentiment ?? { bullish: 0.5, bearish: 0.5 };
  const sentimentShift = sentimentSignal.bullish - sentimentSignal.bearish;
  P = adjustTransitionMatrix(P, sentimentSignal);

  // --- Regime statistics ---
  const logReturns = returns.map(r => Math.log(1 + r));
  const regimeStats = estimateRegimeStats(logReturns, regimeSeq);

  // --- Tier 1c: Polymarket anchors with optional cross-platform validation ---
  let polymarketAnchors = extractPriceThresholds(polymarketMarkets);
  let anchorDivergenceWarnings: AnchorDivergenceWarning[] = [];

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

  // --- Momentum signal ---
  const momentum = computeMomentumSignal(historicalPrices);
  // Halve momentum weight when structural break detected (trend may have broken)
  const momentumAdj = breakResult.detected ? momentum.adjustment * 0.5 : momentum.adjustment;

  // --- HMM forecast (Idea B: Hidden Markov Model) ---
  // Fit a Gaussian HMM on daily returns when we have enough data.
  // Blend its drift/vol forecast with the observable Markov estimate.
  let hmmOverride: { drift: number; vol: number; weight: number } | undefined;
  let hmmMeta: { converged: boolean; iterations: number; states: number; logLikelihood: number } | undefined;
  const HMM_MIN_OBS = 60; // need at least 60 returns for stable HMM
  if (returns.length >= HMM_MIN_OBS) {
    try {
      const hmmResult = baumWelch(returns, 3, 50, 1e-3);
      const hmmForecast = hmmPredict(returns, hmmResult.params, horizon);
      hmmMeta = {
        converged: hmmResult.converged,
        iterations: hmmResult.iterations,
        states: hmmResult.params.nStates,
        logLikelihood: hmmResult.logLikelihood,
      };
      if (hmmResult.converged && Number.isFinite(hmmForecast.expectedReturn)) {
        // Weight HMM at 0.5 when converged; reduce to 0.25 for short series
        const hmmWeight = returns.length >= 120 ? 0.5 : 0.25;
        hmmOverride = {
          drift: hmmForecast.expectedReturn,
          vol: hmmForecast.expectedVolatility,
          weight: hmmWeight,
        };
      }
    } catch {
      // HMM fitting can fail on degenerate data — fall back to observable Markov
    }
  }

  // --- Distribution ---
  const distribution = interpolateDistribution(
    currentPrice, horizon, P, regimeStats, currentRegime, polymarketAnchors, rho,
    20, 1000, ciWidthMultiplier, momentumAdj, hmmOverride,
  );

  // --- Optional R²_OS (leave-one-out on training tail) ---
  let r2os: number | null = null;
  const minHeldOut = 20;
  if (regimeSeq.length >= minHeldOut + 30) {
    const trainStates  = regimeSeq.slice(0, -minHeldOut);
    const trainReturns = logReturns.slice(0, -minHeldOut);
    const testReturns  = logReturns.slice(-minHeldOut);

    const trainP    = estimateTransitionMatrix(trainStates);
    const trainMean = trainReturns.reduce((s, v) => s + v, 0) / trainReturns.length;
    const predicted = testReturns.map((_, i) => {
      const Pn = matPow(trainP, i + 1);
      const stateIdx = STATE_INDEX[trainStates[trainStates.length - 1]];
      const weights  = Pn[stateIdx];
      return REGIME_STATES.reduce(
        (s, state, j) => s + weights[j] * regimeStats[state].meanReturn, 0,
      );
    });
    const baseline = Array(minHeldOut).fill(trainMean);
    const r2model    = computeR2OS(testReturns, predicted);
    const r2base     = computeR2OS(testReturns, baseline);
    r2os = r2model - r2base; // incremental R²_OS over mean baseline
  }

  return {
    ticker,
    currentPrice,
    horizon,
    distribution,
    actionSignal: computeActionSignal(distribution, currentPrice, undefined, undefined, horizon),
    metadata: {
      polymarketAnchors: polymarketAnchors.filter(a => a.trustScore === 'high').length,
      regimeState: currentRegime,
      sentimentAdjustment: sentimentShift,
      historicalDays: returns.length,
      mixingTimeWeight: mixWeight,
      secondEigenvalue: rho,
      outOfSampleR2: r2os,
      stateObservationCounts,
      sparseStates,
      structuralBreakDetected: breakResult.detected,
      structuralBreakDivergence: breakResult.divergence,
      ciWidened: breakResult.detected,
      anchorDivergenceWarnings,
      anchorCoverage: assessAnchorCoverage(polymarketAnchors, currentPrice),
      goodnessOfFit: gofResult,
      hmm: hmmMeta ?? null,
    },
  };
}

// ---------------------------------------------------------------------------
// 9. LangChain tool wrapper
// ---------------------------------------------------------------------------

export const MARKOV_DISTRIBUTION_DESCRIPTION = `
**markov_distribution** — Full probability distribution for a stock/ETF price at a specified horizon.

**Use when:**
- The user asks for a probability distribution of future prices (not a point estimate)
- The query includes a specific price target and horizon (e.g. "Will NVDA hit $1000 in 30 days?")
- You already have ≥2 Polymarket price threshold markets available as anchors
- You want to interpolate between Polymarket anchors using regime-aware Markov transitions

**What it does:**
- Combines Polymarket real-money anchors + historical regime transitions + sentiment
- Returns P(price > X) for 20+ price levels with 90% Monte Carlo confidence intervals
- Flags structural breaks (regime change mid-window), sparse states, and cross-platform divergence
- Automatically applies YES-bias correction (×${YES_BIAS_MULTIPLIER}) to raw Polymarket probabilities

**Do NOT use when:**
- The query is a simple binary probability (use probability_assessment skill instead)
- You have fewer than 10 historical prices (not enough for regime estimation)
- The horizon is > 90 trading days (model accuracy degrades at long horizons)
`.trim();

export const markovDistributionTool = new DynamicStructuredTool({
  name: 'markov_distribution',
  description: `
Generate a full probability distribution for a stock/ETF price at a specified horizon.
Combines Polymarket threshold markets (real-money anchors) with historical Markov regime
transitions to produce P(price > X) for each price level in the distribution.

Use when the query asks for a probability distribution of future prices, not just a point estimate.
Requires: ticker symbol, horizon in trading days (1–90), and access to recent price history.
`.trim(),
  schema: z.object({
    ticker: z.string().describe('Stock/ETF ticker symbol, e.g. NVDA, SPY, BTC-USD'),
    horizon: z.number().int().min(1).max(90).describe('Forecast horizon in trading days'),
    currentPrice: z.number().optional().describe('Current price (fetched automatically if omitted)'),
    historicalPrices: z.array(z.number()).min(10).describe(
      'Daily close prices oldest-first, minimum 30 recommended for regime estimation',
    ),
    polymarketMarkets: z.array(z.object({
      question:    z.string(),
      probability: z.number().min(0).max(1),
      volume:      z.number().optional(),
      createdAt:   z.union([z.string(), z.number()]).optional(),
    })).describe('Polymarket markets with dollar price thresholds (can be empty array)'),
    sentiment: z.object({
      bullish: z.number().min(0).max(1),
      bearish: z.number().min(0).max(1),
    }).optional().describe('Sentiment signal from social_sentiment tool (optional)'),
    kalshiAnchors: z.array(z.object({
      price: z.number(),
      probability: z.number().min(0).max(1),
      volume: z.number().optional(),
    })).optional().describe('Kalshi prediction market anchors for cross-platform validation (optional)'),
  }),
  func: async (input) => {
    const price = input.currentPrice
      ?? input.historicalPrices[input.historicalPrices.length - 1];

    const result = await computeMarkovDistribution({
      ticker:            input.ticker,
      horizon:           input.horizon,
      currentPrice:      price,
      historicalPrices:  input.historicalPrices,
      polymarketMarkets: input.polymarketMarkets,
      sentiment:         input.sentiment,
      kalshiAnchors:     input.kalshiAnchors,
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

    // --- Section 3: Header and metadata ---
    const header = [
      '',
      `📊 Markov Distribution: ${result.ticker} | Horizon: ${result.horizon}d`,
      `Current: $${fmt(result.currentPrice)} | Regime: ${m.regimeState}`,
      `Anchors: ${m.polymarketAnchors} trusted | Anchor quality: ${m.anchorCoverage.quality.toUpperCase()}`,
      `Mixing: ${pct(m.mixingTimeWeight)}% Markov / ${pct(1 - m.mixingTimeWeight)}% Anchors`,
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

    return [
      ...decisionCard,
      ...actionPlan,
      ...whatToDo,
      ...header,
      ...(warnings.length > 0 ? ['', ...warnings] : []),
      ...table,
    ].filter(l => l !== undefined).join('\n');
  },
});
