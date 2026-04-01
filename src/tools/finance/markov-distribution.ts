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
    'GOLD', 'SILVER', 'COPPER',       // common names for precious/base metals
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
    // Cumulative return over next `horizon` days
    let cumReturn = 0;
    for (let j = i; j < i + horizon; j++) {
      cumReturn += returns[j];
    }
    counts[regime].total++;
    if (cumReturn > 0) counts[regime].up++;
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
  let w: number[] = Array.from({ length: n }, (_, i) => (i === 0 ? 0.6 : 0.1));
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
): { mu_n: number; sigma_n: number } {
  const Pn = matPow(P, horizon);
  const stateWeights = Pn[STATE_INDEX[initialState]];

  const mu_obs = REGIME_STATES.reduce(
    (s, state, i) => s + stateWeights[i] * regimeStats[state].meanReturn, 0,
  );
  const sigma_obs = Math.sqrt(
    REGIME_STATES.reduce(
      (s, state, i) => s + stateWeights[i] * regimeStats[state].stdReturn ** 2, 0,
    ),
  );

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
): TrajectoryPoint[] {
  const initialIdx = STATE_INDEX[initialState];
  const trajectory: TrajectoryPoint[] = [];

  // Pre-compute regime weights at each day via matrix powers
  const regimeWeightsPerDay: number[][] = [];
  for (let d = 1; d <= days; d++) {
    const Pd = matPow(P, d);
    regimeWeightsPerDay.push(Pd[initialIdx]);
  }

  // Compute 1-day regime-weighted drift and vol for MC steps
  const { mu_n: drift1d, sigma_n: regimeVol1d } = computeHorizonDriftVol(
    1, P, regimeStats, initialState, momentumAdjustment, hmmOverride,
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
      d, P, regimeStats, initialState, momentumAdjustment, hmmOverride,
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
    horizon, P, regimeStats, initialState, momentumAdjustment, hmmOverride,
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
      const p = studentTSurvival(currentPrice, price, perturbedMu, perturbedVol);
      ciSamples.get(price)!.push(p);
    }
  }

  // Build distribution points
  const rawPoints = prices.map(price => {
    const anchor = findAnchor(price);
    const markovEst = studentTSurvival(currentPrice, price, mu_n, sigma_n);

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
    // Drift-based calibration params (preserves distribution S-shape)
    currentPrice?: number;        // required for drift-based mode
    driftN?: number;              // n-step drift (mu_n) from computeHorizonDriftVol
    volN?: number;                // n-step volatility (sigma_n)
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
  if (regime === 'bull' || regime === 'bear') {
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
    const nu = 5; // Student-t degrees of freedom (must match studentTSurvival)
    const rawPUp = studentTSurvival(cp, cp, driftN, volN);
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
        newProb = studentTSurvival(cp, point.price, calibratedDrift, volN);
      } else {
        // Blended/polymarket point: apply additive delta to preserve anchor contribution
        const oldMarkov = studentTSurvival(cp, point.price, driftN, volN);
        const newMarkov = studentTSurvival(cp, point.price, calibratedDrift, volN);
        newProb = Math.max(0, Math.min(1, point.probability + (newMarkov - oldMarkov)));
      }
      return { ...point, probability: newProb };
      // lowerBound and upperBound preserved from Monte Carlo — no compression
    });

    // Re-enforce monotonicity
    for (let i = calibrated.length - 2; i >= 0; i--) {
      if (calibrated[i].probability < calibrated[i + 1].probability) {
        calibrated[i].probability = calibrated[i + 1].probability;
      }
    }
    return calibrated;
  }

  // --- Legacy fallback: probability-level calibration ---
  // Used when drift params are not available (e.g., unit tests with synthetic data)
  const calibrated = distribution.map(point => ({
    ...point,
    probability: kappa * center + (1 - kappa) * point.probability,
  }));

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
 * Combines four orthogonal signals:
 *   1. **Decisiveness** (40%): |P(up) − 0.5| × 2 — how far from a coin flip.
 *      When P(up)≈0.5, the model has no directional edge; ≈1.0 = very decisive.
 *   2. **Ensemble consensus** (25%): fraction of signals (momentum, mean-reversion,
 *      crossover) that agree. consensus=3 → all agree → max contribution.
 *   3. **HMM convergence** (15%): Baum-Welch converged = +0.15 confidence.
 *   4. **Regime stability** (20%): consecutive days in the same regime / 20.
 *      Longer streaks → more predictable environment.
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
}): number {
  const { pUp, ensembleConsensus, hmmConverged, regimeRunLength, structuralBreak } = options;

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

  // Penalty for structural break (regime change mid-window → unreliable)
  if (structuralBreak) confidence *= 0.6;

  // Asset-type discount: crypto is inherently noisier → scale confidence down.
  // Commodities are driven by supply shocks → moderate discount.
  // ETFs are the most predictable → small boost.
  const assetType = options.assetType;
  if (assetType === 'crypto') {
    confidence *= 0.7;
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
  /** Return day-by-day price trajectory instead of single-horizon snapshot */
  trajectory?: boolean;
  /** Number of days for trajectory (default: horizon, max 30) */
  trajectoryDays?: number;
}): Promise<MarkovDistributionResult> {
  const { ticker, horizon, currentPrice, historicalPrices, polymarketMarkets, sentiment } = params;

  // --- Asset profile (Idea N): per-asset-class parameter tuning ---
  const assetProfile = getAssetProfile(ticker);

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

  // --- Regime statistics (winsorized + drift-capped per asset profile) ---
  const logReturns = returns.map(r => Math.log(1 + r));
  const regimeStats = estimateRegimeStats(logReturns, regimeSeq, assetProfile.maxDailyDrift);

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

  // --- Distribution ---
  // Compute daily volatility for adaptive grid sizing
  const gridDailyVol = returns.length >= 20
    ? Math.sqrt(returns.slice(-20).reduce((s, v) => s + v * v, 0) / 20)
    : undefined;
  const rawDistribution = interpolateDistribution(
    currentPrice, horizon, P, regimeStats, currentRegime, polymarketAnchors, rho,
    20, 1000, ciWidthMultiplier, combinedDriftAdj, hmmOverride, gridDailyVol,
  );

  // Compute drift/vol for drift-based calibration (same values used by interpolateDistribution)
  const { mu_n: calDriftN, sigma_n: calVolN } = computeHorizonDriftVol(
    horizon, P, regimeStats, currentRegime, combinedDriftAdj, hmmOverride,
  );

  // --- Bayesian calibration (Idea I): shrink extreme probabilities toward base rate ---
  // Compute empirical base rate: P(up) from recent returns (Idea L: adaptive center)
  const recentReturns = returns.slice(-Math.min(120, returns.length));
  const baseRate = recentReturns.length > 0
    ? recentReturns.filter(r => r > 0).length / recentReturns.length
    : 0.5;

  // --- Regime-conditional P(up) (Idea T, Round 4) ---
  const regimeUpRates = computeRegimeUpRates(regimeSeq, returns, horizon);
  const Pn = matPow(P, horizon);
  const initialIdx = STATE_INDEX[currentRegime];
  const stateWeightsForUp = Pn[initialIdx];
  const conditionalPUp = REGIME_STATES.reduce(
    (s, state, i) => s + stateWeightsForUp[i] * regimeUpRates[state], 0,
  );
  const conditionalWeight = assetProfile.type === 'etf' ? 0.80
    : assetProfile.type === 'commodity' ? 0.60
    : assetProfile.type === 'equity' ? 0.65
    : 0.40; // crypto
  const calibrationCenter = conditionalWeight * conditionalPUp + (1 - conditionalWeight) * baseRate;

  // Drift-based calibration: shifts the distribution center (P(up)) toward
  // calibrationCenter while preserving the survival curve's S-shape.
  // Unlike the old probability-level shrinkage which compressed ALL probabilities
  // toward center (destroying tails), this only adjusts the drift parameter.
  const distribution = calibrateProbabilities(rawDistribution, {
    ensembleConsensus: ensemble.consensus,
    historicalDays: returns.length,
    hmmConverged: hmmMeta?.converged ?? false,
    baseRate: calibrationCenter,
    kappaMultiplier: assetProfile.kappaMultiplier,
    currentRegime,
    currentPrice,
    driftN: calDriftN,
    volN: calVolN,
  });

  // --- Base-rate floor (Idea S, Round 4) ---
  // Adaptive bear margin: scales with how bullish the base rate is.
  // In strongly bullish markets, require very strong evidence to predict down.
  // Near 50%, the floor is tighter because even small over-prediction of bear hurts.
  const calPUpPreFloor = interpolateSurvival(distribution, currentPrice);
  // bearMargin scales linearly: from 0.05 (at baseRate=0.5) to 0.10 (at baseRate=0.75+)
  // This ensures the model defaults to UP for assets near 50-55% base rate
  const bearMargin = Math.max(0.05, Math.min(0.10, 0.05 + (baseRate - 0.5) * 0.2));
  const pUpFloor = Math.max(0.35, calibrationCenter - bearMargin);
  if (calPUpPreFloor < pUpFloor) {
    const deficit = pUpFloor - calPUpPreFloor;
    for (const pt of distribution) {
      pt.probability = Math.min(1.0, pt.probability + deficit);
    }
  }

  // --- Regime run length: consecutive days ending in the current regime ---
  let regimeRunLength = 1;
  for (let i = regimeSeq.length - 2; i >= 0; i--) {
    if (regimeSeq[i] === currentRegime) regimeRunLength++;
    else break;
  }

  // --- P(up) from both raw and calibrated distributions for confidence scoring ---
  // Raw P(up) reflects model's actual signal strength BEFORE calibration compresses it.
  // This gives a much more discriminative decisiveness score.
  const rawPUp = interpolateSurvival(rawDistribution, currentPrice);
  const calPUp = interpolateSurvival(distribution, currentPrice);

  // Recent daily volatility (20-day std of daily returns)
  const recentDailyVol = returns.length >= 20
    ? Math.sqrt(returns.slice(-20).reduce((s, v) => s + v * v, 0) / 20)
    : undefined;

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
  });

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

  // --- Trajectory computation (optional day-by-day forecast) ---
  let trajectoryResult: TrajectoryPoint[] | undefined;
  if (params.trajectory) {
    const trajDays = Math.min(30, Math.max(1, params.trajectoryDays ?? horizon));
    trajectoryResult = computeTrajectory(
      currentPrice, trajDays, P, regimeStats, currentRegime,
      combinedDriftAdj, hmmOverride, 1000, assetProfile.studentTNu,
      recentDailyVol,
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

  // --- Scenario probabilities (derived from calibrated CDF) ---
  const scenarios = computeScenarioProbabilities(distribution, currentPrice);

  return {
    ticker,
    currentPrice,
    horizon,
    distribution,
    rawDistribution,
    actionSignal: computeActionSignal(distribution, currentPrice, undefined, undefined, horizon,
      recentDailyVol,
    ),
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
      stateObservationCounts,
      sparseStates,
      structuralBreakDetected: breakResult.detected,
      structuralBreakDivergence: breakResult.divergence,
      ciWidened: breakResult.detected,
      anchorDivergenceWarnings,
      anchorCoverage: assessAnchorCoverage(polymarketAnchors, currentPrice),
      goodnessOfFit: gofResult,
      hmm: hmmMeta ?? null,
      ensemble: { consensus: ensemble.consensus, adjustment: ensembleAdj },
    },
  };
}

// ---------------------------------------------------------------------------
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
- Gold/GOLD/XAUUSD → use **GLD** (ETF that tracks gold spot price)
- Silver/SILVER/XAGUSD → use **SLV**
- Oil/Crude/WTICOUSD → use **USO**
- Natural Gas → use **UNG**
GLD, SLV, USO etc. are ETFs that replicate the underlying commodity price.

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
Generate a full probability distribution for a stock/ETF/commodity price at a specified horizon.
Combines Polymarket threshold markets (real-money anchors) with historical Markov regime
transitions to produce P(price > X) for each price level in the distribution.

Use when the query asks for a probability distribution of future prices, not just a point estimate.
Requires: ticker symbol, horizon in trading days (1–90), and access to recent price history.

IMPORTANT — Commodity tickers: For commodities, use the liquid ETF ticker for best data availability:
  Gold → GLD, Silver → SLV, Oil → USO, Natural Gas → UNG, Copper → CPER.
  GLD is an ETF that replicates the price of gold — querying "GLD" and "gold" should yield the same prediction.

Set trajectory=true for a day-by-day price forecast with expected price, 90% CI, and P(up) at each day.
Use trajectoryDays to control the number of days (1–30, default=horizon).
`.trim(),
  schema: z.object({
    ticker: z.string().describe('Stock/ETF/commodity ticker symbol, e.g. NVDA, SPY, BTC-USD, GLD. For commodities, prefer the liquid ETF (GLD for gold, SLV for silver, USO for oil) over futures tickers.'),
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
    trajectory: z.boolean().optional().default(false)
      .describe('Return day-by-day price trajectory instead of single-horizon snapshot'),
    trajectoryDays: z.number().int().min(1).max(30).optional()
      .describe('Number of days for trajectory (default: horizon, max 30)'),
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

    // --- Section 4: Header and metadata ---
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

    return [
      ...decisionCard,
      ...actionPlan,
      ...whatToDo,
      ...scenarioSection,
      ...header,
      ...(warnings.length > 0 ? ['', ...warnings] : []),
      ...table,
      ...trajectorySection,
    ].filter(l => l !== undefined).join('\n');
  },
});
