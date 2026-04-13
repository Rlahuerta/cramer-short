/**
 * Walk-forward backtest engine for the Markov distribution model.
 *
 * Slides a window through historical price data, generates predictions at
 * each step, then compares against realized outcomes.
 */

import {
  computeMarkovDistribution,
  getAssetProfile,
  type MarkovDistributionResult,
} from '../markov-distribution.js';
import type { BacktestStep, DecisionSource, ProbabilitySource } from './metrics.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface WalkForwardConfig {
  sidewaysSplit?: boolean;
  /** Ticker symbol for display purposes */
  ticker: string;
  /** Daily close prices, oldest first */
  prices: number[];
  /** Forecast horizon in trading days */
  horizon: number;
  /** Minimum number of historical days before first prediction (default: 120) */
  warmup?: number;
  /** Step forward by this many days between predictions (default: 5) */
  stride?: number;
  /** Optional override for crypto short-horizon conditional weight (PR3B ablation) */
  cryptoShortHorizonConditionalWeight?: number;
  /** Optional override for crypto short-horizon kappa multiplier (PR3B ablation) */
  cryptoShortHorizonKappaMultiplier?: number;
  /** Optional flag: use raw P(up) for decision generation (PR3 lever ablation) */
  cryptoShortHorizonRawDecisionAblation?: boolean;
  /** Optional flag: use PR3F crypto short-horizon disagreement prior */
  pr3fCryptoShortHorizonDisagreementPrior?: boolean;
  /** Optional override for crypto short-horizon pUp floor clamp (PR3 Stage 2 ablation) */
  cryptoShortHorizonPUpFloor?: number;
  /** Optional override for crypto short-horizon bear margin multiplier (PR3 Stage 2 ablation) */
  cryptoShortHorizonBearMarginMultiplier?: number;
  /** Optional flag: use recency-weighted regime up-rates for crypto short-horizon (PR3G state-model improvement) */
  pr3gCryptoShortHorizonRecencyWeighting?: boolean;
  /** Optional explicit decay parameter for PR3G recency-weighted regime up-rates */
  pr3gCryptoShortHorizonDecay?: number;
  /** Optional flag: reduce overconfidence in BTC 14d mature bull cases */
  matureBullCalibration?: boolean;
  /** Optional flag: replace hard start state with additive mixture over last K states */
  startStateMixture?: boolean;
  /** Optional override for transition-matrix decay weighting (Phase 1 ablation) */
  transitionDecayOverride?: number;
  /** Optional flag: when a full-window run detects a break, rerun with a shorter window */
  postBreakShortWindow?: boolean;
  /** Optional shorter window size used by postBreakShortWindow (default: 60) */
  postBreakWindowSize?: number;
  /** Optional flag: keep break confidence penalty only in trending break contexts (Phase 4 ablation) */
  trendPenaltyOnlyBreakConfidence?: boolean;
}

export interface WalkForwardResult {
  ticker: string;
  horizon: number;
  steps: BacktestStep[];
  errors: Array<{ t: number; error: string }>;
}

// ---------------------------------------------------------------------------
// Walk-forward engine
// ---------------------------------------------------------------------------

/**
 * Run a walk-forward backtest.
 *
 * At each step t (from warmup to end-horizon, stepping by stride):
 * 1. Use prices[0..t] to generate a Markov distribution
 * 2. Record predicted probabilities, CI, recommendation
 * 3. Compare against actual outcome at prices[t + horizon]
 */
export async function walkForward(config: WalkForwardConfig): Promise<WalkForwardResult> {
  const {
    ticker,
    prices,
    horizon,
    warmup = 120,
    stride = 5,
  } = config;

  const steps: BacktestStep[] = [];
  const errors: Array<{ t: number; error: string }> = [];
  const maxT = prices.length - horizon - 1;

  for (let t = warmup; t <= maxT; t += stride) {
    const histPrices = prices.slice(0, t + 1);
    const currentPrice = prices[t];
    const realizedPrice = prices[t + horizon];
    const actualReturn = (realizedPrice - currentPrice) / currentPrice;

    try {
      let result: MarkovDistributionResult = await computeMarkovDistribution({
        ticker,
        horizon,
        currentPrice,
        historicalPrices: histPrices,
        polymarketMarkets: [],  // pure Markov, no anchors
        cryptoShortHorizonConditionalWeight: config.cryptoShortHorizonConditionalWeight,
        cryptoShortHorizonRawDecisionAblation: config.cryptoShortHorizonRawDecisionAblation,
        pr3fCryptoShortHorizonDisagreementPrior: config.pr3fCryptoShortHorizonDisagreementPrior,
        cryptoShortHorizonKappaMultiplier: config.cryptoShortHorizonKappaMultiplier,
        cryptoShortHorizonPUpFloor: config.cryptoShortHorizonPUpFloor,
        cryptoShortHorizonBearMarginMultiplier: config.cryptoShortHorizonBearMarginMultiplier,
        pr3gCryptoShortHorizonRecencyWeighting: config.pr3gCryptoShortHorizonRecencyWeighting,
        pr3gCryptoShortHorizonDecay: config.pr3gCryptoShortHorizonDecay,
        matureBullCalibration: config.matureBullCalibration,
        startStateMixture: config.startStateMixture,
        sidewaysSplit: config.sidewaysSplit,
        transitionDecayOverride: config.transitionDecayOverride,
        trendPenaltyOnlyBreakConfidence: config.trendPenaltyOnlyBreakConfidence,
      });

      const originalStructuralBreakDetected = result.metadata.structuralBreakDetected;
      const originalStructuralBreakDivergence = result.metadata.structuralBreakDivergence;
      let structuralBreakRerunTriggered = false;

      if (config.postBreakShortWindow && result.metadata.structuralBreakDetected) {
        const shortWindowSize = Math.max(30, config.postBreakWindowSize ?? 60);
        if (histPrices.length > shortWindowSize) {
          structuralBreakRerunTriggered = true;
          result = await computeMarkovDistribution({
            ticker,
            horizon,
            currentPrice,
            historicalPrices: histPrices.slice(-shortWindowSize),
            polymarketMarkets: [],
            cryptoShortHorizonConditionalWeight: config.cryptoShortHorizonConditionalWeight,
            cryptoShortHorizonRawDecisionAblation: config.cryptoShortHorizonRawDecisionAblation,
            pr3fCryptoShortHorizonDisagreementPrior: config.pr3fCryptoShortHorizonDisagreementPrior,
            cryptoShortHorizonKappaMultiplier: config.cryptoShortHorizonKappaMultiplier,
            cryptoShortHorizonPUpFloor: config.cryptoShortHorizonPUpFloor,
            cryptoShortHorizonBearMarginMultiplier: config.cryptoShortHorizonBearMarginMultiplier,
            pr3gCryptoShortHorizonRecencyWeighting: config.pr3gCryptoShortHorizonRecencyWeighting,
            pr3gCryptoShortHorizonDecay: config.pr3gCryptoShortHorizonDecay,
            matureBullCalibration: config.matureBullCalibration,
            startStateMixture: config.startStateMixture,
            sidewaysSplit: config.sidewaysSplit,
            transitionDecayOverride: config.transitionDecayOverride,
            trendPenaltyOnlyBreakConfidence: config.trendPenaltyOnlyBreakConfidence,
          });
        }
      }

      // Find P(>currentPrice) from the calibrated distribution by interpolation
      const predictedProb = interpolateSurvival(result.distribution, currentPrice);

      // Find raw (pre-calibration) P(>currentPrice) for PR3A diagnostic tracking.
      // The raw distribution has wider spread and better sign discriminability.
      const rawPredictedProb = interpolateSurvival(result.rawDistribution, currentPrice);

      const { ciLower, ciUpper } = extractCI(result.distribution, currentPrice);

      let decisionSource: DecisionSource = 'default';
      if (result.metadata.pr3fDisagreementBlendActive) {
        decisionSource = 'crypto-short-horizon-disagreement-blend';
      } else if (result.metadata.pr3gRecencyWeightingActive) {
        decisionSource = 'crypto-short-horizon-recency';
      } else if (
        config.cryptoShortHorizonRawDecisionAblation === true &&
        horizon <= 14 &&
        getAssetProfile(ticker).type === 'crypto'
      ) {
        decisionSource = 'crypto-short-horizon-raw';
      }

      steps.push({
        t,
        predictedProb,
        rawPredictedProb,
        actualBinary: realizedPrice > currentPrice ? 1 : 0,
        predictedReturn: result.actionSignal.expectedReturn,
        actualReturn,
        ciLower,
        ciUpper,
        realizedPrice,
        recommendation: result.actionSignal.recommendation,
        gofPasses: result.metadata.goodnessOfFit?.passes ?? null,
        confidence: result.predictionConfidence,
        regime: result.metadata.regimeState,
        anchorQuality: result.metadata.anchorCoverage.quality,
        trustedAnchors: result.metadata.anchorCoverage.trustedAnchors,
        markovWeight: result.metadata.mixingTimeWeight,
        anchorWeight: 1 - result.metadata.mixingTimeWeight,
        validationMetric: result.metadata.validationMetric,
        outOfSampleR2: result.metadata.outOfSampleR2,
        structuralBreakDetected: result.metadata.structuralBreakDetected,
        structuralBreakRerunTriggered,
        originalStructuralBreakDetected,
        structuralBreakDivergence: result.metadata.structuralBreakDivergence,
        originalStructuralBreakDivergence,
        hmmConverged: result.metadata.hmm?.converged ?? null,
        ensembleConsensus: result.metadata.ensemble.consensus,
        probabilitySource: 'calibrated' as ProbabilitySource,
        decisionSource,
        sidewaysSplitActive: result.metadata.sidewaysSplitActive,
        matureBullCalibrationActive: result.metadata.matureBullCalibrationActive,
        trendPenaltyOnlyBreakConfidenceActive: result.metadata.trendPenaltyOnlyBreakConfidenceActive,
      });
    } catch (err) {
      errors.push({ t, error: String(err) });
    }
  }

  return { ticker, horizon, steps, errors };
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Interpolate P(>target) from the distribution curve.
 * The distribution is sorted by price ascending with probability descending.
 */
function interpolateSurvival(
  dist: MarkovDistributionResult['distribution'],
  target: number,
): number {
  if (dist.length === 0) return 0.5;

  // If target is below the lowest price point, P(>target) ≈ 1
  if (target <= dist[0].price) return dist[0].probability;
  // If target is above the highest price point, P(>target) ≈ 0
  if (target >= dist[dist.length - 1].price) return dist[dist.length - 1].probability;

  // Linear interpolation between bracketing points
  for (let i = 0; i < dist.length - 1; i++) {
    if (dist[i].price <= target && target <= dist[i + 1].price) {
      const frac = (target - dist[i].price) / (dist[i + 1].price - dist[i].price);
      return dist[i].probability + frac * (dist[i + 1].probability - dist[i].probability);
    }
  }

  return 0.5; // fallback
}

/**
 * Extract a conservative price interval from the survival distribution.
 *
 * This is used for backtest coverage diagnostics only. The thresholds are tuned
 * empirically for stable coverage and are intentionally more conservative than a
 * literal central 90% interval.
 */
function extractCI(
  dist: MarkovDistributionResult['distribution'],
  currentPrice: number,
): { ciLower: number; ciUpper: number } {
  if (dist.length === 0) return { ciLower: currentPrice * 0.9, ciUpper: currentPrice * 1.1 };

  let ciLower = dist[0].price;
  let ciUpper = dist[dist.length - 1].price;

  // Use conservative survival thresholds — tuned empirically for coverage rather
  // than interpreted as literal 5th/95th percentiles.
  const lowerThreshold = 0.995;
  const upperThreshold = 0.005;

  for (let i = 0; i < dist.length - 1; i++) {
    if (dist[i].probability >= lowerThreshold && dist[i + 1].probability < lowerThreshold) {
      const frac = (lowerThreshold - dist[i + 1].probability) / (dist[i].probability - dist[i + 1].probability);
      ciLower = dist[i + 1].price + frac * (dist[i].price - dist[i + 1].price);
    }
    if (dist[i].probability >= upperThreshold && dist[i + 1].probability < upperThreshold) {
      const frac = (upperThreshold - dist[i + 1].probability) / (dist[i].probability - dist[i + 1].probability);
      ciUpper = dist[i + 1].price + frac * (dist[i].price - dist[i + 1].price);
    }
  }

  return { ciLower, ciUpper };
}
