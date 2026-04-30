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
  type BreakFallbackCandidate,
  type DivergencePenaltySchedule,
  type PredictionConfidenceMode,
} from '../markov-distribution.js';
import {
  AdaptiveConformalPID,
  type AdaptiveConformalRecordDiagnostics,
} from '../conformal.js';
import type { RegimePlattFits } from '../regime-calibrator.js';
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
  /** Optional flag: use raw direction for recommendation while keeping calibrated probabilities/CI (Phase B ablation) */
  rawDirectionHybrid?: boolean;
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
  /** Phase 5 experimental: hybrid structural-break fallback candidate (backtest-only) */
  breakFallbackCandidate?: BreakFallbackCandidate;
  /** Phase 6 experimental: use divergence-weighted break confidence penalties (backtest-only) */
  divergenceWeightedBreakConfidence?: boolean;
  /** Phase 6 experimental: penalty schedule for divergence-weighted mode. Defaults to DEFAULT_DIVERGENCE_PENALTY_SCHEDULE. */
  divergencePenaltySchedule?: DivergencePenaltySchedule;
  /** Phase 7 experimental: use dominant regime's own sigma instead of mixture sigma when regime weights are concentrated (backtest-only) */
  regimeSpecificSigma?: boolean;
  /** Phase 7 experimental: minimum max(stateWeight) to activate regime-specific sigma. Defaults to 0.60. */
  regimeSpecificSigmaThreshold?: number;
  /** Phase D experimental: BTC-only override for regime classification return threshold multiplier */
  btcReturnThresholdMultiplier?: number;
  /** Phase C experimental: BTC-only override for structural break divergence threshold */
  btcBreakDivergenceThreshold?: number;
  /** W3R2 experimental: enable ADWIN-based historical-price trim */
  enableAdwinTrim?: boolean;
  /** W3R2 experimental: ADWIN delta (false-positive rate). Default 0.05 in MD. */
  adwinDelta?: number;
  /** W3R2 experimental: enable Hawkes-based jump intensity amplification */
  enableHawkesIntensity?: boolean;
  /** W3R2 experimental: sigma threshold for Hawkes jump detection. Default 3.0. */
  hawkesSigmaThreshold?: number;
  /** R4 Idea 3: pre-fitted regime-conditional Platt overlay. Default undefined ⇒ no overlay. */
  regimePlattFits?: RegimePlattFits;
  /** R4 Idea 4: enable GARCH(1,1) per-day vol scalars in trajectory MC. Default false ⇒ byte-identical. */
  enableGarchVol?: boolean;
  /** R5 Idea #5 — soft-blend GARCH scalars toward 1.0 past this horizon (in days). */
  garchHorizonCap?: number;
  /** R5 Idea #5 — regime-conditional ceiling for the GARCH scalar. */
  garchRegimeCeiling?: { calm: number; turbulent: number };
  /** R5 Idea #14 — enable transition-entropy CI width modulation. */
  enableEntropyCiModulation?: boolean;
  /** R5 Idea #14 — rolling history size for entropy z-score state. Default 60. */
  entropyWindowSize?: number;
  /** R5 Idea #14 — CI-scale sensitivity to entropy z-score. Default 0.15. */
  entropyKappa?: number;
  /** R5 Idea #11 — enable longshot/favorite shrinkage in RND anchor integration. */
  enableLongshotShrinkage?: boolean;
  /** R6 — enable adaptive conformal calibration in walk-forward CI evaluation. */
  enableAdaptiveConformal?: boolean;
  /** R6 — target conformal miscoverage α. Default 0.1. */
  conformalAlpha?: number;
  /** R6 — break-mode sensitivity / learning-rate multiplier. */
  conformalBreakSensitivity?: number;
  /** R6 — base adaptive conformal learning rate. */
  conformalFastLearningRate?: number;
  /** R6 — number of steps to keep break mode active after a trigger. */
  conformalCooloffWindow?: number;
  /** Item 1 — confidence-score ablation mode. */
  predictionConfidenceMode?: PredictionConfidenceMode;
  /** R4 Idea 1: enable KSWIN variance-aware drift trim. Default false ⇒ byte-identical. */
  enableKswinTrim?: boolean;
  /** KSWIN significance level. Default 0.005. */
  kswinAlpha?: number;
  /** R4 Idea 2: enable cross-asset Lasso bias on trajectory drift. Default false. */
  enableCrossAssetBias?: boolean;
  /** Peer-ticker → daily-return series, time-aligned with the target. */
  crossAssetReturns?: Record<string, number[]>;
  /** Lasso L1 strength. Default 0.005. */
  crossAssetLassoLambda?: number;
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
  const entropyHistory: number[] = [];
  const entropyWindowSize = Math.max(5, config.entropyWindowSize ?? 60);
  let adaptiveConformal: AdaptiveConformalPID | undefined;

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
        rawDirectionHybrid: config.rawDirectionHybrid,
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
        breakFallbackCandidate: config.breakFallbackCandidate,
        divergenceWeightedBreakConfidence: config.divergenceWeightedBreakConfidence,
        divergencePenaltySchedule: config.divergencePenaltySchedule,
        regimeSpecificSigma: config.regimeSpecificSigma,
        regimeSpecificSigmaThreshold: config.regimeSpecificSigmaThreshold,
        btcReturnThresholdMultiplier: config.btcReturnThresholdMultiplier,
        btcBreakDivergenceThreshold: config.btcBreakDivergenceThreshold,
        enableAdwinTrim: config.enableAdwinTrim,
        adwinDelta: config.adwinDelta,
        enableHawkesIntensity: config.enableHawkesIntensity,
        hawkesSigmaThreshold: config.hawkesSigmaThreshold,
        regimePlattFits: config.regimePlattFits,
        enableGarchVol: config.enableGarchVol,
        garchHorizonCap: config.garchHorizonCap,
        garchRegimeCeiling: config.garchRegimeCeiling,
        enableEntropyCiModulation: config.enableEntropyCiModulation,
        transitionEntropyHistory: entropyHistory,
        entropyKappa: config.entropyKappa,
        enableLongshotShrinkage: config.enableLongshotShrinkage,
        enableAdaptiveConformal: config.enableAdaptiveConformal,
        conformalAlpha: config.conformalAlpha,
        conformalBreakSensitivity: config.conformalBreakSensitivity,
        conformalFastLearningRate: config.conformalFastLearningRate,
        conformalCooloffWindow: config.conformalCooloffWindow,
        predictionConfidenceMode: config.predictionConfidenceMode,
        enableKswinTrim: config.enableKswinTrim,
        kswinAlpha: config.kswinAlpha,
        enableCrossAssetBias: config.enableCrossAssetBias,
        crossAssetReturns: config.crossAssetReturns,
        crossAssetLassoLambda: config.crossAssetLassoLambda,
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
            rawDirectionHybrid: config.rawDirectionHybrid,
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
            breakFallbackCandidate: config.breakFallbackCandidate,
            divergenceWeightedBreakConfidence: config.divergenceWeightedBreakConfidence,
            divergencePenaltySchedule: config.divergencePenaltySchedule,
            regimeSpecificSigma: config.regimeSpecificSigma,
            regimeSpecificSigmaThreshold: config.regimeSpecificSigmaThreshold,
            btcReturnThresholdMultiplier: config.btcReturnThresholdMultiplier,
            btcBreakDivergenceThreshold: config.btcBreakDivergenceThreshold,
            enableAdwinTrim: config.enableAdwinTrim,
            adwinDelta: config.adwinDelta,
            enableHawkesIntensity: config.enableHawkesIntensity,
            hawkesSigmaThreshold: config.hawkesSigmaThreshold,
            regimePlattFits: config.regimePlattFits,
            enableGarchVol: config.enableGarchVol,
            garchHorizonCap: config.garchHorizonCap,
            garchRegimeCeiling: config.garchRegimeCeiling,
             enableEntropyCiModulation: config.enableEntropyCiModulation,
             transitionEntropyHistory: entropyHistory,
             entropyKappa: config.entropyKappa,
             enableLongshotShrinkage: config.enableLongshotShrinkage,
            enableAdaptiveConformal: config.enableAdaptiveConformal,
            conformalAlpha: config.conformalAlpha,
            conformalBreakSensitivity: config.conformalBreakSensitivity,
            conformalFastLearningRate: config.conformalFastLearningRate,
            conformalCooloffWindow: config.conformalCooloffWindow,
            predictionConfidenceMode: config.predictionConfidenceMode,
            enableKswinTrim: config.enableKswinTrim,
            kswinAlpha: config.kswinAlpha,
            enableCrossAssetBias: config.enableCrossAssetBias,
            crossAssetReturns: config.crossAssetReturns,
            crossAssetLassoLambda: config.crossAssetLassoLambda,
          });
        }
      }

      // Find P(>currentPrice) from the calibrated distribution by interpolation
      const predictedProb = interpolateSurvival(result.distribution, currentPrice);

      // Find raw (pre-calibration) P(>currentPrice) for PR3A diagnostic tracking.
      // The raw distribution has wider spread and better sign discriminability.
      const rawPredictedProb = interpolateSurvival(result.rawDistribution, currentPrice);

      let { ciLower, ciUpper } = extractCI(result.distribution, currentPrice);
      let conformalApplied: boolean | undefined;
      let conformalRadius: number | undefined;
      let conformalCoverageEstimate: number | undefined;
      let conformalMode: 'normal' | 'break' | undefined;

      if (config.enableAdaptiveConformal === true) {
        const forecastCenter = currentPrice * (1 + result.actionSignal.expectedReturn);
        const recentRealizedVol = computeRecentRealizedVol(histPrices);
        if (!adaptiveConformal) {
          adaptiveConformal = new AdaptiveConformalPID({
            enabled: true,
            alpha: config.conformalAlpha,
            initialRadius: currentPrice * 0.005,
            learningRate: config.conformalFastLearningRate,
            breakLearningRateMultiplier: config.conformalBreakSensitivity,
            cooloffWindow: config.conformalCooloffWindow,
          });
        }

        const adaptiveDiagnostics: AdaptiveConformalRecordDiagnostics = {
          structuralBreak: result.metadata.structuralBreakDetected && (recentRealizedVol ?? 0) >= 0.02,
          realizedVol: recentRealizedVol,
        };
        const adaptiveInterval = adaptiveConformal.wrap(forecastCenter, adaptiveDiagnostics);
        const adaptiveMode = adaptiveConformal.currentMode();
        const adaptiveRadius = adaptiveInterval.high - forecastCenter;
        if (adaptiveMode === 'break') {
          const midpoint = (ciLower + ciUpper) / 2;
          const halfWidth = (ciUpper - ciLower) / 2;
          const widenedHalfWidth = Math.max(
            halfWidth * (1 + ((config.conformalBreakSensitivity ?? 1.5) * 0.35)),
            adaptiveRadius,
          );
          ciLower = Math.min(ciLower, adaptiveInterval.low, midpoint - widenedHalfWidth);
          ciUpper = Math.max(ciUpper, adaptiveInterval.high, midpoint + widenedHalfWidth);
        }
        adaptiveConformal.record(forecastCenter, realizedPrice, adaptiveDiagnostics);
        conformalApplied = true;
        conformalRadius = adaptiveRadius;
        conformalCoverageEstimate = adaptiveConformal.empiricalCoverage();
        conformalMode = adaptiveMode;
      }

      let decisionSource: DecisionSource = 'default';
      if (result.metadata.pr3fDisagreementBlendActive) {
        decisionSource = 'crypto-short-horizon-disagreement-blend';
      } else if (result.metadata.rawDirectionHybridActive) {
        decisionSource = 'crypto-short-horizon-raw-direction-hybrid';
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
        divergenceWeightedBreakConfidenceActive: result.metadata.divergenceWeightedBreakConfidenceActive,
        bearishBreakRecommendationGateActive: result.metadata.bearishBreakRecommendationGateActive,
        breakFallbackCandidateId: result.metadata.breakFallbackCandidateId,
        breakFallbackMode: result.metadata.breakFallbackMode,
        regimeSpecificSigmaActive: result.metadata.regimeSpecificSigmaActive,
        garchVolApplied: result.metadata.garchVolApplied,
        transitionEntropy: result.metadata.transitionEntropy,
        transitionEntropyNorm: result.metadata.transitionEntropyNorm,
        transitionEntropyZ: result.metadata.transitionEntropyZ,
        entropyCiScale: result.metadata.entropyCiScale,
        entropyCiModulationApplied: result.metadata.entropyCiModulationApplied,
        conformalApplied,
        conformalRadius,
        conformalCoverageEstimate,
        conformalMode,
      });
      const entropyNorm = result.metadata.transitionEntropyNorm;
      if (typeof entropyNorm === 'number' && Number.isFinite(entropyNorm)) {
        entropyHistory.push(entropyNorm);
        if (entropyHistory.length > entropyWindowSize) entropyHistory.shift();
      }
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

function computeRecentRealizedVol(prices: readonly number[], lookback = 20): number | undefined {
  if (prices.length < 3) return undefined;
  const start = Math.max(1, prices.length - lookback);
  const returns: number[] = [];
  for (let i = start; i < prices.length; i++) {
    const prev = prices[i - 1];
    const next = prices[i];
    if (!(prev > 0) || !(next > 0)) continue;
    returns.push(Math.log(next / prev));
  }
  if (returns.length < 2) return undefined;
  const mean = returns.reduce((sum, value) => sum + value, 0) / returns.length;
  const variance = returns.reduce((sum, value) => sum + ((value - mean) ** 2), 0) / returns.length;
  return Math.sqrt(Math.max(variance, 0));
}
