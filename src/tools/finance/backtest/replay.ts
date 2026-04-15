import { type WalkForwardConfig, type WalkForwardResult, walkForward } from './walk-forward.js';
import { computeMarkovDistribution, getAssetProfile, type MarkovDistributionResult, type BreakFallbackCandidate } from '../markov-distribution.js';
import type { BacktestStep, DecisionSource, ProbabilitySource } from './metrics.js';

export interface ReplayMarket {
  question: string;
  probability: number;
  volume?: number;
  createdAt?: string | number;
  endDate?: string | null;
  active?: boolean;
  closed?: boolean;
  enableOrderBook?: boolean;
}

export interface ReplaySnapshot {
  date: string;
  markets: ReplayMarket[];
}

export interface ReplayQualityFilters {
  minVolume?: number;
  requirePersistence?: boolean;
  maxProbabilityShock?: number;
  requireHorizonAlignment?: boolean;
}

export function lookupReplaySnapshot(
  targetDate: string,
  snapshots?: ReplaySnapshot[]
): ReplaySnapshot | undefined {
  if (!snapshots || snapshots.length === 0) return undefined;

  let best: ReplaySnapshot | undefined;
  for (const snap of snapshots) {
    if (snap.date <= targetDate) {
      if (!best || snap.date > best.date) {
        best = snap;
      }
    }
  }
  return best;
}

export function filterReplayMarketsByTime(
  markets: ReplayMarket[],
  referenceTimeMs: number
): ReplayMarket[] {
  return markets.filter(m => {
    const createdOk = (() => {
      if (!m.createdAt) return true;
      const createdMs = typeof m.createdAt === 'string' ? Date.parse(m.createdAt) : m.createdAt;
      return Number.isFinite(createdMs) && createdMs <= referenceTimeMs;
    })();

    const unresolvedOk = (() => {
      if (!m.endDate) return true;
      const endMs = Date.parse(m.endDate);
      return Number.isFinite(endMs) && endMs > referenceTimeMs;
    })();

    const tradableOk = m.active !== false && m.closed !== true && m.enableOrderBook !== false;

    return createdOk && unresolvedOk && tradableOk;
  });
}

function isReplayMarketAlignedToHorizon(
  market: ReplayMarket,
  referenceTimeMs: number,
  horizonDays: number
): boolean {
  if (!market.endDate) return true;
  const endMs = Date.parse(market.endDate);
  if (!Number.isFinite(endMs)) return true;
  const daysUntilResolution = (endMs - referenceTimeMs) / 86_400_000;
  return Math.abs(daysUntilResolution - horizonDays) <= Math.max(2, horizonDays * 0.5);
}

function findPreviousReplayObservation(
  question: string,
  snapshotDate: string,
  snapshots: ReplaySnapshot[],
): ReplayMarket | undefined {
  for (let i = snapshots.length - 1; i >= 0; i--) {
    const snapshot = snapshots[i];
    if (snapshot.date >= snapshotDate) continue;
    const match = snapshot.markets.find(m => m.question === question);
    if (match) return match;
  }
  return undefined;
}

export function filterReplayMarketsByQuality(
  markets: ReplayMarket[],
  snapshotDate: string,
  snapshots: ReplaySnapshot[],
  referenceTimeMs: number,
  horizonDays: number,
  filters?: ReplayQualityFilters,
): ReplayMarket[] {
  if (!filters) return markets;

  const {
    minVolume = 0,
    requirePersistence = false,
    maxProbabilityShock = 1,
    requireHorizonAlignment = false,
  } = filters;

  return markets.filter(market => {
    if ((market.volume ?? 0) < minVolume) return false;
    if (requireHorizonAlignment && !isReplayMarketAlignedToHorizon(market, referenceTimeMs, horizonDays)) {
      return false;
    }
    if (!requirePersistence) return true;

    const previous = findPreviousReplayObservation(market.question, snapshotDate, snapshots);
    if (!previous) return false;

    return Math.abs(previous.probability - market.probability) <= maxProbabilityShock;
  });
}

export interface WalkForwardWithReplayConfig extends WalkForwardConfig {
  dates: string[];
  replaySnapshots?: ReplaySnapshot[];
  replayQualityFilters?: ReplayQualityFilters;
}

export async function walkForwardWithReplay(
  config: WalkForwardWithReplayConfig
): Promise<WalkForwardResult> {
  if (!config.replaySnapshots || config.replaySnapshots.length === 0) {
    return walkForward(config);
  }

  const {
    ticker,
    prices,
    dates,
    horizon,
    warmup = 120,
    stride = 5,
    replaySnapshots
  } = config;

  const steps: BacktestStep[] = [];
  const errors: Array<{ t: number; error: string }> = [];
  const maxT = prices.length - horizon - 1;

  for (let t = warmup; t <= maxT; t += stride) {
    const histPrices = prices.slice(0, t + 1);
    const currentPrice = prices[t];
    const realizedPrice = prices[t + horizon];
    const actualReturn = (realizedPrice - currentPrice) / currentPrice;
    
      const stepDate = dates[t];
      const referenceTimeMs = Date.parse(`${stepDate}T23:59:59Z`);

      const snapshot = lookupReplaySnapshot(stepDate, replaySnapshots);
      const snapshotDate = snapshot?.date ?? stepDate;
      const rawMarkets = snapshot?.markets ?? [];
      const timeFilteredMarkets = filterReplayMarketsByTime(rawMarkets, referenceTimeMs);
      const polymarketMarkets = filterReplayMarketsByQuality(
        timeFilteredMarkets,
        snapshotDate,
        replaySnapshots,
        referenceTimeMs,
        horizon,
       config.replayQualityFilters,
     );

    try {
      const result: MarkovDistributionResult = await computeMarkovDistribution({
        ticker,
        horizon,
        currentPrice,
        historicalPrices: histPrices,
        polymarketMarkets,
        referenceTimeMs,
        cryptoShortHorizonConditionalWeight: config.cryptoShortHorizonConditionalWeight,
        cryptoShortHorizonRawDecisionAblation: config.cryptoShortHorizonRawDecisionAblation,
        pr3fCryptoShortHorizonDisagreementPrior: config.pr3fCryptoShortHorizonDisagreementPrior,
        cryptoShortHorizonKappaMultiplier: config.cryptoShortHorizonKappaMultiplier,
        cryptoShortHorizonPUpFloor: config.cryptoShortHorizonPUpFloor,
        cryptoShortHorizonBearMarginMultiplier: config.cryptoShortHorizonBearMarginMultiplier,
        pr3gCryptoShortHorizonRecencyWeighting: config.pr3gCryptoShortHorizonRecencyWeighting,
        pr3gCryptoShortHorizonDecay: config.pr3gCryptoShortHorizonDecay,
        trendPenaltyOnlyBreakConfidence: config.trendPenaltyOnlyBreakConfidence,
        breakFallbackCandidate: config.breakFallbackCandidate,
        divergenceWeightedBreakConfidence: config.divergenceWeightedBreakConfidence,
        divergencePenaltySchedule: config.divergencePenaltySchedule,
        btcReturnThresholdMultiplier: config.btcReturnThresholdMultiplier,
        btcBreakDivergenceThreshold: config.btcBreakDivergenceThreshold,
      });

      const predictedProb = interpolateSurvival(result.distribution, currentPrice);
      const rawPredictedProb = interpolateSurvival(result.rawDistribution, currentPrice);
      const { ciLower, ciUpper } = extractCI(result.distribution, currentPrice);

      const replayAnchorActive = result.metadata.anchorCoverage.trustedAnchors > 0;
      let decisionSource: DecisionSource = replayAnchorActive ? 'replay-anchor' : 'default';
      if (result.metadata.pr3fDisagreementBlendActive) {
        decisionSource = replayAnchorActive
          ? 'crypto-short-horizon-disagreement-blend+replay-anchor'
          : 'crypto-short-horizon-disagreement-blend';
      } else if (result.metadata.pr3gRecencyWeightingActive) {
        decisionSource = replayAnchorActive
          ? 'crypto-short-horizon-recency+replay-anchor'
          : 'crypto-short-horizon-recency';
      } else if (
        config.cryptoShortHorizonRawDecisionAblation === true &&
        horizon <= 14 &&
        getAssetProfile(ticker).type === 'crypto'
      ) {
        decisionSource = replayAnchorActive
          ? 'crypto-short-horizon-raw+replay-anchor'
          : 'crypto-short-horizon-raw';
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
        structuralBreakDivergence: result.metadata.structuralBreakDivergence,
        hmmConverged: result.metadata.hmm?.converged ?? null,
        ensembleConsensus: result.metadata.ensemble.consensus,
        probabilitySource: 'calibrated' as ProbabilitySource,
        decisionSource,
        trendPenaltyOnlyBreakConfidenceActive: result.metadata.trendPenaltyOnlyBreakConfidenceActive,
        divergenceWeightedBreakConfidenceActive: result.metadata.divergenceWeightedBreakConfidenceActive,
        breakFallbackCandidateId: result.metadata.breakFallbackCandidateId,
        breakFallbackMode: result.metadata.breakFallbackMode,
      });
    } catch (err) {
      errors.push({ t, error: String(err) });
    }
  }

  return { ticker, horizon, steps, errors };
}

function interpolateSurvival(dist: MarkovDistributionResult['distribution'], target: number): number {
  if (dist.length === 0) return 0.5;
  if (target <= dist[0].price) return dist[0].probability;
  if (target >= dist[dist.length - 1].price) return dist[dist.length - 1].probability;
  for (let i = 0; i < dist.length - 1; i++) {
    if (dist[i].price <= target && target <= dist[i + 1].price) {
      const frac = (target - dist[i].price) / (dist[i + 1].price - dist[i].price);
      return dist[i].probability + frac * (dist[i + 1].probability - dist[i].probability);
    }
  }
  return 0.5;
}

function extractCI(dist: MarkovDistributionResult['distribution'], currentPrice: number): { ciLower: number; ciUpper: number } {
  if (dist.length === 0) return { ciLower: currentPrice * 0.9, ciUpper: currentPrice * 1.1 };
  let ciLower = dist[0].price;
  let ciUpper = dist[dist.length - 1].price;
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
