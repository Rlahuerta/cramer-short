/**
 * Polymarket Forecast Tool
 *
 * Prediction-market-weighted ensemble price forecast for any asset.
 * Combines Polymarket probability signals with optional news sentiment,
 * fundamental analyst targets, and options skew into a single forecast.
 *
 * Research basis:
 *   Reichenbach & Walther (2025) · Cordoba et al. (2024) · Tsang & Yang (2026)
 */
import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { polymarketBreaker } from '../../utils/circuit-breaker.js';
import {
  fetchPolymarketAnchorMarketsWithQueries,
  fetchPolymarketMarkets,
  type PolymarketMarketResult,
} from './polymarket.js';
import { findSnapshotInWindow, readSnapshotRecords, DEFAULT_POLYMARKET_SNAPSHOTS_PATH } from './polymarket-snapshots.js';
import { extractSignals, scoreMarketRelevance } from './signal-extractor.js';
import { classifyPolymarketQuestion } from './forecast-arbitrator.js';
import { resolveTickerSearchIdentity } from './asset-resolver.js';
import { lookupImpact, inferAssetClass } from './impact-map.js';
import {
  runEnsemble,
  computeCI,
  computePolymarketSignal,
  computeEnsemble,
  computeConditionalReturn,
  adjustYesBias,
  scoreToGrade,
  isCleanThresholdLadder,
  computeThresholdImpliedRawForecast,
  YES_BIAS_MULTIPLIER,
  type MarketInput,
  type ThresholdLadderPoint,
} from '../../utils/ensemble.js';
import {
  buildPriceDistributionChart,
  extractPriceThresholds as extractChartPriceThresholds,
} from './price-distribution-chart.js';
import type { PolymarketSnapshotRecord } from './polymarket-snapshots.js';
import {
  applyCryptoTerminalAnchorFallback,
  buildPolymarketAnchorQueryVariants,
  evaluateAnchorTrust,
  extractPriceThresholds as extractAnchorPriceThresholds,
  type PriceThreshold,
} from './markov-distribution.js';
import {
  appendReplayCachePolymarketCapture,
  createRawPolymarketReplayRow,
  freezePolymarketReplayBlock,
  readArbiterReplayBundles,
  type ArbiterReplayBundle,
  type RawPolymarketReplayRow,
} from './arbiter-replay.js';
import {
  BrierReplayCalibrator,
  predictWithBrierReplayState,
  type BrierReplayCalibratorState,
} from './brier-replay-calibrator.js';
import {
  computeCrossPlatformDelta,
  fetchMetaforecastQuestions,
  findBestMetaforecastMatch,
  shouldFlagCrossPlatform,
  type MetaforecastEstimate,
} from './metaforecast.js';
import {
  fetchKalshiVolSignals,
  type KalshiVolSignal,
} from './kalshi-vol-signals.js';

type RawMarket = {
  marketId?: string;
  assetId?: string;
  question: string;
  probability: number;
  volume24h: number;
  ageDays: number | undefined;
  signalCategory: string;
  endDate?: string | null;
  active?: boolean;
  closed?: boolean;
  enableOrderBook?: boolean;
  priceSpikeDetected: boolean;
  transitoryMove: boolean;
  bidAskSpread?: number;
  priceVelocityPpH?: number;
  priceVelocityLogitPerHour?: number;
  maxHourlyJump?: number;
  maxHourlyLogitJump?: number;
};

type ForecastHistoryReader = typeof readSnapshotRecords;
type ForecastMarketFetcher = typeof fetchPolymarketMarkets;
type ForecastAnchorMarketFetcher = typeof fetchPolymarketAnchorMarketsWithQueries;
type ForecastMetaforecastFetcher = typeof fetchMetaforecastQuestions;
type ForecastKalshiSignalFetcher = typeof fetchKalshiVolSignals;

type ForecastToolDependencies = {
  fetchMarkets?: ForecastMarketFetcher;
  fetchAnchorMarketsWithQueries?: ForecastAnchorMarketFetcher;
  fetchMetaforecastQuestions?: ForecastMetaforecastFetcher;
  fetchKalshiVolSignals?: ForecastKalshiSignalFetcher;
  readRecords?: ForecastHistoryReader;
  readReplayBundles?: typeof readArbiterReplayBundles;
  recordReplayPolymarketCapture?: (capture: {
    rawRow: RawPolymarketReplayRow;
    polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
  }) => void;
};

export interface CrossPlatformEvidence {
  source: 'metaforecast' | 'kalshi';
  kind: 'consensus' | 'macro_event';
  label: string;
  probability: number;
  diagnostic: string;
  flagged: boolean;
  observedAt?: string;
  deltaFromPolymarket?: number;
  intensityBoost?: number;
  url?: string;
}

export interface CrossPlatformConfidenceAdjustment {
  basis: 'none' | 'metaforecast_agreement' | 'metaforecast_divergence';
  applied: boolean;
  qualityScoreDelta: number;
  sigmaMultiplier: number;
  summary?: string;
  warnings: string[];
}

type HistoryFlags = {
  priceSpikeDetected: boolean;
  transitoryMove: boolean;
  warnings: string[];
};

type HistoryHeuristicContext = {
  assetClass?: string;
  horizonDays?: number;
};

type HistoryHeuristicProfile = {
  spikeWindowStartMs: number;
  spikeWindowEndMs: number;
  persistenceWindowStartMs: number;
  persistenceWindowEndMs: number;
  spikeWindowLabel: string;
  persistenceWindowLabel: string;
  baseSpikeThreshold: number;
  transitoryMoveThreshold: number;
  transitoryReversalRatio: number;
  dynamicSpikeThreshold: boolean;
};

type LiveBrierReplayCalibrationDecision =
  | {
      mode: 'disabled' | 'static';
      state?: BrierReplayCalibratorState;
      warnings: string[];
      evidenceRows: 0;
      evidenceBundles: 0;
    }
  | {
      mode: 'blocked';
      warnings: string[];
      evidenceRows: number;
      evidenceBundles: number;
    }
  | {
      mode: 'horizon_aware';
      state: BrierReplayCalibratorState;
      warnings: string[];
      evidenceRows: number;
      evidenceBundles: number;
    };

const DAY_MS = 86_400_000;
const HOUR_MS = 3_600_000;
const TWO_HOURS_MS = 2 * 3_600_000;
const THREE_HOURS_MS = 3 * 3_600_000;
const FOUR_HOURS_MS = 4 * 3_600_000;
const TWELVE_HOURS_MS = 12 * 3_600_000;
const TWENTY_FOUR_HOURS_MS = 24 * 3_600_000;
const THIRTY_SIX_HOURS_MS = 36 * 3_600_000;
const FORTY_EIGHT_HOURS_MS = 48 * 3_600_000;
const LIVE_BRIER_REPLAY_FLAG = 'POLYMARKET_BRIER_REPLAY_CALIBRATOR_ENABLED';
const CROSS_PLATFORM_FUSION_FLAG = 'POLYMARKET_CROSS_PLATFORM_FUSION_ENABLED';
const LIVE_BRIER_REPLAY_STATE: BrierReplayCalibratorState = {
  bias: 0.016276026205423927,
  slope: 1 / 3,
};
const LIVE_BRIER_REPLAY_MIN_PROBABILITY = 0.4;
const LIVE_BRIER_REPLAY_MAX_PROBABILITY = 0.6;
const SHORT_HORIZON_REPLAY_MIN_LABELED_BUNDLES = 3;
const SHORT_HORIZON_REPLAY_MIN_MID_CONFIDENCE_ROWS = 6;
const CROSS_PLATFORM_DIVERGENCE_QUALITY_PENALTY = 8;
const CROSS_PLATFORM_DIVERGENCE_SIGMA_MULTIPLIER = 1.08;

function isLiveBrierReplayCalibratorEnabled(): boolean {
  return /^(1|true|yes|on)$/i.test(process.env[LIVE_BRIER_REPLAY_FLAG] ?? '');
}

function isCrossPlatformFusionEnabled(): boolean {
  return /^(1|true|yes|on)$/i.test(process.env[CROSS_PLATFORM_FUSION_FLAG] ?? '');
}

function isUsablePolymarketProbability(probability: number): boolean {
  return Number.isFinite(probability) && probability >= 0 && probability <= 1;
}

function toRawMarket(
  market: PolymarketMarketResult,
  signalCategory: string,
  history: HistoryFlags,
): RawMarket | null {
  if (!isUsablePolymarketProbability(market.probability)) {
    return null;
  }

  return {
    marketId: market.marketId,
    assetId: market.assetId,
    question: market.question,
    probability: market.probability,
    volume24h: market.volume24h,
    ageDays: market.ageDays,
    endDate: market.endDate,
    signalCategory,
    active: market.active,
    closed: market.closed,
    enableOrderBook: market.enableOrderBook,
    priceSpikeDetected: history.priceSpikeDetected,
    transitoryMove: history.transitoryMove,
    bidAskSpread: market.bidAskSpread,
    priceVelocityPpH: market.priceVelocityPpH,
    priceVelocityLogitPerHour: market.priceVelocityLogitPerHour,
    maxHourlyJump: market.maxHourlyJump,
    maxHourlyLogitJump: market.maxHourlyLogitJump,
  };
}

function isLiveCalibrationApplicable(probability: number): boolean {
  return isUsablePolymarketProbability(probability)
    && probability >= LIVE_BRIER_REPLAY_MIN_PROBABILITY
    && probability <= LIVE_BRIER_REPLAY_MAX_PROBABILITY;
}

function isShortHorizonCryptoHistoryContext(context: HistoryHeuristicContext | undefined): boolean {
  return context?.assetClass === 'crypto'
    && context.horizonDays !== undefined
    && context.horizonDays <= 3;
}

function historyHeuristicProfile(context: HistoryHeuristicContext | undefined): HistoryHeuristicProfile {
  if (isShortHorizonCryptoHistoryContext(context)) {
    return {
      spikeWindowStartMs: THREE_HOURS_MS,
      spikeWindowEndMs: HOUR_MS,
      persistenceWindowStartMs: THIRTY_SIX_HOURS_MS,
      persistenceWindowEndMs: TWELVE_HOURS_MS,
      spikeWindowLabel: '1-3h',
      persistenceWindowLabel: '12-36h',
      baseSpikeThreshold: 0.05,
      transitoryMoveThreshold: 0.08,
      transitoryReversalRatio: 0.45,
      dynamicSpikeThreshold: true,
    };
  }

  return {
    spikeWindowStartMs: FOUR_HOURS_MS,
    spikeWindowEndMs: TWO_HOURS_MS,
    persistenceWindowStartMs: FORTY_EIGHT_HOURS_MS,
    persistenceWindowEndMs: TWENTY_FOUR_HOURS_MS,
    spikeWindowLabel: '2-4h',
    persistenceWindowLabel: '24-48h',
    baseSpikeThreshold: 0.08,
    transitoryMoveThreshold: 0.10,
    transitoryReversalRatio: 0.5,
    dynamicSpikeThreshold: false,
  };
}

function clamp(value: number, min: number, max: number): number {
  return Math.max(min, Math.min(max, value));
}

function median(values: number[]): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const middle = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 0
    ? (sorted[middle - 1]! + sorted[middle]!) / 2
    : sorted[middle]!;
}

function recentProbabilityMoves(
  marketId: string,
  records: PolymarketSnapshotRecord[],
  nowMs: number,
): number[] {
  const recent = records
    .filter((record) => record.marketId === marketId)
    .map((record) => ({
      ...record,
      capturedAtMs: Date.parse(record.capturedAt),
    }))
    .filter((record) =>
      Number.isFinite(record.capturedAtMs)
      && record.capturedAtMs >= nowMs - TWELVE_HOURS_MS
      && record.capturedAtMs <= nowMs,
    )
    .sort((a, b) => a.capturedAtMs - b.capturedAtMs);

  const moves: number[] = [];
  for (let index = 1; index < recent.length; index++) {
    moves.push(Math.abs(recent[index]!.probability - recent[index - 1]!.probability));
  }
  return moves;
}

function shortHorizonCryptoSpikeThreshold(
  market: Pick<PolymarketMarketResult, 'marketId' | 'probability' | 'volume24h'>,
  spikeSnapshot: PolymarketSnapshotRecord,
  records: PolymarketSnapshotRecord[],
  nowMs: number,
  baseThreshold: number,
): number {
  if (!market.marketId) return baseThreshold;
  const recentVolatility = median(recentProbabilityMoves(market.marketId, records, nowMs));
  const volatilityComponent = recentVolatility > 0 ? recentVolatility * 2.25 : 0;
  const volumeRatio = market.volume24h / Math.max(spikeSnapshot.volume24h, 1);
  const volumeComponent = volumeRatio > 1
    ? Math.min(0.03, Math.log10(volumeRatio) * 0.04)
    : 0;

  return clamp(
    Math.max(baseThreshold, volatilityComponent + 0.03) + volumeComponent,
    baseThreshold,
    0.12,
  );
}

function shouldUseShortHorizonReplayCalibration(
  ticker: string,
  assetClass: string,
  horizonDays: number,
): boolean {
  return ticker === 'BTC' && assetClass === 'crypto' && horizonDays >= 1 && horizonDays <= 3;
}

export function shouldAttemptCrossPlatformEvidence(
  assetClass: string,
  horizonDays: number,
): boolean {
  return assetClass === 'crypto' && horizonDays >= 1 && horizonDays <= 30;
}

export function normalizeMetaforecastEvidence(params: {
  marketQuestion: string;
  marketProbability: number;
  match: MetaforecastEstimate;
}): CrossPlatformEvidence {
  const delta = computeCrossPlatformDelta(params.marketProbability, params.match.probability);
  const deltaPp = delta * 100;
  const flagged = shouldFlagCrossPlatform(delta);
  return {
    source: 'metaforecast',
    kind: 'consensus',
    label: params.match.title,
    probability: params.match.probability,
    diagnostic: `${params.match.platform} consensus ${deltaPp.toFixed(1)}pp from Polymarket on "${params.marketQuestion}"${flagged ? ' (material disagreement)' : ''}`,
    flagged,
    deltaFromPolymarket: delta,
    ...(params.match.url ? { url: params.match.url } : {}),
  };
}

export function normalizeKalshiEvidence(signal: KalshiVolSignal): CrossPlatformEvidence {
  return {
    source: 'kalshi',
    kind: 'macro_event',
    label: signal.sourceTitle,
    probability: signal.probability,
    diagnostic: `${signal.eventType.toUpperCase()} event ${signal.eventId} implies ${(signal.probability * 100).toFixed(1)}% with vol boost ${signal.intensityBoost.toFixed(2)}`,
    flagged: false,
    observedAt: signal.eventAt,
    intensityBoost: signal.intensityBoost,
  };
}

export function deriveCrossPlatformConfidenceAdjustment(
  evidence: CrossPlatformEvidence[],
): CrossPlatformConfidenceAdjustment {
  const metaforecastConsensus = evidence.filter((entry) =>
    entry.source === 'metaforecast' && entry.kind === 'consensus',
  );
  const flaggedConsensus = metaforecastConsensus.filter((entry) =>
    shouldFlagCrossPlatform(entry.deltaFromPolymarket ?? 0),
  );

  if (flaggedConsensus.length > 0) {
    const worstDivergence = flaggedConsensus.reduce((worst, entry) =>
      (entry.deltaFromPolymarket ?? 0) > (worst.deltaFromPolymarket ?? 0) ? entry : worst,
    );
    const deltaPp = ((worstDivergence.deltaFromPolymarket ?? 0) * 100).toFixed(1);
    return {
      basis: 'metaforecast_divergence',
      applied: true,
      qualityScoreDelta: -CROSS_PLATFORM_DIVERGENCE_QUALITY_PENALTY,
      sigmaMultiplier: CROSS_PLATFORM_DIVERGENCE_SIGMA_MULTIPLIER,
      summary: 'Cross-platform divergence detected; confidence trimmed conservatively.',
      warnings: [
        `Cross-platform divergence warning: MetaForecast differs from the lead Polymarket market by ${deltaPp}pp; quality score reduced by ${CROSS_PLATFORM_DIVERGENCE_QUALITY_PENALTY} points and the 95% CI was widened modestly.`,
      ],
    };
  }

  if (metaforecastConsensus.length > 0) {
    return {
      basis: 'metaforecast_agreement',
      applied: false,
      qualityScoreDelta: 0,
      sigmaMultiplier: 1,
      summary: 'Cross-platform check: MetaForecast stayed within the 10pp divergence threshold; no confidence adjustment applied.',
      warnings: [],
    };
  }

  return {
    basis: 'none',
    applied: false,
    qualityScoreDelta: 0,
    sigmaMultiplier: 1,
    warnings: [],
  };
}

function toUtcDate(ms: number): string {
  return new Date(ms).toISOString().slice(0, 10);
}

function selectPrimaryCrossPlatformMarket(rawMarkets: RawMarket[]): RawMarket | null {
  if (rawMarkets.length === 0) return null;
  return rawMarkets.reduce((best, market) =>
    market.volume24h > best.volume24h ? market : best,
  );
}

async function collectCrossPlatformEvidence(params: {
  assetClass: string;
  horizonDays: number;
  nowMs: number;
  rawMarkets: RawMarket[];
  fetchMetaforecastQuestions: ForecastMetaforecastFetcher;
  fetchKalshiVolSignals?: ForecastKalshiSignalFetcher;
}): Promise<{ evidence: CrossPlatformEvidence[]; warnings: string[] }> {
  if (!shouldAttemptCrossPlatformEvidence(params.assetClass, params.horizonDays)) {
    return { evidence: [], warnings: [] };
  }

  const primaryMarket = selectPrimaryCrossPlatformMarket(params.rawMarkets);
  const metaforecastPromise = primaryMarket
    ? params.fetchMetaforecastQuestions(primaryMarket.question, { limit: 8 })
    : Promise.resolve([]);
  const kalshiPromise = params.fetchKalshiVolSignals
    ? params.fetchKalshiVolSignals({
      fromDate: toUtcDate(params.nowMs),
      toDate: toUtcDate(params.nowMs + params.horizonDays * DAY_MS),
    })
    : Promise.resolve([]);

  const [metaforecastResult, kalshiResult] = await Promise.allSettled([
    metaforecastPromise,
    kalshiPromise,
  ]);

  const evidence: CrossPlatformEvidence[] = [];
  const warnings: string[] = [];

  if (metaforecastResult.status === 'fulfilled' && primaryMarket) {
    const match = findBestMetaforecastMatch(primaryMarket.question, metaforecastResult.value);
    if (match) {
      evidence.push(normalizeMetaforecastEvidence({
        marketQuestion: primaryMarket.question,
        marketProbability: primaryMarket.probability,
        match,
      }));
    }
  } else if (metaforecastResult.status === 'rejected') {
    warnings.push('Cross-platform evidence unavailable: metaforecast lookup failed.');
  }

  if (kalshiResult.status === 'fulfilled') {
    evidence.push(...kalshiResult.value.map(normalizeKalshiEvidence));
  } else {
    warnings.push('Cross-platform evidence unavailable: Kalshi macro lookup failed.');
  }

  return { evidence, warnings };
}

function fitShortHorizonReplayCalibrationState(
  bundles: ArbiterReplayBundle[],
  ticker: string,
  horizonDays: number,
): LiveBrierReplayCalibrationDecision {
  const horizonBundles = bundles.filter((bundle) =>
    bundle.ticker.toUpperCase() === ticker
    && bundle.horizonDays === horizonDays,
  );

  if (horizonBundles.length === 0) {
    return {
      mode: 'blocked',
      warnings: [`Live replay calibration blocked: no ${horizonDays}d replay bundles were found in the benchmark source.`],
      evidenceRows: 0,
      evidenceBundles: 0,
    };
  }

  const labeledBundles = horizonBundles.filter((bundle) =>
    (bundle.labels?.semantic?.length ?? 0) > 0 && (bundle.polymarket?.selectedMarkets.length ?? 0) > 0,
  );
  if (labeledBundles.length === 0) {
    return {
      mode: 'blocked',
      warnings: [`Live replay calibration blocked: no labeled ${horizonDays}d replay bundles were found; unlabeled bundles are not accuracy proof.`],
      evidenceRows: 0,
      evidenceBundles: 0,
    };
  }

  const rows = labeledBundles.flatMap((bundle) => {
    const marketsById = new Map((bundle.polymarket?.selectedMarkets ?? []).map((market) => [market.marketId, market]));
    return (bundle.labels?.semantic ?? []).flatMap((label) => {
      const market = marketsById.get(label.marketId);
      if (!market || !isLiveCalibrationApplicable(market.probability)) return [];
      if (label.outcome !== 'yes' && label.outcome !== 'no') return [];
      return [{
        probability: market.probability,
        actualBinary: label.outcome === 'yes' ? 1 : 0,
      }];
    });
  });

  if (labeledBundles.length < SHORT_HORIZON_REPLAY_MIN_LABELED_BUNDLES) {
    return {
      mode: 'blocked',
      warnings: [
        `Live replay calibration blocked: only ${labeledBundles.length} labeled ${horizonDays}d replay bundle${labeledBundles.length === 1 ? '' : 's'} were found; need at least ${SHORT_HORIZON_REPLAY_MIN_LABELED_BUNDLES}.`,
      ],
      evidenceRows: rows.length,
      evidenceBundles: labeledBundles.length,
    };
  }

  if (rows.length < SHORT_HORIZON_REPLAY_MIN_MID_CONFIDENCE_ROWS) {
    return {
      mode: 'blocked',
      warnings: [
        `Live replay calibration blocked: only ${rows.length} labeled mid-confidence ${horizonDays}d BTC market outcome${rows.length === 1 ? '' : 's'} were found; need at least ${SHORT_HORIZON_REPLAY_MIN_MID_CONFIDENCE_ROWS}.`,
      ],
      evidenceRows: rows.length,
      evidenceBundles: labeledBundles.length,
    };
  }

  const distinctOutcomes = new Set(rows.map((row) => row.actualBinary));
  if (distinctOutcomes.size < 2) {
    return {
      mode: 'blocked',
      warnings: [`Live replay calibration blocked: labeled ${horizonDays}d BTC mid-confidence outcomes only contain one class, so the fit would be unstable.`],
      evidenceRows: rows.length,
      evidenceBundles: labeledBundles.length,
    };
  }

  const calibrator = new BrierReplayCalibrator({
    learningRate: 0.1,
    midConfidenceWeight: 4,
    maxSlope: 3,
  });
  rows.forEach((row) => {
    calibrator.record(row.probability, row.actualBinary);
  });

  return {
    mode: 'horizon_aware',
    state: calibrator.state(),
    warnings: [],
    evidenceRows: rows.length,
    evidenceBundles: labeledBundles.length,
  };
}

function resolveLiveBrierReplayCalibration(params: {
  ticker: string;
  assetClass: string;
  horizonDays: number;
  readReplayBundles: typeof readArbiterReplayBundles;
}): LiveBrierReplayCalibrationDecision {
  if (!isLiveBrierReplayCalibratorEnabled()) {
    return {
      mode: 'disabled',
      warnings: [],
      evidenceRows: 0,
      evidenceBundles: 0,
    };
  }

  if (!shouldUseShortHorizonReplayCalibration(params.ticker, params.assetClass, params.horizonDays)) {
    return {
      mode: 'static',
      state: LIVE_BRIER_REPLAY_STATE,
      warnings: [],
      evidenceRows: 0,
      evidenceBundles: 0,
    };
  }

  try {
    return fitShortHorizonReplayCalibrationState(
      params.readReplayBundles(),
      params.ticker,
      params.horizonDays,
    );
  } catch {
    return {
      mode: 'blocked',
      warnings: ['Live replay calibration blocked: replay bundles could not be read from disk.'],
      evidenceRows: 0,
      evidenceBundles: 0,
    };
  }
}

export function evaluateMarketHistory(
  market: Pick<PolymarketMarketResult, 'marketId' | 'probability' | 'volume24h'>,
  records: PolymarketSnapshotRecord[],
  nowMs: number,
  context?: HistoryHeuristicContext,
): {
  priceSpikeDetected: boolean;
  transitoryMove: boolean;
  warnings: string[];
} {
  if (!market.marketId) {
    return {
      priceSpikeDetected: false,
      transitoryMove: false,
      warnings: [],
    };
  }
  const warnings: string[] = [];
  const profile = historyHeuristicProfile(context);

  const spikeSnapshot = findSnapshotInWindow(
    records,
    market.marketId,
    nowMs - profile.spikeWindowStartMs,
    nowMs - profile.spikeWindowEndMs,
  );

  const persistenceSnapshot = findSnapshotInWindow(
    records,
    market.marketId,
    nowMs - profile.persistenceWindowStartMs,
    nowMs - profile.persistenceWindowEndMs,
  );

  let priceSpikeDetected = false;
  let transitoryMove = false;

  if (spikeSnapshot) {
    const absDelta = Math.abs(market.probability - spikeSnapshot.probability);
    const spikeThreshold = profile.dynamicSpikeThreshold
      ? shortHorizonCryptoSpikeThreshold(
        market,
        spikeSnapshot,
        records,
        nowMs,
        profile.baseSpikeThreshold,
      )
      : profile.baseSpikeThreshold;
    priceSpikeDetected = profile.dynamicSpikeThreshold
      ? absDelta > spikeThreshold
      : absDelta > spikeThreshold && market.volume24h < 100_000;
  } else {
    warnings.push(
      `Spike detection unavailable: no prior snapshot found for market ${market.marketId} in ${profile.spikeWindowLabel} window`,
    );
  }

  if (!persistenceSnapshot) {
    warnings.push(
      `Persistence test unavailable: no prior snapshot found for market ${market.marketId} in ${profile.persistenceWindowLabel} window`,
    );
  }

  if (spikeSnapshot && persistenceSnapshot) {
    const originalMove = spikeSnapshot.probability - persistenceSnapshot.probability;
    const originalMoveMagnitude = Math.abs(originalMove);
    const movedTowardBaseline =
      Math.abs(market.probability - persistenceSnapshot.probability)
      < Math.abs(spikeSnapshot.probability - persistenceSnapshot.probability);
    const reversalAmount = Math.abs(spikeSnapshot.probability - market.probability);
    transitoryMove =
      originalMoveMagnitude > profile.transitoryMoveThreshold
      && movedTowardBaseline
      && reversalAmount > originalMoveMagnitude * profile.transitoryReversalRatio;
  }

  return {
    priceSpikeDetected,
    transitoryMove,
    warnings,
  };
}

export function evaluateHistoryFlags(
  market: PolymarketMarketResult,
  nowMs: number,
  snapshotFilePath?: string,
  context?: HistoryHeuristicContext,
): HistoryFlags {
  return evaluateHistoryFlagsWithReader(market, nowMs, snapshotFilePath, readSnapshotRecords, context);
}

function evaluateHistoryFlagsWithReader(
  market: PolymarketMarketResult,
  nowMs: number,
  snapshotFilePath: string | undefined,
  readRecords: ForecastHistoryReader,
  context?: HistoryHeuristicContext,
): HistoryFlags {
  if (!market.marketId) {
    return {
      priceSpikeDetected: false,
      transitoryMove: false,
      warnings: [],
    };
  }

  try {
    const records = readRecords(snapshotFilePath, market.marketId);
    return evaluateMarketHistory(market, records, nowMs, context);
  } catch {
    return {
      priceSpikeDetected: false,
      transitoryMove: false,
      warnings: ['Snapshot history unavailable due to filesystem error'],
    };
  }
}

function daysUntilEndDate(endDate: string | null | undefined): number | null {
  if (!endDate) return null;
  const target = new Date(endDate);
  if (Number.isNaN(target.getTime())) return null;
  return (target.getTime() - Date.now()) / 86_400_000;
}

function isResolutionAlignedToHorizon(daysUntilResolution: number | null, horizonDays: number): boolean {
  if (daysUntilResolution === null) return false;
  return Math.abs(daysUntilResolution - horizonDays) <= Math.max(1.5, horizonDays * 0.35);
}

function shouldFilterResolutionMismatch(assetClass: string, horizonDays: number): boolean {
  return assetClass === 'crypto' && horizonDays <= 3;
}

function shouldUseShortHorizonCryptoAnchorRetrieval(assetClass: string, horizonDays: number): boolean {
  return assetClass === 'crypto' && horizonDays <= 3;
}

function requestedHorizonDistance(daysUntilResolution: number | null, horizonDays: number): number {
  return daysUntilResolution === null
    ? Number.POSITIVE_INFINITY
    : Math.abs(daysUntilResolution - horizonDays);
}

function compareShortHorizonResolutionCandidates(
  a: Pick<RawMarket, 'endDate' | 'volume24h' | 'probability'>,
  b: Pick<RawMarket, 'endDate' | 'volume24h' | 'probability'>,
  horizonDays: number,
): number {
  const aDaysToResolution = daysUntilEndDate(a.endDate);
  const bDaysToResolution = daysUntilEndDate(b.endDate);
  const aAligned = isResolutionAlignedToHorizon(aDaysToResolution, horizonDays);
  const bAligned = isResolutionAlignedToHorizon(bDaysToResolution, horizonDays);

  if (aAligned !== bAligned) {
    return aAligned ? -1 : 1;
  }

  const distanceDelta = requestedHorizonDistance(aDaysToResolution, horizonDays)
    - requestedHorizonDistance(bDaysToResolution, horizonDays);
  if (distanceDelta !== 0) {
    return distanceDelta;
  }
  if (a.volume24h !== b.volume24h) {
    return b.volume24h - a.volume24h;
  }
  return b.probability - a.probability;
}

function buildShortHorizonAnchorEndDateFilter(
  horizonDays: number,
  referenceTimeMs: number,
): { end_date_min: string; end_date_max: string } {
  const toleranceDays = Math.max(1.5, horizonDays * 0.35);
  const minDays = Math.max(0, horizonDays - toleranceDays);
  const maxDays = horizonDays + toleranceDays;
  return {
    end_date_min: new Date(referenceTimeMs + minDays * DAY_MS).toISOString(),
    end_date_max: new Date(referenceTimeMs + maxDays * DAY_MS).toISOString(),
  };
}

function resolveAnchorSignalCategory(
  question: string,
  signals: ReturnType<typeof extractSignals>,
): string {
  let bestCategory = signals[0]?.category ?? 'btc_price_target';
  let bestScore = -1;

  for (const signal of signals) {
    const score = scoreMarketRelevance(question, signal.category);
    if (score > bestScore) {
      bestScore = score;
      bestCategory = signal.category;
    }
  }

  return bestScore > 0
    ? bestCategory
    : signals[0]?.category ?? 'btc_price_target';
}

type AnchorCandidateMarket = {
  question: string;
  probability: number;
  volume?: number;
  createdAt?: number;
  endDate?: string | null;
};

type ShortHorizonAnchorSelection = {
  selectedMarkets: RawMarket[];
  selectedThresholds: PriceThreshold[];
  skippedResolutionMismatches: RawMarket[];
};

type ShortHorizonAnchorCandidate = {
  market: RawMarket;
  candidate: AnchorCandidateMarket;
};

function toAnchorCandidateMarket(market: RawMarket, referenceTimeMs: number): AnchorCandidateMarket {
  return {
    question: market.question,
    probability: market.probability,
    volume: market.volume24h,
    createdAt: market.ageDays != null ? referenceTimeMs - market.ageDays * DAY_MS : undefined,
    endDate: market.endDate ?? null,
  };
}

function buildAnchorSelectionKey(
  anchor: Pick<PriceThreshold, 'price' | 'rawProbability' | 'endDate'>,
): string {
  return `${anchor.price}|${anchor.rawProbability}|${anchor.endDate ?? ''}`;
}

function evaluateShortHorizonAnchorTrust(
  market: RawMarket,
  horizonDays: number,
): ReturnType<typeof evaluateAnchorTrust> {
  const daysToResolution = daysUntilEndDate(market.endDate);
  return evaluateAnchorTrust({
    hasVolume: market.volume24h > 0,
    isYoung: (market.ageDays ?? Number.POSITIVE_INFINITY) < 2,
    isShortHorizonCrypto: true,
    isLongHorizonCrypto: false,
    isNearTargetResolution: daysToResolution !== null && Math.abs(daysToResolution - horizonDays) <= 2,
  });
}

function selectPreferredShortHorizonAnchorCandidates(
  candidates: ShortHorizonAnchorCandidate[],
  ticker: string,
  horizonDays: number,
  referenceTimeMs: number,
): ShortHorizonAnchorCandidate[] {
  const bestByPrice = new Map<number, ShortHorizonAnchorCandidate>();

  for (const candidateEntry of candidates) {
    const daysToResolution = daysUntilEndDate(candidateEntry.market.endDate);
    if (daysToResolution === null) continue;

    const [threshold] = extractAnchorPriceThresholds(
      [candidateEntry.candidate],
      { ticker, horizonDays, referenceTimeMs },
    );
    if (!threshold) continue;

    const existing = bestByPrice.get(threshold.price);
    if (
      !existing
      || compareShortHorizonResolutionCandidates(candidateEntry.market, existing.market, horizonDays) < 0
    ) {
      bestByPrice.set(threshold.price, candidateEntry);
    }
  }

  return Array.from(bestByPrice.values());
}

function selectShortHorizonCryptoAnchorMarkets(
  markets: RawMarket[],
  ticker: string,
  horizonDays: number,
  referenceTimeMs: number,
): ShortHorizonAnchorSelection {
  const candidates = markets.map((market) => ({
    market,
    candidate: toAnchorCandidateMarket(market, referenceTimeMs),
  }));
  const preferredCandidates = selectPreferredShortHorizonAnchorCandidates(
    candidates,
    ticker,
    horizonDays,
    referenceTimeMs,
  );
  const strictCandidates = preferredCandidates.filter(({ market }) => {
    const daysToResolution = daysUntilEndDate(market.endDate);
    return daysToResolution !== null && isResolutionAlignedToHorizon(daysToResolution, horizonDays);
  });
  const trustedStrictCandidates = strictCandidates.filter(
    ({ market }) => evaluateShortHorizonAnchorTrust(market, horizonDays).trustScore === 'high',
  );
  const strictThresholdSource = trustedStrictCandidates.length > 0 ? trustedStrictCandidates : strictCandidates;
  const strictThresholds = extractAnchorPriceThresholds(
    strictThresholdSource.map(({ candidate }) => candidate),
    { ticker, horizonDays, referenceTimeMs },
  );
  const selectedThresholds = applyCryptoTerminalAnchorFallback(
    preferredCandidates.map(({ candidate }) => candidate),
    strictThresholds,
    ticker,
    horizonDays,
    referenceTimeMs,
  );

  const marketByKey = new Map<string, RawMarket>();
  for (const { market, candidate } of preferredCandidates) {
    const [threshold] = extractAnchorPriceThresholds([candidate], { ticker, horizonDays, referenceTimeMs });
    if (!threshold) continue;
    const key = buildAnchorSelectionKey(threshold);
    if (!marketByKey.has(key)) {
      marketByKey.set(key, market);
    }
  }

  const selectedMarkets = selectedThresholds
    .map((threshold) => marketByKey.get(buildAnchorSelectionKey(threshold)))
    .filter((market): market is RawMarket => market != null);
  const selectedIds = new Set(selectedMarkets.map((market) => market.marketId ?? market.question));
  const skippedResolutionMismatches = candidates
    .filter(({ market }) => {
      const daysToResolution = daysUntilEndDate(market.endDate);
      return (daysToResolution === null || !isResolutionAlignedToHorizon(daysToResolution, horizonDays))
        && !selectedIds.has(market.marketId ?? market.question);
    })
    .map(({ market }) => market);

  return {
    selectedMarkets,
    selectedThresholds,
    skippedResolutionMismatches,
  };
}

function isThresholdChartAlignedToHorizon(markets: RawMarket[], horizonDays: number): boolean {
  const thresholds = extractChartPriceThresholds(markets);
  if (thresholds.length < 2) return false;

  const datedThresholdMarkets = markets.filter((market) => {
    if (!market.endDate) return false;
    return extractChartPriceThresholds([{ question: market.question, probability: market.probability }]).length > 0;
  });
  if (datedThresholdMarkets.length < 2) return false;

  const alignedMarkets = datedThresholdMarkets.filter((market) => {
    const days = daysUntilEndDate(market.endDate);
    return isResolutionAlignedToHorizon(days, horizonDays);
  });

  return alignedMarkets.length >= 2;
}

// ---------------------------------------------------------------------------
// Description (injected into system prompt)
// ---------------------------------------------------------------------------

export const POLYMARKET_FORECAST_DESCRIPTION = `
Generates a prediction-market-weighted ensemble price forecast for any asset over a 1–365 day horizon.

Combines Polymarket crowd probabilities with optional auxiliary signals (news sentiment, fundamental
analyst targets, options skew) into a single calibrated forecast with a 95% confidence interval and
a quality grade (A–D).

Polymarket hosts markets from 1 day to 12 months out — all are valid inputs. Signal quality is
highest for liquid markets resolving in 1–90 days; longer-dated or low-volume markets are
automatically down-weighted by the quality scoring engine.

## What This Tool Does

1. Extracts relevant Polymarket search signals for the asset (earnings, macro, geopolitical, etc.)
2. Fetches live prediction-market probabilities from Polymarket
3. Maps each market to an asset-return impact using a pre-built δ(YES)/δ(NO) lookup table
4. Blends the Polymarket signal with any auxiliary signals you provide (sentiment, fundamentals, skew)
5. Outputs: forecast price, 95% CI, return percentage, per-signal breakdown, grade, and warnings

## When to Use

- User asks "Where will NVDA trade in a week / month / quarter?"
- User wants a forecast incorporating prediction-market data at any horizon (days to months)
- User asks "What does the market imply for [TICKER] by end of year?"
- You have already fetched sentiment or fundamentals and want to incorporate them into a price forecast
- User asks for a probability-weighted scenario analysis combining multiple market signals

## When NOT to Use

- Real-time stock price — use \`get_market_data\`
- Fundamental company analysis — use \`get_financials\`
- Multi-year DCF valuation (> 2 years) — use the DCF skill instead
- News summarisation — use \`web_search\`

## Signal Quality by Horizon

| Horizon | Polymarket signal strength | Notes |
|---------|--------------------------|-------|
| 1–30 days | ★★★ Strong | Many active markets, high volume, best accuracy |
| 30–90 days | ★★ Moderate | Fewer markets, still actionable |
| 90–365 days | ★ Weaker | Longer-dated markets tend to be less liquid; quality weights auto-adjust |

## Input Tips

- **ALWAYS pass \`current_price\`** (fetch with \`get_market_data\` first). Without it the 95% CI
  is shown as percentages only (relative to base 100), NOT in dollar terms. Call order:
  get_market_data(ticker) → polymarket_forecast(ticker, current_price=<fetched price>).
- **Pass \`sentiment_score\`** (from \`social_sentiment\`) if you have already called that tool —
  it improves forecast quality at no extra cost.
- **Pass \`fundamental_return\`** (analyst 1-year target implied return from \`get_financials\`) if
  available — use the decimal form, e.g. \`0.15\` for a +15% target.
- **Pass \`options_skew\`** if you have options data — use −1 (bearish), 0 (neutral), +1 (bullish).
- The tool fetches Polymarket data itself; you do **not** need to call \`polymarket_search\` first.
- For commodity proxies (USO for oil, GLD for gold, SLV for silver), the tool searches both the ETF ticker and the underlying commodity name (e.g. "WTI", "crude", "OPEC" for oil) to find the broadest set of relevant prediction markets.

## Interpreting the Output

| Grade | Score | Meaning |
|-------|-------|---------|
| A | 80–100 | High conviction — ≥5 liquid markets, multiple corroborating signals |
| B | 60–79  | Moderate conviction — useful directional signal |
| C | 40–59  | Low conviction — treat as indicative, not actionable alone |
| D | 0–39   | Speculative — few or no liquid markets, high uncertainty |

The 95% CI reflects both market-probability variance and a 20% model-uncertainty buffer.
A wide CI (σ > 5%) typically signals Grade C/D and limited predictive power.

## Composability Note

For richer analysis, call \`get_financials\` and \`social_sentiment\` first, then pass their results
to this tool via \`fundamental_return\` and \`sentiment_score\`. This turns two separate lookups into
a unified forecast with a higher quality grade.

\`\`\`
get_financials(NVDA) → fundamental_return = 0.18
social_sentiment(NVDA) → sentiment_score = 0.6
polymarket_forecast(NVDA, current_price=135.50, fundamental_return=0.18, sentiment_score=0.6)
\`\`\`
`.trim();

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const schema = z.object({
  ticker: z.string().describe('Asset ticker or name, e.g. "NVDA", "BTC", "GLD"'),
  horizon_days: z.number().int().min(1).max(365).default(7)
    .describe('Forecast horizon in days (1–365). Default: 7. Polymarket has markets from 1 day to 12 months — all are valid. Signal quality is highest for 1–90 day horizons.'),
  current_price: z.number().optional()
    .describe('Current asset price. If omitted, tool uses a placeholder and notes it.'),
  sentiment_score: z.number().min(-1).max(1).optional()
    .describe('News/social sentiment: -1 bearish, 0 neutral, +1 bullish. Pass if already retrieved.'),
  markov_return: z.number().optional()
    .describe('Markov-chain expected return over the forecast horizon as a decimal, pre-shrunk by Markov weight. Pass from a prior successful markov_distribution result when available.'),
  fundamental_return: z.number().optional()
    .describe('Analyst 1-year price target implied return as decimal (e.g. 0.15 for +15%). Pass if known.'),
  options_skew: z.number().min(-1).max(1).optional()
    .describe('Options skew signal: -1 bearish, 0 neutral, +1 bullish. Pass if available.'),
});

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Map a signal category to a human-readable theme label. */
function catToLabel(category: string): string {
  const map: Record<string, string> = {
    macro_rates:                'Fed / Rates',
    macro_growth:               'Growth / Recession',
    trade_policy:               'Trade Policy',
    tariff_increase:            'Tariffs',
    tariff_relief:              'Tariff Relief',
    geopolitical:               'Geopolitical',
    geopolitical_conflict:      'Conflict Risk',
    earnings:                   'Earnings',
    earnings_beat:              'Earnings Beat',
    earnings_miss:              'Earnings Risk',
    commodity:                  'Commodity',
    oil_spike:                  'Oil / Energy',
    supply_chain:               'Supply Chain',
    government_budget:          'Govt Budget',
    regulatory:                 'Regulation',
    fda_approval:               'FDA Approval',
    fda_rejection:              'FDA Risk',
    crypto_regulation_positive: 'Crypto Reg',
    crypto_regulation_negative: 'Crypto Reg Risk',
    btc_price_target:           'BTC Price Target',
    election_market_friendly:   'Election',
    etf_product:                'ETF Product',
    recession:                  'Recession',
    macro_data_strong:          'Strong Macro Data',
    macro_data_weak:            'Weak Macro Data',
    fed_rate_cut:               'Fed Rate Cut',
    fed_rate_hike:              'Fed Rate Hike',
  };
  return map[category] ?? category.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Truncate string to maxLen, appending '…' if cut. */
function truncCol(s: string, maxLen: number): string {
  return s.length > maxLen ? s.slice(0, maxLen - 1) + '…' : s;
}


function categoryToTier(category: string): 'macro' | 'geopolitical' | 'electoral' {
  const lower = category.toLowerCase();
  if (lower.includes('macro') || lower.includes('fed') || lower.includes('rate') ||
      lower.includes('gdp') || lower.includes('cpi')) return 'macro';
  if (lower.includes('election') || lower.includes('vote') || lower.includes('president')) return 'electoral';
  return 'geopolitical';
}

function sign(n: number): string {
  return n >= 0 ? '+' : '';
}

function pct(n: number, decimals = 2): string {
  return `${sign(n)}${(n * 100).toFixed(decimals)}`;
}

function sentimentLabel(score: number): string {
  if (score >= 0.5) return 'very bullish';
  if (score >= 0.1) return 'bullish';
  if (score <= -0.5) return 'very bearish';
  if (score <= -0.1) return 'bearish';
  return 'neutral';
}

function optionsLabel(skew: number): string {
  if (skew >= 0.5) return 'bullish skew';
  if (skew >= 0.1) return 'mildly bullish';
  if (skew <= -0.5) return 'bearish skew';
  if (skew <= -0.1) return 'mildly bearish';
  return 'neutral skew';
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

export function createPolymarketForecastTool(dependencies: ForecastToolDependencies = {}) {
  const fetchMarkets = dependencies.fetchMarkets ?? fetchPolymarketMarkets;
  const fetchAnchorMarketsWithQueries = dependencies.fetchAnchorMarketsWithQueries ?? fetchPolymarketAnchorMarketsWithQueries;
  const metaforecastQuestionFetcher = dependencies.fetchMetaforecastQuestions ?? fetchMetaforecastQuestions;
  const kalshiSignalFetcher = dependencies.fetchKalshiVolSignals
    ?? (process.env.KALSHI_API_KEY ? fetchKalshiVolSignals : undefined);
  const readRecords = dependencies.readRecords ?? readSnapshotRecords;
  const readReplayBundles = dependencies.readReplayBundles ?? readArbiterReplayBundles;
  const recordReplayPolymarketCapture = dependencies.recordReplayPolymarketCapture
    ?? ((capture: {
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }) => {
      appendReplayCachePolymarketCapture(capture);
    });

  return new DynamicStructuredTool({
    name: 'polymarket_forecast',
    description:
      'Generate a prediction-market-weighted ensemble price forecast for an asset over any horizon (1–365 days), ' +
      'combining Polymarket probabilities (markets span 1 day to 12 months) with optional sentiment, Markov, fundamental, and options signals.',
    schema,
    func: async (input) => {
      if (polymarketBreaker.isOpen()) {
        return formatToolResult({
          error: 'Polymarket API is temporarily unavailable (circuit open). Try again in a few minutes.',
        });
      }

      try {
        const ticker = input.ticker.trim().toUpperCase();
        const horizonDays = input.horizon_days ?? 7;
        const currentPrice = input.current_price;
        const basePrice = currentPrice ?? 100;
        const searchIdentity = resolveTickerSearchIdentity(ticker);
        const assetClass = inferAssetClass(searchIdentity.canonicalTicker);
        const historyWarnings: string[] = [];
        const nowMs = Date.now();
        const replayCapturedAt = new Date(nowMs).toISOString();
        const liveCalibration = resolveLiveBrierReplayCalibration({
          ticker: searchIdentity.canonicalTicker,
          assetClass,
          horizonDays,
          readReplayBundles,
        });
        historyWarnings.push(...liveCalibration.warnings);
        const useShortHorizonCryptoAnchorRetrieval = shouldUseShortHorizonCryptoAnchorRetrieval(assetClass, horizonDays);
        const shortHorizonAnchorEndDateFilter = useShortHorizonCryptoAnchorRetrieval
          ? buildShortHorizonAnchorEndDateFilter(horizonDays, nowMs)
          : undefined;

        // Step 1: Extract signals for this ticker (5 generic; 7 in short-horizon crypto fallback mode)
        const signals = extractSignals(searchIdentity.canonicalTicker, {
          horizonDays,
          preferShortHorizonCryptoSignals: useShortHorizonCryptoAnchorRetrieval,
        }).slice(0, useShortHorizonCryptoAnchorRetrieval ? 7 : 5);
        const genericReplayQuerySet = [...new Set(
          signals.flatMap((sig) => [sig.searchPhrase, ...(sig.queryVariants ?? [])]),
        )];
        const anchorReplayQuerySet = useShortHorizonCryptoAnchorRetrieval
          ? buildPolymarketAnchorQueryVariants(searchIdentity.canonicalTicker, { horizonDays })
          : [];
        const rawMarkets: RawMarket[] = [];
        const skippedResolutionMismatches: RawMarket[] = [];
        let shortHorizonAnchorThresholds: PriceThreshold[] = [];
        const allSnapshotRecords = readRecords(undefined);
        let replayQuerySet = genericReplayQuerySet;
        // Reader reused across all markets — avoids creating a closure per iteration
        const marketReader = (_filePath: string | undefined, marketId?: string) =>
          allSnapshotRecords.filter((r) => !marketId || r.marketId === marketId);
        const addStructuredMarkets = (
          markets: Array<PolymarketMarketResult & { signalCategory: string }>,
          options?: { skipResolutionMismatchFilter?: boolean },
        ) => {
          const seen = new Set<string>();
          for (const market of markets) {
            if (seen.has(market.question)) continue;
            seen.add(market.question);
            const history = evaluateHistoryFlagsWithReader(market, nowMs, undefined, marketReader, {
              assetClass,
              horizonDays,
            });
            historyWarnings.push(...history.warnings);
            const rawMarket = toRawMarket(market, market.signalCategory, history);
            if (!rawMarket) continue;
            if (
              shouldFilterResolutionMismatch(assetClass, horizonDays)
              && !options?.skipResolutionMismatchFilter
            ) {
              const daysToResolution = daysUntilEndDate(rawMarket.endDate);
              if (daysToResolution === null || !isResolutionAlignedToHorizon(daysToResolution, horizonDays)) {
                skippedResolutionMismatches.push(rawMarket);
                continue;
              }
            }
            rawMarkets.push(rawMarket);
          }
        };
        const fetchGenericStructuredMarkets = async () => {
          const allResults = await Promise.allSettled(
            signals.map((sig) => {
                const phrases = [sig.searchPhrase, ...(sig.queryVariants ?? [])];
                return Promise.allSettled(
                  phrases.map((phrase) => fetchMarkets(phrase, 5, {
                    snapshotFilePath: DEFAULT_POLYMARKET_SNAPSHOTS_PATH,
                    enrichMicrostructure: useShortHorizonCryptoAnchorRetrieval,
                  })),
                ).then((settledVariants) =>
                  settledVariants
                  .filter((r): r is PromiseFulfilledResult<PolymarketMarketResult[]> => r.status === 'fulfilled')
                  .flatMap((r) => r.value)
                  .filter((m) => scoreMarketRelevance(m.question, sig.category) > 0)
                  .map((m) => ({ ...m, signalCategory: sig.category })),
              ).then((markets) => shouldFilterResolutionMismatch(assetClass, horizonDays)
                ? markets.sort((a, b) => compareShortHorizonResolutionCandidates(a, b, horizonDays))
                : markets,
              );
            }),
          );

          return allResults
            .filter((result): result is PromiseFulfilledResult<Array<PolymarketMarketResult & { signalCategory: string }>> => result.status === 'fulfilled')
            .flatMap((result) => result.value);
        };

        if (useShortHorizonCryptoAnchorRetrieval) {
          const anchorFrontQueries = anchorReplayQuerySet.slice(0, 6);
          const anchorRetryQueries = anchorReplayQuerySet.slice(anchorFrontQueries.length);
          replayQuerySet = anchorFrontQueries;

          let anchorMarkets = await fetchAnchorMarketsWithQueries(anchorFrontQueries, 12, {
            ticker: searchIdentity.canonicalTicker,
            horizonDays,
            endDateFilter: shortHorizonAnchorEndDateFilter,
            enrichMicrostructure: true,
          });

          if (anchorMarkets.length === 0 && anchorRetryQueries.length > 0) {
            replayQuerySet = [...anchorFrontQueries, ...anchorRetryQueries];
            anchorMarkets = await fetchAnchorMarketsWithQueries(anchorRetryQueries, 12, {
              ticker: searchIdentity.canonicalTicker,
              horizonDays,
              endDateFilter: shortHorizonAnchorEndDateFilter,
              enrichMicrostructure: true,
            });
          }

          addStructuredMarkets(
            anchorMarkets.map((market) => ({
              ...market,
              signalCategory: resolveAnchorSignalCategory(market.question, signals),
            })),
            { skipResolutionMismatchFilter: true },
          );

          const shortHorizonSelection = selectShortHorizonCryptoAnchorMarkets(
            rawMarkets,
            searchIdentity.canonicalTicker,
            horizonDays,
            nowMs,
          );
          rawMarkets.splice(0, rawMarkets.length, ...shortHorizonSelection.selectedMarkets);
          shortHorizonAnchorThresholds = shortHorizonSelection.selectedThresholds;
          skippedResolutionMismatches.push(...shortHorizonSelection.skippedResolutionMismatches);

          if (rawMarkets.length === 0) {
            replayQuerySet = [...new Set([...anchorReplayQuerySet, ...genericReplayQuerySet])];
            addStructuredMarkets(await fetchGenericStructuredMarkets());
          }
        } else {
          addStructuredMarkets(await fetchGenericStructuredMarkets());
        }

        if (skippedResolutionMismatches.length > 0) {
          historyWarnings.push(
            `Skipped ${skippedResolutionMismatches.length} Polymarket market${skippedResolutionMismatches.length === 1 ? '' : 's'} because ${skippedResolutionMismatches.length === 1 ? 'its resolution date was missing, invalid, or did' : 'their resolution dates were missing, invalid, or did'} not align with the requested ${horizonDays}-day horizon.`,
          );
        }

        // Step 3: Build MarketInput array
        let calibratedMarketCount = 0;
        const markets: MarketInput[] = rawMarkets.map((m) => {
          const mImpact = lookupImpact(m.signalCategory, assetClass);
          const shouldApplyLiveCalibration = (
            liveCalibration.mode === 'static' || liveCalibration.mode === 'horizon_aware'
          ) && isLiveCalibrationApplicable(m.probability);
          const probability = shouldApplyLiveCalibration
            ? predictWithBrierReplayState(m.probability, liveCalibration.state!)
            : m.probability;
          const daysToExpiry = daysUntilEndDate(m.endDate);
          if (shouldApplyLiveCalibration && probability !== m.probability) calibratedMarketCount++;
          return {
            question: m.question,
            probability,
            volume24hUsd: m.volume24h,
            ageDays: m.ageDays,
            daysToExpiry: daysToExpiry === null ? undefined : daysToExpiry,
            bidAskSpread: m.bidAskSpread,
            priceVelocityPpH: m.priceVelocityPpH,
            priceVelocityLogitPerHour: m.priceVelocityLogitPerHour,
            maxHourlyJump: m.maxHourlyJump,
            maxHourlyLogitJump: m.maxHourlyLogitJump,
            priceSpikeDetected: m.priceSpikeDetected,
            transitoryMove: m.transitoryMove,
            signalTier: categoryToTier(m.signalCategory),
            deltaYes: mImpact.deltaYes,
            deltaNo: mImpact.deltaNo,
            requestedHorizonDays: shouldFilterResolutionMismatch(assetClass, horizonDays) ? horizonDays : undefined,
            marketSemantics: classifyPolymarketQuestion(m.question),
          };
        });
        const liveCalibrationApplied = (
          liveCalibration.mode === 'static' || liveCalibration.mode === 'horizon_aware'
        ) && calibratedMarketCount > 0;
        const { evidence: crossPlatformEvidence, warnings: crossPlatformWarnings } = await collectCrossPlatformEvidence({
          assetClass,
          horizonDays,
          nowMs,
          rawMarkets,
          fetchMetaforecastQuestions: metaforecastQuestionFetcher,
          fetchKalshiVolSignals: kalshiSignalFetcher,
        });
        const crossPlatformAdjustment = isCrossPlatformFusionEnabled()
          ? deriveCrossPlatformConfidenceAdjustment(crossPlatformEvidence)
          : deriveCrossPlatformConfidenceAdjustment([]);

        // Step 4: Run ensemble — also capture intermediate values for display
        const otherSignals = {
          sentimentScore: input.sentiment_score,
          markovReturn: input.markov_return,
          fundamentalReturn: input.fundamental_return,
          optionsSkew: input.options_skew,
          horizonDays,
        };
        const ensembleOptions = { adaptiveWeighting: true } as const;

        // Compute thresholds once — used for both the threshold-implied raw
        // forecast (below) and the distribution chart display (further down).
        const thresholdChartAligned = shortHorizonAnchorThresholds.length > 0
          ? true
          : isThresholdChartAlignedToHorizon(rawMarkets, horizonDays);

        // In the generic path, restrict to threshold markets whose resolution
        // date aligns with the requested horizon so that misaligned contracts
        // at the same strike cannot contaminate the ladder via probability
        // averaging inside extractChartPriceThresholds.
        const genericThresholdMarkets = (shortHorizonAnchorThresholds.length === 0 && thresholdChartAligned)
          ? rawMarkets.filter((m) => {
              if (!m.endDate) return false;
              if (extractChartPriceThresholds([{ question: m.question, probability: m.probability }]).length === 0) return false;
              return isResolutionAlignedToHorizon(daysUntilEndDate(m.endDate), horizonDays);
            })
          : rawMarkets;

        const thresholds = shortHorizonAnchorThresholds.length > 0
          ? shortHorizonAnchorThresholds
          : extractChartPriceThresholds(genericThresholdMarkets);

        // Build a bias-corrected forecast ladder.
        // shortHorizonAnchorThresholds already stores bias-corrected probabilities;
        // extractChartPriceThresholds returns raw Polymarket quotes, so apply YES_BIAS_MULTIPLIER.
        const forecastLadder: ThresholdLadderPoint[] = shortHorizonAnchorThresholds.length > 0
          ? shortHorizonAnchorThresholds.map((t) => ({ price: t.price, probability: t.probability }))
          : extractChartPriceThresholds(genericThresholdMarkets).map((t) => ({
              price: t.price,
              probability: t.probability * YES_BIAS_MULTIPLIER,
            }));

        const { signal: pmSignal, avgQuality, warnings: pmWarnings } = computePolymarketSignal(markets);
        let rawPolymarketResult = runEnsemble(
          basePrice,
          markets,
          { horizonDays },
          ensembleOptions,
        );
        let thresholdPathActive = false;
        const thresholdPathWarnings: string[] = [];

        if (thresholdChartAligned && forecastLadder.length >= 2) {
          const ladderCheck = isCleanThresholdLadder(forecastLadder);
          thresholdPathWarnings.push(...ladderCheck.warnings);
          if (ladderCheck.clean) {
            rawPolymarketResult = computeThresholdImpliedRawForecast(
              forecastLadder,
              basePrice,
              horizonDays,
            );
            thresholdPathActive = true;
          }
        }

        const { weights } = computeEnsemble(pmSignal, avgQuality, otherSignals, ensembleOptions);
        const result = runEnsemble(basePrice, markets, otherSignals, ensembleOptions);
        const adjustedQualityScore = clamp(
          result.qualityScore + crossPlatformAdjustment.qualityScoreDelta,
          0,
          100,
        );
        const adjustedQualityGrade = scoreToGrade(adjustedQualityScore);
        const adjustedSigma = result.sigma * crossPlatformAdjustment.sigmaMultiplier;
        const adjustedCi = computeCI(result.forecastPrice, adjustedSigma);
        const rawCi = computeCI(rawPolymarketResult.forecastPrice, rawPolymarketResult.sigma);
        const hasAuxiliarySignals = (
          input.sentiment_score !== undefined
          || input.markov_return !== undefined
          || input.fundamental_return !== undefined
          || input.options_skew !== undefined
        );

        if (recordReplayPolymarketCapture) {
          const rawRow = createRawPolymarketReplayRow({
            capturedAt: replayCapturedAt,
            ticker: searchIdentity.canonicalTicker,
            horizonDays,
            currentPrice: currentPrice ?? null,
            querySet: replayQuerySet,
            selectedMarkets: rawMarkets,
            warnings: historyWarnings,
          });
          const polymarket = freezePolymarketReplayBlock({
            querySet: replayQuerySet,
            selectedMarkets: rawMarkets.map((market) => ({
              ...market,
              relevanceScore: scoreMarketRelevance(market.question, market.signalCategory),
            })),
            warnings: historyWarnings,
            forecastReturn: rawPolymarketResult.forecastReturn,
            qualityScore: adjustedQualityScore,
            qualityGrade: adjustedQualityGrade,
            crossPlatformEvidence: crossPlatformEvidence.map((entry) => ({
              source: entry.source,
              kind: entry.kind,
              flagged: entry.flagged,
              ...(entry.deltaFromPolymarket !== undefined
                ? { deltaFromPolymarket: entry.deltaFromPolymarket }
                : {}),
              ...(entry.intensityBoost !== undefined
                ? { intensityBoost: entry.intensityBoost }
                : {}),
            })),
            crossPlatformAdjustment: {
              basis: crossPlatformAdjustment.basis,
              applied: crossPlatformAdjustment.applied,
              qualityScoreDelta: crossPlatformAdjustment.qualityScoreDelta,
              sigmaMultiplier: crossPlatformAdjustment.sigmaMultiplier,
            },
          });

          recordReplayPolymarketCapture({ rawRow, polymarket });
        }

        // Step 5: Format output
        const rawReturnPct = (rawPolymarketResult.forecastReturn * 100).toFixed(2);
        const returnPct = (result.forecastReturn * 100).toFixed(2);
        const rawSigmaPct = (rawPolymarketResult.sigma * 100).toFixed(2);
        const sigmaPct = (adjustedSigma * 100).toFixed(2);
        const rawCiLow = rawCi.low;
        const rawCiHigh = rawCi.high;
        const ciLow = adjustedCi.low;
        const ciHigh = adjustedCi.high;
        const pmPct = pct(result.pmSignal);
        const pmWeightPct = (result.pmNormalizedWeight * 100).toFixed(1);
        const avgQualityStr = result.avgMarketQuality.toFixed(3);
        let thresholdChartWarning: string | null = null;
        const displayLabel = searchIdentity.canonicalNames[0]?.toUpperCase() ?? ticker;

        const lines: string[] = [
          `📊 Polymarket Forecast: ${displayLabel} (${ticker})  |  Horizon: ${horizonDays} days`,
        ];

        if (currentPrice === undefined) {
          lines.push('⚠️  No current price provided — price shown relative to base 100');
        }

        if (horizonDays > 90) {
          lines.push(`⚠️  Horizon ${horizonDays}d > 90 days: Polymarket signal accuracy decreases for longer horizons. Wider CI expected. Consider supplementing with DCF skill for multi-month forecasts.`);
        } else if (horizonDays > 14) {
          lines.push(`ℹ️  Horizon ${horizonDays}d: Polymarket markets exist at this range but signal quality is moderate. 95% CI is wider than short-term forecasts.`);
        }
        if (liveCalibration.mode === 'horizon_aware' && liveCalibrationApplied) {
          lines.push(
            `ℹ️  Horizon-aware replay calibration is active: fitted on ${liveCalibration.evidenceRows} labeled mid-confidence ${horizonDays}d BTC market ${liveCalibration.evidenceRows === 1 ? 'outcome' : 'outcomes'} across ${liveCalibration.evidenceBundles} replay ${liveCalibration.evidenceBundles === 1 ? 'bundle' : 'bundles'}; adjusted ${calibratedMarketCount} market ${calibratedMarketCount === 1 ? 'quote' : 'quotes'} before blending.`,
          );
        } else if (liveCalibration.mode === 'static' && liveCalibrationApplied) {
          lines.push(`ℹ️  Live Brier replay calibration is active: compressed ${calibratedMarketCount} mid-confidence market ${calibratedMarketCount === 1 ? 'quote' : 'quotes'} toward neutral before blending.`);
        }
        if (crossPlatformAdjustment.summary) {
          lines.push(`${crossPlatformAdjustment.applied ? '⚠️' : 'ℹ️'}  ${crossPlatformAdjustment.summary}`);
        }

        lines.push('');
        lines.push(`Current price:   ${currentPrice !== undefined ? '$' + basePrice : 'not provided — CI shown as %'}`);
        lines.push('── Raw Polymarket Forecast ───────────────────────────────────────────────');
        lines.push(`Raw Polymarket forecast: ${currentPrice !== undefined ? '$' + rawPolymarketResult.forecastPrice.toFixed(2) : '(base 100) ' + rawPolymarketResult.forecastPrice.toFixed(2)}  (${sign(rawPolymarketResult.forecastReturn)}${rawReturnPct}%)${thresholdPathActive ? '  [threshold-implied distribution]' : ''}`);
        if (currentPrice !== undefined) {
          lines.push(`Raw Polymarket 95% CI: [$${rawCiLow.toFixed(2)} – $${rawCiHigh.toFixed(2)}]  (σ = ${rawSigmaPct}%)`);
        } else {
          const rawCiLowPct = ((rawCiLow / basePrice - 1) * 100).toFixed(2);
          const rawCiHighPct = ((rawCiHigh / basePrice - 1) * 100).toFixed(2);
          const rawCiHighSign = parseFloat(rawCiHighPct) >= 0 ? '+' : '';
          lines.push(`Raw Polymarket 95% CI: [${rawCiLowPct}% – ${rawCiHighSign}${rawCiHighPct}%]  (σ = ${rawSigmaPct}%)  ← % relative to current price`);
        }
        lines.push(`Raw Polymarket grade: ${rawPolymarketResult.qualityGrade} (${rawPolymarketResult.qualityScore}/100)`);
        lines.push('');

        // ── Polymarket Signal Summary (grouped by theme) ───────────────────────────
        const numThemes = rawMarkets.reduce((s, m) => { s.add(m.signalCategory); return s; }, new Set<string>()).size;
        lines.push(`── Polymarket Signal Summary  (w̄ = ${avgQualityStr} · ${markets.length} markets · ${numThemes} themes) ─`);

        if (markets.length === 0) {
          lines.push('  [No Polymarket markets found for this asset]');
        } else {
          type ThemeRow = {
            category: string;
            label: string;
            netCondReturn: number;
            topQuestion: string;
            topProb: number;
            absContrib: number;
          };

          const byCategory = new Map<string, { question: string; probability: number; condReturn: number }[]>();
          for (const m of rawMarkets) {
            const mImpact = lookupImpact(m.signalCategory, assetClass);
            const condReturn = computeConditionalReturn(adjustYesBias(m.probability), mImpact.deltaYes, mImpact.deltaNo);
            if (!byCategory.has(m.signalCategory)) byCategory.set(m.signalCategory, []);
            byCategory.get(m.signalCategory)!.push({ question: m.question, probability: m.probability, condReturn });
          }

          const rows: ThemeRow[] = [];
          for (const [cat, entries] of byCategory) {
            const net = entries.reduce((s, e) => s + e.condReturn, 0) / entries.length;
            const top = entries.reduce((best, e) => Math.abs(e.condReturn) >= Math.abs(best.condReturn) ? e : best);
            rows.push({ category: cat, label: catToLabel(cat), netCondReturn: net, topQuestion: top.question, topProb: top.probability, absContrib: Math.abs(net) });
          }

          const totalAbs = rows.reduce((s, r) => s + r.absContrib, 0) || 1;
          rows.sort((a, b) => b.absContrib - a.absContrib);

          const W_THEME = 22;
          const W_DIR   = 13;
          const W_SIG   = 48;

          const header = `  ${'Theme'.padEnd(W_THEME)}  ${'Direction'.padEnd(W_DIR)}  ${'Key Signal'.padEnd(W_SIG)}  Contribution`;
          const divider = `  ${'─'.repeat(W_THEME + W_DIR + W_SIG + 18)}`;
          lines.push(header);
          lines.push(divider);

          let bullish = 0;
          let bearish = 0;
          let neutral = 0;
          for (const row of rows) {
            const dir = row.netCondReturn > 0.0005 ? '↑ Bullish' : row.netCondReturn < -0.0005 ? '↓ Bearish' : '→ Neutral';
            if (dir.startsWith('↑')) bullish++;
            else if (dir.startsWith('↓')) bearish++;
            else neutral++;

            const probPct = `${(row.topProb * 100).toFixed(0)}% YES`;
            const keySignal = truncCol(`${row.topQuestion}: ${probPct}`, W_SIG);
            const contrib = `${((row.absContrib / totalAbs) * 100).toFixed(0)}%`;

            lines.push(
              `  ${truncCol(row.label, W_THEME).padEnd(W_THEME)}  ${dir.padEnd(W_DIR)}  ${keySignal.padEnd(W_SIG)}  ${contrib.padStart(5)}`,
            );
          }

          lines.push(divider);

          const netLean = result.pmSignal > 0.005 ? ' (bullish lean)' : result.pmSignal < -0.005 ? ' (bearish lean)' : '';
          lines.push(`  Consensus: ${bullish} bullish · ${bearish} bearish · ${neutral} neutral    Net signal: ${pmPct}%${netLean}`);
        }

        lines.push('');
        lines.push('── Other Signals ──────────────────────────────────────────────────────────');

        const wSent = weights['sentiment'];
        const wMarkov = weights['markov'];
        const wFund = weights['fundamental'];
        const wOpt = weights['options'];

        if (input.sentiment_score !== undefined) {
          const sentContrib = pct(input.sentiment_score * 0.04);
          lines.push(`  News sentiment:     ${sentimentLabel(input.sentiment_score)} → ${sentContrib}%  (weight: ${((wSent ?? 0) * 100).toFixed(1)}%)`);
        } else {
          lines.push('  News sentiment:     [signal omitted — not provided]');
        }

        if (input.fundamental_return !== undefined) {
          const fundContrib = pct(input.fundamental_return * (horizonDays / 365));
          lines.push(`  Fundamentals:       ${fundContrib}%  (weight: ${((wFund ?? 0) * 100).toFixed(1)}%)`);
        } else {
          lines.push('  Fundamentals:       [signal omitted — not provided]');
        }

        if (input.markov_return !== undefined) {
          const markovContrib = pct(input.markov_return);
          lines.push(`  Markov chain:       ${markovContrib}%  (weight: ${((wMarkov ?? 0) * 100).toFixed(1)}%)`);
        } else {
          lines.push('  Markov chain:       [signal omitted — not provided]');
        }

        if (input.options_skew !== undefined) {
          const optContrib = pct(input.options_skew * 0.03);
          lines.push(`  Options skew:       ${optionsLabel(input.options_skew)} → ${optContrib}%  (weight: ${((wOpt ?? 0) * 100).toFixed(1)}%)`);
        } else {
          lines.push('  Options skew:       [signal omitted — not provided]');
        }

        if (crossPlatformEvidence.length > 0) {
          lines.push('');
          lines.push('── Cross-Platform Evidence ───────────────────────────────────────────────');
          for (const evidence of crossPlatformEvidence) {
            const label = evidence.source === 'metaforecast' ? 'MetaForecast' : 'Kalshi';
            lines.push(`  ${label}: ${evidence.diagnostic}`);
          }
        }

        lines.push('');
        lines.push('── Signal Mixing ─────────────────────────────────────────────────────────');
        lines.push(`Blended forecast: ${currentPrice !== undefined ? '$' + result.forecastPrice.toFixed(2) : '(base 100) ' + result.forecastPrice.toFixed(2)}  (${sign(result.forecastReturn)}${returnPct}%)`);
        if (currentPrice !== undefined) {
          lines.push(`Blended 95% CI: [$${ciLow.toFixed(2)} – $${ciHigh.toFixed(2)}]  (σ = ${sigmaPct}%)`);
        } else {
          const ciLowPct = ((ciLow / basePrice - 1) * 100).toFixed(2);
          const ciHighPct = ((ciHigh / basePrice - 1) * 100).toFixed(2);
          const ciHighSign = parseFloat(ciHighPct) >= 0 ? '+' : '';
          lines.push(`Blended 95% CI: [${ciLowPct}% – ${ciHighSign}${ciHighPct}%]  (σ = ${sigmaPct}%)  ← % relative to current price`);
        }
        lines.push(`Blended grade: ${adjustedQualityGrade} (${adjustedQualityScore}/100)`);
        if (hasAuxiliarySignals) {
          lines.push(`Polymarket weight after mixing: ${pmWeightPct}%  (remainder from sentiment / fundamentals / options / Markov)`);
        } else {
          lines.push('Polymarket weight after mixing: 100.0%  (no auxiliary signals were provided)');
        }

        // thresholds and thresholdChartAligned are already computed above (before runEnsemble).
        if (thresholds.length >= 2 && thresholdChartAligned) {
          const chart = buildPriceDistributionChart(thresholds, currentPrice, ticker);
          if (chart) {
            lines.push('');
            lines.push('── Price Distribution (from threshold markets) ────────────────────────────');
            lines.push(chart);
          }
        } else if (thresholds.length >= 2) {
          thresholdChartWarning = 'Threshold-style markets were omitted from the distribution chart because their resolution dates do not align with the requested forecast horizon.';
        }

        lines.push('');
        lines.push('── Warnings ───────────────────────────────────────────────────────────────');

        const allWarnings = [
          ...historyWarnings,
          ...crossPlatformWarnings,
          ...crossPlatformAdjustment.warnings,
          ...(result.warnings ?? []),
          ...pmWarnings.filter((w) => !result.warnings?.includes(w)),
          ...(thresholdChartWarning ? [thresholdChartWarning] : []),
          ...thresholdPathWarnings,
        ];
        const uniqueWarnings = [...new Set(allWarnings)];
        if (uniqueWarnings.length === 0) {
          lines.push('  None');
        } else {
          for (const w of uniqueWarnings) {
            lines.push(`  ⚠ ${w}`);
          }
        }

        lines.push('');
        lines.push('── Research basis: Reichenbach & Walther (2025) · Cordoba et al. (2024) · Tsang & Yang (2026)');

        return formatToolResult({
          result: lines.join('\n'),
          rawForecastReturn: rawPolymarketResult.forecastReturn,
          blendedForecastReturn: result.forecastReturn,
          rawForecastPrice: rawPolymarketResult.forecastPrice,
          blendedForecastPrice: result.forecastPrice,
          crossPlatformEvidence,
          crossPlatformAdjustment,
          qualityScore: adjustedQualityScore,
          qualityGrade: adjustedQualityGrade,
          ...(liveCalibrationApplied ? { forecastReturn: result.forecastReturn } : {}),
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`[polymarket_forecast] ${message}`);
      }
    },
  });
}

export const polymarketForecastTool = createPolymarketForecastTool();
