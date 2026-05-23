import { MS_PER_DAY } from '../../utils/time.js';
import type {
  ArbiterReplayBundle,
  ArbiterReplayPolymarketMarket,
  ArbiterReplaySemanticLabel,
} from './arbiter-replay.js';

export interface ReplayPricePoint {
  at: string;
  price: number;
}

export interface ReplayPriceHistory {
  points: ReplayPricePoint[];
}

export interface ReplayLabelEligibility {
  forecastReady: boolean;
  semanticReady: boolean;
  ready: boolean;
  pendingReasons: string[];
}

function sortPoints(points: ReplayPricePoint[]): ReplayPricePoint[] {
  return [...points]
    .filter((point) => Number.isFinite(Date.parse(point.at)) && Number.isFinite(point.price) && point.price > 0)
    .sort((a, b) => Date.parse(a.at) - Date.parse(b.at));
}

function forecastTargetTimeMs(bundle: ArbiterReplayBundle): number {
  return Date.parse(bundle.capturedAt) + bundle.horizonDays * MS_PER_DAY;
}

function marketTargetTimeMs(
  market: ArbiterReplayPolymarketMarket,
  bundle: ArbiterReplayBundle,
): number {
  const marketEndMs = Date.parse(market.endDate);
  return Number.isFinite(marketEndMs) ? marketEndMs : forecastTargetTimeMs(bundle);
}

function latestPointTimeMs(history: ReplayPriceHistory): number | null {
  const points = sortPoints(history.points);
  if (points.length === 0) return null;
  return Date.parse(points[points.length - 1]!.at);
}

function settlementPoint(
  history: ReplayPriceHistory,
  targetTimeMs: number,
): ReplayPricePoint | null {
  const points = sortPoints(history.points);
  const atOrAfter = points.find((point) => Date.parse(point.at) >= targetTimeMs);
  if (atOrAfter) return atOrAfter;
  return points.length > 0 ? points[points.length - 1]! : null;
}

function pointsInWindow(
  history: ReplayPriceHistory,
  startTimeMs: number,
  endTimeMs: number,
): ReplayPricePoint[] {
  return sortPoints(history.points).filter((point) => {
    const pointMs = Date.parse(point.at);
    return pointMs >= startTimeMs && pointMs <= endTimeMs;
  });
}

function labelTerminalOutcome(question: string, level: number, settlementPrice: number): 'yes' | 'no' {
  const lower = question.toLowerCase();
  if (/\b(at least|not below)\b/.test(lower)) return settlementPrice >= level ? 'yes' : 'no';
  if (/\b(at most|not above)\b/.test(lower)) return settlementPrice <= level ? 'yes' : 'no';
  if (/\b(below|under|less than)\b/.test(lower)) return settlementPrice < level ? 'yes' : 'no';
  return settlementPrice > level ? 'yes' : 'no';
}

function labelBarrierOutcome(
  market: ArbiterReplayPolymarketMarket,
  path: ReplayPricePoint[],
): 'yes' | 'no' | 'unsupported' {
  const level = market.extractedPriceLevels[0];
  if (!Number.isFinite(level)) return 'unsupported';
  const lower = market.question.toLowerCase();

  if (/\btrade at\b/.test(lower)) {
    const touched = path.some((point) => Math.abs(point.price - level) / level <= 0.0025);
    return touched ? 'yes' : 'no';
  }

  const minPrice = path.reduce((min, point) => Math.min(min, point.price), Number.POSITIVE_INFINITY);
  const maxPrice = path.reduce((max, point) => Math.max(max, point.price), Number.NEGATIVE_INFINITY);
  if (!Number.isFinite(minPrice) || !Number.isFinite(maxPrice)) return 'unsupported';

  if (/\b(dip|drop below|fall below|below|under)\b/.test(lower)) {
    return minPrice <= level ? 'yes' : 'no';
  }

  return maxPrice >= level ? 'yes' : 'no';
}

function labelRangeOutcome(
  market: ArbiterReplayPolymarketMarket,
  settlementPrice: number,
): 'yes' | 'no' | 'unsupported' {
  const sortedLevels = [...market.extractedPriceLevels].sort((a, b) => a - b);
  if (sortedLevels.length < 2) return 'unsupported';
  const [lower, upper] = sortedLevels;
  return settlementPrice >= lower! && settlementPrice <= upper! ? 'yes' : 'no';
}

export function evaluateReplayLabelEligibility(
  bundle: ArbiterReplayBundle,
  history: ReplayPriceHistory,
): ReplayLabelEligibility {
  const latestMs = latestPointTimeMs(history);
  const pendingReasons: string[] = [];

  if (latestMs === null) {
    return {
      forecastReady: false,
      semanticReady: false,
      ready: false,
      pendingReasons: ['No historical price points were provided for replay labeling.'],
    };
  }

  const forecastReady = bundle.currentPrice !== null && latestMs >= forecastTargetTimeMs(bundle);
  if (bundle.currentPrice === null) {
    pendingReasons.push('Bundle is missing currentPrice, so forecast labels cannot be computed.');
  } else if (!forecastReady) {
    pendingReasons.push('Forecast horizon has not elapsed in the supplied price history.');
  }

  const scorableMarkets = (bundle.polymarket?.selectedMarkets ?? []).filter((market) =>
    market.semantics === 'terminal'
    || market.semantics === 'barrier_touch'
    || market.semantics === 'range'
    || market.semantics === 'path_dependent'
    || market.semantics === 'unknown',
  );
  const semanticReady = scorableMarkets.every((market) => latestMs >= marketTargetTimeMs(market, bundle));
  if (!semanticReady) {
    pendingReasons.push('At least one captured Polymarket market has not reached its labeling end time.');
  }

  return {
    forecastReady,
    semanticReady,
    ready: forecastReady && semanticReady,
    pendingReasons,
  };
}

export function labelReplaySemanticMarket(
  bundle: ArbiterReplayBundle,
  market: ArbiterReplayPolymarketMarket,
  history: ReplayPriceHistory,
  labeledAt: string,
): ArbiterReplaySemanticLabel {
  const targetTimeMs = marketTargetTimeMs(market, bundle);
  const settlement = settlementPoint(history, targetTimeMs);
  const path = pointsInWindow(history, Date.parse(bundle.capturedAt), targetTimeMs);

  if (!settlement) {
    return {
      marketId: market.marketId,
      semantics: market.semantics,
      outcome: 'unsupported',
      labeledAt,
    };
  }

  if (market.semantics === 'terminal') {
    const level = market.extractedPriceLevels[0];
    return {
      marketId: market.marketId,
      semantics: market.semantics,
      outcome: Number.isFinite(level)
        ? labelTerminalOutcome(market.question, level, settlement.price)
        : 'unsupported',
      labeledAt,
    };
  }

  if (market.semantics === 'barrier_touch') {
    return {
      marketId: market.marketId,
      semantics: market.semantics,
      outcome: labelBarrierOutcome(market, path),
      labeledAt,
    };
  }

  if (market.semantics === 'range') {
    return {
      marketId: market.marketId,
      semantics: market.semantics,
      outcome: labelRangeOutcome(market, settlement.price),
      labeledAt,
    };
  }

  return {
    marketId: market.marketId,
    semantics: market.semantics,
    outcome: 'unsupported',
    labeledAt,
  };
}

export function labelReplayBundle(
  bundle: ArbiterReplayBundle,
  history: ReplayPriceHistory,
  labeledAt: string,
): ArbiterReplayBundle {
  const eligibility = evaluateReplayLabelEligibility(bundle, history);
  if (!eligibility.ready || bundle.currentPrice === null) {
    return bundle;
  }

  const forecastSettlement = settlementPoint(history, forecastTargetTimeMs(bundle));
  if (!forecastSettlement) return bundle;

  const realizedReturn = (forecastSettlement.price - bundle.currentPrice) / bundle.currentPrice;
  const semantic = (bundle.polymarket?.selectedMarkets ?? []).map((market) =>
    labelReplaySemanticMarket(bundle, market, history, labeledAt),
  );

  return {
    ...bundle,
    labels: {
      ...(bundle.labels ?? {}),
      forecast: {
        realizedPrice: forecastSettlement.price,
        realizedReturn,
        actualBinary: realizedReturn > 0 ? 1 : 0,
        labeledAt,
      },
      semantic,
    },
  };
}
