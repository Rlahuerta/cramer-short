import { extractTickers as extractTickersFn } from '../../memory/ticker-extractor.js';
import { resolveAssetIntent } from '../../tools/finance/asset-resolver.js';
import { detectAssetType } from '../../tools/finance/signal-extractor.js';
import { resolveForecastLabMarkovParameterDefaults } from '../../tools/finance/markov-distribution.js';
import {
  isCryptoForecastQuery,
  isDistributionQuery,
  isExplicitTerminalDistributionQuery,
  inferTrajectoryRequest,
  isForecastLabImprovementQuery,
  isExplicitGoldCombinedMarkovPolymarketRequest,
} from './classification.js';
import type { ToolCallRecord } from '../scratchpad.js';

const TRADING_DAYS_PER_WEEK = 5;
const TRADING_DAYS_PER_MONTH = 21;
const TRADING_DAYS_PER_QUARTER = TRADING_DAYS_PER_MONTH * 3;
const QUARTER_END_MONTHS: Record<1 | 2 | 3 | 4, number> = {
  1: 3,
  2: 6,
  3: 9,
  4: 12,
};

export function getBtcSelectiveMarkovConfidenceThreshold(): number {
  return resolveForecastLabMarkovParameterDefaults('btc').recommendedConfidenceThreshold;
}

export function inferDistributionTicker(query: string): string | null {
  const resolved = resolveAssetIntent(query, extractTickersFn(query)[0] ?? null);
  if (resolved.resolvedTicker) {
    if (resolved.assetClass === 'ticker' && /^[A-Z]{2,5}$/.test(resolved.resolvedTicker)) {
      const detected = detectAssetType(query);
      if (detected.type === 'crypto') return `${resolved.resolvedTicker}-USD`;
    }
    return resolved.resolvedTicker;
  }

  const extracted = extractTickersFn(query);
  if (extracted.length > 0) {
    const first = extracted[0]!;
    if (/^[A-Z]{2,5}$/.test(first)) {
      const detected = detectAssetType(query);
      if (detected.type === 'crypto') return `${first}-USD`;
    }
    return first;
  }

  const detected = detectAssetType(query);
  if (detected.type === 'crypto' && detected.ticker) return `${detected.ticker}-USD`;
  return detected.ticker;
}

function startOfUtcDay(date: Date): Date {
  return new Date(Date.UTC(date.getUTCFullYear(), date.getUTCMonth(), date.getUTCDate()));
}

function countWeekdaysUntil(targetDate: Date, referenceDate: Date): number | null {
  const start = startOfUtcDay(referenceDate);
  const target = startOfUtcDay(targetDate);
  if (target <= start) return null;

  let count = 0;
  const cursor = new Date(start);
  while (cursor < target) {
    cursor.setUTCDate(cursor.getUTCDate() + 1);
    const weekday = cursor.getUTCDay();
    if (weekday !== 0 && weekday !== 6) count++;
  }

  return count > 0 ? count : null;
}

function inferQuarterEndTradingDays(query: string, referenceDate: Date): number | null {
  const quarterEndMatch = query.match(/\b(?:by\s+)?end\s+of\s+q([1-4])(?:\s+(\d{4}))?\b/i)
    ?? query.match(/\bthrough\s+q([1-4])(?:\s+(\d{4}))?\b/i);
  if (!quarterEndMatch) return null;

  const quarter = parseInt(quarterEndMatch[1]!, 10) as 1 | 2 | 3 | 4;
  const hasExplicitYear = Boolean(quarterEndMatch[2]);
  let year = quarterEndMatch[2]
    ? parseInt(quarterEndMatch[2]!, 10)
    : referenceDate.getUTCFullYear();

  const buildQuarterEndDate = (targetYear: number) =>
    new Date(Date.UTC(targetYear, QUARTER_END_MONTHS[quarter], 0));

  const start = startOfUtcDay(referenceDate);
  let targetDate = buildQuarterEndDate(year);
  if (!hasExplicitYear && targetDate < start) {
    year += 1;
    targetDate = buildQuarterEndDate(year);
  }

  if (targetDate.getTime() === start.getTime()) return 1;
  if (targetDate < start) return null;

  return countWeekdaysUntil(targetDate, referenceDate);
}

export function inferDistributionHorizon(query: string, referenceDate: Date = new Date()): number | null {
  const lower = query.toLowerCase();

  const tradingDaysMatch = lower.match(/(\d+)\s+trading\s+days?/i);
  if (tradingDaysMatch) return parseInt(tradingDaysMatch[1]!, 10);

  const dayMatch = lower.match(/(\d+)[-\s]days?\b/i);
  if (dayMatch) return parseInt(dayMatch[1]!, 10);

  const hourMatch = lower.match(/(\d+)[-\s]?(?:hours?|hrs?|h)\b/i);
  if (hourMatch) {
    const hours = parseInt(hourMatch[1]!, 10);
    if (Number.isFinite(hours) && hours > 0) {
      return Math.max(1, Math.ceil(hours / 24));
    }
  }

  const weekMatch = lower.match(/(\d+)[-\s]weeks?\b/i);
  if (weekMatch) return parseInt(weekMatch[1]!, 10) * TRADING_DAYS_PER_WEEK;

  const monthMatch = lower.match(/(\d+)[-\s]months?\b/i);
  if (monthMatch) return parseInt(monthMatch[1]!, 10) * TRADING_DAYS_PER_MONTH;

  const quarterMatch = lower.match(/(\d+)[-\s]quarters?\b/i);
  if (quarterMatch) return parseInt(quarterMatch[1]!, 10) * TRADING_DAYS_PER_QUARTER;

  if (/\bnext\s+month\b/i.test(lower)) return TRADING_DAYS_PER_MONTH;
  if (/\bnext\s+quarter\b/i.test(lower)) return TRADING_DAYS_PER_QUARTER;

  const quarterEndTradingDays = inferQuarterEndTradingDays(query, referenceDate);
  if (quarterEndTradingDays !== null) return quarterEndTradingDays;

  return null;
}

export function inferBtcShortHorizonForecastHorizon(query: string): number | null {
  const ticker = inferDistributionTicker(query);
  if (ticker !== 'BTC' && ticker !== 'BTC-USD') return null;

  const horizon = inferDistributionHorizon(query);
  if (horizon !== null) return horizon;
  if (/\bnext\s+week\b/i.test(query)) return TRADING_DAYS_PER_WEEK;

  return null;
}

export function inferMarkovQueryHorizon(query: string): number | null {
  return inferDistributionHorizon(query) ?? inferBtcShortHorizonForecastHorizon(query);
}

export function isBtcShortHorizonForecastQuery(query: string): boolean {
  if (!isCryptoForecastQuery(query)) return false;
  const horizon = inferBtcShortHorizonForecastHorizon(query);
  return horizon !== null && horizon <= 14;
}

export function buildForcedMarkovArgs(query: string): { ticker: string; horizon: number; trajectory?: true; trajectoryDays?: number } | null {
  const ticker = inferDistributionTicker(query);
  const horizon = inferDistributionHorizon(query);
  if (!ticker || !horizon) return null;

  const args: { ticker: string; horizon: number; trajectory?: true; trajectoryDays?: number } = { ticker, horizon };
  if (inferTrajectoryRequest(query)) {
    args.trajectory = true;
    args.trajectoryDays = Math.min(30, horizon);
  }

  return args;
}

export function shouldForceMarkovDistribution(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (isForecastLabImprovementQuery(query)) return false;

  const shouldForceForQuery = isExplicitTerminalDistributionQuery(query)
    || inferTrajectoryRequest(query)
    || (isExplicitGoldCombinedMarkovPolymarketRequest(query) && buildForcedMarkovArgs(query) !== null);

  return shouldForceForQuery && !toolCalls.some((call) => call.tool === 'markov_distribution');
}

export function shouldInjectBtcShortHorizonMixedEvidencePrompt(
  query: string,
  fullToolResults: string,
): boolean {
  if (shouldInjectBtcShortHorizonLowConfidencePrompt(query, fullToolResults)) return false;

  const ticker = inferDistributionTicker(query);
  const horizon = inferDistributionHorizon(query);
  if ((ticker !== 'BTC' && ticker !== 'BTC-USD') || horizon === null || horizon > 14) return false;

  const markovBullish = /"_tool"\s*:\s*"markov_distribution"[\s\S]*?"status"\s*:\s*"ok"[\s\S]*?"expectedReturn"\s*:\s*(0\.0*[1-9]\d*|0\.[1-9]\d*|[1-9]\d*(?:\.\d+)?)/.test(fullToolResults);
  const polymarketFlatOrBearish = /forecast return\s*:\s*-\d+(?:\.\d+)?%/i.test(fullToolResults)
    || /forecast return\s*:\s*[+]?(?:0|0\.0+)%/i.test(fullToolResults);

  return markovBullish && polymarketFlatOrBearish;
}

export function shouldInjectBtcShortHorizonLowConfidencePrompt(
  query: string,
  fullToolResults: string,
): boolean {
  if (!isBtcShortHorizonForecastQuery(query)) return false;

  const match = fullToolResults.match(/"_tool"\s*:\s*"markov_distribution"[\s\S]*?"status"\s*:\s*"ok"[\s\S]*?"predictionConfidence"\s*:\s*([0-9]+(?:\.[0-9]+)?)/);
  if (!match) return false;

  const predictionConfidence = Number.parseFloat(match[1]!);
  return Number.isFinite(predictionConfidence) && predictionConfidence < getBtcSelectiveMarkovConfidenceThreshold();
}
