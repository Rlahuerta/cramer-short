import { routeForecastLabQuery } from '../experiments/forecast-lab/router.js';
import { extractTickers as extractTickersFn } from '../memory/ticker-extractor.js';
import { resolveAssetIntent } from '../tools/finance/asset-resolver.js';
import { resolveForecastLabMarkovParameterDefaults } from '../tools/finance/markov-distribution.js';
import { detectAssetType } from '../tools/finance/signal-extractor.js';
import { discoverSkills } from '../skills/registry.js';
import type { ToolCallRecord } from './scratchpad.js';

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

export function isForecastLabImprovementQuery(query: string): boolean {
  return routeForecastLabQuery(query).intent === 'improvement';
}

export function isDistributionQuery(query: string): boolean {
  return /probability distribution|price distribution|full distribution|distribution for .*price/i.test(query);
}

export function isExplicitTerminalDistributionQuery(query: string): boolean {
  const lower = query.toLowerCase();
  return isDistributionQuery(query)
    || lower.includes('markov distribution')
    || lower.includes('markov chain')
    || lower.includes('terminal threshold');
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

export function isCryptoForecastQuery(query: string): boolean {
  if (isForecastLabImprovementQuery(query)) return false;
  if (isExplicitTerminalDistributionQuery(query)) return false;

  if (/\buse the\s+probability_assessment\s+skill\b/i.test(query)) return false;

  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return false;

  const lower = query.toLowerCase();
  const hasForecastLanguage = /\bforecast\b|\bpredict(?:ion)?\b|\boutlook\b|\bprice target\b|where .* headed|what will .* trade|how .* move/.test(lower);
  const hasFutureHorizon = /over the next\s+\d+\s*(?:day|days|week|weeks)\b|next\s+\d+\s*(?:day|days|week|weeks)\b|in\s+\d+\s*(?:day|days|week|weeks)\b/.test(lower);

  return hasForecastLanguage || hasFutureHorizon;
}

export function isBtcShortHorizonForecastQuery(query: string): boolean {
  if (!isCryptoForecastQuery(query)) return false;
  const horizon = inferBtcShortHorizonForecastHorizon(query);
  return horizon !== null && horizon <= 14;
}

export function isExplicitPolymarketForecastRequest(query: string): boolean {
  const lower = query.toLowerCase();
  const hasPolymarketMention = /\bpolymarket\b/.test(lower) || lower.includes('polymarket_forecast');
  const hasForecastIntent = /\b(?:forecast|prediction|predict|price target)\b/.test(lower);
  return lower.includes('polymarket_forecast')
    || /\bpolymarket forecast\b/.test(lower)
    || /\buse\s+(?:the\s+)?polymarket(?:_forecast| forecast)\b/.test(lower)
    || /\brun\s+(?:the\s+)?polymarket(?:_forecast| forecast)\b/.test(lower)
    || (hasPolymarketMention && hasForecastIntent);
}

export function isExplicitCombinedMarkovPolymarketRequest(query: string): boolean {
  const lower = query.toLowerCase();
  const hasMarkovMention = /\bmarkov(?: chain| distribution)?\b/.test(lower);
  const hasPolymarketMention = /\bpolymarket\b/.test(lower) || lower.includes('polymarket_forecast');
  const hasForecastIntent = /\b(?:forecast|prediction|predict|price target|price outlook|outlook)\b/.test(lower);
  return hasMarkovMention && hasPolymarketMention && hasForecastIntent;
}

export function isExplicitGoldCombinedMarkovPolymarketRequest(query: string): boolean {
  const resolved = resolveAssetIntent(query, extractTickersFn(query)[0] ?? null);
  return resolved.assetClass === 'commodity_gold' && isExplicitCombinedMarkovPolymarketRequest(query);
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

/**
 * Detect whether the user explicitly requested a day-by-day trajectory
 * (as opposed to a single-horizon snapshot).
 */
export function inferTrajectoryRequest(query: string): boolean {
  const lower = query.toLowerCase();
  return /trajectory|day[- ]by[- ]day|day by day|price path|daily forecast|daily projection/i.test(lower);
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
  return (isExplicitTerminalDistributionQuery(query) || inferTrajectoryRequest(query))
    && !toolCalls.some((call) => call.tool === 'markov_distribution');
}

const FORECAST_LAB_PLAN_ONLY_PATTERNS = [
  /\bdo not edit files\b/i,
  /\bdon't edit files\b/i,
  /\bdo not run shell commands\b/i,
  /\bdon't run shell commands\b/i,
  /\bdo not write artifacts\b/i,
  /\bdon't write artifacts\b/i,
  /\bexplain the exact experiment plan\b/i,
  /\bexplain (?:the )?plan\b/i,
  /\bwhat plan would you follow\b/i,
  /\bwhat would you do\b/i,
] as const;

export function isForecastLabPlanOnlyQuery(query: string): boolean {
  return FORECAST_LAB_PLAN_ONLY_PATTERNS.some((pattern) => pattern.test(query));
}

/**
 * Detect non-crypto asset forecast-like queries (stocks, ETFs, commodities).
 * Returns true when the query asks about future price/forecast/outlook for a
 * non-crypto asset — exactly the cases where Markov abstention should trigger
 * a forced get_market_data + polymarket_forecast fallback.
 */
export function isNonCryptoForecastQuery(query: string): boolean {
  if (isForecastLabImprovementQuery(query)) return false;
  if (/\buse the\s+probability_assessment\s+skill\b/i.test(query)) return false;

  // Exclude crypto — that path has its own dedicated forcing
  const detected = detectAssetType(query);
  if (detected.type === 'crypto') return false;
  if (!detected.ticker) return false;

  const lower = query.toLowerCase();
  const isHardDistributionQuery = isDistributionQuery(query)
    || lower.includes('markov distribution')
    || lower.includes('terminal threshold');
  if (isHardDistributionQuery) return false;

  const hasForecastLanguage = /\bforecast\b|\bpredict(?:ion)?\b|\boutlook\b|\bprice target\b|where .* headed|what will .* trade|how .* move|price of|will .* (?:beat|hit|reach|drop|rise|fall|exceed|go above|go below)/.test(lower);
  const hasFutureHorizon = /over the next\s+\d+\s*(?:day|days|week|weeks|month|months|quarter|quarters)\b|next\s+\d+\s*(?:day|days|week|weeks|month|months|quarter|quarters)\b|in\s+\d+\s*(?:day|days|week|weeks|month|months|quarter|quarters)\b|\bnext\s+month\b|\bnext\s+quarter\b|\b(?:by\s+)?end\s+of\s+q[1-4](?:\s+\d{4})?\b|\bthrough\s+q[1-4](?:\s+\d{4})?\b/.test(lower);

  return hasForecastLanguage || hasFutureHorizon;
}

export function detectExplicitSkillRequest(query: string): string | null {
  const match = query.match(/\buse (?:the )?([a-z0-9_-]+) skill\b/i);
  if (!match) return null;

  const requested = match[1].toLowerCase();
  const skill = discoverSkills().find((entry) => entry.name.toLowerCase() === requested);
  return skill?.name ?? null;
}

type ForcedNonCryptoPolymarketForecastArgs = {
  ticker: string;
  horizon_days: number;
  current_price?: number;
  markov_return?: number;
};

type ForecastCoverageArgs = {
  ticker: string;
  horizon_days?: number;
  current_price?: number;
  sentiment_score?: number;
  markov_return?: number;
};

type ForcedForecastArbiterArgs = {
  ticker: string;
  horizon_days: number;
  current_price?: number;
  leverage?: number;
  markov?: {
    forecast_return?: number;
    p_up?: number;
    confidence?: number;
    structural_break?: boolean;
    flat_probability?: number;
    ci_low?: number;
    ci_high?: number;
    trusted_anchors?: number;
    total_anchors?: number;
    anchor_quality?: string;
    conformal?: {
      applied?: boolean;
      radius?: number;
      coverageEstimate?: number | null;
      mode?: 'normal' | 'break';
    };
    summary?: string;
  };
  polymarket?: {
    forecast_return?: number;
    quality_score?: number;
    markets?: Array<{ question: string; probability?: number }>;
    summary?: string;
  };
  whale?: {
    direction?: 'long' | 'short' | 'neutral';
    confidence?: number;
    summary?: string;
  };
};

/**
 * Returns true when a non-crypto forecast query has had markov_distribution
 * abstain and is still missing get_market_data and/or
 * polymarket_forecast — the two tools that should be forced as a fallback.
 */
export function shouldForceNonCryptoForecastFallback(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isNonCryptoForecastQuery(query)) return false;

  // If Markov already produced a non-abstain result, don't force fallback.
  if (hasSuccessfulMarkovDistributionForQuery(query, toolCalls)) return false;

  // This fallback is only for the post-abstain path. If Markov never ran,
  // let the normal model/tool flow decide whether to invoke it first.
  const hasAbstainingMarkov = hasAbstainingMarkovDistributionForQuery(query, toolCalls);
  if (!hasAbstainingMarkov) return false;

  const marketDataArgs = buildForcedNonCryptoMarketDataArgs(query);
  const forecastArgs = buildForcedNonCryptoPolymarketForecastArgs(query, toolCalls);
  const currentPrice = extractCurrentPriceForNonCryptoQuery(query, toolCalls);
  const marketDataAttempted = marketDataArgs !== null && hasMarketDataQuery(toolCalls, marketDataArgs.query);

  const needsCurrentPriceFetch = marketDataArgs !== null && currentPrice === null && !marketDataAttempted;

  const needsForecastRun = forecastArgs !== null
    && !hasPolymarketForecastCoverage(toolCalls, forecastArgs);

  return needsCurrentPriceFetch || needsForecastRun;
}

function parseToolCallData(call: ToolCallRecord): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(call.result) as { data?: unknown };
    return parsed?.data && typeof parsed.data === 'object'
      ? parsed.data as Record<string, unknown>
      : null;
  } catch {
    return null;
  }
}

function hasErrorLikeToolResult(result: string): boolean {
  return /^Error:/i.test(result) || /"error"\s*:/i.test(result);
}

function hasNonEmptyParsedToolData(call: ToolCallRecord): boolean {
  const data = parseToolCallData(call);
  return data !== null && Object.keys(data).length > 0;
}

function extractPositiveNumericValue(value: unknown): number | null {
  if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
    return value;
  }

  if (typeof value === 'string' && value.trim().length > 0) {
    const parsed = Number.parseFloat(value);
    if (Number.isFinite(parsed) && parsed > 0) {
      return parsed;
    }
  }

  return null;
}

function extractPriceFromPayload(value: unknown): number | null {
  if (!value || typeof value !== 'object') return null;
  const record = value as Record<string, unknown>;

  const directPrice = extractPositiveNumericValue(record['price']);
  if (directPrice !== null) {
    return directPrice;
  }

  const closePrice = extractPositiveNumericValue(record['close']);
  if (closePrice !== null) {
    return closePrice;
  }

  const lastTradePrice = extractPositiveNumericValue(record['lastTradePrice']);
  if (lastTradePrice !== null) {
    return lastTradePrice;
  }

  const snapshot = record['snapshot'];
  if (snapshot && typeof snapshot === 'object') {
    const snapshotRecord = snapshot as Record<string, unknown>;
    const snapshotPrice = extractPositiveNumericValue(snapshotRecord['price'])
      ?? extractPositiveNumericValue(snapshotRecord['close'])
      ?? extractPositiveNumericValue(snapshotRecord['lastTradePrice']);
    if (snapshotPrice !== null) {
      return snapshotPrice;
    }
  }

  return null;
}

export function hasMarketDataQuery(toolCalls: ToolCallRecord[], query: string): boolean {
  return toolCalls.some((call) => call.tool === 'get_market_data' && call.args['query'] === query);
}

export function extractCurrentPriceFromMarketDataQuery(toolCalls: ToolCallRecord[], query: string): number | null {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'get_market_data' || call.args['query'] !== query) continue;

    const data = parseToolCallData(call);
    if (!data) continue;

    const direct = extractPriceFromPayload(data);
    if (direct !== null) return direct;

    for (const [key, value] of Object.entries(data)) {
      if (!key.startsWith('get_crypto_price_snapshot_') && !key.startsWith('get_stock_price_')) continue;
      const extracted = extractPriceFromPayload(value);
      if (extracted !== null) return extracted;
    }
  }

  return null;
}

function extractCurrentPriceFromAbstainingMarkovQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferDistributionHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution' || data['status'] !== 'abstain') continue;

    const canonical = data['canonical'];
    if (!canonical || typeof canonical !== 'object') continue;

    const currentPrice = extractPositiveNumericValue((canonical as Record<string, unknown>)['currentPrice']);
    if (currentPrice !== null) {
      return currentPrice;
    }
  }

  return null;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function isFinitePositiveNumber(value: unknown): value is number {
  return isFiniteNumber(value) && value > 0;
}

function numbersApproximatelyMatch(actual: unknown, expected: number): boolean {
  if (!isFiniteNumber(actual)) return false;
  const tolerance = Math.max(1e-6, Math.abs(expected) * 1e-6);
  return Math.abs(actual - expected) <= tolerance;
}

export function extractCurrentPriceFromToolCalls(toolCalls: ToolCallRecord[]): number | null {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'get_market_data') continue;

    const data = parseToolCallData(call);
    if (!data) continue;

    const direct = extractPriceFromPayload(data);
    if (direct !== null) return direct;

    for (const [key, value] of Object.entries(data)) {
      if (!key.startsWith('get_crypto_price_snapshot_') && !key.startsWith('get_stock_price_')) continue;
      const extracted = extractPriceFromPayload(value);
      if (extracted !== null) return extracted;
    }
  }

  return null;
}

function extractCurrentPriceForCryptoQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const marketDataArgs = buildForcedMarketDataArgs(query);
  return marketDataArgs
    ? extractCurrentPriceFromMarketDataQuery(toolCalls, marketDataArgs.query)
    : null;
}

function extractCurrentPriceForNonCryptoQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const marketDataArgs = buildForcedNonCryptoMarketDataArgs(query);
  if (!marketDataArgs) return null;

  const marketDataPrice = extractCurrentPriceFromMarketDataQuery(toolCalls, marketDataArgs.query);
  if (marketDataPrice !== null) return marketDataPrice;

  if (!hasMarketDataQuery(toolCalls, marketDataArgs.query)) return null;

  return extractCurrentPriceFromAbstainingMarkovQuery(query, toolCalls);
}

export function extractSentimentScoreFromToolCalls(toolCalls: ToolCallRecord[]): number | null {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'social_sentiment') continue;

    const data = parseToolCallData(call);
    const report = data?.['result'];
    if (typeof report !== 'string') continue;

    const match = report.match(/score\s*([+-]?\d+)\/100/i);
    if (!match) continue;

    const parsedScore = parseInt(match[1]!, 10) / 100;
    if (Number.isFinite(parsedScore)) {
      return Math.max(-1, Math.min(1, parsedScore));
    }
  }

  return null;
}

function extractSentimentScoreForCryptoQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const desired = buildForcedSocialSentimentArgs(query);
  if (!desired) return null;

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'social_sentiment') continue;
    if (call.args['ticker'] !== desired.ticker) continue;

    const data = parseToolCallData(call);
    const report = data?.['result'];
    if (typeof report !== 'string') continue;

    const match = report.match(/score\s*([+-]?\d+)\/100/i);
    if (!match) continue;

    const parsedScore = parseInt(match[1]!, 10) / 100;
    if (Number.isFinite(parsedScore)) {
      return Math.max(-1, Math.min(1, parsedScore));
    }
  }

  return null;
}

export function extractMarkovReturnFromToolCalls(toolCalls: ToolCallRecord[]): number | null {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution' || data['status'] !== 'ok') continue;

    const canonical = data['canonical'];
    if (!canonical || typeof canonical !== 'object') continue;

    const actionSignal = (canonical as Record<string, unknown>)['actionSignal'];
    const diagnostics = (canonical as Record<string, unknown>)['diagnostics'];
    if (!actionSignal || typeof actionSignal !== 'object' || !diagnostics || typeof diagnostics !== 'object') continue;

    const expectedReturn = (actionSignal as Record<string, unknown>)['expectedReturn'];
    const markovWeight = (diagnostics as Record<string, unknown>)['markovWeight'];
    if (
      typeof expectedReturn === 'number' && Number.isFinite(expectedReturn)
      && typeof markovWeight === 'number' && Number.isFinite(markovWeight)
    ) {
      return expectedReturn * markovWeight;
    }
  }

  return null;
}

function extractMarkovReturnForQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferMarkovQueryHorizon(query);
  const requiresSelectiveBtcGate = isBtcShortHorizonForecastQuery(query);
  const selectiveBtcThreshold = requiresSelectiveBtcGate
    ? getBtcSelectiveMarkovConfidenceThreshold()
    : null;

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution') continue;

    if (data['status'] === 'ok') {
      const canonical = data['canonical'];
      if (!canonical || typeof canonical !== 'object') continue;

      const actionSignal = (canonical as Record<string, unknown>)['actionSignal'];
      const diagnostics = (canonical as Record<string, unknown>)['diagnostics'];
      if (!actionSignal || typeof actionSignal !== 'object' || !diagnostics || typeof diagnostics !== 'object') continue;

      const predictionConfidence = (diagnostics as Record<string, unknown>)['predictionConfidence'];
      if (
        requiresSelectiveBtcGate
        && isFiniteNumber(predictionConfidence)
        && predictionConfidence < selectiveBtcThreshold!
      ) {
        return null;
      }

      const expectedReturn = (actionSignal as Record<string, unknown>)['expectedReturn'];
      const markovWeight = (diagnostics as Record<string, unknown>)['markovWeight'];
      if (
        typeof expectedReturn === 'number' && Number.isFinite(expectedReturn)
        && typeof markovWeight === 'number' && Number.isFinite(markovWeight)
      ) {
        return expectedReturn * markovWeight;
      }
    }
  }

  return null;
}

export function extractMarkovPredictionConfidenceForQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferMarkovQueryHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution') continue;

    if (data['status'] !== 'ok') continue;

    const canonical = data['canonical'];
    if (!canonical || typeof canonical !== 'object') continue;

    const diagnostics = (canonical as Record<string, unknown>)['diagnostics'];
    if (!diagnostics || typeof diagnostics !== 'object') continue;

    const predictionConfidence = (diagnostics as Record<string, unknown>)['predictionConfidence'];
    if (isFiniteNumber(predictionConfidence)) return predictionConfidence;
  }

  return null;
}

function extractMarkovArbiterEvidence(query: string, toolCalls: ToolCallRecord[]): ForcedForecastArbiterArgs['markov'] | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferMarkovQueryHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution') continue;

    if (data['status'] === 'abstain') {
      const evidence: NonNullable<ForcedForecastArbiterArgs['markov']> = {};
      const forecastHint = data['forecastHint'];
      if (forecastHint && typeof forecastHint === 'object') {
        const hintRecord = forecastHint as Record<string, unknown>;
        if (isFiniteNumber(hintRecord['markovReturn'])) evidence.forecast_return = hintRecord['markovReturn'];
        if (isFiniteNumber(hintRecord['confidenceScore'])) evidence.confidence = hintRecord['confidenceScore'];
      }
      const reasons = Array.isArray(data['abstainReasons'])
        ? data['abstainReasons'].filter((reason): reason is string => typeof reason === 'string' && reason.trim().length > 0)
        : [];
      const canonical = data['canonical'] && typeof data['canonical'] === 'object'
        ? data['canonical'] as Record<string, unknown>
        : null;
      const diagnostics = canonical?.['diagnostics'] && typeof canonical['diagnostics'] === 'object'
        ? canonical['diagnostics'] as Record<string, unknown>
        : null;
      if (diagnostics) {
        if (typeof diagnostics['structuralBreakDetected'] === 'boolean') evidence.structural_break = diagnostics['structuralBreakDetected'];
        if (isFiniteNumber(diagnostics['trustedAnchors'])) evidence.trusted_anchors = diagnostics['trustedAnchors'];
        if (isFiniteNumber(diagnostics['totalAnchors'])) evidence.total_anchors = diagnostics['totalAnchors'];
        if (typeof diagnostics['anchorQuality'] === 'string') evidence.anchor_quality = diagnostics['anchorQuality'];
        if (evidence.confidence === undefined && isFiniteNumber(diagnostics['predictionConfidence'])) {
          evidence.confidence = diagnostics['predictionConfidence'];
        }
      }
      const structuralBreakSummary = diagnostics?.['structuralBreakDetected'] === true
        ? [
            'Structural break detected',
            isFiniteNumber(diagnostics['structuralBreakDivergence'])
              ? `divergence ${diagnostics['structuralBreakDivergence'].toFixed(3)}`
              : null,
            diagnostics['ciWidened'] === true ? 'CI widening applied' : null,
          ].filter((part): part is string => part !== null).join(', ')
        : null;
      evidence.summary = [
        reasons.length > 0
          ? `Markov abstained: ${reasons.join('; ')}`
          : 'Markov abstained; treat Markov evidence as diagnostics only.',
        structuralBreakSummary,
      ].filter((part): part is string => Boolean(part)).join(' ');
      return evidence;
    }

    if (data['status'] !== 'ok') continue;

    const canonical = data['canonical'];
    if (!canonical || typeof canonical !== 'object') continue;
    const canonicalRecord = canonical as Record<string, unknown>;
    const scenarios = canonicalRecord['scenarios'];
    const diagnostics = canonicalRecord['diagnostics'];
    const actionSignal = canonicalRecord['actionSignal'];
    const distribution = data['distribution'];

    const evidence: NonNullable<ForcedForecastArbiterArgs['markov']> = {};
    const weightedReturn = extractMarkovReturnForQuery(query, toolCalls);
    if (weightedReturn !== null) evidence.forecast_return = weightedReturn;

    if (scenarios && typeof scenarios === 'object') {
      const scenarioRecord = scenarios as Record<string, unknown>;
      if (isFiniteNumber(scenarioRecord['pUp'])) evidence.p_up = scenarioRecord['pUp'];
      if (isFiniteNumber(scenarioRecord['expectedReturn']) && evidence.forecast_return === undefined) {
        evidence.forecast_return = scenarioRecord['expectedReturn'];
      }
      const buckets = scenarioRecord['buckets'];
      if (Array.isArray(buckets)) {
        const flat = buckets.find((bucket) =>
          bucket && typeof bucket === 'object'
          && typeof (bucket as Record<string, unknown>)['label'] === 'string'
          && ((bucket as Record<string, unknown>)['label'] as string).toLowerCase().includes('flat')
        ) as Record<string, unknown> | undefined;
        if (flat && isFiniteNumber(flat['probability'])) evidence.flat_probability = flat['probability'];
      }
    }

    if (diagnostics && typeof diagnostics === 'object') {
      const diagnosticRecord = diagnostics as Record<string, unknown>;
      if (isFiniteNumber(diagnosticRecord['predictionConfidence'])) evidence.confidence = diagnosticRecord['predictionConfidence'];
      if (typeof diagnosticRecord['structuralBreakDetected'] === 'boolean') evidence.structural_break = diagnosticRecord['structuralBreakDetected'];
      if (isFiniteNumber(diagnosticRecord['trustedAnchors'])) evidence.trusted_anchors = diagnosticRecord['trustedAnchors'];
      if (isFiniteNumber(diagnosticRecord['totalAnchors'])) evidence.total_anchors = diagnosticRecord['totalAnchors'];
      if (typeof diagnosticRecord['anchorQuality'] === 'string') evidence.anchor_quality = diagnosticRecord['anchorQuality'];
      const conformal = diagnosticRecord['conformal'];
      if (conformal && typeof conformal === 'object') {
        const conformalRecord = conformal as Record<string, unknown>;
        const conformed: NonNullable<NonNullable<ForcedForecastArbiterArgs['markov']>['conformal']> = {};
        if (typeof conformalRecord['applied'] === 'boolean') conformed.applied = conformalRecord['applied'];
        if (isFiniteNumber(conformalRecord['radius'])) conformed.radius = conformalRecord['radius'];
        if (conformalRecord['coverageEstimate'] === null || isFiniteNumber(conformalRecord['coverageEstimate'])) {
          conformed.coverageEstimate = conformalRecord['coverageEstimate'] as number | null;
        }
        if (conformalRecord['mode'] === 'normal' || conformalRecord['mode'] === 'break') {
          conformed.mode = conformalRecord['mode'];
        }
        if (Object.keys(conformed).length > 0) evidence.conformal = conformed;
      }
    }

    if (actionSignal && typeof actionSignal === 'object' && typeof (actionSignal as Record<string, unknown>)['confidence'] === 'string') {
      evidence.summary = `Markov action signal confidence ${(actionSignal as Record<string, unknown>)['confidence']}`;
    }

    if (Array.isArray(distribution)) {
      const prices = distribution
        .map((point) => point && typeof point === 'object' ? (point as Record<string, unknown>)['price'] : null)
        .filter((price): price is number => isFinitePositiveNumber(price));
      if (prices.length > 0) {
        evidence.ci_low = Math.min(...prices);
        evidence.ci_high = Math.max(...prices);
      }
    }

    return Object.keys(evidence).length > 0 ? evidence : null;
  }

  return null;
}

export function hasLowConfidenceBtcShortHorizonMarkov(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isBtcShortHorizonForecastQuery(query)) return false;
  const predictionConfidence = extractMarkovPredictionConfidenceForQuery(query, toolCalls);
  return predictionConfidence !== null && predictionConfidence < getBtcSelectiveMarkovConfidenceThreshold();
}

export function buildForcedMarketDataArgs(query: string): { query: string } | null {
  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return null;

  return {
    query: `Current crypto price snapshot for ${detected.ticker}`,
  };
}

export function buildForcedNonCryptoMarketDataArgs(query: string): { query: string } | null {
  if (!isNonCryptoForecastQuery(query)) return null;

  const ticker = inferDistributionTicker(query);
  if (!ticker) return null;

  return {
    query: `${ticker} current price`,
  };
}

export function buildForcedSocialSentimentArgs(query: string): {
  ticker: string;
  include_fear_greed: true;
  limit: number;
} | null {
  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return null;

  return {
    ticker: detected.ticker,
    include_fear_greed: true,
    limit: 25,
  };
}

export function buildForcedPolymarketForecastArgs(query: string, toolCalls: ToolCallRecord[]): {
  ticker: string;
  horizon_days?: number;
  current_price?: number;
  sentiment_score?: number;
  markov_return?: number;
} | null {
  if (!isCryptoForecastQuery(query)) return null;

  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return null;

  const args: {
    ticker: string;
    horizon_days?: number;
    current_price?: number;
    sentiment_score?: number;
    markov_return?: number;
  } = {
    ticker: detected.ticker,
  };

  const horizon = inferDistributionHorizon(query);
  if (horizon) args.horizon_days = horizon;

  const currentPrice = extractCurrentPriceForCryptoQuery(query, toolCalls);
  if (currentPrice !== null) args.current_price = currentPrice;

  const sentimentScore = extractSentimentScoreForCryptoQuery(query, toolCalls);
  if (sentimentScore !== null) args.sentiment_score = sentimentScore;

  const markovReturn = extractMarkovReturnForQuery(query, toolCalls);
  if (markovReturn !== null) args.markov_return = markovReturn;

  return args;
}

export function buildForcedNonCryptoPolymarketForecastArgs(
  query: string,
  toolCalls: ToolCallRecord[],
): ForcedNonCryptoPolymarketForecastArgs | null {
  if (!isNonCryptoForecastQuery(query)) return null;

  const ticker = inferDistributionTicker(query);
  if (!ticker) return null;

  const args: ForcedNonCryptoPolymarketForecastArgs = {
    ticker,
    horizon_days: inferDistributionHorizon(query) ?? 7,
  };

  const currentPrice = extractCurrentPriceForNonCryptoQuery(query, toolCalls);
  if (currentPrice !== null) args.current_price = currentPrice;

  const markovReturn = extractMarkovReturnFromToolCalls(toolCalls);
  if (markovReturn !== null) args.markov_return = markovReturn;

  return args;
}

function getForecastHorizonArg(args: Record<string, unknown>): number {
  return isFinitePositiveNumber(args['horizon_days']) ? Math.trunc(args['horizon_days']) : 7;
}

function getPositiveIntegerArg(args: Record<string, unknown>, key: string): number | null {
  return isFinitePositiveNumber(args[key]) ? Math.trunc(args[key]) : null;
}

function matchesTickerAndOptionalHorizon(
  args: Record<string, unknown>,
  ticker: string | null,
  horizonKey: string,
  horizon: number | null,
): boolean {
  if (ticker) {
    const existingTicker = typeof args['ticker'] === 'string' ? args['ticker'].toUpperCase() : null;
    const expectedTicker = ticker.toUpperCase();
    const equivalentCryptoTicker = expectedTicker.endsWith('-USD')
      && existingTicker === expectedTicker.replace(/-USD$/, '');
    if (existingTicker !== expectedTicker && !equivalentCryptoTicker) return false;
  }

  if (horizon !== null && getPositiveIntegerArg(args, horizonKey) !== horizon) {
    return false;
  }

  return true;
}

function hasMarkovDistributionStatusForQuery(
  query: string,
  toolCalls: ToolCallRecord[],
  status: 'ok' | 'abstain',
): boolean {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferMarkovQueryHorizon(query);

  return toolCalls.some((call) => {
    if (call.tool !== 'markov_distribution') return false;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) return false;

    const data = parseToolCallData(call);
    return data?._tool === 'markov_distribution' && data.status === status;
  });
}

export function hasSuccessfulMarkovDistributionForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  return hasMarkovDistributionStatusForQuery(query, toolCalls, 'ok');
}

export function hasAbstainingMarkovDistributionForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  return hasMarkovDistributionStatusForQuery(query, toolCalls, 'abstain');
}

export function hasCompletedMarkovDistributionForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  return hasSuccessfulMarkovDistributionForQuery(query, toolCalls)
    || hasAbstainingMarkovDistributionForQuery(query, toolCalls);
}

export function hasUsableOnchainResultForCryptoQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  const desired = buildForcedOnchainArgs(query);
  if (!desired) return false;

  return toolCalls.some((call) =>
    call.tool === 'get_onchain_crypto'
    && call.args['ticker'] === desired.ticker
    && !hasErrorLikeToolResult(call.result)
    && hasNonEmptyParsedToolData(call),
  );
}

export function hasUsableFixedIncomeResult(toolCalls: ToolCallRecord[]): boolean {
  const desired = buildForcedFixedIncomeArgs();
  return toolCalls.some((call) =>
    call.tool === 'get_fixed_income'
    && JSON.stringify(call.args['series']) === JSON.stringify(desired.series)
    && !hasErrorLikeToolResult(call.result)
    && hasNonEmptyParsedToolData(call),
  );
}

function hasUsableStructuredToolResult(toolCalls: ToolCallRecord[], toolName: string): boolean {
  return toolCalls.some((call) =>
    call.tool === toolName
    && !hasErrorLikeToolResult(call.result)
    && hasNonEmptyParsedToolData(call),
  );
}

export function hasPolymarketForecastCoverage(
  toolCalls: ToolCallRecord[],
  desired: ForecastCoverageArgs,
): boolean {
  return toolCalls.some((call) => {
    if (call.tool !== 'polymarket_forecast') return false;
    if (hasErrorLikeToolResult(call.result)) return false;

    const existingTicker = typeof call.args['ticker'] === 'string'
      ? call.args['ticker'].toUpperCase()
      : null;
    if (existingTicker !== desired.ticker.toUpperCase()) return false;

    if (desired.horizon_days !== undefined && getForecastHorizonArg(call.args) !== desired.horizon_days) return false;

    if (desired.current_price !== undefined && !numbersApproximatelyMatch(call.args['current_price'], desired.current_price)) {
      return false;
    }

    if (desired.sentiment_score !== undefined && !numbersApproximatelyMatch(call.args['sentiment_score'], desired.sentiment_score)) {
      return false;
    }

    if (desired.markov_return !== undefined && !numbersApproximatelyMatch(call.args['markov_return'], desired.markov_return)) {
      return false;
    }

    return true;
  });
}

export function hasCryptoPolymarketForecastCoverage(query: string, toolCalls: ToolCallRecord[]): boolean {
  const desired = buildForcedPolymarketForecastArgs(query, toolCalls);
  return desired !== null && hasPolymarketForecastCoverage(toolCalls, desired);
}

function hasPolymarketForecastWithMarkovReturn(toolCalls: ToolCallRecord[]): boolean {
  return toolCalls.some((call) => call.tool === 'polymarket_forecast'
    && isFiniteNumber(call.args['markov_return']));
}

export function shouldRerunPolymarketForecastWithMarkov(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isCryptoForecastQuery(query)) return false;
  if (buildForcedCryptoForecastMarkovArgs(query) === null) return false;
  if (!toolCalls.some((call) => call.tool === 'polymarket_forecast')) return false;
  if (extractMarkovReturnForQuery(query, toolCalls) === null) return false;
  return !hasCryptoPolymarketForecastCoverage(query, toolCalls);
}


export function shouldForceGoldCombinedForecastTools(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isExplicitGoldCombinedMarkovPolymarketRequest(query)) return false;
  if (!hasCompletedMarkovDistributionForQuery(query, toolCalls)) return false;

  const marketDataArgs = buildForcedNonCryptoMarketDataArgs(query);
  const forecastArgs = buildForcedNonCryptoPolymarketForecastArgs(query, toolCalls);
  const currentPrice = extractCurrentPriceForNonCryptoQuery(query, toolCalls);
  const marketDataAttempted = marketDataArgs !== null && hasMarketDataQuery(toolCalls, marketDataArgs.query);

  const needsCurrentPriceFetch = marketDataArgs !== null && currentPrice === null && !marketDataAttempted;
  const needsForecastRun = forecastArgs !== null
    && !hasPolymarketForecastCoverage(toolCalls, forecastArgs);

  return needsCurrentPriceFetch || needsForecastRun || shouldForceGoldCombinedForecastArbitrator(query, toolCalls);
}

export function buildForcedOnchainArgs(query: string): { ticker: string; metrics: string[] } | null {
  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return null;

  return {
    ticker: detected.ticker,
    metrics: ['market', 'sentiment'],
  };
}

export function buildForcedFixedIncomeArgs(): { series: string[] } {
  return {
    series: ['treasury_yields', 'yield_curve'],
  };
}

export function buildForcedCryptoForecastMarkovArgs(query: string): {
  ticker: string;
  horizon: number;
  trajectory: true;
  trajectoryDays: number;
} | null {
  if (!isCryptoForecastQuery(query)) return null;

  const ticker = inferDistributionTicker(query);
  let horizon = inferDistributionHorizon(query);

  if (!horizon && ticker === 'BTC-USD') {
    horizon = inferBtcShortHorizonForecastHorizon(query);
  }

  if (!ticker || !horizon || horizon > 14) return null;

  return {
    ticker,
    horizon,
    trajectory: true,
    trajectoryDays: Math.min(30, horizon),
  };
}

export function shouldForceCryptoForecastTools(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isCryptoForecastQuery(query)) return false;

  const marketDataArgs = buildForcedMarketDataArgs(query);
  const hasMarketData = marketDataArgs !== null
    && extractCurrentPriceFromMarketDataQuery(toolCalls, marketDataArgs.query) !== null;
  const hasSocialSentiment = extractSentimentScoreForCryptoQuery(query, toolCalls) !== null;
  const hasPolymarketForecast = hasCryptoPolymarketForecastCoverage(query, toolCalls);
  const hasOnchain = hasUsableOnchainResultForCryptoQuery(query, toolCalls);
  const hasFixedIncome = hasUsableFixedIncomeResult(toolCalls);
  const needsMarkov = buildForcedCryptoForecastMarkovArgs(query) !== null
    && !hasCompletedMarkovDistributionForQuery(query, toolCalls);
  const needsPolymarketRerun = shouldRerunPolymarketForecastWithMarkov(query, toolCalls);
  const needsForecastArbiter = shouldForceForecastArbitrator(query, toolCalls);

  return !hasMarketData || !hasSocialSentiment || !hasPolymarketForecast || !hasOnchain || !hasFixedIncome || needsMarkov || needsPolymarketRerun || needsForecastArbiter;
}

function extractPolymarketForecastReturnForQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferDistributionHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'polymarket_forecast') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon_days', desiredHorizon)) continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(call.result);
    } catch {
      continue;
    }

    const data = parsed && typeof parsed === 'object' ? (parsed as { data?: unknown }).data : null;
    if (!data || typeof data !== 'object') continue;
    const payload = data as Record<string, unknown>;

    const directForecastReturn = payload['forecastReturn'];
    if (typeof directForecastReturn === 'number' && Number.isFinite(directForecastReturn)) {
      return directForecastReturn;
    }

    const resultText = payload['result'];
    if (typeof resultText !== 'string') continue;
    const match = resultText.match(/(?:forecast return|expected return)\s*[:=]?\s*([+-]?\d+(?:\.\d+)?)%/i);
    if (!match) continue;
    const parsedPct = Number.parseFloat(match[1]!);
    if (Number.isFinite(parsedPct)) return parsedPct / 100;
  }

  return null;
}

function extractPolymarketArbiterEvidence(query: string, toolCalls: ToolCallRecord[]): ForcedForecastArbiterArgs['polymarket'] | null {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferDistributionHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'polymarket_forecast') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon_days', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data) continue;

    const evidence: NonNullable<ForcedForecastArbiterArgs['polymarket']> = {};
    const forecastReturn = extractPolymarketForecastReturnForQuery(query, toolCalls);
    if (forecastReturn !== null) evidence.forecast_return = forecastReturn;

    const resultText = data['result'];
    if (typeof resultText === 'string') {
      const scoreMatch = resultText.match(/Grade:\s*[A-Z][^(]*\((\d+)\/100\)/i);
      if (scoreMatch) {
        const score = Number.parseInt(scoreMatch[1]!, 10);
        if (Number.isFinite(score)) evidence.quality_score = score;
      }

      const markets: Array<{ question: string; probability?: number }> = [];
      const marketPattern = /(Will [^:\n|]+?)[:|]\s*(\d{1,3})%\s+YES/gi;
      let marketMatch: RegExpExecArray | null;
      while ((marketMatch = marketPattern.exec(resultText)) !== null && markets.length < 5) {
        const question = marketMatch[1]?.trim();
        const probability = Number.parseInt(marketMatch[2]!, 10) / 100;
        if (question) markets.push({ question, probability });
      }
      if (markets.length > 0) evidence.markets = markets;
      evidence.summary = resultText.slice(0, 600);
    }

    return Object.keys(evidence).length > 0 ? evidence : null;
  }

  return null;
}

function inferLeverageFromQuery(query: string): number | null {
  const match = query.match(/\b(\d{1,3}(?:\.\d+)?)\s*x\b/i);
  if (!match) return null;
  const leverage = Number.parseFloat(match[1]!);
  return Number.isFinite(leverage) && leverage > 0 ? leverage : null;
}

function isTradeDecisionQuery(query: string): boolean {
  return /\b(direction|entry|enter|stop|stop-loss|target|take profit|leverage|leveraged|\d{1,3}(?:\.\d+)?\s*x|long|short|trade setup|trade plan|position|arbitrator|verdict)\b/i.test(query);
}

export function hasForecastArbitratorForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferDistributionHorizon(query);
  return toolCalls.some((call) => {
    if (call.tool !== 'forecast_arbitrator') return false;
    if (hasErrorLikeToolResult(call.result)) return false;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon_days', desiredHorizon)) return false;
    return hasNonEmptyParsedToolData(call);
  });
}

export function buildForcedForecastArbiterArgs(query: string, toolCalls: ToolCallRecord[]): ForcedForecastArbiterArgs | null {
  if (!isCryptoForecastQuery(query)) return null;
  if (!isTradeDecisionQuery(query) && !detectBtcShortHorizonDisagreement(query, toolCalls)) return null;

  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return null;

  const horizon = inferDistributionHorizon(query) ?? inferBtcShortHorizonForecastHorizon(query) ?? 1;
  const markov = extractMarkovArbiterEvidence(query, toolCalls);
  const polymarket = extractPolymarketArbiterEvidence(query, toolCalls);
  if (!markov && !polymarket) return null;

  const args: ForcedForecastArbiterArgs = {
    ticker: detected.ticker,
    horizon_days: horizon,
    whale: {
      direction: 'neutral',
      confidence: 0.35,
      summary: hasUsableOnchainResultForCryptoQuery(query, toolCalls)
        ? 'On-chain/whale tool completed; treat as neutral unless the final synthesis has a stronger confirmed whale signal.'
        : 'No confirmed whale/on-chain signal available.',
    },
  };

  if (markov) args.markov = markov;
  if (polymarket) args.polymarket = polymarket;

  const currentPrice = extractCurrentPriceForCryptoQuery(query, toolCalls);
  if (currentPrice !== null) args.current_price = currentPrice;

  const leverage = inferLeverageFromQuery(query);
  if (leverage !== null) args.leverage = leverage;

  return args;
}

export function shouldForceForecastArbitrator(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (hasForecastArbitratorForQuery(query, toolCalls)) return false;
  return buildForcedForecastArbiterArgs(query, toolCalls) !== null;
}

const GOLD_COMBINED_MATERIAL_DISAGREEMENT_THRESHOLD = 0.01;

function detectGoldCombinedForecastDisagreement(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isExplicitGoldCombinedMarkovPolymarketRequest(query)) return false;

  const markovReturn = extractMarkovReturnForQuery(query, toolCalls);
  const polymarketReturn = extractPolymarketForecastReturnForQuery(query, toolCalls);
  if (markovReturn === null || polymarketReturn === null) return false;

  const oppositeDirections =
    (markovReturn >= 0.001 && polymarketReturn <= -0.001)
    || (markovReturn <= -0.001 && polymarketReturn >= 0.001);

  return oppositeDirections
    && Math.abs(markovReturn - polymarketReturn) >= GOLD_COMBINED_MATERIAL_DISAGREEMENT_THRESHOLD;
}

export function buildForcedGoldCombinedForecastArbiterArgs(
  query: string,
  toolCalls: ToolCallRecord[],
): ForcedForecastArbiterArgs | null {
  const ticker = inferDistributionTicker(query);
  if (!ticker) return null;

  const horizon = inferDistributionHorizon(query) ?? 7;
  const markov = extractMarkovArbiterEvidence(query, toolCalls);
  const polymarket = extractPolymarketArbiterEvidence(query, toolCalls);
  if (!markov || !polymarket) return null;
  const hasDirectionalDisagreement = detectGoldCombinedForecastDisagreement(query, toolCalls);
  const hasStructuralBreakDiagnostics = markov.structural_break === true;
  if (!hasDirectionalDisagreement && !hasStructuralBreakDiagnostics) return null;

  const args: ForcedForecastArbiterArgs = {
    ticker,
    horizon_days: horizon,
    markov,
    polymarket,
  };

  const currentPrice = extractCurrentPriceForNonCryptoQuery(query, toolCalls);
  if (currentPrice !== null) args.current_price = currentPrice;

  const leverage = inferLeverageFromQuery(query);
  if (leverage !== null) args.leverage = leverage;

  return args;
}

export function shouldForceGoldCombinedForecastArbitrator(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (hasForecastArbitratorForQuery(query, toolCalls)) return false;
  return buildForcedGoldCombinedForecastArbiterArgs(query, toolCalls) !== null;
}

export function detectBtcShortHorizonDisagreement(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isCryptoForecastQuery(query)) return false;
  const ticker = inferDistributionTicker(query);
  const horizon = inferDistributionHorizon(query);
  if ((ticker !== 'BTC' && ticker !== 'BTC-USD') || horizon === null || horizon > 14) return false;

  const markovReturn = extractMarkovReturnForQuery(query, toolCalls);
  const polymarketReturn = extractPolymarketForecastReturnForQuery(query, toolCalls);
  if (markovReturn === null || polymarketReturn === null) return false;

  return markovReturn > 0.01 && polymarketReturn <= 0;
}
