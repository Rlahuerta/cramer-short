import { AIMessage } from '@langchain/core/messages';
import { StructuredToolInterface } from '@langchain/core/tools';
import { callLlm, streamCallLlm } from '../model/llm.js';
import { getSetting } from '../utils/config.js';
import { getTools } from '../tools/registry.js';
import { buildSystemPrompt, buildIterationPrompt, loadSoulDocument } from './prompts.js';
import { extractTextContent, hasToolCalls, extractReasoningContent } from '../utils/ai-message.js';
import { InMemoryChatHistory } from '../utils/in-memory-chat-history.js';
import { buildHistoryContext } from '../utils/history-context.js';
import { estimateTokens, CONTEXT_THRESHOLD, KEEP_TOOL_USES, getContextThreshold, getKeepToolUses } from '../utils/tokens.js';
import { formatUserFacingError, isContextOverflowError } from '../utils/errors.js';
import type { AgentConfig, AgentEvent, AnswerStartEvent, AnswerChunkEvent, ContextClearedEvent, ProgressEvent, TokenUsage } from '../agent/types.js';
import { createRunContext, type RunContext } from './run-context.js';
import { AgentToolExecutor } from './tool-executor.js';
import { MemoryManager } from '../memory/index.js';
import { runMemoryFlush, shouldRunMemoryFlush } from '../memory/flush.js';
import { injectMemoryContext } from './memory-injection.js';
import { extractTickers as extractTickersFn } from '../memory/ticker-extractor.js';
import { injectPolymarketContext } from '../tools/finance/polymarket-injector.js';
import { detectAssetType, extractSignals as extractSignalsFn } from '../tools/finance/signal-extractor.js';
import { resolveAssetIntent } from '../tools/finance/asset-resolver.js';
import { fetchPolymarketMarkets } from '../tools/finance/polymarket.js';
import { resolveProvider } from '../providers.js';
import type { ToolCallRecord } from './scratchpad.js';

/** Matches the timeout used in llm.ts — configurable via the same env var. */
const LLM_CALL_TIMEOUT_MS = parseInt(process.env.LLM_CALL_TIMEOUT_MS ?? '120000', 10);


const DEFAULT_MODEL = 'gpt-5.4';
export const DEFAULT_MAX_ITERATIONS = 25;

/**
 * Remove <think>...</think> blocks that Ollama thinking models sometimes embed
 * directly in response text rather than separating into reasoning_content.
 * Also handles orphan </think> tags (e.g. the model output was: <think>…</think>\nAnswer).
 */
export function stripThinkingTags(text: string): string {
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, '') // full <think>…</think> blocks
    .replace(/^[\s\S]*?<\/think>\s*/i, '')      // orphan </think>: strip everything up to and including it
    .trim();
}
const MAX_OVERFLOW_RETRIES = 2;
/** Flush memory to disk every N iterations regardless of context size. */
const PERIODIC_FLUSH_INTERVAL = 5;

/**
 * Build a compact Sources footer from a deduplicated list of URLs.
 * Only appended to answers when the model hasn't already cited inline links.
 * Limits to 10 URLs to keep the footer scannable.
 *
 * Social media post URLs (Reddit, X/Twitter, etc.) are excluded — these are
 * inputs to sentiment analysis, not authoritative research citations.
 */

/** Domains excluded from the Sources footer (social media / UGC). */
const EXCLUDED_SOURCE_DOMAINS = [
  'reddit.com',
  'x.com',
  'twitter.com',
  'threads.net',
  'bsky.app',
  'bluesky.app',
];

export function buildSourcesFooter(urls: string[]): string {
  const filtered = urls.filter(u => {
    try {
      const host = new URL(u).hostname.replace(/^www\./, '');
      return !EXCLUDED_SOURCE_DOMAINS.some(d => host === d || host.endsWith(`.${d}`));
    } catch {
      return false;
    }
  });
  const unique = [...new Set(filtered)].slice(0, 10);
  if (unique.length === 0) return '';
  const lines = unique.map((u, i) => `${i + 1}. ${u}`).join('\n');
  return `\n\n---\n**Sources**\n${lines}`;
}

function formatDiagnosticPrice(value: unknown): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value >= 1000
    ? `$${value.toLocaleString('en-US', { maximumFractionDigits: 0 })}`
    : `$${value.toFixed(2)}`;
}

function formatDiagnosticNumber(value: unknown, digits = 3): string | null {
  if (typeof value !== 'number' || !Number.isFinite(value)) return null;
  return value.toFixed(digits);
}

export function buildAbstainingMarkovAnswer(toolCalls: ToolCallRecord[]): string | null {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(call.result);
    } catch {
      continue;
    }

    const data = parsed && typeof parsed === 'object' ? (parsed as { data?: unknown }).data : null;
    if (!data || typeof data !== 'object') continue;

    const payload = data as Record<string, unknown>;
    if (payload['_tool'] !== 'markov_distribution' || payload['status'] !== 'abstain') continue;

    const canonical = payload['canonical'] && typeof payload['canonical'] === 'object'
      ? payload['canonical'] as Record<string, unknown>
      : {};
    const diagnostics = canonical['diagnostics'] && typeof canonical['diagnostics'] === 'object'
      ? canonical['diagnostics'] as Record<string, unknown>
      : {};
    const abstainReasons = Array.isArray(payload['abstainReasons'])
      ? payload['abstainReasons'].filter((reason): reason is string => typeof reason === 'string' && reason.trim().length > 0)
      : [];

    const ticker = typeof canonical['ticker'] === 'string' ? canonical['ticker'] : 'This asset';
    const horizon = typeof canonical['horizon'] === 'number' ? canonical['horizon'] : null;
    const currentPrice = formatDiagnosticPrice(canonical['currentPrice']);
    const trustedAnchors = typeof diagnostics['trustedAnchors'] === 'number' ? diagnostics['trustedAnchors'] : null;
    const totalAnchors = typeof diagnostics['totalAnchors'] === 'number' ? diagnostics['totalAnchors'] : null;
    const anchorQuality = typeof diagnostics['anchorQuality'] === 'string' ? diagnostics['anchorQuality'] : null;
    const outOfSampleR2 = formatDiagnosticNumber(diagnostics['outOfSampleR2']);
    const structuralBreakDetected = diagnostics['structuralBreakDetected'] === true;
    const structuralBreakDivergence = formatDiagnosticNumber(diagnostics['structuralBreakDivergence']);
    const predictionConfidence = formatDiagnosticNumber(diagnostics['predictionConfidence'], 2);

    const lines = [
      `## ${ticker} ${horizon !== null ? `${horizon}-Day ` : ''}Probability Distribution: Model Abstained`,
      '',
      'A calibrated probability distribution is not available. `markov_distribution` explicitly abstained, so no replacement scenario probabilities are shown.',
      '',
      '## Why it abstained',
      ...(abstainReasons.length > 0
        ? abstainReasons.map((reason) => `- ${reason}`)
        : ['- The tool marked this run as diagnostics-only.']),
      '',
      '## Diagnostics',
      '| Metric | Value |',
      '|--------|-------|',
      ...(currentPrice ? [`| Current price | ${currentPrice} |`] : []),
      `| Trusted anchors | ${trustedAnchors !== null && totalAnchors !== null ? `${trustedAnchors} / ${totalAnchors}` : 'N/A'} |`,
      `| Anchor quality | ${anchorQuality ?? 'N/A'} |`,
      `| Out-of-sample R² | ${outOfSampleR2 ?? 'N/A'} |`,
      `| Structural break | ${structuralBreakDetected ? `Yes${structuralBreakDivergence ? ` (${structuralBreakDivergence})` : ''}` : 'No'} |`,
      `| Prediction confidence | ${predictionConfidence ?? 'N/A'} |`,
      '',
      '## Interpretation',
      'Use later prediction-market searches only as diagnostics about missing or low-quality anchors. They do not justify a replacement calibrated distribution for this horizon.',
      '',
      '## Next best option',
      '- Wait for horizon-matched terminal threshold markets to list, or',
      '- Use `polymarket_forecast` for a point estimate with a confidence interval instead of a full distribution.',
    ];

    return lines.join('\n');
  }

  return null;
}

function isDistributionQuery(query: string): boolean {
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

const TRADING_DAYS_PER_WEEK = 5;
const TRADING_DAYS_PER_MONTH = 21;
const TRADING_DAYS_PER_QUARTER = TRADING_DAYS_PER_MONTH * 3;
const QUARTER_END_MONTHS: Record<1 | 2 | 3 | 4, number> = {
  1: 3,
  2: 6,
  3: 9,
  4: 12,
};

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
  return isExplicitTerminalDistributionQuery(query)
    && !toolCalls.some((call) => call.tool === 'markov_distribution');
}

export function isCryptoForecastQuery(query: string): boolean {
  if (isExplicitTerminalDistributionQuery(query)) return false;

  if (/\buse the\s+probability_assessment\s+skill\b/i.test(query)) return false;

  const detected = detectAssetType(query);
  if (detected.type !== 'crypto' || !detected.ticker) return false;

  const lower = query.toLowerCase();
  const hasForecastLanguage = /\bforecast\b|\bpredict(?:ion)?\b|\boutlook\b|\bprice target\b|where .* headed|what will .* trade|how .* move/.test(lower);
  const hasFutureHorizon = /over the next\s+\d+\s*(?:day|days|week|weeks)\b|next\s+\d+\s*(?:day|days|week|weeks)\b|in\s+\d+\s*(?:day|days|week|weeks)\b/.test(lower);

  return hasForecastLanguage || hasFutureHorizon;
}

/**
 * Detect non-crypto asset forecast-like queries (stocks, ETFs, commodities).
 * Returns true when the query asks about future price/forecast/outlook for a
 * non-crypto asset — exactly the cases where Markov abstention should trigger
 * a forced get_market_data + polymarket_forecast fallback.
 */
export function isNonCryptoForecastQuery(query: string): boolean {
  if (isExplicitTerminalDistributionQuery(query)) return false;

  if (/\buse the\s+probability_assessment\s+skill\b/i.test(query)) return false;

  // Exclude crypto — that path has its own dedicated forcing
  const detected = detectAssetType(query);
  if (detected.type === 'crypto') return false;
  if (!detected.ticker) return false;

  const lower = query.toLowerCase();
  const hasForecastLanguage = /\bforecast\b|\bpredict(?:ion)?\b|\boutlook\b|\bprice target\b|where .* headed|what will .* trade|how .* move|price of|will .* (?:beat|hit|reach|drop|rise|fall|exceed|go above|go below)/.test(lower);
  const hasFutureHorizon = /over the next\s+\d+\s*(?:day|days|week|weeks|month|months|quarter|quarters)\b|next\s+\d+\s*(?:day|days|week|weeks|month|months|quarter|quarters)\b|in\s+\d+\s*(?:day|days|week|weeks|month|months|quarter|quarters)\b|\bnext\s+month\b|\bnext\s+quarter\b|\b(?:by\s+)?end\s+of\s+q[1-4](?:\s+\d{4})?\b|\bthrough\s+q[1-4](?:\s+\d{4})?\b/.test(lower);

  return hasForecastLanguage || hasFutureHorizon;
}

type ForcedNonCryptoPolymarketForecastArgs = {
  ticker: string;
  horizon_days: number;
  current_price?: number;
  markov_return?: number;
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
  const currentPrice = marketDataArgs
    ? extractCurrentPriceFromMarketDataQuery(toolCalls, marketDataArgs.query)
    : null;

  const needsCurrentPriceFetch = marketDataArgs !== null && currentPrice === null;

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

function extractPriceFromPayload(value: unknown): number | null {
  if (!value || typeof value !== 'object') return null;
  const record = value as Record<string, unknown>;

  const directPrice = record['price'];
  if (typeof directPrice === 'number' && Number.isFinite(directPrice) && directPrice > 0) {
    return directPrice;
  }

  const snapshot = record['snapshot'];
  if (snapshot && typeof snapshot === 'object') {
    const snapshotPrice = (snapshot as Record<string, unknown>)['price'];
    if (typeof snapshotPrice === 'number' && Number.isFinite(snapshotPrice) && snapshotPrice > 0) {
      return snapshotPrice;
    }
  }

  return null;
}

function extractCurrentPriceFromMarketDataQuery(toolCalls: ToolCallRecord[], query: string): number | null {
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

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution' || data['status'] !== 'abstain') continue;

    const forecastHint = data['forecastHint'];
    if (!forecastHint || typeof forecastHint !== 'object') continue;

    const usage = (forecastHint as Record<string, unknown>)['usage'];
    const markovReturn = (forecastHint as Record<string, unknown>)['markovReturn'];
    if (usage !== 'forecast_only') continue;
    if (typeof markovReturn === 'number' && Number.isFinite(markovReturn)) {
      return markovReturn;
    }
  }

  return null;
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

  const currentPrice = extractCurrentPriceFromToolCalls(toolCalls);
  if (currentPrice !== null) args.current_price = currentPrice;

  const sentimentScore = extractSentimentScoreFromToolCalls(toolCalls);
  if (sentimentScore !== null) args.sentiment_score = sentimentScore;

  const markovReturn = extractMarkovReturnFromToolCalls(toolCalls);
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

  const marketDataArgs = buildForcedNonCryptoMarketDataArgs(query);
  const currentPrice = marketDataArgs
    ? extractCurrentPriceFromMarketDataQuery(toolCalls, marketDataArgs.query)
    : null;
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
    if (existingTicker !== ticker.toUpperCase()) return false;
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
  const desiredHorizon = inferDistributionHorizon(query);

  return toolCalls.some((call) => {
    if (call.tool !== 'markov_distribution') return false;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) return false;

    const data = parseToolCallData(call);
    return data?._tool === 'markov_distribution' && data.status === status;
  });
}

function hasSuccessfulMarkovDistributionForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  return hasMarkovDistributionStatusForQuery(query, toolCalls, 'ok');
}

function hasAbstainingMarkovDistributionForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  return hasMarkovDistributionStatusForQuery(query, toolCalls, 'abstain');
}

function hasPolymarketForecastCoverage(
  toolCalls: ToolCallRecord[],
  desired: ForcedNonCryptoPolymarketForecastArgs,
): boolean {
  return toolCalls.some((call) => {
    if (call.tool !== 'polymarket_forecast') return false;
    if (/^Error:/i.test(call.result) || /"error"\s*:/i.test(call.result)) return false;

    const existingTicker = typeof call.args['ticker'] === 'string'
      ? call.args['ticker'].toUpperCase()
      : null;
    if (existingTicker !== desired.ticker.toUpperCase()) return false;

    if (getForecastHorizonArg(call.args) !== desired.horizon_days) return false;

    if (desired.current_price !== undefined && !numbersApproximatelyMatch(call.args['current_price'], desired.current_price)) {
      return false;
    }

    if (desired.markov_return !== undefined && !numbersApproximatelyMatch(call.args['markov_return'], desired.markov_return)) {
      return false;
    }

    return true;
  });
}

function hasPolymarketForecastWithMarkovReturn(toolCalls: ToolCallRecord[]): boolean {
  return toolCalls.some((call) => call.tool === 'polymarket_forecast'
    && isFiniteNumber(call.args['markov_return']));
}

export function shouldRerunPolymarketForecastWithMarkov(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isCryptoForecastQuery(query)) return false;
  if (buildForcedCryptoForecastMarkovArgs(query) === null) return false;
  if (!toolCalls.some((call) => call.tool === 'polymarket_forecast')) return false;
  if (extractMarkovReturnFromToolCalls(toolCalls) === null) return false;
  return !hasPolymarketForecastWithMarkovReturn(toolCalls);
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

  const BTC_NEXT_WEEK_TRADING_DAYS = 5;
  if (!horizon && ticker === 'BTC-USD' && /\bnext\s+week\b/i.test(query)) {
    horizon = BTC_NEXT_WEEK_TRADING_DAYS;
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

  const hasMarketData = toolCalls.some((call) => call.tool === 'get_market_data');
  const hasSocialSentiment = toolCalls.some((call) => call.tool === 'social_sentiment');
  const hasPolymarketForecast = toolCalls.some((call) => call.tool === 'polymarket_forecast');
  const hasOnchain = toolCalls.some((call) => call.tool === 'get_onchain_crypto');
  const hasFixedIncome = toolCalls.some((call) => call.tool === 'get_fixed_income');
  const needsMarkov = buildForcedCryptoForecastMarkovArgs(query) !== null
    && !toolCalls.some((call) => call.tool === 'markov_distribution');
  const needsPolymarketRerun = shouldRerunPolymarketForecastWithMarkov(query, toolCalls);

  return !hasMarketData || !hasSocialSentiment || !hasPolymarketForecast || !hasOnchain || !hasFixedIncome || needsMarkov || needsPolymarketRerun;
}

function hasSuccessfulMarkovDistribution(toolCalls: ToolCallRecord[]): boolean {
  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    try {
      const parsed = JSON.parse(call.result) as { data?: { _tool?: string; status?: string } };
      if (parsed?.data?._tool === 'markov_distribution' && parsed.data.status === 'ok') {
        return true;
      }
    } catch {
      continue;
    }
  }
  return false;
}

export function buildUnavailableDistributionAnswer(query: string, toolCalls: ToolCallRecord[]): string | null {
  if (!isDistributionQuery(query)) return null;
  if (hasSuccessfulMarkovDistribution(toolCalls)) return null;

  return [
    '## Probability Distribution Unavailable',
    '',
    'A calibrated price-distribution answer is not available for this query. No successful non-abstaining `markov_distribution` result was produced, so I am not emitting substitute distribution buckets or historical-volatility scenario percentages.',
    '',
    '## Why this answer stops here',
    '- A price-distribution request needs horizon-matched terminal threshold anchors or a validated canonical Markov distribution.',
    '- Neither condition was met in this run, so any bucketed probability table would be speculative rather than grounded.',
    '',
    '## Safe next options',
    '- Wait for terminal threshold markets that match the requested horizon, or',
    '- Use `polymarket_forecast` for a point estimate with a confidence interval instead of a full distribution.',
  ].join('\n');
}

export function buildDistributionWarningPrefix(query: string, toolCalls: ToolCallRecord[]): string | null {
  if (!isDistributionQuery(query)) return null;
  if (hasSuccessfulMarkovDistribution(toolCalls)) return null;

  for (let i = toolCalls.length - 1; i >= 0; i--) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;

    let parsed: unknown;
    try {
      parsed = JSON.parse(call.result);
    } catch {
      continue;
    }

    const data = parsed && typeof parsed === 'object' ? (parsed as { data?: unknown }).data : null;
    if (!data || typeof data !== 'object') continue;

    const payload = data as Record<string, unknown>;
    if (payload['_tool'] !== 'markov_distribution' || payload['status'] !== 'abstain') continue;

    const abstainReasons = Array.isArray(payload['abstainReasons'])
      ? payload['abstainReasons'].filter((reason): reason is string => typeof reason === 'string' && reason.trim().length > 0)
      : [];

    return [
      '## Warning: no calibrated Markov terminal distribution was available',
      '',
      'The Markov distribution workflow abstained for this query. Any fallback analysis below must be treated as non-distribution context unless it comes directly from a validated canonical Markov payload.',
      ...(abstainReasons.length > 0
        ? ['', 'Key abstain reasons:', ...abstainReasons.map((reason) => `- ${reason}`)]
        : []),
      '',
      '---',
      '',
    ].join('\n');
  }

  return [
    '## Warning: no validated Markov distribution was produced',
    '',
    'Cramer-Short did not produce a successful non-abstaining `markov_distribution` result for this distribution query. Any answer below should be read as fallback analysis, not a calibrated probability distribution.',
    '',
    '---',
    '',
  ].join('\n');
}

// ============================================================================
// Context summary helpers (exported for unit tests)
// ============================================================================

/**
 * Numeric fact patterns extracted from tool results before they are cleared
 * from context. Each pattern captures a distinct type of financial data.
 */
export const FACT_PATTERNS: ReadonlyArray<RegExp> = [
  /\$[\d,]+(?:\.\d{1,2})?(?:\s*[BMK](?:illion)?)?/gi,  // prices / market caps
  /[-+]?\d+(?:\.\d+)?%/g,                                // percentages
  /\b(?:IC|ICIR|RankIC)\s*[:=]\s*[-+]?\d+\.\d+/gi,     // factor IC values
  /\bP\/E\s*[:=]?\s*\d+(?:\.\d+)?x?/gi,                 // P/E ratios
  /\bEV\/EBITDA\s*[:=]?\s*\d+(?:\.\d+)?x?/gi,           // EV/EBITDA
  /\bP\/[SB]\s*[:=]?\s*\d+(?:\.\d+)?x?/gi,              // P/S, P/B
  /\b(?:probability|chance|likely)\s+[:=]?\s*\d+(?:\.\d+)?%/gi, // probabilities
  /\bWACC\s*[:=]\s*\d+(?:\.\d+)?%/gi,                   // WACC
  /\bROIC?\s*[:=]\s*\d+(?:\.\d+)?%/gi,                  // ROIC
];

/**
 * Extract up to `maxFacts` unique key numeric facts from a text snippet.
 * Returns them as a compact comma-separated string, or '' when none found.
 */
export function extractKeyFacts(text: string, maxFacts = 10): string {
  const seen = new Set<string>();
  const facts: string[] = [];
  for (const re of FACT_PATTERNS) {
    const pattern = new RegExp(re.source, re.flags);
    for (const m of text.matchAll(pattern)) {
      const key = m[0].toLowerCase().replace(/\s+/g, ' ').trim();
      if (!seen.has(key) && facts.length < maxFacts) {
        seen.add(key);
        facts.push(m[0].trim());
      }
    }
  }
  return facts.join(', ');
}

/** Maps raw JSON field names found in financial tool results to compact labels. */
const METRIC_KEY_MAP: Readonly<Record<string, string>> = {
  revenue: 'rev',
  total_revenue: 'rev',
  net_income: 'NI',
  earnings_per_share: 'EPS',
  eps: 'EPS',
  pe_ratio: 'PE',
  price_to_earnings_ratio: 'PE',
  ev_to_ebitda: 'EV/EBITDA',
  enterprise_value_over_ebitda: 'EV/EBITDA',
  market_cap: 'mktcap',
  market_capitalization: 'mktcap',
  gross_margin: 'GM%',
  operating_margin: 'OpM%',
  price_to_book: 'P/B',
  return_on_equity: 'ROE%',
  return_on_assets: 'ROA%',
  debt_to_equity: 'D/E',
};

/**
 * Parse key financial metrics from a JSON-like tool result snippet.
 * Returns compact `label=value` strings (up to 6) for ticker table rows.
 */
export function extractTickerMetrics(text: string): string[] {
  const metrics: string[] = [];
  const seen = new Set<string>();
  const kvPattern = /"([\w_]+)":\s*"?([^",\n\]}{]+)"?/g;
  for (const m of text.matchAll(kvPattern)) {
    const label = METRIC_KEY_MAP[m[1]!.toLowerCase()];
    if (label) {
      const val = m[2]!.trim().replace(/,$/, '');
      const entry = `${label}=${val}`;
      if (!seen.has(entry) && metrics.length < 6) {
        seen.add(entry);
        metrics.push(entry);
      }
    }
  }
  return metrics;
}

/**
 * Build a merged context summary string from tool results about to be cleared.
 *
 * - Prefixes each line with the tool's ticker/query arg when present so the
 *   LLM retains the ticker→value association (e.g. `get_financials(ticker=NVDA): …`).
 * - Appends a compact ticker→metric table when financial key/value pairs are found.
 * - Snippet length is 400 chars (up from the previous 200) for richer context.
 * - When `existingSummary` is provided the new facts are merged into it instead
 *   of appending a separate entry, preventing 3+ summary blocks stacking up.
 *
 * Returns null when there is nothing to summarise.
 */
export function buildContextSummaryText(
  toSummarise: Array<{ toolName: string; args: Record<string, unknown>; snippet: string }>,
  existingSummary: string | null,
): string | null {
  if (toSummarise.length === 0) return null;

  const lines: string[] = [];
  const tickerRows = new Map<string, string[]>();

  for (const { toolName, args, snippet } of toSummarise) {
    const ticker = typeof args['ticker'] === 'string' ? args['ticker'].toUpperCase() : null;
    const queryArg = typeof args['query'] === 'string' ? args['query'] : null;

    const argsStr = Object.entries(args).map(([k, v]) => `${k}=${v}`).join(', ');
    const condensed = snippet.replace(/\s+/g, ' ').trim().slice(0, 400);
    const keyFacts = extractKeyFacts(snippet);
    const factsNote = keyFacts ? ` [KEY FACTS: ${keyFacts}]` : '';

    // Prefix with ticker/query so the LLM knows which asset the data belongs to.
    const callLabel = ticker
      ? `${toolName}(ticker=${ticker})`
      : queryArg
        ? `${toolName}(query=${queryArg})`
        : `${toolName}(${argsStr})`;
    lines.push(`- ${callLabel}: ${condensed}…${factsNote}`);

    if (ticker) {
      const metrics = extractTickerMetrics(snippet);
      if (metrics.length > 0 && !tickerRows.has(ticker)) {
        tickerRows.set(ticker, metrics);
      }
    }
  }

  let newSummary = `The following ${toSummarise.length} earlier tool result(s) were condensed to save context:\n${lines.join('\n')}`;

  if (tickerRows.size > 0) {
    const tableLines = [...tickerRows.entries()].map(([t, m]) => `${t}: ${m.join(', ')}`);
    newSummary += `\n\nKey metrics by ticker:\n${tableLines.join('\n')}`;
  }

  // Merge into the existing summary rather than appending a second block.
  if (existingSummary) {
    return `${existingSummary}\n\n---\n${newSummary}`;
  }
  return newSummary;
}

/**
 * The core agent class that handles the agent loop and tool execution.
 */
export class Agent {
  private readonly model: string;
  private readonly maxIterations: number;
  private readonly tools: StructuredToolInterface[];
  private readonly toolMap: Map<string, StructuredToolInterface>;
  private readonly toolExecutor: AgentToolExecutor;
  private readonly systemPrompt: string;
  private readonly signal?: AbortSignal;
  private readonly memoryEnabled: boolean;
  private readonly thinkEnabled: boolean | undefined;

  private constructor(
    config: AgentConfig,
    tools: StructuredToolInterface[],
    systemPrompt: string,
  ) {
    this.model = config.model ?? DEFAULT_MODEL;
    this.maxIterations = config.maxIterations ?? getSetting<number>('maxIterations', DEFAULT_MAX_ITERATIONS);
    this.tools = tools;
    this.toolMap = new Map(tools.map(t => [t.name, t]));
    this.toolExecutor = new AgentToolExecutor(this.toolMap, config.signal, config.requestToolApproval, config.sessionApprovedTools);
    this.systemPrompt = systemPrompt;
    this.signal = config.signal;
    this.memoryEnabled = config.memoryEnabled ?? true;
    this.thinkEnabled = config.thinkEnabled;
  }

  /**
   * Create a new Agent instance with tools.
   */
  static async create(config: AgentConfig = {}): Promise<Agent> {
    const model = config.model ?? DEFAULT_MODEL;
    const tools = getTools(model);
    const soulContent = await loadSoulDocument();
    let memoryFiles: string[] = [];
    let memoryContext: string | null = null;

    if (config.memoryEnabled !== false) {
      const memoryManager = await MemoryManager.get();
      memoryFiles = await memoryManager.listFiles();
      const session = await memoryManager.loadSessionContext();
      if (session.text.trim()) {
        memoryContext = session.text;
      }
    }

    const systemPrompt = buildSystemPrompt(
      model,
      soulContent,
      config.channel,
      config.groupContext,
      memoryFiles,
      memoryContext,
    );
    return new Agent(config, tools, systemPrompt);
  }

  /**
   * Run the agent and yield events for real-time UI updates.
   * Anthropic-style context management: full tool results during iteration,
   * with threshold-based clearing of oldest results when context exceeds limit.
   */
  async *run(query: string, inMemoryHistory?: InMemoryChatHistory): AsyncGenerator<AgentEvent> {
    const startTime = Date.now();

    if (this.tools.length === 0) {
      yield { type: 'done', answer: 'No tools available. Please check your API key configuration.', toolCalls: [], iterations: 0, totalTime: Date.now() - startTime };
      return;
    }

    const ctx = createRunContext(query);
    const memoryFlushState = { alreadyFlushed: false };
    const periodicFlushState = { lastFlushedIteration: 0 };

    // Build initial prompt with conversation history context
    let currentPrompt = this.buildInitialPrompt(query, inMemoryHistory);

    // Auto-inject relevant prior research memories based on tickers mentioned
    currentPrompt = await injectMemoryContext(query, currentPrompt, {
      getMemoryManager: () => MemoryManager.get(),
      extractTickers: (text) => extractTickersFn(text),
    });

    // Auto-inject Polymarket prediction market context for detected asset signals
    currentPrompt = await injectPolymarketContext(query, currentPrompt, {
      extractSignals: (text) => extractSignalsFn(text),
      fetchMarkets: (q, limit) => fetchPolymarketMarkets(q, limit),
    });

    // Track whether sequential_thinking has been used at least once this session
    let sequentialThinkingUsed = false;
    // Cap retries for the sequential_thinking compliance reminder to avoid
    // an infinite loop when a model persistently ignores the instruction.
    let sequentialThinkingRetries = 0;
    const MAX_ST_RETRIES = 3;
    // Hard cap on total sequential_thinking calls so planning never burns all
    // iterations before any research tool runs. Models sometimes loop through
    // 10-15 thoughts on complex queries, leaving no budget for actual research.
    let sequentialThinkingCallCount = 0;
    const MAX_SEQUENTIAL_THOUGHTS = 6;

    // Main agent loop
    let overflowRetries = 0;
    while (ctx.iteration < this.maxIterations) {
      ctx.iteration++;
      yield { type: 'progress', iteration: ctx.iteration, maxIterations: this.maxIterations } as ProgressEvent;

      let response: AIMessage | string;
      let usage: TokenUsage | undefined;

      while (true) {
        try {
          const result = await this.callModel(currentPrompt);
          response = result.response;
          usage = result.usage;
          overflowRetries = 0;
          break;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);

          if (isContextOverflowError(errorMessage) && overflowRetries < MAX_OVERFLOW_RETRIES) {
            overflowRetries++;
            const overflowKeep = Math.max(2, getKeepToolUses() - 2);
            this.injectContextSummaryBeforeClearing(ctx, overflowKeep);
            const clearedCount = ctx.scratchpad.clearOldestToolResults(overflowKeep);

            if (clearedCount > 0) {
              yield { type: 'context_cleared', clearedCount, keptCount: overflowKeep };
              currentPrompt = buildIterationPrompt(
                query,
                ctx.scratchpad.getToolResults(),
                ctx.scratchpad.formatToolUsageForPrompt()
              );
              continue;
            }
          }

          const totalTime = Date.now() - ctx.startTime;
          const provider = resolveProvider(this.model).displayName;
          yield {
            type: 'done',
            answer: `Error: ${formatUserFacingError(errorMessage, provider)}`,
            toolCalls: ctx.scratchpad.getToolCallRecords(),
            iterations: ctx.iteration,
            totalTime,
            tokenUsage: ctx.tokenCounter.getUsage(),
            tokensPerSecond: ctx.tokenCounter.getTokensPerSecond(totalTime),
          };
          return;
        }
      }

      ctx.tokenCounter.add(usage);

      // Emit reasoning block from Ollama thinking models (qwen3, deepseek-r1, qwq)
      if (typeof response !== 'string') {
        const reasoning = extractReasoningContent(response as AIMessage);
        if (reasoning) {
          yield { type: 'reasoning', content: reasoning };
        }
      }

      const responseText = typeof response === 'string' ? response : extractTextContent(response);

      // Emit thinking if there are also tool calls (skip whitespace-only responses).
      // Truncate to 500 chars to prevent large JSON blobs from flooding the terminal —
      // some models (e.g. Qwen) embed raw tool-call syntax in their text content.
      if (responseText?.trim() && typeof response !== 'string' && hasToolCalls(response)) {
        const trimmedText = responseText.trim();
        ctx.scratchpad.addThinking(trimmedText);
        const displayText = trimmedText.length > 500 ? trimmedText.slice(0, 500) + '…' : trimmedText;
        yield { type: 'thinking', message: displayText };
      }

      // No tool calls = final answer is in this response
      if (typeof response === 'string' || !hasToolCalls(response)) {
        if (shouldForceCryptoForecastTools(query, ctx.scratchpad.getToolCallRecords())) {
          const forced = yield* this.forceCryptoForecastTools(ctx);
          if (forced) {
            yield* this.manageContextThreshold(ctx, query, memoryFlushState);
            currentPrompt = buildIterationPrompt(
              query,
              ctx.scratchpad.getToolResults(),
              ctx.scratchpad.formatToolUsageForPrompt(),
            );
            continue;
          }
        }

        if (shouldForceNonCryptoForecastFallback(query, ctx.scratchpad.getToolCallRecords())) {
          const forced = yield* this.forceNonCryptoForecastFallback(ctx);
          if (forced) {
            yield* this.manageContextThreshold(ctx, query, memoryFlushState);
            currentPrompt = buildIterationPrompt(
              query,
              ctx.scratchpad.getToolResults(),
              ctx.scratchpad.formatToolUsageForPrompt(),
            );
            continue;
          }
        }

        if (shouldForceMarkovDistribution(query, ctx.scratchpad.getToolCallRecords())) {
          const forced = yield* this.forceMarkovDistribution(ctx);
          if (forced) {
            yield* this.manageContextThreshold(ctx, query, memoryFlushState);
            currentPrompt = buildIterationPrompt(
              query,
              ctx.scratchpad.getToolResults(),
              ctx.scratchpad.formatToolUsageForPrompt(),
            );
            continue;
          }
        }

        yield* this.handleDirectResponse(responseText ?? '', ctx, currentPrompt);
        return;
      }

      // Enforce sequential_thinking as the mandatory first tool call.
      // If the model's first tool call this session is not sequential_thinking,
      // inject a reminder and retry — but only up to MAX_ST_RETRIES times to
      // prevent an infinite loop when a model persistently ignores the reminder.
      if (!sequentialThinkingUsed) {
        const firstTool = (response as AIMessage).tool_calls?.[0]?.name;
        if (firstTool && firstTool !== 'sequential_thinking') {
          if (sequentialThinkingRetries < MAX_ST_RETRIES) {
            sequentialThinkingRetries++;
            ctx.iteration--; // don't charge this iteration
            currentPrompt = `${currentPrompt}\n\nIMPORTANT REMINDER: You MUST call sequential_thinking FIRST before calling any other tool. Start with sequential_thinking to plan your approach, then proceed.`;
            continue;
          }
          // Retries exhausted — proceed without sequential_thinking rather than
          // looping forever. Mark as satisfied so we stop checking.
          sequentialThinkingUsed = true;
        }
      }

      // Mark sequential_thinking as satisfied once it appears in any tool call
      if (!sequentialThinkingUsed) {
        const stToolCalls = (response as AIMessage).tool_calls ?? [];
        if (stToolCalls.some((tc) => tc.name === 'sequential_thinking')) {
          sequentialThinkingUsed = true;
        }
      }

      if (sequentialThinkingUsed && shouldForceMarkovDistribution(query, ctx.scratchpad.getToolCallRecords())) {
        const forced = yield* this.forceMarkovDistribution(ctx);
        if (forced) {
          yield* this.manageContextThreshold(ctx, query, memoryFlushState);
          currentPrompt = buildIterationPrompt(
            query,
            ctx.scratchpad.getToolResults(),
            ctx.scratchpad.formatToolUsageForPrompt(),
          );
          continue;
        }
      }

      if (sequentialThinkingUsed && shouldForceCryptoForecastTools(query, ctx.scratchpad.getToolCallRecords())) {
        const forced = yield* this.forceCryptoForecastTools(ctx);
        if (forced) {
          yield* this.manageContextThreshold(ctx, query, memoryFlushState);
          currentPrompt = buildIterationPrompt(
            query,
            ctx.scratchpad.getToolResults(),
            ctx.scratchpad.formatToolUsageForPrompt(),
          );
          continue;
        }
      }

      if (sequentialThinkingUsed && shouldForceNonCryptoForecastFallback(query, ctx.scratchpad.getToolCallRecords())) {
        const forced = yield* this.forceNonCryptoForecastFallback(ctx);
        if (forced) {
          yield* this.manageContextThreshold(ctx, query, memoryFlushState);
          currentPrompt = buildIterationPrompt(
            query,
            ctx.scratchpad.getToolResults(),
            ctx.scratchpad.formatToolUsageForPrompt(),
          );
          continue;
        }
      }

      // Count sequential_thinking calls before executing tools (needed for nudge below).
      const toolCalls = (response as AIMessage).tool_calls ?? [];
      const stCallsThisIteration = toolCalls.filter((tc) => tc.name === 'sequential_thinking').length;
      sequentialThinkingCallCount += stCallsThisIteration;

      // Execute tools and add results to scratchpad (response is AIMessage here)
      for await (const event of this.toolExecutor.executeAll(response, ctx)) {
        yield event;
        if (event.type === 'tool_denied') {
          const totalTime = Date.now() - ctx.startTime;
          yield {
            type: 'done',
            answer: '',
            toolCalls: ctx.scratchpad.getToolCallRecords(),
            iterations: ctx.iteration,
            totalTime,
            tokenUsage: ctx.tokenCounter.getUsage(),
            tokensPerSecond: ctx.tokenCounter.getTokensPerSecond(totalTime),
          };
          return;
        }
      }
      yield* this.manageContextThreshold(ctx, query, memoryFlushState);

      // Periodic auto-save: flush findings to long-term memory every N iterations
      // so a crash doesn't lose all research done so far.
      if (
        this.memoryEnabled &&
        ctx.iteration - periodicFlushState.lastFlushedIteration >= PERIODIC_FLUSH_INTERVAL
      ) {
        yield* this.runPeriodicMemoryFlush(ctx, query, periodicFlushState);
      }

      // Build iteration prompt with full tool results (Anthropic-style)
      currentPrompt = buildIterationPrompt(
        query,
        ctx.scratchpad.getToolResults(),
        ctx.scratchpad.formatToolUsageForPrompt()
      );

      // After the cap is hit, redirect the model to stop planning and start
      // using research tools. Only inject the nudge once (at the boundary).
      if (stCallsThisIteration > 0 && sequentialThinkingCallCount >= MAX_SEQUENTIAL_THOUGHTS) {
        currentPrompt += '\n\n[SYSTEM NOTE: Planning phase complete. You have used the maximum number of sequential_thinking steps allowed. You MUST now proceed directly to research tools (financial_search, web_search, read_filings, etc.) to gather data and answer the question. Do not call sequential_thinking again.]';
      }
    }

    // Max iterations reached — synthesize a best-effort answer from gathered research
    // rather than yielding a bare failure message. Any data collected is still useful.
    const toolResults = ctx.scratchpad.getToolResults().trim();
    const hasMeaningfulResearch = toolResults.length > 50;

    const synthesisPrompt = hasMeaningfulResearch
      ? buildIterationPrompt(
          query,
          toolResults,
          ctx.scratchpad.formatToolUsageForPrompt(),
        ) +
          `\n\n[SYSTEM NOTE: You have reached the maximum number of research steps (${this.maxIterations}). ` +
          `You MUST now write your best-effort final answer using ONLY the data gathered above. ` +
          `Start your response with "**[Best-effort summary — research may be incomplete]**\\n\\n" ` +
          `then provide the most useful analysis you can from the available data. Do NOT call any more tools.]`
      : query;

    yield* this.handleDirectResponse('', ctx, synthesisPrompt);
  }

  /**
   * Call the LLM with the current prompt.
   * @param prompt - The prompt to send to the LLM
   * @param useTools - Whether to bind tools (default: true). When false, returns string directly.
   */
  private async callModel(prompt: string, useTools: boolean = true): Promise<{ response: AIMessage | string; usage?: TokenUsage }> {
    const result = await callLlm(prompt, {
      model: this.model,
      systemPrompt: this.systemPrompt,
      tools: useTools ? this.tools : undefined,
      signal: this.signal,
      thinkOverride: this.thinkEnabled,
    });
    return { response: result.response, usage: result.usage };
  }

  private async *forceMarkovDistribution(
    ctx: RunContext,
  ): AsyncGenerator<AgentEvent, boolean> {
    const args = buildForcedMarkovArgs(ctx.query);
    if (!args) return false;

    for await (const event of this.toolExecutor.executeTool('markov_distribution', args, ctx)) {
      yield event;
      if (event.type === 'tool_error' || event.type === 'tool_denied') {
        return false;
      }
    }

    return true;
  }

  private async *forceCryptoForecastTools(
    ctx: RunContext,
  ): AsyncGenerator<AgentEvent, boolean> {
    let forcedAny = false;

    const hasTool = (toolName: string) =>
      ctx.scratchpad.getToolCallRecords().some((call) => call.tool === toolName);

    if (!hasTool('get_market_data')) {
      const args = buildForcedMarketDataArgs(ctx.query);
      if (args) {
        let ok = true;
        for await (const event of this.toolExecutor.executeTool('get_market_data', args, ctx)) {
          yield event;
          if (event.type === 'tool_error' || event.type === 'tool_denied') {
            ok = false;
            break;
          }
        }
        forcedAny = forcedAny || ok;
      }
    }

    if (!hasTool('social_sentiment')) {
      const args = buildForcedSocialSentimentArgs(ctx.query);
      if (args) {
        let ok = true;
        for await (const event of this.toolExecutor.executeTool('social_sentiment', args, ctx)) {
          yield event;
          if (event.type === 'tool_error' || event.type === 'tool_denied') {
            ok = false;
            break;
          }
        }
        forcedAny = forcedAny || ok;
      }
    }

    if (!hasTool('markov_distribution')) {
      const args = buildForcedCryptoForecastMarkovArgs(ctx.query);
      if (args) {
        let ok = true;
        for await (const event of this.toolExecutor.executeTool('markov_distribution', args, ctx)) {
          yield event;
          if (event.type === 'tool_error' || event.type === 'tool_denied') {
            ok = false;
            break;
          }
        }
        forcedAny = forcedAny || ok;
      }
    }

    if (!hasTool('polymarket_forecast')) {
      const args = buildForcedPolymarketForecastArgs(ctx.query, ctx.scratchpad.getToolCallRecords());
      if (args) {
        let ok = true;
        for await (const event of this.toolExecutor.executeTool('polymarket_forecast', args, ctx)) {
          yield event;
          if (event.type === 'tool_error' || event.type === 'tool_denied') {
            ok = false;
            break;
          }
        }
        forcedAny = forcedAny || ok;
      }
    }

    if (shouldRerunPolymarketForecastWithMarkov(ctx.query, ctx.scratchpad.getToolCallRecords())) {
      const args = buildForcedPolymarketForecastArgs(ctx.query, ctx.scratchpad.getToolCallRecords());
      if (args) {
        let ok = true;
        for await (const event of this.toolExecutor.executeTool('polymarket_forecast', args, ctx)) {
          yield event;
          if (event.type === 'tool_error' || event.type === 'tool_denied') {
            ok = false;
            break;
          }
        }
        forcedAny = forcedAny || ok;
      }
    }

    if (!hasTool('get_onchain_crypto')) {
      const args = buildForcedOnchainArgs(ctx.query);
      if (args) {
        let ok = true;
        for await (const event of this.toolExecutor.executeTool('get_onchain_crypto', args, ctx)) {
          yield event;
          if (event.type === 'tool_error' || event.type === 'tool_denied') {
            ok = false;
            break;
          }
        }
        forcedAny = forcedAny || ok;
      }
    }

    if (!hasTool('get_fixed_income')) {
      const args = buildForcedFixedIncomeArgs();
      let ok = true;
      for await (const event of this.toolExecutor.executeTool('get_fixed_income', args, ctx)) {
        yield event;
        if (event.type === 'tool_error' || event.type === 'tool_denied') {
          ok = false;
          break;
        }
      }
      forcedAny = forcedAny || ok;
    }

    return forcedAny;
  }

  /** Force get_market_data + polymarket_forecast for non-crypto forecast asks after Markov abstains. */
  private async *forceNonCryptoForecastFallback(
    ctx: RunContext,
  ): AsyncGenerator<AgentEvent, boolean> {
    let forcedAny = false;
    const getToolCalls = () => ctx.scratchpad.getToolCallRecords();

    const marketDataArgs = buildForcedNonCryptoMarketDataArgs(ctx.query);
    if (
      marketDataArgs
      && extractCurrentPriceFromMarketDataQuery(getToolCalls(), marketDataArgs.query) === null
    ) {
      let ok = true;
      for await (const event of this.toolExecutor.executeTool('get_market_data', marketDataArgs, ctx)) {
        yield event;
        if (event.type === 'tool_error' || event.type === 'tool_denied') { ok = false; break; }
      }
      forcedAny = forcedAny || ok;
    }

    const forecastArgs = buildForcedNonCryptoPolymarketForecastArgs(ctx.query, getToolCalls());
    if (forecastArgs && !hasPolymarketForecastCoverage(getToolCalls(), forecastArgs)) {
      let ok = true;
      for await (const event of this.toolExecutor.executeTool('polymarket_forecast', forecastArgs, ctx)) {
        yield event;
        if (event.type === 'tool_error' || event.type === 'tool_denied') { ok = false; break; }
      }
      forcedAny = forcedAny || ok;
    }

    return forcedAny;
  }

  /**
   * Emit the response text as the final answer.
   *
   * When the model has already returned a text answer (non-empty fallbackText),
   * we emit it directly — no second LLM call is needed. Making an extra
   * streamCallLlm round-trip with a large prompt can hang for minutes on
   * heavy models and provides no benefit over the text we already have.
   *
   * The only case where we call streamCallLlm is max-iterations synthesis,
   * where fallbackText is empty and we need the LLM to write a fresh summary.
   * That call is guarded by a hard timeout so it cannot block indefinitely.
   */
  private async *handleDirectResponse(
    fallbackText: string,
    ctx: RunContext,
    currentPrompt?: string,
  ): AsyncGenerator<AgentEvent, void> {
    // Emit answer_start so the TUI can switch to streaming display mode
    yield { type: 'answer_start' } as AnswerStartEvent;

    let streamedAnswer = '';

    const toolCalls = ctx.scratchpad.getToolCallRecords();
    const warningPrefix = buildDistributionWarningPrefix(ctx.query, toolCalls);
    const baseText = stripThinkingTags(fallbackText);
    const text = warningPrefix ? `${warningPrefix}${baseText}` : baseText;

    if (text) {
      // We already have the answer from the non-streaming callLlm response.
      // Fake-stream it so the TUI shows the text appearing progressively.
      const CHUNK_SIZE = 6;
      for (let i = 0; i < text.length; i += CHUNK_SIZE) {
        const chunk = text.slice(i, i + CHUNK_SIZE);
        streamedAnswer += chunk;
        yield { type: 'answer_chunk', chunk } as AnswerChunkEvent;
      }
    } else if (currentPrompt) {
      // No pre-existing answer (e.g. max-iterations synthesis) — request a
      // fresh streaming response. Apply a hard timeout so we never hang.
      const timeoutSignal = AbortSignal.timeout(LLM_CALL_TIMEOUT_MS);
      const combinedSignal = this.signal
        ? AbortSignal.any([this.signal, timeoutSignal])
        : timeoutSignal;
      try {
        for await (const chunk of streamCallLlm(currentPrompt, {
          model: this.model,
          systemPrompt: this.systemPrompt,
          signal: combinedSignal,
        })) {
          streamedAnswer += chunk;
          yield { type: 'answer_chunk', chunk } as AnswerChunkEvent;
        }
      } catch {
        // Synthesis timed out or failed — surface the raw tool results so the
        // user has something to work with rather than seeing a blank answer.
        const toolSummary = ctx.scratchpad.getToolResults().trim();
        if (toolSummary) {
          const fallback =
            '**[Research interrupted — synthesis timed out]**\n\n' +
            'The model did not complete in time. Raw research data gathered:\n\n' +
            toolSummary.slice(0, 3000);
          yield { type: 'answer_chunk', chunk: fallback } as AnswerChunkEvent;
          streamedAnswer = fallback;
        }
      }
    }

    // Append a Sources footer when the answer used web searches or structured
    // financial tools that returned source URLs. Skipped for empty answers and
    // when the answer already contains a markdown link (model cited inline).
    const sourceUrls = ctx.scratchpad.collectSourceUrls();
    if (streamedAnswer && sourceUrls.length > 0 && !streamedAnswer.includes('](http')) {
      const footer = buildSourcesFooter(sourceUrls);
      streamedAnswer += footer;
      yield { type: 'answer_chunk', chunk: footer } as AnswerChunkEvent;
    }

    const totalTime = Date.now() - ctx.startTime;
    yield {
      type: 'done',
      answer: streamedAnswer,
      toolCalls: ctx.scratchpad.getToolCallRecords(),
      iterations: ctx.iteration,
      totalTime,
      tokenUsage: ctx.tokenCounter.getUsage(),
      tokensPerSecond: ctx.tokenCounter.getTokensPerSecond(totalTime),
    };
  }


  /**
   * Clear oldest tool results if context size exceeds threshold.
   */
  private async *manageContextThreshold(
    ctx: RunContext,
    query: string,
    memoryFlushState: { alreadyFlushed: boolean },
  ): AsyncGenerator<ContextClearedEvent | AgentEvent, void> {
    const fullToolResults = ctx.scratchpad.getToolResults();
    const estimatedContextTokens = estimateTokens(this.systemPrompt + ctx.query + fullToolResults);

    if (estimatedContextTokens > getContextThreshold()) {
      if (
        this.memoryEnabled &&
        shouldRunMemoryFlush({
          estimatedContextTokens,
          alreadyFlushed: memoryFlushState.alreadyFlushed,
        })
      ) {
        yield { type: 'memory_flush', phase: 'start' };
        const flushResult = await runMemoryFlush({
          model: this.model,
          systemPrompt: this.systemPrompt,
          query,
          toolResults: fullToolResults,
          signal: this.signal,
        }).catch(() => ({ flushed: false, written: false as const }));
        memoryFlushState.alreadyFlushed = flushResult.flushed;
        yield {
          type: 'memory_flush',
          phase: 'end',
          filesWritten: flushResult.written ? [`${new Date().toISOString().slice(0, 10)}.md`] : [],
        };
      }

      this.injectContextSummaryBeforeClearing(ctx, getKeepToolUses());
      const clearedCount = ctx.scratchpad.clearOldestToolResults(getKeepToolUses());
      if (clearedCount > 0) {
        memoryFlushState.alreadyFlushed = false;
        yield { type: 'context_cleared', clearedCount, keptCount: getKeepToolUses() };
      }
    }
  }

  /**
   * Builds a compact rule-based summary of tool results that are about to be
   * dropped from context and injects it as a context_summary entry so the LLM
   * doesn't lose analysis continuity without incurring an extra LLM call.
   *
   * If a context_summary already exists it merges the new facts into it
   * (via buildContextSummaryText) to prevent multiple summaries stacking up.
   */
  private injectContextSummaryBeforeClearing(ctx: RunContext, keepCount: number): void {
    const toSummarise = ctx.scratchpad.getContentToBeCleared(keepCount);
    if (toSummarise.length === 0) return;

    const existingSummary = ctx.scratchpad.getLatestContextSummary();
    const summary = buildContextSummaryText(toSummarise, existingSummary);
    if (summary) ctx.scratchpad.addContextSummary(summary);
  }
  /**
   * Periodic auto-save: flush research findings to long-term memory every
   * PERIODIC_FLUSH_INTERVAL iterations, independent of context size.
   * This prevents total data loss if the session crashes mid-research.
   */
  private async *runPeriodicMemoryFlush(
    ctx: RunContext,
    query: string,
    state: { lastFlushedIteration: number },
  ): AsyncGenerator<AgentEvent, void> {
    state.lastFlushedIteration = ctx.iteration;
    yield { type: 'memory_flush', phase: 'start' };
    const flushResult = await runMemoryFlush({
      model: this.model,
      systemPrompt: this.systemPrompt,
      query,
      toolResults: ctx.scratchpad.getToolResults(),
      signal: this.signal,
    }).catch(() => ({ flushed: false, written: false as const }));
    yield {
      type: 'memory_flush',
      phase: 'end',
      filesWritten: flushResult.written ? [`${new Date().toISOString().slice(0, 10)}.md`] : [],
    };
  }

  /**
   * Build initial prompt with conversation history context if available
   */
  private buildInitialPrompt(
    query: string,
    inMemoryChatHistory?: InMemoryChatHistory
  ): string {
    if (!inMemoryChatHistory?.hasMessages()) {
      return query;
    }

    const recentTurns = inMemoryChatHistory.getRecentTurns();
    if (recentTurns.length === 0) {
      return query;
    }

    return buildHistoryContext({
      entries: recentTurns,
      currentMessage: query,
    });
  }
}
