import { AIMessage } from '@langchain/core/messages';
import { StructuredToolInterface } from '@langchain/core/tools';
import { callLlm, streamCallLlm, getLlmCallTimeoutMs } from '../model/llm.js';
import { getSetting, loadConfig } from '../utils/config.js';
import { getTools } from '../tools/registry.js';
import {
  buildSystemPrompt,
  buildIterationPrompt,
  injectForecastLabRoutingHint,
  loadSoulDocument,
} from './prompts.js';
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
import { RECOMMENDED_CONFIDENCE_THRESHOLD } from '../tools/finance/markov-distribution.js';
import { resolveProvider } from '../providers.js';
import type { ToolCallRecord } from './scratchpad.js';
import {
  getForecastLabRoutingHint,
  type ForecastLabRoutingHint,
} from './forecast-lab-routing.js';
import { routeForecastLabQuery } from '../experiments/forecast-lab/router.js';
import { listForecastLabProfiles, listForecastLabStructuredMutations } from '../experiments/forecast-lab/profiles.js';
import { discoverSkills } from '../skills/registry.js';
import { extractForecastLabRunToolAnswer } from '../tools/forecast-lab-run.js';

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

function inferBtcShortHorizonForecastHorizon(query: string): number | null {
  const ticker = inferDistributionTicker(query);
  if (ticker !== 'BTC' && ticker !== 'BTC-USD') return null;

  const horizon = inferDistributionHorizon(query);
  if (horizon !== null) return horizon;
  if (/\bnext\s+week\b/i.test(query)) return TRADING_DAYS_PER_WEEK;

  return null;
}

function inferMarkovQueryHorizon(query: string): number | null {
  return inferDistributionHorizon(query) ?? inferBtcShortHorizonForecastHorizon(query);
}

function isBtcShortHorizonForecastQuery(query: string): boolean {
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

export function shouldPreserveAbstainingBtcShortHorizonForecast(
  query: string,
  toolCalls: ToolCallRecord[],
): boolean {
  if (isExplicitPolymarketForecastRequest(query)) return false;
  if (hasForecastArbitratorForQuery(query, toolCalls)) return false;
  if (hasCryptoPolymarketForecastCoverage(query, toolCalls)) return false;
  if (hasUsableOnchainResultForCryptoQuery(query, toolCalls)) return false;
  if (hasUsableFixedIncomeResult(toolCalls)) return false;
  return isBtcShortHorizonForecastQuery(query)
    && hasAbstainingMarkovDistributionForQuery(query, toolCalls);
}

export function buildAbstainingBtcShortHorizonForecastAnswer(
  query: string,
  toolCalls: ToolCallRecord[],
): string | null {
  if (!shouldPreserveAbstainingBtcShortHorizonForecast(query, toolCalls)) return null;

  const diagnostics = buildAbstainingMarkovAnswer(toolCalls);
  if (!diagnostics) return null;

  return [
    diagnostics,
    '',
    '## Decision guidance',
    'Treat this horizon as no-trade / no-calibrated-edge unless new horizon-matched terminal threshold markets appear. Any later fallback tools may still be useful for context, but they do not replace the abstained Markov forecast for BTC short horizons.',
  ].join('\n');
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

export function isForecastLabImprovementQuery(query: string): boolean {
  return routeForecastLabQuery(query).intent === 'improvement';
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

const FORECAST_LAB_APPROVAL_PATTERNS = [
  /\bapprove(?:d|s|ing)?\b/i,
  /\bgo ahead\b/i,
  /\bproceed\b/i,
  /\bpromote\b/i,
  /^\s*yes\b/i,
  /^\s*sure\b/i,
  /^\s*do it\b/i,
] as const;

const FORECAST_LAB_RESET_PATTERNS = [
  /\breset\b/i,
  /\brestore\b/i,
  /\broll\s+back\b/i,
] as const;

const SAFE_FORECAST_LAB_RUN_ID = /[A-Za-z0-9][A-Za-z0-9_.-]*/;

export interface ForecastLabPromotionApprovalHint {
  readonly profileId?: string;
  readonly sourceRunId?: string;
}

export interface ForecastLabResetHint {
  readonly profileId?: string;
  readonly mode: 'defaults' | 'last-known-good';
}

export interface ForecastLabComparisonHint {
  readonly profileId?: string;
  readonly mutationId?: string;
}

export interface ForecastLabResultsHint {
  readonly profileId?: string;
}

export interface ForecastLabMutatorListHint {
  readonly profileId?: string;
}

export interface ForecastLabKeepCurrentBestHint {
  readonly profileId?: string;
}

export interface ForecastLabCatalogExtensionHint {
  readonly profileId?: string;
}

export function isForecastLabPlanOnlyQuery(query: string): boolean {
  return FORECAST_LAB_PLAN_ONLY_PATTERNS.some((pattern) => pattern.test(query));
}

function extractForecastLabProfileId(text: string): string | undefined {
  const lower = text.toLowerCase();
  return listForecastLabProfiles().find((profile) => lower.includes(profile.id.toLowerCase()))?.id;
}

function extractForecastLabSourceRunId(text: string): string | undefined {
  const match = text.match(new RegExp(`(?:source\\s+run|run)\\s+(${SAFE_FORECAST_LAB_RUN_ID.source})`, 'i'));
  const candidate = match?.[1];
  if (!candidate) {
    return undefined;
  }
  if (!candidate.includes('-') && !candidate.includes('.')) {
    return undefined;
  }
  return candidate;
}

function isForecastLabApprovalIntent(query: string): boolean {
  if (
    /\bhow\s+(?:do\s+i\s+)?(?:approve|promote)\b/i.test(query)
    || /\bhow\s+to\s+(?:approve|promote)\b/i.test(query)
    || /\bwhat(?:'s| is)\b[\s\S]{0,40}\b(?:approve|promote)\b/i.test(query)
    || /\bwhich\s+command\b[\s\S]{0,40}\b(?:approve|promote)\b/i.test(query)
  ) {
    return false;
  }
  return FORECAST_LAB_APPROVAL_PATTERNS.some((pattern) => pattern.test(query));
}

function isForecastLabResetIntent(query: string): boolean {
  return FORECAST_LAB_RESET_PATTERNS.some((pattern) => pattern.test(query));
}

function isForecastLabComparisonIntent(query: string): boolean {
  const hasBestReference = /\bcurrent best\b|\bbest kept\b|\bbest lineage\b|\blatest kept\b/i.test(query);
  const hasBaselineReference = /\bshipped default baseline\b|\bshipped baseline\b|\bdefault baseline\b/i.test(query);
  const hasComparisonCue = /\bbetter than\b|\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b|\bstack up\b/i.test(query);
  return (hasBestReference && hasBaselineReference) || (hasBaselineReference && hasComparisonCue);
}

function extractForecastLabNamedMutationId(query: string): string | undefined {
  const candidate = query.match(/\b(markov-[A-Za-z0-9][A-Za-z0-9_.-]*)\b/i)?.[1];
  return candidate?.replace(/[.,!?;:]+$/, '');
}

function extractForecastLabComparisonMutationId(query: string): string | undefined {
  return extractForecastLabNamedMutationId(query);
}

function extractForecastLabRequestedMutationId(text: string): string | undefined {
  const matches = [...text.matchAll(/\brequested\s+mutator\s+id:\s*(markov-[A-Za-z0-9][A-Za-z0-9_.-]*)/gi)];
  const candidate = matches.at(-1)?.[1];
  return candidate?.replace(/[.,!?;:]+$/, '');
}

function isKnownForecastLabStructuredMutationId(mutationId: string): boolean {
  const candidate = mutationId.toLowerCase();
  return listForecastLabProfiles().some((profile) =>
    profile.mutation.mode === 'structured'
    && listForecastLabStructuredMutations(profile.id).some((entry) => entry.id.toLowerCase() === candidate),
  );
}

function isForecastLabMutatorVsActiveIntent(query: string, historyText = ''): boolean {
  const mutationId = extractForecastLabComparisonMutationId(query)
    ?? extractForecastLabRequestedMutationId(historyText);
  if (!mutationId) {
    return false;
  }

  const hasComparisonCue = /\bcompare\b|\bcomparison\b|\bversus\b|\bvs\.?\b/i.test(query);
  const hasMetricCue = /\baccurac\w*\b|\baccurace\b|\bnumbers\b|\bmetrics?\b/i.test(query);
  const hasActiveCue = /\bactive one\b|\bactive baseline\b|\bactive run\b|\bactive mutation\b|\blive one\b|\blive baseline\b|\blive run\b|\bcurrently live\b/i.test(query);
  const hasCreatedCandidateCue = /\bnew\s+mutat(?:e|or)\b|\bnew\s+one\b|\bthat\s+i\s+created\b|\bi\s+created\b|\bnot\s+promoted\b|\bunpromoted\b/i.test(query);
  const contextText = `${query}\n${historyText}`;
  const hasForecastLabContext =
    mutationId !== undefined
    || /\bforecast-lab\b/i.test(contextText)
    || /\bbtc-markov-ultra-short-horizon\b/i.test(contextText)
    || /\bactive baseline\b/i.test(contextText)
    || /\bpromoted parameters\b/i.test(contextText)
    || /src\/tools\/finance\/markov-distribution\.ts/i.test(contextText);

  return hasForecastLabContext && hasActiveCue && (hasComparisonCue || hasMetricCue || hasCreatedCandidateCue);
}

function isForecastLabResultsIntent(query: string): boolean {
  const hasResultsCue = /\bresult(?:s)?\b|\boutcome(?:s)?\b|\bstatus\b|\bsummary\b|\brecap\b|\bwhat happened\b/i.test(query);
  const hasRequestCue = /\bprovide\b|\bshow\b|\bgive\b|\bsummarize\b|\breport\b|\brecap\b|\bwhat\b/i.test(query);
  const hasWorkflowCue = /\boptimi[sz]e\b|\bimprov(?:e|ement)\b|\bworkflow\b|\bforecast-lab\b/i.test(query);
  return hasResultsCue && hasRequestCue && hasWorkflowCue;
}

function isForecastLabMutatorListIntent(query: string): boolean {
  const hasListCue = /\blist\b|\bshow\b|\bgive\b|\bwhat\b|\bwhich\b/i.test(query);
  const hasMutatorCue = /\bmutat(?:e|or|ors|ion|ions)\b/i.test(query);
  const hasCatalogCue = /\bids?\b|\bavail\w*\b|\bshipped\b|\bcatalog\b|\bcurrent\b/i.test(query);
  return hasMutatorCue && hasListCue && hasCatalogCue;
}

function isForecastLabKeepCurrentBestIntent(query: string): boolean {
  return /\bkeep\b[\s\S]{0,30}\bcurrent best candidate\b/i.test(query)
    || /\bkeep\b[\s\S]{0,20}\bbest candidate\b/i.test(query)
    || /\bkeep\b[\s\S]{0,20}\bbest run\b/i.test(query);
}

function hasForecastLabCatalogExtensionContext(text: string): boolean {
  return /\bforecast-lab\b/i.test(text)
    || /\bbtc-markov-ultra-short-horizon\b/i.test(text)
    || /\bbtc\s+1d\/2d\/3d\b/i.test(text)
    || /\bmarkov forecast workflow\b/i.test(text)
    || /\bultra-short-horizon\b/i.test(text)
    || /src\/tools\/finance\/markov-distribution\.ts/i.test(text);
}

function hasForecastLabCatalogImplementationCue(text: string): boolean {
  return /\bkeep it bounded\b/i.test(text)
    || /\badd\b[\s\S]{0,30}\bcatalog\b/i.test(text)
    || /\bvalidate\b[\s\S]{0,40}\b(?:walk-forward|gate)\b/i.test(text)
    || /\bsuggested starting values\b/i.test(text)
    || /\bsearch-replace\b/i.test(text)
    || /\bsoft-regime weighting\b/i.test(text);
}

function isForecastLabCatalogExtensionIntent(query: string, historyText = ''): boolean {
  const requestedMutationId = extractForecastLabNamedMutationId(query)
    ?? extractForecastLabRequestedMutationId(historyText);
  const hasImplementationExecutionCue =
    requestedMutationId !== undefined
    && !isKnownForecastLabStructuredMutationId(requestedMutationId)
    && /\bimplement\b|\bregister\b/i.test(query)
    && /\brun\b|\bre-?run\b|\bexecute\b/i.test(query);
  const hasMutatorCue = /\bmutator\b/i.test(query);
  const hasCatalogCue = /\bcatalog\b|\bshipped\b/i.test(query);
  const hasExtensionCue = /\bdesign\b|\badd\b|\bcreate\b|\bnew\b|\bextend\b|\boutside\b/i.test(query);
  const hasLineageCue = /\blineage\b|\bre-?run\b/i.test(query);
  const contextText = `${query}\n${historyText}`;
  return (
    hasImplementationExecutionCue
    && (
      /\bcatalog-extension plan\b/i.test(historyText)
      || /\brequested mutator id:\s*markov-/i.test(historyText)
      || hasForecastLabCatalogExtensionContext(contextText)
    )
  ) || (
    hasMutatorCue
    && hasCatalogCue
    && hasExtensionCue
    && (
      hasLineageCue
      || hasForecastLabCatalogExtensionContext(contextText)
      || hasForecastLabCatalogImplementationCue(query)
    )
  );
}

export function detectForecastLabPromotionApproval(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabPromotionApprovalHint | null {
  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const assistantHistory = recentTurns
    .filter((entry) => entry.role === 'assistant')
    .map((entry) => entry.content)
    .join('\n\n');
  const contextText = `${query}\n${assistantHistory}`;
  const hasForecastLabContext =
    /\bforecast-lab\b/i.test(contextText)
    || /\bpromotion-ready\b/i.test(contextText)
    || /\bapproval required\b/i.test(contextText);

  if (!hasForecastLabContext || !isForecastLabApprovalIntent(query)) {
    return null;
  }

  const profileId = extractForecastLabProfileId(contextText);
  const sourceRunId = extractForecastLabSourceRunId(query) ?? extractForecastLabSourceRunId(assistantHistory);

  if (!profileId && !sourceRunId && !/\bforecast-lab\b/i.test(query)) {
    return null;
  }

  return {
    ...(profileId ? { profileId } : {}),
    ...(sourceRunId ? { sourceRunId } : {}),
  };
}

export function detectForecastLabResetRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabResetHint | null {
  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const assistantHistory = recentTurns
    .filter((entry) => entry.role === 'assistant')
    .map((entry) => entry.content)
    .join('\n\n');
  const contextText = `${query}\n${assistantHistory}`;
  const hasForecastLabContext =
    /\bforecast-lab\b/i.test(contextText)
    || /\bactive baseline\b/i.test(contextText)
    || /\bpromoted parameters\b/i.test(contextText);

  if (!hasForecastLabContext || !isForecastLabResetIntent(query)) {
    return null;
  }

  const mode = /\bdefault(?:s)?\b|\bshipped defaults\b/i.test(query)
    ? 'defaults'
    : /\blast known good\b|\bprevious(?:ly)? activated\b|\bprevious baseline\b/i.test(query)
      ? 'last-known-good'
      : null;
  if (!mode) {
    return null;
  }

  const profileId = extractForecastLabProfileId(contextText);
  if (!profileId) {
    return null;
  }

  return { profileId, mode };
}

export function detectForecastLabComparisonRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabComparisonHint | null {
  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const historyText = recentTurns
    .map((entry) => entry.content)
    .join('\n\n');
  const contextText = `${query}\n${historyText}`;
  const mutationId = extractForecastLabComparisonMutationId(query)
    ?? extractForecastLabRequestedMutationId(historyText);
  const routedProfileId = routeForecastLabQuery(query).preferredProfileId ?? undefined;
  const profileId = extractForecastLabProfileId(contextText) ?? routedProfileId;

  if (isForecastLabComparisonIntent(query)) {
    return profileId ? { profileId } : {};
  }

  if (!isForecastLabMutatorVsActiveIntent(query, historyText)) {
    return null;
  }

  return {
    ...(profileId ? { profileId } : {}),
    ...(mutationId ? { mutationId } : {}),
  };
}

export function detectForecastLabResultsRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabResultsHint | null {
  if (!isForecastLabResultsIntent(query)) {
    return null;
  }

  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const historyText = recentTurns
    .map((entry) => entry.content)
    .join('\n\n');
  const contextText = `${query}\n${historyText}`;
  const routedProfileId = routeForecastLabQuery(query).preferredProfileId ?? undefined;
  const profileId = extractForecastLabProfileId(contextText) ?? routedProfileId;

  return profileId ? { profileId } : {};
}

export function detectForecastLabMutatorListRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabMutatorListHint | null {
  if (!isForecastLabMutatorListIntent(query)) {
    return null;
  }

  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const historyText = recentTurns
    .map((entry) => entry.content)
    .join('\n\n');
  const contextText = `${query}\n${historyText}`;
  const routedProfileId = routeForecastLabQuery(query).preferredProfileId ?? undefined;
  const profileId = extractForecastLabProfileId(contextText) ?? routedProfileId;

  return profileId ? { profileId } : {};
}

export function detectForecastLabKeepCurrentBestRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabKeepCurrentBestHint | null {
  if (!isForecastLabKeepCurrentBestIntent(query)) {
    return null;
  }

  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const historyText = recentTurns
    .map((entry) => entry.content)
    .join('\n\n');
  const explicitProfileId = extractForecastLabProfileId(query);
  const contextText = explicitProfileId ? `${query}\n${historyText}` : historyText;
  const hasForecastLabContext =
    /\bforecast-lab\b/i.test(query)
    || explicitProfileId !== undefined
    || /\bforecast-lab\b/i.test(historyText)
    || /\bapproval required\b/i.test(historyText)
    || /\bcurrent best\b/i.test(historyText)
    || /\bkept lineage\b/i.test(historyText)
    || /\bbtc-markov-ultra-short-horizon\b/i.test(historyText);
  if (!hasForecastLabContext) {
    return null;
  }

  const profileId = explicitProfileId ?? extractForecastLabProfileId(historyText);
  return profileId ? { profileId } : {};
}

export function detectForecastLabCatalogExtensionRequest(
  query: string,
  inMemoryHistory?: InMemoryChatHistory,
): ForecastLabCatalogExtensionHint | null {
  const recentTurns = inMemoryHistory?.getRecentTurns() ?? [];
  const historyText = recentTurns
    .map((entry) => entry.content)
    .join('\n\n');
  if (!isForecastLabCatalogExtensionIntent(query, historyText)) {
    return null;
  }
  const profileId = extractForecastLabProfileId(`${query}\n${historyText}`);

  return profileId ? { profileId } : {};
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

function hasMarketDataQuery(toolCalls: ToolCallRecord[], query: string): boolean {
  return toolCalls.some((call) => call.tool === 'get_market_data' && call.args['query'] === query);
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
        && predictionConfidence < RECOMMENDED_CONFIDENCE_THRESHOLD
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

function extractMarkovPredictionConfidenceForQuery(query: string, toolCalls: ToolCallRecord[]): number | null {
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
        if (isFiniteNumber(diagnostics['trustedAnchors'])) evidence.trusted_anchors = diagnostics['trustedAnchors'];
        if (isFiniteNumber(diagnostics['totalAnchors'])) evidence.total_anchors = diagnostics['totalAnchors'];
        if (typeof diagnostics['anchorQuality'] === 'string') evidence.anchor_quality = diagnostics['anchorQuality'];
      }
      evidence.summary = reasons.length > 0
        ? `Markov abstained: ${reasons.join('; ')}`
        : 'Markov abstained; treat Markov evidence as diagnostics only.';
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

function hasLowConfidenceBtcShortHorizonMarkov(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isBtcShortHorizonForecastQuery(query)) return false;
  const predictionConfidence = extractMarkovPredictionConfidenceForQuery(query, toolCalls);
  return predictionConfidence !== null && predictionConfidence < RECOMMENDED_CONFIDENCE_THRESHOLD;
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

function hasSuccessfulMarkovDistributionForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  return hasMarkovDistributionStatusForQuery(query, toolCalls, 'ok');
}

function hasAbstainingMarkovDistributionForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  return hasMarkovDistributionStatusForQuery(query, toolCalls, 'abstain');
}

function hasCompletedMarkovDistributionForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  return hasSuccessfulMarkovDistributionForQuery(query, toolCalls)
    || hasAbstainingMarkovDistributionForQuery(query, toolCalls);
}

function hasUsableOnchainResultForCryptoQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
  const desired = buildForcedOnchainArgs(query);
  if (!desired) return false;

  return toolCalls.some((call) =>
    call.tool === 'get_onchain_crypto'
    && call.args['ticker'] === desired.ticker
    && !hasErrorLikeToolResult(call.result)
    && hasNonEmptyParsedToolData(call),
  );
}

function hasUsableFixedIncomeResult(toolCalls: ToolCallRecord[]): boolean {
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

function hasPolymarketForecastCoverage(
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

function hasCryptoPolymarketForecastCoverage(query: string, toolCalls: ToolCallRecord[]): boolean {
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
  return /\b(direction|entry|enter|stop|stop-loss|target|take profit|leverage|leveraged|\d{1,3}(?:\.\d+)?\s*x|long|short|trade setup|position)\b/i.test(query);
}

function hasForecastArbitratorForQuery(query: string, toolCalls: ToolCallRecord[]): boolean {
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

function hasPrematureForecastArbitratorCall(response: AIMessage, query: string, toolCalls: ToolCallRecord[]): boolean {
  const requestedArbiter = response.tool_calls?.some((call) => call.name === 'forecast_arbitrator') ?? false;
  return requestedArbiter
    && buildForcedCryptoForecastMarkovArgs(query) !== null
    && !hasCompletedMarkovDistributionForQuery(query, toolCalls);
}

export function isAcceptedFirstPlanningToolCall(
  response: AIMessage,
  forecastLabRoutingHint?: ForecastLabRoutingHint | null,
  explicitlyRequestedSkill?: string | null,
): boolean {
  const firstToolCall = response.tool_calls?.[0];
  if (!firstToolCall) return true;
  if (firstToolCall.name === 'sequential_thinking') return true;
  if (
    explicitlyRequestedSkill
    && firstToolCall.name === 'skill'
    && firstToolCall.args?.skill === explicitlyRequestedSkill
  ) {
    return true;
  }
  return Boolean(
    forecastLabRoutingHint?.shouldInvokeSkill
      && firstToolCall.name === 'skill'
      && firstToolCall.args?.skill === 'forecast-lab',
  );
}

function detectBtcShortHorizonDisagreement(query: string, toolCalls: ToolCallRecord[]): boolean {
  if (!isCryptoForecastQuery(query)) return false;
  const ticker = inferDistributionTicker(query);
  const horizon = inferDistributionHorizon(query);
  if ((ticker !== 'BTC' && ticker !== 'BTC-USD') || horizon === null || horizon > 14) return false;

  const markovReturn = extractMarkovReturnForQuery(query, toolCalls);
  const polymarketReturn = extractPolymarketForecastReturnForQuery(query, toolCalls);
  if (markovReturn === null || polymarketReturn === null) return false;

  return markovReturn > 0.01 && polymarketReturn <= 0;
}

export function buildForecastDisagreementPrefix(query: string, toolCalls: ToolCallRecord[]): string | null {
  if (!detectBtcShortHorizonDisagreement(query, toolCalls)) return null;

  return [
    '## Warning: BTC short-horizon signals are mixed',
    '',
    'Markov and Polymarket are pointing in different directions for this BTC short-horizon forecast. Read any directional takeaway below as mixed evidence with moderated confidence, not a high-conviction signal.',
    '',
    '---',
    '',
  ].join('\n');
}

export function buildLowConfidenceBtcShortHorizonForecastPrefix(query: string, toolCalls: ToolCallRecord[]): string | null {
  if (!hasLowConfidenceBtcShortHorizonMarkov(query, toolCalls)) return null;

  const predictionConfidence = extractMarkovPredictionConfidenceForQuery(query, toolCalls);
  const confidenceText = predictionConfidence !== null
    ? predictionConfidence.toFixed(2)
    : 'N/A';

  return [
    '## Warning: BTC short-horizon selective Markov gate did not clear',
    '',
    `The Markov run completed, but prediction confidence ${confidenceText} is below the ${RECOMMENDED_CONFIDENCE_THRESHOLD.toFixed(2)} selective threshold. Do not treat the Markov direction here as part of the aggregate selective-accuracy slice or as a validated BTC edge. Read any forecast below as fallback context, not a selective Markov signal.`,
    '',
    '---',
    '',
  ].join('\n');
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
  return Number.isFinite(predictionConfidence) && predictionConfidence < RECOMMENDED_CONFIDENCE_THRESHOLD;
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
    const forecastingConfig = loadConfig().forecasting;
    const forecastLabRoutingHint = getForecastLabRoutingHint(query, {
      enableAutoRoute: forecastingConfig?.enableForecastLabAutoRoute,
      enableSkillHint: forecastingConfig?.enableForecastLabSkillHint,
    });
    const forecastLabResetRequest = detectForecastLabResetRequest(query, inMemoryHistory);
    const forecastLabPromotionApproval = detectForecastLabPromotionApproval(query, inMemoryHistory);
    const forecastLabKeepCurrentBestRequest = detectForecastLabKeepCurrentBestRequest(query, inMemoryHistory);
    const forecastLabCatalogExtensionRequest = detectForecastLabCatalogExtensionRequest(query, inMemoryHistory);
    const forecastLabComparisonRequest = detectForecastLabComparisonRequest(query, inMemoryHistory);
    const forecastLabResultsRequest = detectForecastLabResultsRequest(query, inMemoryHistory);
    const forecastLabMutatorListRequest = detectForecastLabMutatorListRequest(query, inMemoryHistory);
    if (forecastLabResetRequest) {
      yield* this.runForecastLabResetFlow(ctx, forecastLabResetRequest);
      return;
    }
    if (forecastLabPromotionApproval) {
      yield* this.runForecastLabApprovalFlow(ctx, forecastLabPromotionApproval);
      return;
    }
    if (forecastLabKeepCurrentBestRequest) {
      yield* this.runForecastLabResultsFlow(ctx, query, forecastLabKeepCurrentBestRequest);
      return;
    }
    if (forecastLabCatalogExtensionRequest) {
      yield* this.runForecastLabCatalogExtensionFlow(ctx, query, forecastLabCatalogExtensionRequest);
      return;
    }
    if (forecastLabResultsRequest) {
      yield* this.runForecastLabResultsFlow(ctx, query, forecastLabResultsRequest);
      return;
    }
    if (forecastLabMutatorListRequest) {
      yield* this.runForecastLabMutatorListFlow(ctx, query, forecastLabMutatorListRequest);
      return;
    }
    if (forecastLabComparisonRequest) {
      yield* this.runForecastLabComparisonFlow(ctx, query, forecastLabComparisonRequest);
      return;
    }
    if (forecastLabRoutingHint?.shouldInvokeSkill) {
      yield* this.runForecastLabImprovementFlow(ctx, query, forecastLabRoutingHint, isForecastLabPlanOnlyQuery(query));
      return;
    }

    // Build initial prompt with conversation history context
    let currentPrompt = this.buildInitialPrompt(query, inMemoryHistory, forecastLabRoutingHint);

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

    const explicitlyRequestedSkill = detectExplicitSkillRequest(query);

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

    if (explicitlyRequestedSkill) {
      for await (const event of this.toolExecutor.executeTool(
        'skill',
        { skill: explicitlyRequestedSkill },
        ctx,
      )) {
        yield event;
      }

      const toolResults = ctx.scratchpad.getToolResults().trim();
      if (toolResults) {
        currentPrompt = `${currentPrompt}\n\nData retrieved from tool calls:\n${toolResults}`;
      }

      sequentialThinkingUsed = true;
    }

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
                ctx.scratchpad.formatToolUsageForPrompt(),
                forecastLabRoutingHint,
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
              forecastLabRoutingHint,
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
              forecastLabRoutingHint,
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
              forecastLabRoutingHint,
            );
            continue;
          }
        }

        const abstainingBtcForecastAnswer = buildAbstainingBtcShortHorizonForecastAnswer(
          query,
          ctx.scratchpad.getToolCallRecords(),
        );
        if (abstainingBtcForecastAnswer) {
          yield* this.handleDirectResponse(abstainingBtcForecastAnswer, ctx, currentPrompt);
          return;
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
        if (
          firstTool
          && !isAcceptedFirstPlanningToolCall(
            response as AIMessage,
            forecastLabRoutingHint,
            explicitlyRequestedSkill,
          )
        ) {
          if (sequentialThinkingRetries < MAX_ST_RETRIES) {
            sequentialThinkingRetries++;
            ctx.iteration--; // don't charge this iteration
            currentPrompt = forecastLabRoutingHint?.shouldInvokeSkill
              ? `${currentPrompt}\n\nIMPORTANT REMINDER: This routed forecast-lab improvement query must start with skill(\"forecast-lab\") or sequential_thinking. Do NOT start with ordinary forecast/data tools.`
              : `${currentPrompt}\n\nIMPORTANT REMINDER: You MUST call sequential_thinking FIRST before calling any other tool. Start with sequential_thinking to plan your approach, then proceed.`;
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
        if (
          stToolCalls.some((tc) => tc.name === 'sequential_thinking')
          || (
            forecastLabRoutingHint?.shouldInvokeSkill
            && stToolCalls.some((tc) => tc.name === 'skill' && tc.args?.skill === 'forecast-lab')
          )
        ) {
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
            forecastLabRoutingHint,
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
            forecastLabRoutingHint,
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
            forecastLabRoutingHint,
          );
          continue;
        }
      }

      // Count sequential_thinking calls before executing tools (needed for nudge below).
      const toolCalls = (response as AIMessage).tool_calls ?? [];
      if (hasPrematureForecastArbitratorCall(response as AIMessage, query, ctx.scratchpad.getToolCallRecords())) {
        const forced = yield* this.forceCryptoForecastTools(ctx);
        if (forced) {
          yield* this.manageContextThreshold(ctx, query, memoryFlushState);
          currentPrompt = buildIterationPrompt(
            query,
            ctx.scratchpad.getToolResults(),
            ctx.scratchpad.formatToolUsageForPrompt(),
            forecastLabRoutingHint,
          );
          continue;
        }
      }

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
        ctx.scratchpad.formatToolUsageForPrompt(),
        forecastLabRoutingHint,
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

    const abstainingBtcForecastAnswer = buildAbstainingBtcShortHorizonForecastAnswer(
      query,
      ctx.scratchpad.getToolCallRecords(),
    );
    if (abstainingBtcForecastAnswer) {
      yield* this.handleDirectResponse(abstainingBtcForecastAnswer, ctx, currentPrompt);
      return;
    }

    const synthesisPrompt = hasMeaningfulResearch
      ? buildIterationPrompt(
          query,
          toolResults,
          ctx.scratchpad.formatToolUsageForPrompt(),
          forecastLabRoutingHint,
        ) +
          `\n\n[SYSTEM NOTE: You have reached the maximum number of research steps (${this.maxIterations}). ` +
          `You MUST now write your best-effort final answer using ONLY the data gathered above. ` +
          `Start your response with "**[Best-effort summary — research may be incomplete]**\\n\\n" ` +
          `then provide the most useful analysis you can from the available data. Do NOT call any more tools.]`
      : query;

    yield* this.handleDirectResponse('', ctx, synthesisPrompt);
  }

  private extractForecastLabAnswer(ctx: RunContext, fallback: string): string {
    const toolCalls = ctx.scratchpad.getToolCallRecords();
    for (let index = toolCalls.length - 1; index >= 0; index -= 1) {
      const call = toolCalls[index];
      if (call.tool !== 'forecast_lab_run') continue;
      const answer = extractForecastLabRunToolAnswer(call.result);
      if (answer) {
        return answer;
      }
    }

    return fallback;
  }

  private async *runForecastLabImprovementFlow(
    ctx: RunContext,
    query: string,
    forecastLabRoutingHint: ForecastLabRoutingHint,
    planOnly: boolean,
  ): AsyncGenerator<AgentEvent, void> {
    ctx.forecastLabGuard = {
      recommendedProfileId: forecastLabRoutingHint.recommendedProfileId ?? null,
    };
    try {
      for await (const event of this.toolExecutor.executeTool(
        'skill',
        { skill: 'forecast-lab' },
        ctx,
      )) {
        yield event;
      }

      for await (const event of this.toolExecutor.executeTool(
        'forecast_lab_run',
        {
          action: 'guided-improve',
          query,
          ...(forecastLabRoutingHint.recommendedProfileId
            ? { profileId: forecastLabRoutingHint.recommendedProfileId }
            : {}),
          ...(forecastLabRoutingHint.requestedMutatorId
            ? { mutator: forecastLabRoutingHint.requestedMutatorId }
            : {}),
          ...(planOnly ? { execute: false } : {}),
          routingSource: 'auto-routed',
        },
        ctx,
      )) {
        yield event;
      }
    } finally {
      delete ctx.forecastLabGuard;
    }

    const fallback = planOnly
      ? 'Forecast-lab plan generated.'
      : 'Forecast-lab guided improvement finished.';
    yield* this.handleDirectResponse(this.extractForecastLabAnswer(ctx, fallback), ctx);
  }

  private async *runForecastLabApprovalFlow(
    ctx: RunContext,
    approvalHint: ForecastLabPromotionApprovalHint,
  ): AsyncGenerator<AgentEvent, void> {
    ctx.forecastLabGuard = {
      recommendedProfileId: approvalHint.profileId ?? null,
    };
    try {
      for await (const event of this.toolExecutor.executeTool(
        'forecast_lab_run',
        {
          action: 'promote-approved',
          ...(approvalHint.profileId ? { profileId: approvalHint.profileId } : {}),
          ...(approvalHint.sourceRunId ? { sourceRunId: approvalHint.sourceRunId } : {}),
        },
        ctx,
      )) {
        yield event;
      }
    } finally {
      delete ctx.forecastLabGuard;
    }

    yield* this.handleDirectResponse(
      this.extractForecastLabAnswer(ctx, 'Forecast-lab promotion request finished.'),
      ctx,
    );
  }

  private async *runForecastLabResetFlow(
    ctx: RunContext,
    resetHint: ForecastLabResetHint,
  ): AsyncGenerator<AgentEvent, void> {
    ctx.forecastLabGuard = {
      recommendedProfileId: resetHint.profileId ?? null,
    };
    try {
      for await (const event of this.toolExecutor.executeTool(
        'forecast_lab_run',
        {
          action: 'reset-live',
          ...(resetHint.profileId ? { profileId: resetHint.profileId } : {}),
          resetMode: resetHint.mode,
        },
        ctx,
      )) {
        yield event;
      }
    } finally {
      delete ctx.forecastLabGuard;
    }

    yield* this.handleDirectResponse(
      this.extractForecastLabAnswer(ctx, 'Forecast-lab reset request finished.'),
      ctx,
    );
  }

  private async *runForecastLabComparisonFlow(
    ctx: RunContext,
    query: string,
    comparisonHint: ForecastLabComparisonHint,
  ): AsyncGenerator<AgentEvent, void> {
    ctx.forecastLabGuard = {
      recommendedProfileId: comparisonHint.profileId ?? null,
    };
    try {
      for await (const event of this.toolExecutor.executeTool(
        'forecast_lab_run',
        {
          action: 'compare-best-vs-shipped',
          query,
          ...(comparisonHint.profileId ? { profileId: comparisonHint.profileId } : {}),
          ...(comparisonHint.mutationId ? { mutationId: comparisonHint.mutationId } : {}),
        },
        ctx,
      )) {
        yield event;
      }
    } finally {
      delete ctx.forecastLabGuard;
    }

    yield* this.handleDirectResponse(
      this.extractForecastLabAnswer(ctx, 'Forecast-lab comparison finished.'),
      ctx,
    );
  }

  private async *runForecastLabResultsFlow(
    ctx: RunContext,
    query: string,
    resultsHint: ForecastLabResultsHint,
  ): AsyncGenerator<AgentEvent, void> {
    ctx.forecastLabGuard = {
      recommendedProfileId: resultsHint.profileId ?? null,
    };
    try {
      for await (const event of this.toolExecutor.executeTool(
        'forecast_lab_run',
        {
          action: 'compare-best-vs-shipped',
          query,
          ...(resultsHint.profileId ? { profileId: resultsHint.profileId } : {}),
        },
        ctx,
      )) {
        yield event;
      }
    } finally {
      delete ctx.forecastLabGuard;
    }

    yield* this.handleDirectResponse(
      this.extractForecastLabAnswer(ctx, 'Forecast-lab results retrieved.'),
      ctx,
    );
  }

  private async *runForecastLabMutatorListFlow(
    ctx: RunContext,
    query: string,
    mutatorListHint: ForecastLabMutatorListHint,
  ): AsyncGenerator<AgentEvent, void> {
    ctx.forecastLabGuard = {
      recommendedProfileId: mutatorListHint.profileId ?? null,
    };
    try {
      for await (const event of this.toolExecutor.executeTool(
        'forecast_lab_run',
        {
          action: 'list-mutators',
          query,
          ...(mutatorListHint.profileId ? { profileId: mutatorListHint.profileId } : {}),
        },
        ctx,
      )) {
        yield event;
      }
    } finally {
      delete ctx.forecastLabGuard;
    }

    yield* this.handleDirectResponse(
      this.extractForecastLabAnswer(ctx, 'Forecast-lab mutator list retrieved.'),
      ctx,
    );
  }

  private async *runForecastLabCatalogExtensionFlow(
    ctx: RunContext,
    query: string,
    catalogExtensionHint: ForecastLabCatalogExtensionHint,
  ): AsyncGenerator<AgentEvent, void> {
    ctx.forecastLabGuard = {
      recommendedProfileId: catalogExtensionHint.profileId ?? null,
    };
    try {
      for await (const event of this.toolExecutor.executeTool(
        'forecast_lab_run',
        {
          action: 'catalog-extension-plan',
          query,
          ...(catalogExtensionHint.profileId ? { profileId: catalogExtensionHint.profileId } : {}),
        },
        ctx,
      )) {
        yield event;
      }
    } finally {
      delete ctx.forecastLabGuard;
    }

    yield* this.handleDirectResponse(
      this.extractForecastLabAnswer(ctx, 'Forecast-lab catalog-extension guidance generated.'),
      ctx,
    );
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

    const getToolCalls = () => ctx.scratchpad.getToolCallRecords();

    const marketDataArgs = buildForcedMarketDataArgs(ctx.query);
    if (
      marketDataArgs
      && extractCurrentPriceFromMarketDataQuery(getToolCalls(), marketDataArgs.query) === null
    ) {
      const args = marketDataArgs;
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

    if (extractSentimentScoreFromToolCalls(getToolCalls()) === null) {
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

    if (!hasCompletedMarkovDistributionForQuery(ctx.query, getToolCalls())) {
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

    if (!hasCryptoPolymarketForecastCoverage(ctx.query, getToolCalls())) {
      const args = buildForcedPolymarketForecastArgs(ctx.query, getToolCalls());
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

    if (shouldRerunPolymarketForecastWithMarkov(ctx.query, getToolCalls())) {
      const args = buildForcedPolymarketForecastArgs(ctx.query, getToolCalls());
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

    if (!hasUsableOnchainResultForCryptoQuery(ctx.query, getToolCalls())) {
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

    if (!hasUsableFixedIncomeResult(getToolCalls())) {
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

    if (shouldForceForecastArbitrator(ctx.query, getToolCalls())) {
      const args = buildForcedForecastArbiterArgs(ctx.query, getToolCalls());
      if (args) {
        let ok = true;
        for await (const event of this.toolExecutor.executeTool('forecast_arbitrator', args, ctx)) {
          yield event;
          if (event.type === 'tool_error' || event.type === 'tool_denied') {
            ok = false;
            break;
          }
        }
        forcedAny = forcedAny || ok;
      }
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
      && !hasMarketDataQuery(getToolCalls(), marketDataArgs.query)
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
    const lowConfidencePrefix = buildLowConfidenceBtcShortHorizonForecastPrefix(ctx.query, toolCalls);
    const disagreementPrefix = buildForecastDisagreementPrefix(ctx.query, toolCalls);
    const baseText = stripThinkingTags(fallbackText);
    const prefixText = `${warningPrefix ?? ''}${lowConfidencePrefix ?? ''}${disagreementPrefix ?? ''}`;
    const text = baseText ? `${prefixText}${baseText}` : prefixText;

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
      const timeoutSignal = AbortSignal.timeout(getLlmCallTimeoutMs());
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

      if (!streamedAnswer && prefixText) {
        const CHUNK_SIZE = 6;
        for (let i = 0; i < prefixText.length; i += CHUNK_SIZE) {
          const chunk = prefixText.slice(i, i + CHUNK_SIZE);
          streamedAnswer += chunk;
          yield { type: 'answer_chunk', chunk } as AnswerChunkEvent;
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
    inMemoryChatHistory?: InMemoryChatHistory,
    forecastLabRoutingHint?: ForecastLabRoutingHint | null,
  ): string {
    if (!inMemoryChatHistory?.hasMessages()) {
      return injectForecastLabRoutingHint(query, forecastLabRoutingHint);
    }

    const recentTurns = inMemoryChatHistory.getRecentTurns();
    if (recentTurns.length === 0) {
      return injectForecastLabRoutingHint(query, forecastLabRoutingHint);
    }

    return injectForecastLabRoutingHint(buildHistoryContext({
      entries: recentTurns,
      currentMessage: query,
    }), forecastLabRoutingHint);
  }
}
