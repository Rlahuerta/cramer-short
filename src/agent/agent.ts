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
import { extractSignals as extractSignalsFn } from '../tools/finance/signal-extractor.js';
import { fetchPolymarketMarkets } from '../tools/finance/polymarket.js';
import type { MarkovDistributionPoint } from '../tools/finance/markov-distribution.js';
import { resolveProvider } from '../providers.js';
import type { ToolCallRecord } from './scratchpad.js';
import {
  getForecastLabRoutingHint,
  type ForecastLabRoutingHint,
} from './forecast-lab-routing.js';
import { routeForecastLabQuery } from '../experiments/forecast-lab/router.js';
import { listForecastLabProfiles, listForecastLabStructuredMutations } from '../experiments/forecast-lab/profiles.js';
import { extractForecastLabRunToolAnswer } from '../tools/forecast-lab-run.js';
import {
  buildForcedCryptoForecastMarkovArgs,
  buildForcedFixedIncomeArgs,
  buildForcedForecastArbiterArgs,
  buildForcedGoldCombinedForecastArbiterArgs,
  buildForcedMarkovArgs,
  buildForcedMarketDataArgs,
  buildForcedNonCryptoMarketDataArgs,
  buildForcedNonCryptoPolymarketForecastArgs,
  buildForcedOnchainArgs,
  buildForcedPolymarketForecastArgs,
  buildForcedSocialSentimentArgs,
  detectBtcShortHorizonDisagreement,
  detectExplicitSkillRequest,
  extractCurrentPriceFromMarketDataQuery,
  extractCurrentPriceFromToolCalls,
  extractMarkovPredictionConfidenceForQuery,
  extractMarkovReturnFromToolCalls,
  extractSentimentScoreFromToolCalls,
  getBtcSelectiveMarkovConfidenceThreshold,
  hasAbstainingMarkovDistributionForQuery,
  hasCompletedMarkovDistributionForQuery,
  hasCryptoPolymarketForecastCoverage,
  hasForecastArbitratorForQuery,
  hasLowConfidenceBtcShortHorizonMarkov,
  hasMarketDataQuery,
  hasPolymarketForecastCoverage,
  hasUsableFixedIncomeResult,
  hasUsableOnchainResultForCryptoQuery,
  inferBtcShortHorizonForecastHorizon,
  inferDistributionHorizon,
  inferDistributionTicker,
  inferMarkovQueryHorizon,
  inferTrajectoryRequest,
  isBtcShortHorizonForecastQuery,
  isCryptoForecastQuery,
  isDistributionQuery,
  isExplicitGoldCombinedMarkovPolymarketRequest,
  isExplicitPolymarketForecastRequest,
  isExplicitTerminalDistributionQuery,
  isForecastLabImprovementQuery,
  isForecastLabPlanOnlyQuery,
  isNonCryptoForecastQuery,
  shouldForceCryptoForecastTools,
  shouldForceForecastArbitrator,
  shouldForceGoldCombinedForecastArbitrator,
  shouldForceGoldCombinedForecastTools,
  shouldForceMarkovDistribution,
  shouldForceNonCryptoForecastFallback,
  shouldInjectBtcShortHorizonLowConfidencePrompt,
  shouldInjectBtcShortHorizonMixedEvidencePrompt,
  shouldRerunPolymarketForecastWithMarkov,
} from './query-router.js';

export {
  buildForcedCryptoForecastMarkovArgs,
  buildForcedFixedIncomeArgs,
  buildForcedForecastArbiterArgs,
  buildForcedGoldCombinedForecastArbiterArgs,
  buildForcedMarkovArgs,
  buildForcedMarketDataArgs,
  buildForcedNonCryptoMarketDataArgs,
  buildForcedNonCryptoPolymarketForecastArgs,
  buildForcedOnchainArgs,
  buildForcedPolymarketForecastArgs,
  buildForcedSocialSentimentArgs,
  detectExplicitSkillRequest,
  extractCurrentPriceFromToolCalls,
  extractMarkovReturnFromToolCalls,
  extractSentimentScoreFromToolCalls,
  inferDistributionHorizon,
  inferDistributionTicker,
  inferTrajectoryRequest,
  isCryptoForecastQuery,
  isExplicitGoldCombinedMarkovPolymarketRequest,
  isExplicitPolymarketForecastRequest,
  isExplicitTerminalDistributionQuery,
  isForecastLabImprovementQuery,
  isForecastLabPlanOnlyQuery,
  isNonCryptoForecastQuery,
  shouldForceCryptoForecastTools,
  shouldForceForecastArbitrator,
  shouldForceGoldCombinedForecastArbitrator,
  shouldForceGoldCombinedForecastTools,
  shouldForceMarkovDistribution,
  shouldForceNonCryptoForecastFallback,
  shouldInjectBtcShortHorizonLowConfidencePrompt,
  shouldInjectBtcShortHorizonMixedEvidencePrompt,
  shouldRerunPolymarketForecastWithMarkov,
} from './query-router.js';

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

function inferRequestedBucketCount(query: string): number | null {
  const match = query.match(/\b(\d+)\s*(?:parts|buckets|bins|segments)\b/i);
  if (!match) return null;
  const parsed = Number.parseInt(match[1]!, 10);
  return Number.isInteger(parsed) && parsed >= 2 ? parsed : null;
}

function interpolateSurvivalProbability(
  distribution: MarkovDistributionPoint[],
  price: number,
): number | null {
  if (distribution.length === 0) return null;

  const sorted = [...distribution].sort((a, b) => a.price - b.price);
  if (price <= sorted[0]!.price) return sorted[0]!.probability;
  if (price >= sorted[sorted.length - 1]!.price) return sorted[sorted.length - 1]!.probability;

  for (let i = 0; i < sorted.length - 1; i += 1) {
    const left = sorted[i]!;
    const right = sorted[i + 1]!;
    if (price < left.price || price > right.price) continue;
    if (right.price === left.price) return left.probability;

    const weight = (price - left.price) / (right.price - left.price);
    return left.probability + ((right.probability - left.probability) * weight);
  }

  return sorted[sorted.length - 1]!.probability;
}

function estimateBucketProbabilityPct(
  distribution: MarkovDistributionPoint[],
  lower: number | null,
  upper: number | null,
): number | null {
  const lowerSurvival = lower === null ? 1 : interpolateSurvivalProbability(distribution, lower);
  const upperSurvival = upper === null ? 0 : interpolateSurvivalProbability(distribution, upper);
  if (lowerSurvival === null || upperSurvival === null) return null;
  return Math.max(0, Math.min(100, (lowerSurvival - upperSurvival) * 100));
}

function formatDensityPrice(value: number): string {
  return `$${value.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function formatDensityRange(lower: number | null, upper: number | null): string {
  if (lower === null && upper === null) return 'N/A';
  if (lower === null && upper !== null) return `< ${formatDensityPrice(upper)}`;
  if (lower !== null && upper === null) return `> ${formatDensityPrice(lower)}`;
  return `${formatDensityPrice(lower!)}–${formatDensityPrice(upper!)}`;
}

function buildDensityThresholds(
  minPrice: number,
  maxPrice: number,
  bucketCount: number,
): number[] {
  const thresholdCount = bucketCount - 1;
  if (thresholdCount <= 0) return [];

  const useLogSpacing = minPrice > 0;
  return Array.from({ length: thresholdCount }, (_, index) => {
    const weight = (index + 1) / bucketCount;
    if (useLogSpacing) {
      const logPrice = Math.log(minPrice) + ((Math.log(maxPrice) - Math.log(minPrice)) * weight);
      return Math.exp(logPrice);
    }
    return minPrice + ((maxPrice - minPrice) * weight);
  });
}

function buildCanonicalDensityTable(query: string, toolCalls: ToolCallRecord[]): string | null {
  const requestedBucketCount = inferRequestedBucketCount(query);
  if (requestedBucketCount === null) return null;

  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferMarkovQueryHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i -= 1) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution' || data['status'] !== 'ok') continue;
    if (!Array.isArray(data['distribution'])) continue;

    const distribution = data['distribution']
      .filter((point): point is MarkovDistributionPoint => (
        !!point
        && typeof point === 'object'
        && isFinitePositiveNumber((point as Record<string, unknown>)['price'])
        && isFiniteNumber((point as Record<string, unknown>)['probability'])
      ))
      .map((point) => point as MarkovDistributionPoint)
      .sort((a, b) => a.price - b.price);

    if (distribution.length < 2) continue;

    const minPrice = distribution[0]!.price;
    const maxPrice = distribution[distribution.length - 1]!.price;
    if (!(minPrice > 0) || !(maxPrice > minPrice)) continue;

    const thresholds = buildDensityThresholds(minPrice, maxPrice, requestedBucketCount);

    const rows: string[] = [];
    for (let bucketIndex = 0; bucketIndex < requestedBucketCount; bucketIndex += 1) {
      const lower = bucketIndex === 0 ? null : thresholds[bucketIndex - 1]!;
      const upper = bucketIndex === requestedBucketCount - 1 ? null : thresholds[bucketIndex]!;
      const probabilityPct = estimateBucketProbabilityPct(distribution, lower, upper);
      if (probabilityPct === null) continue;

      rows.push(
        `| ${bucketIndex + 1} | ${formatDensityRange(lower, upper)} | ${probabilityPct.toFixed(2)}% |`,
      );
    }

    if (rows.length < requestedBucketCount) continue;

    return [
      `## ${requestedBucketCount}-Part Density Probability Table`,
      '',
      'Canonical scenario breakdown (P(bucket) = probability mass in each price range):',
      '',
      '| Bucket | Price Range | P(bucket) |',
      '|--------|-------------|-----------|',
      ...rows,
    ].join('\n');
  }

  return null;
}

export function ensureStructuredDensityTable(
  answer: string,
  query: string,
  toolCalls: ToolCallRecord[],
): string {
  if (inferRequestedBucketCount(query) === null) return answer;
  const canonicalTable = buildCanonicalDensityTable(query, toolCalls);
  if (!canonicalTable) return answer;

  const densitySectionPattern = /##\s+.*density probability table[\s\S]*?(?=\n---\n|\n##\s|\n###\s|$)/i;
  if (densitySectionPattern.test(answer)) {
    return answer.replace(densitySectionPattern, canonicalTable);
  }

  return `${canonicalTable}\n\n---\n\n${answer}`;
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


function normalizeExplicitGoldCombinedToolCalls(
  response: AIMessage,
  query: string,
  toolCalls: ToolCallRecord[],
): void {
  if (!isExplicitGoldCombinedMarkovPolymarketRequest(query) || !response.tool_calls?.length) return;

  const forcedMarkovArgs = buildForcedMarkovArgs(query);
  const forcedMarketDataArgs = buildForcedNonCryptoMarketDataArgs(query);
  const forcedForecastArgs = buildForcedNonCryptoPolymarketForecastArgs(query, []);
  const forcedArbitratorArgs = buildForcedGoldCombinedForecastArbiterArgs(query, toolCalls);
  const forcedTicker = forcedMarkovArgs?.ticker ?? forcedForecastArgs?.ticker ?? inferDistributionTicker(query);
  const forcedHorizon = forcedMarkovArgs?.horizon ?? forcedForecastArgs?.horizon_days ?? inferDistributionHorizon(query) ?? 7;

  response.tool_calls = response.tool_calls.flatMap((toolCall) => {
    if (toolCall.name === 'social_sentiment' || toolCall.name === 'get_onchain_crypto') {
      return [];
    }

    if (toolCall.name === 'get_market_data' && forcedMarketDataArgs) {
      return [{ ...toolCall, args: forcedMarketDataArgs }];
    }

    if (toolCall.name === 'markov_distribution' && forcedMarkovArgs) {
      const args: Record<string, unknown> = {
        ...(toolCall.args as Record<string, unknown>),
        ...forcedMarkovArgs,
      };
      if (!('trajectory' in forcedMarkovArgs)) {
        delete args['trajectory'];
        delete args['trajectoryDays'];
      }
      return [{ ...toolCall, args }];
    }

    if (toolCall.name === 'polymarket_forecast' && forcedForecastArgs) {
      return [{
        ...toolCall,
        args: {
          ...(toolCall.args as Record<string, unknown>),
          ...forcedForecastArgs,
        },
      }];
    }

    if (toolCall.name === 'forecast_arbitrator') {
      if (forcedArbitratorArgs) {
        return [{ ...toolCall, args: forcedArbitratorArgs }];
      }
      if (!forcedTicker) return [];
      return [{
        ...toolCall,
        args: {
          ...(toolCall.args as Record<string, unknown>),
          ticker: forcedTicker,
          horizon_days: forcedHorizon,
        },
      }];
    }

    return [toolCall];
  });
}



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

const FORECAST_LAB_NAMED_MUTATION_ID = /(?<![A-Za-z0-9_.-])((?:gold-)?markov-[A-Za-z0-9][A-Za-z0-9_.-]*)\b/i;
const FORECAST_LAB_REQUESTED_MUTATION_ID = /\brequested\s+mutator\s+id:\s*((?:gold-)?markov-[A-Za-z0-9][A-Za-z0-9_.-]*)/gi;
const FORECAST_LAB_REQUESTED_MUTATION_ID_CUE = /\brequested mutator id:\s*(?:gold-)?markov-/i;

function extractForecastLabNamedMutationId(query: string): string | undefined {
  const candidate = query.match(FORECAST_LAB_NAMED_MUTATION_ID)?.[1];
  return candidate?.replace(/[.,!?;:]+$/, '');
}

function extractForecastLabComparisonMutationId(query: string): string | undefined {
  return extractForecastLabNamedMutationId(query);
}

function extractForecastLabRequestedMutationId(text: string): string | undefined {
  const matches = [...text.matchAll(FORECAST_LAB_REQUESTED_MUTATION_ID)];
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
      || FORECAST_LAB_REQUESTED_MUTATION_ID_CUE.test(historyText)
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
  const selectiveBtcThreshold = getBtcSelectiveMarkovConfidenceThreshold();
  const confidenceText = predictionConfidence !== null
    ? predictionConfidence.toFixed(2)
    : 'N/A';

  return [
    '## Warning: BTC short-horizon selective Markov gate did not clear',
    '',
    `The Markov run completed, but prediction confidence ${confidenceText} is below the ${selectiveBtcThreshold.toFixed(2)} selective threshold. Do not treat the Markov direction here as part of the aggregate selective-accuracy slice or as a validated BTC edge. Read any forecast below as fallback context, not a selective Markov signal.`,
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
    const tools = config.tools ?? getTools(model, { watchlistEntries: config.watchlistEntries });
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
      config.toolDescriptionsOverride,
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

        if (shouldForceGoldCombinedForecastTools(query, ctx.scratchpad.getToolCallRecords())) {
          const forced = yield* this.forceGoldCombinedForecastTools(ctx);
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

      if (sequentialThinkingUsed && shouldForceGoldCombinedForecastTools(query, ctx.scratchpad.getToolCallRecords())) {
        const forced = yield* this.forceGoldCombinedForecastTools(ctx);
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

      normalizeExplicitGoldCombinedToolCalls(
        response as AIMessage,
        query,
        ctx.scratchpad.getToolCallRecords(),
      );

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

  /** Force the explicit GOLD combined seam after a usable Markov result exists. */
  private async *forceGoldCombinedForecastTools(
    ctx: RunContext,
  ): AsyncGenerator<AgentEvent, boolean> {
    let forcedAny = false;
    const getToolCalls = () => ctx.scratchpad.getToolCallRecords();

    const marketDataArgs = buildForcedNonCryptoMarketDataArgs(ctx.query);
    if (marketDataArgs && !hasMarketDataQuery(getToolCalls(), marketDataArgs.query)) {
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

    if (shouldForceGoldCombinedForecastArbitrator(ctx.query, getToolCalls())) {
      const args = buildForcedGoldCombinedForecastArbiterArgs(ctx.query, getToolCalls());
      if (args) {
        let ok = true;
        for await (const event of this.toolExecutor.executeTool('forecast_arbitrator', args, ctx)) {
          yield event;
          if (event.type === 'tool_error' || event.type === 'tool_denied') { ok = false; break; }
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
    const text = baseText
      ? ensureStructuredDensityTable(`${prefixText}${baseText}`, ctx.query, toolCalls)
      : prefixText;

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
