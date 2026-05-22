import { AIMessage } from '@langchain/core/messages';
import { StructuredToolInterface } from '@langchain/core/tools';
import { callLlm, streamCallLlm, getLlmCallTimeoutMs } from '../model/llm.js';
import { getSetting, loadConfig } from '../utils/config.js';
import { logger } from '../utils/logger.js';
import { buildToolDescriptions, getTools } from '../tools/registry.js';
import {
  buildSystemPrompt,
  buildIterationPrompt,
  loadSoulDocument,
} from './prompts.js';
import { extractTextContent, hasToolCalls, extractReasoningContent } from '../utils/parsing/ai-message.js';
import { InMemoryChatHistory } from '../utils/in-memory-chat-history.js';
import { buildHistoryContext } from '../utils/parsing/history-context.js';
import { estimateTokens, CONTEXT_THRESHOLD, KEEP_TOOL_USES, getContextThreshold, getKeepToolUses } from '../utils/tokens.js';
import { formatUserFacingError, isContextOverflowError } from '../utils/errors.js';
import type { AgentConfig, AgentEvent, AgentMemoryManager, AnswerStartEvent, AnswerChunkEvent, ContextClearedEvent, ProgressEvent, TokenUsage } from '../agent/types.js';
import { createRunContext, type RunContext } from './run-context.js';
import {
  createRunLoopState,
  type MemoryFlushState,
  type PeriodicMemoryFlushState,
  type RunLoopState,
} from './run-loop-state.js';
import { buildContextSummaryText } from './context-summary.js';
import { AgentToolExecutor } from './tool-executor.js';
import { MemoryManager } from '../memory/index.js';
import { runMemoryFlush, shouldRunMemoryFlush } from '../memory/flush.js';
import { injectMemoryContext } from './memory-injection.js';
import { extractTickers as extractTickersFn } from '../memory/ticker-extractor.js';
import { injectPolymarketContext } from '../tools/finance/polymarket-injector.js';
import { extractSignals as extractSignalsFn } from '../tools/finance/signal-extractor.js';
import { fetchPolymarketMarkets } from '../tools/finance/polymarket.js';
import { resolveProvider } from '../providers.js';
import {
  forecastLabRouter,
  type ForecastLabIntentRoute,
  type ForecastLabRoutingHint,
} from './forecast-lab-routing.js';
import { extractForecastLabRunToolAnswer } from '../tools/forecast-lab-run.js';
import {
  hasPrematureForecastArbitratorCall,
  isAcceptedFirstPlanningToolCall,
  normalizeExplicitGoldCombinedToolCalls,
} from './planning-tool-calls.js';
import {
  buildAbstainingBtcShortHorizonForecastAnswer,
  buildDistributionWarningPrefix,
  buildExplicitGoldCombinedForecastAnswer,
  buildForecastDisagreementPrefix,
  buildLowConfidenceBtcShortHorizonForecastPrefix,
  buildSourcesFooter,
  ensureStructuredDensityTable,
  stripThinkingTags,
} from './answer-formatting/index.js';
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
import { hasPolymarketForecastErrorForCoverage } from './query-router/coverage.js';

export {
  FACT_PATTERNS,
  buildContextSummaryText,
  extractKeyFacts,
  extractTickerMetrics,
} from './context-summary.js';
export { isAcceptedFirstPlanningToolCall } from './planning-tool-calls.js';

export {
  buildAbstainingBtcShortHorizonForecastAnswer,
  buildAbstainingMarkovAnswer,
  buildDistributionWarningPrefix,
  buildExplicitGoldCombinedForecastAnswer,
  buildForecastDisagreementPrefix,
  buildLowConfidenceBtcShortHorizonForecastPrefix,
  buildSourcesFooter,
  buildUnavailableDistributionAnswer,
  ensureStructuredDensityTable,
  shouldPreserveAbstainingBtcShortHorizonForecast,
  stripThinkingTags,
} from './answer-formatting/index.js';
type ForecastLabResetHint = NonNullable<ForecastLabIntentRoute['resetRequest']>;
type ForecastLabPromotionApprovalHint = NonNullable<ForecastLabIntentRoute['promotionApproval']>;
type ForecastLabKeepCurrentBestHint = NonNullable<ForecastLabIntentRoute['keepCurrentBestRequest']>;
type ForecastLabCatalogExtensionHint = NonNullable<ForecastLabIntentRoute['catalogExtensionRequest']>;
type ForecastLabComparisonHint = NonNullable<ForecastLabIntentRoute['comparisonRequest']>;
type ForecastLabResultsHint = NonNullable<ForecastLabIntentRoute['resultsRequest']>;
type ForecastLabMutatorListHint = NonNullable<ForecastLabIntentRoute['mutatorListRequest']>;

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
const MAX_OVERFLOW_RETRIES = 2;
/** Flush memory to disk every N iterations regardless of context size. */
const PERIODIC_FLUSH_INTERVAL = 5;

type ForcedToolExecutionStatus = 'success' | 'denied' | 'error';
type ForcedToolRouteStatus = ForcedToolExecutionStatus | 'idle';

function getMemoryManagerFactory(config: AgentConfig): () => Promise<AgentMemoryManager> {
  return config.getMemoryManager ?? (() => MemoryManager.get());
}

function mergeForcedToolStatus(
  current: ForcedToolRouteStatus,
  next: ForcedToolExecutionStatus,
): ForcedToolRouteStatus {
  if (next === 'denied') return 'denied';
  if (next === 'error') return 'error';
  return current === 'idle' ? 'success' : current;
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
  private readonly getMemoryManager: () => Promise<AgentMemoryManager>;
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
    this.getMemoryManager = getMemoryManagerFactory(config);
    this.thinkEnabled = config.thinkEnabled;
  }

  /**
   * Create a new Agent instance with tools.
   */
  static async create(config: AgentConfig = {}): Promise<Agent> {
    const model = config.model ?? DEFAULT_MODEL;
    const memoryEnabled = config.memoryEnabled ?? true;
    const registryOptions = {
      watchlistEntries: config.watchlistEntries,
      memoryEnabled,
    };
    const tools = config.tools ?? getTools(model, registryOptions);
    const soulContentPromise = loadSoulDocument();
    let memoryFiles: string[] = [];
    let memoryContext: string | null = null;

    if (memoryEnabled) {
      const memoryManager = await getMemoryManagerFactory(config)();
      const [files, session] = await Promise.all([
        memoryManager.listFiles(),
        memoryManager.loadSessionContext(),
      ]);
      memoryFiles = files;
      if (session.text.trim()) {
        memoryContext = session.text;
      }
    }
    const soulContent = await soulContentPromise;

    const systemPrompt = buildSystemPrompt(
      model,
      soulContent,
      config.channel,
      config.groupContext,
      memoryFiles,
      memoryContext,
      config.toolDescriptionsOverride ?? buildToolDescriptions(model, registryOptions),
      memoryEnabled,
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
    const runState = createRunLoopState();
    const forecastingConfig = loadConfig().forecasting;
    const forecastLabIntent = forecastLabRouter.routeIntent(query, {
      inMemoryHistory,
      enableAutoRoute: forecastingConfig?.enableForecastLabAutoRoute,
      enableSkillHint: forecastingConfig?.enableForecastLabSkillHint,
    });
    const {
      routingHint: forecastLabRoutingHint,
      resetRequest: forecastLabResetRequest,
      promotionApproval: forecastLabPromotionApproval,
      keepCurrentBestRequest: forecastLabKeepCurrentBestRequest,
      catalogExtensionRequest: forecastLabCatalogExtensionRequest,
      comparisonRequest: forecastLabComparisonRequest,
      resultsRequest: forecastLabResultsRequest,
      mutatorListRequest: forecastLabMutatorListRequest,
    } = forecastLabIntent;
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
    if (this.memoryEnabled) {
      currentPrompt = await injectMemoryContext(query, currentPrompt, {
        getMemoryManager: this.getMemoryManager,
        extractTickers: (text) => extractTickersFn(text),
      });
    }

    // Auto-inject Polymarket prediction market context for detected asset signals
    currentPrompt = await injectPolymarketContext(query, currentPrompt, {
      extractSignals: (text) => extractSignalsFn(text),
      fetchMarkets: (q, limit) => fetchPolymarketMarkets(q, limit),
    });

    const explicitlyRequestedSkill = detectExplicitSkillRequest(query);

    // Cap retries for the sequential_thinking compliance reminder to avoid
    // an infinite loop when a model persistently ignores the instruction.
    const MAX_ST_RETRIES = 3;
    // Hard cap on total sequential_thinking calls so planning never burns all
    // iterations before any research tool runs. Models sometimes loop through
    // 10-15 thoughts on complex queries, leaving no budget for actual research.
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

      runState.sequentialThinkingUsed = true;
    }

    // Main agent loop
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
          runState.overflowRetries = 0;
          break;
        } catch (error) {
          const errorMessage = error instanceof Error ? error.message : String(error);

          if (isContextOverflowError(errorMessage) && runState.overflowRetries < MAX_OVERFLOW_RETRIES) {
            runState.overflowRetries++;
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
          const forcedStatus = yield* this.forceCryptoForecastTools(ctx);
          if (forcedStatus === 'denied') {
            yield* this.finishAfterToolDenied(ctx);
            return;
          }
          if (this.shouldRebuildAfterForcedTools(forcedStatus, runState)) {
            currentPrompt = yield* this.rebuildPromptAfterForcedTools(ctx, query, runState.memoryFlush, forecastLabRoutingHint);
            continue;
          }
        }

        if (shouldForceNonCryptoForecastFallback(query, ctx.scratchpad.getToolCallRecords())) {
          const forcedStatus = yield* this.forceNonCryptoForecastFallback(ctx);
          if (forcedStatus === 'denied') {
            yield* this.finishAfterToolDenied(ctx);
            return;
          }
          if (this.shouldRebuildAfterForcedTools(forcedStatus, runState)) {
            currentPrompt = yield* this.rebuildPromptAfterForcedTools(ctx, query, runState.memoryFlush, forecastLabRoutingHint);
            continue;
          }
        }

        if (shouldForceMarkovDistribution(query, ctx.scratchpad.getToolCallRecords())) {
          const forcedStatus = yield* this.forceMarkovDistribution(ctx);
          if (forcedStatus === 'denied') {
            yield* this.finishAfterToolDenied(ctx);
            return;
          }
          if (this.shouldRebuildAfterForcedTools(forcedStatus, runState)) {
            currentPrompt = yield* this.rebuildPromptAfterForcedTools(ctx, query, runState.memoryFlush, forecastLabRoutingHint);
            continue;
          }
        }

        if (shouldForceGoldCombinedForecastTools(query, ctx.scratchpad.getToolCallRecords())) {
          const forcedStatus = yield* this.forceGoldCombinedForecastTools(ctx);
          if (forcedStatus === 'denied') {
            yield* this.finishAfterToolDenied(ctx);
            return;
          }
          if (this.shouldRebuildAfterForcedTools(forcedStatus, runState)) {
            currentPrompt = yield* this.rebuildPromptAfterForcedTools(ctx, query, runState.memoryFlush, forecastLabRoutingHint);
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

        const explicitGoldCombinedAnswer = buildExplicitGoldCombinedForecastAnswer(
          query,
          ctx.scratchpad.getToolCallRecords(),
        );
        if (explicitGoldCombinedAnswer) {
          yield* this.handleDirectResponse(explicitGoldCombinedAnswer, ctx, currentPrompt);
          return;
        }

        yield* this.handleDirectResponse(responseText ?? '', ctx, currentPrompt);
        return;
      }

      // Enforce sequential_thinking as the mandatory first tool call.
      // If the model's first tool call this session is not sequential_thinking,
      // inject a reminder and retry — but only up to MAX_ST_RETRIES times to
      // prevent an infinite loop when a model persistently ignores the reminder.
      if (!runState.sequentialThinkingUsed) {
        const firstTool = (response as AIMessage).tool_calls?.[0]?.name;
        if (
          firstTool
          && !isAcceptedFirstPlanningToolCall(
            response as AIMessage,
            forecastLabRoutingHint,
            explicitlyRequestedSkill,
          )
        ) {
          if (runState.sequentialThinkingRetries < MAX_ST_RETRIES) {
            runState.sequentialThinkingRetries++;
            ctx.iteration--; // don't charge this iteration
            currentPrompt = forecastLabRoutingHint?.shouldInvokeSkill
              ? `${currentPrompt}\n\nIMPORTANT REMINDER: This routed forecast-lab improvement query must start with skill(\"forecast-lab\") or sequential_thinking. Do NOT start with ordinary forecast/data tools.`
              : `${currentPrompt}\n\nIMPORTANT REMINDER: You MUST call sequential_thinking FIRST before calling any other tool. Start with sequential_thinking to plan your approach, then proceed.`;
            continue;
          }
          // Retries exhausted — proceed without sequential_thinking rather than
          // looping forever. Mark as satisfied so we stop checking.
          runState.sequentialThinkingUsed = true;
        }
      }

      // Mark sequential_thinking as satisfied once it appears in any tool call
      if (!runState.sequentialThinkingUsed) {
        const stToolCalls = (response as AIMessage).tool_calls ?? [];
        if (
          stToolCalls.some((tc) => tc.name === 'sequential_thinking')
          || (
            forecastLabRoutingHint?.shouldInvokeSkill
            && stToolCalls.some((tc) => tc.name === 'skill' && tc.args?.skill === 'forecast-lab')
          )
        ) {
          runState.sequentialThinkingUsed = true;
        }
      }

      if (runState.sequentialThinkingUsed && shouldForceMarkovDistribution(query, ctx.scratchpad.getToolCallRecords())) {
        const forcedStatus = yield* this.forceMarkovDistribution(ctx);
        if (forcedStatus === 'denied') {
          yield* this.finishAfterToolDenied(ctx);
          return;
        }
        if (this.shouldRebuildAfterForcedTools(forcedStatus, runState)) {
          currentPrompt = yield* this.rebuildPromptAfterForcedTools(ctx, query, runState.memoryFlush, forecastLabRoutingHint);
          continue;
        }
      }

      if (runState.sequentialThinkingUsed && shouldForceGoldCombinedForecastTools(query, ctx.scratchpad.getToolCallRecords())) {
        const forcedStatus = yield* this.forceGoldCombinedForecastTools(ctx);
        if (forcedStatus === 'denied') {
          yield* this.finishAfterToolDenied(ctx);
          return;
        }
        if (this.shouldRebuildAfterForcedTools(forcedStatus, runState)) {
          currentPrompt = yield* this.rebuildPromptAfterForcedTools(ctx, query, runState.memoryFlush, forecastLabRoutingHint);
          continue;
        }
      }

      if (runState.sequentialThinkingUsed && shouldForceCryptoForecastTools(query, ctx.scratchpad.getToolCallRecords())) {
        const forcedStatus = yield* this.forceCryptoForecastTools(ctx);
        if (forcedStatus === 'denied') {
          yield* this.finishAfterToolDenied(ctx);
          return;
        }
        if (this.shouldRebuildAfterForcedTools(forcedStatus, runState)) {
          currentPrompt = yield* this.rebuildPromptAfterForcedTools(ctx, query, runState.memoryFlush, forecastLabRoutingHint);
          continue;
        }
      }

      if (runState.sequentialThinkingUsed && shouldForceNonCryptoForecastFallback(query, ctx.scratchpad.getToolCallRecords())) {
        const forcedStatus = yield* this.forceNonCryptoForecastFallback(ctx);
        if (forcedStatus === 'denied') {
          yield* this.finishAfterToolDenied(ctx);
          return;
        }
        if (this.shouldRebuildAfterForcedTools(forcedStatus, runState)) {
          currentPrompt = yield* this.rebuildPromptAfterForcedTools(ctx, query, runState.memoryFlush, forecastLabRoutingHint);
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
        const forcedStatus = yield* this.forceCryptoForecastTools(ctx);
        if (forcedStatus === 'denied') {
          yield* this.finishAfterToolDenied(ctx);
          return;
        }
        if (this.shouldRebuildAfterForcedTools(forcedStatus, runState)) {
          currentPrompt = yield* this.rebuildPromptAfterForcedTools(ctx, query, runState.memoryFlush, forecastLabRoutingHint);
          continue;
        }
      }

      const stCallsThisIteration = toolCalls.filter((tc) => tc.name === 'sequential_thinking').length;
      runState.sequentialThinkingCallCount += stCallsThisIteration;

      // Execute tools and add results to scratchpad (response is AIMessage here)
      for await (const event of this.toolExecutor.executeAll(response, ctx)) {
        yield event;
        if (event.type === 'tool_denied') {
          yield* this.finishAfterToolDenied(ctx);
          return;
        }
      }
      yield* this.manageContextThreshold(ctx, query, runState.memoryFlush);

      // Periodic auto-save: flush findings to long-term memory every N iterations
      // so a crash doesn't lose all research done so far.
      if (
        this.memoryEnabled &&
        ctx.iteration - runState.periodicFlush.lastFlushedIteration >= PERIODIC_FLUSH_INTERVAL
      ) {
        yield* this.runPeriodicMemoryFlush(ctx, query, runState.periodicFlush);
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
      if (stCallsThisIteration > 0 && runState.sequentialThinkingCallCount >= MAX_SEQUENTIAL_THOUGHTS) {
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

    const explicitGoldCombinedAnswer = buildExplicitGoldCombinedForecastAnswer(
      query,
      ctx.scratchpad.getToolCallRecords(),
    );
    if (explicitGoldCombinedAnswer) {
      yield* this.handleDirectResponse(explicitGoldCombinedAnswer, ctx, currentPrompt);
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

  private shouldRebuildAfterForcedTools(
    forcedStatus: ForcedToolRouteStatus,
    runState: RunLoopState,
  ): boolean {
    if (forcedStatus === 'idle') return false;
    if (forcedStatus === 'error') {
      if (runState.forcedToolErrorPromptRebuilt) return false;
      runState.forcedToolErrorPromptRebuilt = true;
    }
    return true;
  }

  private async *rebuildPromptAfterForcedTools(
    ctx: RunContext,
    query: string,
    memoryFlush: MemoryFlushState,
    forecastLabRoutingHint: ForecastLabRoutingHint | null,
  ): AsyncGenerator<AgentEvent, string> {
    yield* this.manageContextThreshold(ctx, query, memoryFlush);
    return buildIterationPrompt(
      query,
      ctx.scratchpad.getToolResults(),
      ctx.scratchpad.formatToolUsageForPrompt(),
      forecastLabRoutingHint,
    );
  }

  private async *finishAfterToolDenied(ctx: RunContext): AsyncGenerator<AgentEvent, void> {
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
  }

  private async runMemoryFlushSafely(
    params: Parameters<typeof runMemoryFlush>[0],
  ): Promise<{ flushed: boolean; written: boolean; content?: string }> {
    try {
      return await runMemoryFlush(params);
    } catch (error) {
      logger.warn('[Agent] memory flush failed', error);
      return { flushed: false, written: false };
    }
  }

  private async *forceMarkovDistribution(
    ctx: RunContext,
  ): AsyncGenerator<AgentEvent, ForcedToolRouteStatus> {
    const args = buildForcedMarkovArgs(ctx.query);
    if (!args) return 'idle';

    return yield* this.executeForcedTool(ctx, 'markov_distribution', args);
  }

  private async *executeForcedTool(
    ctx: RunContext,
    toolName: string,
    args: Record<string, unknown>,
  ): AsyncGenerator<AgentEvent, ForcedToolExecutionStatus> {
    let status: ForcedToolExecutionStatus = 'success';

    for await (const event of this.toolExecutor.executeTool(toolName, args, ctx)) {
      yield event;
      if (event.type === 'tool_denied') {
        return 'denied';
      }
      if (event.type === 'tool_error') {
        status = 'error';
      }
    }

    return status;
  }

  private async *forceCryptoForecastTools(
    ctx: RunContext,
  ): AsyncGenerator<AgentEvent, ForcedToolRouteStatus> {
    let forcedStatus: ForcedToolRouteStatus = 'idle';

    const getToolCalls = () => ctx.scratchpad.getToolCallRecords();

    const marketDataArgs = buildForcedMarketDataArgs(ctx.query);
    if (
      marketDataArgs
      && extractCurrentPriceFromMarketDataQuery(getToolCalls(), marketDataArgs.query) === null
    ) {
      const args = marketDataArgs;
      if (args) {
        const status = yield* this.executeForcedTool(ctx, 'get_market_data', args);
        forcedStatus = mergeForcedToolStatus(forcedStatus, status);
        if (forcedStatus === 'denied') return forcedStatus;
      }
    }

    if (extractSentimentScoreFromToolCalls(getToolCalls()) === null) {
      const args = buildForcedSocialSentimentArgs(ctx.query);
      if (args) {
        const status = yield* this.executeForcedTool(ctx, 'social_sentiment', args);
        forcedStatus = mergeForcedToolStatus(forcedStatus, status);
        if (forcedStatus === 'denied') return forcedStatus;
      }
    }

    if (!hasCompletedMarkovDistributionForQuery(ctx.query, getToolCalls())) {
      const args = buildForcedCryptoForecastMarkovArgs(ctx.query);
      if (args) {
        const status = yield* this.executeForcedTool(ctx, 'markov_distribution', args);
        forcedStatus = mergeForcedToolStatus(forcedStatus, status);
        if (forcedStatus === 'denied') return forcedStatus;
      }
    }

    if (!hasCryptoPolymarketForecastCoverage(ctx.query, getToolCalls())) {
      const args = buildForcedPolymarketForecastArgs(ctx.query, getToolCalls());
      if (args) {
        const status = yield* this.executeForcedTool(ctx, 'polymarket_forecast', args);
        forcedStatus = mergeForcedToolStatus(forcedStatus, status);
        if (forcedStatus === 'denied') return forcedStatus;
      }
    }

    if (shouldRerunPolymarketForecastWithMarkov(ctx.query, getToolCalls())) {
      const args = buildForcedPolymarketForecastArgs(ctx.query, getToolCalls());
      if (args) {
        const status = yield* this.executeForcedTool(ctx, 'polymarket_forecast', args);
        forcedStatus = mergeForcedToolStatus(forcedStatus, status);
        if (forcedStatus === 'denied') return forcedStatus;
      }
    }

    if (!hasUsableOnchainResultForCryptoQuery(ctx.query, getToolCalls())) {
      const args = buildForcedOnchainArgs(ctx.query);
      if (args) {
        const status = yield* this.executeForcedTool(ctx, 'get_onchain_crypto', args);
        forcedStatus = mergeForcedToolStatus(forcedStatus, status);
        if (forcedStatus === 'denied') return forcedStatus;
      }
    }

    if (!hasUsableFixedIncomeResult(getToolCalls())) {
      const args = buildForcedFixedIncomeArgs();
      const status = yield* this.executeForcedTool(ctx, 'get_fixed_income', args);
      forcedStatus = mergeForcedToolStatus(forcedStatus, status);
      if (forcedStatus === 'denied') return forcedStatus;
    }

    if (shouldForceForecastArbitrator(ctx.query, getToolCalls())) {
      const args = buildForcedForecastArbiterArgs(ctx.query, getToolCalls());
      if (args) {
        const status = yield* this.executeForcedTool(ctx, 'forecast_arbitrator', args);
        forcedStatus = mergeForcedToolStatus(forcedStatus, status);
        if (forcedStatus === 'denied') return forcedStatus;
      }
    }

    return forcedStatus;
  }

  /** Force the explicit GOLD combined seam after a usable Markov result exists. */
  private async *forceGoldCombinedForecastTools(
    ctx: RunContext,
  ): AsyncGenerator<AgentEvent, ForcedToolRouteStatus> {
    let forcedStatus: ForcedToolRouteStatus = 'idle';
    const getToolCalls = () => ctx.scratchpad.getToolCallRecords();

    const marketDataArgs = buildForcedNonCryptoMarketDataArgs(ctx.query);
    if (marketDataArgs && !hasMarketDataQuery(getToolCalls(), marketDataArgs.query)) {
      const status = yield* this.executeForcedTool(ctx, 'get_market_data', marketDataArgs);
      forcedStatus = mergeForcedToolStatus(forcedStatus, status);
      if (forcedStatus === 'denied') return forcedStatus;
    }

    const forecastArgs = buildForcedNonCryptoPolymarketForecastArgs(ctx.query, getToolCalls());
    if (
      forecastArgs
      && !hasPolymarketForecastCoverage(getToolCalls(), forecastArgs)
      && !hasPolymarketForecastErrorForCoverage(getToolCalls(), forecastArgs)
    ) {
      const status = yield* this.executeForcedTool(ctx, 'polymarket_forecast', forecastArgs);
      forcedStatus = mergeForcedToolStatus(forcedStatus, status);
      if (forcedStatus === 'denied') return forcedStatus;
    }

    if (shouldForceGoldCombinedForecastArbitrator(ctx.query, getToolCalls())) {
      const args = buildForcedGoldCombinedForecastArbiterArgs(ctx.query, getToolCalls());
      if (args) {
        const status = yield* this.executeForcedTool(ctx, 'forecast_arbitrator', args);
        forcedStatus = mergeForcedToolStatus(forcedStatus, status);
        if (forcedStatus === 'denied') return forcedStatus;
      }
    }

    return forcedStatus;
  }

  /** Force get_market_data + polymarket_forecast for non-crypto forecast asks after Markov abstains. */
  private async *forceNonCryptoForecastFallback(
    ctx: RunContext,
  ): AsyncGenerator<AgentEvent, ForcedToolRouteStatus> {
    let forcedStatus: ForcedToolRouteStatus = 'idle';
    const getToolCalls = () => ctx.scratchpad.getToolCallRecords();

    const marketDataArgs = buildForcedNonCryptoMarketDataArgs(ctx.query);
    if (
      marketDataArgs
      && !hasMarketDataQuery(getToolCalls(), marketDataArgs.query)
    ) {
      const status = yield* this.executeForcedTool(ctx, 'get_market_data', marketDataArgs);
      forcedStatus = mergeForcedToolStatus(forcedStatus, status);
      if (forcedStatus === 'denied') return forcedStatus;
    }

    const forecastArgs = buildForcedNonCryptoPolymarketForecastArgs(ctx.query, getToolCalls());
    if (
      forecastArgs
      && !hasPolymarketForecastCoverage(getToolCalls(), forecastArgs)
      && !hasPolymarketForecastErrorForCoverage(getToolCalls(), forecastArgs)
    ) {
      const status = yield* this.executeForcedTool(ctx, 'polymarket_forecast', forecastArgs);
      forcedStatus = mergeForcedToolStatus(forcedStatus, status);
      if (forcedStatus === 'denied') return forcedStatus;
    }

    return forcedStatus;
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
      } catch (error) {
        logger.warn('[Agent] final synthesis failed', error);
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
    memoryFlushState: MemoryFlushState,
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
        const flushResult = await this.runMemoryFlushSafely({
          model: this.model,
          systemPrompt: this.systemPrompt,
          query,
          toolResults: fullToolResults,
          signal: this.signal,
        });
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
    state: PeriodicMemoryFlushState,
  ): AsyncGenerator<AgentEvent, void> {
    state.lastFlushedIteration = ctx.iteration;
    yield { type: 'memory_flush', phase: 'start' };
    const flushResult = await this.runMemoryFlushSafely({
      model: this.model,
      systemPrompt: this.systemPrompt,
      query,
      toolResults: ctx.scratchpad.getToolResults(),
      signal: this.signal,
    });
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
      return forecastLabRouter.injectRoutingHint(query, forecastLabRoutingHint);
    }

    const recentTurns = inMemoryChatHistory.getRecentTurns();
    if (recentTurns.length === 0) {
      return forecastLabRouter.injectRoutingHint(query, forecastLabRoutingHint);
    }

    return forecastLabRouter.injectRoutingHint(buildHistoryContext({
      entries: recentTurns,
      currentMessage: query,
    }), forecastLabRoutingHint);
  }
}
