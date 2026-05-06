/**
 * Unit tests for Agent.create() and agent.run() using provider-level mocking.
 *
 * Uses the same @langchain/* mock pattern as agent-features.test.ts to avoid
 * Bun module-registry contamination (never mocks llm.js directly).
 */
import { describe, it, expect, mock, beforeEach, afterEach } from 'bun:test';
import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import type { AgentEvent, DoneEvent } from './types.js';

// ---------------------------------------------------------------------------
// Filesystem isolation
// ---------------------------------------------------------------------------
let tmpDir: string;
let originalCwd: string;
let prevOpenAiKey: string | undefined;

beforeEach(() => {
  tmpDir = join(tmpdir(), `agent-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(tmpDir, { recursive: true });
  originalCwd = process.cwd();
  process.chdir(tmpDir);
  prevOpenAiKey = process.env.OPENAI_API_KEY;
  if (!prevOpenAiKey) process.env.OPENAI_API_KEY = 'sk-test-stub';
});

afterEach(() => {
  process.chdir(originalCwd);
  rmSync(tmpDir, { recursive: true, force: true });
  if (!prevOpenAiKey) delete process.env.OPENAI_API_KEY;
  else process.env.OPENAI_API_KEY = prevOpenAiKey;
});

// ---------------------------------------------------------------------------
// Mutable mock state
// ---------------------------------------------------------------------------
const mockState = {
  invokeContent: 'Direct answer',
  invokeToolCalls: [] as any[],
  invokeThrows: false,
  invokeThrowMessage: 'LLM API error',
  streamChunks: ['streaming answer'] as string[],
  streamThrows: false,
  callCount: 0,
};

const forecastLabMockState = {
  calls: [] as Array<Record<string, unknown>>,
  guidedImproveResult: JSON.stringify({
    data: {
      _tool: 'forecast_lab_run',
      action: 'guided-improve',
      status: 'ok',
      execute: true,
      profileId: 'btc-markov-ultra-short-horizon',
      runId: 'btc-markov-ultra-short-horizon.keep-1',
      decision: 'keep',
      reason: 'candidate passed the fixed gates',
      artifactsPath: '.cramer-short/experiments/runs/btc-markov-ultra-short-horizon.keep-1',
      promotionReady: true,
      promotionStatus: 'approval-required',
      sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
      answer: 'Guided keep. Approval required before promotion. Reply "approve forecast-lab promotion for btc-markov-ultra-short-horizon run btc-markov-ultra-short-horizon.keep-1" to continue.',
    },
  }),
  planResult: JSON.stringify({
    data: {
      _tool: 'forecast_lab_run',
      action: 'guided-improve',
      status: 'ok',
      execute: false,
      profileId: 'btc-markov-ultra-short-horizon',
      promotionReady: false,
      answer: 'Forecast-lab bounded plan for btc-markov-ultra-short-horizon. Baseline harness first. Candidate compare. Artifacts stay under .cramer-short/experiments.',
    },
  }),
  promotionResult: JSON.stringify({
    data: {
      _tool: 'forecast_lab_run',
      action: 'promote-approved',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      runId: 'forecast-lab-promo-1',
      sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
      decision: 'keep',
      reason: 'promotion verification passed',
      artifactsPath: '.cramer-short/experiments/runs/forecast-lab-promo-1',
      activationArtifactsPath: '.cramer-short/experiments/runs/forecast-lab-promo-1',
      activeStatePath: '.cramer-short/experiments/active-promotions/btc-markov-ultra-short-horizon.json',
      answer: 'Promotion completed for btc-markov-ultra-short-horizon from kept run btc-markov-ultra-short-horizon.keep-1. Promoted parameters are now live for normal forecasts.',
    },
  }),
  compareResult: JSON.stringify({
    data: {
      _tool: 'forecast_lab_run',
      action: 'compare-best-vs-shipped',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
      decision: 'keep',
      reason: 'candidate passed the fixed gates',
      artifactsPath: '.cramer-short/experiments/runs/btc-markov-ultra-short-horizon.keep-1',
      liveStatus: 'ready-to-promote',
      promotionCommand: 'Approve forecast-lab promotion for btc-markov-ultra-short-horizon run btc-markov-ultra-short-horizon.keep-1.',
      answer: 'Forecast-lab comparison for btc-markov-ultra-short-horizon. Regular forecasts: not live yet. Reply "Approve forecast-lab promotion for btc-markov-ultra-short-horizon run btc-markov-ultra-short-horizon.keep-1." to activate this kept run for ordinary forecast queries.',
    },
  }),
  listMutatorsResult: JSON.stringify({
    data: {
      _tool: 'forecast_lab_run',
      action: 'list-mutators',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      profiles: [
        {
          profileId: 'btc-markov-ultra-short-horizon',
          targetSubsystem: 'markov-distribution',
          mutationMode: 'structured',
          currentCatalogIds: [
            'markov-shorter-reactive-window',
            'markov-faster-decay-reaction',
            'markov-lower-confidence-trend-penalty',
          ],
          allowedOperatorIds: ['search-replace'],
        },
      ],
      dryRunProfiles: ['btc-arbiter-replay', 'polymarket-selection-sanity'],
      frameworkOperatorIds: ['replace-range', 'search-replace', 'insert-block'],
      answer: 'Forecast-lab shipped mutator ids for btc-markov-ultra-short-horizon. Shipped candidate catalog ids: markov-shorter-reactive-window, markov-faster-decay-reaction, markov-lower-confidence-trend-penalty.',
    },
  }),
  catalogExtensionResult: JSON.stringify({
    data: {
      _tool: 'forecast_lab_run',
      action: 'catalog-extension-plan',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      targetSubsystem: 'markov-distribution',
      allowedGlobs: [
        'src/tools/finance/markov-distribution.ts',
        'src/tools/finance/conformal.ts',
        'src/tools/finance/regime-calibrator.ts',
      ],
      mutationMode: 'structured',
      allowedMutatorIds: ['search-replace'],
      currentCatalogIds: [
        'markov-shorter-reactive-window',
        'markov-faster-decay-reaction',
        'markov-lower-confidence-trend-penalty',
      ],
      catalogFiles: [
        'src/experiments/forecast-lab/mutators/markov-parameters.ts',
        'src/experiments/forecast-lab/profiles.ts',
      ],
      validationFiles: [
        'src/experiments/forecast-lab/mutators/markov-parameters.test.ts',
        'src/experiments/forecast-lab/profiles.test.ts',
        'src/tools/finance/markov-distribution.test.ts',
        'src/tools/finance/backtest/walk-forward-r5.test.ts',
      ],
      operatorMutatorIds: ['replace-range', 'search-replace', 'insert-block'],
      answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. This is a bounded code-change plan. It is not a safe runtime mutation request, so I did not inspect experiment artifacts or try to rerun the lineage directly. Catalog files to open directly next: src/experiments/forecast-lab/mutators/markov-parameters.ts, src/experiments/forecast-lab/profiles.ts. Validation files to open directly next: src/experiments/forecast-lab/mutators/markov-parameters.test.ts, src/experiments/forecast-lab/profiles.test.ts, src/tools/finance/markov-distribution.test.ts, src/tools/finance/backtest/walk-forward-r5.test.ts. After the new mutator is implemented and allowed for btc-markov-ultra-short-horizon, rerun the lineage with forecast_lab_run(action="guided-improve", profileId="btc-markov-ultra-short-horizon").',
    },
  }),
  resetResult: JSON.stringify({
    data: {
      _tool: 'forecast_lab_run',
      action: 'reset-live',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      resetMode: 'defaults',
      runId: 'forecast-lab-reset-1',
      artifactsPath: '.cramer-short/experiments/runs/forecast-lab-reset-1',
      resetArtifactPath: '.cramer-short/experiments/runs/forecast-lab-reset-1/reset.json',
      answer: 'Forecast-lab reset completed for btc-markov-ultra-short-horizon. Mode: shipped defaults.',
    },
  }),
};

/** Sequential thinking call stub (satisfies the ST-first enforcement). */
const ST_TOOL_CALL = {
  id: 'st1',
  name: 'sequential_thinking',
  args: { thought: 'analyzing...', nextThoughtNeeded: false, thoughtNumber: 1, totalThoughts: 1 },
  type: 'tool_call' as const,
};

// ---------------------------------------------------------------------------
// Mock llm.js at the module boundary (same pattern as agent-response.test.ts).
//
// Mocking @langchain/* packages is insufficient because agent.ts calls callLlm()
// from model/llm.js, not the LangChain classes directly. agent-response.test.ts
// (which runs first in the same Bun worker) already sets mock.module('../model/llm.js')
// with its own mockState closure, so we must override it here to use THIS file's
// mockState. The @langchain/* mocks below are kept for completeness but are unused.
// ---------------------------------------------------------------------------
mock.module('../model/llm.js', () => ({
  DEFAULT_MODEL: 'gpt-5.4',
  getLlmCallTimeoutMs: () => 120000,
  callLlm: async () => {
    mockState.callCount++;
    if (mockState.invokeThrows) throw new Error(mockState.invokeThrowMessage);
    if (mockState.invokeToolCalls.length > 0) {
      return {
        response: { content: '', tool_calls: mockState.invokeToolCalls, additional_kwargs: {} },
        usage: undefined,
      };
    }
    return {
      response: { content: mockState.invokeContent, tool_calls: [], additional_kwargs: {} },
      usage: undefined,
    };
  },
  streamCallLlm: async function* (_prompt: string) {
    if (mockState.streamThrows) throw new Error('stream timeout');
    for (const chunk of mockState.streamChunks) yield chunk;
  },
  resolveProvider: (model: string) => ({ id: 'openai', displayName: model }),
  formatUserFacingError: (msg: string) => msg,
  isContextOverflowError: () => false,
}));
mock.module('@langchain/openai', () => ({
  ChatOpenAI: class { constructor() {} bindTools() { return this; } },
  OpenAIEmbeddings: class { constructor() {} },
}));
mock.module('@langchain/anthropic', () => ({
  ChatAnthropic: class { constructor() {} bindTools() { return this; } },
}));
mock.module('@langchain/google-genai', () => ({
  ChatGoogleGenerativeAI: class { constructor() {} bindTools() { return this; } },
  GoogleGenerativeAIEmbeddings: class { constructor() {} },
}));
mock.module('@langchain/ollama', () => ({
  ChatOllama: class { constructor() {} bindTools() { return this; } },
  OllamaEmbeddings: class { constructor() {} },
}));

const { Agent } = await import('./agent.js');
const { AgentToolExecutor } = await import('./tool-executor.js');
const {
  inferDistributionTicker,
  inferDistributionHorizon,
  isExplicitTerminalDistributionQuery,
  shouldForceMarkovDistribution,
  inferTrajectoryRequest,
   buildForcedMarkovArgs,
   isCryptoForecastQuery,
   isExplicitPolymarketForecastRequest,
   isNonCryptoForecastQuery,
   buildForcedNonCryptoMarketDataArgs,
   buildForcedNonCryptoPolymarketForecastArgs,
   shouldForceNonCryptoForecastFallback,
   extractCurrentPriceFromToolCalls,
  extractSentimentScoreFromToolCalls,
  buildForcedMarketDataArgs,
  buildForcedSocialSentimentArgs,
   buildForcedPolymarketForecastArgs,
     extractMarkovReturnFromToolCalls,
     buildForecastDisagreementPrefix,
     buildLowConfidenceBtcShortHorizonForecastPrefix,
     shouldInjectBtcShortHorizonMixedEvidencePrompt,
     shouldInjectBtcShortHorizonLowConfidencePrompt,
    shouldRerunPolymarketForecastWithMarkov,
    buildForcedForecastArbiterArgs,
    shouldForceForecastArbitrator,
  buildForcedOnchainArgs,
  buildForcedFixedIncomeArgs,
 buildForcedCryptoForecastMarkovArgs,
  shouldForceCryptoForecastTools,
  shouldPreserveAbstainingBtcShortHorizonForecast,
  isForecastLabImprovementQuery,
  isAcceptedFirstPlanningToolCall,
  detectExplicitSkillRequest,
  isForecastLabPlanOnlyQuery,
  detectForecastLabPromotionApproval,
  detectForecastLabResetRequest,
  detectForecastLabComparisonRequest,
  detectForecastLabResultsRequest,
  detectForecastLabMutatorListRequest,
  detectForecastLabKeepCurrentBestRequest,
  detectForecastLabCatalogExtensionRequest,
} = await import('./agent.js');
 const { getForecastLabRoutingHint } = await import('./forecast-lab-routing.js');
const { InMemoryChatHistory } = await import('../utils/in-memory-chat-history.js');

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
async function collectEvents(gen: AsyncGenerator<AgentEvent>): Promise<AgentEvent[]> {
  const events: AgentEvent[] = [];
  for await (const e of gen) events.push(e);
  return events;
}

function installForecastLabTool(agent: any): void {
  agent.toolMap.set('forecast_lab_run', {
    name: 'forecast_lab_run',
    invoke: async (input: Record<string, unknown>) => {
      forecastLabMockState.calls.push(input);
      if (input.action === 'compare-best-vs-shipped') {
        return forecastLabMockState.compareResult;
      }
      if (input.action === 'list-mutators') {
        return forecastLabMockState.listMutatorsResult;
      }
      if (input.action === 'catalog-extension-plan') {
        return forecastLabMockState.catalogExtensionResult;
      }
      if (input.action === 'promote-approved') {
        return forecastLabMockState.promotionResult;
      }
      if (input.action === 'reset-live') {
        return forecastLabMockState.resetResult;
      }
      if (input.execute === false) {
        return forecastLabMockState.planResult;
      }
      return forecastLabMockState.guidedImproveResult;
    },
  });
  agent.toolExecutor = new AgentToolExecutor(agent.toolMap);
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------
describe('Agent', () => {
  beforeEach(() => {
    // Reset mock state to safe defaults
    mockState.invokeContent = 'Direct answer';
    mockState.invokeToolCalls = [];
    mockState.invokeThrows = false;
    mockState.invokeThrowMessage = 'LLM API error';
    mockState.streamChunks = ['streaming answer'];
    mockState.streamThrows = false;
    mockState.callCount = 0;
    forecastLabMockState.calls = [];
  });

  describe('Agent.create', () => {
    it('creates an Agent instance', async () => {
      const agent = await Agent.create({});
      expect(agent).toBeDefined();
      expect(typeof agent.run).toBe('function');
    });

    it('creates an Agent with custom model', async () => {
      const agent = await Agent.create({ model: 'claude-sonnet-4-20250514' });
      expect(agent).toBeDefined();
    });

    it('creates an Agent with custom maxIterations', async () => {
      const agent = await Agent.create({ maxIterations: 5 });
      expect(agent).toBeDefined();
    });

    it('creates an Agent with memoryEnabled: false', async () => {
      const agent = await Agent.create({ memoryEnabled: false });
      expect(agent).toBeDefined();
    });
  });

  describe('agent.run', () => {
    it('yields a done event when LLM gives a direct answer', async () => {
      mockState.invokeContent = 'Direct answer to query';
      mockState.streamChunks = ['Direct answer to query'];

      const agent = await Agent.create({ maxIterations: 3 });
      const events = await collectEvents(agent.run('What time is it?'));
      const done = events.find(e => e.type === 'done') as DoneEvent | undefined;
      expect(done).toBeDefined();
      expect(done!.answer).toContain('Direct answer to query');
    });

    it('done event includes iterations and totalTime', async () => {
      const agent = await Agent.create({ maxIterations: 3 });
      const events = await collectEvents(agent.run('simple question'));
      const done = events.find(e => e.type === 'done') as DoneEvent | undefined;
      expect(done).toBeDefined();
      expect(typeof done!.iterations).toBe('number');
      expect(typeof done!.totalTime).toBe('number');
      expect(done!.iterations).toBeGreaterThanOrEqual(1);
      expect(done!.totalTime).toBeGreaterThanOrEqual(0);
    });

    it('preloads an explicitly requested runtime skill before the first model turn', async () => {
      const agent = await Agent.create({ maxIterations: 3 });
      const events = await collectEvents(
        agent.run('Use the portfolio_risk skill to analyse risk for AAPL and MSFT'),
      );

      const toolStart = events.find(e => e.type === 'tool_start');
      const done = events.find(e => e.type === 'done') as DoneEvent | undefined;

      expect(toolStart).toBeDefined();
      expect(toolStart?.type).toBe('tool_start');
      if (toolStart?.type === 'tool_start') {
        expect(toolStart.tool).toBe('skill');
      }
      expect(done).toBeDefined();
      expect(done!.answer).toContain('Direct answer');
    });

    it('auto-runs the bounded forecast-lab improvement flow for routed execution queries', async () => {
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const events = await collectEvents(
        agent.run('Improve the BTC 1d/2d/3d Markov forecast workflow.'),
      );

      const toolStarts = events.filter((event) => event.type === 'tool_start');
      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['skill', 'forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'guided-improve',
          query: 'Improve the BTC 1d/2d/3d Markov forecast workflow.',
          profileId: 'btc-markov-ultra-short-horizon',
          routingSource: 'auto-routed',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('Approval required before promotion');
    });

    it('passes explicit mutator ids through the routed forecast-lab improvement flow', async () => {
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const events = await collectEvents(
        agent.run('Improve the BTC 1d/2d/3d Markov forecast workflow using mutator markov-faster-decay-reaction.'),
      );

      const toolStarts = events.filter((event) => event.type === 'tool_start');
      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['skill', 'forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'guided-improve',
          query: 'Improve the BTC 1d/2d/3d Markov forecast workflow using mutator markov-faster-decay-reaction.',
          profileId: 'btc-markov-ultra-short-horizon',
          mutator: 'markov-faster-decay-reaction',
          routingSource: 'auto-routed',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('Approval required before promotion');
    });

    it('uses the bounded forecast-lab plan path when the routed query forbids execution', async () => {
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const events = await collectEvents(
        agent.run('Optimize the BTC 1d/2d/3d Markov forecast workflow. Do not edit files, run shell commands, or write artifacts; explain the exact experiment plan you would follow.'),
      );

      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'guided-improve',
          query: 'Optimize the BTC 1d/2d/3d Markov forecast workflow. Do not edit files, run shell commands, or write artifacts; explain the exact experiment plan you would follow.',
          profileId: 'btc-markov-ultra-short-horizon',
          execute: false,
          routingSource: 'auto-routed',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('bounded plan');
      expect(done?.answer).toContain('.cramer-short/experiments');
    });

    it('runs bounded promotion on explicit approval prompts without another model turn', async () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Improve the BTC 1d/2d/3d Markov forecast workflow.',
        answer: 'Forecast-lab guided improvement finished. Approval required before promotion. Reply "approve forecast-lab promotion for btc-markov-ultra-short-horizon run btc-markov-ultra-short-horizon.keep-1" to continue.',
        summary: null,
      });
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const events = await collectEvents(
        agent.run('Yes, approve forecast-lab promotion for that kept run.', history),
      );

      const toolStarts = events.filter((event) => event.type === 'tool_start');
      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'promote-approved',
          profileId: 'btc-markov-ultra-short-horizon',
          sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('Promotion completed');
      expect(done?.answer).toContain('now live for normal forecasts');
    });

    it('runs bounded reset on explicit reset prompts without another model turn', async () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Approve the BTC ultra-short-horizon promotion.',
        answer: 'Promotion completed for btc-markov-ultra-short-horizon from kept run btc-markov-ultra-short-horizon.keep-1. Promoted parameters are now live for normal forecasts. Active baseline: .cramer-short/experiments/active-promotions/btc-markov-ultra-short-horizon.json',
        summary: null,
      });
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const events = await collectEvents(
        agent.run('Reset the forecast-lab active baseline for btc-markov-ultra-short-horizon back to shipped defaults.', history),
      );

      const toolStarts = events.filter((event) => event.type === 'tool_start');
      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'reset-live',
          profileId: 'btc-markov-ultra-short-horizon',
          resetMode: 'defaults',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('reset completed');
      expect(done?.answer).toContain('shipped defaults');
    });

    it('routes current-best vs shipped-baseline questions into the bounded forecast-lab comparison flow', async () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Improve the BTC 1d/2d/3d Markov forecast workflow.',
        answer: 'Forecast-lab guided improvement finished for btc-markov-ultra-short-horizon. Approval required before promotion.',
        summary: null,
      });
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const events = await collectEvents(
        agent.run('Is the current best better than the shipped default baseline?', history),
      );

      const toolStarts = events.filter((event) => event.type === 'tool_start');
      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'compare-best-vs-shipped',
          query: 'Is the current best better than the shipped default baseline?',
          profileId: 'btc-markov-ultra-short-horizon',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('not live yet');
      expect(done?.answer).toContain('Approve forecast-lab promotion');
    });

    it('routes named mutator-vs-active comparison prompts into the bounded forecast-lab comparison flow', async () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Target anchor trust weighting. Name it something like: markov-entropy-adaptive-anchor-weighting.',
        answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. Requested mutator id: markov-entropy-adaptive-anchor-weighting.',
        summary: null,
      });
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const query = 'I need to compare the markov-entropy-adaptive-anchor-weighting with the active one, I need to see the accurace numbers';
      const events = await collectEvents(agent.run(query, history));

      const toolStarts = events.filter((event) => event.type === 'tool_start');

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'compare-best-vs-shipped',
          query,
          profileId: 'btc-markov-ultra-short-horizon',
          mutationId: 'markov-entropy-adaptive-anchor-weighting',
        },
      ]);
      expect(mockState.callCount).toBe(0);
    });

    it('routes history-based live-vs-new-mutator prompts into the bounded forecast-lab comparison flow', async () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Target anchor trust weighting. Name it something like: markov-entropy-adaptive-anchor-weighting.',
        answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. Requested mutator id: markov-entropy-adaptive-anchor-weighting.',
        summary: null,
      });
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const query = 'I need to compare the live one with the new mutate one that I created and it is not promoted';
      const events = await collectEvents(agent.run(query, history));

      const toolStarts = events.filter((event) => event.type === 'tool_start');

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'compare-best-vs-shipped',
          query,
          profileId: 'btc-markov-ultra-short-horizon',
          mutationId: 'markov-entropy-adaptive-anchor-weighting',
        },
      ]);
      expect(mockState.callCount).toBe(0);
    });

    it('routes bare named mutator-vs-active comparison prompts into the bounded forecast-lab comparison flow', async () => {
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const query = 'I need to compare the markov-entropy-adaptive-anchor-weighting with the active one, I need to see the accurace numbers';
      const events = await collectEvents(agent.run(query));

      const toolStarts = events.filter((event) => event.type === 'tool_start');

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'compare-best-vs-shipped',
          query,
          mutationId: 'markov-entropy-adaptive-anchor-weighting',
        },
      ]);
      expect(mockState.callCount).toBe(0);
    });

    it('routes forecast-lab results prompts into the bounded comparison flow instead of guided improvement', async () => {
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const events = await collectEvents(
        agent.run('provide the results of the Optimize the BTC 1d/2d/3d Markov forecast workflow'),
      );

      const toolStarts = events.filter((event) => event.type === 'tool_start');
      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'compare-best-vs-shipped',
          query: 'provide the results of the Optimize the BTC 1d/2d/3d Markov forecast workflow',
          profileId: 'btc-markov-ultra-short-horizon',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('Approve forecast-lab promotion');
    });

    it('routes mutator-list prompts into the bounded forecast-lab catalog flow', async () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Improve the BTC 1d/2d/3d Markov forecast workflow.',
        answer: 'Forecast-lab guided improvement finished for btc-markov-ultra-short-horizon. Approval required before promotion.',
        summary: null,
      });
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const query = 'List the mutate availible';
      const events = await collectEvents(agent.run(query, history));

      const toolStarts = events.filter((event) => event.type === 'tool_start');
      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'list-mutators',
          query,
          profileId: 'btc-markov-ultra-short-horizon',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('shipped mutator ids');
      expect(done?.answer).toContain('markov-shorter-reactive-window');
    });

    it('routes keep-the-current-best follow-ups into the bounded results flow instead of file edits', async () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'provide the results of the Optimize the BTC 1d/2d/3d Markov forecast workflow',
        answer: 'Forecast-lab comparison for btc-markov-ultra-short-horizon. Current best is not live yet. Reply "Approve forecast-lab promotion for btc-markov-ultra-short-horizon run btc-markov-ultra-short-horizon.keep-1." to activate it.',
        summary: null,
      });
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const events = await collectEvents(
        agent.run('keep the current best candidate', history),
      );

      const toolStarts = events.filter((event) => event.type === 'tool_start');
      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'compare-best-vs-shipped',
          query: 'keep the current best candidate',
          profileId: 'btc-markov-ultra-short-horizon',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('Approve forecast-lab promotion');
    });

    it('routes catalog-extension prompts into the bounded forecast-lab plan flow instead of generic exploration', async () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Use the forecast-lab skill to explain what to do when no shipped structured mutator remains applicable after replaying the kept parent lineage for btc-markov-ultra-short-horizon.',
        answer: 'No shipped structured mutator remains applicable after replaying the kept parent lineage for profile "btc-markov-ultra-short-horizon". Next actions: keep the current best candidate, add a new shipped structured mutator, or intentionally reset the forecast-lab lineage outside the CLI.',
        summary: null,
      });
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const events = await collectEvents(
        agent.run('design a new shipped mutator outside the existing catalog and re-run the lineage', history),
      );

      const toolStarts = events.filter((event) => event.type === 'tool_start');
      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'catalog-extension-plan',
          query: 'design a new shipped mutator outside the existing catalog and re-run the lineage',
          profileId: 'btc-markov-ultra-short-horizon',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('bounded code-change plan');
      expect(done?.answer).toContain('did not inspect experiment artifacts');
    });

    it('routes detailed shipped-mutator implementation briefs into the bounded forecast-lab plan flow', async () => {
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const query = [
        'Target anchor trust weighting.',
        'Add a new shipped structured mutator for btc-markov-ultra-short-horizon that makes the Markov/anchor blend more adaptive under high posterior entropy using the existing soft-regime weighting controls in src/tools/finance/markov-distribution.ts.',
        'Mutator goal:',
        '- reduce retained HMM weight when the regime posterior is ambiguous,',
        '- widen CI slightly more under entropy,',
        '- lower confidence more under entropy,',
        '- keep the change bounded to the existing soft-regime weighting parameters.',
        'Suggested starting values:',
        '- softRegimeConfidenceFloor: 0.65 -> 0.55',
        '- softRegimeConfidenceEntropyWeight: 0.35 -> 0.50',
        '- softRegimeCiEntropyWeight: 0.35 -> 0.50',
        '- softRegimeHmmWeightFloor: 0.50 -> 0.35',
        '- softRegimeHmmWeightEntropyWeight: 0.40 -> 0.60',
        'Name it something like:',
        'markov-entropy-adaptive-anchor-weighting',
        'Keep it bounded, add the shipped mutator to the catalog, and validate it with the existing BTC ultra-short-horizon walk-forward gate.',
      ].join('\n');

      const events = await collectEvents(agent.run(query));
      const toolStarts = events.filter((event) => event.type === 'tool_start');
      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'catalog-extension-plan',
          query,
          profileId: 'btc-markov-ultra-short-horizon',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('src/experiments/forecast-lab/mutators/markov-parameters.ts');
      expect(forecastLabMockState.calls[0]).toMatchObject({
        query: expect.stringContaining('markov-entropy-adaptive-anchor-weighting'),
      });
    });

    it('routes implement-and-run requests for non-shipped mutators into the bounded catalog-extension flow', async () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Target anchor trust weighting. Add a new shipped structured mutator for btc-markov-ultra-short-horizon.',
        answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. Requested mutator id: markov-entropy-adaptive-anchor-weighting.',
        summary: null,
      });
      const agent = await Agent.create({ maxIterations: 3 });
      installForecastLabTool(agent as any);
      const events = await collectEvents(
        agent.run('implement and run the markov-entropy-adaptive-anchor-weighting', history),
      );

      const toolStarts = events.filter((event) => event.type === 'tool_start');
      const done = events.find((event) => event.type === 'done') as DoneEvent | undefined;

      expect(toolStarts.map((event) => (event as { tool: string }).tool)).toEqual(['forecast_lab_run']);
      expect(forecastLabMockState.calls).toEqual([
        {
          action: 'catalog-extension-plan',
          query: 'implement and run the markov-entropy-adaptive-anchor-weighting',
          profileId: 'btc-markov-ultra-short-horizon',
        },
      ]);
      expect(mockState.callCount).toBe(0);
      expect(done?.answer).toContain('bounded code-change plan');
      expect(done?.answer).toContain('did not inspect experiment artifacts');
    });

    it('yields done with error message when LLM throws', async () => {
      mockState.invokeThrows = true;

      const agent = await Agent.create({ maxIterations: 3 });
      const events = await collectEvents(agent.run('error query'));
      const done = events.find(e => e.type === 'done') as DoneEvent | undefined;
      expect(done).toBeDefined();
      expect(done!.answer.length).toBeGreaterThanOrEqual(0); // graceful failure
    });

    it('yields tool_start and tool_end events when a tool is called', async () => {
      // Set ST tool call first (satisfies ST-first enforcement), then clear for answer
      mockState.invokeToolCalls = [ST_TOOL_CALL];

      const requestToolApproval = mock(async () => 'allow' as const);
      const agent = await Agent.create({ maxIterations: 10, requestToolApproval: requestToolApproval as any });

      const events = await collectEvents(agent.run('search something'));
      const done = events.find(e => e.type === 'done');
      expect(done).toBeDefined();
    });

    it('forces the explicit GOLD combined workflow in Markov → market data → Polymarket → arbiter order', async () => {
      const requestToolApproval = mock(async () => 'allow' as const);
      const agent = await Agent.create({ maxIterations: 10, requestToolApproval: requestToolApproval as any });
      const agentAny = agent as any;

      agentAny.toolMap.set('markov_distribution', {
        name: 'markov_distribution',
        invoke: async () => JSON.stringify({
          data: {
            _tool: 'markov_distribution',
            status: 'ok',
            canonical: {
              ticker: 'GLD',
              currentPrice: 312.45,
              scenarios: {
                expectedReturn: 0.018,
                pUp: 0.63,
                buckets: [{ label: 'Up >3%', probability: 0.28 }],
              },
              actionSignal: {
                recommendation: 'BUY',
                expectedReturn: 0.018,
              },
              diagnostics: {
                markovWeight: 0.82,
                predictionConfidence: 0.67,
              },
            },
          },
        }),
      });
      agentAny.toolMap.set('get_market_data', {
        name: 'get_market_data',
        invoke: async () => JSON.stringify({
          data: {
            get_stock_price_GLD: {
              ticker: 'GLD',
              price: 312.45,
            },
          },
        }),
      });
      agentAny.toolMap.set('polymarket_forecast', {
        name: 'polymarket_forecast',
        invoke: async () => JSON.stringify({
          data: {
            forecastReturn: -0.012,
            result: 'Polymarket Forecast: GLD | Horizon: 30 days | Grade: B+ (78/100)\nWill gold finish May above $3,250?: 39% YES',
          },
        }),
      });
      agentAny.toolMap.set('forecast_arbitrator', {
        name: 'forecast_arbitrator',
        invoke: async () => JSON.stringify({
          data: {
            result: {
              verdict: 'NO_TRADE',
              preferredDirection: 'neutral',
              shouldEnterNow: false,
            },
          },
        }),
      });
      agentAny.toolExecutor = new AgentToolExecutor(agentAny.toolMap);

      const events = await collectEvents(
        agent.run('Provide a GOLD price forecast based on markov chain and polymarket for the next 30 days with trade direction'),
      );

      const toolStarts = events
        .filter((event) => event.type === 'tool_start')
        .map((event) => (event as { tool: string }).tool)
        .filter((tool) => ['markov_distribution', 'get_market_data', 'polymarket_forecast', 'forecast_arbitrator'].includes(tool));

      expect(toolStarts).toEqual([
        'markov_distribution',
        'get_market_data',
        'polymarket_forecast',
        'forecast_arbitrator',
      ]);
    });

    it('done event includes toolCalls array', async () => {
      mockState.streamChunks = ['final answer'];
      const agent = await Agent.create({ maxIterations: 3 });
      const events = await collectEvents(agent.run('test'));
      const done = events.find(e => e.type === 'done') as DoneEvent | undefined;
      expect(done).toBeDefined();
      expect(Array.isArray(done!.toolCalls)).toBe(true);
    });

    it('emits answer_start before done', async () => {
      const agent = await Agent.create({ maxIterations: 3 });
      const events = await collectEvents(agent.run('test query'));
      const answerStart = events.find(e => e.type === 'answer_start');
      const done = events.find(e => e.type === 'done');
      expect(answerStart).toBeDefined();
      expect(done).toBeDefined();
      const idxStart = events.indexOf(answerStart!);
      const idxDone = events.indexOf(done!);
      expect(idxStart).toBeLessThan(idxDone);
    });

    it('infers ticker and horizon for explicit terminal distribution queries', () => {
      expect(inferDistributionTicker('What is the probability distribution for BTC-USD in 7 trading days?')).toBe('BTC-USD');
      expect(inferDistributionHorizon('What is the probability distribution for BTC-USD in 7 trading days?')).toBe(7);
    });

    describe('shouldInjectBtcShortHorizonMixedEvidencePrompt', () => {
      it('returns true for BTC short-horizon disagreement', () => {
        const results = [
          '### markov_distribution(ticker=BTC-USD)',
          '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"BUY","expectedReturn":0.032}}}}',
          '### polymarket_forecast(ticker=BTC-USD)',
          'Forecast return: -0.4%\nGrade: B',
        ].join('\n');
        expect(shouldInjectBtcShortHorizonMixedEvidencePrompt('Provide a BTC forecast for the next 14 days', results)).toBe(true);
      });

      it('returns false for non-BTC assets', () => {
        const results = [
          '### markov_distribution(ticker=ETH-USD)',
          '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"BUY","expectedReturn":0.032}}}}',
          '### polymarket_forecast(ticker=ETH-USD)',
          'Forecast return: -0.4%\nGrade: B',
        ].join('\n');
        expect(shouldInjectBtcShortHorizonMixedEvidencePrompt('Provide an ETH forecast for the next 14 days', results)).toBe(false);
      });

      it('returns false for BTC horizons above 14 days', () => {
        const results = [
          '### markov_distribution(ticker=BTC-USD)',
          '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"BUY","expectedReturn":0.032}}}}',
          '### polymarket_forecast(ticker=BTC-USD)',
          'Forecast return: -0.4%\nGrade: B',
        ].join('\n');
        expect(shouldInjectBtcShortHorizonMixedEvidencePrompt('Provide a BTC forecast for the next 30 days', results)).toBe(false);
      });

      it('returns false for unrelated BUY text without bullish Markov payload', () => {
        const results = [
          '### some_other_tool()',
          '{"recommendation":"BUY"}',
          '### polymarket_forecast(ticker=BTC-USD)',
          'Forecast return: -0.4%\nGrade: B',
        ].join('\n');
        expect(shouldInjectBtcShortHorizonMixedEvidencePrompt('Provide a BTC forecast for the next 7 days', results)).toBe(false);
      });

      it('returns false for unrelated bearish forecast text without matching BTC disagreement', () => {
        const results = [
          '### markov_distribution(ticker=BTC-USD)',
          '{"data":{"_tool":"markov_distribution","status":"ok","canonical":{"actionSignal":{"recommendation":"SELL","expectedReturn":-0.012}}}}',
          '### unrelated_text()',
          'Forecast return: -0.4%\nGrade: B',
        ].join('\n');
        expect(shouldInjectBtcShortHorizonMixedEvidencePrompt('Provide a BTC forecast for the next 7 days', results)).toBe(false);
      });
    });

    it('infers month and quarter horizons for non-crypto forecast phrasing', () => {
      expect(inferDistributionHorizon('SPY price target next month')).toBe(21);
      expect(inferDistributionHorizon('Will GLD rally next quarter?')).toBe(63);
      expect(inferDistributionHorizon('Where will oil prices be in 2 months?')).toBe(42);
      expect(inferDistributionHorizon('QQQ outlook in 2 quarters')).toBe(126);
    });

    it('infers hourly short horizons for crypto forecast phrasing', () => {
      expect(inferDistributionHorizon('BTC forecast over the next 24 hours')).toBe(1);
      expect(inferDistributionHorizon('BTC forecast over the next 48h')).toBe(2);
      expect(buildForcedCryptoForecastMarkovArgs('Give me a BTC markov forecast over the next 24 hours')).toEqual({
        ticker: 'BTC-USD',
        horizon: 1,
        trajectory: true,
        trajectoryDays: 1,
      });
    });

    it('infers quarter-end horizons relative to a provided reference date', () => {
      const referenceDate = new Date('2026-04-17T00:00:00.000Z');

      expect(inferDistributionHorizon('Will silver hit $30 by end of Q2?', referenceDate)).toBe(52);
      expect(inferDistributionHorizon('Where will SPY trade through Q3 2026?', referenceDate)).toBe(118);
    });

    it('treats same-day quarter-end requests as a 1-day horizon instead of falling through', () => {
      const referenceDate = new Date('2026-06-30T00:00:00.000Z');

      expect(inferDistributionHorizon('Will silver hit $30 by end of Q2?', referenceDate)).toBe(1);
      expect(inferDistributionHorizon('Will silver hit $30 by end of Q2 2026?', referenceDate)).toBe(1);
    });

    it('self-corrects commodity gold and silver distribution queries to proxy tickers', () => {
      expect(inferDistributionTicker('What is the probability distribution for gold price in 30 trading days?')).toBe('GLD');
      expect(inferDistributionTicker('What is the probability distribution for silver price in 30 trading days?')).toBe('SLV');
      expect(buildForcedMarkovArgs('What is the probability distribution for silver price in 30 trading days?')).toEqual({
        ticker: 'SLV',
        horizon: 30,
      });
    });

    it('routes the open-ended GOLD and SILVER markov prompt shape through commodity proxies', () => {
      expect(inferDistributionTicker('Provide a GOLD forecast based on markov chain for the next 30 days')).toBe('GLD');
      expect(buildForcedMarkovArgs('Provide a GOLD forecast based on markov chain for the next 30 days')).toEqual({
        ticker: 'GLD',
        horizon: 30,
      });

      expect(inferDistributionTicker('Provide a SILVER forecast based on markov chain for the next 30 days')).toBe('SLV');
      expect(buildForcedMarkovArgs('Provide a SILVER forecast based on markov chain for the next 30 days')).toEqual({
        ticker: 'SLV',
        horizon: 30,
      });
    });

    it('routes the exact GOLD markov + polymarket prompt through the commodity proxy path across 1d/2d/3d/14d horizons', () => {
      for (const days of [1, 2, 3, 14]) {
        const query = `Provide a GOLD price forecast based on markov chain and polymarket for the next ${days} day${days === 1 ? '' : 's'}`;
        expect(inferDistributionTicker(query)).toBe('GLD');
        expect(buildForcedMarkovArgs(query)).toEqual({
          ticker: 'GLD',
          horizon: days,
        });
      }
    });

    it('routes open-ended OIL markov prompts through the oil commodity proxy path', () => {
      expect(inferDistributionTicker('Provide a OIL price forecast based on markov chain and polymarket for the next 14 days')).toBe('USO');
      expect(buildForcedMarkovArgs('Provide a OIL price forecast based on markov chain and polymarket for the next 14 days')).toEqual({
        ticker: 'USO',
        horizon: 14,
      });
    });

    it('routes WTI crude oil markov prompts through the oil commodity proxy path', () => {
      expect(inferDistributionTicker('Provide a WTI crude oil forecast based on markov chain for the next 14 days')).toBe('USO');
      expect(buildForcedMarkovArgs('Provide a WTI crude oil forecast based on markov chain for the next 14 days')).toEqual({
        ticker: 'USO',
        horizon: 14,
      });
    });

    it('preserves explicit Barrick Gold equity queries as GOLD ticker', () => {
      expect(inferDistributionTicker('What is the probability distribution for Barrick Gold stock in 30 trading days?')).toBe('GOLD');
    });

    it('flags explicit terminal distribution and trajectory queries for forced markov routing', () => {
      expect(shouldForceMarkovDistribution(
        'What is the probability distribution for BTC-USD in 7 trading days? Use the Markov distribution methodology with terminal threshold markets only.',
        [],
      )).toBe(true);

      expect(shouldForceMarkovDistribution(
        'Show me the day-by-day trajectory for NVDA over 14 days',
        [],
      )).toBe(true);

      expect(shouldForceMarkovDistribution(
        'What is the probability distribution for BTC-USD in 7 trading days?',
        [{ tool: 'markov_distribution', args: { ticker: 'BTC-USD', horizon: 7 }, result: '{"data":{"_tool":"markov_distribution","status":"abstain"}}' }],
      )).toBe(false);

      expect(shouldForceMarkovDistribution(
        'Show me the day-by-day trajectory for NVDA over 14 days',
        [{ tool: 'markov_distribution', args: { ticker: 'NVDA', horizon: 14, trajectory: true, trajectoryDays: 14 }, result: '{"data":{"_tool":"markov_distribution","status":"ok"}}' }],
      )).toBe(false);
    });

    it('treats markov chain forecast wording as a forecast rather than an explicit distribution query', () => {
      expect(isExplicitTerminalDistributionQuery('Provide an NVDA forecast based on markov chain for the next 7 days')).toBe(true);
      expect(isNonCryptoForecastQuery('Provide an NVDA forecast based on markov chain for the next 7 days')).toBe(true);
      expect(shouldForceMarkovDistribution('Provide an NVDA forecast based on markov chain for the next 7 days', [])).toBe(true);
    });

    it('inferTrajectoryRequest detects explicit trajectory/day-by-day queries', () => {
      expect(inferTrajectoryRequest('Show me the day-by-day trajectory for NVDA over 14 days')).toBe(true);
      expect(inferTrajectoryRequest('What is the price path for AAPL over the next 7 trading days?')).toBe(true);
      expect(inferTrajectoryRequest('Give me a daily forecast for SPY')).toBe(true);
      expect(inferTrajectoryRequest('daily projection for BTC-USD 30 day horizon')).toBe(true);
      expect(inferTrajectoryRequest('trajectory analysis for GLD')).toBe(true);
    });

    it('inferTrajectoryRequest returns false for non-trajectory distribution queries', () => {
      expect(inferTrajectoryRequest('What is the probability distribution for BTC-USD in 7 days?')).toBe(false);
      expect(inferTrajectoryRequest('Will NVDA hit $200 in 30 days?')).toBe(false);
      expect(inferTrajectoryRequest('markov distribution for AAPL')).toBe(false);
    });

    it('buildForcedMarkovArgs preserves trajectory flags for trajectory queries', () => {
      expect(buildForcedMarkovArgs('Give me a 7-day price trajectory for AAPL')).toEqual({
        ticker: 'AAPL',
        horizon: 7,
        trajectory: true,
        trajectoryDays: 7,
      });

      expect(buildForcedMarkovArgs('Show me a day-by-day trajectory for BTC-USD over 8 weeks')).toEqual({
        ticker: 'BTC-USD',
        horizon: 40,
        trajectory: true,
        trajectoryDays: 30,
      });

      expect(buildForcedMarkovArgs('What is the probability distribution for BTC-USD in 7 trading days?')).toEqual({
        ticker: 'BTC-USD',
        horizon: 7,
      });
    });

    it('detects crypto forecast queries without matching explicit distribution requests', () => {
      expect(isCryptoForecastQuery('Provide a BTC forecast for the next 7 days')).toBe(true);
      expect(isCryptoForecastQuery('What will ETH trade at next week?')).toBe(true);
      expect(isCryptoForecastQuery('Improve the BTC short-horizon forecast workflow.')).toBe(false);
      expect(isCryptoForecastQuery('Use the probability_assessment skill for BTC price movement in the next 30 days')).toBe(false);
      expect(isCryptoForecastQuery('What is the market cap of BTC?')).toBe(false);
      expect(isCryptoForecastQuery('What is the probability distribution for BTC-USD in 7 days?')).toBe(false);
      expect(isCryptoForecastQuery('Provide an AAPL forecast for the next 7 days')).toBe(false);
    });

    it('classifies routed forecast-lab workflow asks as improvement queries', () => {
      expect(isForecastLabImprovementQuery('Improve the BTC short-horizon forecast workflow.')).toBe(true);
      expect(isForecastLabImprovementQuery('Give me a BTC forecast for the next 7 days.')).toBe(false);
    });

    it('detects plan-only forecast-lab prompts that should not execute the runner', () => {
      expect(
        isForecastLabPlanOnlyQuery(
          'Optimize the BTC 1d/2d/3d workflow. Do not edit files, run shell commands, or write artifacts; explain the exact experiment plan you would follow.',
        ),
      ).toBe(true);
      expect(isForecastLabPlanOnlyQuery('Improve the BTC 1d/2d/3d workflow and run the bounded experiment.')).toBe(false);
    });

    it('detects explicit forecast-lab promotion approvals from history context', () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Improve the BTC 1d/2d/3d Markov forecast workflow.',
        answer: 'Forecast-lab guided improvement finished. Approval required before promotion. Reply "approve forecast-lab promotion for btc-markov-ultra-short-horizon run btc-markov-ultra-short-horizon.keep-1" to continue.',
        summary: null,
      });

      expect(
        detectForecastLabPromotionApproval('Yes, approve forecast-lab promotion for that kept run.', history),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
        sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
      });
      expect(
        detectForecastLabPromotionApproval(
          'Use the forecast-lab skill to explain the full BTC ultra-short-horizon lifecycle after a kept candidate: approval-required promotion, activation for ordinary forecasts, and how to reset to shipped defaults or the last-known-good baseline if the promoted parameters mislead. Do not edit files, run shell commands, or write artifacts.',
          history,
        ),
      ).toBeNull();
      expect(detectForecastLabPromotionApproval('How do I promote these improvements?', history)).toBeNull();
      expect(detectForecastLabPromotionApproval('What is BTC doing next week?', history)).toBeNull();
    });

    it('detects explicit forecast-lab reset requests from history context', () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Approve the BTC ultra-short-horizon promotion.',
        answer: 'Promotion completed for btc-markov-ultra-short-horizon from kept run btc-markov-ultra-short-horizon.keep-1. Promoted parameters are now live for normal forecasts. Active baseline: .cramer-short/experiments/active-promotions/btc-markov-ultra-short-horizon.json',
        summary: null,
      });

      expect(
        detectForecastLabResetRequest(
          'Please reset the forecast-lab baseline for btc-markov-ultra-short-horizon to shipped defaults.',
          history,
        ),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
        mode: 'defaults',
      });
      expect(
        detectForecastLabResetRequest(
          'Restore the forecast-lab baseline for btc-markov-ultra-short-horizon to the last known good activation.',
          history,
        ),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
        mode: 'last-known-good',
      });
      expect(detectForecastLabResetRequest('Reset BTC to defaults.', history)).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
        mode: 'defaults',
      });
      expect(detectForecastLabResetRequest('Reset BTC to defaults.')).toBeNull();
    });

    it('detects current-best vs shipped-baseline comparison requests from history context', () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Improve the BTC 1d/2d/3d Markov forecast workflow.',
        answer: 'Forecast-lab guided improvement finished for btc-markov-ultra-short-horizon. Approval required before promotion.',
        summary: null,
      });
      history.seedMessage({
        query: 'Target anchor trust weighting. Name it something like: markov-entropy-adaptive-anchor-weighting.',
        answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. Requested mutator id: markov-entropy-adaptive-anchor-weighting.',
        summary: null,
      });

      expect(
        detectForecastLabComparisonRequest('Is the current best better than the shipped default baseline?', history),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
      });
      expect(
        detectForecastLabComparisonRequest(
          'I need to compare the markov-entropy-adaptive-anchor-weighting with the active one, I need to see the accurace numbers',
          history,
        ),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
        mutationId: 'markov-entropy-adaptive-anchor-weighting',
      });
      expect(
        detectForecastLabComparisonRequest(
          'I need to compare the live one with the new mutate one that I created and it is not promoted',
          history,
        ),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
        mutationId: 'markov-entropy-adaptive-anchor-weighting',
      });
      expect(
        detectForecastLabComparisonRequest(
          'I need to compare the markov-entropy-adaptive-anchor-weighting with the active one, I need to see the accurace numbers',
        ),
      ).toEqual({
        mutationId: 'markov-entropy-adaptive-anchor-weighting',
      });
      expect(detectForecastLabComparisonRequest('Is BTC going up next week?', history)).toBeNull();
    });

    it('detects forecast-lab results requests and resolves the routed profile', () => {
      expect(
        detectForecastLabResultsRequest('provide the results of the Optimize the BTC 1d/2d/3d Markov forecast workflow'),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
      });
      expect(detectForecastLabResultsRequest('show me BTC results for next week')).toBeNull();
    });

    it('detects mutator-list requests and reuses forecast-lab profile context when available', () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Improve the BTC 1d/2d/3d Markov forecast workflow.',
        answer: 'Forecast-lab guided improvement finished for btc-markov-ultra-short-horizon. Approval required before promotion.',
        summary: null,
      });

      expect(
        detectForecastLabMutatorListRequest('List the mutator ids for btc-markov-ultra-short-horizon.', history),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
      });
      expect(
        detectForecastLabMutatorListRequest('List the mutate availible', history),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
      });
      expect(detectForecastLabMutatorListRequest('List the mutate availible')).toEqual({});
      expect(detectForecastLabMutatorListRequest('List the plugins for my toy parser')).toBeNull();
    });

    it('detects keep-the-current-best follow-ups only in forecast-lab context', () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'provide the results of the Optimize the BTC 1d/2d/3d Markov forecast workflow',
        answer: 'Forecast-lab comparison for btc-markov-ultra-short-horizon. Current best is not live yet.',
        summary: null,
      });

      expect(
        detectForecastLabKeepCurrentBestRequest('keep the current best candidate', history),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
      });
      expect(detectForecastLabKeepCurrentBestRequest('keep the current best candidate')).toBeNull();
    });

    it('detects catalog-extension requests and reuses forecast-lab profile context when available', () => {
      const history = new InMemoryChatHistory();
      history.seedMessage({
        query: 'Use the forecast-lab skill to explain what to do when no shipped structured mutator remains applicable after replaying the kept parent lineage for btc-markov-ultra-short-horizon.',
        answer: 'No shipped structured mutator remains applicable after replaying the kept parent lineage for profile "btc-markov-ultra-short-horizon".',
        summary: null,
      });

      expect(
        detectForecastLabCatalogExtensionRequest(
          'design a new shipped mutator outside the existing catalog and re-run the lineage',
          history,
        ),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
      });
      expect(
        detectForecastLabCatalogExtensionRequest('design a new shipped mutator outside the existing catalog and re-run the lineage'),
      ).toEqual({});
      expect(
        detectForecastLabCatalogExtensionRequest(
          [
            'Target anchor trust weighting.',
            'Add a new shipped structured mutator for btc-markov-ultra-short-horizon that makes the Markov/anchor blend more adaptive under high posterior entropy using the existing soft-regime weighting controls in src/tools/finance/markov-distribution.ts.',
            'Keep it bounded, add the shipped mutator to the catalog, and validate it with the existing BTC ultra-short-horizon walk-forward gate.',
          ].join('\n'),
        ),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
      });
      history.seedMessage({
        query: 'Target anchor trust weighting. Add a new shipped structured mutator for btc-markov-ultra-short-horizon.',
        answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. Requested mutator id: markov-entropy-adaptive-anchor-weighting.',
        summary: null,
      });
      expect(
        detectForecastLabCatalogExtensionRequest(
          'implement and run the markov-entropy-adaptive-anchor-weighting',
          history,
        ),
      ).toEqual({
        profileId: 'btc-markov-ultra-short-horizon',
      });
      expect(detectForecastLabCatalogExtensionRequest('design a new mutator for my toy parser')).toBeNull();
    });

    it('builds forced crypto enrichment args for BTC forecasts', () => {
      expect(buildForcedMarketDataArgs('Provide a BTC forecast for the next 7 days')).toEqual({
        query: 'Current crypto price snapshot for BTC',
      });

      expect(buildForcedSocialSentimentArgs('Provide a BTC forecast for the next 7 days')).toEqual({
        ticker: 'BTC',
        include_fear_greed: true,
        limit: 25,
      });

      expect(buildForcedOnchainArgs('Provide a BTC forecast for the next 7 days')).toEqual({
        ticker: 'BTC',
        metrics: ['market', 'sentiment'],
      });

      expect(buildForcedFixedIncomeArgs()).toEqual({
        series: ['treasury_yields', 'yield_curve'],
      });

      expect(buildForcedCryptoForecastMarkovArgs('Provide a BTC forecast for the next 7 days')).toEqual({
        ticker: 'BTC-USD',
        horizon: 7,
        trajectory: true,
        trajectoryDays: 7,
      });

      expect(buildForcedCryptoForecastMarkovArgs('Provide a BTC forecast for the next 30 days')).toBeNull();
      expect(buildForcedOnchainArgs('Provide a crypto forecast for the next 7 days')).toBeNull();
    });

    it('buildForcedCryptoForecastMarkovArgs uses BTC-only next week fallback of 5 trading days', () => {
      expect(buildForcedCryptoForecastMarkovArgs('Provide a BTC forecast for next week')).toEqual({
        ticker: 'BTC-USD',
        horizon: 5,
        trajectory: true,
        trajectoryDays: 5,
      });

      expect(buildForcedCryptoForecastMarkovArgs('Give me a BTC prediction over the next week')).toEqual({
        ticker: 'BTC-USD',
        horizon: 5,
        trajectory: true,
        trajectoryDays: 5,
      });

      // ETH "next week" must not get the BTC-only fallback
      expect(buildForcedCryptoForecastMarkovArgs('Provide an ETH forecast for next week')).toBeNull();

      // Non-crypto "next week" must not get the fallback
      expect(buildForcedCryptoForecastMarkovArgs('Provide an AAPL forecast for next week')).toBeNull();
    });

    it('extracts current price, sentiment score, and Markov return from prior tool results for forced polymarket args', () => {
      const toolCalls = [
        {
          tool: 'get_market_data',
          args: { query: 'Current crypto price snapshot for BTC' },
          result: JSON.stringify({
            data: {
              get_crypto_price_snapshot_BTC: {
                ticker: 'BTC',
                price: 73300,
              },
            },
          }),
        },
        {
          tool: 'social_sentiment',
          args: { ticker: 'BTC', include_fear_greed: true, limit: 25 },
          result: JSON.stringify({
            data: {
              result: '## Overall: 📈 Bullish (score +42/100)',
            },
          }),
        },
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 7, trajectory: true, trajectoryDays: 7 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'ok',
              canonical: {
                actionSignal: {
                  expectedReturn: 0.04,
                },
                diagnostics: {
                  markovWeight: 0.6,
                },
              },
            },
          }),
        },
      ];

      expect(extractCurrentPriceFromToolCalls(toolCalls)).toBe(73300);
      expect(extractSentimentScoreFromToolCalls(toolCalls)).toBe(0.42);
      expect(extractMarkovReturnFromToolCalls(toolCalls)).toBeCloseTo(0.024, 8);
      expect(buildForcedPolymarketForecastArgs('Provide a BTC forecast for the next 7 days', toolCalls)).toEqual({
        ticker: 'BTC',
        horizon_days: 7,
        current_price: 73300,
        sentiment_score: 0.42,
        markov_return: 0.024,
      });
    });

    it('scopes crypto forced polymarket args to the resolved asset and horizon', () => {
      const toolCalls = [
        {
          tool: 'get_market_data',
          args: { query: 'Current crypto price snapshot for ETH' },
          result: JSON.stringify({
            data: {
              get_crypto_price_snapshot_ETH: {
                ticker: 'ETH',
                price: 3900,
              },
            },
          }),
        },
        {
          tool: 'social_sentiment',
          args: { ticker: 'ETH', include_fear_greed: true, limit: 25 },
          result: JSON.stringify({
            data: {
              result: '## Overall: 📈 Bullish (score +42/100)',
            },
          }),
        },
        {
          tool: 'markov_distribution',
          args: { ticker: 'ETH-USD', horizon: 7, trajectory: true, trajectoryDays: 7 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'ok',
              canonical: {
                actionSignal: { expectedReturn: 0.08 },
                diagnostics: { markovWeight: 0.5 },
              },
            },
          }),
        },
      ];

      expect(buildForcedPolymarketForecastArgs('Provide a BTC forecast for the next 7 days', toolCalls)).toEqual({
        ticker: 'BTC',
        horizon_days: 7,
      });
    });

    it('does not reuse abstain-derived Markov return in forced crypto polymarket args', () => {
      const toolCalls = [
        {
          tool: 'get_market_data',
          args: { query: 'Current crypto price snapshot for BTC' },
          result: JSON.stringify({
            data: {
              get_crypto_price_snapshot_BTC: {
                ticker: 'BTC',
                price: 73300,
              },
            },
          }),
        },
        {
          tool: 'social_sentiment',
          args: { ticker: 'BTC', include_fear_greed: true, limit: 25 },
          result: JSON.stringify({
            data: {
              result: '## Overall: 📈 Bullish (score +42/100)',
            },
          }),
        },
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 7, trajectory: true, trajectoryDays: 7 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              forecastHint: {
                usage: 'forecast_only',
                markovReturn: 0.006,
                confidenceScore: 0.18,
                calibratedDistribution: false,
              },
            },
          }),
        },
      ];

      expect(buildForcedPolymarketForecastArgs('Provide a BTC forecast for the next 7 days', toolCalls)).toEqual({
        ticker: 'BTC',
        horizon_days: 7,
        current_price: 73300,
        sentiment_score: 0.42,
      });
    });

    it('does not reuse low-confidence BTC Markov return in forced crypto polymarket args', () => {
      const toolCalls = [
        {
          tool: 'get_market_data',
          args: { query: 'Current crypto price snapshot for BTC' },
          result: JSON.stringify({
            data: {
              get_crypto_price_snapshot_BTC: {
                ticker: 'BTC',
                price: 73300,
              },
            },
          }),
        },
        {
          tool: 'social_sentiment',
          args: { ticker: 'BTC', include_fear_greed: true, limit: 25 },
          result: JSON.stringify({
            data: {
              result: '## Overall: 📈 Bullish (score +42/100)',
            },
          }),
        },
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 7, trajectory: true, trajectoryDays: 7 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'ok',
              canonical: {
                actionSignal: { expectedReturn: 0.04 },
                diagnostics: {
                  markovWeight: 0.6,
                  predictionConfidence: 0.18,
                },
              },
            },
          }),
        },
      ];

      expect(buildForcedPolymarketForecastArgs('Provide a BTC forecast for the next 7 days', toolCalls)).toEqual({
        ticker: 'BTC',
        horizon_days: 7,
        current_price: 73300,
        sentiment_score: 0.42,
      });
    });

    it('builds forecast arbitrator args after Markov and Polymarket evidence for leveraged trade asks', () => {
      const toolCalls = [
        {
          tool: 'get_market_data',
          args: { query: 'Current crypto price snapshot for BTC' },
          result: JSON.stringify({
            data: {
              get_crypto_price_snapshot_BTC: {
                ticker: 'BTC',
                price: 76029.21,
              },
            },
          }),
        },
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 1, trajectory: true, trajectoryDays: 1 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'ok',
              canonical: {
                scenarios: {
                  expectedReturn: 0.006,
                  pUp: 0.55,
                  buckets: [
                    { label: 'Down >5%', probability: 0.021 },
                    { label: 'Flat +/-3%', probability: 0.828 },
                    { label: 'Up >5%', probability: 0.06 },
                  ],
                },
                actionSignal: { expectedReturn: 0.006, confidence: 'MEDIUM' },
                diagnostics: {
                  markovWeight: 0.68,
                  predictionConfidence: 0.274,
                  structuralBreakDetected: true,
                },
              },
              distribution: [
                { price: 72095, probability: 0.95 },
                { price: 78116, probability: 0.05 },
              ],
            },
          }),
        },
        {
          tool: 'polymarket_forecast',
          args: { ticker: 'BTC', horizon_days: 1, current_price: 76029.21, markov_return: 0.00408 },
          result: JSON.stringify({
            data: {
              forecastReturn: -0.0121,
              result: 'Polymarket Forecast: BTC | Horizon: 1 days | Grade: A (83/100)\nWill Bitcoin dip to $75,000 in April?: 100% YES',
            },
          }),
        },
        {
          tool: 'get_onchain_crypto',
          args: { ticker: 'BTC', metrics: ['market', 'sentiment'] },
          result: JSON.stringify({ data: { result: 'No whale transactions detected.' } }),
        },
      ];

      const query = 'Give me a Polymarket and markov price forecast for BTC over the next 24 hours with entry and stop for 10x leveraged position direction';
      expect(shouldForceForecastArbitrator(query, toolCalls)).toBe(true);
      expect(buildForcedForecastArbiterArgs(query, toolCalls)).toEqual({
        ticker: 'BTC',
        horizon_days: 1,
        current_price: 76029.21,
        leverage: 10,
        markov: {
          forecast_return: 0.00408,
          p_up: 0.55,
          confidence: 0.274,
          structural_break: true,
          flat_probability: 0.828,
          ci_low: 72095,
          ci_high: 78116,
          summary: 'Markov action signal confidence MEDIUM',
        },
        polymarket: {
          forecast_return: -0.0121,
          quality_score: 83,
          markets: [
            { question: 'Will Bitcoin dip to $75,000 in April?', probability: 1 },
          ],
          summary: 'Polymarket Forecast: BTC | Horizon: 1 days | Grade: A (83/100)\nWill Bitcoin dip to $75,000 in April?: 100% YES',
        },
        whale: {
          direction: 'neutral',
          confidence: 0.35,
          summary: 'On-chain/whale tool completed; treat as neutral unless the final synthesis has a stronger confirmed whale signal.',
        },
      });
    });

    it('still forces forecast arbitrator for leveraged trade asks when Markov abstains', () => {
      const toolCalls = [
        {
          tool: 'get_market_data',
          args: { query: 'Current crypto price snapshot for BTC' },
          result: JSON.stringify({
            data: {
              get_crypto_price_snapshot_BTC: { ticker: 'BTC', price: 75504.42 },
            },
          }),
        },
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 1, trajectory: true, trajectoryDays: 1 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              abstainReasons: ['prediction confidence below selective threshold'],
              forecastHint: {
                markovReturn: 0.003,
                confidenceScore: 0.18,
              },
            },
          }),
        },
        {
          tool: 'polymarket_forecast',
          args: { ticker: 'BTC', horizon_days: 1, current_price: 75504.42 },
          result: JSON.stringify({
            data: {
              forecastReturn: -0.0121,
              result: 'Polymarket Forecast: BTC | Horizon: 1 days | Grade: A (83/100)\nWill Bitcoin dip to $75,000 in April?: 100% YES',
            },
          }),
        },
      ];

      const query = 'Give me a BTC forecast over the next 24 hours with entry and stop for a 10x leveraged position direction';
      const args = buildForcedForecastArbiterArgs(query, toolCalls);
      expect(shouldForceForecastArbitrator(query, toolCalls)).toBe(true);
      expect(args?.markov?.summary).toContain('Markov abstained');
      expect(args?.polymarket?.forecast_return).toBe(-0.0121);
      expect(args?.leverage).toBe(10);
    });

    it('preserves the stable Markov arbiter fields even when upstream diagnostics include future conformal details', () => {
      const toolCalls = [
        {
          tool: 'get_market_data',
          args: { query: 'Current crypto price snapshot for BTC' },
          result: JSON.stringify({
            data: {
              get_crypto_price_snapshot_BTC: {
                ticker: 'BTC',
                price: 76000,
              },
            },
          }),
        },
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 7, trajectory: true, trajectoryDays: 7 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'ok',
              canonical: {
                scenarios: {
                  expectedReturn: 0.012,
                  pUp: 0.61,
                  buckets: [
                    { label: 'Flat +/-3%', probability: 0.74 },
                  ],
                },
                diagnostics: {
                  predictionConfidence: 0.58,
                  structuralBreakDetected: true,
                  conformal: {
                    applied: true,
                    radius: 0.088,
                    coverageEstimate: 0.61,
                    mode: 'break',
                  },
                },
              },
              distribution: [
                { price: 71500, probability: 0.95 },
                { price: 79000, probability: 0.05 },
              ],
            },
          }),
        },
        {
          tool: 'polymarket_forecast',
          args: { ticker: 'BTC', horizon_days: 7, current_price: 76000, markov_return: 0.012 },
          result: JSON.stringify({
            data: {
              forecastReturn: -0.009,
              result: 'Polymarket Forecast: BTC | Horizon: 7 days | Grade: A- (81/100)\nWill BTC finish below $75,000 on May 7?: 59% YES',
            },
          }),
        },
      ];

      expect(buildForcedForecastArbiterArgs(
        'Give me a Polymarket and markov price forecast for BTC over the next 7 days with position direction',
        toolCalls,
      )).toMatchObject({
        ticker: 'BTC',
        horizon_days: 7,
        current_price: 76000,
        markov: {
          forecast_return: 0.012,
          p_up: 0.61,
          confidence: 0.58,
          structural_break: true,
          flat_probability: 0.74,
          ci_low: 71500,
          ci_high: 79000,
        },
      });
    });

    it('builds mixed-evidence forecast arbitrator args without collapsing Markov and Polymarket inputs', () => {
      const toolCalls = [
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 7, trajectory: true, trajectoryDays: 7 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'ok',
              canonical: {
                scenarios: {
                  expectedReturn: 0.032,
                },
                actionSignal: {
                  recommendation: 'BUY',
                  expectedReturn: 0.032,
                },
                diagnostics: {
                  predictionConfidence: 0.62,
                  markovWeight: 1,
                },
              },
            },
          }),
        },
        {
          tool: 'polymarket_forecast',
          args: { ticker: 'BTC', horizon_days: 7 },
          result: JSON.stringify({
            data: {
              forecastReturn: -0.004,
              result: 'Polymarket Forecast: BTC | Horizon: 7 days | Grade: B\nWill BTC finish below $75,000 on May 7?: 58% YES',
            },
          }),
        },
      ];

      const prefix = buildForecastDisagreementPrefix(
        'Give me a Polymarket and markov price forecast for BTC over the next 7 days',
        toolCalls,
      );

      expect(shouldForceForecastArbitrator(
        'Give me a Polymarket and markov price forecast for BTC over the next 7 days',
        toolCalls,
      )).toBe(true);
      expect(buildForcedForecastArbiterArgs(
        'Give me a Polymarket and markov price forecast for BTC over the next 7 days',
        toolCalls,
      )).toMatchObject({
        ticker: 'BTC',
        horizon_days: 7,
        markov: {
          forecast_return: 0.032,
          confidence: 0.62,
        },
        polymarket: {
          forecast_return: -0.004,
          markets: [
            {
              question: 'Will BTC finish below $75,000 on May 7?',
              probability: 0.58,
            },
          ],
        },
        whale: {
          direction: 'neutral',
          confidence: 0.35,
          summary: 'No confirmed whale/on-chain signal available.',
        },
      });
      expect(prefix).toContain('signals are mixed');
    });

    it('builds stable non-crypto forced args from resolved asset identity and explicit market-data query', () => {
      const query = 'Provide a GOLD forecast for next month';
      const toolCalls = [
        {
          tool: 'get_market_data',
          args: { query: 'GLD current price' },
          result: JSON.stringify({
            data: {
              get_stock_price_GLD: {
                ticker: 'GLD',
                price: 312.45,
              },
            },
          }),
        },
        {
          tool: 'markov_distribution',
          args: { ticker: 'GLD', horizon: 21 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              forecastHint: {
                usage: 'forecast_only',
                markovReturn: 0.012,
              },
            },
          }),
        },
      ];

      expect(buildForcedNonCryptoMarketDataArgs(query)).toEqual({
        query: 'GLD current price',
      });

      expect(buildForcedNonCryptoPolymarketForecastArgs(query, toolCalls)).toEqual({
        ticker: 'GLD',
        horizon_days: 21,
        current_price: 312.45,
      });
    });

    it('accepts Robinhood-style lastTradePrice strings for non-crypto fallback price enrichment', () => {
      const query = 'Provide a GOLD forecast for next month';
      const toolCalls = [
        {
          tool: 'get_market_data',
          args: { query: 'GLD current price' },
          result: JSON.stringify({
            data: {
              get_stock_price_GLD: {
                symbol: 'GLD',
                lastTradePrice: '295.00',
              },
            },
          }),
        },
        {
          tool: 'markov_distribution',
          args: { ticker: 'GLD', horizon: 21 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              forecastHint: {
                usage: 'forecast_only',
                markovReturn: 0.012,
              },
            },
          }),
        },
      ];

      expect(buildForcedNonCryptoPolymarketForecastArgs(query, toolCalls)).toEqual({
        ticker: 'GLD',
        horizon_days: 21,
        current_price: 295,
      });
    });

    it('falls back to abstaining Markov currentPrice after an explicit current-price query returned no usable price', () => {
      const query = 'Provide a GOLD forecast for next month';
      const toolCalls = [
        {
          tool: 'get_market_data',
          args: { query: 'GLD current price' },
          result: JSON.stringify({ data: {} }),
        },
        {
          tool: 'markov_distribution',
          args: { ticker: 'GLD', horizon: 21 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              canonical: {
                currentPrice: 294.87,
              },
              forecastHint: {
                usage: 'forecast_only',
                markovReturn: 0.012,
              },
            },
          }),
        },
      ];

      expect(buildForcedNonCryptoPolymarketForecastArgs(query, toolCalls)).toEqual({
        ticker: 'GLD',
        horizon_days: 21,
        current_price: 294.87,
      });
    });

    it('ignores Markov return when the structured Markov payload abstains or is missing', () => {
      expect(extractMarkovReturnFromToolCalls([])).toBeNull();

      expect(extractMarkovReturnFromToolCalls([
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 7 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              forecastHint: {
                usage: 'forecast_only',
                markovReturn: 0.006,
                confidenceScore: 0.18,
                calibratedDistribution: false,
              },
              canonical: {
                actionSignal: null,
                diagnostics: {
                  markovWeight: 0.4,
                },
              },
            },
          }),
        },
      ])).toBeNull();
    });

    it('reruns polymarket_forecast only when a Markov signal exists but prior forecast args were not Markov-enriched', () => {
      const toolCalls = [
        { tool: 'get_market_data', args: { query: 'Current crypto price snapshot for BTC' }, result: '{}' },
        { tool: 'social_sentiment', args: { ticker: 'BTC', include_fear_greed: true, limit: 25 }, result: '{}' },
        { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 7, current_price: 73300, sentiment_score: 0.42 }, result: '{}' },
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 7, trajectory: true, trajectoryDays: 7 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'ok',
              canonical: {
                actionSignal: { expectedReturn: 0.04 },
                diagnostics: { markovWeight: 0.6 },
              },
            },
          }),
        },
      ];

      expect(shouldRerunPolymarketForecastWithMarkov('Provide a BTC forecast for the next 7 days', toolCalls)).toBe(true);

      expect(shouldRerunPolymarketForecastWithMarkov('Provide a BTC forecast for the next 7 days', [
        toolCalls[0]!,
        toolCalls[1]!,
        { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 7, current_price: 73300, sentiment_score: 0.42, markov_return: 0.024 }, result: '{}' },
        toolCalls[3]!,
      ])).toBe(false);

      expect(shouldRerunPolymarketForecastWithMarkov('Provide a BTC forecast for the next 7 days', [
        toolCalls[0]!,
        toolCalls[1]!,
        { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 7, current_price: 73300, sentiment_score: 0.42 }, result: '{}' },
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 7, trajectory: true, trajectoryDays: 7 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              forecastHint: {
                usage: 'forecast_only',
                markovReturn: 0.006,
                confidenceScore: 0.18,
                calibratedDistribution: false,
              },
              canonical: {
                actionSignal: null,
                diagnostics: { markovWeight: 0.4 },
              },
            },
          }),
        },
      ])).toBe(false);

      expect(shouldRerunPolymarketForecastWithMarkov('Provide a BTC forecast for the next 7 days', [
        toolCalls[0]!,
        toolCalls[1]!,
        { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 7, current_price: 73300, sentiment_score: 0.42 }, result: '{}' },
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 7, trajectory: true, trajectoryDays: 7 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'ok',
              canonical: {
                actionSignal: { expectedReturn: 0.04 },
                diagnostics: {
                  markovWeight: 0.6,
                  predictionConfidence: 0.18,
                },
              },
            },
          }),
        },
      ])).toBe(false);
    });

    it('forces crypto forecast tools only when required enrichment is missing', () => {
      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for the next 7 days',
        [],
      )).toBe(true);

      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for the next 7 days',
        [
          { tool: 'get_market_data', args: { query: 'Current crypto price snapshot for BTC' }, result: '{}' },
          { tool: 'social_sentiment', args: { ticker: 'BTC', include_fear_greed: true, limit: 25 }, result: '{}' },
          { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 7, current_price: 73300, sentiment_score: 0.42 }, result: '{}' },
          { tool: 'get_onchain_crypto', args: { ticker: 'BTC', metrics: ['market', 'sentiment'] }, result: '{}' },
          { tool: 'get_fixed_income', args: { series: ['treasury_yields', 'yield_curve'] }, result: '{}' },
        ],
      )).toBe(true);

      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for the next 7 days',
        [
          { tool: 'get_market_data', args: { query: 'Current crypto price snapshot for ETH' }, result: JSON.stringify({ data: { get_crypto_price_snapshot_ETH: { price: 3900 } } }) },
          { tool: 'social_sentiment', args: { ticker: 'ETH', include_fear_greed: true, limit: 25 }, result: JSON.stringify({ data: { result: '## Overall: 📈 Bullish (score +42/100)' } }) },
          { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 7, current_price: 73300, sentiment_score: 0.42 }, result: JSON.stringify({ data: { forecast: 74250 } }) },
          { tool: 'get_onchain_crypto', args: { ticker: 'ETH', metrics: ['market', 'sentiment'] }, result: JSON.stringify({ data: { ticker: 'ETH', metrics: { activeAddresses: 650000 } } }) },
          { tool: 'get_fixed_income', args: { series: ['treasury_yields', 'yield_curve'] }, result: JSON.stringify({ data: { treasury_yields: [] } }) },
        ],
      )).toBe(true);

      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for the next 7 days',
        [
          { tool: 'get_market_data', args: { query: 'Current crypto price snapshot for BTC' }, result: JSON.stringify({ data: { get_crypto_price_snapshot_BTC: { price: 73300 } } }) },
          { tool: 'social_sentiment', args: { ticker: 'BTC', include_fear_greed: true, limit: 25 }, result: JSON.stringify({ data: { result: '## Overall: 📈 Bullish (score +42/100)' } }) },
          { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 7, current_price: 73300, sentiment_score: 0.42 }, result: JSON.stringify({ data: { forecast: 74250 } }) },
          { tool: 'get_onchain_crypto', args: { ticker: 'BTC', metrics: ['market', 'sentiment'] }, result: JSON.stringify({ data: { ticker: 'BTC', metrics: { activeAddresses: 1200000 } } }) },
          { tool: 'get_fixed_income', args: { series: ['treasury_yields', 'yield_curve'] }, result: JSON.stringify({ data: { treasury_yields: [] } }) },
          { tool: 'markov_distribution', args: { ticker: 'BTC-USD', horizon: 7, trajectory: true, trajectoryDays: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'ok', canonical: { actionSignal: {}, diagnostics: {} } } }) },
        ],
      )).toBe(false);

      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for the next 30 days',
        [
          { tool: 'get_market_data', args: { query: 'Current crypto price snapshot for BTC' }, result: JSON.stringify({ data: { get_crypto_price_snapshot_BTC: { price: 73300 } } }) },
          { tool: 'social_sentiment', args: { ticker: 'BTC', include_fear_greed: true, limit: 25 }, result: JSON.stringify({ data: { result: '## Overall: 📈 Bullish (score +42/100)' } }) },
          { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 30, current_price: 73300, sentiment_score: 0.42 }, result: JSON.stringify({ data: { forecast: 78100 } }) },
          { tool: 'get_onchain_crypto', args: { ticker: 'BTC', metrics: ['market', 'sentiment'] }, result: JSON.stringify({ data: { ticker: 'BTC', metrics: { activeAddresses: 1200000 } } }) },
          { tool: 'get_fixed_income', args: { series: ['treasury_yields', 'yield_curve'] }, result: JSON.stringify({ data: { treasury_yields: [] } }) },
        ],
      )).toBe(false);

      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for the next 7 days',
        [
          { tool: 'get_market_data', args: { query: 'Current crypto price snapshot for BTC' }, result: JSON.stringify({ data: { get_crypto_price_snapshot_BTC: { price: 73300 } } }) },
          { tool: 'social_sentiment', args: { ticker: 'BTC', include_fear_greed: true, limit: 25 }, result: JSON.stringify({ data: { result: '## Overall: 📈 Bullish (score +42/100)' } }) },
          { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 30, current_price: 73300, sentiment_score: 0.42 }, result: JSON.stringify({ data: { forecast: 75250 } }) },
          { tool: 'get_onchain_crypto', args: { ticker: 'BTC', metrics: ['market', 'sentiment'] }, result: JSON.stringify({ data: { ticker: 'BTC', metrics: { activeAddresses: 1200000 } } }) },
          { tool: 'get_fixed_income', args: { series: ['treasury_yields', 'yield_curve'] }, result: JSON.stringify({ data: { treasury_yields: [] } }) },
        ],
      )).toBe(true);

      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for the next 7 days',
        [
          { tool: 'get_market_data', args: { query: 'Current crypto price snapshot for BTC' }, result: 'Error: rate limit' },
          { tool: 'social_sentiment', args: { ticker: 'BTC', include_fear_greed: true, limit: 25 }, result: JSON.stringify({ data: { result: '## Overall: 📈 Bullish (score +42/100)' } }) },
          { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 7, current_price: 73300, sentiment_score: 0.42 }, result: JSON.stringify({ data: { forecast: 74250 } }) },
          { tool: 'get_onchain_crypto', args: { ticker: 'BTC', metrics: ['market', 'sentiment'] }, result: JSON.stringify({ data: { ticker: 'BTC', metrics: { activeAddresses: 1200000 } } }) },
          { tool: 'get_fixed_income', args: { series: ['treasury_yields', 'yield_curve'] }, result: JSON.stringify({ data: { treasury_yields: [] } }) },
        ],
      )).toBe(true);

      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for the next 7 days',
        [
          { tool: 'get_market_data', args: { query: 'Current crypto price snapshot for BTC' }, result: JSON.stringify({ data: { get_crypto_price_snapshot_BTC: { price: 73300 } } }) },
          { tool: 'social_sentiment', args: { ticker: 'BTC', include_fear_greed: true, limit: 25 }, result: JSON.stringify({ data: { result: '## Overall: 📈 Bullish (score +42/100)' } }) },
          { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 7, current_price: 73300, sentiment_score: 0.35 }, result: JSON.stringify({ data: { forecast: 74250 } }) },
          { tool: 'get_onchain_crypto', args: { ticker: 'BTC', metrics: ['market', 'sentiment'] }, result: JSON.stringify({ data: { ticker: 'BTC', metrics: { activeAddresses: 1200000 } } }) },
          { tool: 'get_fixed_income', args: { series: ['treasury_yields', 'yield_curve'] }, result: JSON.stringify({ data: { treasury_yields: [] } }) },
        ],
      )).toBe(true);

      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for the next 7 days',
        [
          { tool: 'get_market_data', args: { query: 'Current crypto price snapshot for BTC' }, result: JSON.stringify({ data: { get_crypto_price_snapshot_BTC: { price: 73300 } } }) },
          { tool: 'social_sentiment', args: { ticker: 'BTC', include_fear_greed: true, limit: 25 }, result: JSON.stringify({ data: { result: '## Overall: 📈 Bullish (score +42/100)' } }) },
          { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 7, current_price: 73300, sentiment_score: 0.42 }, result: JSON.stringify({ data: { forecast: 74250 } }) },
          { tool: 'get_onchain_crypto', args: { ticker: 'BTC', metrics: ['market', 'sentiment'] }, result: 'Error: upstream failed' },
          { tool: 'get_fixed_income', args: { series: ['treasury_yields', 'yield_curve'] }, result: JSON.stringify({ data: { treasury_yields: [] } }) },
        ],
      )).toBe(true);

      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for the next 7 days',
        [
          { tool: 'get_market_data', args: { query: 'Current crypto price snapshot for BTC' }, result: JSON.stringify({ data: { get_crypto_price_snapshot_BTC: { price: 73300 } } }) },
          { tool: 'social_sentiment', args: { ticker: 'BTC', include_fear_greed: true, limit: 25 }, result: JSON.stringify({ data: { result: '## Overall: 📈 Bullish (score +42/100)' } }) },
          { tool: 'polymarket_forecast', args: { ticker: 'BTC', horizon_days: 7, current_price: 73300, sentiment_score: 0.42 }, result: JSON.stringify({ data: { forecast: 74250 } }) },
          { tool: 'get_onchain_crypto', args: { ticker: 'BTC', metrics: ['market', 'sentiment'] }, result: JSON.stringify({ data: { ticker: 'BTC', metrics: { activeAddresses: 1200000 } } }) },
          { tool: 'get_fixed_income', args: { series: ['treasury_yields', 'yield_curve'] }, result: JSON.stringify({ data: { treasury_yields: [] } }) },
          { tool: 'markov_distribution', args: { ticker: 'BTC-USD', horizon: 7, trajectory: true, trajectoryDays: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }) },
        ],
      )).toBe(false);

      expect(shouldForceCryptoForecastTools(
        'What is the probability distribution for BTC-USD in 7 days?',
        [],
      )).toBe(false);

      expect(shouldForceCryptoForecastTools(
        'Improve the BTC short-horizon forecast workflow.',
        [],
      )).toBe(false);
    });

    it('does not force markov or non-crypto forecast fallbacks for workflow-improvement queries', () => {
      expect(shouldForceMarkovDistribution(
        'Improve the BTC markov short-horizon forecast workflow.',
        [],
      )).toBe(false);

      expect(shouldForceNonCryptoForecastFallback(
        'Improve the NVDA short-horizon forecast workflow.',
        [],
      )).toBe(false);
    });

    it('accepts forecast-lab skill as the first planning tool for routed improvement queries', () => {
      const hint = getForecastLabRoutingHint('Improve the BTC short-horizon forecast workflow.');
      const response = {
        content: '',
        tool_calls: [
          { id: 'skill-1', name: 'skill', args: { skill: 'forecast-lab' }, type: 'tool_call' as const },
        ],
        additional_kwargs: {},
      } as any;

      expect(isAcceptedFirstPlanningToolCall(response, hint)).toBe(true);
      expect(isAcceptedFirstPlanningToolCall(response, null)).toBe(false);
    });

    it('detects explicit skill requests by exact runtime skill name', () => {
      expect(detectExplicitSkillRequest('Use the portfolio_risk skill to analyse AAPL risk')).toBe('portfolio_risk');
      expect(detectExplicitSkillRequest('Use the peer-comparison skill for semis')).toBe('peer-comparison');
      expect(detectExplicitSkillRequest('Use the DCF skill to value Apple')).toBeNull();
    });

    it('accepts an explicitly requested skill as the first planning tool', () => {
      const response = {
        content: '',
        tool_calls: [
          { id: 'skill-1', name: 'skill', args: { skill: 'portfolio_risk' }, type: 'tool_call' as const },
        ],
        additional_kwargs: {},
      } as any;

      expect(isAcceptedFirstPlanningToolCall(response, null, 'portfolio_risk')).toBe(true);
      expect(isAcceptedFirstPlanningToolCall(response, null, 'peer-comparison')).toBe(false);
    });

    it('preserves BTC short-horizon abstention once markov_distribution has abstained', () => {
      const toolCalls = [
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 14, trajectory: true, trajectoryDays: 14 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              abstainReasons: ['No trusted terminal prediction-market anchors are available for this horizon.'],
              canonical: {
                ticker: 'BTC-USD',
                horizon: 14,
                diagnostics: {
                  trustedAnchors: 0,
                  totalAnchors: 5,
                  anchorQuality: 'none',
                },
              },
              forecastHint: {
                usage: 'forecast_only',
                markovReturn: 0.0103,
              },
            },
          }),
        },
      ];

      expect(shouldPreserveAbstainingBtcShortHorizonForecast(
        'Provide a BTC forecast for the next 14 days',
        toolCalls,
      )).toBe(true);
      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for the next 14 days',
        toolCalls,
      )).toBe(true);
    });

    it('preserves BTC next-week abstention once markov_distribution has abstained', () => {
      const toolCalls = [
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 5, trajectory: true, trajectoryDays: 5 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              abstainReasons: ['No trusted terminal prediction-market anchors are available for this horizon.'],
              canonical: {
                ticker: 'BTC-USD',
                horizon: 5,
                diagnostics: {
                  trustedAnchors: 0,
                  totalAnchors: 3,
                  anchorQuality: 'none',
                },
              },
              forecastHint: {
                usage: 'forecast_only',
                markovReturn: 0.009,
              },
            },
          }),
        },
      ];

      expect(shouldPreserveAbstainingBtcShortHorizonForecast(
        'Provide a BTC forecast for next week',
        toolCalls,
      )).toBe(true);
      expect(shouldForceCryptoForecastTools(
        'Provide a BTC forecast for next week',
        toolCalls,
      )).toBe(true);
    });

    it('does not preserve BTC short-horizon selective-gate failures as abstentions', () => {
      const toolCalls = [
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 14, trajectory: true, trajectoryDays: 14 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'ok',
              canonical: {
                ticker: 'BTC-USD',
                horizon: 14,
                actionSignal: { expectedReturn: 0.03 },
                diagnostics: {
                  trustedAnchors: 2,
                  totalAnchors: 5,
                  anchorQuality: 'good',
                  markovWeight: 0.6,
                  predictionConfidence: 0.19,
                },
              },
            },
          }),
        },
      ];

      expect(shouldPreserveAbstainingBtcShortHorizonForecast(
        'Provide a BTC forecast for the next 14 days',
        toolCalls,
      )).toBe(false);
    });

    it('does not treat a non-5-day BTC abstain as matching a next-week query', () => {
      const toolCalls = [
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 14, trajectory: true, trajectoryDays: 14 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              canonical: {
                ticker: 'BTC-USD',
                horizon: 14,
                diagnostics: {
                  trustedAnchors: 0,
                  totalAnchors: 5,
                  anchorQuality: 'none',
                },
              },
            },
          }),
        },
      ];

      expect(shouldPreserveAbstainingBtcShortHorizonForecast(
        'Provide a BTC forecast for next week',
        toolCalls,
      )).toBe(false);
    });

    it('does not preserve BTC short-horizon abstention when polymarket_forecast was explicitly requested', () => {
      const toolCalls = [
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 2, trajectory: true, trajectoryDays: 2 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              canonical: {
                ticker: 'BTC-USD',
                horizon: 2,
                diagnostics: {
                  trustedAnchors: 0,
                  totalAnchors: 4,
                  anchorQuality: 'none',
                },
              },
            },
          }),
        },
      ];

      expect(shouldPreserveAbstainingBtcShortHorizonForecast(
        'Use polymarket_forecast for BTC over the next 2 days',
        toolCalls,
      )).toBe(false);
    });

    it('does not preserve BTC short-horizon abstention for explicit combined Polymarket + Markov forecast prompts', () => {
      const toolCalls = [
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 14, trajectory: true, trajectoryDays: 14 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              canonical: {
                ticker: 'BTC-USD',
                horizon: 14,
                diagnostics: {
                  trustedAnchors: 0,
                  totalAnchors: 5,
                  anchorQuality: 'none',
                },
              },
            },
          }),
        },
      ];

      expect(shouldPreserveAbstainingBtcShortHorizonForecast(
        'Give me a Polymarket and markov price forecast for BTC over the next 14 days',
        toolCalls,
      )).toBe(false);
    });

    it('does not preserve BTC short-horizon abstention once arbitrator evidence exists', () => {
      const toolCalls = [
        {
          tool: 'markov_distribution',
          args: { ticker: 'BTC-USD', horizon: 14, trajectory: true, trajectoryDays: 14 },
          result: JSON.stringify({
            data: {
              _tool: 'markov_distribution',
              status: 'abstain',
              canonical: {
                ticker: 'BTC-USD',
                horizon: 14,
                diagnostics: {
                  trustedAnchors: 0,
                  totalAnchors: 5,
                  anchorQuality: 'none',
                },
              },
            },
          }),
        },
        {
          tool: 'forecast_arbitrator',
          args: { ticker: 'BTC', horizon_days: 14, current_price: 76135, leverage: 10 },
          result: JSON.stringify({
            data: {
              verdict: 'NO_TRADE',
              confidence: 'low',
            },
          }),
        },
      ];

      expect(shouldPreserveAbstainingBtcShortHorizonForecast(
        'Give me a Polymarket and markov price forecast for BTC over the next 14 days and provide the position direction',
        toolCalls,
      )).toBe(false);
    });

    describe('isNonCryptoForecastQuery', () => {
      it('matches stock forecast queries', () => {
        expect(isNonCryptoForecastQuery('Provide an NVDA forecast for the next 7 days')).toBe(true);
        expect(isNonCryptoForecastQuery('What will AAPL trade at next week?')).toBe(true);
        expect(isNonCryptoForecastQuery('Where is SPY headed?')).toBe(true);
        expect(isNonCryptoForecastQuery('SPY price target next month')).toBe(true);
      });

      it('matches commodity forecast queries (gold, silver, oil)', () => {
        expect(isNonCryptoForecastQuery('Gold forecast for the next 30 days')).toBe(true);
        expect(isNonCryptoForecastQuery('Will silver hit $30 by end of Q2?')).toBe(true);
        expect(isNonCryptoForecastQuery('Where will oil prices be in 2 weeks')).toBe(true);
      });

      it('matches ETF forecast queries', () => {
        expect(isNonCryptoForecastQuery('QQQ outlook over the next 14 days')).toBe(true);
      });

      it('rejects crypto forecast queries (handled by isCryptoForecastQuery)', () => {
        expect(isNonCryptoForecastQuery('Provide a BTC forecast for the next 7 days')).toBe(false);
        expect(isNonCryptoForecastQuery('What will ETH trade at next week?')).toBe(false);
      });

      it('rejects explicit markov distribution queries', () => {
        expect(isNonCryptoForecastQuery('What is the probability distribution for SPY in 30 days?')).toBe(false);
      });

      it('rejects probability_assessment skill invocations', () => {
        expect(isNonCryptoForecastQuery('Use the probability_assessment skill for NVDA')).toBe(false);
      });

      it('rejects non-forecast non-crypto queries', () => {
        expect(isNonCryptoForecastQuery('What is AAPL revenue?')).toBe(false);
        expect(isNonCryptoForecastQuery('NVDA P/E ratio')).toBe(false);
      });

      it('rejects macro-only forecast queries without a price ticker target', () => {
        expect(isNonCryptoForecastQuery('Fed rate prediction next meeting')).toBe(false);
      });

      it('detects explicit polymarket_forecast requests by name', () => {
        expect(isExplicitPolymarketForecastRequest('Use polymarket_forecast for BTC over the next 2 days')).toBe(true);
        expect(isExplicitPolymarketForecastRequest('Run the polymarket forecast for NVDA next week')).toBe(true);
        expect(isExplicitPolymarketForecastRequest('Give me a Polymarket and markov price forecast for BTC over the next 14 days')).toBe(true);
        expect(isExplicitPolymarketForecastRequest('Provide a BTC forecast for the next 7 days')).toBe(false);
      });
    });

    describe('shouldForceNonCryptoForecastFallback', () => {
      it('returns false before Markov has run at all', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [],
        )).toBe(false);
      });

      it('returns true when get_market_data is missing after Markov abstains', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            { tool: 'markov_distribution', args: { ticker: 'NVDA', horizon: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }) },
            { tool: 'polymarket_forecast', args: { ticker: 'NVDA', horizon_days: 7 }, result: '{"data":{}}' },
          ],
        )).toBe(true);
      });

      it('returns true when polymarket_forecast is missing after Markov abstains', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            { tool: 'markov_distribution', args: { ticker: 'NVDA', horizon: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }) },
            { tool: 'get_market_data', args: { query: 'NVDA current price' }, result: JSON.stringify({ data: { get_stock_price_NVDA: { price: 921.13 } } }) },
          ],
        )).toBe(true);
      });

      it('returns false when both get_market_data and polymarket_forecast already cover the needed enrichment', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            { tool: 'markov_distribution', args: { ticker: 'NVDA', horizon: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }) },
            { tool: 'get_market_data', args: { query: 'NVDA current price' }, result: JSON.stringify({ data: { get_stock_price_NVDA: { price: 921.13 } } }) },
            { tool: 'polymarket_forecast', args: { ticker: 'NVDA', horizon_days: 7, current_price: 921.13 }, result: '{"data":{}}' },
          ],
        )).toBe(false);
      });

      it('returns false when Markov produced a successful result', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            { tool: 'markov_distribution', args: { ticker: 'NVDA', horizon: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'ok', canonical: { actionSignal: {}, diagnostics: {} } } }) },
          ],
        )).toBe(false);
      });

      it('returns true when Markov abstained (status=abstain)', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            { tool: 'markov_distribution', args: { ticker: 'NVDA', horizon: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }) },
          ],
        )).toBe(true);
      });

      it('keeps combined GOLD requests on the non-crypto fallback path after Markov abstains across 1d/2d/3d/14d horizons', () => {
        for (const days of [1, 2, 3, 14]) {
          expect(shouldForceNonCryptoForecastFallback(
            `Provide a GOLD price forecast based on markov chain and polymarket for the next ${days} day${days === 1 ? '' : 's'}`,
            [
              { tool: 'markov_distribution', args: { ticker: 'GLD', horizon: days }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }) },
            ],
          )).toBe(true);
        }
      });

      it('returns true when forecast rerun is needed to add current_price after market data landed later', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            { tool: 'markov_distribution', args: { ticker: 'NVDA', horizon: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }) },
            { tool: 'polymarket_forecast', args: { ticker: 'NVDA', horizon_days: 7 }, result: '{"data":{}}' },
            { tool: 'get_market_data', args: { query: 'NVDA current price' }, result: JSON.stringify({ data: { get_stock_price_NVDA: { price: 921.13 } } }) },
          ],
        )).toBe(true);
      });

      it('returns false when abstain hints are the only source of markov_return enrichment', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            {
              tool: 'markov_distribution',
              args: { ticker: 'NVDA', horizon: 7 },
              result: JSON.stringify({
                data: {
                  _tool: 'markov_distribution',
                  status: 'abstain',
                  forecastHint: {
                    usage: 'forecast_only',
                    markovReturn: 0.009,
                  },
                },
              }),
            },
            { tool: 'get_market_data', args: { query: 'NVDA current price' }, result: JSON.stringify({ data: { get_stock_price_NVDA: { price: 921.13 } } }) },
            { tool: 'polymarket_forecast', args: { ticker: 'NVDA', horizon_days: 7, current_price: 921.13 }, result: '{"data":{}}' },
          ],
        )).toBe(false);
      });

      it('returns false when a prior generic market-data query exists but the explicit current-price query already succeeded', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            { tool: 'markov_distribution', args: { ticker: 'NVDA', horizon: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }) },
            { tool: 'get_market_data', args: { query: 'NVDA' }, result: '{"data":{}}' },
            { tool: 'get_market_data', args: { query: 'NVDA current price' }, result: JSON.stringify({ data: { get_stock_price_NVDA: { price: 921.13 } } }) },
            { tool: 'polymarket_forecast', args: { ticker: 'NVDA', horizon_days: 7, current_price: 921.13 }, result: '{"data":{}}' },
          ],
        )).toBe(false);
      });

      it('returns false when an explicit current-price query ran but still produced no usable price', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            { tool: 'markov_distribution', args: { ticker: 'NVDA', horizon: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }) },
            { tool: 'get_market_data', args: { query: 'NVDA current price' }, result: JSON.stringify({ data: {} }) },
            { tool: 'polymarket_forecast', args: { ticker: 'NVDA', horizon_days: 7 }, result: '{"data":{}}' },
          ],
        )).toBe(false);
      });

      it('returns false when a prior explicit current-price query failed but Markov diagnostics already provide currentPrice coverage', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide a GOLD forecast for next month',
          [
            {
              tool: 'markov_distribution',
              args: { ticker: 'GLD', horizon: 21 },
              result: JSON.stringify({
                data: {
                  _tool: 'markov_distribution',
                  status: 'abstain',
                  canonical: {
                    currentPrice: 294.87,
                  },
                  forecastHint: {
                    usage: 'forecast_only',
                    markovReturn: 0.012,
                  },
                },
              }),
            },
            { tool: 'get_market_data', args: { query: 'GLD current price' }, result: JSON.stringify({ data: {} }) },
            { tool: 'polymarket_forecast', args: { ticker: 'GLD', horizon_days: 21, current_price: 294.87, markov_return: 0.012 }, result: '{"data":{}}' },
          ],
        )).toBe(false);
      });

      it('returns false when Robinhood-style lastTradePrice already satisfies the explicit current-price enrichment', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide a GOLD forecast for next month',
          [
            {
              tool: 'markov_distribution',
              args: { ticker: 'GLD', horizon: 21 },
              result: JSON.stringify({
                data: {
                  _tool: 'markov_distribution',
                  status: 'abstain',
                },
              }),
            },
            {
              tool: 'get_market_data',
              args: { query: 'GLD current price' },
              result: JSON.stringify({
                data: {
                  get_stock_price_GLD: {
                    symbol: 'GLD',
                    lastTradePrice: '295.00',
                  },
                },
              }),
            },
            { tool: 'polymarket_forecast', args: { ticker: 'GLD', horizon_days: 21, current_price: 295 }, result: '{"data":{}}' },
          ],
        )).toBe(false);
      });

      it('returns true when a matching polymarket_forecast call only recorded an error result', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            {
              tool: 'markov_distribution',
              args: { ticker: 'NVDA', horizon: 7 },
              result: JSON.stringify({
                data: {
                  _tool: 'markov_distribution',
                  status: 'abstain',
                  forecastHint: {
                    usage: 'forecast_only',
                    markovReturn: 0.009,
                  },
                },
              }),
            },
            { tool: 'get_market_data', args: { query: 'NVDA current price' }, result: JSON.stringify({ data: { get_stock_price_NVDA: { price: 921.13 } } }) },
            { tool: 'polymarket_forecast', args: { ticker: 'NVDA', horizon_days: 7, current_price: 921.13, markov_return: 0.009 }, result: 'Error: upstream failed' },
          ],
        )).toBe(true);
      });

      it('returns true when a prior forecast used the wrong current_price value', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            {
              tool: 'markov_distribution',
              args: { ticker: 'NVDA', horizon: 7 },
              result: JSON.stringify({
                data: {
                  _tool: 'markov_distribution',
                  status: 'abstain',
                },
              }),
            },
            { tool: 'get_market_data', args: { query: 'NVDA current price' }, result: JSON.stringify({ data: { get_stock_price_NVDA: { price: 921.13 } } }) },
            { tool: 'polymarket_forecast', args: { ticker: 'NVDA', horizon_days: 7, current_price: 900.0 }, result: '{"data":{}}' },
          ],
        )).toBe(true);
      });

      it('returns false when a prior forecast only differs by an abstain-derived markov_return value', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            {
              tool: 'markov_distribution',
              args: { ticker: 'NVDA', horizon: 7 },
              result: JSON.stringify({
                data: {
                  _tool: 'markov_distribution',
                  status: 'abstain',
                  forecastHint: {
                    usage: 'forecast_only',
                    markovReturn: 0.009,
                  },
                },
              }),
            },
            { tool: 'get_market_data', args: { query: 'NVDA current price' }, result: JSON.stringify({ data: { get_stock_price_NVDA: { price: 921.13 } } }) },
            { tool: 'polymarket_forecast', args: { ticker: 'NVDA', horizon_days: 7, current_price: 921.13, markov_return: 0.005 }, result: '{"data":{}}' },
          ],
        )).toBe(false);
      });

      it('does not let an unrelated successful Markov call suppress the target fallback', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide an NVDA forecast for the next 7 days',
          [
            { tool: 'markov_distribution', args: { ticker: 'SPY', horizon: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'ok', canonical: { actionSignal: {}, diagnostics: {} } } }) },
            { tool: 'markov_distribution', args: { ticker: 'NVDA', horizon: 7 }, result: JSON.stringify({ data: { _tool: 'markov_distribution', status: 'abstain' } }) },
          ],
        )).toBe(true);
      });

      it('returns false for crypto forecast queries', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'Provide a BTC forecast for the next 7 days',
          [],
        )).toBe(false);
      });

      it('returns false for non-forecast queries', () => {
        expect(shouldForceNonCryptoForecastFallback(
          'What is AAPL revenue?',
          [],
        )).toBe(false);
      });
    });
  });
});
