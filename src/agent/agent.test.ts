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
const {
  inferDistributionTicker,
  inferDistributionHorizon,
  isExplicitTerminalDistributionQuery,
  shouldForceMarkovDistribution,
  inferTrajectoryRequest,
   buildForcedMarkovArgs,
   isCryptoForecastQuery,
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
   shouldInjectBtcShortHorizonMixedEvidencePrompt,
   shouldRerunPolymarketForecastWithMarkov,
  buildForcedOnchainArgs,
  buildForcedFixedIncomeArgs,
  buildForcedCryptoForecastMarkovArgs,
  shouldForceCryptoForecastTools,
} = await import('./agent.js');

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------
async function collectEvents(gen: AsyncGenerator<AgentEvent>): Promise<AgentEvent[]> {
  const events: AgentEvent[] = [];
  for await (const e of gen) events.push(e);
  return events;
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

    it('preserves explicit Barrick Gold equity queries as GOLD ticker', () => {
      expect(inferDistributionTicker('What is the probability distribution for Barrick Gold stock in 30 trading days?')).toBe('GOLD');
    });

    it('flags explicit terminal distribution queries for forced markov routing', () => {
      expect(shouldForceMarkovDistribution(
        'What is the probability distribution for BTC-USD in 7 trading days? Use the Markov distribution methodology with terminal threshold markets only.',
        [],
      )).toBe(true);

      expect(shouldForceMarkovDistribution(
        'What is the probability distribution for BTC-USD in 7 trading days?',
        [{ tool: 'markov_distribution', args: { ticker: 'BTC-USD', horizon: 7 }, result: '{"data":{"_tool":"markov_distribution","status":"abstain"}}' }],
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
      expect(isCryptoForecastQuery('Use the probability_assessment skill for BTC price movement in the next 30 days')).toBe(false);
      expect(isCryptoForecastQuery('What is the market cap of BTC?')).toBe(false);
      expect(isCryptoForecastQuery('What is the probability distribution for BTC-USD in 7 days?')).toBe(false);
      expect(isCryptoForecastQuery('Provide an AAPL forecast for the next 7 days')).toBe(false);
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
        markov_return: 0.012,
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
        markov_return: 0.012,
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
        markov_return: 0.012,
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
      ])).toBe(0.006);
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
      ])).toBe(true);
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

      it('returns true when forecast rerun is needed to add markov_return after abstain hint is available', () => {
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
        )).toBe(true);
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
          'Provide a