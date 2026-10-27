import { describe, test, expect, mock, spyOn, beforeEach, afterAll } from 'bun:test';
import { useNoNetwork } from '../../utils/test-no-network.js';
import * as apiRoutingCache from '../../utils/api-routing-cache.js';
import { getAnalystEstimates } from './estimates.js';
import { getKeyRatios } from './key-ratios.js';
import { getYahooAnalystTargets } from './yahoo-finance.js';
import { AIMessage } from '@langchain/core/messages';

useNoNetwork();

// Capture real module before mocking so afterAll can restore
const realLlm = { ...(await import('../../model/llm.js')) };

// ---------------------------------------------------------------------------
// Mock callLlm BEFORE importing the module under test
// ---------------------------------------------------------------------------

const mockCallLlm = mock(async (_prompt: string, _opts: unknown) => ({
  response: new AIMessage({ content: '', tool_calls: [] }),
  usage: undefined,
}));

mock.module('../../model/llm.js', () => ({
  ...realLlm,
  callLlm: mockCallLlm,
}));

// The routing cache lives in .cramer-short/api-routing.json: don't read a local copy or write one.
const routingSpies = [
  spyOn(apiRoutingCache, 'getPreferredApi').mockResolvedValue(null),
  spyOn(apiRoutingCache, 'setPreferredApi').mockResolvedValue(undefined),
];

// These tests only check routing: keep the routed sub-tools off financialdatasets.ai and Yahoo.
const subToolSpies = [getAnalystEstimates, getKeyRatios, getYahooAnalystTargets].map((subTool) =>
  spyOn(subTool, 'invoke').mockResolvedValue(JSON.stringify({ data: {}, sourceUrls: [] })),
);

afterAll(() => {
  mock.module('../../model/llm.js', () => realLlm);
  for (const spy of [...routingSpies, ...subToolSpies]) spy.mockRestore();
});

// ---------------------------------------------------------------------------
// Import after mocking
// ---------------------------------------------------------------------------

const { createGetFinancials } = await import('./get-financials.js');

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeToolCallResponse(toolName: string, args: Record<string, unknown>): AIMessage {
  return new AIMessage({
    content: '',
    tool_calls: [{ name: toolName, args, id: 'call-1', type: 'tool_call' }],
  });
}

function emptyResponse(): AIMessage {
  return new AIMessage({ content: '', tool_calls: [] });
}

const withUsage = (response: AIMessage) => ({ response, usage: undefined as undefined });

function parseResult(raw: unknown): { data: unknown } {
  return JSON.parse(raw as string);
}

// ---------------------------------------------------------------------------
// Router retry behaviour
// ---------------------------------------------------------------------------

describe('get-financials router — retry on zero tool calls', () => {
  const tool = createGetFinancials('test-model');

  beforeEach(() => mockCallLlm.mockReset());

  test('retries when first callLlm returns zero tool calls', async () => {
    // First call: no tool calls → retry
    mockCallLlm.mockResolvedValueOnce({ response: emptyResponse(), usage: undefined });
    // Second call (retry): also no tool calls → keyword fallback
    mockCallLlm.mockResolvedValueOnce({ response: emptyResponse(), usage: undefined });

    await tool.invoke({ query: 'analyst estimates for AAPL' });

    expect(mockCallLlm).toHaveBeenCalledTimes(2);
  });

  test('does NOT retry when first callLlm returns valid tool calls', async () => {
    mockCallLlm.mockResolvedValueOnce({
      response: makeToolCallResponse('get_analyst_estimates', { ticker: 'AAPL', period: 'annual' }),
      usage: undefined,
    });

    await tool.invoke({ query: 'analyst estimates for AAPL' });

    expect(mockCallLlm).toHaveBeenCalledTimes(1);
  });

  test('uses valid tool calls from retry when first attempt is empty', async () => {
    mockCallLlm
      .mockResolvedValueOnce({ response: emptyResponse(), usage: undefined })
      .mockResolvedValueOnce({
        response: makeToolCallResponse('get_analyst_estimates', { ticker: 'AAPL', period: 'annual' }),
        usage: undefined,
      });

    const raw = await tool.invoke({ query: 'analyst estimates for AAPL' });
    const result = parseResult(raw);

    // Should not be a routing error
    expect((result.data as Record<string, unknown>).error).not.toBe('No tools selected for query');
  });
});

// ---------------------------------------------------------------------------
// Keyword fallback behaviour
// ---------------------------------------------------------------------------

describe('get-financials router — keyword fallback after two empty responses', () => {
  const tool = createGetFinancials('test-model');

  beforeEach(() => {
    mockCallLlm.mockReset();
    // Both LLM calls return empty (triggers keyword fallback)
    mockCallLlm.mockResolvedValue({ response: emptyResponse(), usage: undefined });
  });

  test('routes "analyst price targets for VWS.CO" to get_yahoo_analyst_targets', async () => {
    const raw = await tool.invoke({ query: 'analyst price targets for VWS.CO' });
    const result = parseResult(raw);

    // Should not be a "No tools selected" error — keyword fallback should have fired
    expect((result.data as Record<string, unknown>).error).not.toBe('No tools selected for query');
  });

  test('routes "analyst estimates for AAPL" to get_analyst_estimates', async () => {
    const raw = await tool.invoke({ query: 'analyst estimates for AAPL' });
    const result = parseResult(raw);

    expect((result.data as Record<string, unknown>).error).not.toBe('No tools selected for query');
  });

  test.each(['AAPL valuation metrics', 'AAPL P/E ratio', 'tell me about AAPL'])(
    'routes "%s" to the registered get_key_ratios tool',
    async (query) => {
      const raw = await tool.invoke({ query });
      const data = parseResult(raw).data as Record<string, unknown>;

      expect(data._errors).toBeUndefined();
      expect(data).toHaveProperty('get_key_ratios_AAPL');
    },
  );

  test('returns "No tools selected" error only when no ticker can be extracted', async () => {
    const raw = await tool.invoke({ query: 'what is the meaning of life' });
    const result = parseResult(raw);

    expect((result.data as Record<string, unknown>).error).toBe('No tools selected for query');
  });
});

// ---------------------------------------------------------------------------
// Yahoo Finance tools registered in router
// ---------------------------------------------------------------------------

describe('get-financials — Yahoo Finance tools in FINANCE_TOOLS', () => {
  const tool = createGetFinancials('test-model');

  test('router prompt mentions Yahoo Finance tools for international tickers', async () => {
    // Check that at least one callLlm invocation carries the Yahoo tool in its tools list
    mockCallLlm.mockResolvedValueOnce({
      response: makeToolCallResponse('get_yahoo_analyst_targets', { ticker: 'VWS.CO' }),
      usage: undefined,
    });

    await tool.invoke({ query: 'price targets for VWS.CO' });

    const callOpts = mockCallLlm.mock.calls[0]?.[1] as { tools?: Array<{ name: string }> };
    const toolNames = callOpts?.tools?.map((t) => t.name) ?? [];
    expect(toolNames).toContain('get_yahoo_analyst_targets');
    expect(toolNames).toContain('get_yahoo_analyst_recommendations');
  });
});

// ---------------------------------------------------------------------------
// Router prompt only names registered tools
// ---------------------------------------------------------------------------

describe('get-financials — router prompt tool names', () => {
  const tool = createGetFinancials('test-model');

  test('every get_* tool the prompt tells the model to call is registered', async () => {
    mockCallLlm.mockReset();
    mockCallLlm.mockResolvedValueOnce({
      response: makeToolCallResponse('get_key_ratios', { ticker: 'AAPL' }),
      usage: undefined,
    });

    await tool.invoke({ query: 'AAPL P/E' });

    const callOpts = mockCallLlm.mock.calls[0]?.[1] as { systemPrompt: string; tools: Array<{ name: string }> };
    const registered = new Set(callOpts.tools.map((t) => t.name));
    const mentioned = [...callOpts.systemPrompt.matchAll(/\bget_[a-z_]+\b/g)].map((m) => m[0]);
    expect(mentioned.length).toBeGreaterThan(0);
    expect(mentioned.filter((name) => !registered.has(name))).toEqual([]);
  });

  test('prompt routes historical ratio queries to get_historical_key_ratios', async () => {
    mockCallLlm.mockReset();
    mockCallLlm.mockResolvedValueOnce({
      response: makeToolCallResponse('get_key_ratios', { ticker: 'AAPL' }),
      usage: undefined,
    });

    await tool.invoke({ query: 'AAPL P/E' });

    const { systemPrompt } = mockCallLlm.mock.calls[0]?.[1] as { systemPrompt: string };
    const historicalLine = systemPrompt.split('\n').find((l) => l.includes('historical P/E ratio'));
    expect(historicalLine).toContain('get_historical_key_ratios');
  });
});

// ---------------------------------------------------------------------------
// Routing cache records the provider that actually served the call
// ---------------------------------------------------------------------------

describe('get-financials — routing cache provider', () => {
  const tool = createGetFinancials('test-model');
  const setPreferredApiSpy = routingSpies[1];

  beforeEach(() => {
    mockCallLlm.mockReset();
    setPreferredApiSpy.mockClear();
  });

  test('records financial-datasets (not fmp) when a Financial Datasets tool succeeds', async () => {
    mockCallLlm.mockResolvedValueOnce({
      response: makeToolCallResponse('get_key_ratios', { ticker: 'AAPL' }),
      usage: undefined,
    });

    await tool.invoke({ query: 'AAPL P/E' });

    expect(setPreferredApiSpy).toHaveBeenCalledWith('AAPL', 'financial-datasets');
  });

  test('records yahoo when a Yahoo tool succeeds', async () => {
    mockCallLlm.mockResolvedValueOnce({
      response: makeToolCallResponse('get_yahoo_analyst_targets', { ticker: 'VWS.CO' }),
      usage: undefined,
    });

    await tool.invoke({ query: 'price targets for VWS.CO' });

    expect(setPreferredApiSpy).toHaveBeenCalledWith('VWS.CO', 'yahoo');
  });
});
