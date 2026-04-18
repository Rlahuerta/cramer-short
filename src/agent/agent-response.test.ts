/**
 * Regression tests for agent response delivery.
 *
 * Guards three specific fixes made to handleDirectResponse():
 *
 * 1. No redundant streamCallLlm call when callLlm already returned an answer.
 *    The original bug caused multi-minute hangs: streamCallLlm has no timeout
 *    and was called even when the model's text answer was already available.
 *
 * 2. Synthesis stream (streamCallLlm) IS called at max iterations when there
 *    is no pre-existing answer — the one legitimate case for the extra call.
 *
 * 3. Synthesis timeout is graceful: if streamCallLlm throws (e.g. AbortError
 *    from the hard timeout), the agent emits a done event rather than hanging
 *    or producing a blank response.
 *
 * 4. Thinking text is truncated to 500 chars so verbose models (e.g. Qwen)
 *    that embed raw JSON in their reasoning text don't flood the terminal.
 */
import { describe, it, expect, mock, beforeEach, afterEach } from 'bun:test';
import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import type { AgentEvent, DoneEvent, AnswerChunkEvent } from './types.js';
import { Scratchpad } from './scratchpad.js';

// ---------------------------------------------------------------------------
// Mutable mock state — reset before each test.
// ---------------------------------------------------------------------------
const mockState = {
  /** Number of times streamCallLlm was called. Key regression metric. */
  streamCallCount: 0,
  /** When true, callLlm returns tool_calls instead of a direct text answer. */
  invokeReturnsToolCalls: false,
  /** Text content returned alongside tool_calls (simulates model reasoning). */
  invokeThinkingText: '',
  /** Direct answer returned when invokeReturnsToolCalls is false. */
  invokeContent: 'The direct answer from callLlm',
  /** Chunks yielded by streamCallLlm (synthesis path). */
  streamChunks: ['synthesis result'] as string[],
  /** When true, streamCallLlm throws immediately (simulates timeout). */
  streamShouldThrow: false,
  /**
   * Per-call response queue. When non-empty, callLlm shifts from the front
   * instead of using the flag-based logic above. Enables multi-iteration tests
   * (e.g. first call returns tool_calls, second returns text) without requiring
   * a second mock.module() call — which would permanently contaminate the
   * module registry for subsequent test files in the same Bun worker.
   */
  callLlmQueue: [] as Array<{ content: string; toolCalls: typeof ST_TOOL_CALL[] }>,
};

const ST_TOOL_CALL = {
  id: 'st1',
  name: 'sequential_thinking',
  args: {
    thought: 'analyzing...',
    nextThoughtNeeded: false,
    thoughtNumber: 1,
    totalThoughts: 1,
  },
  type: 'tool_call' as const,
};

// ---------------------------------------------------------------------------
// Mock callLlm + streamCallLlm at the module boundary.
// Mocking at this level (rather than @langchain/openai) avoids the providerMap
// cache-poisoning issue and gives precise control over call tracking.
// ---------------------------------------------------------------------------
mock.module('../model/llm.js', () => ({
  DEFAULT_MODEL: 'gpt-5.4',
  callLlm: async () => {
    // Drain the per-call response queue first (used by multi-iteration tests).
    if (mockState.callLlmQueue.length > 0) {
      const next = mockState.callLlmQueue.shift()!;
      return {
        response: { content: next.content, tool_calls: next.toolCalls, additional_kwargs: {} },
        usage: undefined,
      };
    }
    if (mockState.invokeReturnsToolCalls) {
      return {
        response: {
          content: mockState.invokeThinkingText,
          tool_calls: [ST_TOOL_CALL],
          additional_kwargs: {},
        },
        usage: undefined,
      };
    }
    return {
      response: {
        content: mockState.invokeContent,
        tool_calls: [],
        additional_kwargs: {},
      },
      usage: undefined,
    };
  },
  streamCallLlm: async function* (_prompt: string, opts: { signal?: AbortSignal } = {}) {
    mockState.streamCallCount++;
    if (mockState.streamShouldThrow) {
      throw new Error('LLM stream timed out');
    }
    for (const chunk of mockState.streamChunks) {
      if (opts.signal?.aborted) {
        throw new DOMException('The operation was aborted.', 'AbortError');
      }
      yield chunk;
    }
  },
  resolveProvider: (model: string) => ({ id: 'openai', displayName: model }),
  formatUserFacingError: (msg: string) => msg,
  isContextOverflowError: () => false,
}));

// Mock memory to avoid SQLite initialization.
mock.module('../memory/index.js', () => ({
  MemoryManager: {
    get: async () => ({
      listFiles: async () => [],
      loadSessionContext: async () => ({ text: '' }),
      saveAnswer: async () => {},
    }),
  },
}));

// ---------------------------------------------------------------------------
// Dynamic import AFTER mocks are registered.
// ---------------------------------------------------------------------------
const { Agent, buildAbstainingMarkovAnswer, buildDistributionWarningPrefix, buildForecastDisagreementPrefix } = await import('./agent.js');

// ---------------------------------------------------------------------------
// Test environment helpers
// ---------------------------------------------------------------------------
let tmpDir: string;
let originalCwd: string;
let prevOpenAiKey: string | undefined;

beforeEach(() => {
  mockState.streamCallCount = 0;
  mockState.invokeReturnsToolCalls = false;
  mockState.invokeThinkingText = '';
  mockState.invokeContent = 'The direct answer from callLlm';
  mockState.streamChunks = ['synthesis result'];
  mockState.streamShouldThrow = false;
  mockState.callLlmQueue = [];

  tmpDir = join(tmpdir(), `agent-resp-${Date.now()}-${Math.random().toString(36).slice(2)}`);
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

async function collectEvents(gen: AsyncGenerator<AgentEvent>): Promise<AgentEvent[]> {
  const events: AgentEvent[] = [];
  for await (const e of gen) events.push(e);
  return events;
}

// ---------------------------------------------------------------------------
// 1. No redundant streamCallLlm when callLlm already has the answer
// ---------------------------------------------------------------------------

describe('Agent — no redundant stream call for direct answers', () => {
  beforeEach(() => {
    mockState.invokeReturnsToolCalls = false;
  });

  it('streamCallLlm is never invoked when the model returns a text answer', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3, memoryEnabled: false });
    await collectEvents(agent.run('test query'));
    expect(mockState.streamCallCount).toBe(0);
  });

  it('done.answer matches callLlm content, not the stream chunks', async () => {
    mockState.invokeContent = 'Unique answer from invoke';
    mockState.streamChunks = ['Should not appear in answer'];

    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3, memoryEnabled: false });
    const events = await collectEvents(agent.run('test query'));
    const done = events.find((e) => e.type === 'done') as DoneEvent | undefined;

    expect(done?.answer).toContain('Unique answer from invoke');
    expect(done?.answer).not.toContain('Should not appear in answer');
  });

  it('answer_chunk events concatenate to the callLlm response text', async () => {
    mockState.invokeContent = 'Short answer';

    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3, memoryEnabled: false });
    const events = await collectEvents(agent.run('test query'));

    const chunks = events.filter((e) => e.type === 'answer_chunk') as AnswerChunkEvent[];
    const done = events.find((e) => e.type === 'done') as DoneEvent | undefined;

    const assembled = chunks.map((c) => c.chunk).join('');
    expect(assembled).toBe('Short answer');
    expect(done?.answer).toBe('Short answer');
  });

  it('done event is emitted without waiting for a second LLM call', async () => {
    // If streamCallLlm were called (the old bug), it would be a second async op.
    // With the fix we get immediate fake-streaming — done arrives right away.
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3, memoryEnabled: false });
    const start = Date.now();
    await collectEvents(agent.run('test query'));
    const elapsed = Date.now() - start;

    // Should complete in well under 1 second (no real LLM calls, no second trip).
    expect(elapsed).toBeLessThan(1000);
    expect(mockState.streamCallCount).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// 2. streamCallLlm IS used for max-iterations synthesis
// ---------------------------------------------------------------------------

describe('Agent — streamCallLlm used for max-iterations synthesis', () => {
  beforeEach(() => {
    mockState.invokeReturnsToolCalls = true;
    mockState.streamChunks = ['synthesized conclusion'];
  });

  it('streamCallLlm is called when max iterations is hit (synthesis path)', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 2, memoryEnabled: false });
    await collectEvents(agent.run('test query'));
    expect(mockState.streamCallCount).toBeGreaterThanOrEqual(1);
  });

  it('done.answer contains synthesis output when max iterations is hit', async () => {
    mockState.streamChunks = ['synthesis output for max iterations'];
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 2, memoryEnabled: false });
    const events = await collectEvents(agent.run('test query'));
    const done = events.find((e) => e.type === 'done') as DoneEvent | undefined;
    expect(done?.answer).toContain('synthesis output for max iterations');
  });
});

// ---------------------------------------------------------------------------
// 3. Synthesis timeout is graceful (streamCallLlm throws)
// ---------------------------------------------------------------------------

describe('Agent — synthesis timeout graceful fallback', () => {
  beforeEach(() => {
    mockState.invokeReturnsToolCalls = true;
    mockState.streamShouldThrow = true;
  });

  it('done event is always emitted even when streamCallLlm throws', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 2, memoryEnabled: false });
    const events = await collectEvents(agent.run('test query'));
    const doneEvents = events.filter((e) => e.type === 'done');
    expect(doneEvents.length).toBe(1);
  });

  it('done event is the last event emitted after synthesis failure', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 2, memoryEnabled: false });
    const events = await collectEvents(agent.run('test query'));
    expect(events.at(-1)?.type).toBe('done');
  });

  it('answer_start is emitted before done even when synthesis fails', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 2, memoryEnabled: false });
    const events = await collectEvents(agent.run('test query'));

    const startIdx = events.findIndex((e) => e.type === 'answer_start');
    const doneIdx = events.findIndex((e) => e.type === 'done');

    expect(startIdx).toBeGreaterThanOrEqual(0);
    expect(doneIdx).toBeGreaterThan(startIdx);
  });
});

describe('buildAbstainingMarkovAnswer', () => {
  it('returns diagnostics-only answer when markov_distribution abstains', () => {
    const scratchpad = new Scratchpad('btc abstain test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 7 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'abstain',
          abstainReasons: ['No trusted anchors.', 'Out-of-sample Markov R² is -0.011.'],
          canonical: {
            ticker: 'BTC-USD',
            horizon: 7,
            currentPrice: 67713.5,
            diagnostics: {
              trustedAnchors: 0,
              totalAnchors: 2,
              anchorQuality: 'none',
              outOfSampleR2: -0.011,
              structuralBreakDetected: true,
              structuralBreakDivergence: 0.119,
              predictionConfidence: 0.14,
            },
          },
        },
      }),
    );

    const answer = buildAbstainingMarkovAnswer(scratchpad.getToolCallRecords());
    expect(answer).toContain('Model Abstained');
    expect(answer).toContain('no replacement scenario probabilities are shown');
    expect(answer).toContain('No trusted anchors.');
    expect(answer).toContain('Out-of-sample Markov R² is -0.011.');
    expect(answer).toContain('Trusted anchors');
  });

  it('returns null when markov_distribution did not abstain', () => {
    const scratchpad = new Scratchpad('btc ok test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 7 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'ok',
          canonical: { ticker: 'BTC-USD', horizon: 7, scenarios: { buckets: [] } },
        },
      }),
    );

    expect(buildAbstainingMarkovAnswer(scratchpad.getToolCallRecords())).toBeNull();
  });
});

describe('buildDistributionWarningPrefix', () => {
  it('returns warning prefix for distribution queries without successful markov output', () => {
    const answer = buildDistributionWarningPrefix(
      'What is the bitcoin probability distribution in 7 days?',
      [],
    );
    expect(answer).toContain('Warning: no validated Markov distribution was produced');
    expect(answer).toContain('fallback analysis');
  });

  it('includes abstention diagnostics when markov_distribution abstains', () => {
    const scratchpad = new Scratchpad('btc abstain warning test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 7 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'abstain',
          abstainReasons: ['No trusted anchors.'],
          canonical: {
            ticker: 'BTC-USD',
            horizon: 7,
            currentPrice: 67713.5,
            diagnostics: { trustedAnchors: 0, totalAnchors: 2, anchorQuality: 'none' },
          },
        },
      }),
    );

    const answer = buildDistributionWarningPrefix(
      'What is the bitcoin probability distribution in 7 days?',
      scratchpad.getToolCallRecords(),
    );
    expect(answer).toContain('Warning: no calibrated Markov terminal distribution was available');
    expect(answer).toContain('Key abstain reasons:');
    expect(answer).toContain('No trusted anchors.');
    expect(answer).not.toContain('## BTC-USD 7-Day Probability Distribution: Model Abstained');
  });

  it('returns null when successful markov output exists', () => {
    const scratchpad = new Scratchpad('btc ok test for distribution');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 7 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'ok',
          canonical: { ticker: 'BTC-USD', horizon: 7, scenarios: { buckets: [] } },
        },
      }),
    );

    expect(
      buildDistributionWarningPrefix(
        'What is the bitcoin probability distribution in 7 days?',
        scratchpad.getToolCallRecords(),
      ),
    ).toBeNull();
  });
});

describe('buildForecastDisagreementPrefix', () => {
  it('returns a mixed-evidence warning for BTC short-horizon disagreement', () => {
    const scratchpad = new Scratchpad('btc disagreement test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 14 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'ok',
          canonical: {
            actionSignal: { recommendation: 'BUY', expectedReturn: 0.032 },
            diagnostics: { markovWeight: 1 },
          },
        },
      }),
    );
    scratchpad.addToolResult(
      'polymarket_forecast',
      { ticker: 'BTC-USD', horizon_days: 14 },
      JSON.stringify({ data: { result: 'Forecast return: -0.4%\nGrade: B' } }),
    );

    const prefix = buildForecastDisagreementPrefix(
      'Provide a BTC forecast for the next 14 days',
      scratchpad.getToolCallRecords(),
    );

    expect(prefix).toContain('BTC short-horizon signals are mixed');
    expect(prefix).toContain('moderated confidence');
  });

  it('returns null when BTC short-horizon signals do not disagree', () => {
    const scratchpad = new Scratchpad('btc aligned test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 7 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'ok',
          canonical: {
            actionSignal: { recommendation: 'BUY', expectedReturn: 0.025 },
            diagnostics: { markovWeight: 1 },
          },
        },
      }),
    );
    scratchpad.addToolResult(
      'polymarket_forecast',
      { ticker: 'BTC-USD', horizon_days: 7 },
      JSON.stringify({ data: { result: 'Forecast return: +1.2%\nGrade: B' } }),
    );

    expect(
      buildForecastDisagreementPrefix(
        'Provide a BTC forecast for the next 7 days',
        scratchpad.getToolCallRecords(),
      ),
    ).toBeNull();
  });
});

// ---------------------------------------------------------------------------
// 4. Thinking text truncation (prevents raw JSON from flooding the terminal)
// ---------------------------------------------------------------------------

describe('Agent — thinking text truncation', () => {
  beforeEach(() => {
    // Two iterations: first with tool_calls + long thinking, second returns answer.
    mockState.invokeThinkingText = 'T'.repeat(1000); // 1000-char "thinking" blob

    // Prime the per-call queue: first callLlm → tool_calls, second → text answer.
    // Using the queue avoids a second mock.module('../model/llm.js') call here, which
    // would permanently override the module registry for subsequent test files in the
    // same Bun worker and cause the Agent.run tests in agent.test.ts to receive
    // "Final answer" instead of their own mockState.invokeContent.
    mockState.callLlmQueue = [
      { content: mockState.invokeThinkingText, toolCalls: [ST_TOOL_CALL] },
      { content: 'Final answer', toolCalls: [] },
    ];
  });

  it('thinking event message is at most 501 chars (500 + ellipsis)', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 5, memoryEnabled: false });
    const events = await collectEvents(agent.run('test query'));

    const thinkingEvents = events.filter((e) => e.type === 'thinking');
    expect(thinkingEvents.length).toBeGreaterThan(0);

    for (const evt of thinkingEvents) {
      const msg = (evt as { type: 'thinking'; message: string }).message;
      expect(msg.length).toBeLessThanOrEqual(501); // 500 chars + '…'
    }
  });

  it('truncated thinking ends with ellipsis when source text exceeds 500 chars', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 5, memoryEnabled: false });
    const events = await collectEvents(agent.run('test query'));

    const thinkEvt = events.find((e) => e.type === 'thinking') as
      | { type: 'thinking'; message: string }
      | undefined;

    expect(thinkEvt).toBeDefined();
    expect(thinkEvt!.message.endsWith('…')).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// buildSourcesFooter (pure function tests)
// ---------------------------------------------------------------------------

describe('buildSourcesFooter', () => {
  let buildSourcesFooter: (urls: string[]) => string;

  beforeEach(async () => {
    ({ buildSourcesFooter } = await import('./agent.js'));
  });

  it('returns empty string for no URLs', () => {
    expect(buildSourcesFooter([])).toBe('');
  });

  it('includes all URLs as a numbered list', () => {
    const footer = buildSourcesFooter(['https://a.com', 'https://b.com']);
    expect(footer).toContain('1. https://a.com');
    expect(footer).toContain('2. https://b.com');
    expect(footer).toContain('**Sources**');
  });

  it('caps output at 10 URLs', () => {
    const urls = Array.from({ length: 15 }, (_, i) => `https://site${i}.com`);
    const footer = buildSourcesFooter(urls);
    expect(footer).toContain('10. https://site9.com');
    expect(footer).not.toContain('11.');
  });

  it('deduplicates repeated URLs', () => {
    const footer = buildSourcesFooter(['https://a.com', 'https://a.com', 'https://b.com']);
    const matches = footer.match(/https:\/\/a\.com/g) ?? [];
    expect(matches).toHaveLength(1);
  });

  it('excludes Reddit URLs from the footer', () => {
    const footer = buildSourcesFooter([
      'https://reddit.com/r/Bitcoin/comments/abc',
      'https://www.reddit.com/r/investing/comments/xyz',
    ]);
    expect(footer).toBe('');
  });

  it('excludes X/Twitter URLs from the footer', () => {
    const footer = buildSourcesFooter([
      'https://x.com/user/status/123',
      'https://twitter.com/user/status/456',
    ]);
    expect(footer).toBe('');
  });

  it('includes non-social URLs while filtering social ones', () => {
    const footer = buildSourcesFooter([
      'https://reddit.com/r/Bitcoin/comments/abc',
      'https://financialmodelingprep.com/api/v3/profile/BTC',
      'https://polymarket.com/event/bitcoin-price',
    ]);
    expect(footer).not.toContain('reddit.com');
    expect(footer).toContain('financialmodelingprep.com');
    expect(footer).toContain('polymarket.com');
  });
});
