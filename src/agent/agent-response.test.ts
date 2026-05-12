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
import { describe, it, expect, mock, beforeEach, afterEach, beforeAll, afterAll } from 'bun:test';
import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import type { StructuredToolInterface } from '@langchain/core/tools';
import type { AgentEvent, DoneEvent, AnswerChunkEvent } from './types.js';
import { Scratchpad } from './scratchpad.js';
import { RECOMMENDED_CONFIDENCE_THRESHOLD } from '../tools/finance/markov-distribution.js';
import { _setModelFactory } from '../model/llm.js';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import { AIMessage, AIMessageChunk, type BaseMessage } from '@langchain/core/messages';

// ---------------------------------------------------------------------------
// Mutable mock state — reset before each test.
// ---------------------------------------------------------------------------
type MockToolCall = {
  id: string;
  name: string;
  args: Record<string, unknown>;
  type: 'tool_call';
};

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
  callLlmQueue: [] as Array<{ content: string; toolCalls: MockToolCall[] }>,
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
// Scoped model factory DI — avoids permanent mock.module contamination that
// poisons E2E/integration tests sharing the same Bun worker.
// ---------------------------------------------------------------------------
class SpyChatModel extends BaseChatModel {
  constructor(private state: typeof mockState) { super({}); }

  _llmType(): string { return 'spy'; }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async _generate(messages: BaseMessage[], options: any, _runManager?: any) {
    if (this.state.callLlmQueue.length > 0) {
      const next = this.state.callLlmQueue.shift()!;
      return {
        generations: [{
          message: new AIMessage({ content: next.content, tool_calls: next.toolCalls, additional_kwargs: {} }),
          text: next.content,
        }],
      };
    }
    if (this.state.invokeReturnsToolCalls) {
      return {
        generations: [{
          message: new AIMessage({ content: this.state.invokeThinkingText, tool_calls: [ST_TOOL_CALL], additional_kwargs: {} }),
          text: '',
        }],
      };
    }
    return {
      generations: [{
        message: new AIMessage({ content: this.state.invokeContent, additional_kwargs: {} }),
        text: this.state.invokeContent,
      }],
    };
  }

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  async *_streamIterator(input: any, options?: any) {
    this.state.streamCallCount++;
    if (this.state.streamShouldThrow) {
      throw new Error('LLM stream timed out');
    }
    for (const chunk of this.state.streamChunks) {
      yield new AIMessageChunk({ content: chunk });
    }
  }

  bindTools(_tools: any[]): any { return this; }
  withStructuredOutput(_schema: any, _opts?: any): any { return this; }
}

beforeAll(() => { _setModelFactory((_name, _opts, _thinkOverride) => new SpyChatModel(mockState)); });
afterAll(() => {
  _setModelFactory(null);
});

// ---------------------------------------------------------------------------
// Dynamic import after model-factory DI is registered.
// ---------------------------------------------------------------------------
const {
  Agent,
  buildAbstainingMarkovAnswer,
  buildAbstainingBtcShortHorizonForecastAnswer,
  buildDistributionWarningPrefix,
  buildForecastDisagreementPrefix,
  buildLowConfidenceBtcShortHorizonForecastPrefix,
  ensureStructuredDensityTable,
} = await import('./agent.js');

const sequentialThinkingTool = {
  name: 'sequential_thinking',
  invoke: async () => JSON.stringify({ ok: true }),
} satisfies Pick<StructuredToolInterface, 'name' | 'invoke'>;

const markovDistributionTool = {
  name: 'markov_distribution',
  invoke: async () => JSON.stringify({
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
} satisfies Pick<StructuredToolInterface, 'name' | 'invoke'>;

const RESPONSE_TEST_TOOLS = [
  sequentialThinkingTool,
  markovDistributionTool,
] as unknown as StructuredToolInterface[];

function createResponseTestAgent(maxIterations: number) {
  return Agent.create({
    model: 'gpt-5.4',
    maxIterations,
    memoryEnabled: false,
    tools: RESPONSE_TEST_TOOLS,
  });
}

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
    const agent = await createResponseTestAgent(3);
    await collectEvents(agent.run('test query'));
    expect(mockState.streamCallCount).toBe(0);
  });

  it('done.answer matches callLlm content, not the stream chunks', async () => {
    mockState.invokeContent = 'Unique answer from invoke';
    mockState.streamChunks = ['Should not appear in answer'];

    const agent = await createResponseTestAgent(3);
    const events = await collectEvents(agent.run('test query'));
    const done = events.find((e) => e.type === 'done') as DoneEvent | undefined;

    expect(done?.answer).toContain('Unique answer from invoke');
    expect(done?.answer).not.toContain('Should not appear in answer');
  });

  it('answer_chunk events concatenate to the callLlm response text', async () => {
    mockState.invokeContent = 'Short answer';

    const agent = await createResponseTestAgent(3);
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
    const agent = await createResponseTestAgent(3);
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
    const agent = await createResponseTestAgent(2);
    await collectEvents(agent.run('test query'));
    expect(mockState.streamCallCount).toBeGreaterThanOrEqual(1);
  });

  it('done.answer contains synthesis output when max iterations is hit', async () => {
    mockState.streamChunks = ['synthesis output for max iterations'];
    const agent = await createResponseTestAgent(2);
    const events = await collectEvents(agent.run('test query'));
    const done = events.find((e) => e.type === 'done') as DoneEvent | undefined;
    expect(done?.answer).toContain('synthesis output for max iterations');
  });

  it('uses the BTC abstain-preserving answer instead of synthesis output at max iterations', async () => {
    mockState.callLlmQueue = [
      { content: '', toolCalls: [ST_TOOL_CALL] },
      {
        content: '',
        toolCalls: [
          {
            id: 'm1',
            name: 'markov_distribution',
            args: { ticker: 'BTC-USD', horizon: 14, trajectory: true, trajectoryDays: 14 },
            type: 'tool_call' as const,
          },
        ],
      },
    ];
    mockState.streamChunks = ['this synthesis output should be bypassed'];
    mockState.streamShouldThrow = true;

    const agent = await createResponseTestAgent(1);
    const events = await collectEvents(agent.run('Provide a BTC forecast for the next 14 days'));
    const done = events.find((e) => e.type === 'done') as DoneEvent | undefined;

    expect(done?.answer).toContain('Model Abstained');
    expect(done?.answer).toContain('Decision guidance');
    expect(done?.answer).not.toContain('this synthesis output should be bypassed');
  });

  it('prefixes the full answer with BTC low-confidence selective-gate wording', () => {
    const scratchpad = new Scratchpad('btc low confidence full answer test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 7 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'ok',
          canonical: {
            actionSignal: { recommendation: 'BUY', expectedReturn: 0.025 },
            diagnostics: { markovWeight: 0.6, predictionConfidence: 0.18 },
          },
        },
      }),
    );

    const prefix = buildLowConfidenceBtcShortHorizonForecastPrefix(
      'Provide a BTC forecast for the next 7 days',
      scratchpad.getToolCallRecords(),
    );
    const fullAnswer = `${prefix ?? ''}Base forecast text.`;

    expect(fullAnswer).toContain('BTC short-horizon selective Markov gate did not clear');
    expect(fullAnswer).toContain('fallback context');
    expect(fullAnswer).toContain('Base forecast text.');
  });

  it('does not prefix the full answer when BTC short-horizon markov confidence clears the selective gate', () => {
    const scratchpad = new Scratchpad('btc high confidence full answer test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 7 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'ok',
          canonical: {
            actionSignal: { recommendation: 'BUY', expectedReturn: 0.025 },
            diagnostics: { markovWeight: 0.6, predictionConfidence: 0.32 },
          },
        },
      }),
    );

    const prefix = buildLowConfidenceBtcShortHorizonForecastPrefix(
      'Provide a BTC forecast for the next 7 days',
      scratchpad.getToolCallRecords(),
    );
    const fullAnswer = `${prefix ?? ''}Base forecast text.`;

    expect(fullAnswer).not.toContain('BTC short-horizon selective Markov gate did not clear');
    expect(fullAnswer).not.toContain('fallback context');
    expect(fullAnswer).toContain('Base forecast text.');
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
    const agent = await createResponseTestAgent(2);
    const events = await collectEvents(agent.run('test query'));
    const doneEvents = events.filter((e) => e.type === 'done');
    expect(doneEvents.length).toBe(1);
  });

  it('done event is the last event emitted after synthesis failure', async () => {
    const agent = await createResponseTestAgent(2);
    const events = await collectEvents(agent.run('test query'));
    expect(events.at(-1)?.type).toBe('done');
  });

  it('answer_start is emitted before done even when synthesis fails', async () => {
    const agent = await createResponseTestAgent(2);
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

describe('buildAbstainingBtcShortHorizonForecastAnswer', () => {
  it('returns a no-trade abstain answer for BTC short-horizon forecast queries', () => {
    const scratchpad = new Scratchpad('btc short-horizon abstain test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 14 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'abstain',
          abstainReasons: ['No trusted terminal prediction-market anchors are available for this horizon.'],
          canonical: {
            ticker: 'BTC-USD',
            horizon: 14,
            currentPrice: 75785,
            diagnostics: {
              trustedAnchors: 0,
              totalAnchors: 5,
              anchorQuality: 'none',
              outOfSampleR2: 0.162,
              structuralBreakDetected: true,
              structuralBreakDivergence: 0.346,
              predictionConfidence: 0.19,
            },
          },
        },
      }),
    );

    const answer = buildAbstainingBtcShortHorizonForecastAnswer(
      'Provide a BTC forecast for the next 14 days',
      scratchpad.getToolCallRecords(),
    );

    expect(answer).toContain('BTC-USD 14-Day Probability Distribution: Model Abstained');
    expect(answer).toContain('Decision guidance');
    expect(answer).toContain('no-trade / no-calibrated-edge');
    expect(answer).not.toContain('point forecast');
  });

  it('returns null for non-BTC or longer-horizon forecast queries', () => {
    const scratchpad = new Scratchpad('non-btc abstain test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'ETH-USD', horizon: 14 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'abstain',
          canonical: {
            ticker: 'ETH-USD',
            horizon: 14,
            diagnostics: {
              trustedAnchors: 0,
              totalAnchors: 0,
              anchorQuality: 'none',
            },
          },
        },
      }),
    );

    expect(
      buildAbstainingBtcShortHorizonForecastAnswer(
        'Provide an ETH forecast for the next 14 days',
        scratchpad.getToolCallRecords(),
      ),
    ).toBeNull();
  });

  it('treats BTC next-week forecast phrasing as a short-horizon abstain case', () => {
    const scratchpad = new Scratchpad('btc next week abstain test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 5 },
      JSON.stringify({
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
        },
      }),
    );

    const answer = buildAbstainingBtcShortHorizonForecastAnswer(
      'Provide a BTC forecast for next week',
      scratchpad.getToolCallRecords(),
    );

    expect(answer).toContain('Model Abstained');
    expect(answer).toContain('Decision guidance');
  });

  it('does not emit the BTC abstention answer when polymarket_forecast was explicitly requested', () => {
    const scratchpad = new Scratchpad('btc explicit polymarket abstain test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 2 },
      JSON.stringify({
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
    );

    expect(
      buildAbstainingBtcShortHorizonForecastAnswer(
        'Use polymarket_forecast for BTC over the next 2 days',
        scratchpad.getToolCallRecords(),
      ),
    ).toBeNull();
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

describe('buildLowConfidenceBtcShortHorizonForecastPrefix', () => {
  it('returns a selective-gate warning for low-confidence BTC short-horizon markov results', () => {
    const scratchpad = new Scratchpad('btc low confidence test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 7 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'ok',
          canonical: {
            actionSignal: { recommendation: 'BUY', expectedReturn: 0.025 },
            diagnostics: { markovWeight: 0.6, predictionConfidence: 0.18 },
          },
        },
      }),
    );

    const prefix = buildLowConfidenceBtcShortHorizonForecastPrefix(
      'Provide a BTC forecast for the next 7 days',
      scratchpad.getToolCallRecords(),
    );

    expect(prefix).toContain('BTC short-horizon selective Markov gate did not clear');
    expect(prefix).toContain(`${RECOMMENDED_CONFIDENCE_THRESHOLD.toFixed(2)} selective threshold`);
    expect(prefix).toContain('fallback context');
  });

  it('returns null when BTC short-horizon markov confidence clears the threshold', () => {
    const scratchpad = new Scratchpad('btc high confidence test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 7 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'ok',
          canonical: {
            actionSignal: { recommendation: 'BUY', expectedReturn: 0.025 },
            diagnostics: { markovWeight: 0.6, predictionConfidence: 0.32 },
          },
        },
      }),
    );

    expect(
      buildLowConfidenceBtcShortHorizonForecastPrefix(
        'Provide a BTC forecast for the next 7 days',
        scratchpad.getToolCallRecords(),
      ),
    ).toBeNull();
  });
});

describe('ensureStructuredDensityTable', () => {
  const canonicalDistribution = [
    { price: 78000, probability: 0.99, lowerBound: 76000, upperBound: 80000, source: 'markov' as const },
    { price: 80000, probability: 0.88, lowerBound: 78000, upperBound: 82000, source: 'markov' as const },
    { price: 82000, probability: 0.61, lowerBound: 80000, upperBound: 84000, source: 'markov' as const },
    { price: 84000, probability: 0.28, lowerBound: 82000, upperBound: 86000, source: 'markov' as const },
    { price: 86000, probability: 0.08, lowerBound: 84000, upperBound: 88000, source: 'markov' as const },
    { price: 88000, probability: 0.01, lowerBound: 86000, upperBound: 90000, source: 'markov' as const },
  ];

  function buildDensityScratchpad() {
    const scratchpad = new Scratchpad('btc structured density test');
    scratchpad.addToolResult(
      'markov_distribution',
      { ticker: 'BTC-USD', horizon: 1 },
      JSON.stringify({
        data: {
          _tool: 'markov_distribution',
          status: 'ok',
          distribution: canonicalDistribution,
          canonical: {
            ticker: 'BTC-USD',
            horizon: 1,
          },
        },
      }),
    );
    return scratchpad;
  }

  function parseDensityBucketRows(answer: string): Array<{
    bucket: number;
    range: string;
    probability: number;
    lower: number | null;
    upper: number | null;
  }> {
    return answer
      .split('\n')
      .map((line) => line.trim())
      .filter((line) => /^\|\s*\d+\s*\|/.test(line))
      .map((line) => {
        const cells = line.split('|').slice(1, -1).map((cell) => cell.trim());
        const bucket = Number(cells[0]);
        const range = cells[1] ?? '';
        const probability = Number((cells[2]?.match(/([\d.]+)%/) ?? [])[1]);
        const prices = [...range.matchAll(/\$([0-9][0-9,]*(?:\.\d+)?)/g)]
          .map((match) => Number(match[1]!.replace(/,/g, '')));

        if (range.startsWith('<')) {
          return {
            bucket,
            range,
            probability,
            lower: null,
            upper: prices[0] ?? null,
          };
        }

        if (range.startsWith('>')) {
          return {
            bucket,
            range,
            probability,
            lower: prices[0] ?? null,
            upper: null,
          };
        }

        return {
          bucket,
          range,
          probability,
          lower: prices[0] ?? null,
          upper: prices[1] ?? null,
        };
      })
      .filter((row) => Number.isFinite(row.bucket) && Number.isFinite(row.probability));
  }

  it('replaces non-parseable density prose/table with a canonical bucket table', () => {
    const scratchpad = buildDensityScratchpad();

    const answer = ensureStructuredDensityTable(
      [
        '## BTC 24-Hour Forecast',
        '',
        '## 9-Part Density Probability Table',
        '',
        '| Price Range | Probability |',
        '|-------------|-------------|',
        '| $80K–$82K | 20.0% |',
      ].join('\n'),
      'Provide the Polymarket and Markov BTC forecast for 24 hours, also providing the density probabilities for the price range divided into 9 parts.',
      scratchpad.getToolCallRecords(),
    );

    expect(answer).toContain('Canonical scenario breakdown (P(bucket) = probability mass in each price range)');
    expect(answer).toContain('| Bucket | Price Range | P(bucket) |');
    const rows = parseDensityBucketRows(answer);
    expect(rows).toHaveLength(9);

    const firstBucket = rows[0]!;
    const lastBucket = rows[8]!;

    expect(firstBucket.bucket).toBe(1);
    expect(firstBucket.upper).not.toBeNull();
    expect(firstBucket.upper).toBeGreaterThan(canonicalDistribution[0]!.price);
    expect(firstBucket.upper).toBeLessThan(canonicalDistribution[1]!.price);
    expect(firstBucket.probability).toBeGreaterThan(4);

    expect(lastBucket.bucket).toBe(9);
    expect(lastBucket.lower).not.toBeNull();
    expect(lastBucket.lower).toBeGreaterThan(canonicalDistribution[4]!.price);
    expect(lastBucket.lower).toBeLessThan(canonicalDistribution[5]!.price);
    expect(lastBucket.probability).toBeGreaterThan(4);
  });

  it('uses a midpoint split for 2-bucket requests instead of collapsing to an edge bucket', () => {
    const answer = ensureStructuredDensityTable(
      '## BTC 24-Hour Forecast',
      'Provide the Markov BTC forecast for 24 hours with density probabilities divided into 2 parts.',
      buildDensityScratchpad().getToolCallRecords(),
    );

    const rows = parseDensityBucketRows(answer);
    expect(rows).toHaveLength(2);

    const firstBucket = rows[0]!;
    const secondBucket = rows[1]!;

    expect(firstBucket.upper).not.toBeNull();
    expect(secondBucket.lower).not.toBeNull();
    expect(firstBucket.upper).toBeGreaterThan(canonicalDistribution[1]!.price);
    expect(firstBucket.upper).toBeLessThan(canonicalDistribution[4]!.price);
    expect(secondBucket.lower).toBe(firstBucket.upper);
    expect(firstBucket.probability).toBeGreaterThan(20);
    expect(secondBucket.probability).toBeGreaterThan(20);
    expect(Math.abs(firstBucket.probability - secondBucket.probability)).toBeLessThan(20);
  });

  it('leaves already-structured density tables unchanged', () => {
    const structuredAnswer = [
      '## 9-Part Density Probability Table',
      '',
      'Canonical scenario breakdown (P(bucket) = probability mass in each price range):',
      '',
      '| Bucket | Price Range | P(bucket) |',
      '|--------|-------------|-----------|',
      '| 1 | < $78K | 1.00% |',
      '| 2 | $78K–$80K | 4.00% |',
      '| 3 | $80K–$82K | 15.00% |',
      '| 4 | $82K–$84K | 30.00% |',
      '| 5 | $84K–$86K | 25.00% |',
    ].join('\n');

    expect(
      ensureStructuredDensityTable(
        structuredAnswer,
        'Provide the Polymarket and Markov BTC forecast for 24 hours, also providing the density probabilities for the price range divided into 9 parts.',
        [],
      ),
    ).toBe(structuredAnswer);
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
    const agent = await createResponseTestAgent(5);
    const events = await collectEvents(agent.run('test query'));

    const thinkingEvents = events.filter((e) => e.type === 'thinking');
    expect(thinkingEvents.length).toBeGreaterThan(0);

    for (const evt of thinkingEvents) {
      const msg = (evt as { type: 'thinking'; message: string }).message;
      expect(msg.length).toBeLessThanOrEqual(501); // 500 chars + '…'
    }
  });

  it('truncated thinking ends with ellipsis when source text exceeds 500 chars', async () => {
    const agent = await createResponseTestAgent(5);
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
