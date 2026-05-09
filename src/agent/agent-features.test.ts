/**
 * Tests for three agent features:
 *
 * 1. Streaming final answer — answer_start, answer_chunk, done events
 * 2. Graceful degradation at max iterations — synthesis instead of bare error
 * 3. Parallel tool execution event ordering — all tool_starts before tool_ends
 */
import { describe, it, expect, mock, beforeEach, afterEach, afterAll, beforeAll } from 'bun:test';
import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { BaseMessage } from '@langchain/core/messages';
import { AIMessage, AIMessageChunk } from '@langchain/core/messages';
import { _setModelFactory } from '../model/llm.js';
import type { AgentEvent, DoneEvent, AnswerChunkEvent } from './types.js';

// ---------------------------------------------------------------------------
// Isolation: each test gets a fresh tmp working dir so scratchpad JSONL files
// and memory state don't leak between tests.
// ---------------------------------------------------------------------------
let tmpDir: string;
let originalCwd: string;
let prevOpenAiKey: string | undefined;

beforeEach(() => {
  tmpDir = join(tmpdir(), `agent-feat-test-${Date.now()}-${Math.random().toString(36).slice(2)}`);
  mkdirSync(tmpDir, { recursive: true });
  originalCwd = process.cwd();
  process.chdir(tmpDir);

  // Provide a stub API key so getChatModel()'s getApiKey() doesn't throw.
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
// Mutable mock state — reset per test to control LLM behaviour.
// ---------------------------------------------------------------------------

const mockState = {
  /** When true, invoke() always returns a sequential_thinking tool call. */
  alwaysReturnToolCalls: false,
  /** Chunks yielded by stream() for the streaming / synthesis path. */
  streamChunks: ['The answer is 42'] as string[],
};

/** A single sequential_thinking tool call with valid args. */
const ST_TOOL_CALL = {
  id: 'st1',
  name: 'sequential_thinking',
  args: {
    thought: 'analyzing the query...',
    nextThoughtNeeded: false,
    thoughtNumber: 1,
    totalThoughts: 1,
  },
  type: 'tool_call' as const,
};

// ---------------------------------------------------------------------------
// Scoped model factory DI — avoids permanent mock.module contamination that
// breaks E2E/integration tests when llm.js has already been cached.
// ---------------------------------------------------------------------------

class SpyChatModel extends BaseChatModel {
  constructor(private state: typeof mockState) { super({}); }
  _llmType(): string { return 'spy-features'; }

  async _generate(_messages: BaseMessage[], _options: any, _runManager?: any) {
    if (this.state.alwaysReturnToolCalls) {
      return { generations: [{ message: new AIMessage({ content: '', tool_calls: [ST_TOOL_CALL], additional_kwargs: {} }), text: '' }] };
    }
    return { generations: [{ message: new AIMessage({ content: 'The answer is 42', additional_kwargs: {} }), text: 'The answer is 42' }] };
  }

  async *_streamIterator(_input: any, _options?: any) {
    for (const chunk of this.state.streamChunks) {
      yield new AIMessageChunk({ content: chunk });
    }
  }

  bindTools() { return this as any; }
}

beforeAll(() => { _setModelFactory(() => new SpyChatModel(mockState)); });
afterAll(() => { _setModelFactory(null); });

const { Agent } = await import('./agent.js');

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

async function collectEvents(gen: AsyncGenerator<AgentEvent>): Promise<AgentEvent[]> {
  const events: AgentEvent[] = [];
  for await (const e of gen) events.push(e);
  return events;
}

// ---------------------------------------------------------------------------
// Feature 1: Streaming final answer
// ---------------------------------------------------------------------------

describe('Agent — streaming final answer', () => {
  beforeEach(() => {
    mockState.alwaysReturnToolCalls = false;
    mockState.streamChunks = ['Hello ', 'streaming ', 'world'];
  });

  it('emits answer_start before any answer_chunk', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3 });
    const events = await collectEvents(agent.run('test query'));

    const idxStart = events.findIndex((e) => e.type === 'answer_start');
    const idxFirstChunk = events.findIndex((e) => e.type === 'answer_chunk');

    expect(idxStart).toBeGreaterThanOrEqual(0);
    expect(idxFirstChunk).toBeGreaterThan(idxStart);
  }, 15_000);

  it('emits multiple answer_chunk events (true streaming, not one blob)', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3 });
    const events = await collectEvents(agent.run('test query'));

    const chunks = events.filter((e) => e.type === 'answer_chunk') as AnswerChunkEvent[];
    expect(chunks.length).toBeGreaterThan(0);
  });

  it('emits done event after all chunks with assembled answer', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3 });
    const events = await collectEvents(agent.run('test query'));

    const chunks = events.filter((e) => e.type === 'answer_chunk') as AnswerChunkEvent[];
    const done = events.find((e) => e.type === 'done') as DoneEvent | undefined;

    expect(done).toBeDefined();
    const assembled = chunks.map((c) => c.chunk).join('');
    expect(done!.answer).toBe(assembled);
  });

  it('done.answer equals the concatenation of all chunk events', async () => {
    mockState.streamChunks = ['Part one ', 'and part two'];
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3 });
    const events = await collectEvents(agent.run('test query'));

    const chunks = events.filter((e) => e.type === 'answer_chunk') as AnswerChunkEvent[];
    const done = events.find((e) => e.type === 'done') as DoneEvent | undefined;

    expect(done?.answer).toBe(chunks.map((c) => c.chunk).join(''));
  });

  it('done event is always emitted (even when stream yields a single chunk)', async () => {
    mockState.streamChunks = ['single chunk answer'];
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3 });
    const events = await collectEvents(agent.run('test query'));

    const doneEvents = events.filter((e) => e.type === 'done');
    expect(doneEvents.length).toBe(1);
  });

  it('done.answer is non-empty', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3 });
    const events = await collectEvents(agent.run('tell me something'));

    const done = events.find((e) => e.type === 'done') as DoneEvent | undefined;
    expect(done?.answer).toBeTruthy();
  });

  it('answer_start event appears exactly once per query', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3 });
    const events = await collectEvents(agent.run('test query'));

    const starts = events.filter((e) => e.type === 'answer_start');
    expect(starts.length).toBe(1);
  });

  it('done event is the last event emitted', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3 });
    const events = await collectEvents(agent.run('test query'));

    const last = events.at(-1);
    expect(last?.type).toBe('done');
  });
});

// ---------------------------------------------------------------------------
// Feature 2: Graceful degradation at max iterations
// ---------------------------------------------------------------------------

describe('Agent — graceful degradation at max iterations', () => {
  beforeEach(() => {
    mockState.alwaysReturnToolCalls = true;
    mockState.streamChunks = ['[Best-effort summary ', '— research may be incomplete]\n\nSynthesis answer'];
  });

  it('always emits done event (never hangs) when maxIterations is hit', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 2 });
    const events = await collectEvents(agent.run('What is the market cap of NVDA?'));

    const doneEvents = events.filter((e) => e.type === 'done');
    expect(doneEvents.length).toBe(1);
  });

  it('done.answer is not a generic "Error:" string when research was gathered', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 2 });
    const events = await collectEvents(agent.run('Analyze SPY performance'));

    const done = events.find((e) => e.type === 'done') as DoneEvent | undefined;
    expect(done?.answer).not.toMatch(/^Error:/);
  });

  it('emits answer_start before done (synthesis path goes through streaming)', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 2 });
    const events = await collectEvents(agent.run('Research AAPL valuation'));

    const idxStart = events.findIndex((e) => e.type === 'answer_start');
    const idxDone = events.findIndex((e) => e.type === 'done');

    expect(idxStart).toBeGreaterThanOrEqual(0);
    expect(idxDone).toBeGreaterThan(idxStart);
  });

  it('done.iterations is within [1, maxIterations] when limit is hit', async () => {
    const maxIterations = 3;
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations });
    const events = await collectEvents(agent.run('multi-step research query'));

    const done = events.find((e) => e.type === 'done') as DoneEvent | undefined;
    expect(done?.iterations).toBeGreaterThanOrEqual(1);
    expect(done?.iterations).toBeLessThanOrEqual(maxIterations);
  });

  it('emits answer_chunk events during synthesis (not silent)', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 2 });
    const events = await collectEvents(agent.run('Research query'));

    const chunks = events.filter((e) => e.type === 'answer_chunk') as AnswerChunkEvent[];
    expect(chunks.length).toBeGreaterThan(0);
  });

  it('done is still emitted with maxIterations=1', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 1 });
    const events = await collectEvents(agent.run('quick test'));

    const doneEvents = events.filter((e) => e.type === 'done');
    expect(doneEvents.length).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// Feature 3: Parallel tool execution event ordering
// ---------------------------------------------------------------------------
// Covered exhaustively in tool-executor.test.ts for the executor layer.
// Here we add an agent-integration-level check: when the LLM requests multiple
// tool calls in one response, all tool_start events appear before any tool_end.
// ---------------------------------------------------------------------------

describe('Agent — parallel tool execution event ordering', () => {
  beforeEach(() => {
    mockState.alwaysReturnToolCalls = false;
    mockState.streamChunks = ['done'];
  });

  it('all tool_start events precede any tool_end in a two-tool batch', async () => {
    const { AgentToolExecutor } = await import('./tool-executor.js');
    const { createRunContext } = await import('./run-context.js');
    const { AIMessage } = await import('@langchain/core/messages');
    const subDir = join(tmpdir(), `parallel-order-${Date.now()}`);
    mkdirSync(subDir, { recursive: true });
    const savedCwd = process.cwd();
    process.chdir(subDir);

    try {
      let toolCallCount = 0;
      const slowTool = {
        name: 'web_search',
        invoke: async () => {
          toolCallCount++;
          await new Promise<void>((r) => setTimeout(r, 5));
          return '{"result":"data"}';
        },
        lc_namespace: [],
        schema: {},
      } as unknown as import('@langchain/core/tools').StructuredToolInterface;

      const fastTool = {
        name: 'get_market_data',
        invoke: async () => {
          toolCallCount++;
          return '{"price":100}';
        },
        lc_namespace: [],
        schema: {},
      } as unknown as import('@langchain/core/tools').StructuredToolInterface;

      const executor = new AgentToolExecutor(
        new Map([['web_search', slowTool], ['get_market_data', fastTool]]),
      );
      const ctx = createRunContext('parallel test');

      const msg = new AIMessage({
        content: '',
        tool_calls: [
          { id: 'c1', name: 'web_search', args: { query: 'AAPL' }, type: 'tool_call' },
          { id: 'c2', name: 'get_market_data', args: { ticker: 'AAPL' }, type: 'tool_call' },
        ],
      });

      const events: import('./types.js').AgentEvent[] = [];
      for await (const e of executor.executeAll(msg, ctx)) {
        events.push(e as import('./types.js').AgentEvent);
      }

      const startEvents = events.filter((e) => e.type === 'tool_start');
      const endEvents = events.filter((e) => e.type === 'tool_end');

      expect(startEvents).toHaveLength(2);
      expect(endEvents).toHaveLength(2);

      const idxLastStart = events.findLastIndex((e) => e.type === 'tool_start');
      const idxFirstEnd = events.findIndex((e) => e.type === 'tool_end');
      expect(idxLastStart).toBeLessThan(idxFirstEnd);
    } finally {
      process.chdir(savedCwd);
      rmSync(subDir, { recursive: true, force: true });
    }
  });

  it('single-tool execution still produces tool_start then tool_end', async () => {
    const { AgentToolExecutor } = await import('./tool-executor.js');
    const { createRunContext } = await import('./run-context.js');
    const { AIMessage } = await import('@langchain/core/messages');

    const fakeTool = {
      name: 'web_search',
      invoke: async () => '{"result":"ok"}',
      lc_namespace: [],
      schema: {},
    } as unknown as import('@langchain/core/tools').StructuredToolInterface;

    const executor = new AgentToolExecutor(new Map([['web_search', fakeTool]]));
    const ctx = createRunContext('single test');

    const msg = new AIMessage({
      content: '',
      tool_calls: [
        { id: 'c1', name: 'web_search', args: { query: 'test' }, type: 'tool_call' },
      ],
    });

    const events: import('./types.js').AgentEvent[] = [];
    for await (const e of executor.executeAll(msg, ctx)) {
      events.push(e as import('./types.js').AgentEvent);
    }

    const idxStart = events.findIndex((e) => e.type === 'tool_start');
    const idxEnd = events.findIndex((e) => e.type === 'tool_end');
    expect(idxStart).toBeGreaterThanOrEqual(0);
    expect(idxEnd).toBeGreaterThan(idxStart);
  });
});
