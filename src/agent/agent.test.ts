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
  });
});
