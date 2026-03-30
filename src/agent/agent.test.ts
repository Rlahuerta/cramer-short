import { describe, it, expect, mock, beforeEach, afterAll } from 'bun:test';
import { tmpdir } from 'os';
import { join } from 'path';
import { rmSync } from 'fs';

const testDir = join(tmpdir(), `dexter-agent-test-${Date.now()}`);

// ── Mutable LLM implementation ───────────────────────────────────────────────
let mockLlmImpl: () => Promise<any> = () =>
  Promise.resolve({
    response: 'Direct answer',
    usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
  });
const mockCallLlm = mock((..._args: any[]) => mockLlmImpl());

// ── Mutable tool invoke implementation ──────────────────────────────────────
let mockToolInvokeImpl: () => Promise<any> = () => Promise.resolve('{}');
const mockToolInvoke = mock((..._args: any[]) => mockToolInvokeImpl());

// ── Module mocks (must be before any dynamic imports) ────────────────────────
mock.module('../model/llm.js', () => ({
  callLlm: mockCallLlm,
  DEFAULT_MODEL: 'gpt-5.4',
  DEFAULT_PROVIDER: 'openai',
  getFastModel: mock(() => 'gpt-5.4'),
  getChatModel: mock(() => ({})),
}));

mock.module('../tools/registry.js', () => ({
  getTools: mock(() => [{ name: 'web_search', invoke: mockToolInvoke }]),
  buildToolDescriptions: mock(() => 'mock tool descriptions'),
}));

mock.module('../skills/index.js', () => ({
  discoverSkills: mock(() => []),
  buildSkillMetadataSection: mock(() => ''),
  clearSkillCache: mock(() => {}),
  getSkill: mock(() => null),
  parseSkillFile: mock(() => null),
  loadSkillFromPath: mock(() => null),
  extractSkillMetadata: mock(() => null),
}));

mock.module('../utils/paths.js', () => ({
  dexterPath: mock((sub: string) => join(testDir, sub)),
  getDexterDir: mock(() => testDir),
}));

mock.module('../memory/index.js', () => ({
  MemoryManager: {
    get: mock(() =>
      Promise.resolve({
        listFiles: mock(() => Promise.resolve([])),
        loadSessionContext: mock(() => Promise.resolve({ text: '' })),
      })
    ),
  },
}));

mock.module('../memory/flush.js', () => ({
  shouldRunMemoryFlush: mock(() => false),
  runMemoryFlush: mock(() => Promise.resolve({ flushed: false, written: false })),
}));

mock.module('../utils/ai-message.js', () => ({
  extractTextContent: mock((msg: any) => {
    if (typeof msg === 'string') return msg;
    if (Array.isArray(msg?.content)) {
      return msg.content
        .filter((b: any) => b?.type === 'text')
        .map((b: any) => b.text)
        .join('');
    }
    return msg?.content ?? '';
  }),
  hasToolCalls: mock(
    (msg: any) => Array.isArray(msg?.tool_calls) && msg.tool_calls.length > 0,
  ),
}));

mock.module('../utils/history-context.js', () => ({
  buildHistoryContext: mock(({ currentMessage }: any) => currentMessage),
  DEFAULT_HISTORY_LIMIT: 10,
  FULL_ANSWER_TURNS: 3,
  HISTORY_CONTEXT_MARKER: '[Chat history for context]',
  CURRENT_MESSAGE_MARKER: '[Current message - respond to this]',
}));

mock.module('../utils/errors.js', () => ({
  formatUserFacingError: mock((raw: string) => raw),
  isContextOverflowError: mock(() => false),
  isRateLimitError: mock(() => false),
  isBillingError: mock(() => false),
  isAuthError: mock(() => false),
  isTimeoutError: mock(() => false),
  isOverloadedError: mock(() => false),
  isNonRetryableError: mock(() => false),
  classifyError: mock(() => 'unknown'),
  parseApiErrorInfo: mock(() => null),
}));

mock.module('../utils/tokens.js', () => ({
  estimateTokens: mock(() => 0), // Always under threshold
  TOKEN_BUDGET: 150_000,
  CONTEXT_THRESHOLD: 100_000,
  KEEP_TOOL_USES: 5,
}));

mock.module('../providers.js', () => ({
  resolveProvider: mock(() => ({ displayName: 'OpenAI', id: 'openai' })),
  getProviderById: mock(() => ({ displayName: 'OpenAI', id: 'openai', fastModel: 'gpt-5.4' })),
  PROVIDERS: [],
}));

mock.module('../utils/progress-channel.js', () => ({
  createProgressChannel: () => {
    let closedResolve: (() => void) | undefined;
    const closedPromise = new Promise<void>((r) => {
      closedResolve = r;
    });
    return {
      emit: mock(),
      close: mock(() => { closedResolve?.(); }),
      [Symbol.asyncIterator]: async function* () {
        await closedPromise;
      },
    };
  },
}));

const { Agent } = await import('./agent.js');

afterAll(() => {
  rmSync(testDir, { recursive: true, force: true });
});

// Helper to collect all events from an async generator
async function collectEvents(gen: AsyncGenerator<any>): Promise<any[]> {
  const events: any[] = [];
  for await (const e of gen) events.push(e);
  return events;
}

describe('Agent', () => {
  beforeEach(() => {
    // Reset to default: returns a direct string answer
    mockLlmImpl = () =>
      Promise.resolve({
        response: 'Direct answer',
        usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
      });
    mockToolInvokeImpl = () => Promise.resolve('{}');
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
    it('yields a done event with the direct answer when no tool calls', async () => {
      mockLlmImpl = () =>
        Promise.resolve({
          response: 'Direct answer to query',
          usage: { inputTokens: 10, outputTokens: 5, totalTokens: 15 },
        });

      const agent = await Agent.create({});
      const events = await collectEvents(agent.run('What time is it?'));
      const done = events.find(e => e.type === 'done');
      expect(done).toBeDefined();
      expect(done.answer).toBe('Direct answer to query');
    });

    it('done event includes iterations and totalTime', async () => {
      const agent = await Agent.create({});
      const events = await collectEvents(agent.run('simple question'));
      const done = events.find(e => e.type === 'done');
      expect(done).toBeDefined();
      expect(typeof done.iterations).toBe('number');
      expect(typeof done.totalTime).toBe('number');
      expect(done.iterations).toBeGreaterThanOrEqual(1);
      expect(done.totalTime).toBeGreaterThanOrEqual(0);
    });

    it('yields done with error message when LLM throws', async () => {
      mockLlmImpl = () => Promise.reject(new Error('LLM connection failed'));

      const agent = await Agent.create({});
      const events = await collectEvents(agent.run('test query'));
      const done = events.find(e => e.type === 'done');
      expect(done).toBeDefined();
      expect(done.answer).toContain('Error');
    });

    it('executes tool calls and then gets final answer', async () => {
      let callCount = 0;
      mockLlmImpl = () => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve({
            response: {
              content: '',
              tool_calls: [{ name: 'web_search', args: { query: 'AAPL' } }],
            },
            usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10 },
          });
        }
        return Promise.resolve({
          response: 'AAPL is trading at $180.',
          usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10 },
        });
      };

      const agent = await Agent.create({});
      const events = await collectEvents(agent.run('What is AAPL price?'));
      const done = events.find(e => e.type === 'done');
      expect(done).toBeDefined();
      expect(done.answer).toBe('AAPL is trading at $180.');
    });

    it('yields tool_start and tool_end events during tool execution', async () => {
      let callCount = 0;
      mockLlmImpl = () => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve({
            response: {
              content: '',
              tool_calls: [{ name: 'web_search', args: { query: 'test' } }],
            },
            usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10 },
          });
        }
        return Promise.resolve({
          response: 'Final answer',
          usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10 },
        });
      };

      const agent = await Agent.create({});
      const events = await collectEvents(agent.run('search something'));
      expect(events.some(e => e.type === 'tool_start')).toBe(true);
      expect(events.some(e => e.type === 'tool_end')).toBe(true);
    });

    it('yields thinking event when response has text content alongside tool calls', async () => {
      let callCount = 0;
      mockLlmImpl = () => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve({
            response: {
              content: 'I should search for this.',
              tool_calls: [{ name: 'web_search', args: { query: 'test' } }],
            },
            usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10 },
          });
        }
        return Promise.resolve({
          response: 'Final answer',
          usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10 },
        });
      };

      const agent = await Agent.create({});
      const events = await collectEvents(agent.run('test'));
      expect(events.some(e => e.type === 'thinking')).toBe(true);
      const thinking = events.find(e => e.type === 'thinking');
      expect(thinking.message).toContain('search');
    });

    it('reaches max iterations and yields done with max iterations message', async () => {
      mockLlmImpl = () =>
        Promise.resolve({
          response: {
            content: '',
            tool_calls: [{ name: 'web_search', args: { query: 'endless' } }],
          },
          usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10 },
        });

      const agent = await Agent.create({ maxIterations: 2 });
      const events = await collectEvents(agent.run('test endless loop'));
      const done = events.find(e => e.type === 'done');
      expect(done).toBeDefined();
      expect(done.answer).toContain('maximum iterations');
    });

    it('stops agent loop when tool is denied', async () => {
      mockLlmImpl = () =>
        Promise.resolve({
          response: {
            content: '',
            tool_calls: [{ name: 'write_file', args: { path: '/some/file', content: 'data' } }],
          },
          usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10 },
        });

      const requestToolApproval = mock(() => Promise.resolve('deny'));
      const agent = await Agent.create({ requestToolApproval: requestToolApproval as any });
      const events = await collectEvents(agent.run('write something'));
      const done = events.find(e => e.type === 'done');
      expect(done).toBeDefined();
      // When denied, agent stops and yields a done event (possibly with empty answer)
      expect(events.some(e => e.type === 'tool_denied')).toBe(true);
    });

    it('done event includes toolCalls array', async () => {
      let callCount = 0;
      mockLlmImpl = () => {
        callCount++;
        if (callCount === 1) {
          return Promise.resolve({
            response: {
              content: '',
              tool_calls: [{ name: 'web_search', args: { query: 'AAPL' } }],
            },
            usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10 },
          });
        }
        return Promise.resolve({
          response: 'Result',
          usage: { inputTokens: 5, outputTokens: 5, totalTokens: 10 },
        });
      };

      const agent = await Agent.create({});
      const events = await collectEvents(agent.run('test'));
      const done = events.find(e => e.type === 'done');
      expect(Array.isArray(done.toolCalls)).toBe(true);
    });

    it('runs without history (no history context built)', async () => {
      const agent = await Agent.create({});
      const events = await collectEvents(agent.run('standalone query'));
      const done = events.find(e => e.type === 'done');
      expect(done).toBeDefined();
    });
  });
});
