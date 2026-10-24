/**
 * Agent-loop regression tests: prompt context across iterations, failed model
 * calls, forced-route reruns and exit messages. A scripted chat model records
 * every prompt it is sent and replays a queue of responses.
 */
import { FIXED_TEST_DATE, nextTestId } from '@/utils/test-determinism.js';
import { describe, it, expect, spyOn, beforeEach, afterEach, beforeAll, afterAll, setSystemTime } from 'bun:test';
import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { BaseChatModel, type BaseChatModelCallOptions, type BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { BaseLanguageModelInput } from '@langchain/core/language_models/base';
import { AIMessage, AIMessageChunk, type BaseMessage, type ToolCall } from '@langchain/core/messages';
import type { ChatResult } from '@langchain/core/outputs';
import { DynamicStructuredTool, type StructuredToolInterface } from '@langchain/core/tools';
import { z } from 'zod';
import { _setModelFactory } from '../model/llm.js';
import { HISTORY_CONTEXT_MARKER } from '../utils/parsing/history-context.js';
import type { AgentEvent, DoneEvent } from './types.js';

type Scripted = string | ToolCall[] | Error;

const script = {
  responses: [] as Scripted[],
  fallback: 'Final answer' as Scripted,
  prompts: [] as string[],
  streamPrompts: [] as string[],
  streamChunks: ['synthesized answer'] as string[],
  streamError: null as Error | null,
  onStream: null as (() => void) | null,
};

function lastText(messages: BaseMessage[]): string {
  const content = messages.at(-1)?.content;
  return typeof content === 'string' ? content : JSON.stringify(content);
}

class ScriptedChatModel extends BaseChatModel {
  _llmType(): string { return 'scripted'; }

  async _generate(messages: BaseMessage[]): Promise<ChatResult> {
    script.prompts.push(lastText(messages));
    const next = script.responses.length > 0 ? script.responses.shift()! : script.fallback;
    if (next instanceof Error) throw next;
    const message = typeof next === 'string'
      ? new AIMessage({ content: next })
      : new AIMessage({ content: '', tool_calls: next });
    return { generations: [{ message, text: typeof next === 'string' ? next : '' }] };
  }

  async *_streamIterator(input: BaseLanguageModelInput, _options?: BaseChatModelCallOptions): AsyncGenerator<AIMessageChunk> {
    script.streamPrompts.push(Array.isArray(input) ? lastText(input as BaseMessage[]) : String(input));
    script.onStream?.();
    if (script.streamError) throw script.streamError;
    for (const chunk of script.streamChunks) yield new AIMessageChunk({ content: chunk });
  }

  bindTools(_tools: BindToolsInput[]): this { return this; }
}

const ST_CALL: ToolCall = {
  id: 'st1',
  name: 'sequential_thinking',
  args: { thought: 'plan the research steps for this question', nextThoughtNeeded: false, thoughtNumber: 1, totalThoughts: 1 },
  type: 'tool_call',
};

// Keep runs off the network: Polymarket context injection fetches by default.
let fetchSpy: { mockRestore(): void } | undefined;
beforeAll(() => {
  _setModelFactory(() => new ScriptedChatModel({}));
  fetchSpy = spyOn(globalThis, 'fetch').mockImplementation((async () => Response.json([])) as unknown as typeof fetch);
});
afterAll(() => {
  _setModelFactory(null);
  fetchSpy?.mockRestore();
});

let tmpDir: string;
let originalCwd: string;
let prevOpenAiKey: string | undefined;

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
  tmpDir = join(tmpdir(), `agent-loop-test-${nextTestId('path')}`);
  mkdirSync(tmpDir, { recursive: true });
  originalCwd = process.cwd();
  process.chdir(tmpDir);
  prevOpenAiKey = process.env.OPENAI_API_KEY;
  if (!prevOpenAiKey) process.env.OPENAI_API_KEY = 'sk-test-stub';
  script.responses = [];
  script.fallback = 'Final answer';
  script.prompts = [];
  script.streamPrompts = [];
  script.streamChunks = ['synthesized answer'];
  script.streamError = null;
  script.onStream = null;
});

afterEach(() => {
  setSystemTime();
  process.chdir(originalCwd);
  rmSync(tmpDir, { recursive: true, force: true });
  if (!prevOpenAiKey) delete process.env.OPENAI_API_KEY;
  else process.env.OPENAI_API_KEY = prevOpenAiKey;
});

const { Agent } = await import('./agent.js');
const { InMemoryChatHistory } = await import('../utils/in-memory-chat-history.js');
const { AgentToolExecutor } = await import('./tool-executor.js');

type AgentInstance = Awaited<ReturnType<typeof Agent.create>>;

/** Replace (or add) agent tools with fixtures and rebuild its executor around them. */
function installTools(agent: AgentInstance, tools: Record<string, () => string>): void {
  // Throwing fixtures are recorded by the executor as failed calls.
  const toolMap = Object.getOwnPropertyDescriptor(agent, 'toolMap')?.value as Map<string, StructuredToolInterface>;
  for (const [name, invoke] of Object.entries(tools)) {
    toolMap.set(name, new DynamicStructuredTool({
      name,
      description: `${name} fixture`,
      schema: z.record(z.string(), z.unknown()),
      func: async () => invoke(),
    }));
  }
  Object.defineProperty(agent, 'toolExecutor', { value: new AgentToolExecutor(toolMap), configurable: true });
}

async function collect(gen: AsyncGenerator<AgentEvent>): Promise<AgentEvent[]> {
  const events: AgentEvent[] = [];
  for await (const event of gen) events.push(event);
  return events;
}

function doneOf(events: AgentEvent[]): DoneEvent {
  const done = events.find((event) => event.type === 'done');
  if (!done) throw new Error('run produced no done event');
  return done as DoneEvent;
}

function historyWithPriorTurn() {
  const history = new InMemoryChatHistory();
  history.seedMessage({
    query: 'Tell me about Zephyrcorp',
    answer: 'Zephyrcorp is a widget maker.',
    summary: null,
  });
  return history;
}

describe('iteration prompts keep the initial context', () => {
  it('keeps chat history in the prompt after a tool round', async () => {
    script.responses = [[ST_CALL], 'Margins compared'];
    const agent = await Agent.create({ maxIterations: 5, memoryEnabled: false });

    await collect(agent.run('compare its margins', historyWithPriorTurn()));

    expect(script.prompts).toHaveLength(2);
    expect(script.prompts[0]).toContain(HISTORY_CONTEXT_MARKER);
    expect(script.prompts[1]).toContain(HISTORY_CONTEXT_MARKER);
    expect(script.prompts[1]).toContain('Zephyrcorp');
  });

  it('keeps chat history in the max-iterations synthesis prompt', async () => {
    script.fallback = [ST_CALL];
    const agent = await Agent.create({ maxIterations: 1, memoryEnabled: false });

    await collect(agent.run('compare its margins', historyWithPriorTurn()));

    expect(script.streamPrompts).toHaveLength(1);
    expect(script.streamPrompts[0]).toContain('Zephyrcorp');
  });
});

describe('failed model calls', () => {
  it('flags the done event as an error', async () => {
    script.fallback = new Error('provider exploded');
    const agent = await Agent.create({ maxIterations: 3, memoryEnabled: false });

    const done = doneOf(await collect(agent.run('what is up')));

    expect(done.error).toBe(true);
    expect(done.answer).toStartWith('Error:');
  });

  it('leaves the error flag off a normal answer', async () => {
    const agent = await Agent.create({ maxIterations: 3, memoryEnabled: false });

    const done = doneOf(await collect(agent.run('what is up')));

    expect(done.error).toBeUndefined();
  });
});

describe('forced tool routes', () => {
  it('never re-forces a (tool, args) pair it already forced', async () => {
    const agent = await Agent.create({ maxIterations: 6, memoryEnabled: false });
    let markovRuns = 0;
    installTools(agent, {
      // Succeeds without a usable distribution, so the force predicate stays true.
      markov_distribution: () => {
        markovRuns++;
        return JSON.stringify({ data: { _tool: 'markov_distribution', status: 'pending' } });
      },
    });

    const done = doneOf(await collect(agent.run('What is the probability distribution for BTC-USD in 7 trading days?')));

    expect(markovRuns).toBe(1);
    expect(done.iterations).toBeLessThan(6);
    expect(done.answer).toContain('Final answer');
  });
});

describe('exit messages', () => {
  it('does not claim the iteration limit when the repeated-failure loop-breaker stops the run', async () => {
    const failing: ToolCall = { id: 'f1', name: 'lookup_fixture', args: { ticker: 'ZZZ' }, type: 'tool_call' };
    script.responses = [[ST_CALL, failing], [{ ...failing, id: 'f2' }]];
    const agent = await Agent.create({ maxIterations: 10, memoryEnabled: false });
    installTools(agent, { lookup_fixture: () => { throw new Error('unknown ticker'); } });

    await collect(agent.run('look up ZZZ'));

    expect(script.streamPrompts).toHaveLength(1);
    expect(script.streamPrompts[0]).not.toContain('maximum number of research steps');
    expect(script.streamPrompts[0]).toContain('repeated');
  });

  it('reports a user abort during final synthesis as a cancellation, not a timeout', async () => {
    const controller = new AbortController();
    script.fallback = [ST_CALL];
    script.onStream = () => controller.abort();
    script.streamError = Object.assign(new Error('aborted'), { name: 'AbortError' });
    const agent = await Agent.create({ maxIterations: 1, memoryEnabled: false, signal: controller.signal });

    const done = doneOf(await collect(agent.run('look up ZZZ')));

    expect(done.answer).not.toContain('timed out');
    expect(done.answer).toContain('cancelled');
  });

  it('still reports a synthesis timeout as a timeout', async () => {
    script.fallback = [ST_CALL];
    script.streamError = Object.assign(new Error('timed out'), { name: 'TimeoutError' });
    const agent = await Agent.create({ maxIterations: 1, memoryEnabled: false });

    const done = doneOf(await collect(agent.run('look up ZZZ')));

    expect(done.answer).toContain('Research interrupted');
  });
});
