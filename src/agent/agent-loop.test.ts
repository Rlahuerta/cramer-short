/**
 * Agent-loop regression tests: prompt context across iterations, failed model
 * calls, forced-route reruns and exit messages. A scripted chat model records
 * every prompt it is sent and replays a queue of responses.
 */
import { FIXED_TEST_DATE, nextTestId } from '@/utils/test-determinism.js';
import { describe, it, expect, spyOn, beforeEach, afterEach, beforeAll, afterAll, setSystemTime } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { BaseChatModel, type BaseChatModelCallOptions, type BindToolsInput } from '@langchain/core/language_models/chat_models';
import type { BaseLanguageModelInput } from '@langchain/core/language_models/base';
import { AIMessage, AIMessageChunk, type BaseMessage, type ToolCall } from '@langchain/core/messages';
import type { ChatResult } from '@langchain/core/outputs';
import { DynamicStructuredTool, type StructuredToolInterface } from '@langchain/core/tools';
import { z } from 'zod';
import { _setModelFactory } from '../model/llm.js';
import * as config from '../utils/config.js';
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
  tmpDir = mkdtempSync(join(tmpdir(), 'agent-loop-test-'));
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

  it('synthesizes from the research gathered so far instead of discarding it', async () => {
    const research: ToolCall = { id: 'r1', name: 'research_fixture', args: {}, type: 'tool_call' };
    script.responses = [[ST_CALL], [research], new Error('Ollama call timed out after 120s. The model may be slow or unavailable.')];
    const agent = await Agent.create({ maxIterations: 10, memoryEnabled: false });
    installTools(agent, { research_fixture: () => 'AAPL revenue FY2025 = $420B; gross margin 46.9%' });

    const done = doneOf(await collect(agent.run('Summarize Apple fundamentals')));

    expect(done.answer).not.toStartWith('Error:');
    expect(done.answer).toContain('synthesized answer');
    expect(script.streamPrompts[0]).toContain('$420B');
  });

  it('still ends with the error when the user aborted', async () => {
    const controller = new AbortController();
    const research: ToolCall = { id: 'r1', name: 'research_fixture', args: {}, type: 'tool_call' };
    script.responses = [[ST_CALL], [research], Object.assign(new Error('aborted'), { name: 'AbortError' })];
    const agent = await Agent.create({ maxIterations: 10, memoryEnabled: false, signal: controller.signal });
    installTools(agent, { research_fixture: () => { controller.abort(); return 'AAPL revenue FY2025 = $420B'; } });

    const done = doneOf(await collect(agent.run('Summarize Apple fundamentals')));

    expect(done.error).toBe(true);
    expect(script.streamPrompts).toHaveLength(0);
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

  it('stops when the model keeps re-requesting a skill that already ran', async () => {
    const skillCall: ToolCall = { id: 'k1', name: 'skill', args: { skill: 'dcf' }, type: 'tool_call' };
    script.responses = [[ST_CALL, skillCall]];
    script.fallback = [{ ...skillCall, id: 'k2' }];
    const agent = await Agent.create({ maxIterations: 10, memoryEnabled: false });
    let skillRuns = 0;
    installTools(agent, { skill: () => { skillRuns++; return 'Skill dcf instructions: step 1 ... step 9'; } });

    const done = doneOf(await collect(agent.run('value AAPL with a DCF')));

    expect(skillRuns).toBe(1);
    expect(script.prompts).toHaveLength(2);
    expect(done.iterations).toBe(2);
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

  it('asks once more when the model ends with an empty answer', async () => {
    script.responses = [[ST_CALL], ''];
    let streams = 0;
    script.onStream = () => {
      streams++;
      script.streamChunks = streams === 1 ? [] : ['recovered answer'];
    };
    const agent = await Agent.create({ maxIterations: 5, memoryEnabled: false });

    const done = doneOf(await collect(agent.run('look up ZZZ')));

    expect(streams).toBe(2);
    expect(done.answer).toContain('recovered answer');
  });

  it('returns the gathered research instead of a blank answer when the model stays empty', async () => {
    script.responses = [[ST_CALL], ''];
    script.streamChunks = [];
    const agent = await Agent.create({ maxIterations: 5, memoryEnabled: false });

    const done = doneOf(await collect(agent.run('look up ZZZ')));

    expect(done.answer).toContain('Research interrupted');
    expect(done.answer).toContain('empty answer');
  });

  it('flags the interrupted fallback as an error so callers can tell it from an answer', async () => {
    script.fallback = [ST_CALL];
    script.streamError = new Error('provider exploded');
    const agent = await Agent.create({ maxIterations: 1, memoryEnabled: false });

    const done = doneOf(await collect(agent.run('look up ZZZ')));

    expect(done.answer).toContain('Research interrupted');
    expect(done.error).toBe(true);
  });

  it('still reports a synthesis timeout as a timeout', async () => {
    script.fallback = [ST_CALL];
    script.streamError = Object.assign(new Error('timed out'), { name: 'TimeoutError' });
    const agent = await Agent.create({ maxIterations: 1, memoryEnabled: false });

    const done = doneOf(await collect(agent.run('look up ZZZ')));

    expect(done.answer).toContain('Research interrupted');
  });
});

describe('distribution warning prefix', () => {
  // Without a markov_distribution tool the Markov route fails, so the answer gets the warning prefix.
  const DISTRIBUTION_QUERY = 'Give me the probability distribution of AAPL price in 30 days';
  const noop = new DynamicStructuredTool({ name: 'noop', description: 'noop', schema: z.object({}), func: async () => 'ok' });

  it('still synthesizes an answer after the warning when the model returns an empty answer', async () => {
    script.fallback = '';
    const agent = await Agent.create({ tools: [noop], memoryEnabled: false, toolDescriptionsOverride: 'noop' });

    const done = doneOf(await collect(agent.run(DISTRIBUTION_QUERY)));

    expect(done.answer).toStartWith('## Warning');
    expect(done.answer).toContain('synthesized answer');
  });

  it('still synthesizes an answer after the warning at max iterations', async () => {
    script.fallback = [ST_CALL];
    const agent = await Agent.create({ tools: [noop], memoryEnabled: false, maxIterations: 1, toolDescriptionsOverride: 'noop' });

    const done = doneOf(await collect(agent.run(DISTRIBUTION_QUERY)));

    expect(done.answer).toStartWith('## Warning');
    expect(done.answer).toContain('synthesized answer');
  });
});

describe('forecast ticker auto-fix on follow-ups', () => {
  it('runs the earlier asset\'s forecast as the model asked when the follow-up names another asset', async () => {
    const history = new InMemoryChatHistory();
    history.seedMessage({ query: 'BTC 7 day forecast', answer: 'BTC-USD forecast: ...', summary: null });
    const markovCall: ToolCall = { id: 'm1', name: 'markov_distribution', args: { ticker: 'BTC-USD', horizon: 7 }, type: 'tool_call' };
    script.responses = [[ST_CALL, markovCall], 'Compared'];
    const agent = await Agent.create({ maxIterations: 5, memoryEnabled: false });
    const toolMap = Object.getOwnPropertyDescriptor(agent, 'toolMap')?.value as Map<string, StructuredToolInterface>;
    const received: unknown[] = [];
    toolMap.set('markov_distribution', new DynamicStructuredTool({
      name: 'markov_distribution',
      description: 'markov fixture',
      schema: z.object({ ticker: z.string(), horizon: z.number() }),
      func: async ({ ticker }) => { received.push(ticker); return JSON.stringify({ data: { ran_for: ticker } }); },
    }));
    Object.defineProperty(agent, 'toolExecutor', { value: new AgentToolExecutor(toolMap), configurable: true });

    await collect(agent.run('Compare that with gold', history));

    expect(received).toEqual(['BTC-USD']);
  });
});

describe('Agent — parallelToolLimit wiring', () => {
  function executorLimit(agent: AgentInstance): unknown {
    const executor = Object.getOwnPropertyDescriptor(agent, 'toolExecutor')?.value as object;
    return Object.getOwnPropertyDescriptor(executor, 'parallelToolLimit')?.value;
  }

  it('reads the parallelToolLimit setting when the config leaves it unset', async () => {
    const realGetSetting = config.getSetting;
    const spy = spyOn(config, 'getSetting').mockImplementation(((key: string, fallback: unknown) =>
      key === 'parallelToolLimit' ? 2 : realGetSetting(key, fallback)) as typeof config.getSetting);
    try {
      const agent = await Agent.create({ memoryEnabled: false });
      expect(executorLimit(agent)).toBe(2);
    } finally {
      spy.mockRestore();
    }
  });

  it('lets AgentConfig.parallelToolLimit override the setting', async () => {
    const agent = await Agent.create({ memoryEnabled: false, parallelToolLimit: 3 });
    expect(executorLimit(agent)).toBe(3);
  });
});

describe('web_search fallback note after a tool error', () => {
  const NOTE = 'you MUST call web_search next';
  const call = (name: string, id: string): ToolCall => ({ id, name, args: { query: 'ZZZ revenue' }, type: 'tool_call' });

  async function promptAfterFailedTool(opts: { webSearch: 'missing' | 'available' | 'called' }): Promise<string> {
    const agent = await Agent.create({ maxIterations: 4, memoryEnabled: false });
    const toolMap = Object.getOwnPropertyDescriptor(agent, 'toolMap')?.value as Map<string, StructuredToolInterface>;
    toolMap.delete('web_search');
    installTools(agent, {
      zz_data: () => '{"error": "no data for ZZZ"}',
      ...(opts.webSearch === 'missing' ? {} : { web_search: () => '{"results": []}' }),
    });
    script.responses = [
      [ST_CALL],
      opts.webSearch === 'called' ? [call('zz_data', 'z1'), call('web_search', 'w1')] : [call('zz_data', 'z1')],
      'Final answer',
    ];

    await collect(agent.run('look up ZZZ'));

    expect(script.prompts.length).toBeGreaterThanOrEqual(3);
    expect(script.prompts[2]).toContain('no data for ZZZ');
    return script.prompts[2];
  }

  it('asks for web_search when it is registered and has not run', async () => {
    expect(await promptAfterFailedTool({ webSearch: 'available' })).toContain(NOTE);
  });

  it('leaves the note out when web_search is not registered', async () => {
    expect(await promptAfterFailedTool({ webSearch: 'missing' })).not.toContain(NOTE);
  });

  it('leaves the note out once web_search has run', async () => {
    expect(await promptAfterFailedTool({ webSearch: 'called' })).not.toContain(NOTE);
  });
});
