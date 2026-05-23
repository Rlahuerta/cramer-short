/**
 * TDD tests for ProgressEvent emission from the Agent loop (Feature 5).
 *
 * Verifies that:
 * - A `progress` event is emitted at the start of each iteration
 * - `iteration` increments from 1 each loop
 * - `maxIterations` reflects the configured maximum
 * - Iterates do not exceed maxIterations
 */
import { FIXED_TEST_DATE, FIXED_TEST_NOW_MS, deterministicRandom, nextTestId } from '@/utils/test-determinism.js';
import { describe, it, expect, mock, beforeEach, afterEach, afterAll, beforeAll, setSystemTime } from 'bun:test';
import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import { tmpdir } from 'node:os';
import { BaseChatModel } from '@langchain/core/language_models/chat_models';
import type { BaseMessage } from '@langchain/core/messages';
import { AIMessage } from '@langchain/core/messages';
import { _setModelFactory } from '../model/llm.js';

// ---------------------------------------------------------------------------
// Isolation: each test gets its own tmp dir for Scratchpad persistence
// ---------------------------------------------------------------------------
let tmpDir: string;
let originalCwd: string;

beforeEach(() => {
  tmpDir = join(tmpdir(), `agent-progress-test-${nextTestId('path')}`);
  mkdirSync(tmpDir, { recursive: true });
  originalCwd = process.cwd();
  process.chdir(tmpDir);
});

afterEach(() => {
  process.chdir(originalCwd);
  rmSync(tmpDir, { recursive: true, force: true });
});

// ---------------------------------------------------------------------------
// Scoped model factory DI — avoids permanent mock.module contamination that
// breaks E2E/integration tests when llm.js has already been cached.
// ---------------------------------------------------------------------------

class SpyChatModel extends BaseChatModel {
  constructor() { super({}); }
  _llmType(): string { return 'spy-progress'; }
  async _generate(_messages: BaseMessage[], _options: any, _runManager?: any) {
    return { generations: [{ message: new AIMessage({ content: 'done', additional_kwargs: {} }), text: 'done' }] };
  }
  bindTools() { return this as any; }
}

beforeAll(() => { _setModelFactory(() => new SpyChatModel()); });
afterAll(() => { _setModelFactory(null); });

const { Agent } = await import('./agent.js');

import type { ProgressEvent, AgentEvent } from './types.js';

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
});

afterEach(() => {
  setSystemTime();
});

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

async function collectEvents(gen: AsyncGenerator<AgentEvent>): Promise<AgentEvent[]> {
  const events: AgentEvent[] = [];
  for await (const e of gen) events.push(e);
  return events;
}

// ---------------------------------------------------------------------------
// ProgressEvent tests
// ---------------------------------------------------------------------------
describe('Agent — ProgressEvent emission', () => {
  it('emits a progress event at the start of each iteration', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3 });
    const events = await collectEvents(agent.run('What is 1 + 1?'));

    const progressEvents = events.filter((e) => e.type === 'progress') as ProgressEvent[];
    expect(progressEvents.length).toBeGreaterThan(0);
  });

  it('progress event contains the current iteration number starting at 1', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 3 });
    const events = await collectEvents(agent.run('Simple test query'));

    const progressEvents = events.filter((e) => e.type === 'progress') as ProgressEvent[];
    expect(progressEvents[0]?.iteration).toBe(1);
  });

  it('progress event iteration increments each loop', async () => {
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations: 5 });
    const events = await collectEvents(agent.run('Query that may need multiple iterations'));

    const progressEvents = events.filter((e) => e.type === 'progress') as ProgressEvent[];
    for (let i = 0; i < progressEvents.length; i++) {
      expect(progressEvents[i].iteration).toBe(i + 1);
    }
  });

  it('progress event contains maxIterations from config', async () => {
    const maxIterations = 7;
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations });
    const events = await collectEvents(agent.run('Test query'));

    const progressEvents = events.filter((e) => e.type === 'progress') as ProgressEvent[];
    expect(progressEvents.length).toBeGreaterThan(0);
    expect(progressEvents[0]?.maxIterations).toBe(maxIterations);
  });

  it('does not emit more progress events than maxIterations', async () => {
    const maxIterations = 4;
    const agent = await Agent.create({ model: 'gpt-5.4', maxIterations });
    const events = await collectEvents(agent.run('Test bounded query'));

    const progressEvents = events.filter((e) => e.type === 'progress') as ProgressEvent[];
    expect(progressEvents.length).toBeLessThanOrEqual(maxIterations);
  });
});
