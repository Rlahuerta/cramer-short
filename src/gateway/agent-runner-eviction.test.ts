/**
 * Tests for session LRU eviction in gateway/agent-runner.ts.
 * Stubs AgentRunnerController methods to avoid real LLM / API calls.
 */

import { describe, it, expect, spyOn, beforeEach, afterAll } from 'bun:test';
import { AgentRunnerController } from '../controllers/agent-runner.js';

// Stub the controller's runtime methods on the prototype (restored afterAll) instead of
// mock.module, which would replace the controller module for every later test file.
const controllerSpies = [
  spyOn(AgentRunnerController.prototype, 'setModel').mockImplementation(() => {}),
  spyOn(AgentRunnerController.prototype, 'setRunOptions').mockImplementation(() => {}),
  spyOn(AgentRunnerController.prototype, 'cancelExecution').mockImplementation(() => {}),
  spyOn(AgentRunnerController.prototype, 'runQuery').mockImplementation(async () => ({ answer: 'ok' }) as never),
];
afterAll(() => {
  for (const spy of controllerSpies) spy.mockRestore();
});

const {
  runAgentForMessage,
  getSessionCount,
  _clearSessionsForTest,
  _addSessionForTest,
  _hasSessionForTest,
  MAX_SESSIONS,
} = await import('./agent-runner.js');

beforeEach(() => {
  _clearSessionsForTest();
});

const baseReq = {
  query: 'test',
  model: 'gpt-4o',
  modelProvider: 'openai',
};

describe('sessions LRU eviction', () => {
  it('creates a session on first call and increments count', async () => {
    await runAgentForMessage({ ...baseReq, sessionKey: 'alice' });
    expect(getSessionCount()).toBe(1);
  });

  it('reuses the same session on repeated calls for the same key', async () => {
    await runAgentForMessage({ ...baseReq, sessionKey: 'alice' });
    await runAgentForMessage({ ...baseReq, sessionKey: 'alice' });
    expect(getSessionCount()).toBe(1);
  });

  it('evicts the least recently used session when MAX_SESSIONS is exceeded', async () => {
    for (let i = 0; i < MAX_SESSIONS; i++) {
      _addSessionForTest(`pre-existing-${i}`);
    }
    expect(getSessionCount()).toBe(MAX_SESSIONS);

    await runAgentForMessage({ ...baseReq, sessionKey: 'new-session' });

    expect(getSessionCount()).toBe(MAX_SESSIONS);
    expect(_hasSessionForTest('pre-existing-0')).toBe(false);
    expect(_hasSessionForTest('pre-existing-1')).toBe(true);
    expect(_hasSessionForTest('new-session')).toBe(true);
  });

  it('refreshes an existing real session so a later overflow evicts the next least-recently-used key', async () => {
    await runAgentForMessage({ ...baseReq, sessionKey: 'alice' });
    for (let i = 0; i < MAX_SESSIONS - 1; i++) {
      _addSessionForTest(`pre-existing-${i}`);
    }

    await runAgentForMessage({ ...baseReq, sessionKey: 'alice' });
    _addSessionForTest('overflow');

    expect(getSessionCount()).toBe(MAX_SESSIONS);
    expect(_hasSessionForTest('alice')).toBe(true);
    expect(_hasSessionForTest('pre-existing-0')).toBe(false);
    expect(_hasSessionForTest('overflow')).toBe(true);
  });
});
