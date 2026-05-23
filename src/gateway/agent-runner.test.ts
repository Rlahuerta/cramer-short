import { describe, expect, it, mock, beforeEach } from 'bun:test';
import {
  BufferedEventForwarder,
  getSessionCount,
  _addSessionForTest,
  _clearSessionsForTest,
  _hasSessionForTest,
  MAX_SESSIONS,
} from './agent-runner.js';
import type { AgentEvent } from '../agent/types.js';

beforeEach(() => {
  _clearSessionsForTest();
});

describe('BufferedEventForwarder', () => {
  it('delivers events emitted before the request handler is attached', async () => {
    const forwarder = new BufferedEventForwarder();
    const earlyEvent = { type: 'thinking', message: 'initializing' } satisfies AgentEvent;
    const onEvent = mock(async (_event: AgentEvent) => {});

    await forwarder.forward(earlyEvent);
    await forwarder.setHandler(onEvent);

    expect(onEvent).toHaveBeenCalledTimes(1);
    expect(onEvent).toHaveBeenCalledWith(earlyEvent);
  });

  it('does not deliver inactive-session events to the next request handler', async () => {
    const forwarder = new BufferedEventForwarder();
    const firstHandler = mock(async (_event: AgentEvent) => {});
    const secondHandler = mock(async (_event: AgentEvent) => {});

    await forwarder.setHandler(firstHandler);
    forwarder.clearHandler();
    await forwarder.forward({ type: 'thinking', message: 'late event' });
    await forwarder.setHandler(secondHandler);

    expect(secondHandler).not.toHaveBeenCalled();
  });
});

describe('sessions Map — bounded LRU eviction', () => {
  it('getSessionCount() returns 0 after clear', () => {
    expect(getSessionCount()).toBe(0);
  });

  it('_addSessionForTest fills and getSessionCount tracks accurately', () => {
    for (let i = 0; i < 5; i++) _addSessionForTest(`s${i}`);
    expect(getSessionCount()).toBe(5);
  });

  it('MAX_SESSIONS constant is 50', () => {
    expect(MAX_SESSIONS).toBe(50);
  });

  it('evicts the least recently used session when the test insertion path exceeds MAX_SESSIONS', () => {
    for (let i = 0; i < MAX_SESSIONS; i++) _addSessionForTest(`session-${i}`);
    expect(getSessionCount()).toBe(MAX_SESSIONS);

    _addSessionForTest('overflow');

    expect(getSessionCount()).toBe(MAX_SESSIONS);
    expect(_hasSessionForTest('session-0')).toBe(false);
    expect(_hasSessionForTest('session-1')).toBe(true);
    expect(_hasSessionForTest('overflow')).toBe(true);
  });

  it('refreshes recency when an existing test session is inserted again', () => {
    for (let i = 0; i < MAX_SESSIONS; i++) _addSessionForTest(`session-${i}`);

    _addSessionForTest('session-0');
    _addSessionForTest('overflow');

    expect(getSessionCount()).toBe(MAX_SESSIONS);
    expect(_hasSessionForTest('session-0')).toBe(true);
    expect(_hasSessionForTest('session-1')).toBe(false);
  });
});
