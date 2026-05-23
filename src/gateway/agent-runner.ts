import { AgentRunnerController } from '../controllers/agent-runner.js';
import { InMemoryChatHistory } from '../utils/in-memory-chat-history.js';
import { HEARTBEAT_OK_TOKEN } from './heartbeat/suppression.js';
import type { AgentEvent } from '../agent/types.js';
import type { GroupContext } from '../agent/prompts.js';

type EventForwarder = {
  forward(event: AgentEvent): Promise<void>;
  setHandler(handler: ((event: AgentEvent) => void | Promise<void>) | undefined): Promise<void>;
  clearHandler(): void;
};

type SessionState = {
  controller: AgentRunnerController;
  inMemoryChatHistory: InMemoryChatHistory;
  tail: Promise<void>;
  eventForwarder: EventForwarder;
};

type SessionEntry =
  | { kind: 'active'; session: SessionState }
  | { kind: 'test-placeholder' };

/** Maximum number of concurrent sessions to keep in memory. LRU eviction on overflow. */
const MAX_SESSIONS = 50;

const sessions = new Map<string, SessionEntry>();

function setBoundedSessionEntry(sessionKey: string, entry: SessionEntry): void {
  if (sessions.has(sessionKey)) {
    sessions.delete(sessionKey);
  } else if (sessions.size >= MAX_SESSIONS) {
    const oldest = sessions.keys().next();
    if (!oldest.done) sessions.delete(oldest.value);
  }
  sessions.set(sessionKey, entry);
}

function setBoundedSession(sessionKey: string, session: SessionState): void {
  setBoundedSessionEntry(sessionKey, { kind: 'active', session });
}

/** Exposed for observability and testing only. */
export function getSessionCount(): number {
  return sessions.size;
}

/** @internal Test-only: inject a fake session entry to drive eviction tests. */
export function _addSessionForTest(key: string): void {
  setBoundedSessionEntry(key, { kind: 'test-placeholder' });
}

/** @internal Test-only: inspect session-key membership without exposing the Map. */
export function _hasSessionForTest(key: string): boolean {
  return sessions.has(key);
}

/** @internal Test-only: clear all sessions for test isolation. */
export function _clearSessionsForTest(): void {
  sessions.clear();
}

/** @internal Test-only: expose MAX_SESSIONS for assertions. */
export { MAX_SESSIONS };

export class BufferedEventForwarder implements EventForwarder {
  private handler: ((event: AgentEvent) => void | Promise<void>) | undefined;
  private bufferedEvents: AgentEvent[] = [];
  private hasAttachedHandler = false;

  async forward(event: AgentEvent): Promise<void> {
    if (this.handler) {
      await this.handler(event);
      return;
    }

    if (!this.hasAttachedHandler) {
      this.bufferedEvents.push(event);
    }
  }

  async setHandler(handler: ((event: AgentEvent) => void | Promise<void>) | undefined): Promise<void> {
    this.hasAttachedHandler = true;
    this.handler = handler;

    if (!handler) {
      this.bufferedEvents = [];
      return;
    }

    const bufferedEvents = this.bufferedEvents;
    this.bufferedEvents = [];
    for (const event of bufferedEvents) {
      await handler(event);
    }
  }

  clearHandler(): void {
    this.handler = undefined;
    this.bufferedEvents = [];
  }
}

function getSession(sessionKey: string, model: string, modelProvider: string): SessionState {
  const existing = sessions.get(sessionKey);
  if (existing?.kind === 'active') {
    sessions.delete(sessionKey);
    sessions.set(sessionKey, existing);
    return existing.session;
  }

  const inMemoryChatHistory = new InMemoryChatHistory(model);
  const eventForwarder = new BufferedEventForwarder();
  const controller = new AgentRunnerController(
    { model, modelProvider },
    inMemoryChatHistory,
    undefined,
    { onEvent: (event) => eventForwarder.forward(event) },
  );

  const created: SessionState = {
    controller,
    inMemoryChatHistory,
    tail: Promise.resolve(),
    eventForwarder,
  };
  setBoundedSession(sessionKey, created);
  return created;
}

export type AgentRunRequest = {
  sessionKey: string;
  query: string;
  model: string;
  modelProvider: string;
  maxIterations?: number;
  signal?: AbortSignal;
  onEvent?: (event: AgentEvent) => void | Promise<void>;
  isHeartbeat?: boolean;
  channel?: string;
  groupContext?: GroupContext;
};

export async function runAgentForMessage(req: AgentRunRequest): Promise<string> {
  const session = getSession(req.sessionKey, req.model, req.modelProvider);
  let finalAnswer = '';

  const run = async () => {
    session.controller.setModel(req.model, req.modelProvider);
    session.controller.setRunOptions({
      maxIterations: req.maxIterations,
      channel: req.channel,
      groupContext: req.groupContext,
    });

    const abort = () => session.controller.cancelExecution();
    if (req.signal?.aborted) {
      session.eventForwarder.clearHandler();
      return;
    }
    req.signal?.addEventListener('abort', abort, { once: true });

    try {
      await session.eventForwarder.setHandler(req.onEvent);
      const result = await session.controller.runQuery(req.query);
      if (result?.answer) {
        finalAnswer = result.answer;
      }

      // Prune HEARTBEAT_OK turns to avoid context pollution.
      if (req.isHeartbeat && finalAnswer.trim().toUpperCase().includes(HEARTBEAT_OK_TOKEN)) {
        session.inMemoryChatHistory.pruneLastTurn();
      }
    } finally {
      session.eventForwarder.clearHandler();
      req.signal?.removeEventListener('abort', abort);
    }
  };

  // Serialize per-session turns while allowing cross-session concurrency.
  session.tail = session.tail.then(run, run);
  await session.tail;
  return finalAnswer;
}
