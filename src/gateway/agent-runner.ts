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

const sessions = new Map<string, SessionState>();

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
  if (existing) {
    return existing;
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
  sessions.set(sessionKey, created);
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
