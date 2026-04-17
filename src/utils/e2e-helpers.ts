/**
 * E2E test helper — runs the full Cramer-Short agent against the configured Ollama
 * model and returns a structured result for assertion.
 *
 * Environment variables:
 *   E2E_MODEL      — Ollama model to use (default: 'ollama:minimax-m2.7:cloud')
 *   OLLAMA_BASE_URL — Ollama endpoint (default: 'http://127.0.0.1:11434')
 *   E2E_TIMEOUT_MS  — Hard timeout in ms (default: 360 000)
 */
import { Agent } from '../agent/agent.js';
import type { AgentEvent, DoneEvent } from '../agent/types.js';
import { isTimeoutError } from './errors.js';
import { withRetry } from './retry.js';

export const E2E_MODEL = process.env.E2E_MODEL ?? 'ollama:minimax-m2.7:cloud';
export const E2E_TIMEOUT_MS = parseInt(process.env.E2E_TIMEOUT_MS ?? '360000', 10);

export interface E2EResult {
  /** Full final answer text from the done event */
  answer: string;
  /** Ordered list of tool names that were called (tool_start events) */
  toolsCalled: string[];
  /** All raw events emitted by the agent */
  events: AgentEvent[];
  /** Wall-clock duration in ms */
  durationMs: number;
  /** Number of agent iterations */
  iterations: number;
}

interface E2EOptions {
  maxIterations?: number;
  model?: string;
}

interface E2ERetryOptions extends E2EOptions {
  retryAttempts?: number;
}

/**
 * Run the Cramer-Short agent end-to-end with the E2E model.
 *
 * Handles the `--deep` CLI flag: strips it from the query string and maps it
 * to `maxIterations: 40` (in the CLI it is a process.argv flag, not part of
 * the query, so passing it literally confuses the LLM).
 *
 * Throws if no `done` event is received within E2E_TIMEOUT_MS.
 */
export async function runAgentE2E(
  query: string,
  opts: E2EOptions = {},
): Promise<E2EResult> {
  // Strip --deep CLI flag and boost iteration budget accordingly
  let actualQuery = query;
  let defaultMaxIter = 10;
  if (query.trimStart().startsWith('--deep ')) {
    actualQuery = query.trimStart().slice('--deep '.length);
    defaultMaxIter = 40;
  }

  const model = opts.model ?? E2E_MODEL;
  const maxIterations = opts.maxIterations ?? defaultMaxIter;

  const agent = await Agent.create({
    model,
    maxIterations,
    memoryEnabled: false, // keep E2E tests hermetic — no cross-test memory bleed
  });

  const events: AgentEvent[] = [];
  const toolsCalled: string[] = [];
  const start = Date.now();

  // Race the agent run against the hard wall-clock timeout
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), E2E_TIMEOUT_MS);

  try {
    for await (const event of agent.run(actualQuery)) {
      if (ac.signal.aborted) break;
      events.push(event);
      if (event.type === 'tool_start') toolsCalled.push(event.tool);
    }
  } finally {
    clearTimeout(timer);
  }

  const doneEvent = events.find((e): e is DoneEvent => e.type === 'done');
  if (!doneEvent) {
    throw new Error(
      `E2E: no 'done' event received after ${Date.now() - start}ms. ` +
        `Tools called: [${toolsCalled.join(', ')}]. Events: [${events.map((e) => e.type).join(', ')}]`,
    );
  }

  return {
    answer: doneEvent.answer,
    toolsCalled,
    events,
    durationMs: Date.now() - start,
    iterations: doneEvent.iterations,
  };
}

/**
 * Retry whole-agent E2E runs when the final answer is just a timeout error.
 * This stabilizes E2E suites that intermittently hit provider/model latency
 * spikes under full-suite load.
 */
export async function runAgentE2EWithTimeoutRetry(
  query: string,
  opts: E2ERetryOptions = {},
): Promise<E2EResult> {
  const { retryAttempts = 2, ...runOpts } = opts;

  return withRetry(async () => {
    const result = await runAgentE2E(query, runOpts);
    if (isTimeoutError(result.answer)) {
      throw new Error(result.answer);
    }
    return result;
  }, {
    maxAttempts: retryAttempts,
    baseDelayMs: 2_000,
    maxDelayMs: 5_000,
    jitter: 0,
    shouldRetry: (error) => error instanceof Error && isTimeoutError(error.message),
  });
}
