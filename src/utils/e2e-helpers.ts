/**
 * E2E test helper — runs the full Cramer-Short agent against the configured Ollama
 * model and returns a structured result for assertion.
 *
 * Environment variables:
 *   E2E_MODEL      — Ollama model to use (default: 'ollama:kimi-k2.6:cloud')
 *   OLLAMA_BASE_URL — Ollama endpoint (default: 'http://127.0.0.1:11434')
 *   E2E_TIMEOUT_MS  — Hard timeout in ms (default: 600 000)
 */
import type { AgentEvent, DoneEvent } from '../agent/types.js';
import { InMemoryChatHistory } from './in-memory-chat-history.js';
import { isTimeoutError } from './errors.js';
import { withRetry } from './retry.js';

export const E2E_MODEL = process.env.E2E_MODEL ?? 'ollama:kimi-k2.6:cloud';
export const E2E_TIMEOUT_MS = parseInt(process.env.E2E_TIMEOUT_MS ?? '600000', 10);
export const CHILD_RESULT_MARKER = '__CRAMER_E2E_RESULT__';

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

export interface E2ESeedMessage {
  query: string;
  answer: string;
  summary: string | null;
}

interface E2EOptions {
  maxIterations?: number;
  model?: string;
  historySeed?: readonly E2ESeedMessage[];
}

interface E2ERetryOptions extends E2EOptions {
  retryAttempts?: number;
}

interface E2EChildPayload {
  query: string;
  opts: E2EOptions;
}

function encodeChildPayload(payload: E2EChildPayload): string {
  return Buffer.from(JSON.stringify(payload), 'utf8').toString('base64');
}

function decodeChildPayload(payload: string): E2EChildPayload {
  return JSON.parse(Buffer.from(payload, 'base64').toString('utf8')) as E2EChildPayload;
}

function buildChildErrorMessage(prefix: string, stdout: string, stderr: string): string {
  const trimmedStdout = stdout.trim();
  const trimmedStderr = stderr.trim();
  const sections = [prefix];

  if (trimmedStdout) {
    sections.push(`stdout:\n${trimmedStdout.slice(-4000)}`);
  }
  if (trimmedStderr) {
    sections.push(`stderr:\n${trimmedStderr.slice(-4000)}`);
  }

  return sections.join('\n\n');
}

function parseChildResult(stdout: string, stderr: string): E2EResult {
  const markerLine = stdout
    .split(/\r?\n/)
    .find((line) => line.startsWith(`${CHILD_RESULT_MARKER}:`));
  if (!markerLine) {
    throw new Error(buildChildErrorMessage('E2E child exited without a result payload.', stdout, stderr));
  }

  const encoded = markerLine.slice(CHILD_RESULT_MARKER.length + 1);
  return JSON.parse(Buffer.from(encoded, 'base64').toString('utf8')) as E2EResult;
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
export async function runAgentE2EInProcess(
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

  // Dynamic import to avoid mock.module contamination from unit test files
  // that load earlier in the same Bun worker. At this point afterAll hooks
  // from unit tests have already restored the real module exports.
  const { Agent: AgentCtor } = await import('../agent/agent.js');

  const agent = await AgentCtor.create({
    model,
    maxIterations,
    memoryEnabled: false, // keep E2E tests hermetic — no cross-test memory bleed
  });
  const inMemoryHistory = opts.historySeed?.length
    ? (() => {
        const history = new InMemoryChatHistory(model);
        for (const message of opts.historySeed) {
          history.seedMessage(message);
        }
        return history;
      })()
    : undefined;

  const events: AgentEvent[] = [];
  const toolsCalled: string[] = [];
  const start = Date.now();

  // Race the agent run against the hard wall-clock timeout
  const ac = new AbortController();
  const timer = setTimeout(() => ac.abort(), E2E_TIMEOUT_MS);

  try {
    for await (const event of agent.run(actualQuery, inMemoryHistory)) {
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
 * Run the Cramer-Short agent end-to-end in a fresh Bun subprocess so Bun's
 * file-scoped mock.module() pollution from unit tests cannot leak into E2E runs.
 */
export async function runAgentE2E(
  query: string,
  opts: E2EOptions = {},
): Promise<E2EResult> {
  if (process.env.CRAMER_E2E_CHILD === '1') {
    return runAgentE2EInProcess(query, opts);
  }

  const childScriptPath = new URL('./e2e-agent-child.ts', import.meta.url).pathname;
  const payload = encodeChildPayload({ query, opts });
  const proc = Bun.spawn({
    cmd: [process.execPath, 'run', childScriptPath, payload],
    cwd: process.cwd(),
    env: {
      ...process.env,
      CRAMER_E2E_CHILD: '1',
    },
    stdout: 'pipe',
    stderr: 'pipe',
  });

  const stdoutPromise = new Response(proc.stdout).text();
  const stderrPromise = new Response(proc.stderr).text();
  let timedOut = false;
  const timer = setTimeout(() => {
    timedOut = true;
    proc.kill();
  }, E2E_TIMEOUT_MS);

  const exitCode = await proc.exited;
  clearTimeout(timer);
  const [stdout, stderr] = await Promise.all([stdoutPromise, stderrPromise]);

  if (timedOut) {
    throw new Error(buildChildErrorMessage(
      `E2E child timed out after ${E2E_TIMEOUT_MS}ms.`,
      stdout,
      stderr,
    ));
  }

  if (exitCode !== 0) {
    throw new Error(buildChildErrorMessage(
      `E2E child exited with code ${exitCode}.`,
      stdout,
      stderr,
    ));
  }

  return parseChildResult(stdout, stderr);
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

export function readE2EChildPayloadFromArgv(): E2EChildPayload {
  const payload = process.argv[2];
  if (!payload) {
    throw new Error('Missing E2E child payload argument.');
  }
  return decodeChildPayload(payload);
}
