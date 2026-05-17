/**
 * E2E test helper — runs the full Cramer-Short agent against the configured live
 * model and returns a structured result for assertion.
 *
 * Environment variables:
 *   E2E_MODEL      — model to use (default: auto-resolved Ollama cloud model)
 *   OLLAMA_BASE_URL — Ollama endpoint (default: 'http://127.0.0.1:11434')
 *   E2E_TIMEOUT_MS  — Hard timeout in ms (default: 600 000)
 */
import { mkdirSync, rmSync } from 'node:fs';
import { join } from 'node:path';
import type { AgentEvent, DoneEvent } from '../shared/agent-events.js';
import { loadAgentCtor } from '../shared/agent-loader.js';
import { getEnv, getEnvironment, getEnvOrDefault } from './env.js';
import {
  isAuthError,
  isBillingError,
  isOverloadedError,
  isRateLimitError,
  isTimeoutError,
} from './errors.js';
import { logger } from './logger.js';
import { withRetry } from './retry.js';
import { resolveProvider } from '../providers.js';

const DEFAULT_E2E_TIMEOUT_MS = 600_000;

function resolveE2ETimeoutMs(): number {
  const raw = getEnv('E2E_TIMEOUT_MS')?.trim();
  if (!raw) {
    return DEFAULT_E2E_TIMEOUT_MS;
  }

  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed >= 40_000 && parsed <= 900_000
    ? parsed
    : DEFAULT_E2E_TIMEOUT_MS;
}

export const E2E_TIMEOUT_MS = resolveE2ETimeoutMs();
export const CHILD_RESULT_MARKER = '__CRAMER_E2E_RESULT__';

const E2E_RESULT_DIR = join(process.cwd(), '.cramer-short', 'e2e-results');
const E2E_CHILD_TIMEOUT_MS = Math.max(30_000, E2E_TIMEOUT_MS - 10_000);
const DEFAULT_E2E_PREFLIGHT_TIMEOUT_MS = 30_000;
const E2E_MODEL_FALLBACK = 'ollama:glm-5:cloud';
const E2E_MODEL_PREFERENCES = [
  'glm-5.1:cloud',
  'minimax-m2.7:cloud',
  'glm-5:cloud',
  'kimi-k2.6:cloud',
  'qwen3.5:397b-cloud',
  'qwen3-next:80b-cloud',
] as const;

let resolvedDefaultE2EModelPromise: Promise<string> | null = null;
const preflightPromises = new Map<string, Promise<E2EPreflightStatus>>();
let dynamicSkipReason: string | null = null;

export class E2EPreflightSkipError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'E2EPreflightSkipError';
  }
}

export function isE2EPreflightSkipError(error: unknown): error is E2EPreflightSkipError {
  return error instanceof E2EPreflightSkipError || (
    error instanceof Error && error.name === 'E2EPreflightSkipError'
  );
}

export function markE2ESkippedFromError(error: unknown): boolean {
  if (!isE2EPreflightSkipError(error)) {
    return false;
  }
  dynamicSkipReason = error.message;
  console.warn(error.message);
  return true;
}

export function getE2EDynamicSkipReason(): string | null {
  return dynamicSkipReason;
}

export interface E2EPreflightStatus {
  available: boolean;
  model: string;
  reason?: string;
}

function resolveE2EPreflightTimeoutMs(): number {
  const raw = getEnv('E2E_PREFLIGHT_TIMEOUT_MS')?.trim();
  if (!raw) {
    return DEFAULT_E2E_PREFLIGHT_TIMEOUT_MS;
  }

  const parsed = Number(raw);
  return Number.isFinite(parsed) && parsed >= 1_000 && parsed <= 120_000
    ? parsed
    : DEFAULT_E2E_PREFLIGHT_TIMEOUT_MS;
}

function normalizeOllamaModel(model: string): string {
  return model.replace(/^ollama:/i, '');
}

function isOllamaModel(model: string): boolean {
  return model.toLowerCase().startsWith('ollama:');
}

function stringifyError(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function isE2EExternalModelError(error: unknown): boolean {
  const message = stringifyError(error);
  if (!message || /E2E: no 'done' event received/i.test(message)) {
    return false;
  }

  const lower = message.toLowerCase();
  return isTimeoutError(message)
    || isRateLimitError(message)
    || isOverloadedError(message)
    || isAuthError(message)
    || isBillingError(message)
    || /\b(ECONNREFUSED|ECONNRESET|ENOTFOUND|ETIMEDOUT)\b/i.test(message)
    || lower.includes('ollama is unavailable')
    || lower.includes('model may be slow or unavailable')
    || lower.includes('model is unavailable')
    || lower.includes('llm request failed')
    || lower.includes('api key')
    || lower.includes('this operation was aborted');
}

function getProviderAvailabilityReason(model: string): string | null {
  const provider = resolveProvider(model);
  if (!provider.apiKeyEnvVar) {
    return null;
  }

  return getEnv(provider.apiKeyEnvVar)
    ? null
    : `${provider.displayName} API key is missing (${provider.apiKeyEnvVar})`;
}

async function probeOllamaModel(model: string, timeoutMs: number): Promise<string | null> {
  const baseUrl = getEnvOrDefault('OLLAMA_BASE_URL', 'http://127.0.0.1:11434');
  const ollamaModel = normalizeOllamaModel(model);

  async function fetchWithHardTimeout(url: string, init: RequestInit, label: string): Promise<Response> {
    const ac = new AbortController();
    let hardTimer: ReturnType<typeof setTimeout> | undefined;
    const abortTimer = setTimeout(() => ac.abort(), timeoutMs);
    try {
      return await Promise.race([
        fetch(url, { ...init, signal: ac.signal }),
        new Promise<Response>((_, reject) => {
          hardTimer = setTimeout(
            () => reject(new Error(`${label} timed out after ${timeoutMs}ms`)),
            timeoutMs + 100,
          );
        }),
      ]);
    } finally {
      clearTimeout(abortTimer);
      if (hardTimer) clearTimeout(hardTimer);
    }
  }

  try {
    const tagsResponse = await fetchWithHardTimeout(`${baseUrl}/api/tags`, {
      headers: { Accept: 'application/json' },
    }, 'Ollama health check');
    if (!tagsResponse.ok) {
      return `Ollama health check failed at ${baseUrl}/api/tags with HTTP ${tagsResponse.status}`;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return `Ollama is unavailable at ${baseUrl}: ${message}`;
  }

  try {
    const response = await fetchWithHardTimeout(`${baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
      body: JSON.stringify({
        model: ollamaModel,
        prompt: 'Reply with OK.',
        stream: false,
        options: { num_predict: 1 },
      }),
    }, `Ollama model ${ollamaModel} preflight`);

    if (!response.ok) {
      const text = await response.text().catch(() => '');
      return `Ollama model ${ollamaModel} is unavailable (HTTP ${response.status}${text ? `: ${text.slice(0, 240)}` : ''})`;
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return `Ollama model ${ollamaModel} did not respond within ${timeoutMs}ms: ${message}`;
  }

  return null;
}

function probeProviderModel(model: string): string | null {
  const apiKeyReason = getProviderAvailabilityReason(model);
  if (apiKeyReason) {
    return apiKeyReason;
  }
  return null;
}

async function resolveDefaultE2EModel(): Promise<string> {
  const override = getEnv('E2E_MODEL')?.trim();
  if (override) {
    return override;
  }

  if (!resolvedDefaultE2EModelPromise) {
    resolvedDefaultE2EModelPromise = (async () => {
      const { getOllamaModels } = await import('./ollama.js');
      const models = await getOllamaModels();

      for (const candidate of E2E_MODEL_PREFERENCES) {
        if (models.includes(candidate)) {
          return `ollama:${candidate}`;
        }
      }

      const firstCloudModel = models
        .filter((model) => model.includes(':cloud'))
        .sort((a, b) => a.localeCompare(b))[0];
      return firstCloudModel ? `ollama:${firstCloudModel}` : E2E_MODEL_FALLBACK;
    })();
  }

  return resolvedDefaultE2EModelPromise;
}

export async function getE2EPreflightStatus(modelOverride?: string): Promise<E2EPreflightStatus> {
  const model = modelOverride ?? await resolveDefaultE2EModel();
  const cacheKey = `${model}\0${getEnv('OLLAMA_BASE_URL') ?? ''}\0${getEnv('E2E_PREFLIGHT_TIMEOUT_MS') ?? ''}`;
  const cached = preflightPromises.get(cacheKey);
  if (cached) {
    return cached;
  }

  const promise = (async (): Promise<E2EPreflightStatus> => {
    const timeoutMs = resolveE2EPreflightTimeoutMs();
    const reason = isOllamaModel(model)
      ? await probeOllamaModel(model, timeoutMs)
      : probeProviderModel(model);
    return reason
      ? { available: false, model, reason }
      : { available: true, model };
  })();
  preflightPromises.set(cacheKey, promise);
  return promise;
}

async function assertE2EPreflight(modelOverride?: string): Promise<void> {
  const status = await getE2EPreflightStatus(modelOverride);
  if (!status.available) {
    throw new E2EPreflightSkipError(
      `Skipping live E2E for ${status.model}: ${status.reason ?? 'preflight failed'}`,
    );
  }
}

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
  resultFilePath?: string;
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

function createChildResultFilePath(): string {
  mkdirSync(E2E_RESULT_DIR, { recursive: true });
  return join(
    E2E_RESULT_DIR,
    `cramer-short-e2e-${process.pid}-${Date.now()}-${Math.random().toString(36).slice(2)}.json`,
  );
}

function findChildResultPayload(stdout: string): string | null {
  const markerLine = stdout
    .split(/\r?\n/)
    .filter((line) => line.startsWith(`${CHILD_RESULT_MARKER}:`))
    .at(-1);
  return markerLine ? markerLine.slice(CHILD_RESULT_MARKER.length + 1).trim() : null;
}

async function readChildResultFile(resultFilePath: string): Promise<E2EResult> {
  const file = Bun.file(resultFilePath);
  if (!await file.exists()) {
    throw new Error(`E2E child result file not found: ${resultFilePath}`);
  }
  const json = await file.text();
  return JSON.parse(json) as E2EResult;
}

async function parseChildResult(
  stdout: string,
  stderr: string,
  expectedResultFilePath?: string,
): Promise<E2EResult> {
  const payload = findChildResultPayload(stdout);
  const markerResultFilePath =
    payload && payload.startsWith('file:')
      ? payload.slice('file:'.length)
      : null;
  const resultFilePath = expectedResultFilePath ?? markerResultFilePath;

  if (resultFilePath) {
    try {
      return await readChildResultFile(resultFilePath);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(buildChildErrorMessage(
        `E2E child emitted an unreadable result file: ${message}`,
        stdout,
        stderr,
      ));
    } finally {
      try {
        rmSync(resultFilePath, { force: true });
      } catch (error) {
        logger.debug('Failed to remove E2E child result file', { error, resultFilePath });
      }
    }
  }

  if (!payload) {
    throw new Error(buildChildErrorMessage('E2E child exited without a result payload.', stdout, stderr));
  }

  try {
    return JSON.parse(Buffer.from(payload, 'base64').toString('utf8')) as E2EResult;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    throw new Error(buildChildErrorMessage(
      `E2E child emitted an unreadable inline result payload: ${message}`,
      stdout,
      stderr,
    ));
  }
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

  const model = opts.model ?? await resolveDefaultE2EModel();
  await assertE2EPreflight(model);
  const maxIterations = opts.maxIterations ?? defaultMaxIter;
  const historySeed = opts.historySeed;

  // Dynamic import to avoid mock.module contamination from unit test files
  // that load earlier in the same Bun worker. At this point afterAll hooks
  // from unit tests have already restored the real module exports.
  const AgentCtor = await loadAgentCtor();

  const ac = new AbortController();
  const agent = await AgentCtor.create({
    model,
    maxIterations,
    memoryEnabled: false, // keep E2E tests hermetic — no cross-test memory bleed
    signal: ac.signal,
  });
  const inMemoryHistory = historySeed?.length
    ? await (async () => {
        const { InMemoryChatHistory } = await import('./in-memory-chat-history.js');
        const history = new InMemoryChatHistory(model);
        for (const message of historySeed) {
          history.seedMessage(message);
        }
        return history;
      })()
    : undefined;

  const events: AgentEvent[] = [];
  const toolsCalled: string[] = [];
  const start = Date.now();

  // Race the agent run against the hard wall-clock timeout
  const timer = setTimeout(() => ac.abort(), E2E_TIMEOUT_MS);

  try {
    for await (const event of agent.run(actualQuery, inMemoryHistory)) {
      events.push(event);
      if (event.type === 'tool_start') toolsCalled.push(event.tool);
      if (ac.signal.aborted) break;
    }
  } finally {
    clearTimeout(timer);
  }

  const doneEvent = events.find((e): e is DoneEvent => e.type === 'done');
  if (!doneEvent) {
    if (ac.signal.aborted) {
      throw new Error(
        `E2E agent timed out after ${Date.now() - start}ms. ` +
          `Tools called: [${toolsCalled.join(', ')}]. Events: [${events.map((e) => e.type).join(', ')}]`,
      );
    }
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
  if (getEnv('CRAMER_E2E_CHILD') === '1') {
    return runAgentE2EInProcess(query, opts);
  }

  const childScriptPath = new URL('./e2e-agent-child.ts', import.meta.url).pathname;
  const model = opts.model ?? await resolveDefaultE2EModel();
  await assertE2EPreflight(model);
  const resultFilePath = createChildResultFilePath();
  const payload = encodeChildPayload({ query, opts, resultFilePath });
  const proc = Bun.spawn({
    cmd: [process.execPath, 'run', childScriptPath, payload],
    cwd: process.cwd(),
    env: {
      ...getEnvironment(),
      CRAMER_E2E_CHILD: '1',
    },
    stdout: 'pipe',
    stderr: 'pipe',
  });

  const stdoutPromise = new Response(proc.stdout).text();
  const stderrPromise = new Response(proc.stderr).text();
  let timedOut = false;
  let killTimer: ReturnType<typeof setTimeout> | undefined;
  const timer = setTimeout(() => {
    timedOut = true;
    proc.kill('SIGTERM');
    killTimer = setTimeout(() => {
      proc.kill('SIGKILL');
    }, 5_000);
  }, E2E_CHILD_TIMEOUT_MS);

  const exitCode = await proc.exited;
  clearTimeout(timer);
  if (killTimer) {
    clearTimeout(killTimer);
  }
  const [stdout, stderr] = await Promise.all([stdoutPromise, stderrPromise]);

  if (timedOut) {
    throw new Error(buildChildErrorMessage(
      `E2E child timed out after ${E2E_CHILD_TIMEOUT_MS}ms.`,
      stdout,
      stderr,
    ));
  }

  if (exitCode !== 0) {
    const message = buildChildErrorMessage(
      `E2E child exited with code ${exitCode}.`,
      stdout,
      stderr,
    );
    throw new Error(message);
  }

  return await parseChildResult(stdout, stderr, resultFilePath);
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

  try {
    return await withRetry(async () => {
      const result = await runAgentE2E(query, runOpts);
      if (isE2EExternalModelError(result.answer)) {
        throw new Error(result.answer);
      }
      return result;
    }, {
      maxAttempts: retryAttempts,
      baseDelayMs: 2_000,
      maxDelayMs: 5_000,
      jitter: 0,
      shouldRetry: (error) => error instanceof Error
        && (isTimeoutError(error.message) || isOverloadedError(error.message) || isRateLimitError(error.message)),
    });
  } catch (error) {
    if (isE2EPreflightSkipError(error)) {
      throw error;
    }
    throw error;
  }
}

export function readE2EChildPayloadFromArgv(): E2EChildPayload {
  const payload = process.argv[2];
  if (!payload) {
    throw new Error('Missing E2E child payload argument.');
  }
  return decodeChildPayload(payload);
}
