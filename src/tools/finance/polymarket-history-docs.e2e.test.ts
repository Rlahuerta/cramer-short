/**
 * E2E tests — validate documented Polymarket history guide examples
 * against the default live Ollama cloud E2E model.
 *
 * These tests assert on stable structural behaviour (tool calls triggered,
 * heading formats, horizon labels, warning phrases), not on volatile market
 * numbers that drift between runs.
 *
 * Run with:  bun run test:e2e
 * Model:     auto-resolved live Ollama cloud model (override via E2E_MODEL)
 */
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { describe, expect, beforeEach } from 'bun:test';
import { e2eIt } from '@/utils/test-guards.js';
import { runAgentE2EWithTimeoutRetry, E2E_TIMEOUT_MS } from '@/utils/e2e-helpers.js';
import type { ToolStartEvent, ToolEndEvent } from '@/agent/types.js';

// Load the real ChatOllama via require(CJS filesystem path) to bypass Bun's
// mock.module contamination from agent.test.ts (which stubs @langchain/ollama).
// We inject a factory that creates instances from the real class so that
// getChatModel() in llm.ts uses real Ollama models.
const RealChatOllama = require(
  '../../../node_modules/@langchain/ollama/dist/chat_models.cjs',
).ChatOllama;

import { _setModelFactory } from '@/model/llm.js';

const OLLAMA_BASE_URL = process.env.OLLAMA_BASE_URL ?? 'http://127.0.0.1:11434';
const THINKING_RE = /qwen3|deepseek-r1|deepseek-v4|qwq|nemotron|gemma4|kimi-k2\.6/i;

function ollamaFactory(name: string, opts: { streaming: boolean }, thinkOverride?: boolean) {
  const modelName = name.replace(/^ollama:/i, '');
  const useThink = thinkOverride !== undefined
    ? thinkOverride
    : THINKING_RE.test(modelName);
  return new RealChatOllama({
    model: modelName,
    ...opts,
    ...(useThink ? { think: true } : {}),
    ...(OLLAMA_BASE_URL ? { baseUrl: OLLAMA_BASE_URL } : {}),
  });
}

// Inject the real-model factory at file scope and before each test — guards
// against agent.test.ts's beforeAll which injects SpyChatModel.
_setModelFactory(ollamaFactory);
beforeEach(() => { _setModelFactory(ollamaFactory); });

const SNAPSHOT_FILE = '.cramer-short/polymarket-snapshots.jsonl';

async function withFreshSnapshotFile<T>(fn: () => Promise<T>): Promise<T> {
  const hadOriginal = existsSync(SNAPSHOT_FILE);
  const original = hadOriginal ? readFileSync(SNAPSHOT_FILE, 'utf8') : null;

  rmSync(SNAPSHOT_FILE, { force: true });

  try {
    return await fn();
  } finally {
    if (hadOriginal) {
      mkdirSync(dirname(SNAPSHOT_FILE), { recursive: true });
      writeFileSync(SNAPSHOT_FILE, original ?? '', 'utf8');
    } else {
      rmSync(SNAPSHOT_FILE, { force: true });
    }
  }
}

function findToolStartEvent(result: { events: unknown[] }, tool: string): ToolStartEvent | undefined {
  return result.events.find((event): event is ToolStartEvent => {
    if (!event || typeof event !== 'object') return false;
    const candidate = event as { type?: string; tool?: string };
    return candidate.type === 'tool_start' && candidate.tool === tool;
  });
}

function findToolEndEvent(result: { events: unknown[] }, tool: string): ToolEndEvent | undefined {
  return result.events.find((event): event is ToolEndEvent => {
    if (!event || typeof event !== 'object') return false;
    const candidate = event as { type?: string; tool?: string };
    return candidate.type === 'tool_end' && candidate.tool === tool;
  });
}

function extractToolResultText(result: string): string {
  try {
    const payload = JSON.parse(result) as { data?: { result?: string; error?: string } };
    return payload.data?.result ?? payload.data?.error ?? result;
  } catch {
    return result;
  }
}

describe('Polymarket history docs — guide examples vs live Ollama cloud model', () => {
  beforeEach(() => { _setModelFactory(ollamaFactory); });

  e2eIt(
    '(a) BTC price-market search calls polymarket_search and returns the Polymarket heading',
    async () => {
      _setModelFactory(ollamaFactory);
      const result = await withFreshSnapshotFile(() => runAgentE2EWithTimeoutRetry(
        'Search Polymarket for Bitcoin price markets and summarize the top results.',
      ));

      expect(result.toolsCalled).toContain('polymarket_search');

      const searchEnd = findToolEndEvent(result, 'polymarket_search');
      expect(searchEnd).toBeDefined();
      const text = extractToolResultText(searchEnd!.result);

      // The polymarket_search tool formats results starting with
      // "Polymarket — prediction market probabilities for: ..."
      // but the agent may shorten the tool query from "Bitcoin price" to "Bitcoin".
      expect(text).toMatch(/Polymarket.*prediction market probabilities/i);
      expect(text).toMatch(/Bitcoin/i);
      expect(text).toMatch(/price of Bitcoin|Bitcoin reach|Bitcoin dip/i);

      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    '(b) BTC 7-day forecast calls polymarket_forecast with correct heading, horizon, and cold-start warnings',
    async () => {
      _setModelFactory(ollamaFactory);
      const result = await withFreshSnapshotFile(() => runAgentE2EWithTimeoutRetry(
        'Give me a Polymarket-based forecast for BTC over the next 7 days. Current price is 68000.',
      ));

      expect(result.toolsCalled).toContain('polymarket_forecast');

      const forecastStart = findToolStartEvent(result, 'polymarket_forecast');
      expect(forecastStart).toBeDefined();
      expect(forecastStart?.args.ticker).toBe('BTC');
      expect(forecastStart?.args.horizon_days).toBe(7);

      const forecastEnd = findToolEndEvent(result, 'polymarket_forecast');
      expect(forecastEnd).toBeDefined();
      const text = extractToolResultText(forecastEnd!.result);

      // Structural heading assertions — no price numbers
      expect(text).toContain('Polymarket Forecast: BTC (BTC)');
      expect(text).toContain('Horizon: 7 days');

      // Deterministic cold-start warnings due to fresh snapshot file
      expect(text).toMatch(/Spike detection unavailable|Persistence test unavailable/i);
      expect(text).toMatch(/Persistence test unavailable/i);

      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    '(c) BTC 180-day forecast calls polymarket_forecast with heading and >90-day horizon warning',
    async () => {
      _setModelFactory(ollamaFactory);
      const result = await withFreshSnapshotFile(() => runAgentE2EWithTimeoutRetry(
        'Give me a Polymarket-based forecast for BTC over the next 180 days. Current price is 68000.',
      ));

      expect(result.toolsCalled).toContain('polymarket_forecast');

      const forecastStart = findToolStartEvent(result, 'polymarket_forecast');
      expect(forecastStart).toBeDefined();
      expect(forecastStart?.args.ticker).toBe('BTC');
      expect(forecastStart?.args.horizon_days).toBe(180);

      const forecastEnd = findToolEndEvent(result, 'polymarket_forecast');
      expect(forecastEnd).toBeDefined();
      const text = extractToolResultText(forecastEnd!.result);

      // Structural heading
      expect(text).toContain('Polymarket Forecast: BTC (BTC)');

      // >90-day horizon warning
      expect(text).toMatch(/Horizon.*180.*[dD].*>.*90/);

      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );

  e2eIt(
    '(d) whale-aware BTC prompt still routes through polymarket_forecast and surfaces history insufficiency warnings',
    async () => {
      _setModelFactory(ollamaFactory);
      const result = await withFreshSnapshotFile(() => runAgentE2EWithTimeoutRetry(
        'Give me a Polymarket-based forecast for BTC over the next 7 days. Current price is 68000. Pay special attention to the warning section and tell me whether any low-volume spike suggests possible whale activity or whether the history is still insufficient.',
      ));

      expect(result.toolsCalled).toContain('polymarket_forecast');

      const forecastStart = findToolStartEvent(result, 'polymarket_forecast');
      expect(forecastStart).toBeDefined();
      expect(forecastStart?.args.ticker).toBe('BTC');
      expect(forecastStart?.args.horizon_days).toBe(7);

      const forecastEnd = findToolEndEvent(result, 'polymarket_forecast');
      expect(forecastEnd).toBeDefined();
      const text = extractToolResultText(forecastEnd!.result);

      expect(text).toContain('Polymarket Forecast: BTC (BTC)');
      expect(text).toMatch(/Spike detection unavailable|Persistence test unavailable/i);

      expect(result.durationMs).toBeLessThan(E2E_TIMEOUT_MS);
    },
    E2E_TIMEOUT_MS,
  );
});
