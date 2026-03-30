/**
 * Integration tests — Ollama cloud model provider routing and discovery.
 *
 * Tests the routing layer (resolveProvider, isThinkingModel) and the Ollama
 * model discovery API (getOllamaModels). No live LLM inference here —
 * see llm.e2e.test.ts for end-to-end prompt tests.
 *
 * Run with:
 *   bun run test:integration
 *   # or
 *   RUN_INTEGRATION=1 bun test --filter integration
 */

import { describe, expect } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import { getOllamaModels } from '@/utils/ollama.js';
import { resolveProvider } from '@/providers.js';
import { isThinkingModel } from '@/model/llm.js';

// ---------------------------------------------------------------------------
// Provider routing
// ---------------------------------------------------------------------------

describe('Ollama provider routing', () => {
  integrationIt('resolveProvider identifies ollama: prefix correctly', () => {
    const provider = resolveProvider('ollama:nemotron-3-super:cloud');
    expect(provider.id).toBe('ollama');
    expect(provider.displayName).toBe('Ollama');
    expect(provider.apiKeyEnvVar).toBeUndefined();
  });

  integrationIt('resolveProvider returns no apiKeyEnvVar for Ollama (local model)', () => {
    const provider = resolveProvider('ollama:deepseek-v3.2:cloud');
    expect(provider.id).toBe('ollama');
    expect(provider.apiKeyEnvVar).toBeUndefined();
  });

  integrationIt('isThinkingModel returns true for nemotron (thinking-capable)', () => {
    expect(isThinkingModel('ollama:nemotron-3-super:cloud')).toBe(true);
  });

  integrationIt('isThinkingModel returns true for qwen3 models', () => {
    expect(isThinkingModel('ollama:qwen3.5:cloud')).toBe(true);
    expect(isThinkingModel('ollama:qwen3-coder-next:cloud')).toBe(true);
  });

  integrationIt('isThinkingModel returns false for non-reasoning cloud models', () => {
    expect(isThinkingModel('ollama:deepseek-v3.2:cloud')).toBe(false);
    expect(isThinkingModel('ollama:minimax-m2.7:cloud')).toBe(false);
    expect(isThinkingModel('ollama:glm-5:cloud')).toBe(false);
  });

  integrationIt('non-ollama models are not identified as Ollama provider', () => {
    expect(resolveProvider('gpt-5.4').id).toBe('openai');
    expect(resolveProvider('claude-sonnet-4').id).toBe('anthropic');
    expect(resolveProvider('gemini-3-flash-preview').id).toBe('google');
  });
});

// ---------------------------------------------------------------------------
// Model discovery
// ---------------------------------------------------------------------------

describe('Ollama model discovery', () => {
  integrationIt('getOllamaModels returns a non-empty array of model names', async () => {
    const models = await getOllamaModels();
    expect(Array.isArray(models)).toBe(true);
    expect(models.length).toBeGreaterThan(0);
    expect(models.every((m) => typeof m === 'string')).toBe(true);
  });

  integrationIt('at least one cloud model is available', async () => {
    const models = await getOllamaModels();
    const cloudModels = models.filter((m) => m.includes(':cloud'));
    expect(cloudModels.length).toBeGreaterThan(0);
  });

  integrationIt('nemotron-3-super:cloud is listed in available models', async () => {
    const models = await getOllamaModels();
    expect(models).toContain('nemotron-3-super:cloud');
  });

  integrationIt('getOllamaModels returns [] when Ollama URL is unreachable', async () => {
    const original = process.env.OLLAMA_BASE_URL;
    process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:19999';
    const models = await getOllamaModels();
    process.env.OLLAMA_BASE_URL = original;
    expect(models).toEqual([]);
  });
});
