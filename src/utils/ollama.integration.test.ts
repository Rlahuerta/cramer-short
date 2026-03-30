/**
 * Integration tests — Ollama HTTP API utility.
 *
 * Exercises getOllamaModels() against the live Ollama daemon running on this
 * machine. Requires Ollama to be running with at least one model downloaded.
 *
 * Run with:
 *   bun run test:integration
 */

import { afterEach, beforeEach, describe, expect } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import { getOllamaModels } from '@/utils/ollama.js';

// ---------------------------------------------------------------------------
// Live model list
// ---------------------------------------------------------------------------

describe('getOllamaModels — live Ollama', () => {
  integrationIt('returns a non-empty array when Ollama is running', async () => {
    const models = await getOllamaModels();
    expect(Array.isArray(models)).toBe(true);
    expect(models.length).toBeGreaterThan(0);
  });

  integrationIt('all returned values are non-empty strings', async () => {
    const models = await getOllamaModels();
    for (const m of models) {
      expect(typeof m).toBe('string');
      expect(m.length).toBeGreaterThan(0);
    }
  });

  integrationIt('nomic-embed-text is available (required for memory embeddings)', async () => {
    const models = await getOllamaModels();
    const embedModels = models.filter((m) => m.includes('nomic-embed-text'));
    expect(embedModels.length).toBeGreaterThan(0);
  });

  integrationIt('at least one cloud model is available', async () => {
    const models = await getOllamaModels();
    const cloudModels = models.filter((m) => m.endsWith(':cloud'));
    expect(cloudModels.length).toBeGreaterThan(0);
  });

  integrationIt('model names use colon-separated tag format (name:tag)', async () => {
    const models = await getOllamaModels();
    // Every model returned by Ollama should be in "name:tag" format
    const invalid = models.filter((m) => !m.includes(':'));
    expect(invalid).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// OLLAMA_BASE_URL override
// ---------------------------------------------------------------------------

describe('getOllamaModels — OLLAMA_BASE_URL override', () => {
  const originalBaseUrl = process.env.OLLAMA_BASE_URL;

  afterEach(() => {
    if (originalBaseUrl === undefined) {
      delete process.env.OLLAMA_BASE_URL;
    } else {
      process.env.OLLAMA_BASE_URL = originalBaseUrl;
    }
  });

  integrationIt('respects OLLAMA_BASE_URL pointing to live daemon', async () => {
    process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:11434';
    const models = await getOllamaModels();
    expect(models.length).toBeGreaterThan(0);
  });

  integrationIt('returns [] when OLLAMA_BASE_URL points to unreachable host', async () => {
    process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:19999';
    const models = await getOllamaModels();
    expect(models).toEqual([]);
  });

  integrationIt('returns [] when OLLAMA_BASE_URL points to invalid URL', async () => {
    process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:29999';
    const models = await getOllamaModels();
    expect(models).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Specific model availability checks
// ---------------------------------------------------------------------------

describe('getOllamaModels — expected models on this machine', () => {
  let models: string[];

  beforeEach(async () => {
    models = await getOllamaModels();
  });

  integrationIt('deepseek-v3.2:cloud is available', async () => {
    expect(models).toContain('deepseek-v3.2:cloud');
  });

  integrationIt('qwen3.5:397b-cloud is available', async () => {
    expect(models).toContain('qwen3.5:397b-cloud');
  });

  integrationIt('nemotron-3-super:cloud is available', async () => {
    expect(models).toContain('nemotron-3-super:cloud');
  });

  integrationIt('mxbai-embed-large:latest is available (embedding model)', async () => {
    expect(models).toContain('mxbai-embed-large:latest');
  });
});
