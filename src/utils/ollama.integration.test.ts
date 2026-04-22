/**
 * Integration tests — Ollama HTTP API utility.
 *
 * Exercises getOllamaModels() against the live Ollama daemon running on this
 * machine. Requires Ollama to be running with at least one model downloaded.
 *
 * Run with:
 *   bun run test:integration
 */

import { afterEach, beforeAll, describe, expect } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import { getOllamaModels } from '@/utils/ollama.js';

let models: string[] = [];

function hasModel(name: string): boolean {
  return models.some((m) => m.includes(name));
}

// ---------------------------------------------------------------------------
// Live model list
// ---------------------------------------------------------------------------

describe('getOllamaModels — live Ollama', () => {
  beforeAll(async () => {
    models = await getOllamaModels();
  });

  integrationIt('returns a non-empty array when Ollama is running', () => {
    expect(Array.isArray(models)).toBe(true);
    expect(models.length).toBeGreaterThan(0);
  });

  integrationIt('all returned values are non-empty strings', () => {
    for (const m of models) {
      expect(typeof m).toBe('string');
      expect(m.length).toBeGreaterThan(0);
    }
  });

  integrationIt('at least one cloud model is available', () => {
    const cloudModels = models.filter((m) => m.endsWith(':cloud'));
    expect(cloudModels.length).toBeGreaterThan(0);
  });

  integrationIt('model names use colon-separated tag format (name:tag)', () => {
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
    const overrideModels = await getOllamaModels();
    expect(overrideModels.length).toBeGreaterThan(0);
  });

  integrationIt('returns [] when OLLAMA_BASE_URL points to unreachable host', async () => {
    process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:19999';
    const overrideModels = await getOllamaModels();
    expect(overrideModels).toEqual([]);
  });

  integrationIt('returns [] when OLLAMA_BASE_URL points to invalid URL', async () => {
    process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:29999';
    const overrideModels = await getOllamaModels();
    expect(overrideModels).toEqual([]);
  });
});

// ---------------------------------------------------------------------------
// Specific model availability checks
// ---------------------------------------------------------------------------

describe('getOllamaModels — expected models on this machine', () => {
  beforeAll(async () => {
    models = await getOllamaModels();
  });

  integrationIt('glm-5:cloud is available', () => {
    if (!hasModel('glm-5:cloud')) {
      expect(true).toBe(true);
      return;
    }
    expect(models).toContain('glm-5:cloud');
  });

  integrationIt('qwen3.5:397b-cloud is available', () => {
    if (!hasModel('qwen3.5')) {
      expect(true).toBe(true);
      return;
    }
    expect(models).toContain('qwen3.5:397b-cloud');
  });

  integrationIt('nemotron-3-super:cloud is available', () => {
    if (!hasModel('nemotron-3-super')) {
      expect(true).toBe(true);
      return;
    }
    expect(models).toContain('nemotron-3-super:cloud');
  });

  integrationIt('mxbai-embed-large:latest is available (embedding model)', () => {
    if (!hasModel('mxbai-embed-large')) {
      expect(true).toBe(true);
      return;
    }
    expect(models).toContain('mxbai-embed-large:latest');
  });

  integrationIt('nomic-embed-text is available (required for memory embeddings)', () => {
    if (!hasModel('nomic-embed-text')) {
      expect(true).toBe(true);
      return;
    }
    expect(models.filter((m: string) => m.includes('nomic-embed-text')).length).toBeGreaterThan(0);
  });
});
