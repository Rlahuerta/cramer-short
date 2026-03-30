/**
 * E2E tests — live Ollama embeddings via OllamaEmbeddings.
 *
 * Runs in isolation to avoid @langchain/ollama mock contamination from
 * agent.test.ts (which permanently mocks the module for the Bun worker).
 *
 * Requires:
 *   - Ollama running on http://127.0.0.1:11434
 *   - nomic-embed-text:latest downloaded
 *   - mxbai-embed-large:latest downloaded
 *
 * Run with:
 *   bun run test:e2e
 *   # or directly:
 *   RUN_E2E=1 bun test src/memory/embeddings.e2e.test.ts
 */

import { beforeEach, describe, expect } from 'bun:test';
import { e2eIt } from '@/utils/test-guards.js';
import { createEmbeddingClient, embedSingleQuery } from '@/memory/embeddings.js';

describe('createEmbeddingClient — live Ollama embeddings', () => {
  let client: ReturnType<typeof createEmbeddingClient>;

  beforeEach(() => {
    client = createEmbeddingClient({ provider: 'ollama' });
  });

  e2eIt(
    'embedSingleQuery returns a numeric vector',
    async () => {
      const vector = await embedSingleQuery(client, 'Apple quarterly revenue');
      expect(vector).not.toBeNull();
      expect(Array.isArray(vector)).toBe(true);
      expect(vector!.length).toBeGreaterThan(0);
      expect(vector!.every((v) => typeof v === 'number')).toBe(true);
    },
    30_000,
  );

  e2eIt(
    'embedding vector has consistent dimensionality across queries',
    async () => {
      const v1 = await embedSingleQuery(client, 'revenue growth');
      const v2 = await embedSingleQuery(client, 'earnings per share');
      expect(v1!.length).toBe(v2!.length);
    },
    30_000,
  );

  e2eIt(
    'different queries produce different vectors (non-trivial cosine similarity)',
    async () => {
      const v1 = await embedSingleQuery(client, 'Apple stock price');
      const v2 = await embedSingleQuery(client, 'quantum physics equations');
      const dot = v1!.reduce((sum, val, i) => sum + val * v2![i]!, 0);
      const mag1 = Math.sqrt(v1!.reduce((s, v) => s + v * v, 0));
      const mag2 = Math.sqrt(v2!.reduce((s, v) => s + v * v, 0));
      const cosine = dot / (mag1 * mag2);
      expect(cosine).toBeLessThan(0.99);
    },
    30_000,
  );

  e2eIt(
    'embedSingleQuery with mxbai-embed-large returns a non-empty vector',
    async () => {
      const largeClient = createEmbeddingClient({
        provider: 'ollama',
        model: 'mxbai-embed-large',
      });
      const vector = await embedSingleQuery(largeClient, 'financial analysis');
      expect(vector).not.toBeNull();
      expect(vector!.length).toBeGreaterThan(0);
    },
    30_000,
  );

  e2eIt(
    'embedding a batch of texts returns correct count',
    async () => {
      const texts = ['text one', 'text two', 'text three'];
      const vectors = await client!.embed(texts);
      expect(vectors.length).toBe(3);
      expect(vectors.every((v) => v.length > 0)).toBe(true);
    },
    30_000,
  );
});
