/**
 * Integration tests — embedding client provider resolution.
 *
 * Tests createEmbeddingClient() provider routing logic — pure unit tests with
 * no live network calls. No @langchain/ollama embedDocuments() invocations here
 * (those are in embeddings.e2e.test.ts to avoid mock contamination from agent.test.ts).
 *
 * Run with:
 *   bun run test:integration
 */

import { afterEach, describe, expect, it } from 'bun:test';
import { createEmbeddingClient, embedSingleQuery } from '@/memory/embeddings.js';

// ---------------------------------------------------------------------------
// Provider resolution — pure logic (no network)
// ---------------------------------------------------------------------------

describe('createEmbeddingClient — provider resolution', () => {
  const savedOpenAI = process.env.OPENAI_API_KEY;
  const savedGoogle = process.env.GOOGLE_API_KEY;
  const savedOllama = process.env.OLLAMA_BASE_URL;

  afterEach(() => {
    if (savedOpenAI === undefined) delete process.env.OPENAI_API_KEY;
    else process.env.OPENAI_API_KEY = savedOpenAI;
    if (savedGoogle === undefined) delete process.env.GOOGLE_API_KEY;
    else process.env.GOOGLE_API_KEY = savedGoogle;
    if (savedOllama === undefined) delete process.env.OLLAMA_BASE_URL;
    else process.env.OLLAMA_BASE_URL = savedOllama;
  });

  it('provider=ollama always returns a client (no API key needed)', () => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    const client = createEmbeddingClient({ provider: 'ollama' });
    expect(client).not.toBeNull();
    expect(client!.provider).toBe('ollama');
  });

  it('provider=ollama uses default nomic-embed-text model', () => {
    const client = createEmbeddingClient({ provider: 'ollama' });
    expect(client!.model).toBe('nomic-embed-text');
  });

  it('provider=ollama accepts a custom model override', () => {
    const client = createEmbeddingClient({ provider: 'ollama', model: 'mxbai-embed-large' });
    expect(client!.model).toBe('mxbai-embed-large');
  });

  it('provider=auto resolves to ollama when only OLLAMA_BASE_URL is set', () => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:11434';
    const client = createEmbeddingClient({ provider: 'auto' });
    expect(client).not.toBeNull();
    expect(client!.provider).toBe('ollama');
  });

  it('provider=auto returns null when no provider is configured', () => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    delete process.env.OLLAMA_BASE_URL;
    const client = createEmbeddingClient({ provider: 'auto' });
    expect(client).toBeNull();
  });

  it('provider=auto prefers openai over ollama when both are set', () => {
    process.env.OPENAI_API_KEY = 'sk-test-key';
    process.env.OLLAMA_BASE_URL = 'http://127.0.0.1:11434';
    const client = createEmbeddingClient({ provider: 'auto' });
    expect(client!.provider).toBe('openai');
  });

  it('provider=none returns null', () => {
    const client = createEmbeddingClient({ provider: 'none' });
    expect(client).toBeNull();
  });

  it('embedSingleQuery returns null when client is null', async () => {
    const result = await embedSingleQuery(null, 'test query');
    expect(result).toBeNull();
  });
});
