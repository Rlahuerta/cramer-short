import { describe, expect, it, afterEach, beforeEach } from 'bun:test';
import { perplexitySearch } from './perplexity.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makePerplexityResponse(overrides: Record<string, unknown> = {}) {
  return {
    choices: [{ message: { content: 'The answer is 42.' } }],
    citations: ['https://source1.com', 'https://source2.com'],
    search_results: [
      { title: 'Result 1', url: 'https://source1.com', snippet: 'Some context here.' },
      { title: 'Result 2', url: 'https://source2.com' },
    ],
    ...overrides,
  };
}

const savedFetch = globalThis.fetch;
const savedKey = process.env.PERPLEXITY_API_KEY;

afterEach(() => {
  globalThis.fetch = savedFetch;
  if (savedKey !== undefined) {
    process.env.PERPLEXITY_API_KEY = savedKey;
  } else {
    delete process.env.PERPLEXITY_API_KEY;
  }
});

beforeEach(() => {
  process.env.PERPLEXITY_API_KEY = 'test-key';
});

// ---------------------------------------------------------------------------
// perplexitySearch tool
// ---------------------------------------------------------------------------

describe('perplexitySearch', () => {
  it('returns formatted answer with citations and search results', async () => {
    const payload = makePerplexityResponse();
    globalThis.fetch = async () =>
      new Response(JSON.stringify(payload), { status: 200 }) as Response;

    const result = await perplexitySearch.invoke({ query: 'what is 42?' });
    expect(result).toContain('The answer is 42.');
    expect(result).toContain('source1.com');
  });

  it('deduplicates URLs from citations and search_results', async () => {
    const payload = makePerplexityResponse({
      citations: ['https://same.com'],
      search_results: [{ title: 'Same', url: 'https://same.com' }],
    });
    globalThis.fetch = async () =>
      new Response(JSON.stringify(payload), { status: 200 }) as Response;

    const result = await perplexitySearch.invoke({ query: 'dedupe test' });
    // Just verify it doesn't throw and returns content
    expect(result).toContain('same.com');
  });

  it('handles response with no citations (null)', async () => {
    const payload = makePerplexityResponse({ citations: null, search_results: null });
    globalThis.fetch = async () =>
      new Response(JSON.stringify(payload), { status: 200 }) as Response;

    const result = await perplexitySearch.invoke({ query: 'no citations' });
    expect(result).toContain('The answer is 42.');
  });

  it('handles response with no search_results but has citations', async () => {
    const payload = makePerplexityResponse({ search_results: null });
    globalThis.fetch = async () =>
      new Response(JSON.stringify(payload), { status: 200 }) as Response;

    const result = await perplexitySearch.invoke({ query: 'citations only' });
    expect(result).toContain('source1.com');
  });

  it('handles empty choices array gracefully', async () => {
    const payload = makePerplexityResponse({ choices: [] });
    globalThis.fetch = async () =>
      new Response(JSON.stringify(payload), { status: 200 }) as Response;

    const result = await perplexitySearch.invoke({ query: 'empty choices' });
    expect(typeof result).toBe('string');
  });

  it('throws when PERPLEXITY_API_KEY is not set', async () => {
    delete process.env.PERPLEXITY_API_KEY;
    await expect(perplexitySearch.invoke({ query: 'no key' })).rejects.toThrow(
      'PERPLEXITY_API_KEY is not set',
    );
  });

  it('throws on non-ok HTTP response with status code', async () => {
    globalThis.fetch = async () =>
      new Response('Unauthorized', { status: 401 }) as Response;

    await expect(perplexitySearch.invoke({ query: 'bad key' })).rejects.toThrow('[Perplexity API]');
  });

  it('throws on HTTP 429 rate limit', async () => {
    globalThis.fetch = async () =>
      new Response('Too Many Requests', { status: 429 }) as Response;

    await expect(perplexitySearch.invoke({ query: 'rate limit' })).rejects.toThrow('[Perplexity API]');
  });

  it('adds search_results URLs not already in citations', async () => {
    const payload = makePerplexityResponse({
      citations: ['https://citation-only.com'],
      search_results: [
        { title: 'New', url: 'https://result-only.com' },
        { title: 'Dup', url: 'https://citation-only.com' },
      ],
    });
    globalThis.fetch = async () =>
      new Response(JSON.stringify(payload), { status: 200 }) as Response;

    const result = await perplexitySearch.invoke({ query: 'url union' });
    expect(result).toContain('citation-only.com');
    expect(result).toContain('result-only.com');
  });

  it('sends the query to the Perplexity API endpoint', async () => {
    let capturedBody = '';
    globalThis.fetch = async (_url: unknown, opts?: RequestInit) => {
      capturedBody = opts?.body as string ?? '';
      return new Response(JSON.stringify(makePerplexityResponse()), { status: 200 }) as Response;
    };

    await perplexitySearch.invoke({ query: 'my specific query' });
    const body = JSON.parse(capturedBody);
    expect(body.messages[0].content).toBe('my specific query');
    expect(body.model).toBe('sonar');
  });
});
