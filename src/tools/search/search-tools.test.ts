import { FIXED_TEST_DATE, FIXED_TEST_NOW_MS, deterministicRandom, nextTestId } from '@/utils/test-determinism.js';
import { mock, describe, it, expect, beforeAll, afterAll, beforeEach, afterEach, setSystemTime } from 'bun:test';

// exa.ts requires a key before it builds the (mocked) client; don't depend on a local .env.
const originalExaKey = process.env.EXASEARCH_API_KEY;
beforeAll(() => {
  process.env.EXASEARCH_API_KEY = 'test-exa-key';
});
afterAll(() => {
  if (originalExaKey === undefined) delete process.env.EXASEARCH_API_KEY;
  else process.env.EXASEARCH_API_KEY = originalExaKey;
});

beforeEach(() => {
  setSystemTime(FIXED_TEST_DATE);
});

afterEach(() => {
  setSystemTime();
});

// Mock external search dependencies BEFORE importing the search tools
const mockExaInvoke = mock(() =>
  Promise.resolve({ results: [{ url: 'https://example.com', text: 'content' }] })
);

// Snapshot the real modules so afterAll can undo mock.module (Bun never does);
// otherwise these mocks leak into every test file that runs later.
const realExaJsModule = { ...(await import('exa-js')) };
const realExaModule = { ...(await import('@langchain/exa')) };
const realTavilyModule = { ...(await import('@langchain/tavily')) };
afterAll(() => {
  mock.module('exa-js', () => realExaJsModule);
  mock.module('@langchain/exa', () => realExaModule);
  mock.module('@langchain/tavily', () => realTavilyModule);
});

mock.module('exa-js', () => ({
  default: class Exa {
    constructor(_apiKey?: string) {}
  },
}));

mock.module('@langchain/exa', () => ({
  ExaSearchResults: class {
    invoke = mockExaInvoke;
    constructor(_opts?: unknown) {}
  },
}));

const mockTavilyInvoke = mock(() =>
  Promise.resolve([{ url: 'https://example.com', content: 'result' }])
);

mock.module('@langchain/tavily', () => ({
  TavilySearch: class {
    invoke = mockTavilyInvoke;
    constructor(_opts?: unknown) {}
  },
}));

// Dynamic imports after mocking.
// Use cache-busting ?t= imports so that when fundamentals.test.ts / key-ratios.test.ts /
// stock-price.test.ts mock '../search/tavily.js' directly (and thus override the module
// registry entry for 'src/tools/search/tavily.js'), these imports resolve to a DIFFERENT
// module cache key that bypasses the override and loads the real tavily.ts — which then
// picks up our mock.module('@langchain/tavily') above.
const t = FIXED_TEST_NOW_MS;
const { exaSearch } = await import(`./exa.js?t=${t}`) as typeof import('./exa.js');
const { tavilySearch } = await import(`./tavily.js?t=${t}`) as typeof import('./tavily.js');

// ---------------------------------------------------------------------------
// exaSearch
// ---------------------------------------------------------------------------

describe('exaSearch', () => {
  it('returns formatted JSON with urls on success', async () => {
    mockExaInvoke.mockResolvedValueOnce({
      results: [
        { url: 'https://exa.ai/result1', text: 'First result' },
        { url: 'https://exa.ai/result2', text: 'Second result' },
      ],
    });

    const output = await exaSearch.invoke({ query: 'test query' });
    const parsed = JSON.parse(output);
    expect(parsed.data).toBeDefined();
    expect(parsed.sourceUrls).toContain('https://exa.ai/result1');
    expect(parsed.sourceUrls).toContain('https://exa.ai/result2');
  });

  it('rethrows errors with [Exa API] prefix', async () => {
    mockExaInvoke.mockRejectedValueOnce(new Error('rate limit exceeded'));
    await expect(exaSearch.invoke({ query: 'failing query' })).rejects.toThrow(
      '[Exa API] rate limit exceeded'
    );
  });
});

// ---------------------------------------------------------------------------
// tavilySearch
// ---------------------------------------------------------------------------

describe('tavilySearch', () => {
  it('returns formatted JSON with urls on success', async () => {
    mockTavilyInvoke.mockResolvedValueOnce([
      { url: 'https://tavily.com/result1', content: 'First result' },
      { url: 'https://tavily.com/result2', content: 'Second result' },
    ]);

    const output = await tavilySearch.invoke({ query: 'test query' });
    const parsed = JSON.parse(output);
    expect(parsed.data).toBeDefined();
    expect(parsed.sourceUrls).toContain('https://tavily.com/result1');
    expect(parsed.sourceUrls).toContain('https://tavily.com/result2');
  });

  it('rethrows errors with [Tavily API] prefix', async () => {
    mockTavilyInvoke.mockRejectedValueOnce(new Error('API key invalid'));
    await expect(tavilySearch.invoke({ query: 'failing query' })).rejects.toThrow(
      '[Tavily API] API key invalid'
    );
  });
});
