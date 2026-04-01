import { describe, expect, it, afterEach } from 'bun:test';
import {
  fetchGdeltArticles,
  parseGdeltDate,
  deduplicateArticles,
  type GdeltArticle,
} from './gdelt.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeArticle(url: string, overrides: Partial<GdeltArticle> = {}): GdeltArticle {
  return {
    url,
    title: 'Test Article',
    seendate: '20240115143022',
    domain: 'reuters.com',
    language: 'English',
    sourcecountry: 'US',
    ...overrides,
  };
}

function makeRaw(url = 'https://reuters.com/article/1') {
  return {
    url,
    title: 'Test Article',
    seendate: '20240115143022',
    domain: 'reuters.com',
    language: 'English',
    sourcecountry: 'US',
    tone: '2.5',
  };
}

const savedFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = savedFetch; });

// ---------------------------------------------------------------------------
// parseGdeltDate — pure function
// ---------------------------------------------------------------------------

describe('parseGdeltDate', () => {
  it('parses a valid 14-digit GDELT timestamp', () => {
    const d = parseGdeltDate('20240115143022');
    expect(d.getFullYear()).toBe(2024);
    expect(d.getMonth()).toBe(0); // January = 0
    expect(d.getDate()).toBe(15);
    expect(d.getHours()).toBe(14);
    expect(d.getMinutes()).toBe(30);
    expect(d.getSeconds()).toBe(22);
  });

  it('returns epoch (Date(0)) for invalid length string', () => {
    const d = parseGdeltDate('2024');
    expect(d.getTime()).toBe(0);
  });

  it('returns epoch for empty string', () => {
    expect(parseGdeltDate('').getTime()).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// deduplicateArticles — pure function
// ---------------------------------------------------------------------------

describe('deduplicateArticles', () => {
  it('returns all articles when there are no duplicates', () => {
    const articles = [makeArticle('url-1'), makeArticle('url-2')];
    expect(deduplicateArticles(articles)).toHaveLength(2);
  });

  it('removes duplicate articles by URL', () => {
    const articles = [makeArticle('url-1'), makeArticle('url-2'), makeArticle('url-1')];
    const result = deduplicateArticles(articles);
    expect(result).toHaveLength(2);
    expect(result.map((a) => a.url)).toEqual(['url-1', 'url-2']);
  });

  it('keeps the first occurrence when duplicates exist', () => {
    const first = makeArticle('url-1', { title: 'first' });
    const second = makeArticle('url-1', { title: 'second' });
    expect(deduplicateArticles([first, second])[0].title).toBe('first');
  });

  it('returns empty array for empty input', () => {
    expect(deduplicateArticles([])).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// fetchGdeltArticles — fetch mock
// ---------------------------------------------------------------------------

describe('fetchGdeltArticles', () => {
  it('returns normalized articles on a successful response', async () => {
    const raw = { articles: [makeRaw()] };
    globalThis.fetch = (async () => new Response(JSON.stringify(raw), { status: 200 })) as unknown as typeof fetch;

    const articles = await fetchGdeltArticles('ukraine war');
    expect(articles).toHaveLength(1);
    expect(articles[0].url).toBe('https://reuters.com/article/1');
    expect(articles[0].tone).toBe(2.5);
  });

  it('returns empty array when articles field is absent', async () => {
    globalThis.fetch = (async () => new Response(JSON.stringify({}), { status: 200 })) as unknown as typeof fetch;
    const articles = await fetchGdeltArticles('empty');
    expect(articles).toHaveLength(0);
  });

  it('normalizes missing raw fields to defaults', async () => {
    const raw = { articles: [{}] };
    globalThis.fetch = (async () => new Response(JSON.stringify(raw), { status: 200 })) as unknown as typeof fetch;
    const articles = await fetchGdeltArticles('minimal');
    expect(articles[0].url).toBe('');
    expect(articles[0].title).toBe('');
    expect(articles[0].tone).toBeUndefined();
  });

  it('includes query and timespan in the URL', async () => {
    let capturedUrl = '';
    globalThis.fetch = (async (url: string | URL | Request) => {
      capturedUrl = url.toString();
      return new Response(JSON.stringify({ articles: [] }), { status: 200 }) as Response;
    }) as unknown as typeof fetch;

    await fetchGdeltArticles('oil conflict', { timespan: '7d' });
    expect(capturedUrl).toContain('timespan=10080'); // 7d = 10080 minutes
    expect(capturedUrl).toContain('oil');
  });

  it('applies domain filter when domains are provided', async () => {
    let capturedUrl = '';
    globalThis.fetch = (async (url: string | URL | Request) => {
      capturedUrl = url.toString();
      return new Response(JSON.stringify({ articles: [] }), { status: 200 }) as Response;
    }) as unknown as typeof fetch;

    await fetchGdeltArticles('conflict', { domains: ['reuters.com', 'bbc.com'] });
    expect(capturedUrl).toContain('reuters.com');
    expect(capturedUrl).toContain('bbc.com');
  });

  it('includes sourcelang in the query', async () => {
    let capturedUrl = '';
    globalThis.fetch = (async (url: string | URL | Request) => {
      capturedUrl = url.toString();
      return new Response(JSON.stringify({ articles: [] }), { status: 200 }) as Response;
    }) as unknown as typeof fetch;

    await fetchGdeltArticles('news', { sourceLanguage: 'spanish' });
    expect(capturedUrl).toContain('spanish');
  });

  it('respects maxRecords option', async () => {
    let capturedUrl = '';
    globalThis.fetch = (async (url: string | URL | Request) => {
      capturedUrl = url.toString();
      return new Response(JSON.stringify({ articles: [] }), { status: 200 }) as Response;
    }) as unknown as typeof fetch;

    await fetchGdeltArticles('test', { maxRecords: 50 });
    expect(capturedUrl).toContain('maxrecords=50');
  });

  it('throws on HTTP error response', async () => {
    globalThis.fetch = (async () =>
      new Response('Service Unavailable', { status: 503, statusText: 'Service Unavailable' })) as unknown as typeof fetch;
    await expect(fetchGdeltArticles('fail')).rejects.toThrow('GDELT HTTP 503');
  });

  it('parses numeric tone correctly', async () => {
    const raw = { articles: [{ ...makeRaw(), tone: 3.7 }] };
    globalThis.fetch = (async () => new Response(JSON.stringify(raw), { status: 200 })) as unknown as typeof fetch;
    const articles = await fetchGdeltArticles('tone');
    expect(articles[0].tone).toBe(3.7);
  });

  it('uses all timespan values correctly', async () => {
    const timespanMap: Record<string, string> = {
      '1d': '1440', '3d': '4320', '7d': '10080', '14d': '20160', '30d': '43200',
    };
    for (const [ts, minutes] of Object.entries(timespanMap)) {
      let capturedUrl = '';
      globalThis.fetch = (async (url: string | URL | Request) => {
        capturedUrl = url.toString();
        return new Response(JSON.stringify({ articles: [] }), { status: 200 }) as Response;
      }) as unknown as typeof fetch;
      await fetchGdeltArticles('test', { timespan: ts as never });
      expect(capturedUrl).toContain(`timespan=${minutes}`);
    }
  });
});
