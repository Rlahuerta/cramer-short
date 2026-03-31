import { describe, expect, it, afterEach, beforeEach } from 'bun:test';
import { xSearchTool, X_SEARCH_DESCRIPTION } from './x-search.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makeUser(id = 'u1', username = 'testuser') {
  return { id, username, name: 'Test User', public_metrics: { followers_count: 100 } };
}

function makeTweet(id = 't1', userId = 'u1') {
  return {
    id,
    text: 'This is a tweet',
    author_id: userId,
    created_at: '2024-01-15T12:00:00Z',
    public_metrics: { like_count: 5, retweet_count: 2, reply_count: 1, impression_count: 500 },
  };
}

function makeSearchResponse(tweets = [makeTweet()], users = [makeUser()]) {
  return {
    data: tweets,
    includes: { users },
    meta: {},
  };
}

function makeProfileResponse(user = makeUser()) {
  return { data: user };
}

const savedFetch = globalThis.fetch;
const savedToken = process.env.X_BEARER_TOKEN;

beforeEach(() => {
  process.env.X_BEARER_TOKEN = 'test-bearer-token';
});

afterEach(() => {
  globalThis.fetch = savedFetch;
  if (savedToken !== undefined) {
    process.env.X_BEARER_TOKEN = savedToken;
  } else {
    delete process.env.X_BEARER_TOKEN;
  }
});

// ---------------------------------------------------------------------------
// X_SEARCH_DESCRIPTION
// ---------------------------------------------------------------------------

describe('X_SEARCH_DESCRIPTION', () => {
  it('is a non-empty string', () => {
    expect(typeof X_SEARCH_DESCRIPTION).toBe('string');
    expect(X_SEARCH_DESCRIPTION.length).toBeGreaterThan(50);
  });

  it('mentions all three commands', () => {
    expect(X_SEARCH_DESCRIPTION).toContain('search');
    expect(X_SEARCH_DESCRIPTION).toContain('profile');
    expect(X_SEARCH_DESCRIPTION).toContain('thread');
  });
});

// ---------------------------------------------------------------------------
// Missing X_BEARER_TOKEN
// ---------------------------------------------------------------------------

describe('xSearchTool — missing token', () => {
  it('throws when X_BEARER_TOKEN is not set', async () => {
    delete process.env.X_BEARER_TOKEN;
    await expect(
      xSearchTool.invoke({ command: 'search', query: 'AAPL' }),
    ).rejects.toThrow('X_BEARER_TOKEN is not set');
  });
});

// ---------------------------------------------------------------------------
// search command
// ---------------------------------------------------------------------------

describe('xSearchTool — search command', () => {
  it('returns tweets on a successful search', async () => {
    const resp = makeSearchResponse();
    globalThis.fetch = async () =>
      new Response(JSON.stringify(resp), { status: 200 }) as Response;

    const result = await xSearchTool.invoke({ command: 'search', query: 'AAPL earnings' });
    expect(result).toContain('testuser');
    expect(result).toContain('This is a tweet');
  });

  it('auto-appends -is:retweet when not present in query', async () => {
    let capturedUrl = '';
    globalThis.fetch = async (url: string | URL | Request) => {
      capturedUrl = url.toString();
      return new Response(JSON.stringify(makeSearchResponse()), { status: 200 }) as Response;
    };

    await xSearchTool.invoke({ command: 'search', query: 'NVDA' });
    expect(decodeURIComponent(capturedUrl)).toContain('-is:retweet');
  });

  it('does NOT double-append -is:retweet when already present', async () => {
    let capturedUrl = '';
    globalThis.fetch = async (url: string | URL | Request) => {
      capturedUrl = url.toString();
      return new Response(JSON.stringify(makeSearchResponse()), { status: 200 }) as Response;
    };

    await xSearchTool.invoke({ command: 'search', query: 'NVDA -is:retweet' });
    const decoded = decodeURIComponent(capturedUrl);
    expect((decoded.match(/-is:retweet/g) ?? []).length).toBe(1);
  });

  it('filters tweets by min_likes', async () => {
    const tweets = [makeTweet('t1'), makeTweet('t2')];
    tweets[0].public_metrics.like_count = 20;
    tweets[1].public_metrics.like_count = 2;
    globalThis.fetch = async () =>
      new Response(JSON.stringify(makeSearchResponse(tweets)), { status: 200 }) as Response;

    const result = await xSearchTool.invoke({ command: 'search', query: 'test', min_likes: 10 });
    // Only tweet with 20 likes should pass (t1). t2 (2 likes) should be filtered out.
    expect(result).toContain('"id":"t1"');
    expect(result).not.toContain('"id":"t2"');
  });

  it('sorts by likes descending when sort=likes', async () => {
    const tweets = [makeTweet('t1'), makeTweet('t2')];
    tweets[0].public_metrics.like_count = 5;
    tweets[1].public_metrics.like_count = 100;
    globalThis.fetch = async () =>
      new Response(JSON.stringify(makeSearchResponse(tweets)), { status: 200 }) as Response;

    const result = await xSearchTool.invoke({ command: 'search', query: 'sort test', sort: 'likes' });
    // tweet t2 (100 likes) should appear before t1 (5 likes)
    const idx1 = result.indexOf('"id":"t1"');
    const idx2 = result.indexOf('"id":"t2"');
    expect(idx2).toBeLessThan(idx1);
  });

  it('throws when query is missing for search command', async () => {
    await expect(
      xSearchTool.invoke({ command: 'search' }),
    ).rejects.toThrow('query is required');
  });

  it('includes since parameter in URL when provided', async () => {
    let capturedUrl = '';
    globalThis.fetch = async (url: string | URL | Request) => {
      capturedUrl = url.toString();
      return new Response(JSON.stringify(makeSearchResponse()), { status: 200 }) as Response;
    };

    await xSearchTool.invoke({ command: 'search', query: 'news', since: '1h' });
    expect(capturedUrl).toContain('start_time=');
  });

  it('handles rate limit (429) with descriptive error', async () => {
    const resetTime = String(Math.floor(Date.now() / 1000) + 60);
    globalThis.fetch = async () =>
      new Response('Too Many Requests', {
        status: 429,
        headers: { 'x-rate-limit-reset': resetTime },
      }) as Response;

    await expect(xSearchTool.invoke({ command: 'search', query: 'test' })).rejects.toThrow(
      'rate limited',
    );
  });

  it('handles HTTP error with status code', async () => {
    globalThis.fetch = async () =>
      new Response('Forbidden', { status: 403 }) as Response;

    await expect(xSearchTool.invoke({ command: 'search', query: 'test' })).rejects.toThrow('403');
  });

  it('deduplicates tweets across pages', async () => {
    let callCount = 0;
    globalThis.fetch = async () => {
      callCount++;
      const resp = {
        ...makeSearchResponse([makeTweet('t1')]),
        meta: callCount === 1 ? { next_token: 'page2' } : {},
      };
      return new Response(JSON.stringify(resp), { status: 200 }) as Response;
    };

    const result = await xSearchTool.invoke({ command: 'search', query: 'dup', pages: 2 });
    // t1 should appear only once despite being returned on both pages
    const matches = (result.match(/"id":"t1"/g) ?? []).length;
    expect(matches).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// profile command
// ---------------------------------------------------------------------------

describe('xSearchTool — profile command', () => {
  it('returns user info and tweets', async () => {
    let callCount = 0;
    globalThis.fetch = async () => {
      callCount++;
      if (callCount === 1) {
        // First call: user lookup
        return new Response(JSON.stringify(makeProfileResponse()), { status: 200 }) as Response;
      }
      // Second call: tweet search
      return new Response(JSON.stringify(makeSearchResponse()), { status: 200 }) as Response;
    };

    const result = await xSearchTool.invoke({ command: 'profile', username: 'testuser' });
    expect(result).toContain('testuser');
  });

  it('throws when username is missing for profile command', async () => {
    await expect(
      xSearchTool.invoke({ command: 'profile' }),
    ).rejects.toThrow('username is required');
  });

  it('throws when user is not found', async () => {
    globalThis.fetch = async () =>
      new Response(JSON.stringify({ data: null }), { status: 200 }) as Response;

    await expect(
      xSearchTool.invoke({ command: 'profile', username: 'ghost' }),
    ).rejects.toThrow('not found');
  });
});

// ---------------------------------------------------------------------------
// thread command
// ---------------------------------------------------------------------------

describe('xSearchTool — thread command', () => {
  it('returns thread tweets', async () => {
    globalThis.fetch = async () =>
      new Response(JSON.stringify(makeSearchResponse()), { status: 200 }) as Response;

    const result = await xSearchTool.invoke({ command: 'thread', query: '123456789' });
    expect(result).toContain('testuser');
  });

  it('throws when tweet ID (query) is missing for thread command', async () => {
    await expect(
      xSearchTool.invoke({ command: 'thread' }),
    ).rejects.toThrow('tweet ID');
  });
});

// ---------------------------------------------------------------------------
// parseSince — tested indirectly through the search URL
// ---------------------------------------------------------------------------

describe('parseSince (via search URL)', () => {
  async function captureUrl(since: string) {
    let url = '';
    globalThis.fetch = async (u: string | URL | Request) => {
      url = u.toString();
      return new Response(JSON.stringify(makeSearchResponse()), { status: 200 }) as Response;
    };
    await xSearchTool.invoke({ command: 'search', query: 'x', since });
    return url;
  }

  it('parses "1h" into an ISO start_time roughly 1 hour ago', async () => {
    const before = Date.now() - 3_700_000;
    const url = await captureUrl('1h');
    const match = url.match(/start_time=([^&]+)/);
    expect(match).toBeTruthy();
    const t = new Date(decodeURIComponent(match![1])).getTime();
    expect(t).toBeGreaterThan(before);
  });

  it('parses "3d" into an ISO start_time roughly 3 days ago', async () => {
    const before = Date.now() - 3 * 86_400_000 - 5000;
    const url = await captureUrl('3d');
    const match = url.match(/start_time=([^&]+)/);
    const t = new Date(decodeURIComponent(match![1])).getTime();
    expect(t).toBeGreaterThan(before);
  });

  it('passes a full ISO timestamp directly as start_time', async () => {
    const url = await captureUrl('2024-01-15T12:00:00Z');
    expect(url).toContain('start_time=');
  });

  it('skips invalid since values (no start_time in URL)', async () => {
    const url = await captureUrl('not-valid');
    expect(url).not.toContain('start_time=');
  });
});
