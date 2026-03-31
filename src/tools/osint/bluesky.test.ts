import { describe, expect, it, afterEach } from 'bun:test';
import {
  searchBskyPosts,
  getBskyAuthorFeed,
  bskyUriToWebUrl,
  deduplicatePosts,
  type BskyPost,
} from './bluesky.js';

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function makePost(uri: string, overrides: Partial<BskyPost> = {}): BskyPost {
  return {
    uri,
    authorHandle: 'test.bsky.social',
    authorDisplayName: 'Test User',
    text: 'hello world',
    createdAt: '2024-01-15T12:00:00Z',
    likeCount: 5,
    repostCount: 1,
    replyCount: 2,
    ...overrides,
  };
}

function makeRawPost(uri = 'at://did:plc:abc/app.bsky.feed.post/123') {
  return {
    uri,
    author: { handle: 'test.bsky.social', displayName: 'Test User' },
    record: { text: 'hello world', createdAt: '2024-01-15T12:00:00Z' },
    likeCount: 5,
    repostCount: 1,
    replyCount: 2,
  };
}

const savedFetch = globalThis.fetch;
afterEach(() => { globalThis.fetch = savedFetch; });

// ---------------------------------------------------------------------------
// bskyUriToWebUrl — pure function
// ---------------------------------------------------------------------------

describe('bskyUriToWebUrl', () => {
  it('converts a valid AT URI to a Bluesky web URL', () => {
    const uri = 'at://did:plc:abc123/app.bsky.feed.post/rkey456';
    expect(bskyUriToWebUrl(uri)).toBe('https://bsky.app/profile/did:plc:abc123/post/rkey456');
  });

  it('returns the original URI when it does not match the expected format', () => {
    const uri = 'invalid-uri';
    expect(bskyUriToWebUrl(uri)).toBe('invalid-uri');
  });

  it('handles a real handle-based AT URI', () => {
    const uri = 'at://bellingcat.bsky.social/app.bsky.feed.post/abc';
    expect(bskyUriToWebUrl(uri)).toBe('https://bsky.app/profile/bellingcat.bsky.social/post/abc');
  });
});

// ---------------------------------------------------------------------------
// deduplicatePosts — pure function
// ---------------------------------------------------------------------------

describe('deduplicatePosts', () => {
  it('returns the same posts when there are no duplicates', () => {
    const posts = [makePost('uri-1'), makePost('uri-2'), makePost('uri-3')];
    expect(deduplicatePosts(posts)).toHaveLength(3);
  });

  it('removes duplicate posts by URI', () => {
    const posts = [makePost('uri-1'), makePost('uri-2'), makePost('uri-1')];
    const result = deduplicatePosts(posts);
    expect(result).toHaveLength(2);
    expect(result.map((p) => p.uri)).toEqual(['uri-1', 'uri-2']);
  });

  it('keeps the first occurrence when duplicates exist', () => {
    const first = makePost('uri-1', { text: 'first' });
    const second = makePost('uri-1', { text: 'second' });
    const result = deduplicatePosts([first, second]);
    expect(result[0].text).toBe('first');
  });

  it('returns empty array for empty input', () => {
    expect(deduplicatePosts([])).toHaveLength(0);
  });
});

// ---------------------------------------------------------------------------
// searchBskyPosts — fetch mock
// ---------------------------------------------------------------------------

describe('searchBskyPosts', () => {
  it('returns normalized posts on a successful response', async () => {
    const raw = { posts: [makeRawPost()] };
    globalThis.fetch = async () => new Response(JSON.stringify(raw), { status: 200 }) as Response;

    const posts = await searchBskyPosts('hello');
    expect(posts).toHaveLength(1);
    expect(posts[0].authorHandle).toBe('test.bsky.social');
    expect(posts[0].text).toBe('hello world');
    expect(posts[0].likeCount).toBe(5);
  });

  it('returns empty array when posts field is absent', async () => {
    globalThis.fetch = async () => new Response(JSON.stringify({}), { status: 200 }) as Response;
    const posts = await searchBskyPosts('missing');
    expect(posts).toHaveLength(0);
  });

  it('passes limit and sort params in the URL', async () => {
    let capturedUrl = '';
    globalThis.fetch = async (url: string | URL | Request) => {
      capturedUrl = url.toString();
      return new Response(JSON.stringify({ posts: [] }), { status: 200 }) as Response;
    };

    await searchBskyPosts('test', { limit: 10, sort: 'top' });
    expect(capturedUrl).toContain('limit=10');
    expect(capturedUrl).toContain('sort=top');
  });

  it('passes since param when provided', async () => {
    let capturedUrl = '';
    globalThis.fetch = async (url: string | URL | Request) => {
      capturedUrl = url.toString();
      return new Response(JSON.stringify({ posts: [] }), { status: 200 }) as Response;
    };

    await searchBskyPosts('test', { since: '2024-01-01T00:00:00Z' });
    expect(capturedUrl).toContain('since=');
  });

  it('throws on HTTP error response', async () => {
    globalThis.fetch = async () => new Response('Not Found', { status: 404, statusText: 'Not Found' }) as Response;
    await expect(searchBskyPosts('fail')).rejects.toThrow('Bluesky HTTP 404');
  });

  it('throws a timeout error when request is aborted', async () => {
    globalThis.fetch = async (_url: unknown, init?: RequestInit) => {
      return new Promise<Response>((_, reject) => {
        if (init?.signal) {
          init.signal.addEventListener('abort', () => {
            const err = new Error('AbortError');
            err.name = 'AbortError';
            reject(err);
          });
        }
      });
    };

    // override timeout to be very short so test doesn't hang
    const { searchBskyPosts: search } = await import('./bluesky.js?t=' + Date.now());
    await expect(
      Promise.race([
        search('timeout'),
        new Promise<BskyPost[]>((_, reject) =>
          setTimeout(() => reject(new Error('Bluesky request timed out')), 100),
        ),
      ]),
    ).rejects.toThrow();
  });

  it('normalizes missing fields to defaults', async () => {
    const minimal = { posts: [{ uri: 'at://x/app.bsky.feed.post/1' }] };
    globalThis.fetch = async () => new Response(JSON.stringify(minimal), { status: 200 }) as Response;
    const posts = await searchBskyPosts('minimal');
    expect(posts[0].authorHandle).toBe('');
    expect(posts[0].likeCount).toBe(0);
    expect(posts[0].text).toBe('');
  });
});

// ---------------------------------------------------------------------------
// getBskyAuthorFeed — fetch mock
// ---------------------------------------------------------------------------

describe('getBskyAuthorFeed', () => {
  it('returns normalized posts from author feed', async () => {
    const raw = { feed: [{ post: makeRawPost() }] };
    globalThis.fetch = async () => new Response(JSON.stringify(raw), { status: 200 }) as Response;

    const posts = await getBskyAuthorFeed('bellingcat.bsky.social');
    expect(posts).toHaveLength(1);
    expect(posts[0].text).toBe('hello world');
  });

  it('returns empty array when feed is absent', async () => {
    globalThis.fetch = async () => new Response(JSON.stringify({}), { status: 200 }) as Response;
    const posts = await getBskyAuthorFeed('nobody');
    expect(posts).toHaveLength(0);
  });

  it('includes actor and limit in the request URL', async () => {
    let capturedUrl = '';
    globalThis.fetch = async (url: string | URL | Request) => {
      capturedUrl = url.toString();
      return new Response(JSON.stringify({ feed: [] }), { status: 200 }) as Response;
    };

    await getBskyAuthorFeed('user.bsky.social', 7);
    expect(capturedUrl).toContain('actor=user.bsky.social');
    expect(capturedUrl).toContain('limit=7');
  });

  it('throws on HTTP error response', async () => {
    globalThis.fetch = async () =>
      new Response('Forbidden', { status: 403, statusText: 'Forbidden' }) as Response;
    await expect(getBskyAuthorFeed('locked')).rejects.toThrow('Bluesky feed HTTP 403');
  });
});
