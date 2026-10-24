import { afterEach, beforeEach, spyOn } from 'bun:test';

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

/**
 * Test-only: fail every test in the calling file that fetches a non-local
 * host. Such calls are blocked (they reject) and reported afterEach, so stub
 * them at the source instead. Call at module top level.
 */
export function useNoNetwork(): void {
  let blocked: string[] = [];
  let spy: ReturnType<typeof spyOn<typeof globalThis, 'fetch'>> | undefined;
  beforeEach(() => {
    blocked = [];
    const realFetch = globalThis.fetch;
    spy = spyOn(globalThis, 'fetch').mockImplementation(((input: Parameters<typeof fetch>[0], init?: RequestInit) => {
      const url = input instanceof Request ? input.url : String(input);
      if (LOCAL_HOSTS.has(new URL(url).hostname)) return realFetch(input, init);
      blocked.push(url);
      return Promise.reject(new Error(`unit test blocked network call to ${new URL(url).host}`));
    }) as typeof fetch);
  });
  afterEach(() => {
    spy?.mockRestore();
    if (blocked.length > 0) {
      throw new Error(`unit test made ${blocked.length} network call(s): ${[...new Set(blocked.map((u) => new URL(u).host))].join(', ')}`);
    }
  });
}
