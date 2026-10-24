import { afterEach } from 'bun:test';

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

/**
 * Unit-tier preload (bunfig.toml `[test] preload`): unit tests must not reach
 * the network. Non-local fetches reject, and the next afterEach fails with
 * the hosts they tried, so stub them at the source. Off when RUN_INTEGRATION
 * or RUN_E2E opts into a live tier (scripts/run-isolated-bun-tests.ts).
 */
export const NETWORK_GUARD_ACTIVE = !['RUN_INTEGRATION', 'RUN_E2E'].some((name) =>
  /^(1|true|yes|on)$/i.test(process.env[name] ?? ''),
);

let blocked: string[] = [];

/** Test-only: return and clear the fetches the guard blocked since the last check. */
export function takeBlockedFetches(): string[] {
  const taken = blocked;
  blocked = [];
  return taken;
}

if (NETWORK_GUARD_ACTIVE) {
  const realFetch = globalThis.fetch;
  globalThis.fetch = Object.assign(
    (input: Parameters<typeof fetch>[0], init?: RequestInit) => {
      const url = input instanceof Request ? input.url : String(input);
      const { host, hostname } = new URL(url);
      if (LOCAL_HOSTS.has(hostname)) return realFetch(input, init);
      blocked.push(url);
      return Promise.reject(new Error(`unit test blocked network call to ${host}`));
    },
    { preconnect: realFetch.preconnect },
  );
  afterEach(() => {
    const urls = takeBlockedFetches();
    if (urls.length > 0) {
      const hosts = [...new Set(urls.map((u) => new URL(u).host))].join(', ');
      throw new Error(`unit test made ${urls.length} network call(s): ${hosts}. Stub them at the source.`);
    }
  });
}
