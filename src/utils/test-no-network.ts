import { afterEach, beforeEach, spyOn } from 'bun:test';
import * as dns from 'node:dns';
import * as dnsPromises from 'node:dns/promises';
import { isIP } from 'node:net';

const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]']);

/**
 * Unit-tier preload (bunfig.toml `[test] preload`): unit tests must not reach
 * the network. Non-local fetches and node:dns lookups fail, and the next
 * afterEach fails with the hosts they tried, so stub them at the source. Off when RUN_INTEGRATION
 * or RUN_E2E opts into a live tier (scripts/run-isolated-bun-tests.ts).
 */
export const NETWORK_GUARD_ACTIVE = !['RUN_INTEGRATION', 'RUN_E2E'].some((name) =>
  /^(1|true|yes|on)$/i.test(process.env[name] ?? ''),
);

let blocked: string[] = [];

/** Test-only: return and clear the calls the guard blocked since the last check. */
export function takeBlockedCalls(): string[] {
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
  // Spies, not assignment: named imports of node builtins only see a namespace spy.
  const realLookup = dns.lookup;
  const realPromisesLookup = dnsPromises.lookup;
  const isLocal = (hostname: string) => hostname === 'localhost' || isIP(hostname) !== 0;
  const blockDns = (hostname: string) => {
    blocked.push(`dns://${hostname}`);
    return new Error(`unit test blocked DNS lookup for ${hostname}`);
  };
  const installDnsGuard = () => {
    spyOn(dnsPromises, 'lookup').mockImplementation(((hostname: string, ...rest: [never]) =>
      isLocal(hostname) ? realPromisesLookup(hostname, ...rest) : Promise.reject(blockDns(hostname))
    ) as typeof dnsPromises.lookup);
    spyOn(dns, 'lookup').mockImplementation(((hostname: string, ...rest: [never, never]) => {
      if (isLocal(hostname)) return realLookup(hostname, ...rest);
      const callback = rest.at(-1) as unknown as (error: Error) => void;
      process.nextTick(() => callback(blockDns(hostname)));
    }) as typeof dns.lookup);
  };
  installDnsGuard();
  // Re-arm per test: a test's mock.restore() resets every spy, these included.
  beforeEach(installDnsGuard);
  afterEach(() => {
    const urls = takeBlockedCalls();
    if (urls.length > 0) {
      const hosts = [...new Set(urls.map((u) => new URL(u).host))].join(', ');
      throw new Error(`unit test made ${urls.length} network call(s): ${hosts}. Stub them at the source.`);
    }
  });
}
