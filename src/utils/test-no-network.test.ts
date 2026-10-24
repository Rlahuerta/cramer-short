import { expect, test } from 'bun:test';
import * as dns from 'node:dns';
import { lookup } from 'node:dns/promises';
import { NETWORK_GUARD_ACTIVE, takeBlockedCalls } from './test-no-network.js';

test.skipIf(!NETWORK_GUARD_ACTIVE)('the unit-tier preload blocks and reports non-local fetches', async () => {
  await expect(fetch('https://example.com/data')).rejects.toThrow('blocked network call to example.com');
  expect(takeBlockedCalls()).toEqual(['https://example.com/data']);
});

test.skipIf(!NETWORK_GUARD_ACTIVE)('the unit-tier preload blocks and reports non-local DNS lookups', async () => {
  await expect(lookup('example.com', { all: true })).rejects.toThrow('blocked DNS lookup for example.com');
  const callbackError = await new Promise<Error | null>((resolve) => dns.lookup('example.org', (error) => resolve(error)));
  expect(callbackError?.message).toContain('blocked DNS lookup for example.org');
  expect(takeBlockedCalls()).toEqual(['dns://example.com', 'dns://example.org']);

  expect((await lookup('localhost')).address).toMatch(/^(127\.0\.0\.1|::1)$/);
});
