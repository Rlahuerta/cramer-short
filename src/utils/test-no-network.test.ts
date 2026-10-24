import { expect, test } from 'bun:test';
import { NETWORK_GUARD_ACTIVE, takeBlockedFetches } from './test-no-network.js';

test.skipIf(!NETWORK_GUARD_ACTIVE)('the unit-tier preload blocks and reports non-local fetches', async () => {
  await expect(fetch('https://example.com/data')).rejects.toThrow('blocked network call to example.com');
  expect(takeBlockedFetches()).toEqual(['https://example.com/data']);
});
