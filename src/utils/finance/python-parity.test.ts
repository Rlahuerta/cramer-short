import { describe, test, expect } from 'bun:test';

// Import a fresh, cache-busted module instance so this assertion is independent
// of whether parity test files (which run in the same Bun process and call
// enablePythonParity()) have already enabled the shared module instance.
const freshModulePath = `./python-parity.js?gate=${Date.now()}-${Math.random()}`;

describe('python-parity execution gate', () => {
  test('runPython throws when parity is not enabled', async () => {
    const mod = (await import(freshModulePath)) as typeof import('./python-parity.js');
    await expect(mod.runPython('print(1)')).rejects.toThrow(/disabled outside parity tests/);
  });

  test('enablePythonParity is exported for test opt-in', async () => {
    const mod = (await import(freshModulePath)) as typeof import('./python-parity.js');
    expect(typeof mod.enablePythonParity).toBe('function');
  });
});
