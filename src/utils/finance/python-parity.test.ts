import { describe, test, expect, beforeEach, afterEach } from 'bun:test';
import { existsSync, mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

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

describe('pythonParityAvailable', () => {
  let fakeHome: string;

  beforeEach(() => {
    fakeHome = mkdtempSync(join(tmpdir(), 'parity-home-'));
  });

  afterEach(() => {
    rmSync(fakeHome, { recursive: true, force: true });
  });

  // os.homedir() is cached per process, so evaluate the gate in a child with a fake $HOME.
  function runGate(env: Record<string, string> = {}): { available: string; stderr: string } {
    const modulePath = join(import.meta.dir, 'python-parity.ts');
    const { REQUIRE_PARITY: _ignored, ...baseEnv } = process.env;
    const proc = Bun.spawnSync({
      cmd: [process.execPath, '-e', `const m = await import(${JSON.stringify(modulePath)}); console.log(m.pythonParityAvailable());`],
      env: { ...baseEnv, HOME: fakeHome, ...env },
    });
    return { available: proc.stdout.toString().trim(), stderr: proc.stderr.toString() };
  }

  test('is true when the cramer-research conda env exists', () => {
    const binDir = join(fakeHome, 'miniconda3', 'envs', 'cramer-research', 'bin');
    mkdirSync(binDir, { recursive: true });
    writeFileSync(join(binDir, 'python'), '');
    expect(runGate().available).toBe('true');
  });

  test.skipIf(['/opt/anaconda3', '/opt/miniconda3'].some((root) => existsSync(join(root, 'envs', 'cramer-research'))))(
    'is false with a skip warning when the env is missing',
    () => {
      const { available, stderr } = runGate();
      expect(available).toBe('false');
      expect(stderr).toMatch(/cramer-research.*REQUIRE_PARITY=1/s);
    },
  );

  test('REQUIRE_PARITY=1 keeps parity tests running even when the env is missing', () => {
    expect(runGate({ REQUIRE_PARITY: '1' }).available).toBe('true');
  });
});
