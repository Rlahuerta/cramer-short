/**
 * PI-46: write_file rejects the documented @~/ home path form
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `expandPath` supports the `@` prefix (`normalizeAtPrefix`) and its unit test
 * asserts `expandPath('@~') === homedir()`; the TUI/README hand the model
 * `@~/reports/...` paths. But `writeFileTool` decides the sandbox root from the
 * RAW string (`input.path.startsWith('~/')`), so `@~/reports/x.md` is treated as
 * cwd-relative, resolves to `<home>/reports/x.md`, and is then rejected with
 * "Path escapes sandbox root". The same path without `@` succeeds, so a
 * documented invocation fails for a reason that has nothing to do with the
 * sandbox it is supposed to enforce.
 */

import assert from 'node:assert/strict';
import { mkdir, readFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));

class HarnessFailure extends Error {}

const ANSI = /\x1b\[[0-9;?]*[ -/]*[@-~]/g;
function normalize(text: string, root: string): string {
  return text
    .replace(ANSI, '')
    .split(root)
    .join('<TMP>')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/g, '<TS>');
}

const EXTRA_SCRUB = [
  'OLLAMA_BASE_URL',
  'DEXTER_GATEWAY_CONFIG',
  'DEXTER_PAIRING_PATH',
  'CRAMER_SHORT_SESSIONS_DIR',
  'KALSHI_API_KEY',
  'OPENROUTER_API_KEY',
  'E2E_MODEL',
  'FORECAST_LAB_OUTPUT_METRICS',
  'COVERAGE_LINES_MIN',
  'TEST_API_KEY',
];

async function prepare(): Promise<{ root: string; restore: () => void }> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) throw new HarnessFailure('CS_REPRO_ROOT is not set');

  const originalCwd = process.cwd();
  const saved = new Map<string, string | undefined>();
  const remember = (key: string) => {
    if (!saved.has(key)) saved.set(key, process.env[key]);
  };

  process.chdir(root);
  if (homedir() !== root) throw new HarnessFailure(`homedir escaped run root: ${homedir()}`);
  if (tmpdir() !== root) throw new HarnessFailure(`tmpdir escaped run root: ${tmpdir()}`);

  const envKeysPath = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues/pre-work.envkeys');
  const listed = (await readFile(envKeysPath, 'utf8')).split(/\r?\n/).map((s) => s.trim()).filter(Boolean);
  const exempt = new Set(['HOME', 'USERPROFILE']);
  for (const key of new Set([...listed, ...EXTRA_SCRUB])) {
    if (exempt.has(key)) continue;
    remember(key);
    delete process.env[key];
  }
  for (const key of Object.keys(process.env)) {
    if (exempt.has(key)) continue;
    if (/(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/i.test(key)) {
      remember(key);
      delete process.env[key];
    }
  }
  const surviving = Object.keys(process.env).filter(
    (key) => !exempt.has(key) && /(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/i.test(key),
  );
  if (surviving.length > 0) throw new HarnessFailure(`surviving secret-like env keys: ${surviving.join(',')}`);

  globalThis.fetch = (async (input: unknown) => {
    throw new Error(`network disabled: ${String(input)}`);
  }) as unknown as typeof fetch;

  return {
    root,
    restore: () => {
      process.chdir(originalCwd);
      for (const [key, value] of saved) {
        if (value === undefined) delete process.env[key];
        else process.env[key] = value;
      }
    },
  };
}

async function attempt(
  tool: { invoke: (input: { path: string; content: string }) => Promise<unknown> },
  path: string,
): Promise<{ path: string; ok: boolean; detail: string }> {
  try {
    await tool.invoke({ path, content: 'repro\n' });
    return { path, ok: true, detail: 'written' };
  } catch (error) {
    return { path, ok: false, detail: (error as Error).message };
  }
}

async function main(): Promise<void> {
  let prepared: { root: string; restore: () => void } | undefined;
  try {
    prepared = await prepare();
    const root = prepared.root;

    // The agent's cwd is a project folder inside home, as in a normal checkout.
    await mkdir('repo', { recursive: true });
    process.chdir('repo');

    const { writeFileTool } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/filesystem/write-file.ts')).href
    );
    const { expandPath } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/filesystem/utils/path-utils.ts')).href
    );

    const documented = await attempt(writeFileTool as never, '@~/reports/analysis.md');
    const control = await attempt(writeFileTool as never, '~/reports/analysis.md');

    const observed = JSON.stringify(
      {
        expandPathAtTilde: expandPath('@~') === root ? '<HOME>' : expandPath('@~'),
        documented,
        control,
      },
      null,
      2,
    );
    const expected = '@~/reports/analysis.md is accepted exactly like ~/reports/analysis.md';
    const actual = `@~/reports/analysis.md ok=${documented.ok} (${documented.detail}); ~/reports/analysis.md ok=${control.ok}`;

    try {
      assert.equal(control.ok, true, 'control path should succeed');
      assert.equal(documented.ok, true, 'at-prefixed home path was rejected');
    } catch {
      console.log('OBSERVED:');
      for (const line of normalize(observed, root).split('\n')) console.log(`  ${line}`);
      console.log(`EXPECTED: ${expected}`);
      console.log(`ACTUAL: ${actual}`);
      process.exitCode = 42;
      return;
    }

    console.log('NO BUG');
  } catch (error) {
    if (error instanceof HarnessFailure) {
      console.log(`HARNESS-FAILURE: ${error.message}`);
      process.exitCode = 43;
      return;
    }
    throw error;
  } finally {
    prepared?.restore();
  }
}

await main();
