/**
 * PI-10: read_file secret deny-list misses .npmrc/.netrc and other key stores
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `isSecretPath` blocks `.env*`, `*.pem`, `*.key`, `.cramer-short/settings.json`
 * and `.cramer-short/credentials/**`, but it matches on the basename only, so
 * other canonical secret stores inside the workspace are read straight into the
 * model context: `.npmrc` (registry auth token), `.netrc` (machine credentials),
 * `*.p12`, and `~/.ssh/id_rsa` when the agent runs from the home directory.
 *
 * The script writes throwaway fixtures under the temp run root — no real secret
 * is touched or printed.
 */

import assert from 'node:assert/strict';
import { readFile, writeFile } from 'node:fs/promises';
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

type Probe = { path: string; read: boolean; snippet: string };

async function probe(
  tool: { invoke: (input: { path: string }) => Promise<unknown> },
  path: string,
): Promise<Probe> {
  try {
    const raw = String(await tool.invoke({ path }));
    const data = JSON.parse(raw).data as { content?: string };
    return { path, read: true, snippet: String(data.content ?? '').slice(0, 48) };
  } catch {
    return { path, read: false, snippet: '' };
  }
}

async function main(): Promise<void> {
  let prepared: { root: string; restore: () => void } | undefined;
  try {
    prepared = await prepare();
    const root = prepared.root;

    const { readFileTool } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/filesystem/read-file.ts')).href
    );

    await writeFile('.npmrc', '//registry.npmjs.org/:_authToken=FAKE_NPM_TOKEN\n', 'utf-8');
    await writeFile('.netrc', 'machine example.com login bot password FAKE_NETRC_PASS\n', 'utf-8');
    await writeFile('.env', 'FAKE_ENV_VALUE=1\n', 'utf-8');

    const results = [
      await probe(readFileTool, '.npmrc'),
      await probe(readFileTool, '.netrc'),
      await probe(readFileTool, '.env'),
    ];

    const leaked = results.filter((result) => result.path !== '.env' && result.read).map((result) => result.path);
    const observed = JSON.stringify(
      {
        results,
        blockedControl: results.find((result) => result.path === '.env')?.read === false,
        leakedSecretStores: leaked,
      },
      null,
      2,
    );
    const expected = 'read_file denies .npmrc and .netrc the same way it denies .env';
    const actual = `.npmrc read=${results[0].read}, .netrc read=${results[1].read}, .env read=${results[2].read}`;

    try {
      assert.deepEqual(leaked, [], 'secret-bearing paths were readable');
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
