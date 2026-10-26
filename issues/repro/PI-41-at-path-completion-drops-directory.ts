/**
 * PI-41: @-path completion drops the directory prefix after a trailing slash
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * Typing `@src` lists `src/`; selecting it produces `@src/`. The next
 * completion pass for `@src/` derives the relative path with
 * `join(dirname('src/'), name)`, and `dirname('src/') === '.'`, so the
 * suggestion value becomes `@cli.ts`. Applying that suggestion replaces
 * `@src/` with `@cli.ts`, silently attaching a same-named file from the
 * repo root instead of the file inside `src/`.
 *
 * The provider is constructed with `fdPath = null`, so the readdirSync
 * fallback runs and no child process is spawned.
 */
import assert from 'node:assert/strict';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));

const EXPLICIT_SCRUB = [
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
const SECRET_RE = /(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/;

function normalize(text: string, root: string): string {
  return text
    .replaceAll(root, '<TMP>')
    .replace(/\u001b\[[0-9;]*m/g, '')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/g, '<TS>');
}

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) {
    console.log('HARNESS-FAILURE: CS_REPRO_ROOT unset');
    process.exitCode = 43;
    return;
  }

  const originalCwd = process.cwd();
  const savedEnv = new Map<string, string | undefined>();
  const realFetch = globalThis.fetch;

  const fail = (reason: string): never => {
    console.log(`HARNESS-FAILURE: ${reason}`);
    process.exitCode = 43;
    throw new Error(`HARNESS-FAILURE: ${reason}`);
  };

  try {
    process.chdir(root);
    if (homedir() !== root) fail(`homedir escape: ${homedir()}`);
    if (tmpdir() !== root) fail(`tmpdir escape: ${tmpdir()}`);

    // --- env scrub (before every dynamic import) -------------------------
    const envKeysRaw = await readFile(
      join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues/pre-work.envkeys'),
      'utf8',
    );
    const keys = new Set<string>([
      ...envKeysRaw.split(/\r?\n/).map((k) => k.trim()).filter(Boolean),
      ...EXPLICIT_SCRUB,
    ]);
    for (const k of Object.keys(process.env)) {
      if (SECRET_RE.test(k)) keys.add(k);
    }
    keys.delete('HOME');
    keys.delete('USERPROFILE');
    for (const k of keys) {
      savedEnv.set(k, process.env[k]);
      delete process.env[k];
    }
    for (const k of Object.keys(process.env)) {
      if (SECRET_RE.test(k)) fail(`secret key survived scrub: ${k}`);
    }

    // --- network trap ----------------------------------------------------
    globalThis.fetch = (async () => {
      throw new Error('network disabled');
    }) as unknown as typeof globalThis.fetch;

    // --- fixture + production import after chdir + scrub -----------------
    const project = join(root, 'project');
    await mkdir(join(project, 'src'), { recursive: true });
    await writeFile(join(project, 'cli.ts'), '// root-level cli\n', 'utf8');
    await writeFile(join(project, 'src', 'cli.ts'), '// nested cli\n', 'utf8');

    const { AtPathAutocompleteProvider } = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/components/at-path-provider.ts')).href
    )) as typeof import('../../src/components/at-path-provider.ts');

    const provider = new AtPathAutocompleteProvider([], project, null);

    const line = '@src/';
    const result = provider.getSuggestions([line], 0, line.length);
    assert.ok(result, 'no suggestions returned for @src/');

    const item = result.items.find((entry) => entry.label === 'cli.ts');
    assert.ok(item, `no cli.ts suggestion; got ${result.items.map((i) => i.value).join(', ')}`);

    const applied = provider.applyCompletion([line], 0, line.length, item, result.prefix);
    const inserted = applied.lines[0] ?? '';

    const observed = `suggestionValue=${JSON.stringify(item.value)} insertedLine=${JSON.stringify(inserted)}`;
    assert.equal(item.value, '@src/cli.ts', `${observed}; directory prefix lost in suggestion value`);
    assert.equal(inserted.trimEnd(), '@src/cli.ts', `${observed}; completion rewrote the path`);

    console.log('NO BUG');
  } catch (error) {
    if (process.exitCode !== 43) {
      const message = error instanceof Error ? error.message : String(error);
      console.log('OBSERVED:');
      console.log(`  ${normalize(message, root)}`);
      console.log(
        'EXPECTED: @src/ completion suggests @src/cli.ts and inserts "@src/cli.ts "',
      );
      console.log(
        'ACTUAL: @src/ completion suggests @cli.ts and inserts "@cli.ts ", dropping the src/ directory',
      );
      process.exitCode = 42;
    }
  } finally {
    globalThis.fetch = realFetch;
    for (const [k, v] of savedEnv) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
    process.chdir(originalCwd);
  }
}

await main();
