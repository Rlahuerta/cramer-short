/**
 * PI-3: edit_file fuzzy fallback rewrites unrelated lines of the file
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `fuzzyFindText` normalises trailing whitespace, smart quotes, en/em dashes and
 * exotic spaces across the WHOLE file (`normalizeForFuzzyMatch`) and then returns
 * that normalised copy as `contentForReplacement`. `edit_file` builds the new file
 * from that copy, so whenever the exact match fails and the fuzzy path is taken,
 * every untouched line is silently rewritten: trailing spaces disappear and
 * typographic characters become their ASCII equivalents. A file containing
 * `const c = “x”;` becomes `const c = "x";` after an edit somewhere else.
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

const ORIGINAL = 'const a = \u201cit\u2019s\u201d;\nconst b = 1;   \nconst c = \u201Cx\u201D;\n';
const REPLACED = 'const a = "REPLACED";\nconst b = 1;   \nconst c = \u201Cx\u201D;\n';

async function main(): Promise<void> {
  let prepared: { root: string; restore: () => void } | undefined;
  try {
    prepared = await prepare();
    const root = prepared.root;

    const { editFileTool } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/filesystem/edit-file.ts')).href
    );

    await writeFile('sample.ts', ORIGINAL, 'utf-8');
    // Straight quotes in old_text force the fuzzy path (the file has curly quotes).
    await editFileTool.invoke({
      path: 'sample.ts',
      old_text: 'const a = "it\'s";',
      new_text: 'const a = "REPLACED";',
    });
    const after = await readFile('sample.ts', 'utf-8');

    const beforeLines = ORIGINAL.split('\n');
    const afterLines = after.split('\n');
    const untouchedLinesRewritten = afterLines
      .map((line, index) => ({ before: beforeLines[index], after: line }))
      .filter((entry, index) => index > 0 && entry.before !== entry.after);

    const observed = JSON.stringify(
      {
        before: ORIGINAL,
        after,
        untouchedLinesRewritten,
        targetLineApplied: after.startsWith('const a = "REPLACED";'),
      },
      null,
      2,
    );
    const expected = 'only the edited line changes; every untouched line keeps its exact bytes';
    const actual = `lines 2-3 rewritten: ${JSON.stringify(ORIGINAL.split('\n').slice(1, 3))} -> ${JSON.stringify(after.split('\n').slice(1, 3))}`;

    try {
      assert.equal(after, REPLACED, 'fuzzy edit rewrote untouched lines');
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
