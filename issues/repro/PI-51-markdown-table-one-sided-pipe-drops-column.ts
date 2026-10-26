/**
 * PI-51
 * A markdown table with only one outer pipe loses a whole header/data column
 * when rendered as a box table.
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const ANSI = /\u001B\[[0-?]*[ -/]*[@-~]/g;

function normalize(text: string, root: string): string {
  return text
    .replace(ANSI, '')
    .split(root).join('<TMP>')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/g, '<TS>');
}

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) {
    console.log('HARNESS-FAILURE: CS_REPRO_ROOT is unset');
    process.exitCode = 43;
    return;
  }
  const originalCwd = process.cwd();
  const originalEnv = { ...process.env };
  try {
    process.chdir(root);

    assert.equal(homedir(), root, 'HARNESS-FAILURE: homedir escaped CS_REPRO_ROOT');
    assert.equal(tmpdir(), root, 'HARNESS-FAILURE: tmpdir escaped CS_REPRO_ROOT');

    const listed = readFileSync(
      join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues/pre-work.envkeys'),
      'utf8',
    ).split('\n').map((line) => line.trim()).filter(Boolean);
    for (const name of listed) if (name !== 'HOME' && name !== 'USERPROFILE') delete process.env[name];
    for (const name of [
      'OLLAMA_BASE_URL', 'DEXTER_GATEWAY_CONFIG', 'DEXTER_PAIRING_PATH',
      'CRAMER_SHORT_SESSIONS_DIR', 'KALSHI_API_KEY', 'OPENROUTER_API_KEY',
      'E2E_MODEL', 'FORECAST_LAB_OUTPUT_METRICS', 'COVERAGE_LINES_MIN', 'TEST_API_KEY',
    ]) delete process.env[name];
    for (const name of Object.keys(process.env)) {
      if (/(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/.test(name)) delete process.env[name];
    }
    for (const name of Object.keys(process.env)) {
      if (/(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/.test(name)) {
        console.log(`HARNESS-FAILURE: secret survived ${name}`);
        process.exitCode = 43;
        return;
      }
    }

    const realFetch = globalThis.fetch;
    globalThis.fetch = (async (input: string | URL | Request) => {
      throw new Error(`network disabled: ${String(input)}`);
    }) as unknown as typeof fetch;
    try {
      const { formatResponse } = await import(
        pathToFileURL(join(REPO_ROOT, 'src/utils/ui/markdown-table.ts')).href
      );

      // GFM permits omitting the leading OR trailing pipe (not both). Both forms
      // are valid tables and must keep every cell when rendered.
      const inputs = [
        '| A | B\n|---|---|\n| 1 | 2',
        'A | B |\n|---|---|\n1 | 2 |',
      ];
      const wants = ['A', 'B', '1', '2'];
      const present = (text: string, cell: string) =>
        new RegExp(`(^|[^\\w])${cell}([^\\w]|$)`).test(text);

      for (const input of inputs) {
        const rendered = normalize(formatResponse(input), root);
        const missing = wants.filter((cell) => !present(rendered, cell));
        let ok = false;
        try {
          assert.deepEqual(missing, [], 'every header and data cell must survive table rendering');
          ok = true;
        } catch {
          ok = false;
        }
        if (!ok) {
          console.log('OBSERVED:');
          for (const line of rendered.split('\n')) console.log(`  ${line}`);
          console.log(`  input=${JSON.stringify(input)}`);
          console.log('EXPECTED:');
          console.log('  Both header cells (A, B) and both data cells (1, 2) appear in the rendered table.');
          console.log('ACTUAL:');
          console.log(`  Missing cells: ${missing.join(', ')}`);
          process.exitCode = 42;
          return;
        }
      }
      console.log('NO BUG');
    } finally {
      globalThis.fetch = realFetch;
    }
  } catch (err) {
    console.log(`HARNESS-FAILURE: ${err instanceof Error ? err.message : String(err)}`);
    process.exitCode = 43;
  } finally {
    process.chdir(originalCwd);
    for (const name of Object.keys(process.env)) if (!(name in originalEnv)) delete process.env[name];
    Object.assign(process.env, originalEnv);
  }
}

await main();
