/**
 * PI-59
 * Box-table column widths strip snake_case underscores as if they were italic
 * markers, but the italic pass refuses to style them, so rows overflow the border.
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

      // Snake_case identifiers are common in financial tables and are NOT
      // markdown italics (the italic pass guards with word boundaries).
      const input = '| metric | note |\n|---|---|\n| foo_bar_baz | x |';
      const rendered = normalize(formatResponse(input), root);
      const lines = rendered.split('\n');
      const widths = lines.map((line) => [...line].length);
      const aligned = new Set(widths).size === 1;

      let ok = false;
      try {
        assert.ok(aligned, 'every rendered table line must have the same visible width');
        ok = true;
      } catch {
        ok = false;
      }
      if (!ok) {
        console.log('OBSERVED:');
        for (const line of rendered.split('\n')) console.log(`  ${line}`);
        console.log(`  line widths: ${widths.join(', ')}`);
        console.log('EXPECTED:');
        console.log('  All rendered table lines have equal visible width (box borders align).');
        console.log('ACTUAL:');
        console.log(`  The data row is ${Math.max(...widths) - Math.min(...widths)} columns wider than the border.`);
        process.exitCode = 42;
        return;
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
