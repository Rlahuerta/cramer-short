/**
 * PI-58
 * The rendered-line cache key truncates content to 120 characters, so two
 * different answers of equal length with a shared prefix collide and the second
 * answer gets the first answer's line count.
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
      const { countRenderedTuiMarkdownLines, renderTuiMarkdownLines } = await import(
        pathToFileURL(join(REPO_ROOT, 'src/utils/ui/markdown-table.ts')).href
      );

      const width = 40;
      const sharedPrefix = 'x'.repeat(120);
      const firstAnswer = `${sharedPrefix}\n${'b'.repeat(19)}`;
      // Same length (140) and same first 120 chars as firstAnswer.
      const secondAnswer = `${sharedPrefix}\n${Array.from({ length: 10 }, () => 'c').join('\n')}`;
      assert.equal(firstAnswer.length, secondAnswer.length, 'HARNESS-FAILURE: fixtures must collide on length');

      // Ground truth comes from the renderer, which never consults the cache.
      const trueFirst = renderTuiMarkdownLines(firstAnswer, width).length;
      const trueSecond = renderTuiMarkdownLines(secondAnswer, width).length;
      assert.notEqual(trueFirst, trueSecond, 'HARNESS-FAILURE: fixtures must render a different number of lines');

      // Production path: count the first answer (seeds the cache), then the second.
      const countedFirst = countRenderedTuiMarkdownLines(firstAnswer, width);
      const countedSecond = countRenderedTuiMarkdownLines(secondAnswer, width);

      let ok = false;
      try {
        assert.equal(countedFirst, trueFirst, 'first answer line count');
        assert.equal(countedSecond, trueSecond, 'second answer line count must not reuse the first cache entry');
        ok = true;
      } catch {
        ok = false;
      }
      if (!ok) {
        console.log('OBSERVED:');
        console.log(`  firstAnswer  true=${trueFirst} counted=${countedFirst}`);
        console.log(`  secondAnswer true=${trueSecond} counted=${countedSecond}`);
        console.log(`  cache key (truncated content) = ${secondAnswer.length}:${width}:${sharedPrefix.slice(0, 40)}...`);
        console.log('EXPECTED:');
        console.log('  countRenderedTuiMarkdownLines(secondAnswer) equals the renderer line count for secondAnswer.');
        console.log('ACTUAL:');
        console.log(`  countRenderedTuiMarkdownLines(secondAnswer) returned ${countedSecond} (the first answer's ${trueFirst}).`);
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
