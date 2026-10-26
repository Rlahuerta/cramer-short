/**
 * PI-50
 * A non-numeric COVERAGE_LINES_MIN becomes NaN, the comparison is always false,
 * and check-coverage.ts reports success for any coverage instead of failing.
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 */
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
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
      // A valid lcov record with 0% line coverage. The gate must reject the
      // invalid threshold; instead it treats it as NaN and passes everything.
      mkdirSync(join(root, 'coverage'), { recursive: true });
      writeFileSync(join(root, 'coverage/lcov.info'), 'SF:sample.ts\nLF:100\nLH:0\nend_of_record\n');
      process.env.COVERAGE_LINES_MIN = 'not-a-number';

      let rejected = false;
      let errorMessage = '';
      try {
        await import(pathToFileURL(join(REPO_ROOT, 'scripts/check-coverage.ts')).href);
      } catch (err) {
        rejected = true;
        errorMessage = err instanceof Error ? err.message : String(err);
      }

      let ok = false;
      try {
        assert.ok(rejected, 'an invalid COVERAGE_LINES_MIN must be rejected, not silently disabled');
        ok = true;
      } catch {
        ok = false;
      }
      if (!ok) {
        console.log('OBSERVED:');
        console.log('  COVERAGE_LINES_MIN="not-a-number"');
        console.log('  coverage/lcov.info: LF:100 LH:0 (0.00% line coverage)');
        console.log(`  import resolved without error (rejected=${rejected})`);
        console.log('EXPECTED:');
        console.log('  The gate throws for an invalid threshold before evaluating coverage.');
        console.log('ACTUAL:');
        console.log('  The threshold becomes NaN, the comparison is always false, so 0% coverage passes the gate.');
        process.exitCode = 42;
        return;
      }
      assert.ok(errorMessage.length > 0, 'HARNESS-FAILURE: rejection must carry a message');
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
