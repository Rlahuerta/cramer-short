/**
 * PI-57
 * The dependency-cruiser layering rules only match CLI paths with a trailing
 * slash, so top-level entry files such as src/cli.ts are outside the guard.
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: static
 *
 * Execution is impossible: the artifact is a declarative dependency-cruiser
 * config, and running depcruise would traverse the whole repository/node_modules
 * instead of testing the one rule under audit, so the rule pattern is asserted
 * directly against the paths it is meant to cover.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const ARTIFACT = '.dependency-cruiser.cjs';
const CLI_LAYER_PATHS = [
  'src/cli.ts',
  'src/cli-schedule.ts',
  'src/cli-forecast-lab.ts',
  'src/cli-replay-label.ts',
  'src/components/chat-log.ts',
  'src/controllers/slash-commands/core.ts',
];

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
      const artifactPath = join(REPO_ROOT, ARTIFACT);
      const raw = readFileSync(artifactPath, 'utf8');
      const loaded = await import(pathToFileURL(artifactPath).href);
      const config = (loaded.default ?? loaded) as {
        forbidden: Array<{ name: string; from?: { path?: string }; to?: { path?: string } }>;
      };

      const rules = config.forbidden.filter(
        (rule) => rule.name === 'tools-no-import-cli' || rule.name === 'agent-no-import-cli',
      );
      assert.equal(rules.length, 2, 'HARNESS-FAILURE: both CLI layering rules must exist');

      const misses: string[] = [];
      for (const rule of rules) {
        assert.ok(rule.to?.path, `HARNESS-FAILURE: ${rule.name} must have a to.path`);
        const pattern = new RegExp(rule.to.path);
        for (const candidate of CLI_LAYER_PATHS) {
          if (!pattern.test(candidate)) misses.push(`${rule.name} does not match ${candidate}`);
        }
      }

      let ok = false;
      try {
        assert.deepEqual(misses, [], 'both CLI layering rules must cover every CLI-layer entry path');
        ok = true;
      } catch {
        ok = false;
      }
      if (!ok) {
        const numbered = raw.split('\n')
          .map((line, index) => ({ line, number: index + 1 }))
          .filter((entry) => entry.line.includes('^src/(cli|components|controllers)/'));
        console.log('OBSERVED:');
        for (const entry of numbered) console.log(`  ${entry.number}: ${entry.line.trim()}`);
        for (const miss of misses) console.log(`  ${miss}`);
        console.log('EXPECTED:');
        console.log('  Every CLI-layer entry file (src/cli.ts, src/cli-schedule.ts, ...) matches both layering rules.');
        console.log('ACTUAL:');
        console.log(`  ${misses.length} uncovered CLI-layer path(s); the rules only match paths with a trailing slash.`);
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
