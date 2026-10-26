/**
 * PI-14: Conformal reset drops the configured initial radius
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * TS `ConformalPID.reset()` with no radius keeps the adapted radius; the
 * Python mirror (`research/models/conformal.py`) restores the constructor's
 * `initial_radius` (asserted by research/tests/test_conformal_parity.py:244).
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const ENV_KEYS_PATH = join(
  REPO_ROOT,
  '.omo/evidence/bug-hunt-reproducible-issues/pre-work.envkeys',
);
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
const SECRET_RE = /(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/;

function normalize(text: string, root: string): string {
  const stripped = text.replace(/\u001b\[[0-9;]*m/g, '');
  return stripped
    .replaceAll(root, '<TMP>')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/g, '<TS>');
}

function harnessFailure(reason: string): void {
  console.log(`HARNESS-FAILURE: ${reason}`);
  process.exitCode = 43;
}

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) {
    harnessFailure('CS_REPRO_ROOT is unset');
    return;
  }

  const originalCwd = process.cwd();
  const originalEnv = { ...process.env };
  try {
    process.chdir(root);

    if (homedir() !== root) {
      harnessFailure(`homedir escaped to ${homedir()}`);
      return;
    }
    if (tmpdir() !== root) {
      harnessFailure(`tmpdir escaped to ${tmpdir()}`);
      return;
    }

    const envKeys = readFileSync(ENV_KEYS_PATH, 'utf8')
      .split('\n')
      .map((line) => line.trim())
      .filter(Boolean);
    for (const key of envKeys) {
      if (key !== 'HOME' && key !== 'USERPROFILE') delete process.env[key];
    }
    for (const key of EXTRA_SCRUB) delete process.env[key];
    for (const key of Object.keys(process.env)) {
      if (SECRET_RE.test(key)) delete process.env[key];
    }
    for (const key of Object.keys(process.env)) {
      if (SECRET_RE.test(key)) {
        harnessFailure(`secret env var survived scrub: ${key}`);
        return;
      }
    }

    globalThis.fetch = (async () => {
      throw new Error('network disabled');
    }) as unknown as typeof fetch;

    const { ConformalPID } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/conformal.js')).href
    );

    const initialRadius = 1.0;
    const pid = new ConformalPID({ initialRadius, alpha: 0.1 });
    pid.record(0, 5); // miss: residual 5 > 1 => the PID widens q
    const radiusAfterMiss = pid.currentRadius();
    pid.reset();
    const radiusAfterReset = pid.currentRadius();

    let mismatch: { expected: number; actual: number } | undefined;
    try {
      assert.equal(radiusAfterReset, initialRadius);
    } catch {
      mismatch = { expected: initialRadius, actual: radiusAfterReset };
    }

    if (!mismatch) {
      console.log('NO BUG');
      return;
    }

    const observed = normalize(
      [
        `initialRadius=${initialRadius}`,
        `radiusAfterMiss=${radiusAfterMiss}`,
        `radiusAfterReset=${radiusAfterReset}`,
      ].join('\n'),
      root,
    );
    console.log('OBSERVED:');
    for (const line of observed.split('\n')) console.log(`  ${line}`);
    console.log('EXPECTED:');
    console.log(`  radiusAfterReset=${mismatch.expected}`);
    console.log('ACTUAL:');
    console.log(`  radiusAfterReset=${mismatch.actual}`);
    process.exitCode = 42;
  } finally {
    process.chdir(originalCwd);
    for (const key of Object.keys(process.env)) {
      if (!(key in originalEnv)) delete process.env[key];
    }
    for (const [key, value] of Object.entries(originalEnv)) {
      if (value !== undefined) process.env[key] = value;
    }
  }
}

await main();
