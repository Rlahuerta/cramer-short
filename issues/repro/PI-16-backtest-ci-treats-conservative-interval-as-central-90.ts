/**
 * PI-16: backtest metrics read the conservative CI as a central-90 interval
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * walk-forward/replay store a *conservative* interval in ciLower/ciUpper
 * (survival thresholds 0.995/0.005) and the literal central-90 interval in
 * central90CiLower/central90CiUpper. crps/scaledCrps/tail-weighted CRPS,
 * murphyWinklerDecomposition and summarizeCalibrationSteps.coverageError all
 * assume ciLower/ciUpper is a central-90 interval, so every score is computed
 * against the wrong interval whenever the literal central-90 fields exist.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const EVIDENCE = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues');

class HarnessFailure extends Error {}

function harnessFailure(reason: string): never {
  console.log(`HARNESS-FAILURE: ${reason}`);
  throw new HarnessFailure(reason);
}

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) harnessFailure('CS_REPRO_ROOT is unset');
  if (homedir() !== root) harnessFailure(`homedir() ${homedir()} escaped ${root}`);
  if (tmpdir() !== root) harnessFailure(`tmpdir() ${tmpdir()} escaped ${root}`);

  const savedCwd = process.cwd();
  const savedEnv: Record<string, string | undefined> = { ...process.env };
  const keys = readFileSync(join(EVIDENCE, 'pre-work.envkeys'), 'utf-8')
    .split(/\s+/)
    .filter(Boolean);
  const extra = [
    'OLLAMA_BASE_URL', 'DEXTER_GATEWAY_CONFIG', 'DEXTER_PAIRING_PATH',
    'CRAMER_SHORT_SESSIONS_DIR', 'KALSHI_API_KEY', 'OPENROUTER_API_KEY',
    'E2E_MODEL', 'FORECAST_LAB_OUTPUT_METRICS', 'COVERAGE_LINES_MIN', 'TEST_API_KEY',
  ];
  const secretPattern = /(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/;

  try {
    process.chdir(root);
    for (const key of [...keys, ...extra]) {
      if (key !== 'HOME' && key !== 'USERPROFILE') delete process.env[key];
    }
    for (const key of Object.keys(process.env)) {
      if (secretPattern.test(key)) delete process.env[key];
    }
    for (const key of Object.keys(process.env)) {
      if (secretPattern.test(key)) harnessFailure(`secret survived: ${key}`);
    }
    globalThis.fetch = (async () => {
      throw new Error('network disabled');
    }) as typeof fetch;

    const metrics = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/backtest/metrics.ts')).href
    );
    const harness = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/backtest/comparison-harness.ts')).href
    );

    type Step = typeof metrics extends { crps: unknown } ? Parameters<typeof metrics.crps>[0][number] : never;
    const step = (overrides: Record<string, unknown>): Step => ({
      t: 0,
      predictedProb: 0.5,
      actualBinary: 1,
      predictedReturn: 0,
      actualReturn: 0.01,
      ciLower: 90,
      ciUpper: 110,
      realizedPrice: 101,
      recommendation: 'BUY',
      gofPasses: null,
      confidence: 0.8,
      ...overrides,
    } as Step);

    // Same literal central-90 interval; only the conservative helper differs.
    const narrow = step({ ciLower: 95, ciUpper: 105 });
    const widened = step({ ciLower: 90, ciUpper: 110, central90CiLower: 95, central90CiUpper: 105 });

    const crpsNarrow = metrics.crps([narrow]);
    const crpsWidened = metrics.crps([widened]);
    const mw = metrics.murphyWinklerDecomposition([widened]);
    // Realized 108 is inside the conservative interval but outside central-90.
    const coverageStep = step({
      ciLower: 90,
      ciUpper: 110,
      central90CiLower: 95,
      central90CiUpper: 105,
      realizedPrice: 108,
      actualReturn: 0.08,
    });
    const calibration = harness.summarizeCalibrationSteps([coverageStep], { targetCiCoverage: 0.9 });

    console.log('OBSERVED:');
    console.log(`  crps(ci=95..105, no central90) = ${crpsNarrow.toFixed(6)}`);
    console.log(`  crps(ci=90..110, central90=95..105) = ${crpsWidened.toFixed(6)}`);
    console.log(`  murphyWinkler meanWidth = ${mw.meanWidth.toFixed(6)}`);
    console.log(`  summarizeCalibrationSteps coverageError = ${calibration.coverageError.toFixed(6)}`);
    console.log(`  centralIntervalCoverage(0.9) = ${metrics.centralIntervalCoverage([coverageStep], 0.9).toFixed(6)}`);

    let failure: string | null = null;
    try {
      assert.ok(
        Math.abs(crpsNarrow - crpsWidened) < 1e-9,
        'CRPS changed when only the conservative helper interval changed (central-90 is identical)',
      );
      assert.ok(
        Math.abs(mw.meanWidth - 10) < 1e-9,
        'Murphy-Winkler meanWidth must be the central-90 width (10), not the conservative width (20)',
      );
      assert.ok(
        Math.abs(calibration.coverageError - 0.9) < 1e-9,
        'coverageError must use central-90 coverage (0/1) vs the 0.9 target, giving 0.9',
      );
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      console.log('EXPECTED:');
      console.log('  CRPS equal for identical central-90; MW meanWidth=10; coverageError=0.9');
      console.log('ACTUAL:');
      console.log(`  ${failure}`);
      process.exitCode = 42;
      return;
    }
    console.log('NO BUG');
  } finally {
    process.chdir(savedCwd);
    for (const key of Object.keys(process.env)) delete process.env[key];
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value !== undefined) process.env[key] = value;
    }
  }
}

try {
  await main();
} catch (error) {
  if (error instanceof HarnessFailure) {
    process.exitCode = 43;
  } else {
    throw error;
  }
}
