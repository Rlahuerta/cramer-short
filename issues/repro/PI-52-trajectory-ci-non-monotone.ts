/**
 * PI-52: trajectory CI can narrow as the horizon grows
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `computeTrajectory` documents (and its test asserts) that CI widths are
 * monotonically non-decreasing with horizon. Shared-path Monte Carlo does not
 * guarantee that for sample percentiles: with seed 76 the 90% band narrows from
 * 36.07 to 35.38 between day 11 and day 12. The Python mirror explicitly clamps
 * each day's band to the previous day's (`simulation.py` "Enforce monotonically
 * widening CIs"); the TS implementation has no such clamp.
 */

import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const FOUND_AT = 'e3e2088b21acc4963b3c20661e5d15c739244409';

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

async function main(): Promise<void> {
  let prepared: { root: string; restore: () => void } | undefined;
  try {
    prepared = await prepare();
    const root = prepared.root;

    const ci = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/markov-distribution/confidence-intervals.ts')).href
    );
    const transition = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/markov-distribution/transition.ts')).href
    );

    const stats = {
      bull: { meanReturn: 0.01, stdReturn: 0.03 },
      bear: { meanReturn: -0.01, stdReturn: 0.03 },
      sideways: { meanReturn: 0, stdReturn: 0.03 },
    };

    const trajectory = ci.computeTrajectory(
      100,                        // currentPrice
      14,                         // days
      transition.buildDefaultMatrix(),
      stats,
      'sideways',                 // initialState
      0,                          // momentumAdjustment
      undefined,                  // hmmOverride
      1000,                       // nSamples
      5,                          // nu
      undefined,                  // empiricalDailyVol
      undefined,                  // startMixture
      undefined,                  // jumpSpec
      undefined,                  // garchScales
      1.0,                        // uncertaintyCiScale
      76,                         // randomState
    );

    const widths = trajectory.map((p: { day: number; lowerBound: number; upperBound: number }) => ({
      day: p.day,
      width: p.upperBound - p.lowerBound,
    }));
    const violations = widths
      .slice(1)
      .map((entry: { day: number; width: number }, i: number) => ({
        day: entry.day,
        previousWidth: widths[i].width,
        width: entry.width,
      }))
      .filter((entry: { previousWidth: number; width: number }) => entry.width < entry.previousWidth - 0.05);

    const observed = JSON.stringify(
      {
        seed: 76,
        days: 14,
        nSamples: 1000,
        worstViolation: violations[0] ?? null,
        violations,
        days11to13: trajectory.slice(10, 13),
      },
      null,
      2,
    );
    const expected = "each trajectory day's 90% CI width is at least the previous day's width (within one cent of rounding)";
    const actual = violations.length === 0
      ? 'no narrowing observed'
      : `day ${violations[0].day} width ${violations[0].width.toFixed(2)} < day ${violations[0].day - 1} width ${violations[0].previousWidth.toFixed(2)}`;

    try {
      assert.equal(violations.length, 0, 'trajectory CI narrowed with horizon');
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
