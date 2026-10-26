/**
 * PI-4: cross-platform merge treats a below-market raw YES probability as P(above)
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `extractPriceThresholds` stores the raw YES probability for both "above" and
 * "below" markets, but `PriceThreshold.probability` is always P(price > strike).
 * For a below market the two are complementary. `mergeAnchorsWithCrossPlatformValidation`
 * compares and averages `rawProbability` as if it were P(above), so two agreeing
 * platforms merge to ~0.49 instead of ~0.24 and a false 56pp divergence warning fires.
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

    const md = await import(pathToFileURL(join(REPO_ROOT, 'src/tools/finance/markov-distribution.ts')).href);

    const referenceTimeMs = Date.parse('2026-03-01T00:00:00Z');
    const endDate = new Date(referenceTimeMs + 7 * 86_400_000).toISOString();
    const markets = [
      {
        question: 'Will BTC close below $90,000 on March 8, 2026?',
        probability: 0.8,
        volume: 100_000,
        createdAt: referenceTimeMs - 30 * 86_400_000,
        endDate,
      },
    ];

    const anchors = md.extractPriceThresholds(markets, {
      ticker: 'SPY',
      horizonDays: 7,
      referenceTimeMs,
    });
    const poly = anchors[0];

    // Kalshi prices the same strike at the same P(above); the two platforms agree.
    const kalshi = [{ price: 90_000, probability: 0.24, volume: 10_000 }];
    const merged = md.mergeAnchorsWithCrossPlatformValidation(anchors, kalshi);
    const mergedAnchor = merged.anchors.find((a: { price: number }) => a.price === 90_000);

    const observed = JSON.stringify(
      {
        polymarket: { price: poly?.price, rawYes: poly?.rawProbability, pAbove: poly?.probability },
        kalshi: kalshi[0],
        merged: mergedAnchor,
        warnings: merged.warnings,
      },
      null,
      2,
    );
    const expected =
      'merged P(above) stays near the agreed 0.24 and no divergence warning is emitted';
    const actual = `merged P(above)=${String(mergedAnchor?.probability)} with ${merged.warnings.length} spurious warning(s)`;

    try {
      assert.ok(
        poly !== undefined
          && Math.abs(poly.probability - 0.24) < 1e-9
          && mergedAnchor !== undefined
          && Math.abs(mergedAnchor.probability - 0.24) <= 0.02
          && merged.warnings.length === 0,
        'below-anchor merge inverted the survival probability',
      );
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
