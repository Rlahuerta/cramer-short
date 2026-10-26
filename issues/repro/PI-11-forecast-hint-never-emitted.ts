/**
 * PI-11: BTC short-horizon abstain never emits its forecastHint
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * The tool builds `forecastHint` only on the canonical branch, but
 * `buildForecastHint` itself returns null when `canEmitCanonical` is true.
 * The two conditions are mutually exclusive, so the hint is always null.
 * The only consumer (`query-router/tool-call-extractors.ts`) reads it in the
 * `status === 'abstain'` branch, where it is never produced.
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
    const xiphos = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/markov-distribution/xiphos-btc-profile.ts')).href
    );

    const prices = Array.from({ length: 40 }, (_, i) => Math.round((100 + i * 0.4) * 100) / 100);
    const currentPrice = prices[prices.length - 1];
    const polymarketMarkets = [{ question: 'Will it rain tomorrow?', probability: 0.5 }];

    const raw = await (md.markovDistributionTool as any).func({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets,
      trajectory: false,
    });
    const payload = JSON.parse(raw).data;
    const diagnostics = payload.canonical?.diagnostics;

    // Same deterministic computation the wrapper runs, so the helper receives the
    // exact expectedReturn / mixingTimeWeight the tool used.
    const direct = await md.computeMarkovDistribution({
      ...xiphos.xiphosBtcLiveParams(),
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice,
      historicalPrices: prices,
      polymarketMarkets,
      btcBreakDivergenceThreshold: 0.15,
    });

    const contractHint = md.buildForecastHint({
      canEmitCanonical: false,
      ticker: 'BTC-USD',
      horizon: 7,
      expectedReturn: direct.actionSignal.expectedReturn,
      mixingTimeWeight: direct.metadata.mixingTimeWeight,
      predictionConfidence: diagnostics?.predictionConfidence ?? NaN,
    });

    const sameRun =
      Math.abs((diagnostics?.mixingTimeWeight ?? NaN) - direct.metadata.mixingTimeWeight) < 1e-12
      && (diagnostics?.predictionConfidence ?? NaN) === direct.predictionConfidence;

    const observed = JSON.stringify(
      {
        status: payload.status,
        canEmitCanonical: diagnostics?.canEmitCanonical,
        predictionConfidence: diagnostics?.predictionConfidence,
        expectedReturn: direct.actionSignal.expectedReturn,
        mixingTimeWeight: direct.metadata.mixingTimeWeight,
        sameRunAsDirectCompute: sameRun,
        contractHint,
        emittedForecastHint: payload.forecastHint,
      },
      null,
      2,
    );
    const expected =
      'a forecastHint object for the BTC-USD h=7 abstain payload (canEmitCanonical=false, confidence >= 0.10)';
    const actual = `forecastHint=${JSON.stringify(payload.forecastHint)} while buildForecastHint(contract) returns ${JSON.stringify(contractHint)}`;

    try {
      assert.ok(
        sameRun
          && payload.status === 'abstain'
          && diagnostics?.canEmitCanonical === false
          && contractHint !== null
          && payload.forecastHint !== null,
        'forecastHint missing on abstain payload',
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
