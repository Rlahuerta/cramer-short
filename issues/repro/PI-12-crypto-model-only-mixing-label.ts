/**
 * PI-12: crypto model-only output claims 100% Polymarket anchors
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `cryptoModelOnly` is a first-class calibration bypass (no anchors, model-only
 * emission) exactly like `commodityModelOnly`, but the mixing line and canonical
 * diagnostics only special-case the commodity flag. A BTC model-only run with
 * zero trusted anchors therefore reports "Mixing: 0.0% Markov / 100.0% Anchors"
 * and `markovWeight ≈ 0`, even though `anchorBypassApplied` is true and the
 * distribution is 100% Markov-model driven.
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

function deterministicSeries(length: number, drift: number, vol: number, seed: number): number[] {
  let state = seed >>> 0;
  const random = () => {
    state = (state * 1_664_525 + 1_013_904_223) >>> 0;
    return state / 0x1_0000_0000;
  };
  const gaussian = () =>
    Math.sqrt(-2 * Math.log(Math.max(1e-12, random()))) * Math.cos(2 * Math.PI * random());
  const prices = [100];
  let price = 100;
  for (let i = 1; i < length; i++) {
    price *= Math.exp(drift + gaussian() * vol);
    prices.push(price);
  }
  return prices;
}

async function main(): Promise<void> {
  let prepared: { root: string; restore: () => void } | undefined;
  try {
    prepared = await prepare();
    const root = prepared.root;

    const md = await import(pathToFileURL(join(REPO_ROOT, 'src/tools/finance/markov-distribution.ts')).href);

    const prices = deterministicSeries(70, 0.002, 0.02, 1);
    const raw = await (md.markovDistributionTool as any).func({
      ticker: 'BTC-USD',
      horizon: 7,
      currentPrice: prices[prices.length - 1],
      historicalPrices: prices,
      polymarketMarkets: [{ question: 'Will it rain tomorrow?', probability: 0.5 }],
      trajectory: false,
    });
    const payload = JSON.parse(raw).data;
    const diagnostics = payload.canonical?.diagnostics;
    const mixingLine = String(payload.report)
      .split('\n')
      .find((line: string) => line.startsWith('Mixing:') || line.startsWith('Calibration: model-only'));

    const observed = JSON.stringify(
      {
        status: payload.status,
        calibrationMode: diagnostics?.calibrationMode,
        anchorBypassApplied: diagnostics?.anchorBypassApplied,
        trustedAnchors: diagnostics?.trustedAnchors,
        markovWeight: diagnostics?.markovWeight,
        anchorWeight: diagnostics?.anchorWeight,
        mixingLine,
      },
      null,
      2,
    );
    const expected =
      'a model-only run reports 100% Markov (markovWeight=1, anchorWeight=0) and a model-only calibration line';
    const actual = `markovWeight=${String(diagnostics?.markovWeight)}, anchorWeight=${String(diagnostics?.anchorWeight)}, line="${String(mixingLine)}"`;

    try {
      assert.ok(
        payload.status === 'ok'
          && diagnostics?.anchorBypassApplied === true
          && diagnostics?.trustedAnchors === 0
          && diagnostics?.calibrationMode === 'model_only'
          && diagnostics?.markovWeight === 1
          && diagnostics?.anchorWeight === 0
          && typeof mixingLine === 'string'
          && mixingLine.includes('model-only'),
        'crypto model-only provenance reported anchors',
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
