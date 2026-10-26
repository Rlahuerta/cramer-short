/**
 * PI-19: sentiment variance ignores the sentiment signal magnitude
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * computeVariance() declares a `sentSignal` parameter and documents that the
 * sentiment score "contributes to the variance term", but the parameter is
 * never read.  Weak and strong sentiment therefore produce an identical
 * forecast sigma, so the 95% CI is over-wide for weak sentiment and no
 * sentiment magnitude reaches the uncertainty estimate at all.
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const FOUND_AT = 'e3e2088b21acc4963b3c20661e5d15c739244409';

const EXPLICIT_SCRUB = [
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
  return text
    .replaceAll(root, '<TMP>')
    .replace(/\u001b\[[0-9;]*m/g, '')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/g, '<TS>');
}

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) {
    console.log('HARNESS-FAILURE: CS_REPRO_ROOT unset');
    process.exitCode = 43;
    return;
  }

  const originalCwd = process.cwd();
  const savedEnv = new Map<string, string | undefined>();
  const realFetch = globalThis.fetch;

  const fail = (reason: string): never => {
    console.log(`HARNESS-FAILURE: ${reason}`);
    process.exitCode = 43;
    throw new Error(`HARNESS-FAILURE: ${reason}`);
  };

  try {
    process.chdir(root);
    if (homedir() !== root) fail(`homedir escape: ${homedir()}`);
    if (tmpdir() !== root) fail(`tmpdir escape: ${tmpdir()}`);

    // --- env scrub (before every dynamic import) -------------------------
    const envKeysRaw = await readFile(
      join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues/pre-work.envkeys'),
      'utf8',
    );
    const keys = new Set<string>([
      ...envKeysRaw.split(/\r?\n/).map((k) => k.trim()).filter(Boolean),
      ...EXPLICIT_SCRUB,
    ]);
    for (const k of Object.keys(process.env)) {
      if (SECRET_RE.test(k)) keys.add(k);
    }
    keys.delete('HOME');
    keys.delete('USERPROFILE');
    for (const k of keys) {
      savedEnv.set(k, process.env[k]);
      delete process.env[k];
    }
    for (const k of Object.keys(process.env)) {
      if (SECRET_RE.test(k)) fail(`secret key survived scrub: ${k}`);
    }

    // --- network trap ----------------------------------------------------
    globalThis.fetch = (async () => {
      throw new Error('network disabled');
    }) as unknown as typeof globalThis.fetch;

    // --- production import after chdir + scrub ---------------------------
    const ensemble = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/utils/finance/ensemble.ts')).href
    )) as typeof import('../../src/utils/finance/ensemble.ts');

    const market = {
      question: 'Will test market resolve YES?',
      probability: 0.6,
      volume24hUsd: 800_000,
      ageDays: 21,
      signalTier: 'macro' as const,
      deltaYes: 0.06,
      deltaNo: -0.04,
    };

    // Same market, weights and probability; only the sentiment magnitude differs.
    const sigmaNoSentiment = ensemble.computeVariance([market], 0.4, 0.2, 0);
    const sigmaWeak = ensemble.computeVariance([market], 0.4, 0.2, 0.01);
    const sigmaStrong = ensemble.computeVariance([market], 0.4, 0.2, 1.0);

    const observed =
      `sigma(sentSignal=0)=${sigmaNoSentiment.toFixed(12)}, ` +
      `sigma(sentSignal=0.01)=${sigmaWeak.toFixed(12)}, ` +
      `sigma(sentSignal=1.0)=${sigmaStrong.toFixed(12)}`;

    assert.equal(
      sigmaWeak !== sigmaStrong,
      true,
      `${observed}; weak and strong sentiment produce identical sigma`,
    );

    console.log('NO BUG');
  } catch (error) {
    if (process.exitCode !== 43) {
      const message = error instanceof Error ? error.message : String(error);
      console.log('OBSERVED:');
      console.log(`  ${normalize(message, root)}`);
      console.log('EXPECTED: sigma(sentSignal=0.01) < sigma(sentSignal=1.0)');
      console.log('ACTUAL: sigma is identical for sentSignal=0, 0.01 and 1.0');
      process.exitCode = 42;
    }
  } finally {
    globalThis.fetch = realFetch;
    for (const [k, v] of savedEnv) {
      if (v === undefined) delete process.env[k];
      else process.env[k] = v;
    }
    process.chdir(originalCwd);
  }
}

await main();
