/**
 * PI-36: Fractional projection years silently produce NaN valuations
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * Both dcf_valuation and reverse_dcf accept `years` as z.number() (no
 * .int()).  projectFcf() loops `i < years` (6 values for 5.5), but the
 * engines read `projectedFcf[years - 1]` / `impliedFcfProjection[years - 1]`,
 * i.e. index 4.5 -> undefined -> NaN.  dcf_valuation then returns a null fair
 * value with a "NaN" sensitivity grid and no error about `years`, and
 * reverse_dcf reports resultStatus "converged" at the HIGH boundary with a
 * null terminal value.
 */
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));

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

    // --- production imports after chdir + scrub --------------------------
    const { dcfValuationTool } = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/dcf-valuation.ts')).href
    )) as typeof import('../../src/tools/finance/dcf-valuation.ts');
    const { reverseDcfTool } = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/reverse-dcf.ts')).href
    )) as typeof import('../../src/tools/finance/reverse-dcf.ts');

    type ToolData = Record<string, unknown>;
    const invokeMaybeRejected = async (
      fn: () => Promise<unknown>,
    ): Promise<ToolData | null> => {
      try {
        const raw = await fn();
        return (JSON.parse(String(raw)) as { data: ToolData }).data;
      } catch {
        // A schema rejection for a fractional horizon is an acceptable fix.
        return null;
      }
    };

    const dcf = await invokeMaybeRejected(() =>
      dcfValuationTool.invoke({
        ticker: 'AAPL',
        base_fcf: 100,
        net_debt: 0,
        diluted_shares: 10,
        wacc: 0.1,
        growth_rate: 0.15,
        terminal_growth_rate: 0.025,
        years: 5.5,
        decay: 0.05,
      }),
    );

    const reverse = await invokeMaybeRejected(() =>
      reverseDcfTool.invoke({
        ticker: 'AAPL',
        current_price: 100,
        base_fcf: 100,
        wacc: 0.1,
        terminal_growth_rate: 0.025,
        years: 5.5,
        decay: 0.05,
        net_debt: 0,
        diluted_shares: 10,
      }),
    );

    const dcfFair = dcf?.fairValuePerShare;
    const dcfErrors = (dcf?.validation as { errors?: string[] } | undefined)?.errors ?? [];
    const reverseStatus = reverse?.resultStatus;
    const reverseTv = reverse?.impliedTv;
    const reverseGrowth = reverse?.impliedGrowthRate;

    const observed =
      `years=5.5 dcf.fairValuePerShare=${String(dcfFair)} ` +
      `dcf.validation.errors=${JSON.stringify(dcfErrors)} ` +
      `reverse.status=${String(reverseStatus)} ` +
      `reverse.impliedGrowth=${String(reverseGrowth)} ` +
      `reverse.impliedTv=${String(reverseTv)}`;

    const dcfOk =
      dcf === null || (typeof dcfFair === 'number' && Number.isFinite(dcfFair));
    assert.ok(
      dcfOk,
      `${observed}; a fractional horizon silently yields a non-finite fair value`,
    );

    const reverseOk =
      reverse === null ||
      !(
        reverseStatus === 'converged' &&
        (typeof reverseTv !== 'number' || !Number.isFinite(reverseTv))
      );
    assert.ok(
      reverseOk,
      `${observed}; reverse DCF reports converged with a null terminal value`,
    );

    console.log('NO BUG');
  } catch (error) {
    if (process.exitCode !== 43) {
      const message = error instanceof Error ? error.message : String(error);
      console.log('OBSERVED:');
      console.log(`  ${normalize(message, root)}`);
      console.log(
        'EXPECTED: years=5.5 is rejected at the schema boundary, or a finite fair value is returned',
      );
      console.log(
        'ACTUAL: dcf.fairValuePerShare=null with a NaN sensitivity grid, and reverse_dcf reports converged at impliedGrowth=2 with impliedTv=null',
      );
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
