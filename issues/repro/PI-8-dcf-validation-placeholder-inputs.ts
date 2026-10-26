/**
 * PI-8: DCF valuation fabricates an EV error and a value-destruction
 * warning on every run
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * dcf_valuation feeds runDcfValidation() placeholder values it never receives:
 * `roic: 0`, `reportedEv: 0`, and a hardcoded `riskFreeRate: 0.043`.
 * validateEvSanity() treats reportedEv <= 0 as a hard error, and
 * validateWaccVsRoic() sees ROIC 0 and always emits "potential value
 * destruction".  A healthy DCF therefore reports one error and one warning
 * that have nothing to do with the company.
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

    // --- production import after chdir + scrub ---------------------------
    const { dcfValuationTool } = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/dcf-valuation.ts')).href
    )) as typeof import('../../src/tools/finance/dcf-valuation.ts');

    const raw = await dcfValuationTool.invoke({
      ticker: 'AAPL',
      base_fcf: 100,
      net_debt: 0,
      diluted_shares: 10,
      wacc: 0.1,
      growth_rate: 0.15,
      terminal_growth_rate: 0.025,
      years: 5,
      decay: 0.05,
    });
    const parsed = JSON.parse(raw) as {
      data: { validation?: { errors?: string[]; warnings?: string[] } };
    };
    const errors = parsed.data.validation?.errors ?? [];
    const warnings = parsed.data.validation?.warnings ?? [];

    const evErrors = errors.filter((e) => /enterprise value/i.test(e));
    const destruction = warnings.filter((w) => /value destruction/i.test(w));

    const observed =
      `errors=${JSON.stringify(errors)} warnings=${JSON.stringify(warnings)}`;

    assert.equal(
      evErrors.length,
      0,
      `${observed}; EV-sanity error fired although no reported EV was supplied`,
    );
    assert.equal(
      destruction.length,
      0,
      `${observed}; value-destruction warning fired although ROIC is unknown`,
    );

    console.log('NO BUG');
  } catch (error) {
    if (process.exitCode !== 43) {
      const message = error instanceof Error ? error.message : String(error);
      console.log('OBSERVED:');
      console.log(`  ${normalize(message, root)}`);
      console.log('EXPECTED: errors=[] and no "value destruction" warning for a healthy DCF');
      console.log(
        'ACTUAL: errors=["Reported enterprise value must be positive"] and "potential value destruction" warning on every call',
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
