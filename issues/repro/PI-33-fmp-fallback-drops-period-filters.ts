/**
 * PI-33: FMP statement fallback drops the report-period filters
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * get_income_statements advertises report_period_gt/gte/lt/lte, but when the
 * primary provider fails the FMP fallback is invoked with only
 * {ticker, period, limit} (fundamentals.ts:111). The requested fiscal window is
 * never applied, so a query scoped to FY2019 returns whatever the latest four
 * FMP periods are and presents them as the answer.
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const EVIDENCE = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues');
const GTE = '2019-01-01';
const LTE = '2019-12-31';

class HarnessFailure extends Error {}

function harnessFailure(reason: string): never {
  console.log(`HARNESS-FAILURE: ${reason}`);
  throw new HarnessFailure(reason);
}

function normalize(text: string, root: string): string {
  return text
    .replaceAll(root, '<TMP>')
    .replace(/\u001b\[[0-9;]*m/g, '')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/g, '<TS>');
}

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) harnessFailure('CS_REPRO_ROOT is unset');
  if (homedir() !== root) harnessFailure(`homedir() ${homedir()} escaped ${root}`);
  if (tmpdir() !== root) harnessFailure(`tmpdir() ${tmpdir()} escaped ${root}`);

  const savedCwd = process.cwd();
  const savedEnv: Record<string, string | undefined> = { ...process.env };
  const savedFetch = globalThis.fetch;
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
    process.env.FMP_API_KEY = 'dummy-fmp-key';

    let fmpUrl = '';
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      if (url.includes('financialmodelingprep.com')) {
        fmpUrl = url;
        // Minimal stand-in for the FMP statement endpoint: it honours from/to
        // bounds when the caller sends them and otherwise returns the latest rows.
        const rows = [
          {
            date: '2024-09-28',
            symbol: 'AAPL',
            revenue: 391035000000,
            grossProfit: 180683000000,
            operatingIncome: 123216000000,
            netIncome: 93736000000,
            eps: 6.11,
            ebitda: 134661000000,
          },
          {
            date: '2019-09-28',
            symbol: 'AAPL',
            revenue: 260174000000,
            grossProfit: 98392000000,
            operatingIncome: 63930000000,
            netIncome: 55256000000,
            eps: 2.97,
            ebitda: 76577000000,
          },
        ];
        const params = new URL(url).searchParams;
        const from = params.get('from') ?? params.get('fromDate');
        const to = params.get('to') ?? params.get('toDate');
        const bounded = rows.filter((row) => (!from || row.date >= from) && (!to || row.date <= to));
        return Response.json(bounded);
      }
      if (url.includes('financialdatasets.ai')) {
        return new Response('payment required', { status: 402, statusText: 'Payment Required' });
      }
      throw new Error(`network disabled: ${url}`);
    }) as typeof fetch;

    const { getIncomeStatements } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/fundamentals.ts')).href
    );
    const raw = await getIncomeStatements.invoke({
      ticker: 'AAPL',
      period: 'annual',
      limit: 4,
      report_period_gte: GTE,
      report_period_lte: LTE,
    });
    const parsed = JSON.parse(raw as string) as { data?: unknown };
    const rows = Array.isArray(parsed.data)
      ? (parsed.data as Array<{ date?: string }>)
      : [];
    const dates = rows.map((row) => String(row.date ?? ''));
    const inRange = dates.length > 0
      && dates.every((date) => date >= GTE && date <= LTE);

    const emit = (line: string): void => { console.log(normalize(line, root)); };
    const observed = [
      `  requested report_period_gte = ${GTE}, report_period_lte = ${LTE}`,
      `  FMP request = ${fmpUrl}`,
      `  returned statement dates = ${JSON.stringify(dates)}`,
    ];

    let failure: string | null = null;
    try {
      assert.ok(fmpUrl.length > 0, 'FMP fallback never ran');
      assert.ok(dates.length > 0, 'FMP fallback returned no statements');
      assert.ok(inRange, `returned dates ${JSON.stringify(dates)} are outside ${GTE}..${LTE}`);
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      emit('OBSERVED:');
      for (const line of observed) emit(line);
      emit('EXPECTED:');
      emit(`  every returned statement date inside ${GTE}..${LTE}`);
      emit('ACTUAL:');
      emit(`  fallback returned an unfiltered FMP period: ${JSON.stringify(dates)}`);
      process.exitCode = 42;
      return;
    }
    console.log('NO BUG');
  } finally {
    globalThis.fetch = savedFetch;
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
