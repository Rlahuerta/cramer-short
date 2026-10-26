/**
 * PI-43: Theme semantic colors (success/error/warning/accent) render black
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `theme.ts` stores `success: 'green'`, `error: 'red'`, `warning: 'yellow'`
 * and `accent: 'cyan'` but routes every palette entry through `chalk.hex()`.
 * chalk parses a non-hex name as invalid and falls back to rgb(0, 0, 0), so
 * every success/error/warning/skill indicator is painted black instead of
 * its semantic color. On a dark terminal the error/warning lines become
 * unreadable.
 *
 * Color output is forced on with FORCE_COLOR=3 so the mapping is observable
 * off a TTY; this mirror the real TUI, where chalk's color level is > 0.
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
const BLACK_SGR = '38;2;0;0;0';
const SEMANTIC_KEYS = ['success', 'error', 'warning', 'accent'] as const;

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

    // chalk reads FORCE_COLOR when it is first imported below.
    savedEnv.set('FORCE_COLOR', process.env.FORCE_COLOR);
    process.env.FORCE_COLOR = '3';

    const { theme } = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/theme.ts')).href
    )) as typeof import('../../src/theme.ts');

    const primary = theme.primary('X');
    if (!/\u001b\[/.test(primary)) {
      fail(`chalk color output disabled; primary=${JSON.stringify(primary)}`);
    }

    const rendered = SEMANTIC_KEYS.map((key) => ({ key, value: theme[key]('X') }));
    const blackKeys = rendered.filter((entry) => entry.value.includes(BLACK_SGR)).map((e) => e.key);
    const observed = rendered
      .map((entry) => `${entry.key}=${JSON.stringify(entry.value)}`)
      .join(' ');

    assert.equal(
      blackKeys.length,
      0,
      `blackForegrounds=${JSON.stringify(blackKeys)} ${observed}`,
    );

    console.log('NO BUG');
  } catch (error) {
    if (process.exitCode !== 43) {
      const message = error instanceof Error ? error.message : String(error);
      console.log('OBSERVED:');
      console.log(`  ${normalize(message, root)}`);
      console.log(
        'EXPECTED: success/error/warning/accent each render a non-black semantic foreground color',
      );
      console.log(
        'ACTUAL: all four render rgb(0,0,0) because chalk.hex() is applied to named colors',
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
