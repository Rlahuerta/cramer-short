/**
 * PI-40: heartbeat active-hours gate compares "HH:MM" strings, not clock times
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * isWithinActiveHours() builds "HH:MM" strings and compares them with >= / <=.
 * With a legitimate unpadded start ("9:30"), the 10:00–16:00 window is
 * evaluated as false ("10:05" < "9:30" lexicographically), so the heartbeat
 * silently never runs even though the current time is inside the window.
 */
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const SCRIPT_NAME = 'PI-40-heartbeat-active-hours-string-compare.ts';
const FIXED_ISO = '2026-10-26T14:05:00.000Z'; // Monday 10:05 America/New_York (EDT)

function harnessFailure(reason: string): never {
  console.log(`HARNESS-FAILURE: ${reason}`);
  process.exit(43);
}

const ENV_KEY_FILE = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues/pre-work.envkeys');
const FORCED_SCRUB = [
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

function scrubSecrets(): Map<string, string | undefined> {
  const saved = new Map<string, string | undefined>();
  const listed = readFileSync(ENV_KEY_FILE, 'utf8').split(/\r?\n/).filter(Boolean);
  for (const key of new Set([...listed, ...FORCED_SCRUB])) {
    if (key === 'HOME' || key === 'USERPROFILE') continue;
    saved.set(key, process.env[key]);
    delete process.env[key];
  }
  for (const key of Object.keys(process.env)) {
    if (SECRET_RE.test(key)) {
      saved.set(key, process.env[key]);
      delete process.env[key];
    }
  }
  const leaked = Object.keys(process.env).filter((key) => SECRET_RE.test(key));
  if (leaked.length > 0) harnessFailure(`secret env keys survived the scrub: ${leaked.join(', ')}`);
  return saved;
}

function restoreSecrets(saved: Map<string, string | undefined>): void {
  for (const [key, value] of saved) {
    if (value === undefined) delete process.env[key];
    else process.env[key] = value;
  }
}

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) harnessFailure('CS_REPRO_ROOT is not set');
  if (homedir() !== root) harnessFailure(`homedir() escaped the run root: ${homedir()}`);
  if (tmpdir() !== root) harnessFailure(`tmpdir() escaped the run root: ${tmpdir()}`);

  const realSetTimeout = globalThis.setTimeout;
  const RealDate = Date;
  const fixedMs = RealDate.parse(FIXED_ISO);
  const savedCwd = process.cwd();
  let savedEnv: Map<string, string | undefined> | null = null;

  try {
    process.chdir(root);
    savedEnv = scrubSecrets();
    globalThis.fetch = (() => {
      throw new Error('network disabled');
    }) as unknown as typeof fetch;

    const dir = join(root, '.cramer-short');
    mkdirSync(dir, { recursive: true });
    writeFileSync(
      join(dir, 'gateway.json'),
      JSON.stringify({
        gateway: {
          accountId: 'default',
          heartbeat: {
            enabled: true,
            intervalMinutes: 5,
            activeHours: {
              start: '9:30',
              end: '16:00',
              timezone: 'America/New_York',
              daysOfWeek: [1],
            },
          },
        },
        channels: { whatsapp: { enabled: true, accounts: {}, allowFrom: [] } },
        bindings: [],
      }),
      'utf8',
    );

    class FakeDate extends RealDate {
      constructor(...args: unknown[]) {
        if (args.length === 0) super(fixedMs);
        else super(...(args as [string]));
      }
      static now(): number {
        return fixedMs;
      }
    }
    globalThis.Date = FakeDate as unknown as DateConstructor;

    const scheduled: Array<() => void> = [];
    globalThis.setTimeout = ((callback: () => void) => {
      scheduled.push(callback);
      return { unref() {}, __fake: true } as unknown as ReturnType<typeof setTimeout>;
    }) as unknown as typeof setTimeout;

    const heartbeat = await import(
      pathToFileURL(join(REPO_ROOT, 'src/gateway/heartbeat/runner.js')).href
    );
    const runner = heartbeat.startHeartbeatRunner({});
    assert.ok(scheduled.length > 0, 'heartbeat runner did not schedule its first tick');
    scheduled[0]!(); // run tick() at the fixed Monday 10:05 ET
    await new Promise<void>((resolve) => realSetTimeout(resolve, 25));
    runner.stop();

    const log = readFileSync(join(dir, 'gateway-debug.log'), 'utf8');
    const outsideActiveHours = log.includes('outside active hours');

    try {
      assert.equal(
        outsideActiveHours,
        false,
        'Monday 10:05 ET is inside 9:30–16:00 America/New_York; the heartbeat must not skip',
      );
      console.log('NO BUG');
    } catch {
      console.log('OBSERVED:');
      console.log('  now=2026-10-26T14:05:00Z (Mon 10:05 America/New_York), activeHours={"start":"9:30","end":"16:00"}');
      console.log(`  gateway-debug.log contains "outside active hours": ${outsideActiveHours}`);
      console.log('EXPECTED:');
      console.log('  the active-hours gate passes at 10:05 inside the 9:30–16:00 window');
      console.log('ACTUAL:');
      console.log('  string compare "10:05" >= "9:30" is false, so every heartbeat in this window is skipped');
      console.log(`  script=${SCRIPT_NAME}`);
      process.exitCode = 42;
    }
  } finally {
    globalThis.Date = RealDate;
    globalThis.setTimeout = realSetTimeout;
    if (savedEnv) restoreSecrets(savedEnv);
    process.chdir(savedCwd);
  }
}

await main();
