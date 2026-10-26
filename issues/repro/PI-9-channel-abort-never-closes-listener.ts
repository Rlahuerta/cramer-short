/**
 * PI-9: monitorWhatsAppChannel ignores abortSignal while awaiting onClose
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * The gateway's stop path (SIGINT/SIGTERM) aborts the account AbortSignal and
 * then awaits the channel task.  monitorWhatsAppChannel only checks the signal
 * between connection attempts; while it is awaiting listener.onClose nothing
 * closes the listener, so stopAll() (and therefore `server.stop()`) never
 * resolves.  Reproduced with a stubbed inbound transport (no daemon/network).
 */
import { mock } from 'bun:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const SCRIPT_NAME = 'PI-9-channel-abort-never-closes-listener.ts';
const ABORT_SETTLE_BUDGET_MS = 400;

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

  const realTimers = {
    setTimeout: globalThis.setTimeout,
    setInterval: globalThis.setInterval,
    clearInterval: globalThis.clearInterval,
  };
  const savedCwd = process.cwd();
  let savedEnv: Map<string, string | undefined> | null = null;

  try {
    process.chdir(root);
    savedEnv = scrubSecrets();

    // Never allow a real network call from the module graph.
    globalThis.fetch = (() => {
      throw new Error('network disabled');
    }) as unknown as typeof fetch;

    // The channel runtime has no injectable transport seam; stub the inbound
    // module it imports.  The fake listener never closes on its own, exactly
    // like a live Baileys socket that is merely idle.
    const inboundFile = join(REPO_ROOT, 'src/gateway/channels/whatsapp/inbound.ts');
    let fakeInboundUsed = false;
    mock.module(inboundFile, () => ({
      monitorWebInbox: async () => {
        fakeInboundUsed = true;
        let resolveClose: (reason: { status: number; isLoggedOut: boolean }) => void = () => {};
        const onClose = new Promise<{ status: number; isLoggedOut: boolean }>((resolve) => {
          resolveClose = resolve;
        });
        return {
          sock: {},
          onClose,
          // Matches monitorWebInbox.close(): resolveClose(...) makes the
          // runtime's `await listener.onClose` settle.
          close: async () => resolveClose({ status: 499, isLoggedOut: false }),
        };
      },
    }));

    // Keep the channel's heartbeat/watchdog intervals from holding the
    // process open; the repro only needs the abort/await path.
    globalThis.setInterval = ((_fn: unknown, _ms?: number) =>
      ({ unref() {} }) as unknown as ReturnType<typeof setInterval>) as typeof setInterval;
    globalThis.clearInterval = (() => {}) as typeof clearInterval;

    const runtime = await import(
      pathToFileURL(join(REPO_ROOT, 'src/gateway/channels/whatsapp/runtime.js')).href
    );

    const abort = new AbortController();
    let settled = false;
    const channelRun = runtime
      .monitorWhatsAppChannel({
        accountId: 'default',
        authDir: join(root, 'credentials', 'whatsapp', 'default'),
        verbose: false,
        allowFrom: ['+15551234567'],
        dmPolicy: 'allowlist',
        groupPolicy: 'disabled',
        groupAllowFrom: [],
        abortSignal: abort.signal,
        onMessage: async () => {},
      })
      .then(() => {
        settled = true;
      });

    // Let the fake connection reach `await listener.onClose`.
    await new Promise<void>((resolve) => realTimers.setTimeout(resolve, 50));
    if (!fakeInboundUsed) harnessFailure('stubbed monitorWebInbox was not reached');

    abort.abort(); // what gateway manager stopAccount() does

    const timedOut = await Promise.race([
      channelRun.then(() => false),
      new Promise<boolean>((resolve) => realTimers.setTimeout(() => resolve(true), ABORT_SETTLE_BUDGET_MS)),
    ]);

    try {
      assert.equal(
        timedOut,
        false,
        'monitorWhatsAppChannel must resolve after abortSignal aborts (listener closed by the runtime)',
      );
      console.log('NO BUG');
    } catch {
      console.log('OBSERVED:');
      console.log(`  abort() called at t=50ms; monitorWhatsAppChannel still pending at t=${50 + ABORT_SETTLE_BUDGET_MS}ms (settled=${settled})`);
      console.log('EXPECTED:');
      console.log('  aborting the account signal closes the active listener so monitorWhatsAppChannel resolves');
      console.log('ACTUAL:');
      console.log('  runtime awaits listener.onClose forever; manager.stopAccount()/stopAll() hangs (Ctrl+C hang)');
      console.log(`  script=${SCRIPT_NAME}`);
      process.exitCode = 42;
    }
  } finally {
    globalThis.setTimeout = realTimers.setTimeout;
    globalThis.setInterval = realTimers.setInterval;
    globalThis.clearInterval = realTimers.clearInterval;
    if (savedEnv) restoreSecrets(savedEnv);
    process.chdir(savedCwd);
  }
}

await main();
