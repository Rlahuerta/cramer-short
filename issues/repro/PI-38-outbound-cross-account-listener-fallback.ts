/**
 * PI-38: outbound sends through another account's socket when the requested one is missing
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * getActive(accountId) falls back to the first registered listener when the
 * requested account has no socket.  With two linked accounts, a message
 * processed for a disconnected account is sent from the other account's WhatsApp
 * number (the reply leaves from the wrong identity).
 */
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const SCRIPT_NAME = 'PI-38-outbound-cross-account-listener-fallback.ts';

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

  const savedCwd = process.cwd();
  let savedEnv: Map<string, string | undefined> | null = null;
  try {
    process.chdir(root);
    savedEnv = scrubSecrets();
    globalThis.fetch = (() => {
      throw new Error('network disabled');
    }) as unknown as typeof fetch;

    const configPath = join(root, 'gateway.json');
    mkdirSync(join(root, '.cramer-short'), { recursive: true });
    writeFileSync(
      configPath,
      JSON.stringify({
        gateway: { accountId: 'work' },
        channels: {
          whatsapp: {
            enabled: true,
            accounts: {
              work: { enabled: true, dmPolicy: 'allowlist', allowFrom: ['+15551234567'], groupPolicy: 'disabled', groupAllowFrom: [] },
              personal: { enabled: true, dmPolicy: 'allowlist', allowFrom: ['+15557654321'], groupPolicy: 'disabled', groupAllowFrom: [] },
            },
            allowFrom: [],
          },
        },
        bindings: [],
      }),
      'utf8',
    );

    const outbound = await import(
      pathToFileURL(join(REPO_ROOT, 'src/gateway/channels/whatsapp/outbound.js')).href
    );

    const sentVia: string[] = [];
    outbound.setActiveWebListener('personal', {
      sendMessage: async (_jid: string, _payload: unknown) => {
        sentVia.push('personal');
        return { key: { id: 'msg-personal' } };
      },
      sendPresenceUpdate: async () => {},
    } as never);

    let sendResult: { messageId: string; toJid: string } | null = null;
    let sendError: string | null = null;
    const realLog = console.log;
    console.log = () => {}; // outbound.ts logs the send; keep the OBSERVED block stable
    try {
      sendResult = await outbound.sendMessageWhatsApp({
        to: '+15551234567',
        body: 'reply for the work account',
        accountId: 'work',
        configPath,
      });
    } catch (error) {
      sendError = error instanceof Error ? error.message : String(error);
    } finally {
      console.log = realLog;
      outbound.setActiveWebListener('personal', null);
    }

    try {
      assert.ok(sendError, 'sending for a disconnected account must fail, not borrow another account socket');
      assert.deepEqual(sentVia, [], 'no socket should have transmitted the message');
      assert.equal(sendResult, null, 'no send result when the matching account socket is absent');
      console.log('NO BUG');
    } catch {
      console.log('OBSERVED:');
      console.log(`  requested account=work (no listener registered); sockets used=${JSON.stringify(sentVia)}`);
      console.log(`  result=${JSON.stringify(sendResult)} error=${sendError}`);
      console.log('EXPECTED:');
      console.log('  sendMessageWhatsApp({accountId:"work"}) throws when the work socket is absent');
      console.log('ACTUAL:');
      console.log('  getActive("work") falls back to the first socket (personal) and sends from the wrong number');
      console.log(`  script=${SCRIPT_NAME}`);
      process.exitCode = 42;
    }
  } finally {
    if (savedEnv) restoreSecrets(savedEnv);
    process.chdir(savedCwd);
  }
}

await main();
