/**
 * PI-39: setup wizard's "*" allowlist cannot send replies
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `gateway login` bot mode explicitly offers "* to allow anyone" and persists
 * it (src/gateway/index.ts:61/:68/:84/:89).  Inbound access control honours the
 * wildcard, but the outbound guard filters it out and then throws because no
 * explicit recipient remains, so the gateway accepts messages it can never
 * answer.
 */
import assert from 'node:assert/strict';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const SCRIPT_NAME = 'PI-39-outbound-wildcard-allowlist-blocks-replies.ts';

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

    // Exactly what promptSetupMode() writes for "*" in `gateway login` bot mode.
    const configPath = join(root, 'gateway.json');
    mkdirSync(join(root, '.cramer-short'), { recursive: true });
    writeFileSync(
      configPath,
      JSON.stringify({
        gateway: { accountId: 'default' },
        channels: {
          whatsapp: {
            enabled: true,
            accounts: {
              default: {
                enabled: true,
                dmPolicy: 'allowlist',
                allowFrom: ['*'],
                groupPolicy: 'disabled',
                groupAllowFrom: [],
                sendReadReceipts: true,
              },
            },
            allowFrom: ['*'],
          },
        },
        bindings: [],
      }),
      'utf8',
    );

    const outbound = await import(
      pathToFileURL(join(REPO_ROOT, 'src/gateway/channels/whatsapp/outbound.js')).href
    );
    const access = await import(
      pathToFileURL(join(REPO_ROOT, 'src/gateway/access-control.js')).href
    );

    const inbound = await access.checkInboundAccessControl({
      accountId: 'default',
      from: '+15551234567',
      selfE164: '+15550000000',
      senderE164: '+15551234567',
      group: false,
      isFromMe: false,
      dmPolicy: 'allowlist',
      groupPolicy: 'disabled',
      allowFrom: ['*'],
      groupAllowFrom: [],
      reply: async () => {},
    });

    let outboundError: string | null = null;
    try {
      outbound.assertOutboundAllowed({ to: '+15551234567', accountId: 'default', configPath });
    } catch (error) {
      outboundError = error instanceof Error ? error.message : String(error);
    }

    try {
      assert.equal(inbound.allowed, true, 'inbound wildcard must be allowed by checkInboundAccessControl');
      assert.equal(
        outboundError,
        null,
        'a wildcard allowlist accepted inbound must also let the gateway send its reply',
      );
      console.log('NO BUG');
    } catch {
      console.log('OBSERVED:');
      console.log(`  inbound allowed=${inbound.allowed} (allowFrom=["*"])`);
      console.log(`  outbound assertOutboundAllowed threw: ${outboundError}`);
      console.log('EXPECTED:');
      console.log('  inbound allowed=true and outbound allowed (same "*" allowlist)');
      console.log('ACTUAL:');
      console.log('  inbound accepts the sender, outbound rejects every recipient -> gateway never replies');
      console.log(`  script=${SCRIPT_NAME}`);
      process.exitCode = 42;
    }
  } finally {
    if (savedEnv) restoreSecrets(savedEnv);
    process.chdir(savedCwd);
  }
}

await main();
