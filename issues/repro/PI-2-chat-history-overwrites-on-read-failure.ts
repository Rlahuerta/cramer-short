/**
 * PI-2: Chat history file is replaced after a failed read
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `LongTermChatHistory.load` converts any read failure (EACCES, EIO, EMFILE) or
 * a parse failure into "no history" and sets `loaded = true`. The next
 * `addUserMessage` then writes the (empty) in-memory list back over the file,
 * destroying the previous history that was never read.
 */

import assert from 'node:assert/strict';
import { chmodSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { readFile } from 'node:fs/promises';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));

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
  'OLLAMA_BASE_URL', 'DEXTER_GATEWAY_CONFIG', 'DEXTER_PAIRING_PATH',
  'CRAMER_SHORT_SESSIONS_DIR', 'KALSHI_API_KEY', 'OPENROUTER_API_KEY',
  'E2E_MODEL', 'FORECAST_LAB_OUTPUT_METRICS', 'COVERAGE_LINES_MIN', 'TEST_API_KEY',
];

async function prepare(): Promise<{ root: string; restore: () => void }> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) throw new HarnessFailure('CS_REPRO_ROOT is not set');
  const originalCwd = process.cwd();
  const saved = new Map<string, string | undefined>();
  const remember = (key: string) => { if (!saved.has(key)) saved.set(key, process.env[key]); };

  process.chdir(root);
  if (homedir() !== root) throw new HarnessFailure(`homedir escaped run root: ${homedir()}`);
  if (tmpdir() !== root) throw new HarnessFailure(`tmpdir escaped run root: ${tmpdir()}`);

  const envKeysPath = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues/pre-work.envkeys');
  const listed = (await readFile(envKeysPath, 'utf8'))
    .split(/\r?\n/)
    .map((v) => v.trim())
    .filter(Boolean);
  const exempt = new Set(['HOME', 'USERPROFILE']);
  for (const key of new Set([...listed, ...EXTRA_SCRUB])) {
    if (exempt.has(key)) continue;
    remember(key);
    delete process.env[key];
  }
  for (const key of Object.keys(process.env)) {
    if (exempt.has(key)) continue;
    if (/(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/i.test(key)) { remember(key); delete process.env[key]; }
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

async function main(): Promise<void> {
  let prepared: { root: string; restore: () => void } | undefined;
  try {
    prepared = await prepare();
    const root = prepared.root;

    const { LongTermChatHistory } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/utils/long-term-chat-history.ts')).href
    );
    const historyDir = join(root, '.cramer-short');
    mkdirSync(join(historyDir, 'messages'), { recursive: true });
    const file = join(historyDir, 'messages', 'chat_history.json');
    writeFileSync(
      file,
      JSON.stringify({
        messages: [
          {
            id: '1',
            timestamp: '2026-01-01T00:00:00.000Z',
            userMessage: 'IMPORTANT OLD HISTORY',
            agentResponse: 'kept',
          },
        ],
      }),
    );

    chmodSync(file, 0o000);
    const history = new LongTermChatHistory(root);
    let loadError: string | null = null;
    try {
      await history.load();
    } catch (error) {
      loadError = error instanceof Error ? error.message : String(error);
    }
    let addError: string | null = null;
    try {
      await history.addUserMessage('new question');
    } catch (error) {
      addError = error instanceof Error ? error.message : String(error);
    }
    chmodSync(file, 0o644);

    const after = readFileSync(file, 'utf-8');
    const preserved = after.includes('IMPORTANT OLD HISTORY');
    const observed = JSON.stringify(
      {
        loadRejected: loadError !== null,
        addRejected: addError !== null,
        originalPreserved: preserved,
      },
      null,
      2,
    );
    const expected = 'the unreadable chat history file is preserved instead of being replaced by the next save';
    const actual = preserved
      ? 'the original history survived'
      : 'chat_history.json was replaced by the next save, so the original history is gone';

    try {
      assert.ok(preserved, 'chat history was overwritten after a failed read');
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
