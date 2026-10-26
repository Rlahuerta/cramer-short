/**
 * PI-48: Streaming LLM calls never retry transient provider errors
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `streamCallLlm()` wraps `llm.stream(...)` in `withRetry()`, but LangChain's
 * `stream()` is a lazy async generator: calling it performs no I/O, so the
 * retry loop always sees a resolved AsyncGenerator and never observes the
 * provider error, which is thrown on the first `for await` step. The
 * non-streaming `callLlm()` path retries the same 429 three times. A transient
 * rate limit during max-iterations synthesis therefore degrades straight to the
 * "synthesis failed" raw-tool-results fallback.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const EVIDENCE = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues');

class HarnessFailure extends Error {}

function harnessFailure(reason: string): never {
  console.log(`HARNESS-FAILURE: ${reason}`);
  throw new HarnessFailure(reason);
}

function normalize(text: string, root: string): string {
  return text.split(root).join('<TMP>').split(REPO_ROOT).join('<REPO>');
}

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) harnessFailure('CS_REPRO_ROOT is unset');
  if (homedir() !== root) harnessFailure(`homedir() ${homedir()} escaped ${root}`);
  if (tmpdir() !== root) harnessFailure(`tmpdir() ${tmpdir()} escaped ${root}`);

  const savedCwd = process.cwd();
  const savedEnv: Record<string, string | undefined> = { ...process.env };
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
    globalThis.fetch = (async () => {
      throw new Error('network disabled');
    }) as unknown as typeof fetch;

    const llm = await import(pathToFileURL(join(REPO_ROOT, 'src/model/llm.ts')).href);

    // Streaming path: a transient 429 thrown on the first chunk.
    let streamAttempts = 0;
    llm._setModelFactory((() => ({
      stream: async function* () {
        streamAttempts += 1;
        throw new Error('429 Too Many Requests: rate limit exceeded');
      },
    })) as never);
    let streamError = '';
    try {
      for await (const _chunk of llm.streamCallLlm('reply', { model: 'ollama:llama3.1:8b' })) {
        // no chunks expected
      }
    } catch (error) {
      streamError = error instanceof Error ? error.message : String(error);
    }

    // Non-streaming control: the same transient 429 must be retried 3 times.
    let callAttempts = 0;
    llm._setModelFactory((() => ({
      invoke: async () => {
        callAttempts += 1;
        throw new Error('429 Too Many Requests: rate limit exceeded');
      },
    })) as never);
    let callError = '';
    try {
      await llm.callLlm('reply', {
        model: 'ollama:llama3.1:8b',
        systemPrompt: 'reply',
        timeoutMs: 10_000,
      });
    } catch (error) {
      callError = error instanceof Error ? error.message : String(error);
    }

    console.log('OBSERVED:');
    console.log(`  streamCallLlm attempts on 429: ${streamAttempts}`);
    console.log(`  streamCallLlm error: ${normalize(streamError, root)}`);
    console.log(`  callLlm attempts on 429 (control): ${callAttempts}`);
    console.log(`  callLlm error: ${normalize(callError, root)}`);

    let failure: string | null = null;
    try {
      assert.equal(
        streamAttempts,
        3,
        `streamCallLlm must retry a transient 429 like callLlm does, got ${streamAttempts} attempt(s)`,
      );
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      console.log('EXPECTED:');
      console.log('  streamCallLlm makes 3 attempts on a retryable 429 (same policy as callLlm)');
      console.log('ACTUAL:');
      console.log(`  streamCallLlm made ${streamAttempts} attempt(s) and surfaced "${normalize(streamError, root)}"`);
      process.exitCode = 42;
      return;
    }
    console.log('NO BUG');
  } finally {
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
