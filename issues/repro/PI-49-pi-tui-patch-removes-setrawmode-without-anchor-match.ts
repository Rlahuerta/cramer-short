/**
 * PI-49
 * The postinstall patch removes the original setRawMode() call even when the
 * insertion anchor did not match, so a pi-tui layout change silently leaves the
 * terminal in raw mode instead of failing loudly.
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: static
 *
 * Execution is impossible: running the patch would rewrite the installed
 * node_modules/@mariozechner/pi-tui file, which the audit contract forbids, so
 * the artifact is inspected for the required guard instead.
 */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const ARTIFACT = 'patches/pi-tui-terminal-fix.cjs';

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) {
    console.log('HARNESS-FAILURE: CS_REPRO_ROOT is unset');
    process.exitCode = 43;
    return;
  }
  const originalCwd = process.cwd();
  const originalEnv = { ...process.env };
  try {
    process.chdir(root);

    assert.equal(homedir(), root, 'HARNESS-FAILURE: homedir escaped CS_REPRO_ROOT');
    assert.equal(tmpdir(), root, 'HARNESS-FAILURE: tmpdir escaped CS_REPRO_ROOT');

    const listed = readFileSync(
      join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues/pre-work.envkeys'),
      'utf8',
    ).split('\n').map((line) => line.trim()).filter(Boolean);
    for (const name of listed) if (name !== 'HOME' && name !== 'USERPROFILE') delete process.env[name];
    for (const name of [
      'OLLAMA_BASE_URL', 'DEXTER_GATEWAY_CONFIG', 'DEXTER_PAIRING_PATH',
      'CRAMER_SHORT_SESSIONS_DIR', 'KALSHI_API_KEY', 'OPENROUTER_API_KEY',
      'E2E_MODEL', 'FORECAST_LAB_OUTPUT_METRICS', 'COVERAGE_LINES_MIN', 'TEST_API_KEY',
    ]) delete process.env[name];
    for (const name of Object.keys(process.env)) {
      if (/(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/.test(name)) delete process.env[name];
    }
    for (const name of Object.keys(process.env)) {
      if (/(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/.test(name)) {
        console.log(`HARNESS-FAILURE: secret survived ${name}`);
        process.exitCode = 43;
        return;
      }
    }

    const realFetch = globalThis.fetch;
    globalThis.fetch = (async (input: string | URL | Request) => {
      throw new Error(`network disabled: ${String(input)}`);
    }) as unknown as typeof fetch;
    try {
      const raw = readFileSync(join(REPO_ROOT, ARTIFACT), 'utf8');
      const lines = raw.split('\n');

      // Required form: the replacement result is captured and compared with the
      // pre-patch content, and the script aborts (non-zero) when the anchor did
      // not match, so the old setRawMode removal can never run on a no-op patch.
      const hasAnchorGuard =
        raw.includes('patched === content') &&
        raw.includes('refusing to patch');
      const insertionIsConditional =
        /if \(patched === content\)[\s\S]{0,200}process\.exit\(1\)/.test(raw);

      let ok = false;
      try {
        assert.ok(hasAnchorGuard, 'patch must verify the anchor matched before removing the old call');
        assert.ok(insertionIsConditional, 'patch must exit non-zero when the anchor did not match');
        ok = true;
      } catch {
        ok = false;
      }
      if (!ok) {
        console.log('OBSERVED:');
        lines.forEach((line, index) => {
          if (
            line.includes('content = content.replace(originalBlock, replacementBlock)') ||
            line.includes('const oldSetRawMode') ||
            line.includes('if (oldSetRawMode.test(content))') ||
            line.includes('content = content.replace(oldSetRawMode') ||
            line.includes('fs.writeFileSync(terminalPath, content')
          ) {
            console.log(`  ${index + 1}: ${line.trim()}`);
          }
        });
        console.log('EXPECTED:');
        console.log('  The script compares the patch result with the pre-patch content and exits non-zero when the anchor missed.');
        console.log('ACTUAL:');
        console.log('  The insertion result is discarded and the old setRawMode block is removed unconditionally, so a missed anchor deletes the fix.');
        process.exitCode = 42;
        return;
      }
      console.log('NO BUG');
    } finally {
      globalThis.fetch = realFetch;
    }
  } catch (err) {
    console.log(`HARNESS-FAILURE: ${err instanceof Error ? err.message : String(err)}`);
    process.exitCode = 43;
  } finally {
    process.chdir(originalCwd);
    for (const name of Object.keys(process.env)) if (!(name in originalEnv)) delete process.env[name];
    Object.assign(process.env, originalEnv);
  }
}

await main();
