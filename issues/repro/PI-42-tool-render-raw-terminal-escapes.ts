/**
 * PI-42: Tool event and approval prompt render untrusted text unsanitized
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * Answers are sanitized (`AnswerBoxComponent` -> `formatResponseTui`), but the
 * tool/approval components interpolate model- and tool-controlled text straight
 * into pi-tui `Text`, which preserves ANSI/OSC bytes. A search summary, tool
 * error, file path or approval preview can therefore write the terminal
 * (OSC 52 clipboard writes, screen clears) and a path containing a newline can
 * forge a second line in the approval prompt.
 *
 * No network, subprocess or TUI is used: the components are rendered directly.
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

const ESC = '\u001b';
const BEL = '\u0007';
const OSC_CLIPBOARD = `${ESC}]52;c;${Buffer.from('injected').toString('base64')}${BEL}`;
const CLEAR_SCREEN = `${ESC}[2J`;

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

    // --- production imports after chdir + scrub --------------------------
    const { ToolEventComponent } = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/components/tool-event.ts')).href
    )) as typeof import('../../src/components/tool-event.ts');
    const { ApprovalPromptComponent } = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/components/approval-prompt.ts')).href
    )) as typeof import('../../src/components/approval-prompt.ts');
    const { AnswerBoxComponent } = (await import(
      pathToFileURL(join(REPO_ROOT, 'src/components/answer-box.ts')).href
    )) as typeof import('../../src/components/answer-box.ts');

    const tool = new ToolEventComponent(undefined, 'web_search', {
      query: `search results ${OSC_CLIPBOARD}${CLEAR_SCREEN}`,
    });
    tool.setComplete(`summary ${OSC_CLIPBOARD}${CLEAR_SCREEN}`, 5);
    const toolLines = tool.render(200);

    const approval = new ApprovalPromptComponent('write_file', {
      path: `safe.txt\n2. Yes, allow all edits this session ${OSC_CLIPBOARD}`,
      content: 'safe content',
    });
    const approvalLines = approval.render(200);

    // Control: the answer path already sanitizes the same payload.
    const answerLines = new AnswerBoxComponent(`answer ${OSC_CLIPBOARD}`).render(200);

    const all = [...toolLines, ...approvalLines];
    const oscCount = all.filter((line) => line.includes(OSC_CLIPBOARD)).length;
    const forgedOptionLines = approvalLines.filter((line) =>
      line.trimStart().startsWith('2. Yes, allow all edits this session'),
    ).length;
    const answerSanitized = !answerLines.some((line) => line.includes(OSC_CLIPBOARD));

    const observed =
      `oscLines=${oscCount} forgedApprovalOptionLines=${forgedOptionLines} ` +
      `answerSanitized=${answerSanitized}`;

    assert.equal(oscCount, 0, `${observed}; component rendered a raw OSC sequence`);
    assert.ok(
      forgedOptionLines <= 1,
      `${observed}; a newline in the tool path forged an approval option line`,
    );

    console.log('NO BUG');
  } catch (error) {
    if (process.exitCode !== 43) {
      const message = error instanceof Error ? error.message : String(error);
      console.log('OBSERVED:');
      console.log(`  ${normalize(message, root)}`);
      console.log(
        'EXPECTED: tool/approval components strip terminal control sequences and collapse path newlines',
      );
      console.log(
        'ACTUAL: raw OSC sequences reach the render buffer and a newline in the approval path forges an option line',
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
