/**
 * PI-56: Invalid project skill overrides are silently discarded
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * `scanSkillDirectory()` catches every error from `extractSkillMetadata()` and
 * drops the skill with no diagnostic. A malformed project override under
 * `.cramer-short/skills/<name>/SKILL.md` therefore never shadows the builtin of
 * the same name and never appears in the prompt or `/skills` panel; the user's
 * customized workflow silently does not run. The catch is the only place that
 * knows the file and the reason, and it throws both away.
 */
import assert from 'node:assert/strict';
import { mkdirSync, writeFileSync } from 'node:fs';
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

    // A malformed project override: `description` is required and missing.
    const overrideDir = join(root, '.cramer-short', 'skills', 'dcf-valuation');
    mkdirSync(overrideDir, { recursive: true });
    const overridePath = join(overrideDir, 'SKILL.md');
    writeFileSync(
      overridePath,
      ['---', 'name: dcf-valuation', '---', 'PROJECT OVERRIDE BODY', ''].join('\n'),
      'utf-8',
    );

    // Also a brand-new project skill with the same malformed shape.
    const newDir = join(root, '.cramer-short', 'skills', 'my-custom-skill');
    mkdirSync(newDir, { recursive: true });
    writeFileSync(newDir + '/SKILL.md', ['---', '---', 'BODY', ''].join('\n'), 'utf-8');

    const { discoverSkills, getSkill } = await import(
      pathToFileURL(join(REPO_ROOT, 'src/skills/registry.ts')).href
    );

    // Capture any diagnostics emitted while the invalid overrides are scanned.
    const diagnostics: string[] = [];
    const origWarn = console.warn;
    const origError = console.error;
    console.warn = (...args: unknown[]) => { diagnostics.push(args.map(String).join(' ')); };
    console.error = (...args: unknown[]) => { diagnostics.push(args.map(String).join(' ')); };
    let resolved: { source: string; path: string } | undefined;
    let listed: string[] = [];
    try {
      listed = discoverSkills().map(
        (skill: { name: string; source: string }) => `${skill.name}:${skill.source}`,
      );
      const dcf = getSkill('dcf-valuation');
      resolved = dcf ? { source: dcf.source, path: normalize(dcf.path, root) } : undefined;
    } finally {
      console.warn = origWarn;
      console.error = origError;
    }

    const overrideDiagnostics = diagnostics.filter((line) => line.includes('dcf-valuation') || line.includes('my-custom-skill'));

    console.log('OBSERVED:');
    console.log(`  diagnostics mentioning the two invalid overrides: ${overrideDiagnostics.length}`);
    console.log(`  diagnostics total: ${diagnostics.length}`);
    console.log(`  dcf-valuation resolved source: ${resolved?.source ?? '<missing>'}`);
    console.log(`  dcf-valuation resolved path: ${resolved?.path ?? '<missing>'}`);
    console.log(`  my-custom-skill listed: ${listed.includes('my-custom-skill:project')}`);

    let failure: string | null = null;
    try {
      assert.ok(
        overrideDiagnostics.length > 0,
        'an invalid project SKILL.md must be reported (file + reason), got no diagnostic',
      );
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      console.log('EXPECTED:');
      console.log('  scanning the two invalid overrides emits at least one diagnostic naming the file and the missing field');
      console.log('ACTUAL:');
      console.log(`  no diagnostic emitted; dcf-valuation silently resolves to the builtin (${resolved?.source ?? '<missing>'}) and my-custom-skill is absent`);
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
