import { afterEach, beforeEach, describe, expect, spyOn, test } from 'bun:test';
import { mkdirSync, mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

describe('discoverSkills with invalid project skill overrides (PI-56)', () => {
  let root: string;
  let savedCwd: string;
  let warn: ReturnType<typeof spyOn>;
  let messages: string[];

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), 'cs-skills-'));
    savedCwd = process.cwd();
    process.chdir(root);
    messages = [];
    warn = spyOn(console, 'warn').mockImplementation((...args: unknown[]) => {
      messages.push(args.map(String).join(' '));
    });
  });

  afterEach(() => {
    warn.mockRestore();
    process.chdir(savedCwd);
    rmSync(root, { recursive: true, force: true });
  });

  function writeProjectSkill(dir: string, lines: string[]): string {
    const skillDir = join(root, '.cramer-short', 'skills', dir);
    mkdirSync(skillDir, { recursive: true });
    const file = join(skillDir, 'SKILL.md');
    writeFileSync(file, lines.join('\n'), 'utf-8');
    return file;
  }

  test('warns with the file and the missing field, skips the file, and keeps discovering the rest', async () => {
    const noDescription = writeProjectSkill('dcf-valuation', ['---', 'name: dcf-valuation', '---', 'BODY', '']);
    const noName = writeProjectSkill('my-custom-skill', ['---', 'description: no name here', '---', 'BODY', '']);
    writeProjectSkill('my-good-skill', ['---', 'name: my-good-skill', 'description: valid', '---', 'BODY', '']);

    // SKILL_DIRECTORIES resolves the project dir from cwd at import time: load a fresh copy after chdir.
    const { discoverSkills } = (await import(`./registry.js?pi56=${Date.now()}`)) as typeof import('./registry.js');
    const skills = discoverSkills();

    expect(messages.some((m) => m.includes(noDescription) && m.includes("'description'"))).toBe(true);
    expect(messages.some((m) => m.includes(noName) && m.includes("'name'"))).toBe(true);

    expect(skills.find((s) => s.name === 'my-good-skill')?.source).toBe('project');
    expect(skills.find((s) => s.name === 'dcf-valuation')?.source).toBe('builtin');
    expect(skills.find((s) => s.name === 'my-custom-skill')).toBeUndefined();
  });

  test('warns on malformed YAML frontmatter instead of throwing', async () => {
    const broken = writeProjectSkill('broken-yaml', ['---', 'name: [unclosed', 'description: x', '---', 'BODY', '']);

    const { discoverSkills } = (await import(`./registry.js?pi56yaml=${Date.now()}`)) as typeof import('./registry.js');
    expect(() => discoverSkills()).not.toThrow();

    expect(messages.some((m) => m.includes(broken))).toBe(true);
  });
});
