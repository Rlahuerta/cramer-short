import { describe, expect, it } from 'bun:test';
import { parseSkillFile } from './loader.js';

describe('parseSkillFile', () => {
  it('parses YAML frontmatter', () => {
    const skill = parseSkillFile('---\nname: demo\ndescription: d\n---\nbody', '/x/SKILL.md', 'builtin');
    expect(skill.name).toBe('demo');
  });

  for (const lang of ['js', 'javascript']) {
    it(`rejects ---${lang} frontmatter without evaluating it`, () => {
      const g = globalThis as { __skillPwned?: boolean };
      delete g.__skillPwned;
      const content = `---${lang}\n(globalThis.__skillPwned = true, { name: 'x', description: 'y' })\n---\nbody`;
      expect(() => parseSkillFile(content, '/x/SKILL.md', 'builtin')).toThrow(/not allowed/);
      expect(g.__skillPwned).toBeUndefined();
    });
  }
});
