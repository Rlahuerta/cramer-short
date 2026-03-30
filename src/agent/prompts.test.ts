import { describe, it, expect, mock } from 'bun:test';
import { join } from 'path';
import { tmpdir } from 'os';

const testDir = join(tmpdir(), `dexter-prompts-test-${Date.now()}`);

mock.module('../tools/registry.js', () => ({
  buildToolDescriptions: mock(() => 'mock tool descriptions'),
  getTools: mock(() => []),
  getToolRegistry: mock(() => []),
}));

// skills/index.js intentionally NOT mocked here — it re-exports from registry.js
// and loader.js via ESM live bindings. Any mock.module() call for index.js
// propagates back through those bindings into registry.js/loader.js, breaking
// skill.test.ts files that import directly from those sub-modules in the same
// Bun worker. The real discoverSkills/buildSkillMetadataSection are safe to use
// in tests because no assertion here checks for skill presence or absence.

mock.module('../utils/paths.js', () => ({
  dexterPath: mock((sub: string) => join(testDir, sub ?? 'SOUL.md')),
  getDexterDir: mock(() => testDir),
}));

const {
  getCurrentDate,
  buildSystemPrompt,
  buildIterationPrompt,
  buildGroupSection,
  loadSoulDocument,
} = await import('./prompts.js');

describe('getCurrentDate', () => {
  it('returns a non-empty string', () => {
    const date = getCurrentDate();
    expect(typeof date).toBe('string');
    expect(date.length).toBeGreaterThan(0);
  });

  it('includes the current year', () => {
    const date = getCurrentDate();
    const currentYear = new Date().getFullYear().toString();
    expect(date).toContain(currentYear);
  });

  it('contains the day of the week', () => {
    const date = getCurrentDate();
    const days = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
    expect(days.some(d => date.includes(d))).toBe(true);
  });
});

describe('loadSoulDocument', () => {
  it('returns a value (string or null)', async () => {
    const result = await loadSoulDocument();
    // The bundled SOUL.md exists in the repo, so it should be a string
    expect(typeof result === 'string' || result === null).toBe(true);
  });

  it('returns a string since the bundled SOUL.md exists', async () => {
    const result = await loadSoulDocument();
    // The SOUL.md in the project root acts as the bundled fallback
    expect(result).not.toBeNull();
    expect(typeof result).toBe('string');
  });
});

describe('buildSystemPrompt', () => {
  it('contains tool descriptions', () => {
    const prompt = buildSystemPrompt('gpt-5.4');
    expect(prompt).toContain('mock tool descriptions');
  });

  it('contains the current date', () => {
    const prompt = buildSystemPrompt('gpt-5.4');
    const currentYear = new Date().getFullYear().toString();
    expect(prompt).toContain(currentYear);
  });

  it('includes soul content when provided', () => {
    const soulContent = 'I am a focused financial analyst.';
    const prompt = buildSystemPrompt('gpt-5.4', soulContent);
    expect(prompt).toContain(soulContent);
  });

  it('does not include Identity section when soul is null', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null);
    expect(prompt).not.toContain('## Identity');
  });

  it('uses WhatsApp profile preamble when channel=whatsapp', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'whatsapp');
    expect(prompt.toLowerCase()).toContain('whatsapp');
  });

  it('uses CLI profile when channel=cli', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'cli');
    expect(prompt).toContain('CLI');
  });

  it('omits tables section for whatsapp channel', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'whatsapp');
    expect(prompt).not.toContain('## Tables');
  });

  it('includes tables section for CLI channel', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'cli');
    expect(prompt).toContain('Tables');
  });

  it('includes memory section', () => {
    const prompt = buildSystemPrompt('gpt-5.4');
    expect(prompt).toContain('Memory');
  });

  it('includes memory files list when provided', () => {
    const prompt = buildSystemPrompt('gpt-5.4', null, 'cli', undefined, ['goals.md', 'daily.md']);
    expect(prompt).toContain('goals.md');
    expect(prompt).toContain('daily.md');
  });

  it('includes memory context when provided', () => {
    const prompt = buildSystemPrompt(
      'gpt-5.4',
      null,
      'cli',
      undefined,
      [],
      'User is a long-term investor focused on dividends.',
    );
    expect(prompt).toContain('long-term investor');
  });

  it('includes group section when groupContext is provided', () => {
    const groupCtx = { groupName: 'Investors Club', activationMode: 'mention' as const };
    const prompt = buildSystemPrompt('gpt-5.4', null, 'whatsapp', groupCtx);
    expect(prompt).toContain('Investors Club');
    expect(prompt).toContain('Group Chat');
  });

  it('mentions Dexter as the assistant name', () => {
    const prompt = buildSystemPrompt('gpt-5.4');
    expect(prompt).toContain('Dexter');
  });
});

describe('buildIterationPrompt', () => {
  it('includes the original query', () => {
    const prompt = buildIterationPrompt('What is AAPL price?', '');
    expect(prompt).toContain('What is AAPL price?');
  });

  it('includes tool results when non-empty', () => {
    const results = '### web_search(query=AAPL)\n{"price":180}';
    const prompt = buildIterationPrompt('AAPL price?', results);
    expect(prompt).toContain('web_search');
    expect(prompt).toContain('180');
  });

  it('omits tool results section when empty', () => {
    const prompt = buildIterationPrompt('What is 2+2?', '');
    expect(prompt).not.toContain('Data retrieved');
  });

  it('omits tool results section when whitespace-only', () => {
    const prompt = buildIterationPrompt('test', '   ');
    expect(prompt).not.toContain('Data retrieved');
  });

  it('includes toolUsageStatus when provided', () => {
    const status = '## Tool Usage This Query\n\n- web_search: 2/3 calls';
    const prompt = buildIterationPrompt('test', '', status);
    expect(prompt).toContain('Tool Usage');
    expect(prompt).toContain('web_search: 2/3 calls');
  });

  it('does not include toolUsageStatus when not provided', () => {
    const prompt = buildIterationPrompt('test', '');
    expect(prompt).not.toContain('Tool Usage This Query');
  });

  it('always includes continuation instruction', () => {
    const prompt = buildIterationPrompt('test', '');
    expect(prompt).toContain('Continue working');
  });
});

describe('buildGroupSection', () => {
  it('includes the group name when provided', () => {
    const section = buildGroupSection({
      groupName: 'Stock Traders',
      activationMode: 'mention',
    });
    expect(section).toContain('Stock Traders');
  });

  it('handles missing group name gracefully', () => {
    const section = buildGroupSection({ activationMode: 'mention' });
    expect(section).toContain('WhatsApp group chat');
    expect(section).not.toContain('undefined');
  });

  it('includes members list when provided', () => {
    const section = buildGroupSection({
      activationMode: 'mention',
      membersList: 'Alice, Bob, Charlie',
    });
    expect(section).toContain('Alice, Bob, Charlie');
    expect(section).toContain('Group members');
  });

  it('includes activation mode mention text', () => {
    const section = buildGroupSection({ activationMode: 'mention' });
    expect(section).toContain('@-mentioned');
  });

  it('always includes ## Group Chat header', () => {
    const section = buildGroupSection({ activationMode: 'mention' });
    expect(section).toContain('## Group Chat');
  });

  it('includes group behavior guidelines', () => {
    const section = buildGroupSection({ activationMode: 'mention' });
    expect(section).toContain('Group behavior');
  });
});
