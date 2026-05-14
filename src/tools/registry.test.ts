import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import type { SkillMetadata } from '../skills/types.js';

let availableSkills: SkillMetadata[] = [];

const { getToolRegistry, getTools, buildToolDescriptions } = await import('./registry.js');
const { exaSearch, perplexitySearch, tavilySearch, xSearchTool } =
  await import('./search/index.js') as typeof import('./search/index.js');
const { skillTool } = await import('./skill.js') as typeof import('./skill.js');

const GATED_ENV_KEYS = [
  'EXASEARCH_API_KEY',
  'PERPLEXITY_API_KEY',
  'TAVILY_API_KEY',
  'X_BEARER_TOKEN',
] as const;

const savedEnv = new Map<string, string | undefined>();

function names(): string[] {
  return getToolRegistry('gpt-5.4', { discoverSkills: () => availableSkills }).map((entry) => entry.name);
}

beforeEach(() => {
  availableSkills = [];
  for (const key of GATED_ENV_KEYS) {
    savedEnv.set(key, process.env[key]);
    delete process.env[key];
  }
});

afterEach(() => {
  for (const key of GATED_ENV_KEYS) {
    const value = savedEnv.get(key);
    if (value === undefined) {
      delete process.env[key];
    } else {
      process.env[key] = value;
    }
  }
  savedEnv.clear();
});

describe('getToolRegistry core tools', () => {
  it('always registers core local tools without gated env vars', () => {
    expect(names()).toEqual(expect.arrayContaining([
      'sequential_thinking',
      'get_financials',
      'get_market_data',
      'web_fetch',
      'browser',
      'read_file',
      'write_file',
      'edit_file',
      'memory_search',
      'memory_get',
      'memory_update',
      'recall_financial_context',
      'store_financial_insight',
    ]));
  });

  it('omits web_search, x_search, and skill when their gates are closed', () => {
    expect(names()).not.toContain('web_search');
    expect(names()).not.toContain('x_search');
    expect(names()).not.toContain('skill');
  });

  it('getTools returns only tool instances and descriptions include registered names', () => {
    availableSkills = [{ name: 'local-skill', description: 'test', path: 'unused', source: 'project' }];
    process.env.X_BEARER_TOKEN = 'x-test';

    const options = { discoverSkills: () => availableSkills };
    const registry = getToolRegistry('gpt-5.4', options);
    const tools = getTools('gpt-5.4', options);
    expect(tools).toHaveLength(registry.length);
    expect(tools.map((tool) => tool.name)).toEqual(registry.map((entry) => entry.name));

    const descriptions = buildToolDescriptions('gpt-5.4', options);
    expect(descriptions).toContain('### sequential_thinking');
    expect(descriptions).toContain('### x_search');
    expect(descriptions).toContain('### skill');
  });
});

describe('getToolRegistry env-gated tools', () => {
  it('uses Exa for web_search before Perplexity and Tavily', () => {
    process.env.EXASEARCH_API_KEY = 'exa-test';
    process.env.PERPLEXITY_API_KEY = 'perplexity-test';
    process.env.TAVILY_API_KEY = 'tavily-test';

    const webSearch = getToolRegistry('gpt-5.4', { discoverSkills: () => availableSkills }).find((entry) => entry.name === 'web_search');
    expect(webSearch?.tool).toBe(exaSearch);
  });

  it('uses Perplexity for web_search when Exa is absent', () => {
    process.env.PERPLEXITY_API_KEY = 'perplexity-test';
    process.env.TAVILY_API_KEY = 'tavily-test';

    const webSearch = getToolRegistry('gpt-5.4', { discoverSkills: () => availableSkills }).find((entry) => entry.name === 'web_search');
    expect(webSearch?.tool).toBe(perplexitySearch);
  });

  it('uses Tavily for web_search when it is the only configured provider', () => {
    process.env.TAVILY_API_KEY = 'tavily-test';

    const webSearch = getToolRegistry('gpt-5.4', { discoverSkills: () => availableSkills }).find((entry) => entry.name === 'web_search');
    expect(webSearch?.tool).toBe(tavilySearch);
  });

  it('registers x_search only when X_BEARER_TOKEN is present', () => {
    expect(getToolRegistry('gpt-5.4', { discoverSkills: () => availableSkills }).find((entry) => entry.name === 'x_search')).toBeUndefined();

    process.env.X_BEARER_TOKEN = 'x-test';
    const xSearch = getToolRegistry('gpt-5.4', { discoverSkills: () => availableSkills }).find((entry) => entry.name === 'x_search');
    expect(xSearch?.tool).toBe(xSearchTool);
  });

  it('registers skill only when skill discovery returns at least one skill', () => {
    expect(getToolRegistry('gpt-5.4', { discoverSkills: () => availableSkills }).find((entry) => entry.name === 'skill')).toBeUndefined();

    availableSkills = [{ name: 'dcf', description: 'DCF workflow', path: 'unused', source: 'builtin' }];
    const skill = getToolRegistry('gpt-5.4', { discoverSkills: () => availableSkills }).find((entry) => entry.name === 'skill');
    expect(skill?.tool).toBe(skillTool);
  });
});
