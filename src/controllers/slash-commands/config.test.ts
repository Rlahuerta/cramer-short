import { describe, expect, it } from 'bun:test';
import { buildConfigSummary, handleConfigSlashCommand } from './config.js';

function createHarness() {
  const answers: string[] = [];
  const queries: string[] = [];
  const errors: Array<string | null> = [];
  const statuses: string[] = [];
  let renderCount = 0;

  return {
    answers,
    queries,
    errors,
    statuses,
    get renderCount() {
      return renderCount;
    },
    options: {
      chatLog: {
        clearAll: () => {},
        addQuery: (query: string) => queries.push(query),
        resetToolGrouping: () => {},
        finalizeAnswer: (answer: string) => answers.push(answer),
      },
      currentModel: () => 'gpt-5.4',
      refreshError: () => {},
      requestRender: () => { renderCount += 1; },
      setError: (message: string | null) => errors.push(message),
      setStatus: (message: string) => statuses.push(message),
    },
  };
}

describe('config slash commands', () => {
  it('formats the current config summary', () => {
    const summary = buildConfigSummary();

    expect(summary).toContain('Current Configuration:');
    expect(summary).toContain('maxIterations');
    expect(summary).toContain('llmCallTimeoutMs');
  });

  it('handles /config show through the extracted handler', () => {
    const harness = createHarness();

    expect(handleConfigSlashCommand('/config show', harness.options)).toBe(true);
    expect(harness.queries).toEqual(['/config show']);
    expect(harness.answers[0]).toContain('Current Configuration:');
    expect(harness.renderCount).toBe(1);
  });

  it('reports usage errors for incomplete config set commands', () => {
    const harness = createHarness();

    expect(handleConfigSlashCommand('/config set ', harness.options)).toBe(true);
    expect(harness.errors).toEqual(['Usage: /config set <key> <value>']);
    expect(harness.renderCount).toBe(1);
  });

  it('ignores non-config slash commands', () => {
    const harness = createHarness();

    expect(handleConfigSlashCommand('/help', harness.options)).toBe(false);
    expect(harness.answers).toEqual([]);
    expect(harness.renderCount).toBe(0);
  });
});
