import { describe, expect, it } from 'bun:test';
import { handleExitCommand } from './core.js';
import type { HistoryItem } from '../types.js';

type ExitOptions = Parameters<typeof handleExitCommand>[1];

function createHarness() {
  const calls: string[] = [];
  const exits: Array<number | undefined> = [];
  const options: ExitOptions = {
    history: (): HistoryItem[] => [],
    currentModel: () => 'test-model',
    safeStopTui: () => { calls.push('safeStopTui'); },
    writeSessionDailySummary: async () => { calls.push('writeSessionDailySummary'); },
    flushSession: async () => { calls.push('flushSession'); },
    // The real default is process.exit; stub it so the test process survives.
    exitProcess: ((code?: number) => { exits.push(code); return undefined as never; }) as (code?: number) => never,
  };
  return { options, calls, exits };
}

describe('handleExitCommand', () => {
  for (const variant of ['/exit', '/quit', 'exit', 'quit']) {
    it(`exits for "${variant}" (case-insensitive, trimmed)`, async () => {
      const harness = createHarness();
      const handled = await handleExitCommand(`  ${variant.toUpperCase()}  `, harness.options);

      expect(handled).toBe(true);
      expect(harness.calls).toEqual([
        'safeStopTui',
        'writeSessionDailySummary',
        'flushSession',
      ]);
      expect(harness.exits).toEqual([0]);
    });
  }

  it('accepts mixed-case bare "Quit"', async () => {
    const harness = createHarness();
    expect(await handleExitCommand('Quit', harness.options)).toBe(true);
    expect(harness.exits).toEqual([0]);
  });

  it('does not exit for "/exitx"', async () => {
    const harness = createHarness();
    expect(await handleExitCommand('/exitx', harness.options)).toBe(false);
    expect(harness.calls).toEqual([]);
    expect(harness.exits).toEqual([]);
  });

  it('does not exit for a sentence containing exit', async () => {
    const harness = createHarness();
    expect(await handleExitCommand('exit now', harness.options)).toBe(false);
    expect(harness.calls).toEqual([]);
  });
});
