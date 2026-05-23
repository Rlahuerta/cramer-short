import { describe, it, expect } from 'bun:test';
import { routeCommand, type IndexHandlers } from './index-routing.js';

function makeHandlers(): { calls: Record<string, string[][]>; handlers: IndexHandlers } {
  const calls: Record<string, string[][]> = { schedule: [], lab: [], replayLabel: [], cli: [] };
  const handlers: IndexHandlers = {
    schedule: async (args) => { calls.schedule.push(args); },
    lab: async (args) => { calls.lab.push(args); },
    replayLabel: async (args) => { calls.replayLabel.push(args); },
    cli: async () => { calls.cli.push([]); },
  };
  return { calls, handlers };
}

describe('index argv routing', () => {
  it('routes replay-label to replayLabel handler with argv.slice(3) and does not call cli', async () => {
    const { calls, handlers } = makeHandlers();
    await routeCommand(['bun', 'src/index.tsx', 'replay-label', 'run', '--input', 'x.jsonl'], handlers);
    expect(calls.replayLabel).toEqual([['run', '--input', 'x.jsonl']]);
    expect(calls.cli).toEqual([]);
    expect(calls.schedule).toEqual([]);
    expect(calls.lab).toEqual([]);
  });

  it('routes schedule to schedule handler with argv.slice(3)', async () => {
    const { calls, handlers } = makeHandlers();
    await routeCommand(['bun', 'src/index.tsx', 'schedule', 'list'], handlers);
    expect(calls.schedule).toEqual([['list']]);
    expect(calls.cli).toEqual([]);
  });

  it('routes lab to lab handler with argv.slice(3)', async () => {
    const { calls, handlers } = makeHandlers();
    await routeCommand(['bun', 'src/index.tsx', 'lab', 'run'], handlers);
    expect(calls.lab).toEqual([['run']]);
    expect(calls.cli).toEqual([]);
  });

  it('falls through to cli for an unrecognised subcommand', async () => {
    const { calls, handlers } = makeHandlers();
    await routeCommand(['bun', 'src/index.tsx', 'something-else'], handlers);
    expect(calls.cli).toEqual([[]]);
    expect(calls.replayLabel).toEqual([]);
  });

  it('falls through to cli when no subcommand is supplied', async () => {
    const { calls, handlers } = makeHandlers();
    await routeCommand(['bun', 'src/index.tsx'], handlers);
    expect(calls.cli).toEqual([[]]);
    expect(calls.replayLabel).toEqual([]);
  });
});
