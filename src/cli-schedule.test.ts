import { afterEach, describe, expect, it } from 'bun:test';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { runScheduleCommand } from './cli-schedule.js';
import type { ScheduleCommandOptions, ScheduleJob } from './cli-schedule.js';
import type { AgentEvent } from './agent/types.js';
import type { ForecastLabRunOptions, ForecastLabRunResult } from './experiments/forecast-lab/runner.js';

const TEST_ROOT = join('.cramer-short', 'cli-schedule-test');
const TEST_HOME = join(TEST_ROOT, 'home');
const SCHEDULES_PATH = join(TEST_HOME, '.cramer-short', 'schedules.json');
const NOW = new Date('2026-05-02T12:34:56.000Z');

function cleanup(): void {
  rmSync(TEST_ROOT, { recursive: true, force: true });
}

function writeSchedules(jobs: readonly ScheduleJob[]): void {
  mkdirSync(dirname(SCHEDULES_PATH), { recursive: true });
  writeFileSync(SCHEDULES_PATH, `${JSON.stringify(jobs, null, 2)}\n`, 'utf8');
}

function makeOptions(overrides: ScheduleCommandOptions = {}): {
  options: ScheduleCommandOptions;
  output: string[];
  errors: string[];
  writes: string[];
  exitCodes: number[];
} {
  const output: string[] = [];
  const errors: string[] = [];
  const writes: string[] = [];
  const exitCodes: number[] = [];

  return {
    output,
    errors,
    writes,
    exitCodes,
    options: {
      schedulesPath: SCHEDULES_PATH,
      homeDir: TEST_HOME,
      cwd: process.cwd(),
      now: () => NOW,
      log: (message) => output.push(message),
      error: (message) => errors.push(message),
      write: (message) => writes.push(message),
      exit: (code) => exitCodes.push(code),
      ...overrides,
    },
  };
}

async function* fakeAgentRun(query: string, seenQueries: string[]): AsyncGenerator<AgentEvent> {
  seenQueries.push(query);
  yield { type: 'tool_start', tool: 'memory_search', args: { q: 'portfolio' } };
  yield { type: 'answer_chunk', chunk: 'Agent answer' };
  yield {
    type: 'done',
    answer: 'Agent answer',
    toolCalls: [],
    iterations: 1,
    totalTime: 5,
  };
}

function fakeForecastLabResult(
  profileId = 'btc-markov-short-horizon',
  options: {
    mutationMode?: 'structured';
    mutatorId?: 'replace-range' | 'search-replace' | 'insert-block';
    lineage?: {
      rootRunId: string;
      parentRunId?: string;
      generation: number;
    };
    candidateWorkspace?: {
      kind: 'current-worktree' | 'candidate-worktree';
      rootDir: string;
      branch: string;
    };
  } = {},
): ForecastLabRunResult {
  return {
    runId: 'schedule-forecast-run',
    manifest: {
      runId: 'schedule-forecast-run',
      startedAt: NOW.toISOString(),
      profileId,
      targetSubsystem: 'markov-distribution',
      candidateBranch: 'forecast-lab/schedule-forecast-run',
      allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
      artifactsPath: join('.cramer-short', 'experiments', 'runs', 'schedule-forecast-run'),
      mutationMode: options.mutationMode,
      lineage: options.lineage,
      mutationSpecSummary: options.mutatorId
        ? {
            mutatorId: options.mutatorId,
            targetFiles: ['src/tools/finance/markov-distribution.ts'],
            summary: 'Adjust Markov stability window',
          }
        : undefined,
      candidateWorkspace: options.candidateWorkspace,
    },
    baseline: { exitCode: 0 },
    candidate: { exitCode: 0 },
    decision: {
      decision: 'keep',
      reason: 'candidate passed',
      metrics: [
        {
          name: 'walkForwardShortHorizonTestExitCode',
          baseline: 0,
          candidate: 0,
          delta: 0,
        },
      ],
    },
    ledgerEntry: {
      runId: 'schedule-forecast-run',
      startedAt: NOW.toISOString(),
      profileId,
      targetSubsystem: 'markov-distribution',
      candidateBranch: 'forecast-lab/schedule-forecast-run',
      allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
      mutationMode: options.mutationMode,
      lineage: options.lineage,
      mutationSpecSummary: options.mutatorId
        ? {
            mutatorId: options.mutatorId,
            targetFiles: ['src/tools/finance/markov-distribution.ts'],
            summary: 'Adjust Markov stability window',
          }
        : undefined,
      candidateWorkspace: options.candidateWorkspace,
      baselineSummary: { exitCode: 0 },
      candidateSummary: { exitCode: 0 },
      decision: 'keep',
      reason: 'candidate passed',
      artifactsPath: join('.cramer-short', 'experiments', 'runs', 'schedule-forecast-run'),
    },
  };
}

afterEach(cleanup);

describe('schedule command', () => {
  it('lists existing agent jobs without requiring an explicit kind', async () => {
    writeSchedules([
      {
        id: 'morning-briefing',
        description: 'Daily watchlist briefing',
        query: 'Run the watchlist-briefing skill',
        outputFile: '~/.cramer-short/reports/{date}-briefing.md',
      },
    ]);
    const { options, output } = makeOptions();

    await runScheduleCommand(['list'], options);

    const text = output.join('\n');
    expect(text).toContain('Configured jobs');
    expect(text).toContain('morning-briefing');
    expect(text).toContain('Daily watchlist briefing');
    expect(text).toContain('output: ~/.cramer-short/reports/{date}-briefing.md');
    expect(text).not.toContain('kind: agent');
  });

  it('runs a backward-compatible agent job with an injected agent', async () => {
    writeSchedules([
      {
        id: 'morning-briefing',
        description: 'Daily watchlist briefing',
        query: 'Run the watchlist-briefing skill',
        outputFile: '~/.cramer-short/reports/{date}-briefing.md',
      },
    ]);
    const seenQueries: string[] = [];
    const { options, writes, errors } = makeOptions({
      createAgent: async () => ({
        run: (query) => fakeAgentRun(query, seenQueries),
      }),
    });

    await runScheduleCommand(['run', 'morning-briefing'], options);

    const outPath = join(TEST_HOME, '.cramer-short', 'reports', '2026-05-02-briefing.md');
    expect(seenQueries).toEqual(['Run the watchlist-briefing skill']);
    expect(writes.join('')).toBe('Agent answer');
    expect(errors).toEqual([]);
    expect(readFileSync(outPath, 'utf8')).toContain('Agent answer');
  });

  it('dispatches forecast_lab jobs to runForecastLab in dry-run mode by default', async () => {
    writeSchedules([
      {
        id: 'nightly-lab',
        kind: 'forecast_lab',
        description: 'Nightly forecast lab',
        profileId: 'btc-markov-short-horizon',
        maxIterations: 3,
        outputFile: `${TEST_ROOT}/reports/{date}-forecast-lab.md`,
      },
    ]);
    const calls: ForecastLabRunOptions[] = [];
    const { options, output } = makeOptions({
      runLab: async (runOptions) => {
        calls.push(runOptions);
        return fakeForecastLabResult(runOptions.profileId);
      },
    });

    await runScheduleCommand(['run', 'nightly-lab'], options);

    expect(calls).toEqual([
      expect.objectContaining({
        profileId: 'btc-markov-short-horizon',
        dryRun: true,
        skipMutation: false,
        progress: expect.any(Function),
        output: expect.any(Function),
      }),
    ]);
    expect(output.join('\n')).toContain('forecast-lab keep: candidate passed');
  });

  it('fails loudly before dispatching forecast_lab jobs with dryRun false and no explicit mutation mode', async () => {
    writeSchedules([
      {
        id: 'unsafe-lab',
        kind: 'forecast_lab',
        description: 'Unsafe lab',
        profileId: 'btc-markov-short-horizon',
        dryRun: false,
      },
    ]);
    let dispatchCount = 0;
    const { options, errors, exitCodes } = makeOptions({
      runLab: async () => {
        dispatchCount += 1;
        return fakeForecastLabResult();
      },
    });

    await runScheduleCommand(['run', 'unsafe-lab'], options);

    expect(dispatchCount).toBe(0);
    expect(exitCodes).toEqual([1]);
    expect(errors.join('\n')).toContain('dryRun: false without an explicit mutationMode');
    expect(errors.join('\n')).toContain('mutationMode: "structured"');
  });

  it('allows forecast_lab jobs with dryRun false when mutationMode is explicit', async () => {
    writeSchedules([
      {
        id: 'mutating-lab',
        kind: 'forecast_lab',
        profileId: 'btc-markov-short-horizon',
        dryRun: false,
        mutationMode: 'structured',
        keepWorktree: true,
        mutator: 'markov-longer-stability-window',
      },
    ]);
    const calls: ForecastLabRunOptions[] = [];
    const { options, errors } = makeOptions({
      runLab: async (runOptions) => {
        calls.push(runOptions);
        return fakeForecastLabResult(runOptions.profileId);
      },
    });

    await runScheduleCommand(['run', 'mutating-lab'], options);

    expect(errors).toEqual([]);
    expect(calls).toEqual([
      expect.objectContaining({
        profileId: 'btc-markov-short-horizon',
        dryRun: false,
        skipMutation: false,
        mutationMode: 'structured',
        keepWorktree: true,
        mutator: 'markov-longer-stability-window',
        progress: expect.any(Function),
        output: expect.any(Function),
      }),
    ]);
  });

  it('allows forecast_lab jobs with dryRun false when skipMutation is true', async () => {
    writeSchedules([
      {
        id: 'no-mutation-lab',
        kind: 'forecast_lab',
        profileId: 'btc-markov-short-horizon',
        dryRun: false,
        skipMutation: true,
      },
    ]);
    const calls: ForecastLabRunOptions[] = [];
    const { options, errors } = makeOptions({
      runLab: async (runOptions) => {
        calls.push(runOptions);
        return fakeForecastLabResult(runOptions.profileId);
      },
    });

    await runScheduleCommand(['run', 'no-mutation-lab'], options);

    expect(errors).toEqual([]);
    expect(calls).toEqual([
      expect.objectContaining({
        profileId: 'btc-markov-short-horizon',
        dryRun: false,
        skipMutation: true,
        keepWorktree: false,
        progress: expect.any(Function),
        output: expect.any(Function),
      }),
    ]);
  });

  it('fails loudly when mutation debugging controls are used without an explicit structured mutation mode', async () => {
    writeSchedules([
      {
        id: 'bad-debug-lab',
        kind: 'forecast_lab',
        profileId: 'btc-markov-short-horizon',
        dryRun: true,
        keepWorktree: true,
      } as ScheduleJob,
    ]);
    let dispatchCount = 0;
    const { options, errors, exitCodes } = makeOptions({
      runLab: async () => {
        dispatchCount += 1;
        return fakeForecastLabResult();
      },
    });

    await runScheduleCommand(['run', 'bad-debug-lab'], options);

    expect(dispatchCount).toBe(0);
    expect(exitCodes).toEqual([1]);
    expect(errors.join('\n')).toContain('keepWorktree/mutator debugging controls');
    expect(errors.join('\n')).toContain('mutationMode: "structured"');
  });

  it('writes forecast_lab output summaries when outputFile is configured', async () => {
    writeSchedules([
      {
        id: 'nightly-lab',
        kind: 'forecast_lab',
        description: 'Nightly forecast lab',
        profileId: 'btc-markov-short-horizon',
        outputFile: `${TEST_ROOT}/reports/{date}-forecast-lab.md`,
      },
    ]);
    const { options } = makeOptions({
      runLab: async (runOptions) => fakeForecastLabResult(runOptions.profileId),
    });

    await runScheduleCommand(['run', 'nightly-lab'], options);

    const outPath = join(TEST_ROOT, 'reports', '2026-05-02-forecast-lab.md');
    const text = readFileSync(outPath, 'utf8');
    expect(text).toContain('# Nightly forecast lab');
    expect(text).toContain('- Profile: btc-markov-short-horizon');
    expect(text).toContain('- Decision: keep');
    expect(text).toContain('| walkForwardShortHorizonTestExitCode | 0 | 0 | 0 |');
  });

  it('includes structured mutation metadata in schedule summaries', async () => {
    writeSchedules([
      {
        id: 'mutating-lab',
        kind: 'forecast_lab',
        description: 'Mutating forecast lab',
        profileId: 'btc-markov-short-horizon',
        dryRun: false,
        mutationMode: 'structured',
        mutator: 'markov-longer-stability-window',
        outputFile: `${TEST_ROOT}/reports/{date}-mutating-forecast-lab.md`,
      },
    ]);
    const { options, output } = makeOptions({
      runLab: async (runOptions) => fakeForecastLabResult(runOptions.profileId, {
        mutationMode: 'structured',
        mutatorId: 'search-replace',
        lineage: {
          rootRunId: 'forecast-lab-root-run',
          parentRunId: 'forecast-lab-parent-run',
          generation: 2,
        },
        candidateWorkspace: {
          kind: 'candidate-worktree',
          rootDir: join(TEST_ROOT, 'worktrees', 'schedule-forecast-run'),
          branch: 'forecast-lab/schedule-forecast-run',
        },
      }),
    });

    await runScheduleCommand(['run', 'mutating-lab'], options);

    const text = readFileSync(join(TEST_ROOT, 'reports', '2026-05-02-mutating-forecast-lab.md'), 'utf8');
    expect(output.join('\n')).toContain('mutation: structured');
    expect(output.join('\n')).toContain('selected mutator: markov-longer-stability-window');
    expect(output.join('\n')).toContain('mutation operator: search-replace');
    expect(output.join('\n')).toContain('lineage: root=forecast-lab-root-run, parent=forecast-lab-parent-run, generation=2');
    expect(output.join('\n')).toContain(`candidate workspace: candidate-worktree ${join(TEST_ROOT, 'worktrees', 'schedule-forecast-run')} (forecast-lab/schedule-forecast-run)`);
    expect(text).toContain('- Mutation mode: structured');
    expect(text).toContain('- Selected mutator: markov-longer-stability-window');
    expect(text).toContain('- Mutation operator: search-replace');
    expect(text).toContain('- Lineage: root=forecast-lab-root-run, parent=forecast-lab-parent-run, generation=2');
    expect(text).toContain(`- Candidate workspace: candidate-worktree @ ${join(TEST_ROOT, 'worktrees', 'schedule-forecast-run')}`);
    expect(text).toContain('- Candidate branch: forecast-lab/schedule-forecast-run');
  });

  it('fails loudly and does not dispatch forecast_lab jobs with missing profile ids', async () => {
    writeSchedules([
      {
        id: 'broken-lab',
        kind: 'forecast_lab',
        description: 'Broken lab',
      } as ScheduleJob,
    ]);
    let dispatchCount = 0;
    const { options, errors, exitCodes } = makeOptions({
      runLab: async () => {
        dispatchCount += 1;
        return fakeForecastLabResult();
      },
    });

    await runScheduleCommand(['run', 'broken-lab'], options);

    expect(dispatchCount).toBe(0);
    expect(exitCodes).toEqual([1]);
    expect(errors.join('\n')).toContain('missing profileId');
  });

  it('fails loudly and does not dispatch forecast_lab jobs with invalid profile ids', async () => {
    writeSchedules([
      {
        id: 'bad-lab',
        kind: 'forecast_lab',
        profileId: 'unknown-profile',
      },
    ]);
    let dispatchCount = 0;
    const { options, errors, exitCodes } = makeOptions({
      runLab: async () => {
        dispatchCount += 1;
        return fakeForecastLabResult();
      },
    });

    await runScheduleCommand(['run', 'bad-lab'], options);

    expect(dispatchCount).toBe(0);
    expect(exitCodes).toEqual([1]);
    expect(errors.join('\n')).toContain('Unknown forecast-lab profile id: unknown-profile');
  });

  it('fails loudly before dispatching forecast_lab jobs with conflicting skipMutation and mutationMode flags', async () => {
    writeSchedules([
      {
        id: 'ambiguous-lab',
        kind: 'forecast_lab',
        profileId: 'btc-markov-short-horizon',
        dryRun: false,
        skipMutation: true,
        mutationMode: 'structured',
      },
    ]);
    let dispatchCount = 0;
    const { options, errors, exitCodes } = makeOptions({
      runLab: async () => {
        dispatchCount += 1;
        return fakeForecastLabResult();
      },
    });

    await runScheduleCommand(['run', 'ambiguous-lab'], options);

    expect(dispatchCount).toBe(0);
    expect(exitCodes).toEqual([1]);
    expect(errors.join('\n')).toContain('cannot combine skipMutation: true with mutationMode');
  });

  it('reports unknown job ids without running any jobs', async () => {
    writeSchedules([
      {
        id: 'morning-briefing',
        description: 'Daily watchlist briefing',
        query: 'Run the watchlist-briefing skill',
        outputFile: '~/.cramer-short/reports/{date}-briefing.md',
      },
    ]);
    const { options, errors, exitCodes } = makeOptions();

    await runScheduleCommand(['run', 'missing-job'], options);

    expect(exitCodes).toEqual([1]);
    expect(errors.join('\n')).toContain('No job found with id "missing-job"');
    expect(existsSync(join(TEST_HOME, '.cramer-short', 'reports', '2026-05-02-briefing.md'))).toBe(false);
  });
});
