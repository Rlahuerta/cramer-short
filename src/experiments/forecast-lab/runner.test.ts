import { afterEach, describe, expect, it } from 'bun:test';
import { EventEmitter } from 'node:events';
import { existsSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { spawnSync } from 'node:child_process';
import { join, resolve, sep } from 'node:path';
import { PassThrough } from 'node:stream';
import { runForecastLabCommand } from '../../cli-forecast-lab.js';
import { getExperimentRunDir, getExperimentRunManifestPath } from '../../utils/paths.js';
import {
  getForecastLabCandidateWorktreePath,
  makeForecastLabCandidateBranch,
} from './git.js';
import { readLedgerEntries, writeRunManifest } from './ledger.js';
import type { ForecastLabCommandRunner } from './runner.js';
import {
  ForecastLabRunnerError,
  createForecastLabCommandRunner,
  defaultForecastLabCommandRunner,
  runForecastLab,
} from './runner.js';

const TEST_LEDGER_DIR = join('.cramer-short', 'experiments', '__runner_test__');
const TEST_LEDGER_PATH = join(TEST_LEDGER_DIR, 'forecast-results.tsv');
const TEST_ROUTING_STATS_PATH = join('.cramer-short', '__runner_test__', 'forecast-lab-routing-stats.json');
const RUN_IDS = [
  'runner-test-dry-run',
  'runner-test-unknown',
  'runner-test-failed-candidate',
  'runner-test-structured',
  'runner-test-structured-dirty-live-checkout',
  'runner-test-structured-parent-root',
  'runner-test-structured-parent-child',
  'runner-test-structured-auto-parent',
  'runner-test-structured-auto-child',
  'runner-test-structured-payload-parent',
  'runner-test-structured-payload-child',
  'runner-test-structured-keep-worktree',
  'runner-test-skip-mutation',
  'runner-test-outside-ledger',
];

function cleanup(): void {
  rmSync(TEST_LEDGER_DIR, { recursive: true, force: true });
  rmSync(TEST_ROUTING_STATS_PATH, { force: true });
  for (const runId of RUN_IDS) {
    const worktreePath = getForecastLabCandidateWorktreePath(runId);
    if (existsSync(worktreePath)) {
      spawnSync('git', ['worktree', 'remove', '--force', worktreePath], { stdio: 'ignore' });
    }
    spawnSync('git', ['branch', '-D', makeForecastLabCandidateBranch(runId)], { stdio: 'ignore' });
    rmSync(getExperimentRunDir(runId), { recursive: true, force: true });
  }
}

function passingRunner(calls: string[]): ForecastLabCommandRunner {
  return async (command, context) => {
    calls.push(`${context.phase}:${command.id}`);
    return {
      id: command.id,
      command: command.command,
      exitCode: 0,
      stdout: `${context.phase} ok`,
      stderr: '',
      durationMs: 1,
      timedOut: false,
    };
  };
}

class FakeSpawnedChild extends EventEmitter {
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  readonly killSignals: Array<number | NodeJS.Signals | undefined> = [];

  kill(signal?: number | NodeJS.Signals): boolean {
    this.killSignals.push(signal);
    return true;
  }
}

afterEach(cleanup);

describe('forecast-lab runner', () => {
  it('dry-run completes baseline, candidate, decision, and ledger write with an injected command runner', async () => {
    const calls: string[] = [];
    const progress: string[] = [];
    const output: string[] = [];
    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      dryRun: true,
      runId: 'runner-test-dry-run',
      now: () => new Date('2026-05-02T00:00:00.000Z'),
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner(calls),
      progress: (message) => progress.push(message),
      output: (chunk) => output.push(chunk),
    });

    expect(calls).toEqual([
      'baseline:walk-forward-short-horizon',
      'candidate:walk-forward-short-horizon',
    ]);
    expect(result.decision.decision).toBe('keep');
    expect(result.manifest.profileId).toBe('multi-asset-markov-short-horizon');
    expect(result.manifest.baselineCommit).toMatch(/^[0-9a-f]{40}$/);
    expect(result.manifest.candidateWorkspace).toBeUndefined();
    expect(result.manifest.effectiveMutationContract).toEqual({
      mode: 'structured',
      mutableFiles: [
        'src/tools/finance/markov-distribution.ts',
        'src/tools/finance/conformal.ts',
        'src/tools/finance/regime-calibrator.ts',
      ],
      allowedMutatorIds: ['search-replace'],
      allowMultipleCandidateAttempts: false,
    });

    const runDir = getExperimentRunDir('runner-test-dry-run');
    for (const fileName of ['manifest.json', 'baseline.json', 'candidate.json', 'decision.json']) {
      expect(existsSync(join(runDir, fileName))).toBe(true);
      expect(resolve(join(runDir, fileName)).startsWith(resolve('.cramer-short', 'experiments') + sep)).toBe(true);
    }

    const candidate = JSON.parse(readFileSync(join(runDir, 'candidate.json'), 'utf8')) as Record<string, unknown>;
    expect(candidate.mutation).toBe('dry-run: no code mutation attempted');
    expect(result.candidate as unknown).toEqual(candidate);
    const manifest = JSON.parse(readFileSync(join(runDir, 'manifest.json'), 'utf8')) as Record<string, unknown>;
    expect(manifest.baselineCommit).toMatch(/^[0-9a-f]{40}$/);
    expect(manifest).not.toHaveProperty('candidateWorkspace');
    expect(manifest.effectiveMutationContract).toEqual(result.manifest.effectiveMutationContract);

    const entries = readLedgerEntries(TEST_LEDGER_PATH);
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      runId: 'runner-test-dry-run',
      decision: 'keep',
      artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-dry-run'),
      effectiveMutationContract: result.manifest.effectiveMutationContract,
    });
    expect(entries[0]).not.toHaveProperty('candidateWorkspace');
    expect(progress).toEqual([
      'forecast-lab: started multi-asset-markov-short-horizon (runner-test-dry-run)',
      `forecast-lab: manifest written to ${join('.cramer-short', 'experiments', 'runs', 'runner-test-dry-run', 'manifest.json')}`,
      'forecast-lab: starting baseline gate',
      'baseline: running walk-forward-short-horizon — bun test src/tools/finance/backtest/walk-forward-short-horizon.test.ts --timeout 480000',
      'baseline: completed walk-forward-short-horizon (exit 0, 1ms)',
      `forecast-lab: baseline results written to ${join('.cramer-short', 'experiments', 'runs', 'runner-test-dry-run', 'baseline.json')}`,
      'forecast-lab: starting candidate gate (dry-run)',
      'candidate: running walk-forward-short-horizon — bun test src/tools/finance/backtest/walk-forward-short-horizon.test.ts --timeout 480000',
      'candidate: completed walk-forward-short-horizon (exit 0, 1ms)',
      `forecast-lab: candidate results written to ${join('.cramer-short', 'experiments', 'runs', 'runner-test-dry-run', 'candidate.json')}`,
      'forecast-lab: decision keep — candidate walk-forward short-horizon test command must pass; candidate walk-forward short-horizon test command must not regress versus baseline',
      `forecast-lab: ledger appended at ${TEST_LEDGER_PATH}`,
    ]);
    expect(output).toEqual([]);
  });

  it('rejects unknown profiles before running commands', async () => {
    const calls: string[] = [];

    await expect(runForecastLab({
      profileId: 'unknown-profile',
      dryRun: true,
      runId: 'runner-test-unknown',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner(calls),
    })).rejects.toThrow(/Unknown forecast-lab profile id/);

    expect(calls).toEqual([]);
    expect(existsSync(getExperimentRunDir('runner-test-unknown'))).toBe(false);
  });

  it('drops a failed candidate from command exit codes', async () => {
    const result = await runForecastLab({
      profileId: 'polymarket-selection-sanity',
      dryRun: true,
      runId: 'runner-test-failed-candidate',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: async (command, context) => ({
        id: command.id,
        command: command.command,
        exitCode: context.phase === 'candidate' ? 1 : 0,
        stdout: '',
        stderr: context.phase === 'candidate' ? 'failed' : '',
        durationMs: 1,
        timedOut: false,
      }),
    });

    expect(result.decision.decision).toBe('drop');
    expect(result.decision.reason).toMatch(/drop when the Polymarket forecast sanity test command fails/);
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).toMatchObject({
      runId: 'runner-test-failed-candidate',
      decision: 'drop',
    });
  });

  it('persists routing context to manifests and ledgers while updating per-profile routing stats', async () => {
    const autoRoutedResult = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      dryRun: true,
      runId: 'runner-test-dry-run',
      now: () => new Date('2026-05-02T00:00:00.000Z'),
      ledgerPath: TEST_LEDGER_PATH,
      routingStatsPath: TEST_ROUTING_STATS_PATH,
      routingContext: {
        originatingQuery: 'Improve the short-horizon Markov calibration.',
        selectedProfileId: 'multi-asset-markov-short-horizon',
        routerReason: 'Matched improvement intent and Markov short-horizon routing keywords.',
        invocationSource: 'auto-routed',
      },
      commandRunner: passingRunner([]),
    });

    const manualResult = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      dryRun: true,
      runId: 'runner-test-failed-candidate',
      now: () => new Date('2026-05-03T00:00:00.000Z'),
      ledgerPath: TEST_LEDGER_PATH,
      routingStatsPath: TEST_ROUTING_STATS_PATH,
      routingContext: {
        originatingQuery: 'Run the Markov short-horizon skill manually.',
        selectedProfileId: 'multi-asset-markov-short-horizon',
        routerReason: 'User explicitly requested the shipped Markov short-horizon skill.',
        invocationSource: 'manual-request',
      },
      commandRunner: async (command, context) => ({
        id: command.id,
        command: command.command,
        exitCode: context.phase === 'candidate' ? 1 : 0,
        stdout: '',
        stderr: context.phase === 'candidate' ? 'failed' : '',
        durationMs: 1,
        timedOut: false,
      }),
    });

    expect(autoRoutedResult.manifest.routingContext).toEqual({
      originatingQuery: 'Improve the short-horizon Markov calibration.',
      selectedProfileId: 'multi-asset-markov-short-horizon',
      routerReason: 'Matched improvement intent and Markov short-horizon routing keywords.',
      invocationSource: 'auto-routed',
    });
    expect(manualResult.ledgerEntry.routingContext).toEqual({
      originatingQuery: 'Run the Markov short-horizon skill manually.',
      selectedProfileId: 'multi-asset-markov-short-horizon',
      routerReason: 'User explicitly requested the shipped Markov short-horizon skill.',
      invocationSource: 'manual-request',
    });

    expect(
      JSON.parse(readFileSync(join(getExperimentRunDir('runner-test-dry-run'), 'manifest.json'), 'utf8')),
    ).toMatchObject({
      routingContext: autoRoutedResult.manifest.routingContext,
    });

    expect(readLedgerEntries(TEST_LEDGER_PATH)).toEqual([
      expect.objectContaining({
        runId: 'runner-test-dry-run',
        routingContext: autoRoutedResult.manifest.routingContext,
        decision: 'keep',
      }),
      expect.objectContaining({
        runId: 'runner-test-failed-candidate',
        routingContext: manualResult.manifest.routingContext,
        decision: 'drop',
      }),
    ]);

    expect(JSON.parse(readFileSync(TEST_ROUTING_STATS_PATH, 'utf8'))).toEqual({
      profiles: {
        'multi-asset-markov-short-horizon': {
          autoRoutedRuns: 1,
          droppedRuns: 1,
          keptRuns: 1,
          lastDecision: 'drop',
          lastRunAt: '2026-05-03T00:00:00.000Z',
          manualRequestedRuns: 1,
          totalRuns: 2,
        },
      },
      version: 1,
    });
  });

  it('runs one real structured mutation inside an isolated candidate workspace and records its metadata', async () => {
    const calls: string[] = [];
    const progress: string[] = [];
    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      runId: 'runner-test-structured',
      ledgerPath: TEST_LEDGER_PATH,
      progress: (message) => progress.push(message),
      commandRunner: async (command, context) => {
        calls.push(`${context.phase}:${command.id}:${context.cwd ?? ''}`);

        if (context.phase === 'candidate') {
          const candidateRoot = context.cwd!;
          expect(candidateRoot).toBe(resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured'));
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            '  momentumLookback: 14,',
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/conformal.ts'), 'utf8')).toContain(
            '  scoreAggregationCalibrationWindow: 96,',
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/regime-calibrator.ts'), 'utf8')).toContain(
            '  minSamplesPerRegime: 24,',
          );
        } else {
          expect(context.cwd).toBe(process.cwd());
        }

        return {
          id: command.id,
          command: command.command,
          exitCode: 0,
          stdout: `${context.phase} ok`,
          stderr: '',
          durationMs: 1,
          timedOut: false,
        };
      },
    });

    expect(calls).toEqual([
      `baseline:walk-forward-short-horizon:${process.cwd()}`,
      `candidate:walk-forward-short-horizon:${resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured')}`,
    ]);
    expect(result.decision.decision).toBe('keep');
    expect(result.manifest.mutationMode).toBe('structured');
    expect(result.manifest.parentRunId).toBeUndefined();
    expect(result.manifest.mutationId).toBe('markov-shorter-reactive-window');
    expect(result.manifest.mutationSummary).toBe(
      'Shorten Markov/conformal calibration windows for faster short-horizon adaptation.',
    );
    expect(result.manifest.lineage).toEqual({
      rootRunId: 'runner-test-structured',
      generation: 0,
    });
    expect(result.manifest.mutationSpecSummary).toEqual({
      mutatorId: 'search-replace',
      targetFiles: [
        'src/tools/finance/markov-distribution.ts',
        'src/tools/finance/conformal.ts',
        'src/tools/finance/regime-calibrator.ts',
      ],
      summary: 'Shorten Markov/conformal calibration windows for faster short-horizon adaptation.',
    });
    expect(result.manifest.mutationReplayPayload).toMatchObject({
      kind: 'markov-parameter-candidate',
      id: 'markov-shorter-reactive-window',
      profileId: 'multi-asset-markov-short-horizon',
    });
    const candidateWorkspace = result.manifest.candidateWorkspace;
    expect(candidateWorkspace).toEqual({
      kind: 'candidate-worktree',
      rootDir: resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured'),
      branch: 'topic/forecast-lab-runner-test-structured',
    });
    expect(candidateWorkspace).toBeDefined();
    expect(existsSync(candidateWorkspace!.rootDir)).toBe(false);

    const runDir = getExperimentRunDir('runner-test-structured');
    const candidate = JSON.parse(readFileSync(join(runDir, 'candidate.json'), 'utf8')) as Record<string, unknown>;
    expect(candidate.mutationMode).toBe('structured');
    expect(candidate.mutatedFiles).toEqual([
      'src/tools/finance/markov-distribution.ts',
      'src/tools/finance/conformal.ts',
      'src/tools/finance/regime-calibrator.ts',
    ]);
    expect(result.candidate as unknown).toEqual(candidate);
    expect(candidate.selectedMutator).toEqual({
      id: 'markov-shorter-reactive-window',
      mutatorId: 'search-replace',
    });
    expect(candidate.patchSummary).toEqual([
      'markov-distribution.ts: momentumLookback 20 → 14',
      'markov-distribution.ts: structuralBreakMinLength 60 → 48',
      'conformal.ts: scoreAggregationMinSamples 20 → 16',
      'conformal.ts: scoreAggregationCalibrationWindow 120 → 96',
      'regime-calibrator.ts: minSamplesPerRegime 30 → 24',
    ]);

    const decision = JSON.parse(readFileSync(join(runDir, 'decision.json'), 'utf8')) as Record<string, unknown>;
    expect(decision.mutationMode).toBe('structured');
    expect(decision.candidateWorkspace).toEqual(candidateWorkspace);

    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).toMatchObject({
      runId: 'runner-test-structured',
      decision: 'keep',
      mutationMode: 'structured',
      mutationId: 'markov-shorter-reactive-window',
      mutationSummary: 'Shorten Markov/conformal calibration windows for faster short-horizon adaptation.',
      lineage: {
        rootRunId: 'runner-test-structured',
        generation: 0,
      },
      mutationSpecSummary: result.manifest.mutationSpecSummary,
      candidateWorkspace,
    });
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).not.toHaveProperty('parentRunId');
    expect(progress).toContain(
      `forecast-lab: candidate workspace ${resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured')}`,
    );
    expect(progress).toContain('forecast-lab: selected mutator markov-shorter-reactive-window (search-replace)');
    expect(progress).toContain(
      'forecast-lab: mutated files src/tools/finance/markov-distribution.ts, src/tools/finance/conformal.ts, src/tools/finance/regime-calibrator.ts',
    );
    expect(progress).toContain(
      'forecast-lab: patch summary markov-distribution.ts: momentumLookback 20 → 14 | markov-distribution.ts: structuralBreakMinLength 60 → 48 | conformal.ts: scoreAggregationMinSamples 20 → 16 | conformal.ts: scoreAggregationCalibrationWindow 120 → 96 | regime-calibrator.ts: minSamplesPerRegime 30 → 24',
    );
    expect(progress.indexOf('forecast-lab: selected mutator markov-shorter-reactive-window (search-replace)'))
      .toBeLessThan(progress.indexOf('forecast-lab: starting baseline gate'));
  });

  it('selects structured mutations against the clean candidate workspace instead of a dirty live checkout', async () => {
    const liveCheckoutPath = join(process.cwd(), 'src/tools/finance/markov-distribution.ts');
    const originalContents = readFileSync(liveCheckoutPath, 'utf8');
    expect(originalContents).toContain('  momentumLookback: 20,');

    writeFileSync(liveCheckoutPath, originalContents.replace('  momentumLookback: 20,', '  momentumLookback: 28,'));

    try {
      const result = await runForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        mutationMode: 'structured',
        runId: 'runner-test-structured-dirty-live-checkout',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: async (command, context) => {
          if (context.phase === 'candidate') {
            const candidateRoot = context.cwd!;
            expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
              '  momentumLookback: 14,',
            );
            expect(readFileSync(liveCheckoutPath, 'utf8')).toContain('  momentumLookback: 28,');
          }

          return {
            id: command.id,
            command: command.command,
            exitCode: 0,
            stdout: `${context.phase} ok`,
            stderr: '',
            durationMs: 1,
            timedOut: false,
          };
        },
      });

      expect(result.manifest.mutationId).toBe('markov-shorter-reactive-window');
    } finally {
      writeFileSync(liveCheckoutPath, originalContents);
    }
  });

  it('seeds the next structured mutation from the last kept structured run lineage', async () => {
    await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      runId: 'runner-test-structured-parent-root',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    const progress: string[] = [];
    const calls: string[] = [];
    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      mutator: 'markov-faster-decay-reaction',
      runId: 'runner-test-structured-parent-child',
      ledgerPath: TEST_LEDGER_PATH,
      progress: (message) => progress.push(message),
      commandRunner: async (command, context) => {
        calls.push(`${context.phase}:${command.id}:${context.cwd ?? ''}`);
        if (context.phase === 'candidate') {
          const candidateRoot = context.cwd!;
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            '  momentumLookback: 14,',
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            '  transitionDecay: 0.94,',
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/conformal.ts'), 'utf8')).toContain(
            '  scoreAggregationCalibrationWindow: 96,',
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/conformal.ts'), 'utf8')).toContain(
            '  adaptiveBreakLearningRateMultiplier: 1.75,',
          );
        }

        return {
          id: command.id,
          command: command.command,
          exitCode: 0,
          stdout: `${context.phase} ok`,
          stderr: '',
          durationMs: 1,
          timedOut: false,
        };
      },
    });

    expect(calls).toEqual([
      `baseline:walk-forward-short-horizon:${process.cwd()}`,
      `candidate:walk-forward-short-horizon:${resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured-parent-child')}`,
    ]);
    expect(result.manifest.parentRunId).toBe('runner-test-structured-parent-root');
    expect(result.manifest.mutationId).toBe('markov-faster-decay-reaction');
    expect(result.manifest.mutationSummary).toBe(
      'Lower transition decay and raise adaptive conformal sensitivity for quicker regime resets.',
    );
    expect(result.manifest.lineage).toEqual({
      rootRunId: 'runner-test-structured-parent-root',
      parentRunId: 'runner-test-structured-parent-root',
      generation: 1,
    });
    expect(progress).toContain(
      'forecast-lab: seeded from kept run runner-test-structured-parent-root (1 replayed mutation)',
    );
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).toMatchObject({
      runId: 'runner-test-structured-parent-child',
      parentRunId: 'runner-test-structured-parent-root',
      mutationId: 'markov-faster-decay-reaction',
      mutationSummary: 'Lower transition decay and raise adaptive conformal sensitivity for quicker regime resets.',
      lineage: {
        rootRunId: 'runner-test-structured-parent-root',
        parentRunId: 'runner-test-structured-parent-root',
        generation: 1,
      },
    });
  });

  it('auto-selects the first applicable unused structured mutation after replay', async () => {
    await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      runId: 'runner-test-structured-auto-parent',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    const progress: string[] = [];
    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      runId: 'runner-test-structured-auto-child',
      ledgerPath: TEST_LEDGER_PATH,
      progress: (message) => progress.push(message),
      commandRunner: async (command, context) => {
        if (context.phase === 'candidate') {
          const candidateRoot = context.cwd!;
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            '  momentumLookback: 14,',
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            '  transitionDecay: 0.94,',
          );
        }

        return {
          id: command.id,
          command: command.command,
          exitCode: 0,
          stdout: `${context.phase} ok`,
          stderr: '',
          durationMs: 1,
          timedOut: false,
        };
      },
    });

    expect(result.manifest.parentRunId).toBe('runner-test-structured-auto-parent');
    expect(result.manifest.mutationId).toBe('markov-faster-decay-reaction');
    expect(progress).toContain('forecast-lab: selected mutator markov-faster-decay-reaction (search-replace)');
  });

  it('replays the persisted mutation payload even when the catalog entry is no longer available by mutationId', async () => {
    const parent = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      mutator: 'markov-shorter-reactive-window',
      runId: 'runner-test-structured-payload-parent',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    expect(parent.manifest.mutationReplayPayload).toBeDefined();
    writeRunManifest(getExperimentRunManifestPath('runner-test-structured-payload-parent'), {
      ...parent.manifest,
      mutationId: 'archived-shorter-reactive-window',
      mutationReplayPayload: {
        ...parent.manifest.mutationReplayPayload!,
        id: 'archived-shorter-reactive-window',
      },
    });

    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      mutator: 'markov-faster-decay-reaction',
      runId: 'runner-test-structured-payload-child',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: async (command, context) => {
        if (context.phase === 'candidate') {
          const candidateRoot = context.cwd!;
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            '  momentumLookback: 14,',
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            '  transitionDecay: 0.94,',
          );
        }

        return {
          id: command.id,
          command: command.command,
          exitCode: 0,
          stdout: `${context.phase} ok`,
          stderr: '',
          durationMs: 1,
          timedOut: false,
        };
      },
    });

    expect(result.manifest.parentRunId).toBe('runner-test-structured-payload-parent');
    expect(result.manifest.mutationId).toBe('markov-faster-decay-reaction');
  });

  it('preserves the candidate workspace when keepWorktree is enabled for a structured mutation run', async () => {
    const worktreePath = getForecastLabCandidateWorktreePath('runner-test-structured-keep-worktree');
    const progress: string[] = [];

    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      keepWorktree: true,
      mutator: 'markov-longer-stability-window',
      runId: 'runner-test-structured-keep-worktree',
      ledgerPath: TEST_LEDGER_PATH,
      progress: (message) => progress.push(message),
      commandRunner: passingRunner([]),
    });

    expect(result.decision.decision).toBe('keep');
    expect(result.manifest.candidateWorkspace?.rootDir).toBe(worktreePath);
    expect(existsSync(worktreePath)).toBe(true);
    expect(readFileSync(join(worktreePath, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
      '  momentumLookback: 28,',
    );
    expect(progress).toContain(`forecast-lab: keeping candidate workspace ${worktreePath}`);

    const candidate = JSON.parse(
      readFileSync(join(getExperimentRunDir('runner-test-structured-keep-worktree'), 'candidate.json'), 'utf8'),
    ) as Record<string, unknown>;
    expect(candidate.selectedMutator).toEqual({
      id: 'markov-longer-stability-window',
      mutatorId: 'search-replace',
    });
  });

  it('rejects ambiguous real mutation runs with no explicit mutation mode', async () => {
    await expect(runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      runId: 'runner-test-skip-mutation',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    })).rejects.toThrow(/explicit mutationMode/i);
  });

  it('fails loudly when real mutation is requested for a profile without a shipped structured catalog', async () => {
    await expect(runForecastLab({
      profileId: 'btc-arbiter-replay',
      mutationMode: 'structured',
      runId: 'runner-test-skip-mutation',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    })).rejects.toThrow(/requires a structured profile with a shipped catalog/i);
  });

  it('drops --skip-mutation runs because no candidate code change exists', async () => {
    const result = await runForecastLab({
      profileId: 'btc-arbiter-replay',
      skipMutation: true,
      runId: 'runner-test-skip-mutation',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    expect(result.decision.decision).toBe('drop');
    expect(result.decision.reason).toContain('mutation skipped by --skip-mutation');
    expect(result.manifest.candidateWorkspace).toBeUndefined();
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).toMatchObject({
      runId: 'runner-test-skip-mutation',
      decision: 'drop',
      effectiveMutationContract: {
        mode: 'dry-run',
        mutableFiles: ['src/tools/finance/forecast-arbitrator.ts', 'src/tools/finance/forecast-hooks.ts'],
        allowMultipleCandidateAttempts: false,
      },
    });
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).not.toHaveProperty('candidateWorkspace');
  });

  it('returns a failed result when the child emits a spawn error and clears timeout cleanup', async () => {
    const child = new FakeSpawnedChild();
    const output: string[] = [];
    const runner = createForecastLabCommandRunner((() => {
      queueMicrotask(() => {
        child.stderr.write('spawn stderr\n');
        child.emit('error', new Error('bad cwd'));
        child.emit('close');
      });
      return child as unknown as ReturnType<typeof import('node:child_process').spawn>;
    }) as typeof import('node:child_process').spawn);

    const result = await runner(
      {
        id: 'spawn-error-command',
        command: 'bun --version',
        timeoutMs: 10,
      },
      {
        phase: 'candidate',
        profile: {} as never,
        runId: 'runner-test-spawn-error',
        output: (chunk) => output.push(chunk),
      },
    );

    expect(result).toMatchObject({
      id: 'spawn-error-command',
      command: 'bun --version',
      exitCode: 1,
      stdout: '',
      timedOut: false,
    });
    expect(result.stderr).toBe('spawn stderr\nFailed to start command "spawn-error-command": bad cwd');
    expect(output).toEqual(['spawn stderr\n']);

    await new Promise((resolve) => setTimeout(resolve, 25));
    expect(child.killSignals).toEqual([]);
  });

  it('rejects unsafe profile commands before shell execution', async () => {
    await expect(defaultForecastLabCommandRunner(
      {
        id: 'unsafe-command',
        command: 'bun test src/tools/finance/polymarket-forecast.test.ts; git commit -m bad',
      },
      {
        phase: 'baseline',
        profile: {} as never,
        runId: 'runner-test-unsafe-command',
      },
    )).rejects.toThrow(/Unsafe forecast-lab command/);
  });

  it('refuses broad path writes outside .cramer-short/experiments', async () => {
    await expect(runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      dryRun: true,
      runId: 'runner-test-outside-ledger',
      ledgerPath: join('.cramer-short', 'forecast-results.tsv'),
      commandRunner: passingRunner([]),
    })).rejects.toThrow(/outside \.cramer-short\/experiments/);
  });
});

describe('forecast-lab CLI module', () => {
  it('lists profiles without launching the TUI', async () => {
    const output: string[] = [];

    await runForecastLabCommand(['list'], { log: (message) => output.push(message) });

    expect(output.join('\n')).toContain('Forecast-lab profiles:');
    expect(output.join('\n')).toContain('multi-asset-markov-short-horizon');
  });

  it('prints useful usage for missing commands', async () => {
    const output: string[] = [];

    await runForecastLabCommand([], { log: (message) => output.push(message) });

    expect(output.join('\n')).not.toContain('  cramer-short lab run <profileId>\n');
    expect(output.join('\n')).toContain('cramer-short lab run <profileId> --dry-run');
    expect(output.join('\n')).toContain('--mutation structured');
  });

  it('prints useful usage for missing run profile ids', async () => {
    const output: string[] = [];
    const errors: string[] = [];
    let exitCode = 0;

    await runForecastLabCommand(['run'], {
      log: (message) => output.push(message),
      error: (message) => errors.push(message),
      exit: (code) => {
        exitCode = code;
      },
    });

    expect(exitCode).toBe(1);
    expect(errors.join('\n')).toContain('Missing forecast-lab profile id.');
    expect(output.join('\n')).toContain('cramer-short lab run <profileId> --dry-run');
  });

  it('rejects unknown run flags instead of defaulting into a real mutation run', async () => {
    const output: string[] = [];
    const errors: string[] = [];
    let exitCode = 0;
    let runLabCalls = 0;

    await runForecastLabCommand(['run', 'multi-asset-markov-short-horizon', '--dryrun'], {
      log: (message) => output.push(message),
      error: (message) => errors.push(message),
      exit: (code) => {
        exitCode = code;
      },
      runLab: async () => {
        runLabCalls += 1;
        throw new Error('runLab should not be called');
      },
    });

    expect(exitCode).toBe(1);
    expect(runLabCalls).toBe(0);
    expect(errors.join('\n')).toContain('Unknown forecast-lab flag: "--dryrun"');
    expect(output.join('\n')).not.toContain('Running forecast-lab profile');
  });

  it('rejects conflicting no-mutation flags', async () => {
    const errors: string[] = [];
    let exitCode = 0;
    let runLabCalls = 0;

    await runForecastLabCommand(['run', 'multi-asset-markov-short-horizon', '--dry-run', '--skip-mutation'], {
      error: (message) => errors.push(message),
      exit: (code) => {
        exitCode = code;
      },
      runLab: async () => {
        runLabCalls += 1;
        throw new Error('runLab should not be called');
      },
    });

    expect(exitCode).toBe(1);
    expect(runLabCalls).toBe(0);
    expect(errors.join('\n')).toContain(
      'Conflicting forecast-lab flags: --dry-run and --skip-mutation cannot be used together.',
    );
  });

  it('requires an explicit mutation mode before running a real mutation', async () => {
    const errors: string[] = [];
    let exitCode = 0;
    let runLabCalls = 0;

    await runForecastLabCommand(['run', 'multi-asset-markov-short-horizon'], {
      error: (message) => errors.push(message),
      exit: (code) => {
        exitCode = code;
      },
      runLab: async () => {
        runLabCalls += 1;
        throw new Error('runLab should not be called');
      },
    });

    expect(exitCode).toBe(1);
    expect(runLabCalls).toBe(0);
    expect(errors.join('\n')).toContain('Real forecast-lab mutation requires an explicit flag: --mutation structured.');
  });

  it('passes explicit structured mutation controls through to the runner', async () => {
    const output: string[] = [];
    const runLabCalls: Array<Record<string, unknown>> = [];

    await runForecastLabCommand([
      'run',
      'multi-asset-markov-short-horizon',
      '--mutation',
      'structured',
      '--mutator',
      'markov-longer-stability-window',
      '--keep-worktree',
    ], {
      log: (message) => output.push(message),
      runLab: async (options) => {
        runLabCalls.push(options as unknown as Record<string, unknown>);
        return {
          runId: 'runner-test-explicit-structured',
          manifest: {
            runId: 'runner-test-explicit-structured',
            startedAt: '2026-05-02T00:00:00.000Z',
            profileId: 'multi-asset-markov-short-horizon',
            targetSubsystem: 'markov-distribution',
            baselineCommit: '0123456789abcdef0123456789abcdef01234567',
            candidateBranch: 'topic/forecast-lab-runner-test-explicit-structured',
            allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
            artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-explicit-structured'),
          },
          baseline: { exitCode: 0 },
          candidate: { exitCode: 0 },
          decision: {
            decision: 'keep',
            reason: 'candidate passed',
            metrics: [],
          },
          ledgerEntry: {
            runId: 'runner-test-explicit-structured',
            startedAt: '2026-05-02T00:00:00.000Z',
            profileId: 'multi-asset-markov-short-horizon',
            targetSubsystem: 'markov-distribution',
            candidateBranch: 'topic/forecast-lab-runner-test-explicit-structured',
            allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
            baselineSummary: { exitCode: 0 },
            candidateSummary: { exitCode: 0 },
            decision: 'keep',
            reason: 'candidate passed',
            artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-explicit-structured'),
          },
        };
      },
    });

    expect(runLabCalls).toEqual([
      expect.objectContaining({
        profileId: 'multi-asset-markov-short-horizon',
        dryRun: false,
        skipMutation: false,
        mutationMode: 'structured',
        keepWorktree: true,
        mutator: 'markov-longer-stability-window',
        progress: expect.any(Function),
        output: expect.any(Function),
      }),
    ]);
    expect(output.join('\n')).toContain('Running forecast-lab profile "multi-asset-markov-short-horizon" with structured mutation...');
  });

  it('rejects mutator overrides without an explicit structured mutation mode', async () => {
    const errors: string[] = [];
    let exitCode = 0;
    let runLabCalls = 0;

    await runForecastLabCommand(['run', 'multi-asset-markov-short-horizon', '--mutator', 'markov-longer-stability-window'], {
      error: (message) => errors.push(message),
      exit: (code) => {
        exitCode = code;
      },
      runLab: async () => {
        runLabCalls += 1;
        throw new Error('runLab should not be called');
      },
    });

    expect(exitCode).toBe(1);
    expect(runLabCalls).toBe(0);
    expect(errors.join('\n')).toContain('--keep-worktree and --mutator require --mutation structured');
  });

  it('prints evolution and parameter summaries after a successful run', async () => {
    const output: string[] = [];
    const writes: string[] = [];

    await runForecastLabCommand(['run', 'btc-markov-ultra-short-horizon', '--mutation', 'structured'], {
      log: (message) => output.push(message),
      write: (chunk) => writes.push(chunk),
      runLab: async () => ({
        runId: 'runner-test-structured-summary',
        manifest: {
          runId: 'runner-test-structured-summary',
          startedAt: '2026-05-02T00:00:00.000Z',
          profileId: 'btc-markov-ultra-short-horizon',
          targetSubsystem: 'markov-distribution',
          baselineCommit: '0123456789abcdef0123456789abcdef01234567',
          candidateBranch: 'topic/forecast-lab-runner-test-structured-summary',
          allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
          mutationMode: 'structured',
          mutationId: 'markov-shorter-reactive-window',
          mutationSummary: 'Shorten Markov/conformal calibration windows for faster short-horizon adaptation.',
          mutationReplayPayload: {
            kind: 'markov-parameter-candidate',
            id: 'markov-shorter-reactive-window',
            profileId: 'btc-markov-ultra-short-horizon',
            mutatorId: 'search-replace',
            specSummary: {
              mutatorId: 'search-replace',
              targetFiles: [
                'src/tools/finance/markov-distribution.ts',
                'src/tools/finance/conformal.ts',
              ],
              summary: 'Shorten Markov/conformal calibration windows for faster short-horizon adaptation.',
            },
            patchSummary: [
              'markov-distribution.ts: momentumLookback 20 → 14',
              'conformal.ts: scoreAggregationCalibrationWindow 120 → 96',
            ],
            edits: [
              {
                kind: 'search-replace',
                parameterId: 'momentumLookback',
                filePath: 'src/tools/finance/markov-distribution.ts',
                beforeValue: 20,
                afterValue: 14,
                search: '  momentumLookback: 20,',
                replace: '  momentumLookback: 14,',
                expectedReplacements: 1,
              },
              {
                kind: 'search-replace',
                parameterId: 'scoreAggregationCalibrationWindow',
                filePath: 'src/tools/finance/conformal.ts',
                beforeValue: 120,
                afterValue: 96,
                search: '  scoreAggregationCalibrationWindow: 120,',
                replace: '  scoreAggregationCalibrationWindow: 96,',
                expectedReplacements: 1,
              },
            ],
          },
          candidateWorkspace: {
            kind: 'candidate-worktree',
            rootDir: resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured-summary'),
            branch: 'topic/forecast-lab-runner-test-structured-summary',
          },
          artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-structured-summary'),
        },
        baseline: { exitCode: 0 },
        candidate: { exitCode: 0 },
        decision: {
          decision: 'keep',
          reason: 'candidate BTC ultra-short-horizon test command must pass',
          metrics: [
            {
              name: 'walkForwardBtcUltraShortHorizonTestExitCode',
              baseline: 0,
              candidate: 0,
              delta: 0,
            },
          ],
        },
        ledgerEntry: {
          runId: 'runner-test-structured-summary',
          startedAt: '2026-05-02T00:00:00.000Z',
          profileId: 'btc-markov-ultra-short-horizon',
          targetSubsystem: 'markov-distribution',
          candidateBranch: 'topic/forecast-lab-runner-test-structured-summary',
          allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
          mutationMode: 'structured',
          mutationId: 'markov-shorter-reactive-window',
          mutationSummary: 'Shorten Markov/conformal calibration windows for faster short-horizon adaptation.',
          candidateWorkspace: {
            kind: 'candidate-worktree',
            rootDir: resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured-summary'),
            branch: 'topic/forecast-lab-runner-test-structured-summary',
          },
          baselineSummary: { exitCode: 0 },
          candidateSummary: { exitCode: 0 },
          decision: 'keep',
          reason: 'candidate BTC ultra-short-horizon test command must pass',
          artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-structured-summary'),
        },
      }),
    });

    const text = output.join('\n');
    expect(text).toContain('Running forecast-lab profile "btc-markov-ultra-short-horizon" with structured mutation...');
    expect(text).toContain('Evolution summary:');
    expect(text).toContain('baseline exitCode: 0');
    expect(text).toContain('candidate exitCode: 0');
    expect(text).toContain('Mutation summary: Shorten Markov/conformal calibration windows for faster short-horizon adaptation.');
    expect(text).toContain('mutation id: markov-shorter-reactive-window');
    expect(text).toContain('Previous parameters (baseline defaults):');
    expect(text).toContain('New parameters (candidate mutation):');
    expect(text).toContain('markov-distribution.ts: momentumLookback = 20');
    expect(text).toContain('markov-distribution.ts: momentumLookback = 14');
    expect(text).toContain('conformal.ts: scoreAggregationCalibrationWindow = 120');
    expect(text).toContain('conformal.ts: scoreAggregationCalibrationWindow = 96');
    expect(text).toContain('forecast-lab keep: candidate BTC ultra-short-horizon test command must pass');
    expect(writes).toEqual([]);
  });
});
