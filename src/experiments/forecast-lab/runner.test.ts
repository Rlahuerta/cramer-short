import { afterEach, describe, expect, it } from 'bun:test';
import { EventEmitter } from 'node:events';
import { existsSync, readFileSync, rmSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';
import { PassThrough } from 'node:stream';
import { runForecastLabCommand } from '../../cli-forecast-lab.js';
import { getExperimentRunDir } from '../../utils/paths.js';
import { readLedgerEntries } from './ledger.js';
import type { ForecastLabCommandRunner } from './runner.js';
import {
  ForecastLabRunnerError,
  createForecastLabCommandRunner,
  defaultForecastLabCommandRunner,
  runForecastLab,
} from './runner.js';

const TEST_LEDGER_DIR = join('.cramer-short', 'experiments', '__runner_test__');
const TEST_LEDGER_PATH = join(TEST_LEDGER_DIR, 'forecast-results.tsv');
const RUN_IDS = [
  'runner-test-dry-run',
  'runner-test-unknown',
  'runner-test-failed-candidate',
  'runner-test-structured',
  'runner-test-skip-mutation',
  'runner-test-outside-ledger',
];

function cleanup(): void {
  rmSync(TEST_LEDGER_DIR, { recursive: true, force: true });
  for (const runId of RUN_IDS) {
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
      profileId: 'btc-markov-short-horizon',
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
    expect(result.manifest.profileId).toBe('btc-markov-short-horizon');
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
      'forecast-lab: started btc-markov-short-horizon (runner-test-dry-run)',
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

  it('runs one real structured mutation inside an isolated candidate workspace and records its metadata', async () => {
    const calls: string[] = [];
    const progress: string[] = [];
    const result = await runForecastLab({
      profileId: 'btc-markov-short-horizon',
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
      lineage: {
        rootRunId: 'runner-test-structured',
        generation: 0,
      },
      mutationSpecSummary: result.manifest.mutationSpecSummary,
      candidateWorkspace,
    });
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
  });

  it('fails loudly when real mutation is requested for a profile without a shipped structured catalog', async () => {
    await expect(runForecastLab({
      profileId: 'btc-arbiter-replay',
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
      profileId: 'btc-markov-short-horizon',
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
    expect(output.join('\n')).toContain('btc-markov-short-horizon');
  });

  it('prints useful usage for missing commands', async () => {
    const output: string[] = [];

    await runForecastLabCommand([], { log: (message) => output.push(message) });

    expect(output.join('\n')).toContain('cramer-short lab run <profileId>');
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
    expect(output.join('\n')).toContain('cramer-short lab run <profileId>');
  });

  it('rejects unknown run flags instead of defaulting into a real mutation run', async () => {
    const output: string[] = [];
    const errors: string[] = [];
    let exitCode = 0;
    let runLabCalls = 0;

    await runForecastLabCommand(['run', 'btc-markov-short-horizon', '--dryrun'], {
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

    await runForecastLabCommand(['run', 'btc-markov-short-horizon', '--dry-run', '--skip-mutation'], {
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

  it('prints evolution and parameter summaries after a successful run', async () => {
    const output: string[] = [];
    const writes: string[] = [];

    await runForecastLabCommand(['run', 'btc-markov-short-horizon', '--dry-run'], {
      log: (message) => output.push(message),
      write: (chunk) => writes.push(chunk),
      runLab: async () => ({
        runId: 'runner-test-dry-run',
        manifest: {
          runId: 'runner-test-dry-run',
          startedAt: '2026-05-02T00:00:00.000Z',
          profileId: 'btc-markov-short-horizon',
          targetSubsystem: 'markov-distribution',
          baselineCommit: '0123456789abcdef0123456789abcdef01234567',
          candidateBranch: 'topic/forecast-lab-runner-test-dry-run',
          allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
          candidateWorkspace: {
            kind: 'candidate-worktree',
            rootDir: resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-dry-run'),
            branch: 'topic/forecast-lab-runner-test-dry-run',
          },
          artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-dry-run'),
        },
        baseline: { exitCode: 0 },
        candidate: { exitCode: 0 },
        decision: {
          decision: 'keep',
          reason: 'candidate walk-forward short-horizon test command must pass',
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
          runId: 'runner-test-dry-run',
          startedAt: '2026-05-02T00:00:00.000Z',
          profileId: 'btc-markov-short-horizon',
          targetSubsystem: 'markov-distribution',
          candidateBranch: 'topic/forecast-lab-runner-test-dry-run',
          allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
          candidateWorkspace: {
            kind: 'candidate-worktree',
            rootDir: resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-dry-run'),
            branch: 'topic/forecast-lab-runner-test-dry-run',
          },
          baselineSummary: { exitCode: 0 },
          candidateSummary: { exitCode: 0 },
          decision: 'keep',
          reason: 'candidate walk-forward short-horizon test command must pass',
          artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-dry-run'),
        },
      }),
    });

    const text = output.join('\n');
    expect(text).toContain('Running forecast-lab profile "btc-markov-short-horizon"...');
    expect(text).toContain('Evolution summary:');
    expect(text).toContain('baseline exitCode: 0');
    expect(text).toContain('candidate exitCode: 0');
    expect(text).toContain('Previous parameters (baseline gate):');
    expect(text).toContain('New parameters (candidate gate):');
    expect(text).toContain('walk-forward-short-horizon: command=bun test src/tools/finance/backtest/walk-forward-short-horizon.test.ts --timeout 480000');
    expect(text).toContain('forecast-lab keep: candidate walk-forward short-horizon test command must pass');
    expect(writes).toEqual([]);
  });
});
