import { afterEach, describe, expect, it } from 'bun:test';
import { existsSync, readFileSync, rmSync } from 'node:fs';
import { join, resolve, sep } from 'node:path';
import { runForecastLabCommand } from '../../cli-forecast-lab.js';
import { getExperimentRunDir } from '../../utils/paths.js';
import { readLedgerEntries } from './ledger.js';
import type { ForecastLabCommandRunner } from './runner.js';
import { ForecastLabRunnerError, defaultForecastLabCommandRunner, runForecastLab } from './runner.js';

const TEST_LEDGER_DIR = join('.cramer-short', 'experiments', '__runner_test__');
const TEST_LEDGER_PATH = join(TEST_LEDGER_DIR, 'forecast-results.tsv');
const RUN_IDS = [
  'runner-test-dry-run',
  'runner-test-unknown',
  'runner-test-failed-candidate',
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

afterEach(cleanup);

describe('forecast-lab runner', () => {
  it('dry-run completes baseline, candidate, decision, and ledger write with an injected command runner', async () => {
    const calls: string[] = [];
    const result = await runForecastLab({
      profileId: 'btc-markov-short-horizon',
      dryRun: true,
      runId: 'runner-test-dry-run',
      now: () => new Date('2026-05-02T00:00:00.000Z'),
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner(calls),
    });

    expect(calls).toEqual([
      'baseline:walk-forward-short-horizon',
      'candidate:walk-forward-short-horizon',
    ]);
    expect(result.decision.decision).toBe('keep');
    expect(result.manifest.profileId).toBe('btc-markov-short-horizon');

    const runDir = getExperimentRunDir('runner-test-dry-run');
    for (const fileName of ['manifest.json', 'baseline.json', 'candidate.json', 'decision.json']) {
      expect(existsSync(join(runDir, fileName))).toBe(true);
      expect(resolve(join(runDir, fileName)).startsWith(resolve('.cramer-short', 'experiments') + sep)).toBe(true);
    }

    const candidate = JSON.parse(readFileSync(join(runDir, 'candidate.json'), 'utf8')) as Record<string, unknown>;
    expect(candidate.mutation).toBe('dry-run: no code mutation attempted');

    const entries = readLedgerEntries(TEST_LEDGER_PATH);
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      runId: 'runner-test-dry-run',
      decision: 'keep',
      artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-dry-run'),
    });
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

  it('fails loudly when real mutation is requested in V1', async () => {
    await expect(runForecastLab({
      profileId: 'btc-arbiter-replay',
      runId: 'runner-test-dry-run',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    })).rejects.toThrow(ForecastLabRunnerError);
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
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).toMatchObject({
      runId: 'runner-test-skip-mutation',
      decision: 'drop',
    });
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

    expect(output.join('\n')).toContain('cramer-short lab run <profileId> --dry-run');
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
});
