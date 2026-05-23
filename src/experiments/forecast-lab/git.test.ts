import { afterEach, describe, expect, it, setDefaultTimeout } from 'bun:test';
import { existsSync, mkdirSync, readFileSync, rmSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { getForecastLabProfile } from './profiles.js';
import {
  ForecastLabGitError,
  applyForecastLabCandidateEdits,
  prepareForecastLabCandidateWorkspace,
  prepareForecastLabPromotionWorkspace,
  assertSafeForecastLabWorktreePath,
  assertSafeGitBranchName,
  getForecastLabCandidateWorktreePath,
  getForecastLabPromotionWorktreePath,
  makeForecastLabCandidateBranch,
  makeForecastLabPromotionBranch,
  withForecastLabCandidateWorkspace,
} from './git.js';

const WORKTREE_RUN_IDS = [
  'git-test-safe-path',
  'git-test-cleanup',
  'git-test-edit-policy',
  'git-test-existing-worktree',
  'git-test-cleanup-secondary-error',
  'git-test-promotion-safe-path',
  'git-test-promotion-cleanup',
] as const;

setDefaultTimeout(120_000);

function listForecastLabTestBranches(): ReadonlySet<string> {
  const result = spawnSync('git', ['branch', '--list', 'topic/forecast-lab-git-test*', 'topic/forecast-lab-promote-git-test*'], {
    cwd: process.cwd(),
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  return new Set(
    (result.stdout ?? '')
      .split('\n')
      .map((line) => line.replace(/^\*\s*/, '').trim())
      .filter(Boolean),
  );
}

function cleanupWorktree(runId: string, branches?: ReadonlySet<string>): void {
  const branch = makeForecastLabCandidateBranch(runId);
  const worktreePath = getForecastLabCandidateWorktreePath(runId);

  if (existsSync(worktreePath)) {
    spawnSync('git', ['worktree', 'remove', '--force', worktreePath], {
      cwd: process.cwd(),
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  }
  rmSync(worktreePath, { recursive: true, force: true });
  if (branches?.has(branch) ?? branchExists(branch)) {
    spawnSync('git', ['branch', '-D', branch], {
      cwd: process.cwd(),
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  }
}

function cleanupPromotionWorktree(runId: string, branches?: ReadonlySet<string>): void {
  const branch = makeForecastLabPromotionBranch(runId);
  const worktreePath = getForecastLabPromotionWorktreePath(runId);

  if (existsSync(worktreePath)) {
    spawnSync('git', ['worktree', 'remove', '--force', worktreePath], {
      cwd: process.cwd(),
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  }
  rmSync(worktreePath, { recursive: true, force: true });
  if (branches?.has(branch) ?? branchExists(branch)) {
    spawnSync('git', ['branch', '-D', branch], {
      cwd: process.cwd(),
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
  }
}

function branchExists(branch: string): boolean {
  const result = spawnSync('git', ['show-ref', '--verify', '--quiet', `refs/heads/${branch}`], {
    cwd: process.cwd(),
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  return result.status === 0;
}

afterEach(() => {
  const branches = listForecastLabTestBranches();
  for (const runId of WORKTREE_RUN_IDS) {
    cleanupWorktree(runId, branches);
    cleanupPromotionWorktree(runId, branches);
  }
});

describe('forecast-lab git helpers', () => {
  it('accepts safe branch names and rejects unsafe ones', () => {
    expect(makeForecastLabCandidateBranch('git-test-safe-path')).toBe('topic/forecast-lab-git-test-safe-path');
    expect(() => assertSafeGitBranchName('topic/forecast-lab-safe_branch.1')).not.toThrow();
    expect(() => assertSafeGitBranchName('../bad')).toThrow(/Unsafe git branch name/);
    expect(() => assertSafeGitBranchName('topic//double-slash')).toThrow(/Unsafe git branch name/);
    expect(() => assertSafeGitBranchName('topic/with space')).toThrow(/Unsafe git branch name/);
  });

  it('returns safe worktree paths under .cramer-short/experiments/worktrees', () => {
    const worktreePath = getForecastLabCandidateWorktreePath('git-test-safe-path');

    expect(worktreePath).toBe(resolve('.cramer-short', 'experiments', 'worktrees', 'git-test-safe-path'));
    expect(() => assertSafeForecastLabWorktreePath(worktreePath)).not.toThrow();
    expect(() => assertSafeForecastLabWorktreePath(resolve('.cramer-short', 'experiments', 'runs', 'git-test-safe-path'))).toThrow(
      /Unsafe forecast-lab worktree path/,
    );
    expect(() => assertSafeForecastLabWorktreePath(resolve('.cramer-short', 'experiments', 'worktrees'))).toThrow(
      /Unsafe forecast-lab worktree path/,
    );
  });

  it('creates promotion staging worktrees with isolated branch and path names', () => {
    const runId = 'git-test-promotion-safe-path';

    expect(makeForecastLabPromotionBranch(runId)).toBe('topic/forecast-lab-promote-git-test-promotion-safe-path');
    expect(getForecastLabPromotionWorktreePath(runId)).toBe(
      resolve('.cramer-short', 'experiments', 'worktrees', 'promote-git-test-promotion-safe-path'),
    );
  });

  it('cleans up the candidate branch and worktree after callback failure', async () => {
    const runId = 'git-test-cleanup';
    const branch = makeForecastLabCandidateBranch(runId);
    const worktreePath = getForecastLabCandidateWorktreePath(runId);
    const profile = getForecastLabProfile('multi-asset-markov-short-horizon');

    await expect(
      withForecastLabCandidateWorkspace(runId, () => {
        expect(existsSync(worktreePath)).toBe(true);
        expect(branchExists(branch)).toBe(true);
        throw new Error('boom');
      }, {
        materializePaths: profile.mutation.mutableFiles,
      }),
    ).rejects.toThrow('boom');

    expect(existsSync(worktreePath)).toBe(false);
    expect(branchExists(branch)).toBe(false);
  });

  it('cleans up the promotion staging branch and worktree after a verification failure', () => {
    const runId = 'git-test-promotion-cleanup';
    const branch = makeForecastLabPromotionBranch(runId);
    const worktreePath = getForecastLabPromotionWorktreePath(runId);
    const profile = getForecastLabProfile('multi-asset-markov-short-horizon');
    const workspace = prepareForecastLabPromotionWorkspace(runId, {
      materializePaths: profile.mutation.mutableFiles,
    });

    expect(workspace.metadata.branch).toBe(branch);
    expect(workspace.metadata.rootDir).toBe(worktreePath);
    expect(existsSync(worktreePath)).toBe(true);
    expect(branchExists(branch)).toBe(true);

    workspace.cleanup();

    expect(existsSync(worktreePath)).toBe(false);
    expect(branchExists(branch)).toBe(false);
  });

  it('refuses to reuse a pre-existing worktree path before git unregisters it', () => {
    const runId = 'git-test-existing-worktree';
    const worktreePath = getForecastLabCandidateWorktreePath(runId);
    const calls: string[] = [];

    mkdirSync(worktreePath, { recursive: true });

    expect(() =>
      prepareForecastLabCandidateWorkspace(runId, {
        gitCommandRunner: (args) => {
          calls.push(args.join(' '));

          if (args[0] === 'rev-parse') {
            return {
              exitCode: 0,
              stdout: '0123456789abcdef0123456789abcdef01234567\n',
              stderr: '',
            };
          }

          return {
            exitCode: 0,
            stdout: '',
            stderr: '',
          };
        },
      }),
    ).toThrow(/Refusing to reuse existing forecast-lab worktree path/);

    expect(calls).toEqual(['rev-parse HEAD']);
    expect(existsSync(worktreePath)).toBe(true);
  });

  it('preserves the callback error when cleanup also fails', async () => {
    const runId = 'git-test-cleanup-secondary-error';
    const callbackError = new Error('boom');
    const worktreePath = getForecastLabCandidateWorktreePath(runId);
    let thrown: unknown;

    try {
      await withForecastLabCandidateWorkspace(
        runId,
        () => {
          throw callbackError;
        },
        {
          gitCommandRunner: (args) => {
            if (args[0] === 'rev-parse') {
              return {
                exitCode: 0,
                stdout: '0123456789abcdef0123456789abcdef01234567\n',
                stderr: '',
              };
            }

            if (args[0] === 'worktree' && args[1] === 'add') {
              mkdirSync(worktreePath, { recursive: true });
              return {
                exitCode: 0,
                stdout: '',
                stderr: '',
              };
            }

            if (args[0] === 'worktree' && args[1] === 'remove') {
              return {
                exitCode: 1,
                stdout: '',
                stderr: 'stuck worktree',
              };
            }

            if (args[0] === 'show-ref') {
              return {
                exitCode: 1,
                stdout: '',
                stderr: '',
              };
            }

            return {
              exitCode: 0,
              stdout: '',
              stderr: '',
            };
          },
        },
      );
    } catch (error) {
      thrown = error;
    }

    expect(thrown).toBe(callbackError);
    expect(thrown).toBeInstanceOf(Error);
    expect((thrown as Error).message).toBe('boom');
    expect((thrown as Error).stack).toContain('Suppressed forecast-lab workspace cleanup failure');
    expect((thrown as Error).stack).toContain('worktree remove failed: stuck worktree');
    expect(existsSync(worktreePath)).toBe(true);
  });

  it('applies edits only inside the allowed mutation set', async () => {
    const runId = 'git-test-edit-policy';
    const profile = getForecastLabProfile('multi-asset-markov-short-horizon');

    await withForecastLabCandidateWorkspace(runId, (workspace) => {
      const allowedPath = profile.mutation.mutableFiles.find((filePath) => (
        existsSync(join(workspace.metadata.rootDir, filePath))
      ))!;
      const allowedFile = join(workspace.metadata.rootDir, allowedPath);
      const original = readFileSync(allowedFile, 'utf8');
      const updated = `${original}\n// forecast-lab git policy test\n`;

      expect(
        applyForecastLabCandidateEdits(
          workspace.metadata.rootDir,
          [{ path: allowedPath, contents: updated }],
          {
            allowedPaths: profile.mutation.mutableFiles,
            readOnlyPaths: profile.readOnlyHarnessFiles,
          },
        ),
      ).toEqual([allowedPath]);
      expect(readFileSync(allowedFile, 'utf8')).toBe(updated);

      const blockedDocsPath = 'docs/__forecast_lab_blocked__/guide.md';
      for (const path of [
        blockedDocsPath,
        'src/tools/finance/backtest/walk-forward.ts',
        'src/tools/finance/markov-distribution.test.ts',
        'src/tools/finance/polymarket.ts',
      ]) {
        expect(() =>
          applyForecastLabCandidateEdits(
            workspace.metadata.rootDir,
            [{ path, contents: 'blocked\n' }],
            {
              allowedPaths: profile.mutation.mutableFiles,
              readOnlyPaths: profile.readOnlyHarnessFiles,
            },
          ),
        ).toThrow(ForecastLabGitError);
      }

      expect(existsSync(join(workspace.metadata.rootDir, blockedDocsPath))).toBe(false);
      expect(readFileSync(allowedFile, 'utf8')).toBe(updated);
    }, {
      materializePaths: profile.mutation.mutableFiles,
    });
  });
});
