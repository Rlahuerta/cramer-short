import { existsSync, mkdirSync, rmSync, writeFileSync } from 'node:fs';
import { dirname, relative, resolve, sep } from 'node:path';
import { spawnSync } from 'node:child_process';
import type { ForecastLabCandidateWorkspaceMetadata } from './mutation.js';
import { getExperimentsDir } from '../../utils/paths.js';

export class ForecastLabGitError extends Error {
  override name = 'ForecastLabGitError';
}

const SAFE_BRANCH_SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const SAFE_BASELINE_COMMIT = /^[0-9a-f]{40}$/i;

export interface ForecastLabCandidateEdit {
  readonly path: string;
  readonly contents: string;
}

export interface ForecastLabCandidateEditPolicy {
  readonly allowedPaths: readonly string[];
  readonly readOnlyPaths?: readonly string[];
}

export interface ForecastLabPreparedCandidateWorkspace {
  readonly baselineCommit: string;
  readonly metadata: ForecastLabCandidateWorkspaceMetadata;
  cleanup(): void;
}

export interface ForecastLabGitCommandResult {
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
}

export type ForecastLabGitCommandRunner = (
  args: readonly string[],
  options?: { readonly cwd?: string },
) => ForecastLabGitCommandResult;

type ForecastLabErrorWithCleanupContext = Error & {
  suppressedCleanupError?: unknown;
};

function defaultForecastLabGitCommandRunner(
  args: readonly string[],
  options?: { readonly cwd?: string },
): ForecastLabGitCommandResult {
  const result = spawnSync('git', [...args], {
    cwd: options?.cwd ?? process.cwd(),
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });

  if (result.error) {
    throw new ForecastLabGitError(result.error.message);
  }

  return {
    exitCode: result.status ?? 1,
    stdout: result.stdout ?? '',
    stderr: result.stderr ?? '',
  };
}

function normalizeRepoRelativePath(path: string): string {
  const normalized = path.replaceAll('\\', '/').trim();

  if (normalized === '') {
    throw new ForecastLabGitError('Forecast-lab candidate edit path must not be empty');
  }

  if (normalized.startsWith('/') || normalized.startsWith('../') || normalized === '..') {
    throw new ForecastLabGitError(`Forecast-lab candidate edit path must stay inside the repo: ${path}`);
  }

  const segments = normalized.split('/');
  if (segments.some((segment) => segment === '' || segment === '.' || segment === '..')) {
    throw new ForecastLabGitError(`Forecast-lab candidate edit path must be a clean repo-relative path: ${path}`);
  }

  return normalized;
}

function assertInsideRoot(rootDir: string, targetPath: string): void {
  const resolvedRoot = resolve(rootDir);
  const resolvedTarget = resolve(targetPath);

  if (resolvedTarget !== resolvedRoot && !resolvedTarget.startsWith(resolvedRoot + sep)) {
    throw new ForecastLabGitError(`Refusing to access path outside candidate workspace: ${targetPath}`);
  }
}

function runGitCommandOrThrow(
  gitCommandRunner: ForecastLabGitCommandRunner,
  args: readonly string[],
  options?: { readonly cwd?: string; readonly errorPrefix?: string },
): string {
  const result = gitCommandRunner(args, { cwd: options?.cwd });

  if (result.exitCode !== 0) {
    const detail = result.stderr.trim() || result.stdout.trim() || `exit ${result.exitCode}`;
    throw new ForecastLabGitError(`${options?.errorPrefix ?? `git ${args.join(' ')}`} failed: ${detail}`);
  }

  return result.stdout.trim();
}

function gitBranchExists(
  branchName: string,
  gitCommandRunner: ForecastLabGitCommandRunner,
  repoRoot: string,
): boolean {
  const result = gitCommandRunner(['show-ref', '--verify', '--quiet', `refs/heads/${branchName}`], { cwd: repoRoot });
  return result.exitCode === 0;
}

export function assertSafeGitBranchName(branchName: string): void {
  const segments = branchName.split('/');

  if (
    branchName.trim() === '' ||
    branchName.includes('..') ||
    branchName.startsWith('/') ||
    branchName.endsWith('/') ||
    segments.some((segment) => !SAFE_BRANCH_SEGMENT.test(segment))
  ) {
    throw new ForecastLabGitError(`Unsafe git branch name: ${branchName}`);
  }
}

function makeForecastLabWorkspaceBranch(runId: string, prefix: string): string {
  const branchName = `topic/${prefix}-${runId}`;
  assertSafeGitBranchName(branchName);
  return branchName;
}

export function makeForecastLabCandidateBranch(runId: string): string {
  return makeForecastLabWorkspaceBranch(runId, 'forecast-lab');
}

export function makeForecastLabPromotionBranch(runId: string): string {
  return makeForecastLabWorkspaceBranch(runId, 'forecast-lab-promote');
}

export function assertSafeForecastLabWorktreePath(worktreePath: string, repoRoot = process.cwd()): void {
  const worktreesRoot = resolve(repoRoot, getExperimentsDir(), 'worktrees');
  const resolvedWorktreePath = resolve(worktreePath);
  const relativePath = relative(worktreesRoot, resolvedWorktreePath);

  if (
    relativePath === '' ||
    relativePath === '.' ||
    relativePath.startsWith('..') ||
    relativePath.includes(`${sep}..${sep}`) ||
    relativePath.endsWith(`${sep}..`)
  ) {
    throw new ForecastLabGitError(`Unsafe forecast-lab worktree path: ${worktreePath}`);
  }
}

function getForecastLabWorkspacePath(worktreeDirName: string, repoRoot = process.cwd()): string {
  const worktreePath = resolve(repoRoot, getExperimentsDir(), 'worktrees', worktreeDirName);
  assertSafeForecastLabWorktreePath(worktreePath, repoRoot);
  return worktreePath;
}

export function getForecastLabCandidateWorktreePath(runId: string, repoRoot = process.cwd()): string {
  return getForecastLabWorkspacePath(runId, repoRoot);
}

export function getForecastLabPromotionWorktreePath(runId: string, repoRoot = process.cwd()): string {
  return getForecastLabWorkspacePath(`promote-${runId}`, repoRoot);
}

function describeForecastLabWorkspace(
  runId: string,
  options?: {
    readonly branchName?: string;
    readonly repoRoot?: string;
    readonly branchFactory?: (runId: string) => string;
    readonly worktreePathFactory?: (runId: string, repoRoot?: string) => string;
  },
): ForecastLabCandidateWorkspaceMetadata {
  const branch = options?.branchName ?? (options?.branchFactory ?? makeForecastLabCandidateBranch)(runId);
  return {
    kind: 'candidate-worktree',
    rootDir: (options?.worktreePathFactory ?? getForecastLabCandidateWorktreePath)(runId, options?.repoRoot),
    branch,
  };
}

export function describeForecastLabCandidateWorkspace(
  runId: string,
  options?: {
    readonly branchName?: string;
    readonly repoRoot?: string;
  },
): ForecastLabCandidateWorkspaceMetadata {
  return describeForecastLabWorkspace(runId, options);
}

export function describeForecastLabPromotionWorkspace(
  runId: string,
  options?: {
    readonly branchName?: string;
    readonly repoRoot?: string;
  },
): ForecastLabCandidateWorkspaceMetadata {
  return describeForecastLabWorkspace(runId, {
    ...options,
    branchFactory: makeForecastLabPromotionBranch,
    worktreePathFactory: getForecastLabPromotionWorktreePath,
  });
}

export function getForecastLabBaselineCommit(
  options?: {
    readonly repoRoot?: string;
    readonly gitCommandRunner?: ForecastLabGitCommandRunner;
  },
): string {
  const baselineCommit = runGitCommandOrThrow(
    options?.gitCommandRunner ?? defaultForecastLabGitCommandRunner,
    ['rev-parse', 'HEAD'],
    {
      cwd: options?.repoRoot ?? process.cwd(),
      errorPrefix: 'forecast-lab baseline commit lookup',
    },
  );

  if (!SAFE_BASELINE_COMMIT.test(baselineCommit)) {
    throw new ForecastLabGitError(`Unexpected forecast-lab baseline commit: ${baselineCommit}`);
  }

  return baselineCommit;
}

function prepareForecastLabWorkspace(
  runId: string,
  options?: {
    readonly branchName?: string;
    readonly repoRoot?: string;
    readonly gitCommandRunner?: ForecastLabGitCommandRunner;
    readonly describeWorkspace?: (
      runId: string,
      options?: {
        readonly branchName?: string;
        readonly repoRoot?: string;
      },
    ) => ForecastLabCandidateWorkspaceMetadata;
  },
): ForecastLabPreparedCandidateWorkspace {
  const repoRoot = options?.repoRoot ?? process.cwd();
  const gitCommandRunner = options?.gitCommandRunner ?? defaultForecastLabGitCommandRunner;
  const metadata = (options?.describeWorkspace ?? describeForecastLabCandidateWorkspace)(runId, {
    branchName: options?.branchName,
    repoRoot,
  });
  const baselineCommit = getForecastLabBaselineCommit({ repoRoot, gitCommandRunner });

  if (existsSync(metadata.rootDir)) {
    throw new ForecastLabGitError(
      `Refusing to reuse existing forecast-lab worktree path before git unregisters it: ${metadata.rootDir}`,
    );
  }

  mkdirSync(dirname(metadata.rootDir), { recursive: true });
  runGitCommandOrThrow(gitCommandRunner, ['worktree', 'add', '-b', metadata.branch, metadata.rootDir, baselineCommit], {
    cwd: repoRoot,
    errorPrefix: `forecast-lab worktree create (${metadata.branch})`,
  });

  return {
    baselineCommit,
    metadata,
    cleanup() {
      const failures: string[] = [];

      const worktreeRemoval = gitCommandRunner(['worktree', 'remove', '--force', metadata.rootDir], { cwd: repoRoot });
      if (worktreeRemoval.exitCode !== 0 && existsSync(metadata.rootDir)) {
        const detail = worktreeRemoval.stderr.trim() || worktreeRemoval.stdout.trim() || `exit ${worktreeRemoval.exitCode}`;
        failures.push(`worktree remove failed: ${detail}`);
      } else {
        rmSync(metadata.rootDir, { recursive: true, force: true });
      }

      if (gitBranchExists(metadata.branch, gitCommandRunner, repoRoot)) {
        const branchDeletion = gitCommandRunner(['branch', '-D', metadata.branch], { cwd: repoRoot });
        if (branchDeletion.exitCode !== 0) {
          const detail = branchDeletion.stderr.trim() || branchDeletion.stdout.trim() || `exit ${branchDeletion.exitCode}`;
          failures.push(`branch delete failed: ${detail}`);
        }
      }

      if (failures.length > 0) {
        throw new ForecastLabGitError(`Failed to clean forecast-lab candidate workspace: ${failures.join('; ')}`);
      }
    },
  };
}

export function prepareForecastLabCandidateWorkspace(
  runId: string,
  options?: {
    readonly branchName?: string;
    readonly repoRoot?: string;
    readonly gitCommandRunner?: ForecastLabGitCommandRunner;
  },
): ForecastLabPreparedCandidateWorkspace {
  return prepareForecastLabWorkspace(runId, options);
}

export function prepareForecastLabPromotionWorkspace(
  runId: string,
  options?: {
    readonly branchName?: string;
    readonly repoRoot?: string;
    readonly gitCommandRunner?: ForecastLabGitCommandRunner;
  },
): ForecastLabPreparedCandidateWorkspace {
  return prepareForecastLabWorkspace(runId, {
    ...options,
    describeWorkspace: describeForecastLabPromotionWorkspace,
  });
}

async function withForecastLabWorkspace<T>(
  runId: string,
  callback: (workspace: ForecastLabPreparedCandidateWorkspace) => Promise<T> | T,
  options?: {
    readonly branchName?: string;
    readonly repoRoot?: string;
    readonly gitCommandRunner?: ForecastLabGitCommandRunner;
    readonly prepareWorkspace?: (
      runId: string,
      options?: {
        readonly branchName?: string;
        readonly repoRoot?: string;
        readonly gitCommandRunner?: ForecastLabGitCommandRunner;
      },
    ) => ForecastLabPreparedCandidateWorkspace;
  },
): Promise<T> {
  const workspace = (options?.prepareWorkspace ?? prepareForecastLabCandidateWorkspace)(runId, options);
  let callbackError: unknown;
  let result: T | undefined;

  try {
    result = await callback(workspace);
  } catch (error) {
    callbackError = error;
  }

  try {
    workspace.cleanup();
  } catch (cleanupError) {
    if (callbackError !== undefined) {
      if (callbackError instanceof Error) {
        const errorWithCleanupContext = callbackError as ForecastLabErrorWithCleanupContext;
        errorWithCleanupContext.suppressedCleanupError = cleanupError;
        const cleanupDetail = cleanupError instanceof Error
          ? (cleanupError.stack ?? `${cleanupError.name}: ${cleanupError.message}`)
          : String(cleanupError);
        const baseStack = callbackError.stack ?? `${callbackError.name}: ${callbackError.message}`;
        callbackError.stack = `${baseStack}\nSuppressed forecast-lab workspace cleanup failure: ${cleanupDetail}`;
      }

      throw callbackError;
    }

    throw cleanupError;
  }

  if (callbackError !== undefined) {
    throw callbackError;
  }

  return result as T;
}

export async function withForecastLabCandidateWorkspace<T>(
  runId: string,
  callback: (workspace: ForecastLabPreparedCandidateWorkspace) => Promise<T> | T,
  options?: {
    readonly branchName?: string;
    readonly repoRoot?: string;
    readonly gitCommandRunner?: ForecastLabGitCommandRunner;
  },
): Promise<T> {
  return withForecastLabWorkspace(runId, callback, options);
}

export async function withForecastLabPromotionWorkspace<T>(
  runId: string,
  callback: (workspace: ForecastLabPreparedCandidateWorkspace) => Promise<T> | T,
  options?: {
    readonly branchName?: string;
    readonly repoRoot?: string;
    readonly gitCommandRunner?: ForecastLabGitCommandRunner;
  },
): Promise<T> {
  return withForecastLabWorkspace(runId, callback, {
    ...options,
    prepareWorkspace: prepareForecastLabPromotionWorkspace,
  });
}

export function isPathAllowedByForecastLab(path: string, allowedGlobs: readonly string[]): boolean {
  const normalizedPath = normalizeRepoRelativePath(path);
  return allowedGlobs.some((allowedPath) => normalizeRepoRelativePath(allowedPath) === normalizedPath);
}

export function assertForecastLabCandidateEditPolicy(
  paths: readonly string[],
  policy: ForecastLabCandidateEditPolicy,
): void {
  const readOnlyPaths = new Set((policy.readOnlyPaths ?? []).map((path) => normalizeRepoRelativePath(path)));

  for (const path of paths) {
    const normalizedPath = normalizeRepoRelativePath(path);

    if (normalizedPath.startsWith('docs/') || normalizedPath.endsWith('.test.ts') || normalizedPath.includes('/backtest/')) {
      throw new ForecastLabGitError(`Forecast-lab candidate edits must not touch harness/test/docs files: ${normalizedPath}`);
    }

    if (readOnlyPaths.has(normalizedPath)) {
      throw new ForecastLabGitError(`Forecast-lab candidate edits must not touch read-only harness files: ${normalizedPath}`);
    }

    if (!isPathAllowedByForecastLab(normalizedPath, policy.allowedPaths)) {
      throw new ForecastLabGitError(`Forecast-lab candidate edit is outside the allowed mutation set: ${normalizedPath}`);
    }
  }
}

export function applyForecastLabCandidateEdits(
  workspaceRootDir: string,
  edits: readonly ForecastLabCandidateEdit[],
  policy: ForecastLabCandidateEditPolicy,
): readonly string[] {
  const normalizedPaths = edits.map((edit) => normalizeRepoRelativePath(edit.path));
  assertForecastLabCandidateEditPolicy(normalizedPaths, policy);

  for (let index = 0; index < edits.length; index += 1) {
    const edit = edits[index]!;
    const normalizedPath = normalizedPaths[index]!;
    const targetPath = resolve(workspaceRootDir, normalizedPath);

    assertInsideRoot(workspaceRootDir, targetPath);
    mkdirSync(dirname(targetPath), { recursive: true });
    writeFileSync(targetPath, edit.contents, 'utf8');
  }

  return normalizedPaths;
}
