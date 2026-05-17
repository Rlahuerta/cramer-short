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

const materializedFileContentsCache = new Map<string, string>();

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

function runGitCommandStdoutOrThrow(
  gitCommandRunner: ForecastLabGitCommandRunner,
  args: readonly string[],
  options?: { readonly cwd?: string; readonly errorPrefix?: string },
): string {
  const result = gitCommandRunner(args, { cwd: options?.cwd });

  if (result.exitCode !== 0) {
    const detail = result.stderr.trim() || result.stdout.trim() || `exit ${result.exitCode}`;
    throw new ForecastLabGitError(`${options?.errorPrefix ?? `git ${args.join(' ')}`} failed: ${detail}`);
  }

  return result.stdout;
}

function readForecastLabMaterializedFilesBatch(
  repoRoot: string,
  baselineCommit: string,
  paths: readonly string[],
): Map<string, string> {
  const result = spawnSync('git', ['cat-file', '--batch'], {
    cwd: repoRoot,
    input: `${paths.map((path) => `${baselineCommit}:${path}`).join('\n')}\n`,
    stdio: ['pipe', 'pipe', 'pipe'],
  });

  if (result.error) {
    throw new ForecastLabGitError(result.error.message);
  }
  if (result.status !== 0) {
    const detail = result.stderr.toString('utf8').trim() || result.stdout.toString('utf8').trim() || `exit ${result.status ?? 1}`;
    throw new ForecastLabGitError(`forecast-lab workspace materialize batch failed: ${detail}`);
  }

  const contentsByPath = new Map<string, string>();
  const stdout = result.stdout;
  let offset = 0;

  for (const path of paths) {
    const headerEnd = stdout.indexOf(0x0a, offset);
    if (headerEnd === -1) {
      throw new ForecastLabGitError(`forecast-lab workspace materialize (${path}) failed: truncated git cat-file header`);
    }

    const header = stdout.subarray(offset, headerEnd).toString('utf8');
    offset = headerEnd + 1;
    if (header.endsWith(' missing')) {
      throw new ForecastLabGitError(`forecast-lab workspace materialize (${path}) failed: missing blob at ${baselineCommit}`);
    }

    const parts = header.split(' ');
    const size = Number(parts[2]);
    if (parts.length !== 3 || parts[1] !== 'blob' || !Number.isInteger(size) || size < 0) {
      throw new ForecastLabGitError(`forecast-lab workspace materialize (${path}) failed: unexpected git cat-file header "${header}"`);
    }
    if (offset + size > stdout.length) {
      throw new ForecastLabGitError(`forecast-lab workspace materialize (${path}) failed: truncated git cat-file body`);
    }

    contentsByPath.set(path, stdout.subarray(offset, offset + size).toString('utf8'));
    offset += size;
    if (stdout[offset] === 0x0a) {
      offset += 1;
    }
  }

  return contentsByPath;
}

function materializeForecastLabWorkspacePathsFromCommit(
  repoRoot: string,
  workspaceRootDir: string,
  baselineCommit: string,
  gitCommandRunner: ForecastLabGitCommandRunner,
  paths: readonly string[],
): void {
  const normalizedPaths = paths.map((path) => normalizeRepoRelativePath(path));
  const missingCachePaths = normalizedPaths.filter((path) => !materializedFileContentsCache.has(`${baselineCommit}:${path}`));

  if (missingCachePaths.length > 1 && gitCommandRunner === defaultForecastLabGitCommandRunner) {
    const materialized = readForecastLabMaterializedFilesBatch(repoRoot, baselineCommit, missingCachePaths);
    for (const [path, contents] of materialized.entries()) {
      materializedFileContentsCache.set(`${baselineCommit}:${path}`, contents);
    }
  }

  for (const normalizedPath of normalizedPaths) {
    const targetPath = resolve(workspaceRootDir, normalizedPath);
    assertInsideRoot(workspaceRootDir, targetPath);
    mkdirSync(dirname(targetPath), { recursive: true });
    const cacheKey = `${baselineCommit}:${normalizedPath}`;
    let contents = materializedFileContentsCache.get(cacheKey);
    if (contents === undefined) {
      contents = runGitCommandStdoutOrThrow(gitCommandRunner, ['show', `${baselineCommit}:${normalizedPath}`], {
        cwd: repoRoot,
        errorPrefix: `forecast-lab workspace materialize (${normalizedPath})`,
      });
      materializedFileContentsCache.set(cacheKey, contents);
    }
    writeFileSync(targetPath, contents, 'utf8');
  }
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
    readonly materializePaths?: readonly string[];
    readonly lightweight?: boolean;
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
  if (options?.lightweight) {
    try {
      mkdirSync(metadata.rootDir, { recursive: true });
      if (options.materializePaths) {
        materializeForecastLabWorkspacePathsFromCommit(
          repoRoot,
          metadata.rootDir,
          baselineCommit,
          gitCommandRunner,
          options.materializePaths,
        );
      }
    } catch (error) {
      rmSync(metadata.rootDir, { recursive: true, force: true });
      throw error;
    }

    return {
      baselineCommit,
      metadata,
      cleanup() {
        rmSync(metadata.rootDir, { recursive: true, force: true });
      },
    };
  }

  if (gitBranchExists(metadata.branch, gitCommandRunner, repoRoot)) {
    throw new ForecastLabGitError(`Refusing to reuse existing forecast-lab branch before git unregisters it: ${metadata.branch}`);
  }

  mkdirSync(dirname(metadata.rootDir), { recursive: true });
  try {
    runGitCommandOrThrow(
      gitCommandRunner,
      [
        'worktree',
        'add',
        ...(options?.materializePaths ? ['--no-checkout'] : []),
        '-b',
        metadata.branch,
        metadata.rootDir,
        baselineCommit,
      ],
      {
        cwd: repoRoot,
        errorPrefix: `forecast-lab worktree create (${metadata.branch})`,
      },
    );
    if (options?.materializePaths) {
      materializeForecastLabWorkspacePathsFromCommit(
        repoRoot,
        metadata.rootDir,
        baselineCommit,
        gitCommandRunner,
        options.materializePaths,
      );
    }
  } catch (error) {
    const cleanupFailures: string[] = [];
    const originalMessage = error instanceof Error ? error.message : String(error);

    const worktreeRemoval = gitCommandRunner(['worktree', 'remove', '--force', metadata.rootDir], { cwd: repoRoot });
    if (worktreeRemoval.exitCode !== 0 && existsSync(metadata.rootDir)) {
      const detail = worktreeRemoval.stderr.trim() || worktreeRemoval.stdout.trim() || `exit ${worktreeRemoval.exitCode}`;
      cleanupFailures.push(`worktree remove failed: ${detail}`);
      try {
        rmSync(metadata.rootDir, { recursive: true, force: true });
      } catch (cleanupError) {
        const detail = cleanupError instanceof Error
          ? cleanupError.message
          : String(cleanupError);
        cleanupFailures.push(`worktree directory removal failed: ${detail}`);
      }
    } else if (existsSync(metadata.rootDir)) {
      try {
        rmSync(metadata.rootDir, { recursive: true, force: true });
      } catch (cleanupError) {
        const detail = cleanupError instanceof Error
          ? cleanupError.message
          : String(cleanupError);
        cleanupFailures.push(`worktree directory removal failed: ${detail}`);
      }
    }

    if (gitBranchExists(metadata.branch, gitCommandRunner, repoRoot)) {
      const branchDeletion = gitCommandRunner(['branch', '-D', metadata.branch], { cwd: repoRoot });
      if (branchDeletion.exitCode !== 0) {
        const detail = branchDeletion.stderr.trim() || branchDeletion.stdout.trim() || `exit ${branchDeletion.exitCode}`;
        cleanupFailures.push(`branch delete failed: ${detail}`);
      }
    }

    if (cleanupFailures.length > 0) {
      throw new ForecastLabGitError(
        `${originalMessage}; forecast-lab workspace create cleanup failed: ${cleanupFailures.join('; ')}`,
      );
    }

    if (error instanceof Error) {
      throw error;
    }
    throw new ForecastLabGitError(originalMessage);
  }

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
    readonly materializePaths?: readonly string[];
    readonly lightweight?: boolean;
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
    readonly materializePaths?: readonly string[];
    readonly lightweight?: boolean;
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
    readonly materializePaths?: readonly string[];
    readonly prepareWorkspace?: (
      runId: string,
      options?: {
        readonly branchName?: string;
        readonly repoRoot?: string;
        readonly gitCommandRunner?: ForecastLabGitCommandRunner;
        readonly materializePaths?: readonly string[];
        readonly lightweight?: boolean;
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
    readonly materializePaths?: readonly string[];
    readonly lightweight?: boolean;
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
    readonly materializePaths?: readonly string[];
    readonly lightweight?: boolean;
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
