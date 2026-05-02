export class ForecastLabGitError extends Error {
  override name = 'ForecastLabGitError';
}

const SAFE_BRANCH_SEGMENT = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;

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

export function makeForecastLabCandidateBranch(runId: string): string {
  const branchName = `topic/forecast-lab-${runId}`;
  assertSafeGitBranchName(branchName);
  return branchName;
}

export function isPathAllowedByForecastLab(path: string, allowedGlobs: readonly string[]): boolean {
  return allowedGlobs.includes(path);
}

