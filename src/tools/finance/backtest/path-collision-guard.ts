import { existsSync, realpathSync } from 'node:fs';
import { dirname, relative, resolve } from 'node:path';

interface CanonicalPathEntry {
  label: string;
  rawPath: string;
  canonicalPath: string;
}

function toRealPath(path: string): string {
  return typeof realpathSync.native === 'function'
    ? realpathSync.native(path)
    : realpathSync(path);
}

function findNearestExistingPath(path: string): string {
  let current = path;
  while (!existsSync(current)) {
    const parent = dirname(current);
    if (parent === current) {
      return current;
    }
    current = parent;
  }
  return current;
}

export function canonicalizeCollisionGuardPath(path: string): string {
  const resolvedPath = resolve(path);
  const existingPath = findNearestExistingPath(resolvedPath);

  let canonicalExistingPath = existingPath;
  try {
    canonicalExistingPath = toRealPath(existingPath);
  } catch {
    canonicalExistingPath = existingPath;
  }

  const suffix = relative(existingPath, resolvedPath);
  return suffix === ''
    ? canonicalExistingPath
    : resolve(canonicalExistingPath, suffix);
}

function findCanonicalPathCollision(paths: Record<string, string>): [CanonicalPathEntry, CanonicalPathEntry] | null {
  const seen = new Map<string, CanonicalPathEntry>();

  for (const [label, rawPath] of Object.entries(paths)) {
    const entry: CanonicalPathEntry = {
      label,
      rawPath,
      canonicalPath: canonicalizeCollisionGuardPath(rawPath),
    };
    const prior = seen.get(entry.canonicalPath);
    if (prior) {
      return [prior, entry];
    }
    seen.set(entry.canonicalPath, entry);
  }

  return null;
}

export function assertNoCanonicalPathCollision(
  scope: string,
  message: string,
  paths: Record<string, string>,
): void {
  const collision = findCanonicalPathCollision(paths);
  if (!collision) {
    return;
  }

  const [first, second] = collision;
  throw new Error(
    `${scope}: ${message} Got: ${first.label}=${first.rawPath}, ${second.label}=${second.rawPath}`,
  );
}
