import { afterAll, spyOn } from 'bun:test';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import * as paths from './paths.js';

/**
 * Test-only: point getCramerShortDir() — and every path built from it — at a
 * fresh `<os tmpdir>/…/.cramer-short` for the calling test file, so tests never
 * read or write the working tree's real `.cramer-short/`. Restored and deleted
 * afterAll. Call at module top level, before importing modules that resolve
 * paths at load time.
 *
 * Prefer this over mock.module('…/paths.js'): Bun never undoes mock.module, so
 * such mocks leak into every test file that runs later in the same process.
 */
export function useTempCramerShortDir(): string {
  const root = mkdtempSync(join(tmpdir(), 'cramer-short-test-'));
  const dir = join(root, '.cramer-short');
  const spy = spyOn(paths, 'getCramerShortDir').mockReturnValue(dir);
  afterAll(() => {
    spy.mockRestore();
    rmSync(root, { recursive: true, force: true });
  });
  return dir;
}
