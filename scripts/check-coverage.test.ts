import { afterEach, describe, expect, it } from 'bun:test';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const SCRIPT = join(import.meta.dir, 'check-coverage.ts');
const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

/** Run the gate on a synthetic lcov file with `hit` of 100 lines covered. */
function runGate(hit: number, minimum: string | undefined): { status: number; stderr: string; stdout: string } {
  const root = mkdtempSync(join(tmpdir(), 'check-coverage-'));
  roots.push(root);
  const lcov = join(root, 'lcov.info');
  writeFileSync(lcov, `SF:sample.ts\nLF:100\nLH:${hit}\nend_of_record\n`);

  const env: Record<string, string | undefined> = { ...process.env };
  if (minimum === undefined) delete env.COVERAGE_LINES_MIN;
  else env.COVERAGE_LINES_MIN = minimum;

  const result = Bun.spawnSync([process.execPath, 'run', SCRIPT, lcov], { cwd: root, env });
  return { status: result.exitCode, stderr: result.stderr.toString(), stdout: result.stdout.toString() };
}

describe('check-coverage gate', () => {
  it('defaults the minimum to 22% when COVERAGE_LINES_MIN is unset', () => {
    expect(runGate(30, undefined).status).toBe(0);

    const failing = runGate(10, undefined);
    expect(failing.status).not.toBe(0);
    expect(failing.stderr).toContain('below required 22.00%');
  });

  it('honours a numeric COVERAGE_LINES_MIN', () => {
    expect(runGate(40, '30').status).toBe(0);
    expect(runGate(40, '50').status).not.toBe(0);
  });

  it.each(['not-a-number', 'NaN', '', '   '])(
    'rejects COVERAGE_LINES_MIN=%j instead of silently disabling the gate',
    (value) => {
      // 0% coverage: a disabled gate would pass it.
      const result = runGate(0, value);

      expect(result.status).not.toBe(0);
      expect(result.stderr).toContain('COVERAGE_LINES_MIN must be a number');
    },
  );
});
