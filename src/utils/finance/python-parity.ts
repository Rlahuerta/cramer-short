import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';
import { getEnvironment } from '../env.js';

const CONDA_ENV_NAME = 'cramer-research';

const CONDA_INSTALL_CANDIDATES = [
  join(homedir(), 'anaconda3'),
  join(homedir(), 'miniconda3'),
  '/opt/anaconda3',
  '/opt/miniconda3',
];

function condaPythonCandidates(): string[] {
  return CONDA_INSTALL_CANDIDATES.map((root) => join(root, 'envs', CONDA_ENV_NAME, 'bin', 'python'));
}

/**
 * Test-only gate for *.parity.test.ts suites. False (with a warning) when the
 * conda env is missing, so parity suites skip on machines/CI without it;
 * REQUIRE_PARITY=1 forces true so a missing env fails loudly instead.
 */
export function pythonParityAvailable(): boolean {
  if (getEnvironment().REQUIRE_PARITY === '1') return true;
  if (condaPythonCandidates().some((python) => existsSync(python))) return true;
  console.warn(
    `[parity] Skipping TS/Python parity tests: conda env "${CONDA_ENV_NAME}" not found ` +
    `(see environment-research.yml). Set REQUIRE_PARITY=1 to fail instead.`,
  );
  return false;
}

function findCondaPython(): string {
  for (const python of condaPythonCandidates()) {
    if (existsSync(python)) return python;
  }
  throw new Error(
    `Could not find conda environment "${CONDA_ENV_NAME}". ` +
    `Searched: ${condaPythonCandidates().join(', ')}`,
  );
}

function findRepoRoot(): string {
  // Walk up from this source file to find the git root
  let dir = dirname(fileURLToPath(import.meta.url));
  for (let i = 0; i < 10; i++) {
    if (existsSync(join(dir, 'package.json')) && existsSync(join(dir, '.git'))) {
      return dir;
    }
    const parent = join(dir, '..');
    if (parent === dir) break;
    dir = parent;
  }
  throw new Error('Could not find repo root');
}

/**
 * Run a Python one-liner script and return its stdout trimmed.
 * Uses the cramer-research conda environment.
 *
 * **INTERNAL USE ONLY**: This function executes arbitrary Python code.
 * Do NOT expose to untrusted callers or external input.
 * Current usage is limited to internal parity testing.
 *
 * Execution is gated behind {@link enablePythonParity}: production code that
 * never calls it cannot trigger Python execution even if it imports runPython.
 */
let pythonParityEnabled = false;

/**
 * Opt in to Python execution. Test-only: call this from parity-test setup
 * before invoking {@link runPython}. Never call from production code paths.
 */
export function enablePythonParity(): void {
  pythonParityEnabled = true;
}

export function runPython(script: string): Promise<string> {
  if (!pythonParityEnabled) {
    return Promise.reject(
      new Error('runPython is disabled outside parity tests; call enablePythonParity() first.'),
    );
  }
  const python = findCondaPython();
  const repoRoot = findRepoRoot();

  return new Promise((resolve, reject) => {
    const proc = Bun.spawn({
      cmd: [python, '-c', script],
      cwd: repoRoot,
      env: { ...getEnvironment(), PYTHONPATH: repoRoot },
      stdout: 'pipe',
      stderr: 'pipe',
    });
    const stdout: string[] = [];
    const stderr: string[] = [];
    proc.stdout.pipeTo(
      new WritableStream({
        write(chunk) {
          stdout.push(new TextDecoder().decode(chunk));
        },
      }),
    );
    proc.stderr.pipeTo(
      new WritableStream({
        write(chunk) {
          stderr.push(new TextDecoder().decode(chunk));
        },
      }),
    );
    proc.exited.then((code) => {
      if (code !== 0) reject(new Error(`Python exited ${code}: ${stderr.join('')}`));
      else resolve(stdout.join('').trim());
    });
  });
}
