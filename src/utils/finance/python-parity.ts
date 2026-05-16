import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { homedir } from 'node:os';

const CONDA_ENV_NAME = 'cramer-research';

const CONDA_INSTALL_CANDIDATES = [
  join(homedir(), 'anaconda3'),
  join(homedir(), 'miniconda3'),
  '/opt/anaconda3',
  '/opt/miniconda3',
];

function findCondaPython(): string {
  for (const root of CONDA_INSTALL_CANDIDATES) {
    const python = join(root, 'envs', CONDA_ENV_NAME, 'bin', 'python');
    if (existsSync(python)) return python;
  }
  throw new Error(
    `Could not find conda environment "${CONDA_ENV_NAME}". ` +
    `Searched: ${CONDA_INSTALL_CANDIDATES.map(d => join(d, 'envs', CONDA_ENV_NAME, 'bin', 'python')).join(', ')}`,
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
 */
export function runPython(script: string): Promise<string> {
  const python = findCondaPython();
  const repoRoot = findRepoRoot();

  return new Promise((resolve, reject) => {
    const proc = Bun.spawn({
      cmd: [python, '-c', script],
      cwd: repoRoot,
      env: { ...process.env, PYTHONPATH: repoRoot },
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
