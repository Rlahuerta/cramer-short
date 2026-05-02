import { mkdirSync } from 'node:fs';
import { join } from 'node:path';

const CRAMER_SHORT_DIR = '.cramer-short';
const EXPERIMENTS_DIR = 'experiments';
const EXPERIMENT_RUNS_DIR = 'runs';
const FORECAST_RESULTS_LEDGER = 'forecast-results.tsv';
const SAFE_RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*(\.[A-Za-z0-9_-]+)*$/;

type PathOptions = {
  create?: boolean;
};

export function getCramerShortDir(): string {
  return CRAMER_SHORT_DIR;
}

export function cramerShortPath(...segments: string[]): string {
  return join(getCramerShortDir(), ...segments);
}

function maybeCreateDir(path: string, options?: PathOptions): string {
  if (options?.create) {
    mkdirSync(path, { recursive: true });
  }

  return path;
}

function assertSafeExperimentRunId(runId: string): void {
  if (!SAFE_RUN_ID_PATTERN.test(runId)) {
    throw new Error('experiment runId must be a non-empty safe path segment');
  }
}

export function experimentsPath(...segments: string[]): string {
  return cramerShortPath(EXPERIMENTS_DIR, ...segments);
}

export function getExperimentsDir(options?: PathOptions): string {
  return maybeCreateDir(experimentsPath(), options);
}

export function getExperimentLedgerPath(options?: PathOptions): string {
  if (options?.create) {
    getExperimentsDir({ create: true });
  }

  return experimentsPath(FORECAST_RESULTS_LEDGER);
}

export function getExperimentRunsDir(options?: PathOptions): string {
  return maybeCreateDir(experimentsPath(EXPERIMENT_RUNS_DIR), options);
}

export function getExperimentRunDir(runId: string, options?: PathOptions): string {
  assertSafeExperimentRunId(runId);
  return maybeCreateDir(experimentsPath(EXPERIMENT_RUNS_DIR, runId), options);
}

export function getExperimentRunArtifactsDir(runId: string, options?: PathOptions): string {
  assertSafeExperimentRunId(runId);
  return maybeCreateDir(experimentsPath(EXPERIMENT_RUNS_DIR, runId, 'artifacts'), options);
}

export function getExperimentRunManifestPath(runId: string, options?: PathOptions): string {
  assertSafeExperimentRunId(runId);

  if (options?.create) {
    getExperimentRunDir(runId, { create: true });
  }

  return experimentsPath(EXPERIMENT_RUNS_DIR, runId, 'manifest.json');
}
