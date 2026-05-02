import { exec } from 'node:child_process';
import type { ExecException } from 'node:child_process';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname, resolve, sep } from 'node:path';
import {
  getExperimentLedgerPath,
  getExperimentRunDir,
  getExperimentRunManifestPath,
  getExperimentsDir,
} from '../../utils/paths.js';
import { appendLedgerEntry, stableJsonStringify, writeRunManifest } from './ledger.js';
import { getForecastLabProfile } from './profiles.js';
import type { ForecastLabCommand, ForecastLabProfile } from './profiles.js';
import type { ForecastLabDecision, ForecastLabLedgerEntry, ForecastLabRunManifest, JsonValue } from './types.js';
import { makeForecastLabCandidateBranch } from './git.js';

export type ForecastLabGatePhase = 'baseline' | 'candidate';

export interface ForecastLabCommandRunContext {
  readonly phase: ForecastLabGatePhase;
  readonly profile: ForecastLabProfile;
  readonly runId: string;
}

export interface ForecastLabCommandResult {
  readonly id: string;
  readonly command: string;
  readonly exitCode: number;
  readonly stdout: string;
  readonly stderr: string;
  readonly durationMs: number;
  readonly timedOut: boolean;
}

export type ForecastLabCommandRunner = (
  command: ForecastLabCommand,
  context: ForecastLabCommandRunContext,
) => Promise<ForecastLabCommandResult>;

interface ForecastLabGateSummary {
  readonly phase: ForecastLabGatePhase;
  readonly exitCode: number;
  readonly commands: readonly ForecastLabCommandResult[];
}

interface ForecastLabMetricEvaluation {
  readonly name: string;
  readonly baseline: number;
  readonly candidate: number;
  readonly delta: number;
}

interface ForecastLabDecisionSummary {
  readonly decision: ForecastLabDecision;
  readonly reason: string;
  readonly metrics: readonly ForecastLabMetricEvaluation[];
}

export interface ForecastLabRunOptions {
  readonly profileId: string;
  readonly dryRun?: boolean;
  readonly skipMutation?: boolean;
  readonly runId?: string;
  readonly now?: () => Date;
  readonly commandRunner?: ForecastLabCommandRunner;
  readonly ledgerPath?: string;
}

export interface ForecastLabRunResult {
  readonly runId: string;
  readonly manifest: ForecastLabRunManifest;
  readonly baseline: JsonValue;
  readonly candidate: JsonValue;
  readonly decision: ForecastLabDecisionSummary;
  readonly ledgerEntry: ForecastLabLedgerEntry;
}

export class ForecastLabRunnerError extends Error {
  override name = 'ForecastLabRunnerError';
}

const UNSAFE_SHELL_COMMAND_PATTERN = /[;&|`<>]|\$\(/;
const UNSAFE_GIT_COMMAND_PATTERN = /\bgit\s+(?:add|commit|push|reset|checkout|clean)\b/;

function makeRunId(profileId: string, now: Date): string {
  return `forecast-lab-${profileId}-${now.toISOString().replace(/[:.]/g, '-')}`;
}

function assertInsideExperiments(path: string): void {
  const experimentsRoot = resolve(getExperimentsDir());
  const resolved = resolve(path);

  if (resolved !== experimentsRoot && !resolved.startsWith(experimentsRoot + sep)) {
    throw new ForecastLabRunnerError(`Refusing to write outside .cramer-short/experiments: ${path}`);
  }
}

function writeJsonArtifact(path: string, value: unknown): void {
  assertInsideExperiments(path);
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${stableJsonStringify(value)}\n`, 'utf8');
}

function toCommandResult(
  command: ForecastLabCommand,
  startedAtMs: number,
  error: ExecException | null,
  stdout: string,
  stderr: string,
): ForecastLabCommandResult {
  return {
    id: command.id,
    command: command.command,
    exitCode: typeof error?.code === 'number' ? error.code : error ? 1 : 0,
    stdout,
    stderr,
    durationMs: Date.now() - startedAtMs,
    timedOut: Boolean(error?.killed),
  };
}

function assertSafeProfileCommand(command: ForecastLabCommand): void {
  if (UNSAFE_SHELL_COMMAND_PATTERN.test(command.command) || UNSAFE_GIT_COMMAND_PATTERN.test(command.command)) {
    throw new ForecastLabRunnerError(`Unsafe forecast-lab command "${command.id}" is not allowed`);
  }
}

export const defaultForecastLabCommandRunner: ForecastLabCommandRunner = async (command) => {
  assertSafeProfileCommand(command);
  const startedAtMs = Date.now();

  return await new Promise<ForecastLabCommandResult>((resolveResult) => {
    exec(
      command.command,
      {
        cwd: process.cwd(),
        env: { ...process.env, ...(command.env ?? {}) },
        maxBuffer: 10 * 1024 * 1024,
        timeout: command.timeoutMs,
      },
      (error, stdout, stderr) => {
        resolveResult(toCommandResult(command, startedAtMs, error, stdout, stderr));
      },
    );
  });
};

async function runGate(
  phase: ForecastLabGatePhase,
  profile: ForecastLabProfile,
  runId: string,
  commandRunner: ForecastLabCommandRunner,
): Promise<ForecastLabGateSummary> {
  const commands = phase === 'baseline' ? profile.baselineCommands : profile.candidateCommands;
  const results: ForecastLabCommandResult[] = [];

  for (const command of commands) {
    results.push(await commandRunner(command, { phase, profile, runId }));
  }

  return {
    phase,
    exitCode: results.some((result) => result.exitCode !== 0) ? 1 : 0,
    commands: results,
  };
}

function getPathNumber(root: unknown, path: string): number | undefined {
  let current: unknown = root;

  for (const segment of path.split('.')) {
    if (!current || typeof current !== 'object' || !(segment in current)) {
      return undefined;
    }

    current = (current as Record<string, unknown>)[segment];
  }

  return typeof current === 'number' && Number.isFinite(current) ? current : undefined;
}

function evaluateCriterion(
  criterion: ForecastLabProfile['keepDropRule']['keepWhen']['all'][number],
  metrics: readonly ForecastLabMetricEvaluation[],
): boolean {
  const metric = metrics.find((candidateMetric) => candidateMetric.name === criterion.metric);

  if (!metric) {
    return false;
  }

  switch (criterion.operator) {
    case 'candidate-delta-gte':
      return metric.delta >= criterion.value;
    case 'candidate-delta-lte':
      return metric.delta <= criterion.value;
    case 'candidate-value-gte':
      return metric.candidate >= criterion.value;
    case 'candidate-value-lte':
      return metric.candidate <= criterion.value;
  }
}

function decideRun(
  profile: ForecastLabProfile,
  baseline: ForecastLabGateSummary,
  candidate: ForecastLabGateSummary,
): ForecastLabDecisionSummary {
  const metricRoot = { baseline, candidate };
  const metrics: ForecastLabMetricEvaluation[] = [];

  for (const metric of profile.minimumMetrics) {
    const baselineValue = getPathNumber(metricRoot, metric.baselinePath);
    const candidateValue = getPathNumber(metricRoot, metric.candidatePath);

    if (baselineValue === undefined || candidateValue === undefined) {
      return {
        decision: 'drop',
        reason: `missing required metric: ${metric.name}`,
        metrics,
      };
    }

    metrics.push({
      name: metric.name,
      baseline: baselineValue,
      candidate: candidateValue,
      delta: candidateValue - baselineValue,
    });
  }

  const dropCriterion = profile.keepDropRule.dropWhen.any.find((criterion) => evaluateCriterion(criterion, metrics));
  if (dropCriterion) {
    return { decision: 'drop', reason: dropCriterion.reason, metrics };
  }

  const keepCriteria = profile.keepDropRule.keepWhen.all;
  if (keepCriteria.length > 0 && keepCriteria.every((criterion) => evaluateCriterion(criterion, metrics))) {
    return {
      decision: 'keep',
      reason: keepCriteria.map((criterion) => criterion.reason).join('; '),
      metrics,
    };
  }

  return {
    decision: profile.keepDropRule.defaultDecision,
    reason: profile.keepDropRule.defaultDecision === 'drop' ? 'default drop: no keep rule satisfied' : 'default keep',
    metrics,
  };
}

function dropSkippedMutation(decision: ForecastLabDecisionSummary): ForecastLabDecisionSummary {
  return {
    ...decision,
    decision: 'drop',
    reason: 'mutation skipped by --skip-mutation; no candidate code change to keep',
  };
}

function summarizeForLedger(summary: ForecastLabGateSummary): JsonValue {
  return {
    exitCode: summary.exitCode,
    commands: summary.commands.map((command) => ({
      id: command.id,
      exitCode: command.exitCode,
      durationMs: command.durationMs,
      timedOut: command.timedOut,
    })),
  };
}

export async function runForecastLab(options: ForecastLabRunOptions): Promise<ForecastLabRunResult> {
  const profile = getForecastLabProfile(options.profileId);
  const now = options.now ?? (() => new Date());
  const startedAt = now().toISOString();
  const runId = options.runId ?? makeRunId(profile.id, new Date(startedAt));
  const dryRun = options.dryRun === true;
  const skipMutation = options.skipMutation === true;

  if (!dryRun && !skipMutation) {
    throw new ForecastLabRunnerError(
      'Forecast-lab V1 does not support real mutation yet. Re-run with --dry-run or --skip-mutation.',
    );
  }

  const candidateBranch = makeForecastLabCandidateBranch(runId);
  const runDir = getExperimentRunDir(runId, { create: true });
  const manifestPath = getExperimentRunManifestPath(runId, { create: true });
  const baselinePath = `${runDir}/baseline.json`;
  const candidatePath = `${runDir}/candidate.json`;
  const decisionPath = `${runDir}/decision.json`;
  const ledgerPath = options.ledgerPath ?? getExperimentLedgerPath({ create: true });

  for (const path of [runDir, manifestPath, baselinePath, candidatePath, decisionPath, ledgerPath]) {
    assertInsideExperiments(path);
  }

  const manifest: ForecastLabRunManifest = {
    runId,
    startedAt,
    profileId: profile.id,
    targetSubsystem: profile.targetSubsystem,
    candidateBranch,
    allowedGlobs: [...profile.allowedGlobs],
    artifactsPath: runDir,
  };

  writeRunManifest(manifestPath, manifest);

  const commandRunner = options.commandRunner ?? defaultForecastLabCommandRunner;
  const baseline = await runGate('baseline', profile, runId, commandRunner);
  writeJsonArtifact(baselinePath, baseline);

  const candidate = await runGate('candidate', profile, runId, commandRunner);
  const mutationStatus = dryRun ? 'dry-run: no code mutation attempted' : 'skipped by --skip-mutation';
  writeJsonArtifact(candidatePath, {
    ...candidate,
    mutation: mutationStatus,
  });

  const measuredDecision = decideRun(profile, baseline, candidate);
  const decision = skipMutation ? dropSkippedMutation(measuredDecision) : measuredDecision;
  writeJsonArtifact(decisionPath, decision);

  const ledgerEntry: ForecastLabLedgerEntry = {
    runId,
    startedAt,
    profileId: profile.id,
    targetSubsystem: profile.targetSubsystem,
    candidateBranch,
    allowedGlobs: [...profile.allowedGlobs],
    baselineSummary: summarizeForLedger(baseline),
    candidateSummary: summarizeForLedger(candidate),
    decision: decision.decision,
    reason: decision.reason,
    artifactsPath: runDir,
  };

  appendLedgerEntry(ledgerPath, ledgerEntry);

  return {
    runId,
    manifest,
    baseline: baseline as unknown as JsonValue,
    candidate: candidate as unknown as JsonValue,
    decision,
    ledgerEntry,
  };
}
