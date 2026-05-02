import { spawn } from 'node:child_process';
import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve, sep } from 'node:path';
import {
  getExperimentLedgerPath,
  getExperimentRunDir,
  getExperimentRunManifestPath,
  getExperimentsDir,
} from '../../utils/paths.js';
import { appendLedgerEntry, stableJsonStringify, writeRunManifest } from './ledger.js';
import { applyForecastLabCandidateEdits, withForecastLabCandidateWorkspace } from './git.js';
import { getForecastLabProfile, listForecastLabStructuredMutations } from './profiles.js';
import type { ForecastLabCommand, ForecastLabProfile } from './profiles.js';
import type { ForecastLabDecision, ForecastLabLedgerEntry, ForecastLabRunManifest, JsonValue } from './types.js';
import {
  getForecastLabBaselineCommit,
  makeForecastLabCandidateBranch,
} from './git.js';
import type { ForecastLabMutationLineage } from './mutation.js';
import type { ForecastLabMarkovParameterMutationCandidate } from './mutators/markov-parameters.js';

export type ForecastLabGatePhase = 'baseline' | 'candidate';

export interface ForecastLabCommandRunContext {
  readonly phase: ForecastLabGatePhase;
  readonly profile: ForecastLabProfile;
  readonly runId: string;
  readonly cwd?: string;
  readonly output?: (chunk: string) => void;
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
  readonly cwd: string;
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

interface ForecastLabStructuredMutationSelection {
  readonly mutationMode: 'structured';
  readonly lineage: ForecastLabMutationLineage;
  readonly selectedMutatorId: string;
  readonly mutatorId: ForecastLabMarkovParameterMutationCandidate['mutatorId'];
  readonly mutatedFiles: readonly string[];
  readonly patchSummary: readonly string[];
  readonly mutationSpecSummary: ForecastLabMarkovParameterMutationCandidate['specSummary'];
}

export interface ForecastLabRunOptions {
  readonly profileId: string;
  readonly dryRun?: boolean;
  readonly skipMutation?: boolean;
  readonly runId?: string;
  readonly now?: () => Date;
  readonly commandRunner?: ForecastLabCommandRunner;
  readonly ledgerPath?: string;
  readonly progress?: (message: string) => void;
  readonly output?: (chunk: string) => void;
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

function assertSafeProfileCommand(command: ForecastLabCommand): void {
  if (UNSAFE_SHELL_COMMAND_PATTERN.test(command.command) || UNSAFE_GIT_COMMAND_PATTERN.test(command.command)) {
    throw new ForecastLabRunnerError(`Unsafe forecast-lab command "${command.id}" is not allowed`);
  }
}

export function createForecastLabCommandRunner(spawnProcess: typeof spawn): ForecastLabCommandRunner {
  return async (command, context) => {
    assertSafeProfileCommand(command);
    const startedAtMs = Date.now();

    return await new Promise<ForecastLabCommandResult>((resolveResult) => {
      let stdout = '';
      let stderr = '';
      let timedOut = false;
      let timeoutHandle: ReturnType<typeof setTimeout> | undefined;
      let killHandle: ReturnType<typeof setTimeout> | undefined;
      let settled = false;

      const finish = (exitCode: number) => {
        if (settled) {
          return;
        }
        settled = true;

        if (timeoutHandle) {
          clearTimeout(timeoutHandle);
        }
        if (killHandle) {
          clearTimeout(killHandle);
        }

        resolveResult({
          id: command.id,
          command: command.command,
          exitCode,
          stdout,
          stderr,
          durationMs: Date.now() - startedAtMs,
          timedOut,
        });
      };

      let child: ReturnType<typeof spawn>;
      try {
        child = spawnProcess(command.command, {
          cwd: context.cwd ?? process.cwd(),
          env: { ...process.env, ...(command.env ?? {}) },
          shell: true,
          stdio: ['ignore', 'pipe', 'pipe'],
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        stderr = `Failed to start command "${command.id}": ${message}`;
        finish(1);
        return;
      }

      if (command.timeoutMs) {
        timeoutHandle = setTimeout(() => {
          timedOut = true;
          child.kill('SIGTERM');
          killHandle = setTimeout(() => {
            child.kill('SIGKILL');
          }, 5_000);
        }, command.timeoutMs);
      }

      child.stdout?.on('data', (chunk: Buffer | string) => {
        const text = chunk.toString();
        stdout += text;
        context.output?.(text);
      });

      child.stderr?.on('data', (chunk: Buffer | string) => {
        const text = chunk.toString();
        stderr += text;
        context.output?.(text);
      });

      child.once('error', (error) => {
        const message = error instanceof Error ? error.message : String(error);
        stderr = stderr
          ? `${stderr}${stderr.endsWith('\n') ? '' : '\n'}Failed to start command "${command.id}": ${message}`
          : `Failed to start command "${command.id}": ${message}`;
        finish(1);
      });

      child.once('close', (code) => {
        finish(typeof code === 'number' ? code : 1);
      });
    });
  };
}

export const defaultForecastLabCommandRunner = createForecastLabCommandRunner(spawn);

async function runGate(
  phase: ForecastLabGatePhase,
  profile: ForecastLabProfile,
  runId: string,
  commandRunner: ForecastLabCommandRunner,
  cwd: string,
  progress?: (message: string) => void,
  output?: (chunk: string) => void,
): Promise<ForecastLabGateSummary> {
  const commands = phase === 'baseline' ? profile.baselineCommands : profile.candidateCommands;
  const results: ForecastLabCommandResult[] = [];

  for (const command of commands) {
    progress?.(`${phase}: running ${command.id} — ${command.command}`);
    const result = await commandRunner(command, { phase, profile, runId, cwd, output });
    progress?.(`${phase}: completed ${command.id} (exit ${result.exitCode}, ${result.durationMs}ms)`);
    results.push(result);
  }

  return {
    phase,
    cwd,
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

function snapshotEffectiveMutationContract(profile: ForecastLabProfile): ForecastLabRunManifest['effectiveMutationContract'] {
  if (profile.mutation.mode === 'structured') {
    return {
      mode: 'structured',
      mutableFiles: [...profile.mutation.mutableFiles],
      allowedMutatorIds: [...profile.mutation.allowedMutatorIds],
      allowMultipleCandidateAttempts: profile.mutation.allowMultipleCandidateAttempts,
    };
  }

  return {
    mode: profile.mutation.mode,
    mutableFiles: [...profile.mutation.mutableFiles],
    allowMultipleCandidateAttempts: profile.mutation.allowMultipleCandidateAttempts,
  };
}

function buildCandidateArtifact(
  candidate: ForecastLabGateSummary,
  manifest: ForecastLabRunManifest,
  structuredMutation: ForecastLabStructuredMutationSelection | undefined,
  dryRun: boolean,
): JsonValue {
  if (structuredMutation) {
    return {
      ...candidate,
      mutationMode: structuredMutation.mutationMode,
      lineage: structuredMutation.lineage,
      mutationSpecSummary: structuredMutation.mutationSpecSummary,
      candidateWorkspace: manifest.candidateWorkspace,
      selectedMutator: {
        id: structuredMutation.selectedMutatorId,
        mutatorId: structuredMutation.mutatorId,
      },
      mutatedFiles: [...structuredMutation.mutatedFiles],
      patchSummary: [...structuredMutation.patchSummary],
    } as unknown as JsonValue;
  }

  return {
    ...candidate,
    mutation: dryRun ? 'dry-run: no code mutation attempted' : 'skipped by --skip-mutation',
  } as unknown as JsonValue;
}

function countOccurrences(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

function selectStructuredMutation(profile: ForecastLabProfile): ForecastLabMarkovParameterMutationCandidate {
  if (profile.mutation.mode !== 'structured') {
    throw new ForecastLabRunnerError(
      `Real forecast-lab mutation requires a structured profile with a shipped catalog. Profile "${profile.id}" uses "${profile.mutation.mode}".`,
    );
  }

  const catalog = listForecastLabStructuredMutations(profile.id);
  if (catalog.length === 0) {
    throw new ForecastLabRunnerError(
      `Real forecast-lab mutation requires a shipped structured mutator catalog. Profile "${profile.id}" has none.`,
    );
  }

  const allowedMutatorIds = new Set(profile.mutation.allowedMutatorIds);
  const selected = catalog.find((candidate) => allowedMutatorIds.has(candidate.mutatorId));
  if (!selected) {
    throw new ForecastLabRunnerError(
      `No shipped structured mutator matches the allowed mutator contract for profile "${profile.id}".`,
    );
  }

  return selected;
}

function applyStructuredMutation(
  workspaceRootDir: string,
  profile: ForecastLabProfile,
  selectedMutation: ForecastLabMarkovParameterMutationCandidate,
  runId: string,
): ForecastLabStructuredMutationSelection {
  const updatedFiles = new Map<string, string>();

  for (const edit of selectedMutation.edits) {
    const existingContents = updatedFiles.get(edit.filePath)
      ?? readFileSync(resolve(workspaceRootDir, edit.filePath), 'utf8');
    const replacementCount = countOccurrences(existingContents, edit.search);

    if (replacementCount !== edit.expectedReplacements) {
      throw new ForecastLabRunnerError(
        `Structured mutation "${selectedMutation.id}" expected ${edit.expectedReplacements} match(es) for ${edit.parameterId} in ${edit.filePath}, found ${replacementCount}.`,
      );
    }

    const nextContents = existingContents.replace(edit.search, edit.replace);
    if (nextContents === existingContents) {
      throw new ForecastLabRunnerError(
        `Structured mutation "${selectedMutation.id}" did not change ${edit.filePath} for ${edit.parameterId}.`,
      );
    }

    updatedFiles.set(edit.filePath, nextContents);
  }

  const mutatedFiles = applyForecastLabCandidateEdits(
    workspaceRootDir,
    [...updatedFiles.entries()].map(([path, contents]) => ({ path, contents })),
    {
      allowedPaths: profile.mutation.mutableFiles,
      readOnlyPaths: profile.readOnlyHarnessFiles,
    },
  );

  return {
    mutationMode: 'structured',
    lineage: {
      rootRunId: runId,
      generation: 0,
    },
    selectedMutatorId: selectedMutation.id,
    mutatorId: selectedMutation.mutatorId,
    mutatedFiles,
    patchSummary: [...selectedMutation.patchSummary],
    mutationSpecSummary: selectedMutation.specSummary,
  };
}

export async function runForecastLab(options: ForecastLabRunOptions): Promise<ForecastLabRunResult> {
  const profile = getForecastLabProfile(options.profileId);
  const now = options.now ?? (() => new Date());
  const startedAt = now().toISOString();
  const runId = options.runId ?? makeRunId(profile.id, new Date(startedAt));
  const dryRun = options.dryRun === true;
  const skipMutation = options.skipMutation === true;
  const progress = options.progress;
  const output = options.output;

  const runRealMutation = !dryRun && !skipMutation;
  const selectedMutation = runRealMutation ? selectStructuredMutation(profile) : undefined;

  const candidateBranch = makeForecastLabCandidateBranch(runId);
  const baselineCommit = getForecastLabBaselineCommit();
  const runDir = getExperimentRunDir(runId, { create: true });
  const manifestPath = getExperimentRunManifestPath(runId, { create: true });
  const baselinePath = `${runDir}/baseline.json`;
  const candidatePath = `${runDir}/candidate.json`;
  const decisionPath = `${runDir}/decision.json`;
  const ledgerPath = options.ledgerPath ?? getExperimentLedgerPath({ create: true });
  const effectiveMutationContract = snapshotEffectiveMutationContract(profile);

  for (const path of [runDir, manifestPath, baselinePath, candidatePath, decisionPath, ledgerPath]) {
    assertInsideExperiments(path);
  }

  const manifest: ForecastLabRunManifest = {
    runId,
    startedAt,
    profileId: profile.id,
    targetSubsystem: profile.targetSubsystem,
    baselineCommit,
    candidateBranch,
    allowedGlobs: [...profile.allowedGlobs],
    effectiveMutationContract,
    artifactsPath: runDir,
  };

  writeRunManifest(manifestPath, manifest);
  progress?.(`forecast-lab: started ${profile.id} (${runId})`);
  progress?.(`forecast-lab: manifest written to ${manifestPath}`);

  const commandRunner = options.commandRunner ?? defaultForecastLabCommandRunner;
  const baselineCwd = process.cwd();
  progress?.('forecast-lab: starting baseline gate');
  const baseline = await runGate('baseline', profile, runId, commandRunner, baselineCwd, progress, output);
  writeJsonArtifact(baselinePath, baseline);
  progress?.(`forecast-lab: baseline results written to ${baselinePath}`);

  let candidate: ForecastLabGateSummary;
  let structuredMutation: ForecastLabStructuredMutationSelection | undefined;

  if (runRealMutation) {
    if (!selectedMutation) {
      throw new ForecastLabRunnerError(`Missing structured mutation selection for profile "${profile.id}".`);
    }

    const candidateRun = await withForecastLabCandidateWorkspace(runId, async (workspace) => {
      const mutation = applyStructuredMutation(workspace.metadata.rootDir, profile, selectedMutation, runId);
      manifest.mutationMode = mutation.mutationMode;
      manifest.lineage = mutation.lineage;
      manifest.mutationSpecSummary = mutation.mutationSpecSummary;
      manifest.candidateWorkspace = workspace.metadata;
      writeRunManifest(manifestPath, manifest);

      progress?.(`forecast-lab: candidate workspace ${workspace.metadata.rootDir}`);
      progress?.(`forecast-lab: selected mutator ${mutation.selectedMutatorId} (${mutation.mutatorId})`);
      progress?.(`forecast-lab: mutated files ${mutation.mutatedFiles.join(', ')}`);
      progress?.(`forecast-lab: patch summary ${mutation.patchSummary.join(' | ')}`);
      progress?.('forecast-lab: starting candidate gate (structured mutation)');

      const gate = await runGate('candidate', profile, runId, commandRunner, workspace.metadata.rootDir, progress, output);
      return { gate, mutation };
    });

    candidate = candidateRun.gate;
    structuredMutation = candidateRun.mutation;
  } else {
    progress?.(`forecast-lab: starting candidate gate (${dryRun ? 'dry-run' : 'skip-mutation'})`);
    candidate = await runGate('candidate', profile, runId, commandRunner, process.cwd(), progress, output);
  }

  const candidateArtifact = buildCandidateArtifact(candidate, manifest, structuredMutation, dryRun);

  writeJsonArtifact(candidatePath, candidateArtifact);
  progress?.(`forecast-lab: candidate results written to ${candidatePath}`);

  const measuredDecision = decideRun(profile, baseline, candidate);
  const decision = skipMutation ? dropSkippedMutation(measuredDecision) : measuredDecision;
  writeJsonArtifact(decisionPath, structuredMutation
    ? {
        ...decision,
        mutationMode: structuredMutation.mutationMode,
        lineage: structuredMutation.lineage,
        mutationSpecSummary: structuredMutation.mutationSpecSummary,
        candidateWorkspace: manifest.candidateWorkspace,
        selectedMutator: {
          id: structuredMutation.selectedMutatorId,
          mutatorId: structuredMutation.mutatorId,
        },
        mutatedFiles: [...structuredMutation.mutatedFiles],
        patchSummary: [...structuredMutation.patchSummary],
      }
    : decision);
  progress?.(`forecast-lab: decision ${decision.decision} — ${decision.reason}`);

  const ledgerEntry: ForecastLabLedgerEntry = {
    runId,
    startedAt,
    profileId: profile.id,
    targetSubsystem: profile.targetSubsystem,
    candidateBranch,
    allowedGlobs: [...profile.allowedGlobs],
    effectiveMutationContract,
    mutationMode: structuredMutation?.mutationMode,
    lineage: structuredMutation?.lineage,
    mutationSpecSummary: structuredMutation?.mutationSpecSummary,
    candidateWorkspace: manifest.candidateWorkspace,
    baselineSummary: summarizeForLedger(baseline),
    candidateSummary: summarizeForLedger(candidate),
    decision: decision.decision,
    reason: decision.reason,
    artifactsPath: runDir,
  };

  appendLedgerEntry(ledgerPath, ledgerEntry);
  progress?.(`forecast-lab: ledger appended at ${ledgerPath}`);

  return {
    runId,
    manifest,
    baseline: baseline as unknown as JsonValue,
    candidate: candidateArtifact,
    decision,
    ledgerEntry,
  };
}
