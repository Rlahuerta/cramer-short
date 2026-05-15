import { existsSync, readFileSync } from 'node:fs';
import { join } from 'node:path';
import { DynamicStructuredTool } from '@langchain/core/tools';
import type { RunnableConfig } from '@langchain/core/runnables';
import { z } from 'zod';
import { findLatestKeptLedgerEntry, readLedgerEntries, readRunManifest } from '../experiments/forecast-lab/ledger.js';
import {
  getForecastLabProfile,
  listForecastLabProfiles,
  listForecastLabStructuredMutations,
  type ForecastLabProfile,
} from '../experiments/forecast-lab/profiles.js';
import {
  rankForecastLabMutators,
  type ForecastLabRankedMutator,
} from '../experiments/forecast-lab/mutator-ranker.js';
import {
  promoteForecastLab,
  resetForecastLab,
  runForecastLab,
  type ForecastLabPromotionOptions,
  type ForecastLabPromotionResult,
  type ForecastLabResetMode,
  type ForecastLabResetOptions,
  type ForecastLabResetResult,
  type ForecastLabRunOptions,
  type ForecastLabRunResult,
} from '../experiments/forecast-lab/runner.js';
import { runForecastLabProfileImprovementLoop } from '../experiments/forecast-lab/improvement-loop.js';
import type {
  ForecastLabImprovementMetrics,
  ForecastLabProfileImprovementLoopResult,
} from '../experiments/forecast-lab/improvement-loop.js';
import { listForecastLabMutatorIds } from '../experiments/forecast-lab/mutation.js';
import { routeForecastLabQuery } from '../experiments/forecast-lab/router.js';
import type {
  ForecastLabLedgerEntry,
  ForecastLabPromotionStatus,
  ForecastLabRoutingInvocationSource,
  ForecastLabRunManifest,
} from '../experiments/forecast-lab/types.js';
import { experimentsPath, getExperimentLedgerPath, getExperimentRunManifestPath } from '../utils/paths.js';
import { formatToolResult } from './types.js';

export const FORECAST_LAB_RUN_DESCRIPTION = `
Run the bounded forecast-lab workflow for repo-native forecast experiments.

## When to Use

- Routed forecast-lab improvement requests that should stay inside the bounded baseline-vs-candidate workflow
- Bounded current-best vs shipped-baseline comparisons for a forecast-lab profile
- Bounded shipped-mutator catalog queries for a forecast-lab profile
- Plan-only forecast-lab requests that need the exact approved experiment plan without running commands
- Explicitly approved forecast-lab promotion requests for a kept structured run

## Actions

- \`guided-improve\`: resolve the profile, then either return the exact bounded plan (\`execute=false\`) or run the guided improvement path
- \`compare-best-vs-shipped\`: compare either the latest kept structured run against its shipped-default baseline, or a named mutator against the active/live promoted run when \`mutationId\` is supplied
- \`list-mutators\`: list the shipped mutator ids for one forecast-lab profile, or summarize every structured profile when no unique profile is specified
- \`catalog-extension-plan\`: explain the bounded code-change path for adding a new shipped mutator outside the current catalog, without reading experiment artifacts or running lineage commands
- \`mutator-scorecard\`: summarize shipped mutators for a profile in a readable table for operator decision support, showing per-mutator status, attempts, kept, regressed, health, and score
- \`batch-replay-mutators\`: replay shipped mutators from a common baseline (shipped defaults) and return a readable comparative matrix for operator decision support
- \`iterative-improve-mutators\`: run a bounded trial loop around the current best shipped mutator for one supported short-horizon profile and return the best trial found so far
- \`promote-approved\`: run the bounded promotion verification path only after the user explicitly approves promotion, then activate the verified parameters for normal forecasts
- \`reset-live\`: roll the live forecast-lab baseline back to shipped defaults or the last known-good activated baseline

## Hard Rules

- Never auto-promote. A kept result only becomes promotion-ready until the user explicitly approves promotion.
- Never bypass the bounded promotion path when activating promoted work.
- Promotion never retries mutation or broadens the editable surface.
- Use this tool instead of generic repo read/edit flows once forecast-lab execution has started.
`.trim();

const forecastLabRunSchema = z.object({
  action: z.enum(['guided-improve', 'compare-best-vs-shipped', 'list-mutators', 'catalog-extension-plan', 'mutator-scorecard', 'batch-replay-mutators', 'iterative-improve-mutators', 'promote-approved', 'reset-live']),
  query: z.string().optional().describe('Original user request. Required when profileId is omitted for guided-improve.'),
  profileId: z.string().optional().describe('Forecast-lab profile id. Optional when the tool can resolve a unique profile automatically.'),
  sourceRunId: z.string().optional().describe('Kept run id to promote. Optional when there is a unique pending promotion source.'),
  mutationId: z.string().optional().describe('For compare-best-vs-shipped: optionally compare a named kept mutator against the active/live promoted run.'),
  resetMode: z.enum(['defaults', 'last-known-good']).optional().describe('For reset-live: restore shipped defaults or the last known-good activated baseline.'),
  execute: z.boolean().optional().describe('For guided-improve: when false, return the bounded plan only and do not run commands or write artifacts. Defaults to true.'),
  mutator: z.string().optional().describe('Optional structured mutator override for guided-improve execution.'),
  rankMutators: z.boolean().optional().describe('Enable ledger-based mutator ranking for guided-improve execution.'),
  routingSource: z.enum(['auto-routed', 'manual-request']).optional().describe('How the improvement request reached forecast-lab. Defaults to manual-request.'),
  limit: z.number().int().min(1).max(50).optional().describe('For batch-replay-mutators: maximum number of mutators to replay. Defaults to 5. Must be between 1 and 50.'),
  iterations: z.number().int().min(1).max(5).optional().describe('For iterative-improve-mutators: maximum trial-loop iterations. Defaults to 2. Must be between 1 and 5.'),
});

type ForecastLabRunToolInput = z.infer<typeof forecastLabRunSchema>;

interface ForecastLabPlanPayload {
  readonly _tool: 'forecast_lab_run';
  readonly action: 'guided-improve';
  readonly status: 'ok';
  readonly execute: false;
  readonly profileId: string;
  readonly allowedGlobs: readonly string[];
  readonly baselineCommands: readonly string[];
  readonly candidateCommands: readonly string[];
  readonly mutationMode: ForecastLabProfile['mutation']['mode'];
  readonly promotionReady: false;
  readonly answer: string;
}

interface ForecastLabGuidedImprovePayload {
  readonly _tool: 'forecast_lab_run';
  readonly action: 'guided-improve';
  readonly status: 'ok';
  readonly execute: true;
  readonly profileId: string;
  readonly runId: string;
  readonly decision: 'keep' | 'drop';
  readonly reason: string;
  readonly allowedGlobs: readonly string[];
  readonly artifactsPath: string;
  readonly promotionReady: boolean;
  readonly promotionStatus?: ForecastLabPromotionStatus;
  readonly sourceRunId?: string;
  readonly answer: string;
}

interface ForecastLabPromotePayload {
  readonly _tool: 'forecast_lab_run';
  readonly action: 'promote-approved';
  readonly status: 'ok';
  readonly profileId: string;
  readonly runId: string;
  readonly sourceRunId: string;
  readonly decision: 'keep';
  readonly reason: string;
  readonly artifactsPath: string;
  readonly activationArtifactsPath: string;
  readonly activeStatePath: string;
  readonly answer: string;
}

interface ForecastLabMetricComparison {
  readonly horizon: string;
  readonly baselineDirAcc: number;
  readonly candidateDirAcc: number;
  readonly deltaDirAcc: number;
  readonly baselineBrier: number;
  readonly candidateBrier: number;
  readonly deltaBrier: number;
  readonly baselineCiCov: number;
  readonly candidateCiCov: number;
  readonly deltaCiCov: number;
}

type ForecastLabCompareLiveStatus = 'already-live' | 'ready-to-promote' | 'not-live';

interface ForecastLabComparePayload {
  readonly _tool: 'forecast_lab_run';
  readonly action: 'compare-best-vs-shipped';
  readonly status: 'ok';
  readonly profileId: string;
  readonly comparisonTarget?: 'shipped-baseline' | 'active-live';
  readonly sourceRunId: string;
  readonly activeSourceRunId?: string;
  readonly decision: 'keep' | 'drop';
  readonly reason: string;
  readonly artifactsPath: string;
  readonly liveStatus: ForecastLabCompareLiveStatus;
  readonly promotionCommand?: string;
  readonly mutationId?: string;
  readonly activeMutationId?: string;
  readonly mutationSummary?: string;
  readonly metrics?: readonly ForecastLabMetricComparison[];
  readonly answer: string;
}

interface ForecastLabCatalogExtensionPayload {
  readonly _tool: 'forecast_lab_run';
  readonly action: 'catalog-extension-plan';
  readonly status: 'ok';
  readonly profileId?: string;
  readonly targetSubsystem?: string;
  readonly allowedGlobs?: readonly string[];
  readonly mutationMode?: ForecastLabProfile['mutation']['mode'];
  readonly allowedMutatorIds?: readonly string[];
  readonly currentCatalogIds?: readonly string[];
  readonly catalogFiles: readonly string[];
  readonly validationFiles: readonly string[];
  readonly requestedMutatorId?: string;
  readonly requestedParameterChanges?: readonly string[];
  readonly operatorMutatorIds: readonly string[];
  readonly answer: string;
}

interface ForecastLabMutatorListProfileSummary {
  readonly profileId: string;
  readonly targetSubsystem: ForecastLabProfile['targetSubsystem'];
  readonly mutationMode: ForecastLabProfile['mutation']['mode'];
  readonly currentCatalogIds: readonly string[];
  readonly allowedOperatorIds: readonly string[];
}

interface ForecastLabMutatorListPayload {
  readonly _tool: 'forecast_lab_run';
  readonly action: 'list-mutators';
  readonly status: 'ok';
  readonly profileId?: string;
  readonly profiles: readonly ForecastLabMutatorListProfileSummary[];
  readonly dryRunProfiles: readonly string[];
  readonly frameworkOperatorIds: readonly string[];
  readonly answer: string;
}

interface ForecastLabErrorPayload {
  readonly _tool: 'forecast_lab_run';
  readonly action: ForecastLabRunToolInput['action'];
  readonly status: 'error';
  readonly error: string;
  readonly answer: string;
}

interface ForecastLabResetPayload {
  readonly _tool: 'forecast_lab_run';
  readonly action: 'reset-live';
  readonly status: 'ok';
  readonly profileId: string;
  readonly resetMode: ForecastLabResetMode;
  readonly runId: string;
  readonly artifactsPath: string;
  readonly resetArtifactPath: string;
  readonly activeStatePath?: string;
  readonly answer: string;
}

interface ForecastLabMutatorScorecardPayload {
  readonly _tool: 'forecast_lab_run';
  readonly action: 'mutator-scorecard';
  readonly status: 'ok';
  readonly profileId: string;
  readonly totalStructuredRuns: number;
  readonly rankedMutators: readonly {
    readonly id: string;
    readonly mutatorId: string;
    readonly applicable: boolean;
    readonly unused: boolean;
    readonly attempts: number;
    readonly keptRuns: number;
    readonly regressedRuns: number;
    readonly health: string;
    readonly score: number;
  }[];
  readonly answer: string;
}

interface ForecastLabBatchReplayMutatorResult {
  readonly mutatorId: string;
  readonly decision: 'keep' | 'drop';
  readonly reason: string;
  readonly behaviorSummary: string;
  readonly metrics?: readonly ForecastLabMetricComparison[];
}

interface ForecastLabBatchReplayPayload {
  readonly _tool: 'forecast_lab_run';
  readonly action: 'batch-replay-mutators';
  readonly status: 'ok';
  readonly profileId: string;
  readonly baselineDescription: string;
  readonly replayedCount: number;
  readonly results: readonly ForecastLabBatchReplayMutatorResult[];
  readonly answer: string;
}

interface ForecastLabIterativeImprovePayload {
  readonly _tool: 'forecast_lab_run';
  readonly action: 'iterative-improve-mutators';
  readonly status: 'ok';
  readonly profileId: string;
  readonly iterationsRun: number;
  readonly bestMutationId: string;
  readonly bestDecision: 'keep' | 'drop';
  readonly keepSatisfied: number;
  readonly keepTotal: number;
  readonly bestObjectiveScore: number;
  readonly bestPrimaryScore: number;
  readonly baselineMetrics: ForecastLabImprovementMetrics;
  readonly bestMetrics: ForecastLabImprovementMetrics;
  readonly answer: string;
}

export type ForecastLabRunToolPayload =
  | ForecastLabPlanPayload
  | ForecastLabGuidedImprovePayload
  | ForecastLabComparePayload
  | ForecastLabMutatorListPayload
  | ForecastLabCatalogExtensionPayload
  | ForecastLabMutatorScorecardPayload
  | ForecastLabBatchReplayPayload
  | ForecastLabIterativeImprovePayload
  | ForecastLabPromotePayload
  | ForecastLabResetPayload
  | ForecastLabErrorPayload;

type CreateForecastLabRunToolDeps = {
  readonly runForecastLabFn?: (options: ForecastLabRunOptions) => Promise<ForecastLabRunResult>;
  readonly runForecastLabImprovementLoopFn?: (options: {
    readonly profileId: string;
    readonly seedMutatorId?: string;
    readonly maxIterations?: number;
    readonly progress?: (message: string) => void;
  }) => Promise<ForecastLabProfileImprovementLoopResult>;
  readonly promoteForecastLabFn?: (options: ForecastLabPromotionOptions) => Promise<ForecastLabPromotionResult>;
  readonly resetForecastLabFn?: (options: ForecastLabResetOptions) => Promise<ForecastLabResetResult>;
  readonly readLedgerEntriesFn?: (path: string) => ForecastLabLedgerEntry[];
  readonly getLedgerPathFn?: () => string;
  readonly findLatestKeptLedgerEntryFn?: (
    path: string,
    profileId: string,
    options?: { readonly mutationMode?: ForecastLabLedgerEntry['mutationMode'] },
  ) => ForecastLabLedgerEntry | undefined;
  readonly readRunManifestFn?: (path: string) => ForecastLabRunManifest;
  readonly readTextFileFn?: (path: string) => string;
  readonly existsSyncFn?: (path: string) => boolean;
};

function commandStrings(commands: readonly ForecastLabProfile['baselineCommands'][number][]): string[] {
  return commands.map((command) => command.command);
}

function profileListLabel(): string {
  return listForecastLabProfiles().map((profile) => profile.id).join(', ');
}

function buildPlanAnswer(
  profile: ForecastLabProfile,
  routingReason: string,
  mutator?: string,
): string {
  const candidateStep = profile.mutation.mode === 'structured'
    ? `Run one bounded structured mutation inside the approved editable surface${mutator ? ` using mutator "${mutator}"` : ''}.`
    : 'This shipped profile is dry-run only, so keep the workflow at plan/baseline level until a structured mutation catalog exists.';

  return [
    `Forecast-lab bounded plan for ${profile.id}.`,
    '',
    `Experiment scope: ${profile.targetSubsystem}. Approved editable files: ${profile.allowedGlobs.join(', ')}.`,
    `Baseline: run the fixed harness first (${commandStrings(profile.baselineCommands).join(' ; ')}).`,
    `Candidate: ${candidateStep}`,
    `Gates: compare the same harness/metric set on the candidate (${commandStrings(profile.candidateCommands).join(' ; ')}).`,
    'Decision: keep only if the candidate passes the fixed gates; otherwise drop, revert, or discard it.',
    'Artifacts: execution writes under .cramer-short/experiments/, but this plan-only run did not execute commands or write artifacts.',
    `Routing: ${routingReason}`,
    'Promotion: any kept structured result becomes promotion-ready and still requires explicit approval before promotion.',
  ].join('\n');
}

function buildGuidedImproveAnswer(
  profile: ForecastLabProfile,
  result: ForecastLabRunResult,
  requestedMutator?: string,
): string {
  const promotion = result.ledgerEntry.promotion;
  const promotionReady = promotion?.status === 'approval-required';
  const approvalLine = promotionReady
    ? `Promotion-ready: yes. Approval required before promotion. Reply "approve forecast-lab promotion for ${profile.id} run ${result.runId}" to continue.`
    : profile.mutation.mode === 'structured'
      ? 'Promotion-ready: no. This run was kept only if the fixed gates passed, but it is not waiting on approval.'
      : 'Promotion-ready: no. This profile is still dry-run only, so there is no promotion path yet.';

  return [
    `Forecast-lab guided improvement finished for ${profile.id}.`,
    '',
    ...(requestedMutator ? [`Requested mutator: ${requestedMutator}.`] : []),
    `Decision: ${result.decision.decision.toUpperCase()} — ${result.decision.reason}`,
    `Artifacts: ${result.manifest.artifactsPath}`,
    approvalLine,
    'No activation was executed.',
  ].join('\n');
}

function buildPromoteAnswer(profile: ForecastLabProfile, result: ForecastLabPromotionResult): string {
  return [
    `Forecast-lab promotion completed for ${profile.id}.`,
    '',
    `Source run: ${result.sourceRunId}`,
    `Promotion run: ${result.runId}`,
    `Verification: KEEP — ${result.decision.reason}`,
    `Artifacts: ${result.manifest.artifactsPath}`,
    `Activation artifacts: ${result.activation.artifactsPath}`,
    `Active baseline: ${result.activeStatePath}`,
    'Activation: promoted parameters are now live for normal forecasts in this session and future restarts from the updated checkout.',
  ].join('\n');
}

function formatSignedDelta(value: number, unit = ''): string {
  const sign = value > 0 ? '+' : value < 0 ? '-' : '';
  const magnitude = Math.abs(value);
  return `${sign}${magnitude.toFixed(unit === 'pp' ? 1 : 3)}${unit}`;
}

interface ForecastLabUltraShortMetrics {
  readonly dirAcc: number;
  readonly brier: number;
  readonly ciCov: number;
}

function parseUltraShortBaselineVariantMetrics(stdout: string): Map<string, ForecastLabUltraShortMetrics> {
  const metrics = new Map<string, ForecastLabUltraShortMetrics>();
  let currentHorizon: string | null = null;

  for (const line of stdout.split(/\r?\n/)) {
    const horizonMatch = line.match(/^BTC-USD horizon (\d+d)$/i);
    if (horizonMatch) {
      currentHorizon = horizonMatch[1]!.toLowerCase();
      continue;
    }

    if (!currentHorizon) {
      continue;
    }

    const variantMatch = line.match(
      /^baseline warmup=120 stride=3\s*│\s*\d+\s*│\s*\d+\s*│\s*([0-9.]+)%\s*│\s*([0-9.]+)\s*│\s*([0-9.]+)%/i,
    );
    if (!variantMatch) {
      continue;
    }

    metrics.set(currentHorizon, {
      dirAcc: Number(variantMatch[1]),
      brier: Number(variantMatch[2]),
      ciCov: Number(variantMatch[3]),
    });
  }

  return metrics;
}

function buildMetricComparisons(
  baselineStdout: string,
  candidateStdout: string,
): readonly ForecastLabMetricComparison[] {
  const baseline = parseUltraShortBaselineVariantMetrics(baselineStdout);
  const candidate = parseUltraShortBaselineVariantMetrics(candidateStdout);

  return ['1d', '2d', '3d']
    .map((horizon) => {
      const baselineMetrics = baseline.get(horizon);
      const candidateMetrics = candidate.get(horizon);
      if (!baselineMetrics || !candidateMetrics) {
        return null;
      }

      return {
        horizon,
        baselineDirAcc: baselineMetrics.dirAcc,
        candidateDirAcc: candidateMetrics.dirAcc,
        deltaDirAcc: candidateMetrics.dirAcc - baselineMetrics.dirAcc,
        baselineBrier: baselineMetrics.brier,
        candidateBrier: candidateMetrics.brier,
        deltaBrier: candidateMetrics.brier - baselineMetrics.brier,
        baselineCiCov: baselineMetrics.ciCov,
        candidateCiCov: candidateMetrics.ciCov,
        deltaCiCov: candidateMetrics.ciCov - baselineMetrics.ciCov,
      } satisfies ForecastLabMetricComparison;
    })
    .filter((value): value is ForecastLabMetricComparison => value !== null);
}

function extractExplicitProfileId(text: string): string | undefined {
  const lower = text.toLowerCase();
  return listForecastLabProfiles().find((profile) => lower.includes(profile.id.toLowerCase()))?.id;
}

function resolveComparisonProfile(
  input: ForecastLabRunToolInput,
  ledgerEntries?: readonly ForecastLabLedgerEntry[],
): { profile: ForecastLabProfile } | { error: string } {
  if (input.profileId?.trim()) {
    return { profile: getForecastLabProfile(input.profileId) };
  }

  if (input.mutationId?.trim() && ledgerEntries) {
    const matchingProfiles = [...new Set(
      ledgerEntries
        .filter((entry) => entry.decision === 'keep' && entry.mutationMode === 'structured' && entry.mutationId === input.mutationId)
        .map((entry) => entry.profileId),
    )];
    if (matchingProfiles.length === 1) {
      return { profile: getForecastLabProfile(matchingProfiles[0]!) };
    }
    if (matchingProfiles.length > 1) {
      return {
        error: `Could not resolve a unique forecast-lab profile from mutationId "${input.mutationId}". Matching profiles: ${matchingProfiles.join(', ')}`,
      };
    }
  }

  const query = input.query?.trim();
  if (!query) {
    return {
      error: `compare-best-vs-shipped requires profileId or query. Available profiles: ${profileListLabel()}`,
    };
  }

  const explicitProfileId = extractExplicitProfileId(query);
  if (explicitProfileId) {
    return { profile: getForecastLabProfile(explicitProfileId) };
  }

  const route = routeForecastLabQuery(query);
  if (route.preferredProfileId) {
    return { profile: getForecastLabProfile(route.preferredProfileId) };
  }

  return {
    error: `Could not resolve a unique forecast-lab profile from the comparison request. Specify profileId. Available profiles: ${profileListLabel()}`,
  };
}

function extractComparisonMutationId(query: string | undefined): string | undefined {
  return query?.match(/\b(markov-[A-Za-z0-9][A-Za-z0-9_.-]*)\b/i)?.[1];
}

function findLatestKeptStructuredMutationEntry(
  entries: readonly ForecastLabLedgerEntry[],
  profileId: string,
  mutationId: string,
): ForecastLabLedgerEntry | undefined {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index]!;
    if (
      entry.profileId === profileId
      && entry.decision === 'keep'
      && entry.mutationMode === 'structured'
      && entry.mutationId === mutationId
    ) {
      return entry;
    }
  }

  return undefined;
}

function findLedgerEntryByRunId(
  entries: readonly ForecastLabLedgerEntry[],
  runId: string,
): ForecastLabLedgerEntry | undefined {
  for (let index = entries.length - 1; index >= 0; index -= 1) {
    const entry = entries[index]!;
    if (entry.runId === runId) {
      return entry;
    }
  }

  return undefined;
}

function readActiveSourceRunId(
  profileId: string,
  existsSyncFn: (path: string) => boolean,
  readTextFileFn: (path: string) => string,
): { activeStatePath: string; sourceRunId?: string } {
  const activeStatePath = experimentsPath('active-promotions', `${profileId}.json`);
  if (!existsSyncFn(activeStatePath)) {
    return { activeStatePath };
  }

  const activeState = readJsonObject(activeStatePath, readTextFileFn);
  const sourceRunId = typeof activeState.sourceRunId === 'string'
    ? activeState.sourceRunId
    : undefined;
  return { activeStatePath, sourceRunId };
}

function extractComparisonStdout(
  artifact: Record<string, unknown>,
  preferredCommandId = 'walk-forward-btc-ultra-short-horizon',
): string | undefined {
  if (!Array.isArray(artifact.commands)) {
    return undefined;
  }

  const preferred = artifact.commands.find((command) =>
    command && typeof command === 'object' && command['id'] === preferredCommandId && typeof command['stdout'] === 'string'
  );
  if (preferred && typeof preferred === 'object' && typeof preferred['stdout'] === 'string') {
    return preferred['stdout'];
  }

  const fallback = artifact.commands.find((command) =>
    command && typeof command === 'object' && typeof command['stdout'] === 'string'
  );
  return fallback && typeof fallback === 'object' && typeof fallback['stdout'] === 'string'
    ? fallback['stdout']
    : undefined;
}

function buildMutatorVsActiveAnswer(
  profile: ForecastLabProfile,
  requestedEntry: ForecastLabLedgerEntry,
  activeEntry: ForecastLabLedgerEntry,
  comparisons: readonly ForecastLabMetricComparison[],
): string {
  const sameRun = requestedEntry.runId === activeEntry.runId;
  const metricLines = comparisons.length === 0
    ? []
    : [
        '',
        'Accuracy snapshot (requested mutator vs active/live run):',
        ...comparisons.map((comparison) =>
          `${comparison.horizon}: Active Dir Acc ${comparison.baselineDirAcc.toFixed(1)}% vs Requested Dir Acc ${comparison.candidateDirAcc.toFixed(1)}% (${formatSignedDelta(comparison.deltaDirAcc, 'pp')}), Active Brier ${comparison.baselineBrier.toFixed(3)} vs Requested Brier ${comparison.candidateBrier.toFixed(3)} (${formatSignedDelta(comparison.deltaBrier)}), Active CI Cov ${comparison.baselineCiCov.toFixed(1)}% vs Requested CI Cov ${comparison.candidateCiCov.toFixed(1)}% (${formatSignedDelta(comparison.deltaCiCov, 'pp')})`,
        ),
      ];

  return [
    `Forecast-lab mutator-vs-active comparison for ${profile.id}.`,
    '',
    `Requested mutator: ${requestedEntry.mutationId ?? requestedEntry.runId}${requestedEntry.mutationSummary ? ` — ${requestedEntry.mutationSummary}` : ''}`,
    `Requested source run: ${requestedEntry.runId}`,
    `Active mutator: ${activeEntry.mutationId ?? activeEntry.runId}${activeEntry.mutationSummary ? ` — ${activeEntry.mutationSummary}` : ''}`,
    `Active source run: ${activeEntry.runId}`,
    sameRun
      ? 'Active/live state: the requested mutator is already the active promoted run.'
      : 'Active/live state: the requested mutator is not the active promoted run yet; the table below compares their kept candidate metrics directly.',
    ...metricLines,
  ].join('\n');
}

function readJsonObject(path: string, readTextFileFn: (path: string) => string): Record<string, unknown> {
  const parsed = JSON.parse(readTextFileFn(path)) as unknown;
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new Error(`Expected JSON object at ${path}`);
  }
  return parsed as Record<string, unknown>;
}

function getLatestKeptStructuredEntry(
  profileId: string,
  ledgerPath: string,
  findLatestKeptLedgerEntryFn: NonNullable<CreateForecastLabRunToolDeps['findLatestKeptLedgerEntryFn']>,
): ForecastLabLedgerEntry | undefined {
  return findLatestKeptLedgerEntryFn(ledgerPath, profileId, { mutationMode: 'structured' });
}

function buildNoCurrentBestCompareAnswer(profile: ForecastLabProfile): string {
  return [
    `Forecast-lab comparison for ${profile.id}.`,
    '',
    'Current best: no kept structured run exists yet, so there is nothing to compare against the shipped baseline.',
    'Shipped baseline: still the active reference because no kept candidate has been recorded.',
    'Regular forecasts: not live yet. Run a guided improvement first, then explicitly approve a kept run before ordinary forecast queries can use it.',
  ].join('\n');
}

function buildCompareAnswer(
  profile: ForecastLabProfile,
  latestKeptEntry: ForecastLabLedgerEntry,
  liveStatus: ForecastLabCompareLiveStatus,
  promotionCommand: string | undefined,
  comparisons: readonly ForecastLabMetricComparison[],
): string {
  const headline = latestKeptEntry.decision === 'keep'
    ? 'Yes — the current best is better under the bounded forecast-lab gate.'
    : 'No — the current best did not beat the shipped baseline under the bounded forecast-lab gate.';
  const metricLines = comparisons.length === 0
    ? []
    : [
        '',
        'Horizon snapshot from the latest kept run:',
        ...comparisons.map((comparison) =>
          `${comparison.horizon}: Dir Acc ${comparison.baselineDirAcc.toFixed(1)}% -> ${comparison.candidateDirAcc.toFixed(1)}% (${formatSignedDelta(comparison.deltaDirAcc, 'pp')}), Brier ${comparison.baselineBrier.toFixed(3)} -> ${comparison.candidateBrier.toFixed(3)} (${formatSignedDelta(comparison.deltaBrier)}), CI Cov ${comparison.baselineCiCov.toFixed(1)}% -> ${comparison.candidateCiCov.toFixed(1)}% (${formatSignedDelta(comparison.deltaCiCov, 'pp')})`,
        ),
      ];
  const liveLine = liveStatus === 'already-live'
    ? 'Regular forecasts: this kept run is already live, so ordinary forecast queries are already using it.'
    : promotionCommand
      ? `Regular forecasts: not live yet. Reply "${promotionCommand}" to activate this kept run for ordinary forecast queries.`
      : 'Regular forecasts: not live yet. Promote this kept run before expecting ordinary forecast queries to use it.';

  return [
    `Forecast-lab comparison for ${profile.id}.`,
    '',
    `Latest kept structured run: ${latestKeptEntry.runId}`,
    ...(latestKeptEntry.mutationId ? [`Mutation: ${latestKeptEntry.mutationId}${latestKeptEntry.mutationSummary ? ` — ${latestKeptEntry.mutationSummary}` : ''}`] : []),
    `Verdict: ${headline}`,
    ...metricLines,
    '',
    liveLine,
  ].join('\n');
}

function buildResetAnswer(
  profile: ForecastLabProfile,
  result: ForecastLabResetResult,
): string {
  return [
    `Forecast-lab reset completed for ${profile.id}.`,
    '',
    `Mode: ${result.mode === 'defaults' ? 'shipped defaults' : 'last known-good activated baseline'}`,
    `Artifacts: ${result.artifactsPath}`,
    `Reset record: ${result.resetArtifactPath}`,
    ...(result.activeStatePath
      ? [`Active baseline: ${result.activeStatePath}`]
      : ['Active baseline: shipped defaults (no promoted baseline is currently live).']),
  ].join('\n');
}

function resolveCatalogExtensionProfile(
  input: ForecastLabRunToolInput,
): { profile?: ForecastLabProfile; routingReason: string } {
  if (input.profileId?.trim()) {
    return {
      profile: getForecastLabProfile(input.profileId),
      routingReason: 'Profile supplied explicitly.',
    };
  }

  const query = input.query?.trim();
  if (!query) {
    return {
      routingReason: 'No profile was supplied. Returning generic catalog-extension guidance instead of guessing.',
    };
  }

  const explicitProfileId = extractExplicitProfileId(query);
  if (explicitProfileId) {
    return {
      profile: getForecastLabProfile(explicitProfileId),
      routingReason: 'Profile inferred from an explicit profile id in the request.',
    };
  }

  const route = routeForecastLabQuery(query);
  if (route.preferredProfileId) {
    return {
      profile: getForecastLabProfile(route.preferredProfileId),
      routingReason: route.reasons.join(' '),
    };
  }

  return {
    routingReason: 'No unique profile could be inferred from the catalog-extension request. Returning generic guidance instead of choosing a profile heuristically.',
  };
}

function listCatalogExtensionCatalogFiles(profile: ForecastLabProfile | undefined): readonly string[] {
  if (!profile || profile.targetSubsystem === 'markov-distribution') {
    return [
      'src/experiments/forecast-lab/mutators/markov-parameters.ts',
      'src/experiments/forecast-lab/profiles.ts',
    ];
  }

  return ['src/experiments/forecast-lab/profiles.ts'];
}

function listCatalogExtensionValidationFiles(profile: ForecastLabProfile | undefined): readonly string[] {
  if (!profile || profile.targetSubsystem === 'markov-distribution') {
    return [
      'src/experiments/forecast-lab/mutators/markov-parameters.test.ts',
      'src/experiments/forecast-lab/profiles.test.ts',
      'src/tools/finance/markov-distribution.test.ts',
      'src/tools/finance/backtest/walk-forward-r5.test.ts',
    ];
  }

  return ['src/experiments/forecast-lab/profiles.test.ts'];
}

function extractCatalogExtensionMutatorId(query: string | undefined): string | undefined {
  if (!query) {
    return undefined;
  }

  const explicitName = query.match(/\bname it(?: something like)?\s*:\s*([a-z0-9-]+)/i)?.[1];
  if (explicitName) {
    return explicitName;
  }

  const quotedName = query.match(/\bname it(?: something like)?\s*"?([a-z0-9-]+)"?/i)?.[1];
  return quotedName;
}

function extractCatalogExtensionParameterChanges(query: string | undefined): readonly string[] {
  if (!query) {
    return [];
  }

  return [...query.matchAll(/^\s*[-*•]\s*([A-Za-z0-9_]+:\s*[^\n]*->\s*[^\n]+)$/gm)]
    .map((match) => match[1]?.trim())
    .filter((line): line is string => Boolean(line));
}

function buildCatalogExtensionAnswer(
  profile: ForecastLabProfile | undefined,
  routingReason: string,
  query?: string,
): string {
  const structuredProfiles = listForecastLabProfiles()
    .filter((candidate) => candidate.mutation.mode === 'structured')
    .map((candidate) => candidate.id);
  const dryRunProfiles = listForecastLabProfiles()
    .filter((candidate) => candidate.mutation.mode !== 'structured')
    .map((candidate) => candidate.id);
  const operatorMutatorIds = listForecastLabMutatorIds().join(', ');
  const catalogFiles = listCatalogExtensionCatalogFiles(profile);
  const validationFiles = listCatalogExtensionValidationFiles(profile);
  const requestedMutatorId = extractCatalogExtensionMutatorId(query);
  const requestedParameterChanges = extractCatalogExtensionParameterChanges(query);

  if (!profile) {
    return [
      'Forecast-lab catalog extension is a bounded code-change plan, not a runtime lineage action.',
      '',
      'I did not pick a profile heuristically, so I did not read `.cramer-short/experiments/` artifacts or run a lineage command.',
      `Structured lineage profiles today: ${structuredProfiles.join(', ')}.`,
      `Dry-run-only profiles today: ${dryRunProfiles.join(', ')}.`,
      `Catalog files to open directly before editing: ${catalogFiles.join(', ')}.`,
      `Validation files to update directly: ${validationFiles.join(', ')}.`,
      `Built-in structured mutator operators: ${operatorMutatorIds}.`,
      'To add a new shipped mutator outside the current catalog, update the profile-specific catalog in code, allow it in the target profile contract, add focused tests, then rerun the lineage only after that code exists.',
      'If you want a profile-specific plan, specify the target profile id in the request.',
      `Routing: ${routingReason}`,
    ].join('\n');
  }

  const allowedMutatorIds = profile.mutation.mode === 'structured'
    ? [...profile.mutation.allowedMutatorIds]
    : [];
  const currentCatalogIds = profile.mutation.mode === 'structured'
    ? listForecastLabStructuredMutations(profile.id).map((candidate) => candidate.id)
    : [];
  const rerunLine = profile.mutation.mode === 'structured'
    ? `After the new mutator is implemented and allowed for ${profile.id}, rerun the lineage with forecast_lab_run(action="guided-improve", profileId="${profile.id}").`
    : `Profile ${profile.id} is ${profile.mutation.mode} today, so a real lineage rerun also requires adding a structured mutation path before forecast-lab can execute it.`;
  const requestedSpecLines = requestedMutatorId || requestedParameterChanges.length > 0
    ? [
        requestedMutatorId
          ? `Requested mutator id: ${requestedMutatorId}.`
          : 'Requested mutator id: none supplied explicitly.',
        ...(requestedParameterChanges.length > 0
          ? ['Requested parameter deltas:', ...requestedParameterChanges.map((line) => `- ${line}`)]
          : []),
      ]
    : [];

  return [
    `Forecast-lab catalog-extension plan for ${profile.id}.`,
    '',
    'This is a bounded code-change plan. It is not a safe runtime mutation request, so I did not inspect experiment artifacts or try to rerun the lineage directly.',
    `Target subsystem: ${profile.targetSubsystem}.`,
    `Approved editable files for this profile: ${profile.allowedGlobs.join(', ')}.`,
    `Catalog files to open directly next: ${catalogFiles.join(', ')}.`,
    `Validation files to open directly next: ${validationFiles.join(', ')}.`,
    `Current mutation mode: ${profile.mutation.mode}.`,
    `Allowed structured operator ids for this profile: ${allowedMutatorIds.length > 0 ? allowedMutatorIds.join(', ') : 'none'}.`,
    `Current shipped candidate catalog ids: ${currentCatalogIds.length > 0 ? currentCatalogIds.join(', ') : 'none'}.`,
    `Built-in structured mutator operators in the framework: ${operatorMutatorIds}.`,
    ...(requestedSpecLines.length > 0 ? ['', ...requestedSpecLines] : []),
    'Code-change path: add the new candidate/spec in the profile catalog, extend the mutation operator layer only if the new mutator needs a new operator kind, then update the profile contract and focused tests before rerunning.',
    rerunLine,
    `Routing: ${routingReason}`,
  ].join('\n');
}

function summarizeForecastLabMutators(profile: ForecastLabProfile): ForecastLabMutatorListProfileSummary {
  return {
    profileId: profile.id,
    targetSubsystem: profile.targetSubsystem,
    mutationMode: profile.mutation.mode,
    currentCatalogIds: profile.mutation.mode === 'structured'
      ? listForecastLabStructuredMutations(profile.id).map((candidate) => candidate.id)
      : [],
    allowedOperatorIds: profile.mutation.mode === 'structured'
      ? [...profile.mutation.allowedMutatorIds]
      : [],
  } satisfies ForecastLabMutatorListProfileSummary;
}

function buildMarkdownTable(
  headers: readonly string[],
  rows: readonly (readonly string[])[],
): string {
  const normalizeCell = (value: string | undefined): string => {
    const normalized = (value ?? '—').replaceAll('|', '\\|').replaceAll('\n', ' ').trim();
    return normalized.length > 0 ? normalized : '—';
  };

  return [
    `| ${headers.map((header) => normalizeCell(header)).join(' | ')} |`,
    `| ${headers.map(() => '---').join(' | ')} |`,
    ...rows.map((row) => `| ${headers.map((_, index) => normalizeCell(row[index])).join(' | ')} |`),
  ].join('\n');
}

function buildIndexedListRows(values: readonly string[]): string[][] {
  if (values.length === 0) {
    return [['—', 'none']];
  }

  return values.map((value, index) => [String(index + 1), value]);
}

function buildBatchReplayAnswer(
  profile: ForecastLabProfile,
  results: readonly ForecastLabBatchReplayMutatorResult[],
): string {
  const rows = results.map((result) => {
    const metricsStr = result.metrics && result.metrics.length > 0
      ? result.metrics.map((m) => `${m.horizon}: ${formatSignedDelta(m.deltaDirAcc, 'pp')}`).join(', ')
      : '—';
    return [
      result.mutatorId,
      result.decision.toUpperCase(),
      result.behaviorSummary,
      result.reason,
      metricsStr,
    ];
  });

  return [
    `Forecast-lab batch replay for ${profile.id}.`,
    '',
    `Baseline: shipped defaults (fresh runs with no parent lineage)`,
    `Replayed: ${results.length} mutator${results.length === 1 ? '' : 's'}`,
    '',
    buildMarkdownTable(
      ['Mutator ID', 'Verdict', 'Behavior', 'Reason', 'Horizon Deltas'],
      rows,
    ),
    '',
    'These runs were replayed from shipped defaults, not current kept lineage.',
  ].join('\n');
}

function buildIterativeImproveAnswer(
  profile: ForecastLabProfile,
  result: ForecastLabProfileImprovementLoopResult,
): string {
  const best = result.bestResult;
  const historyLine = result.history.length > 0
    ? result.history.map((entry) => `${entry.mutation.id} (${entry.keepSatisfied}/${entry.keepTotal} keep checks, objective ${entry.objectiveScore.toFixed(3)})`).join(' -> ')
    : 'no trial improved on the best shipped seed';

  return [
    `Forecast-lab iterative improvement loop for ${profile.id}.`,
    '',
    `Baseline: shipped defaults on ${profile.id}.`,
    `Best trial: ${best.mutation.id}`,
    `Decision if promoted today: ${best.decision.toUpperCase()}`,
    `Gate progress: ${best.keepSatisfied}/${best.keepTotal} keep checks satisfied`,
    `Objective score: ${best.objectiveScore.toFixed(3)}`,
    `Primary score: ${best.primaryScore.toFixed(3)}`,
    `Iterations run: ${result.iterationsRun}`,
    `Loop history: ${historyLine}.`,
    '',
    buildMarkdownTable(
      ['Horizon', 'Baseline DirAcc', 'Best DirAcc', 'Baseline Brier', 'Best Brier'],
      ['h1', 'h2', 'h3', 'h7', 'h14'].map((horizon) => {
        const baseline = result.baselineMetrics[horizon as keyof ForecastLabImprovementMetrics];
        const candidate = best.metrics[horizon as keyof ForecastLabImprovementMetrics];
        return [
          horizon,
          `${(baseline.directionalAccuracy * 100).toFixed(1)}%`,
          `${(candidate.directionalAccuracy * 100).toFixed(1)}%`,
          baseline.brierScore.toFixed(4),
          candidate.brierScore.toFixed(4),
        ];
      }),
    ),
    '',
    'This loop is bounded and diagnostic-only: it searches around the current shipped mutator space but does not auto-promote or rewrite the catalog.',
  ].join('\n');
}

function resolveMutatorListProfiles(
  input: ForecastLabRunToolInput,
): { profiles: readonly ForecastLabProfile[]; routingReason: string } | { error: string } {
  if (input.profileId?.trim()) {
    return {
      profiles: [getForecastLabProfile(input.profileId)],
      routingReason: 'Profile supplied explicitly.',
    };
  }

  const query = input.query?.trim();
  if (!query) {
    return {
      profiles: listForecastLabProfiles().filter((profile) => profile.mutation.mode === 'structured'),
      routingReason: 'No profile or query supplied. Returning every structured forecast-lab profile.',
    };
  }

  const explicitProfileId = extractExplicitProfileId(query);
  if (explicitProfileId) {
    return {
      profiles: [getForecastLabProfile(explicitProfileId)],
      routingReason: 'Profile supplied explicitly.',
    };
  }

  const route = routeForecastLabQuery(query);
  if (route.preferredProfileId) {
    return {
      profiles: [getForecastLabProfile(route.preferredProfileId)],
      routingReason: route.reasons.join(' '),
    };
  }

  const structuredProfiles = listForecastLabProfiles().filter((profile) => profile.mutation.mode === 'structured');
  if (structuredProfiles.length === 0) {
    return {
      error: 'No structured forecast-lab profiles are available.',
    };
  }

  return {
    profiles: structuredProfiles,
    routingReason: 'No unique forecast-lab profile could be inferred from the mutator-list request. Returning every structured profile instead of guessing.',
  };
}

function buildMutatorListAnswer(
  profiles: readonly ForecastLabProfile[],
  routingReason: string,
): string {
  const summaries = profiles.map((profile) => summarizeForecastLabMutators(profile));
  const frameworkOperatorIds = listForecastLabMutatorIds();
  const dryRunProfiles = listForecastLabProfiles()
    .filter((profile) => profile.mutation.mode !== 'structured')
    .map((profile) => profile.id);

  if (summaries.length === 1) {
    const summary = summaries[0]!;
    return [
      `Forecast-lab shipped mutator ids for ${summary.profileId}.`,
      '',
      buildMarkdownTable(
        ['Field', 'Value'],
        [
          ['Target subsystem', summary.targetSubsystem],
          ['Current mutation mode', summary.mutationMode],
          ['Allowed structured operator ids', summary.allowedOperatorIds.length > 0 ? summary.allowedOperatorIds.join(', ') : 'none'],
          ['Built-in structured mutator operators', frameworkOperatorIds.join(', ')],
          ['Dry-run-only profiles today', dryRunProfiles.length > 0 ? dryRunProfiles.join(', ') : 'none'],
          ['Routing', routingReason],
        ],
      ),
      '',
      'Shipped candidate catalog ids:',
      buildMarkdownTable(['#', 'Mutator id'], buildIndexedListRows(summary.currentCatalogIds)),
    ].join('\n');
  }

  return [
    'Forecast-lab shipped mutator catalog summary.',
    '',
    buildMarkdownTable(
      ['Profile id', 'Target subsystem', 'Mutation mode', 'Shipped ids', 'Allowed operators'],
      summaries.map((summary) => [
        summary.profileId,
        summary.targetSubsystem,
        summary.mutationMode,
        String(summary.currentCatalogIds.length),
        summary.allowedOperatorIds.length > 0 ? summary.allowedOperatorIds.join(', ') : 'none',
      ]),
    ),
    '',
    ...summaries.flatMap((summary) => [
      `Shipped candidate catalog ids for ${summary.profileId}:`,
      buildMarkdownTable(['#', 'Mutator id'], buildIndexedListRows(summary.currentCatalogIds)),
      '',
    ]),
    buildMarkdownTable(
      ['Field', 'Value'],
      [
        ['Built-in structured mutator operators', frameworkOperatorIds.join(', ')],
        ['Dry-run-only profiles today', dryRunProfiles.length > 0 ? dryRunProfiles.join(', ') : 'none'],
        ['Routing', routingReason],
      ],
    ),
    '',
    'If you want one profile only, specify the profile id in the request.',
  ].join('\n');
}

function resolveScorecardProfile(
  input: ForecastLabRunToolInput,
): { profile: ForecastLabProfile; routingReason: string } | { error: string } {
  if (input.profileId?.trim()) {
    const profile = getForecastLabProfile(input.profileId);
    if (profile.mutation.mode !== 'structured') {
      return {
        error: `Profile "${profile.id}" does not have structured mutations enabled.`,
      };
    }
    return {
      profile,
      routingReason: 'Profile supplied explicitly.',
    };
  }

  const query = input.query?.trim();
  if (query) {
    const explicitProfileId = extractExplicitProfileId(query);
    if (explicitProfileId) {
      const profile = getForecastLabProfile(explicitProfileId);
      if (profile.mutation.mode !== 'structured') {
        return {
          error: `Profile "${profile.id}" does not have structured mutations enabled.`,
        };
      }
      return {
        profile,
        routingReason: 'Profile supplied explicitly.',
      };
    }

    const route = routeForecastLabQuery(query);
    if (route.preferredProfileId) {
      const profile = getForecastLabProfile(route.preferredProfileId);
      if (profile.mutation.mode !== 'structured') {
        return {
          error: `Profile "${profile.id}" does not have structured mutations enabled.`,
        };
      }
      return {
        profile,
        routingReason: route.reasons.join(' '),
      };
    }
  }

  const structuredProfiles = listForecastLabProfiles().filter((profile) => profile.mutation.mode === 'structured');
  if (structuredProfiles.length === 0) {
    return {
      error: 'No structured forecast-lab profiles are available.',
    };
  }

  if (structuredProfiles.length === 1) {
    return {
      profile: structuredProfiles[0]!,
      routingReason: 'Only one structured profile exists.',
    };
  }

  return {
    error: `Multiple structured forecast-lab profiles exist (${structuredProfiles.map((p) => p.id).join(', ')}). Please specify profileId.`,
  };
}

function getMutatorStatusLabel(mutator: ForecastLabRankedMutator): string {
  if (!mutator.applicable) {
    return 'inapplicable';
  }
  if (!mutator.unused) {
    return 'already applied';
  }
  return 'available';
}

function buildMutatorScorecardAnswer(
  profile: ForecastLabProfile,
  totalStructuredRuns: number,
  rankedMutators: readonly ForecastLabRankedMutator[],
  routingReason: string,
): string {
  const mutatorRows = rankedMutators.map((mutator) => [
    mutator.id,
    getMutatorStatusLabel(mutator),
    mutator.mutatorId,
    String(mutator.attempts),
    String(mutator.keptRuns),
    String(mutator.regressedRuns),
    mutator.health,
    String(mutator.score),
  ]);

  return [
    `Forecast-lab mutator scorecard for ${profile.id}.`,
    '',
    `Profile: ${profile.id} | Total structured runs: ${totalStructuredRuns}`,
    '',
    buildMarkdownTable(
      ['Mutator ID', 'Status', 'Behavior', 'Attempts', 'Kept', 'Regressed', 'Health', 'Score'],
      mutatorRows,
    ),
    '',
    `Routing: ${routingReason}`,
  ].join('\n');
}

function extractErrorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function splitCommaList(text: string | undefined): string[] {
  return (text ?? '')
    .split(',')
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

function splitNextActions(text: string | undefined): string[] {
  return (text ?? '')
    .split(/,\s+|\s+or\s+/)
    .map((part) => part.trim())
    .filter((part) => part.length > 0);
}

function formatStructuredMutationApplicabilityError(
  action: ForecastLabRunToolInput['action'],
  message: string,
): string | null {
  const requestedMutatorMatch = message.match(
    /^Forecast-lab mutator "([^"]+)" is not applicable after replaying the kept parent lineage for profile "([^"]+)"\.\s*(.*)$/s,
  );
  const exhaustedMatch = message.match(
    /^No shipped structured mutator remains applicable after replaying the kept parent lineage for profile "([^"]+)"\.\s*(.*)$/s,
  );

  const requestedMutatorId = requestedMutatorMatch?.[1];
  const profileId = requestedMutatorMatch?.[2] ?? exhaustedMatch?.[1];
  const remainder = requestedMutatorMatch?.[3] ?? exhaustedMatch?.[2];
  if (!profileId || !remainder) {
    return null;
  }

  const appliedCandidateIds = splitCommaList(
    remainder.match(/Current kept lineage already applied: ([^.]+)\./)?.[1],
  );
  const inapplicableCandidateIds = splitCommaList(
    remainder.match(/Remaining shipped mutators checked and found inapplicable: ([^.]+)\./)?.[1],
  );
  const alternativeCandidateIds = splitCommaList(
    remainder.match(/Try one of the remaining applicable shipped mutators instead: ([^.]+)\./)?.[1],
  );
  const nextActions = splitNextActions(
    remainder.match(/Next actions: ([^.]+)\./)?.[1],
  );
  const requestedMutatorState = requestedMutatorId
    ? appliedCandidateIds.includes(requestedMutatorId)
      ? 'Already applied in the current kept lineage'
      : 'Not applicable in the current kept lineage state'
    : 'No unused shipped mutator remains applicable';
  const catalogStatus = exhaustedMatch
    ? 'Shipped structured mutator catalog exhausted for this lineage'
    : alternativeCandidateIds.length > 0
      ? 'Requested mutator rejected, but other shipped options remain'
      : 'Requested mutator rejected and no shipped fallback remains';

  return [
    `Forecast-lab ${action} could not continue.`,
    '',
    buildMarkdownTable(
      ['Field', 'Value'],
      [
        ['Profile', profileId],
        ...(requestedMutatorId ? [['Requested mutator', requestedMutatorId]] : []),
        ['Mutator status', requestedMutatorState],
        ['Catalog status', catalogStatus],
      ],
    ),
    ...(appliedCandidateIds.length > 0
      ? [
          '',
          'Already applied in the kept lineage:',
          buildMarkdownTable(['#', 'Mutator id'], buildIndexedListRows(appliedCandidateIds)),
        ]
      : []),
    ...(inapplicableCandidateIds.length > 0
      ? [
          '',
          'Remaining shipped mutators checked and found inapplicable:',
          buildMarkdownTable(['#', 'Mutator id'], buildIndexedListRows(inapplicableCandidateIds)),
        ]
      : []),
    ...(alternativeCandidateIds.length > 0
      ? [
          '',
          'Remaining applicable shipped mutators:',
          buildMarkdownTable(['#', 'Mutator id'], buildIndexedListRows(alternativeCandidateIds)),
        ]
      : []),
    ...(nextActions.length > 0
      ? [
          '',
          'Next actions:',
          ...nextActions.map((step, index) => `${index + 1}. ${step.charAt(0).toUpperCase()}${step.slice(1)}`),
        ]
      : []),
  ].join('\n');
}

function buildErrorPayload(action: ForecastLabRunToolInput['action'], message: string): string {
  const formattedAnswer = formatStructuredMutationApplicabilityError(action, message)
    ?? `Forecast-lab ${action} failed: ${message}`;
  return formatToolResult({
    _tool: 'forecast_lab_run',
    action,
    status: 'error',
    error: message,
    answer: formattedAnswer,
  } satisfies ForecastLabErrorPayload);
}

function isPromotableLedgerEntry(entry: ForecastLabLedgerEntry): boolean {
  return entry.decision === 'keep'
    && entry.mutationMode === 'structured'
    && (entry.promotion?.status === 'approval-required' || entry.promotion?.status === 'approved');
}

function resolvePromotionSelection(
  input: ForecastLabRunToolInput,
  ledgerEntries: readonly ForecastLabLedgerEntry[],
): { profileId: string; sourceRunId?: string } | { error: string } {
  if (input.profileId?.trim()) {
    const profileId = getForecastLabProfile(input.profileId).id;
    return {
      profileId,
      ...(input.sourceRunId ? { sourceRunId: input.sourceRunId } : {}),
    };
  }

  if (input.sourceRunId?.trim()) {
    const match = ledgerEntries.findLast((entry) => entry.runId === input.sourceRunId && isPromotableLedgerEntry(entry));
    if (!match) {
      return {
        error: `No promotable kept structured run matched sourceRunId "${input.sourceRunId}".`,
      };
    }

    return {
      profileId: match.profileId,
      sourceRunId: match.runId,
    };
  }

  const promotable = ledgerEntries.filter(isPromotableLedgerEntry);
  if (promotable.length === 0) {
    return {
      error: 'No approval-required forecast-lab promotion source is available.',
    };
  }

  const uniqueProfiles = new Set(promotable.map((entry) => entry.profileId));
  if (uniqueProfiles.size > 1) {
    return {
      error: `Multiple promotable forecast-lab runs are waiting for approval. Specify profileId or sourceRunId. Candidates: ${promotable.map((entry) => `${entry.profileId}:${entry.runId}`).join(', ')}`,
    };
  }

  const latest = promotable[promotable.length - 1]!;
  return {
    profileId: latest.profileId,
    sourceRunId: latest.runId,
  };
}

function resolveGuidedImproveProfile(
  input: ForecastLabRunToolInput,
): { profile: ForecastLabProfile; routingReason: string } | { error: string } {
  if (input.profileId?.trim()) {
    return {
      profile: getForecastLabProfile(input.profileId),
      routingReason: 'Profile supplied explicitly.',
    };
  }

  const query = input.query?.trim();
  if (!query) {
    return {
      error: `guided-improve requires profileId or query. Available profiles: ${profileListLabel()}`,
    };
  }

  const route = routeForecastLabQuery(query);
  if (route.intent !== 'improvement' || !route.preferredProfileId) {
    return {
      error: `Could not resolve a unique forecast-lab profile from the query. Available profiles: ${profileListLabel()}`,
    };
  }

  return {
    profile: getForecastLabProfile(route.preferredProfileId),
    routingReason: route.reasons.join(' '),
  };
}

function buildRoutingContext(
  input: ForecastLabRunToolInput,
  profile: ForecastLabProfile,
  routingReason: string,
): ForecastLabRunOptions['routingContext'] | undefined {
  if (!input.query?.trim()) {
    return undefined;
  }

  const invocationSource: ForecastLabRoutingInvocationSource = input.routingSource ?? 'manual-request';
  return {
    originatingQuery: input.query,
    selectedProfileId: profile.id,
    routerReason: routingReason,
    invocationSource,
  };
}

export function parseForecastLabRunToolPayload(result: string): ForecastLabRunToolPayload | null {
  try {
    const parsed = JSON.parse(result) as { data?: unknown };
    const data = parsed?.data;
    if (!data || typeof data !== 'object') {
      return null;
    }

    const payload = data as Record<string, unknown>;
    if (payload['_tool'] !== 'forecast_lab_run') {
      return null;
    }

    return payload as unknown as ForecastLabRunToolPayload;
  } catch {
    return null;
  }
}

export function extractForecastLabRunToolAnswer(result: string): string | null {
  const payload = parseForecastLabRunToolPayload(result);
  return payload && typeof payload.answer === 'string' ? payload.answer : null;
}

export function createForecastLabRunTool({
  runForecastLabFn = runForecastLab,
  runForecastLabImprovementLoopFn = runForecastLabProfileImprovementLoop,
  promoteForecastLabFn = promoteForecastLab,
  resetForecastLabFn = resetForecastLab,
  readLedgerEntriesFn = readLedgerEntries,
  getLedgerPathFn = () => getExperimentLedgerPath({ create: true }),
  findLatestKeptLedgerEntryFn = findLatestKeptLedgerEntry,
  readRunManifestFn = readRunManifest,
  readTextFileFn = (path: string) => readFileSync(path, 'utf8'),
  existsSyncFn = existsSync,
}: CreateForecastLabRunToolDeps = {}): DynamicStructuredTool {
  /** Creates the forecast lab execution tool. */
  return new DynamicStructuredTool({
    name: 'forecast_lab_run',
    description: 'Execute the bounded forecast-lab improvement, comparison, promotion, or reset workflow.',
    schema: forecastLabRunSchema,
    func: async (input, _runManager, config?: RunnableConfig) => {
      const onProgress = config?.metadata?.onProgress as ((message: string) => void) | undefined;

      try {
        if (input.action === 'guided-improve') {
          const resolved = resolveGuidedImproveProfile(input);
          if ('error' in resolved) {
            return buildErrorPayload(input.action, resolved.error);
          }

          const { profile, routingReason } = resolved;
          if (input.execute === false) {
            return formatToolResult({
              _tool: 'forecast_lab_run',
              action: 'guided-improve',
              status: 'ok',
              execute: false,
              profileId: profile.id,
              allowedGlobs: [...profile.allowedGlobs],
              baselineCommands: commandStrings(profile.baselineCommands),
              candidateCommands: commandStrings(profile.candidateCommands),
              mutationMode: profile.mutation.mode,
              promotionReady: false,
              answer: buildPlanAnswer(profile, routingReason, input.mutator),
            } satisfies ForecastLabPlanPayload);
          }

          if (input.mutator && profile.mutation.mode !== 'structured') {
            return buildErrorPayload(
              input.action,
              `Profile "${profile.id}" does not support structured mutator overrides.`,
            );
          }

          const result = await runForecastLabFn({
            profileId: profile.id,
            ...(profile.mutation.mode === 'structured'
              ? {
                  mutationMode: 'structured',
                  ...(input.mutator ? { mutator: input.mutator } : {}),
                  ...(input.rankMutators !== undefined ? { rankMutators: input.rankMutators } : {}),
                }
              : { dryRun: true }),
            routingContext: buildRoutingContext(input, profile, routingReason),
            progress: onProgress,
          });
          const promotionStatus = result.ledgerEntry.promotion?.status;

          return formatToolResult({
            _tool: 'forecast_lab_run',
            action: 'guided-improve',
            status: 'ok',
            execute: true,
            profileId: profile.id,
            runId: result.runId,
            decision: result.decision.decision,
            reason: result.decision.reason,
            allowedGlobs: [...profile.allowedGlobs],
            artifactsPath: result.manifest.artifactsPath,
            promotionReady: promotionStatus === 'approval-required',
            ...(promotionStatus ? { promotionStatus } : {}),
            ...(promotionStatus === 'approval-required' ? { sourceRunId: result.runId } : {}),
            answer: buildGuidedImproveAnswer(profile, result, input.mutator?.trim() || undefined),
          } satisfies ForecastLabGuidedImprovePayload);
        }

        if (input.action === 'compare-best-vs-shipped') {
          const ledgerPath = getLedgerPathFn();
          const ledgerEntries = readLedgerEntriesFn(ledgerPath);
          const resolved = resolveComparisonProfile(input, ledgerEntries);
          if ('error' in resolved) {
            return buildErrorPayload(input.action, resolved.error);
          }

          const { profile } = resolved;
          const requestedMutationId = input.mutationId?.trim() ?? extractComparisonMutationId(input.query);
          if (requestedMutationId) {
            const requestedEntry = findLatestKeptStructuredMutationEntry(ledgerEntries, profile.id, requestedMutationId);
            if (!requestedEntry) {
              return buildErrorPayload(
                input.action,
                `No kept structured run matched mutationId "${requestedMutationId}" for profile "${profile.id}".`,
              );
            }

            const { sourceRunId: activeSourceRunId } = readActiveSourceRunId(profile.id, existsSyncFn, readTextFileFn);
            if (!activeSourceRunId) {
              return buildErrorPayload(
                input.action,
                `No active promoted baseline is recorded for profile "${profile.id}", so a mutator-vs-active comparison is not available yet.`,
              );
            }

            const activeEntry = findLedgerEntryByRunId(ledgerEntries, activeSourceRunId);
            if (!activeEntry) {
              return buildErrorPayload(
                input.action,
                `Active promotion source run "${activeSourceRunId}" for profile "${profile.id}" was not found in the forecast-lab ledger.`,
              );
            }

            const requestedCandidateArtifact = readJsonObject(join(requestedEntry.artifactsPath, 'candidate.json'), readTextFileFn);
            const activeCandidateArtifact = requestedEntry.runId === activeEntry.runId
              ? requestedCandidateArtifact
              : readJsonObject(join(activeEntry.artifactsPath, 'candidate.json'), readTextFileFn);
            const requestedCandidateStdout = extractComparisonStdout(requestedCandidateArtifact);
            const activeCandidateStdout = extractComparisonStdout(activeCandidateArtifact);
            const comparisons = typeof activeCandidateStdout === 'string' && typeof requestedCandidateStdout === 'string'
              ? buildMetricComparisons(activeCandidateStdout, requestedCandidateStdout)
              : [];

            return formatToolResult({
              _tool: 'forecast_lab_run',
              action: 'compare-best-vs-shipped',
              status: 'ok',
              profileId: profile.id,
              comparisonTarget: 'active-live',
              sourceRunId: requestedEntry.runId,
              activeSourceRunId,
              decision: requestedEntry.decision,
              reason: requestedEntry.reason,
              artifactsPath: requestedEntry.artifactsPath,
              liveStatus: requestedEntry.runId === activeEntry.runId ? 'already-live' : 'not-live',
              mutationId: requestedEntry.mutationId,
              activeMutationId: activeEntry.mutationId,
              ...(requestedEntry.mutationSummary ? { mutationSummary: requestedEntry.mutationSummary } : {}),
              ...(comparisons.length > 0 ? { metrics: comparisons } : {}),
              answer: buildMutatorVsActiveAnswer(profile, requestedEntry, activeEntry, comparisons),
            } satisfies ForecastLabComparePayload);
          }

          const latestKeptEntry = getLatestKeptStructuredEntry(profile.id, ledgerPath, findLatestKeptLedgerEntryFn);
          if (!latestKeptEntry) {
            return formatToolResult({
              _tool: 'forecast_lab_run',
              action: 'compare-best-vs-shipped',
              status: 'error',
              error: `No kept structured run is recorded yet for profile "${profile.id}".`,
              answer: buildNoCurrentBestCompareAnswer(profile),
            } satisfies ForecastLabErrorPayload);
          }
          const manifestPath = latestKeptEntry.promotion?.source.manifestPath ?? getExperimentRunManifestPath(latestKeptEntry.runId);
          const manifest = readRunManifestFn(manifestPath);
          const artifactsPath = latestKeptEntry.artifactsPath;
          const baselineArtifact = readJsonObject(join(artifactsPath, 'baseline.json'), readTextFileFn);
          const candidateArtifact = readJsonObject(join(artifactsPath, 'candidate.json'), readTextFileFn);
          const { sourceRunId: activeSourceRunId } = readActiveSourceRunId(profile.id, existsSyncFn, readTextFileFn);
          const liveStatus: ForecastLabCompareLiveStatus = activeSourceRunId === latestKeptEntry.runId
            ? 'already-live'
            : latestKeptEntry.decision === 'keep'
              ? 'ready-to-promote'
              : 'not-live';
          const promotionCommand = liveStatus === 'already-live'
            ? undefined
            : `Approve forecast-lab promotion for ${profile.id} run ${latestKeptEntry.runId}.`;
          const baselineStdout = Array.isArray(baselineArtifact.commands)
            ? extractComparisonStdout(baselineArtifact)
            : undefined;
          const candidateStdout = Array.isArray(candidateArtifact.commands)
            ? extractComparisonStdout(candidateArtifact)
            : undefined;
          const comparisons = typeof baselineStdout === 'string' && typeof candidateStdout === 'string'
            ? buildMetricComparisons(baselineStdout, candidateStdout)
            : [];

          return formatToolResult({
            _tool: 'forecast_lab_run',
            action: 'compare-best-vs-shipped',
            status: 'ok',
            profileId: profile.id,
            comparisonTarget: 'shipped-baseline',
            sourceRunId: latestKeptEntry.runId,
            decision: latestKeptEntry.decision,
            reason: latestKeptEntry.reason,
            artifactsPath,
            liveStatus,
            ...(promotionCommand ? { promotionCommand } : {}),
            ...(manifest.mutationId ? { mutationId: manifest.mutationId } : {}),
            ...(manifest.mutationSummary ? { mutationSummary: manifest.mutationSummary } : {}),
            ...(comparisons.length > 0 ? { metrics: comparisons } : {}),
            answer: buildCompareAnswer(profile, latestKeptEntry, liveStatus, promotionCommand, comparisons),
          } satisfies ForecastLabComparePayload);
        }

        if (input.action === 'list-mutators') {
          const resolved = resolveMutatorListProfiles(input);
          if ('error' in resolved) {
            return buildErrorPayload(input.action, resolved.error);
          }

          const profiles = resolved.profiles.map((profile) => summarizeForecastLabMutators(profile));
          const dryRunProfiles = listForecastLabProfiles()
            .filter((profile) => profile.mutation.mode !== 'structured')
            .map((profile) => profile.id);

          return formatToolResult({
            _tool: 'forecast_lab_run',
            action: 'list-mutators',
            status: 'ok',
            ...(profiles.length === 1 ? { profileId: profiles[0]!.profileId } : {}),
            profiles,
            dryRunProfiles,
            frameworkOperatorIds: listForecastLabMutatorIds(),
            answer: buildMutatorListAnswer(resolved.profiles, resolved.routingReason),
          } satisfies ForecastLabMutatorListPayload);
        }

        if (input.action === 'catalog-extension-plan') {
          const { profile, routingReason } = resolveCatalogExtensionProfile(input);
          const allowedMutatorIds = profile?.mutation.mode === 'structured'
            ? [...profile.mutation.allowedMutatorIds]
            : undefined;
          const currentCatalogIds = profile?.mutation.mode === 'structured'
            ? listForecastLabStructuredMutations(profile.id).map((candidate) => candidate.id)
            : undefined;
          const catalogFiles = [...listCatalogExtensionCatalogFiles(profile)];
          const validationFiles = [...listCatalogExtensionValidationFiles(profile)];
          const requestedMutatorId = extractCatalogExtensionMutatorId(input.query);
          const requestedParameterChanges = [...extractCatalogExtensionParameterChanges(input.query)];

          return formatToolResult({
            _tool: 'forecast_lab_run',
            action: 'catalog-extension-plan',
            status: 'ok',
            ...(profile ? {
              profileId: profile.id,
              targetSubsystem: profile.targetSubsystem,
              allowedGlobs: [...profile.allowedGlobs],
              mutationMode: profile.mutation.mode,
            } : {}),
            ...(allowedMutatorIds ? { allowedMutatorIds } : {}),
            ...(currentCatalogIds ? { currentCatalogIds } : {}),
            catalogFiles,
            validationFiles,
            ...(requestedMutatorId ? { requestedMutatorId } : {}),
            ...(requestedParameterChanges.length > 0 ? { requestedParameterChanges } : {}),
            operatorMutatorIds: [...listForecastLabMutatorIds()],
            answer: buildCatalogExtensionAnswer(profile, routingReason, input.query),
          } satisfies ForecastLabCatalogExtensionPayload);
        }

        if (input.action === 'mutator-scorecard') {
          const resolved = resolveScorecardProfile(input);
          if ('error' in resolved) {
            return buildErrorPayload(input.action, resolved.error);
          }

          const { profile, routingReason } = resolved;
          const ledgerPath = getLedgerPathFn();
          const ledgerEntries = readLedgerEntriesFn(ledgerPath);
          const catalog = listForecastLabStructuredMutations(profile.id);

          const ranking = rankForecastLabMutators({
            profileId: profile.id,
            catalog,
            ledgerEntries,
          });

          return formatToolResult({
            _tool: 'forecast_lab_run',
            action: 'mutator-scorecard',
            status: 'ok',
            profileId: profile.id,
            totalStructuredRuns: ranking.totalStructuredRuns,
            rankedMutators: ranking.rankedMutators.map((mutator) => ({
              id: mutator.id,
              mutatorId: mutator.mutatorId,
              applicable: mutator.applicable,
              unused: mutator.unused,
              attempts: mutator.attempts,
              keptRuns: mutator.keptRuns,
              regressedRuns: mutator.regressedRuns,
              health: mutator.health,
              score: mutator.score,
            })),
            answer: buildMutatorScorecardAnswer(profile, ranking.totalStructuredRuns, ranking.rankedMutators, routingReason),
          } satisfies ForecastLabMutatorScorecardPayload);
        }

        if (input.action === 'batch-replay-mutators') {
          if (!input.profileId?.trim()) {
            return buildErrorPayload(input.action, 'batch-replay-mutators requires profileId.');
          }

          const profile = getForecastLabProfile(input.profileId);
          if (profile.mutation.mode !== 'structured') {
            return buildErrorPayload(
              input.action,
              `Profile "${profile.id}" does not support structured mutation replay.`,
            );
          }

          const catalog = listForecastLabStructuredMutations(profile.id);
          const limit = input.limit ?? 5;
          const mutatorsToReplay = catalog.slice(0, limit);

          if (mutatorsToReplay.length === 0) {
            return buildErrorPayload(
              input.action,
              `Profile "${profile.id}" has no shipped mutators to replay.`,
            );
          }

          const results: ForecastLabBatchReplayMutatorResult[] = [];

          for (const mutator of mutatorsToReplay) {
            try {
              onProgress?.(`batch-replay: running ${mutator.id}`);
              const result = await runForecastLabFn({
                profileId: profile.id,
                mutationMode: 'structured',
                mutator: mutator.id,
                forceNoParent: true,
                diagnosticOnly: true,
                progress: onProgress,
              });

              const baselineArtifact = readJsonObject(join(result.manifest.artifactsPath, 'baseline.json'), readTextFileFn);
              const candidateArtifact = readJsonObject(join(result.manifest.artifactsPath, 'candidate.json'), readTextFileFn);
              const baselineStdout = extractComparisonStdout(baselineArtifact);
              const candidateStdout = extractComparisonStdout(candidateArtifact);
              const comparisons = typeof baselineStdout === 'string' && typeof candidateStdout === 'string'
                ? buildMetricComparisons(baselineStdout, candidateStdout)
                : [];

              results.push({
                mutatorId: mutator.id,
                decision: result.decision.decision,
                reason: result.decision.reason,
                behaviorSummary: mutator.specSummary.summary,
                ...(comparisons.length > 0 ? { metrics: comparisons } : {}),
              });
            } catch (error) {
              const errorMessage = extractErrorMessage(error);
              results.push({
                mutatorId: mutator.id,
                decision: 'drop',
                reason: `error: ${errorMessage}`,
                behaviorSummary: mutator.specSummary.summary,
              });
            }
          }

          return formatToolResult({
            _tool: 'forecast_lab_run',
            action: 'batch-replay-mutators',
            status: 'ok',
            profileId: profile.id,
            baselineDescription: 'shipped defaults (fresh runs with no parent lineage)',
            replayedCount: results.length,
            results,
            answer: buildBatchReplayAnswer(profile, results),
          } satisfies ForecastLabBatchReplayPayload);
        }

        if (input.action === 'iterative-improve-mutators') {
          if (!input.profileId?.trim()) {
            return buildErrorPayload(input.action, 'iterative-improve-mutators requires profileId.');
          }

          const profile = getForecastLabProfile(input.profileId);
          const result = await runForecastLabImprovementLoopFn({
            profileId: profile.id,
            seedMutatorId: input.mutator?.trim() || undefined,
            maxIterations: input.iterations,
            progress: onProgress,
          });

          return formatToolResult({
            _tool: 'forecast_lab_run',
            action: 'iterative-improve-mutators',
            status: 'ok',
            profileId: profile.id,
            iterationsRun: result.iterationsRun,
            bestMutationId: result.bestResult.mutation.id,
            bestDecision: result.bestResult.decision,
            keepSatisfied: result.bestResult.keepSatisfied,
            keepTotal: result.bestResult.keepTotal,
            bestObjectiveScore: result.bestResult.objectiveScore,
            bestPrimaryScore: result.bestResult.primaryScore,
            baselineMetrics: result.baselineMetrics,
            bestMetrics: result.bestResult.metrics,
            answer: buildIterativeImproveAnswer(profile, result),
          } satisfies ForecastLabIterativeImprovePayload);
        }

        if (input.action === 'promote-approved') {
          const ledgerPath = getLedgerPathFn();
          const selection = resolvePromotionSelection(input, readLedgerEntriesFn(ledgerPath));
          if ('error' in selection) {
            return buildErrorPayload(input.action, selection.error);
          }

          const profile = getForecastLabProfile(selection.profileId);
          if (profile.mutation.mode !== 'structured') {
            return buildErrorPayload(
              input.action,
              `Profile "${profile.id}" does not have a structured promotion path.`,
            );
          }

          const result = await promoteForecastLabFn({
            profileId: profile.id,
            ...(selection.sourceRunId ? { sourceRunId: selection.sourceRunId } : {}),
            progress: onProgress,
          });

          return formatToolResult({
            _tool: 'forecast_lab_run',
            action: 'promote-approved',
            status: 'ok',
            profileId: profile.id,
            runId: result.runId,
            sourceRunId: result.sourceRunId,
            decision: 'keep',
            reason: result.decision.reason,
            artifactsPath: result.manifest.artifactsPath,
            activationArtifactsPath: result.activation.artifactsPath,
            activeStatePath: result.activeStatePath,
            answer: buildPromoteAnswer(profile, result),
          } satisfies ForecastLabPromotePayload);
        }

        if (!input.profileId?.trim()) {
          return buildErrorPayload(input.action, 'reset-live requires profileId.');
        }
        if (!input.resetMode) {
          return buildErrorPayload(input.action, 'reset-live requires resetMode ("defaults" or "last-known-good").');
        }

        const profile = getForecastLabProfile(input.profileId);
        const result = await resetForecastLabFn({
          profileId: profile.id,
          mode: input.resetMode,
          progress: onProgress,
        });

        return formatToolResult({
          _tool: 'forecast_lab_run',
          action: 'reset-live',
          status: 'ok',
          profileId: profile.id,
          resetMode: result.mode,
          runId: result.runId,
          artifactsPath: result.artifactsPath,
          resetArtifactPath: result.resetArtifactPath,
          ...(result.activeStatePath ? { activeStatePath: result.activeStatePath } : {}),
          answer: buildResetAnswer(profile, result),
        } satisfies ForecastLabResetPayload);
      } catch (error) {
        return buildErrorPayload(input.action, extractErrorMessage(error));
      }
    },
  });
}

export const forecastLabRunTool = createForecastLabRunTool();
