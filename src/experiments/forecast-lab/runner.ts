import { spawn, spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readFileSync, unlinkSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve, sep } from 'node:path';
import {
  getExperimentLedgerPath,
  getExperimentRunDir,
  getExperimentRunManifestPath,
  getExperimentsDir,
} from '../../utils/paths.js';
import { loadConfig, type Config } from '../../utils/config.js';
import { getEnvironment } from '../../utils/env.js';
import {
  appendLedgerEntry,
  findLatestKeptLedgerEntry,
  readLedgerEntries,
  readRunManifest,
  stableJsonStringify,
  writeRunManifest,
} from './ledger.js';
import { updateForecastLabRoutingStats } from './router-memory.js';
import {
  applyForecastLabCandidateEdits,
  prepareForecastLabCandidateWorkspace,
  prepareForecastLabPromotionWorkspace,
  withForecastLabCandidateWorkspace,
} from './git.js';
import { getForecastLabProfile, listForecastLabStructuredMutations } from './profiles.js';
import type { ForecastLabCommand, ForecastLabProfile } from './profiles.js';
import type {
  ForecastLabDecision,
  ForecastLabLedgerEntry,
  ForecastLabPromotionActivationRef,
  ForecastLabPromotionState,
  ForecastLabRoutingContext,
  ForecastLabRunManifest,
  JsonValue,
} from './types.js';
import {
  getForecastLabBaselineCommit,
  makeForecastLabCandidateBranch,
} from './git.js';
import type { ForecastLabMutationLineage, ForecastLabMutationMode } from './mutation.js';
import {
  isForecastLabMarkovMutatorProfileId,
  replayForecastLabMarkovParameterMutation,
  snapshotForecastLabMarkovParameterMutation,
  validateForecastLabMarkovParameterMutationReplayPayload,
} from './mutators/markov-parameters.js';
import type {
  ForecastLabMarkovParameterMutationCandidate,
  ForecastLabMarkovParameterMutationReplayPayload,
  ForecastLabMutationScalarValue,
} from './mutators/markov-parameters.js';
import {
  formatForecastLabMutatorHealthProgress,
  rankForecastLabMutators,
} from './mutator-ranker.js';
import type { ForecastLabMutatorRanking } from './mutator-ranker.js';
import {
  FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS,
  getForecastLabMarkovRuntimeDefaults,
  setForecastLabMarkovRuntimeDefaults,
} from '../../tools/finance/markov-distribution.js';
import {
  FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS,
  getForecastLabConformalRuntimeDefaults,
  setForecastLabConformalRuntimeDefaults,
} from '../../tools/finance/conformal.js';
import {
  FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS,
  getForecastLabRegimeCalibratorRuntimeDefaults,
  setForecastLabRegimeCalibratorRuntimeDefaults,
} from '../../tools/finance/regime-calibrator.js';
import type { ForecastLabRuntimeAssetScope as FinanceForecastLabRuntimeAssetScope } from '../../tools/finance/forecast-lab-runtime-defaults.js';

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
  readonly metrics?: JsonValue;
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
  readonly metrics?: JsonValue;
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
  readonly parentRunId?: string;
  readonly mutationId: string;
  readonly mutationSummary: string;
  readonly lineage: ForecastLabMutationLineage;
  readonly selectedMutatorId: string;
  readonly mutatorId: ForecastLabMarkovParameterMutationCandidate['mutatorId'];
  readonly mutatedFiles: readonly string[];
  readonly patchSummary: readonly string[];
  readonly mutationSpecSummary: ForecastLabMarkovParameterMutationCandidate['specSummary'];
  readonly mutationReplayPayload: ForecastLabMarkovParameterMutationReplayPayload;
  readonly mutatorRanking?: ForecastLabMutatorRanking;
}

interface ForecastLabStructuredMutationSeed {
  readonly parentRunId: string;
  readonly rootRunId: string;
  readonly generation: number;
  readonly replayedMutations: readonly ForecastLabMarkovParameterMutationCandidate[];
  readonly usedMutationIds: ReadonlySet<string>;
}

interface ForecastLabPromotionSourceSelection {
  readonly ledgerEntry: ForecastLabLedgerEntry;
  readonly manifestPath: string;
  readonly manifest: ForecastLabRunManifest;
  readonly replayPayload: ForecastLabMarkovParameterMutationReplayPayload;
}

export interface ForecastLabRunOptions {
  readonly profileId: string;
  readonly dryRun?: boolean;
  readonly skipMutation?: boolean;
  readonly mutationMode?: ForecastLabMutationMode;
  readonly keepWorktree?: boolean;
  readonly mutator?: string;
  readonly rankMutators?: boolean;
  readonly runId?: string;
  readonly forceNoParent?: boolean;
  readonly diagnosticOnly?: boolean;
  readonly now?: () => Date;
  readonly commandRunner?: ForecastLabCommandRunner;
  readonly ledgerPath?: string;
  readonly routingContext?: ForecastLabRoutingContext;
  readonly routingStatsPath?: string;
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

export interface ForecastLabPromotionOptions {
  readonly profileId: string;
  readonly sourceRunId?: string;
  readonly runId?: string;
  readonly now?: () => Date;
  readonly commandRunner?: ForecastLabCommandRunner;
  readonly ledgerPath?: string;
  readonly progress?: (message: string) => void;
  readonly output?: (chunk: string) => void;
}

export interface ForecastLabPromotionResult {
  readonly runId: string;
  readonly sourceRunId: string;
  readonly manifest: ForecastLabRunManifest;
  readonly sourceManifest: ForecastLabRunManifest;
  readonly baseline: JsonValue;
  readonly candidate: JsonValue;
  readonly decision: ForecastLabDecisionSummary;
  readonly activation: ForecastLabPromotionActivationRef;
  readonly activeStatePath: string;
}

export type ForecastLabResetMode = 'defaults' | 'last-known-good';

export interface ForecastLabResetOptions {
  readonly profileId: string;
  readonly mode: ForecastLabResetMode;
  readonly runId?: string;
  readonly now?: () => Date;
  readonly progress?: (message: string) => void;
}

export interface ForecastLabResetResult {
  readonly runId: string;
  readonly profileId: string;
  readonly mode: ForecastLabResetMode;
  readonly artifactsPath: string;
  readonly resetArtifactPath: string;
  readonly activeStatePath?: string;
}

export class ForecastLabRunnerError extends Error {
  override name = 'ForecastLabRunnerError';
}

function isJsonValue(value: unknown): value is JsonValue {
  if (
    value === null
    || typeof value === 'string'
    || typeof value === 'boolean'
  ) {
    return true;
  }
  if (typeof value === 'number') {
    return Number.isFinite(value);
  }
  if (Array.isArray(value)) {
    return value.every(isJsonValue);
  }
  if (typeof value === 'object' && value !== null) {
    return Object.values(value).every(isJsonValue);
  }
  return false;
}

function toJsonValue(value: unknown, context: string): JsonValue {
  if (!isJsonValue(value)) {
    throw new ForecastLabRunnerError(`${context} must be JSON-serializable`);
  }
  return value;
}

interface StructuredMutationCatalogState {
  readonly appliedCandidateIds: readonly string[];
  readonly applicableCandidateIds: readonly string[];
  readonly inapplicableCandidateIds: readonly string[];
}

type ForecastingConfig = Config['forecasting'];

const UNSAFE_SHELL_COMMAND_PATTERN = /[;&|`<>]|\$\(/;
const UNSAFE_GIT_COMMAND_PATTERN = /\bgit\s+(?:add|commit|push|reset|checkout|clean)\b/;
const SAFE_PROFILE_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9._-]*$/;
const FORECAST_LAB_ACTIVE_STATE_DIR = 'active-promotions';
const FORECAST_LAB_METRICS_PREFIX = 'FORECAST_LAB_METRICS ';

type ForecastLabMutableParameterDefaults = Record<string, ForecastLabMutationScalarValue>;

interface ForecastLabRuntimeDefaultsOverride {
  readonly filePath: string;
  readonly parameterId: string;
  readonly value: ForecastLabMutationScalarValue;
}

interface ForecastLabRuntimeDefaultsSnapshot {
  readonly assetScope: ForecastLabRuntimeAssetScope;
  readonly filePath: string;
  readonly overrides?: ForecastLabMutableParameterDefaults;
}

export type ForecastLabRuntimeAssetScope = FinanceForecastLabRuntimeAssetScope;

export interface ForecastLabRuntimeDefaultsActivation {
  readonly profileId: string;
  readonly assetScope: ForecastLabRuntimeAssetScope;
  readonly overrides: readonly ForecastLabRuntimeDefaultsOverride[];
}

export interface ForecastLabResolvedRuntimeDefaults {
  readonly markov: typeof FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS;
  readonly conformal: typeof FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS;
  readonly regime: typeof FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS;
}

interface ForecastLabActivePromotionSnapshot {
  readonly profileId: string;
  readonly sourceRunId: string;
  readonly promotionRunId: string;
  readonly activatedAt: string;
  readonly promotion: Extract<ForecastLabPromotionState, { status: 'activated' }>;
  readonly mutationReplayPayload: ForecastLabMarkovParameterMutationReplayPayload;
  readonly mutatedFiles: readonly string[];
  readonly patchSummary: readonly string[];
}

interface ForecastLabActivePromotionRecord extends ForecastLabActivePromotionSnapshot {
  readonly version: 1;
  readonly promotion: Extract<ForecastLabPromotionState, { status: 'activated' }>;
  readonly activeStatePath: string;
  readonly previousActive?: ForecastLabActivePromotionSnapshot;
}

const LIVE_PARAMETER_DEFAULT_TARGETS: Record<string, {
  readonly get: (assetScope: ForecastLabRuntimeAssetScope) => ForecastLabMutableParameterDefaults | undefined;
  readonly set: (assetScope: ForecastLabRuntimeAssetScope, overrides?: ForecastLabMutableParameterDefaults) => void;
}> = {
  'src/tools/finance/markov-distribution/core.ts': {
    get: (assetScope) => getForecastLabMarkovRuntimeDefaults(assetScope) as ForecastLabMutableParameterDefaults | undefined,
    set: (assetScope, overrides) => setForecastLabMarkovRuntimeDefaults(assetScope, overrides),
  },
  'src/tools/finance/conformal.ts': {
    get: (assetScope) => getForecastLabConformalRuntimeDefaults(assetScope) as ForecastLabMutableParameterDefaults | undefined,
    set: (assetScope, overrides) => setForecastLabConformalRuntimeDefaults(assetScope, overrides),
  },
  'src/tools/finance/regime-calibrator.ts': {
    get: (assetScope) => getForecastLabRegimeCalibratorRuntimeDefaults(assetScope) as ForecastLabMutableParameterDefaults | undefined,
    set: (assetScope, overrides) => setForecastLabRegimeCalibratorRuntimeDefaults(assetScope, overrides),
  },
};

const SHIPPED_PARAMETER_DEFAULT_TARGETS: Record<string, ForecastLabMutableParameterDefaults> = {
  'src/tools/finance/markov-distribution/core.ts': { ...FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS },
  'src/tools/finance/conformal.ts': { ...FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS },
  'src/tools/finance/regime-calibrator.ts': { ...FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS },
};

const STRUCTURED_MUTATION_SOURCE_FILE_SNAPSHOTS = new Map(
  ['src/tools/finance/markov-distribution/core.ts']
    .filter((filePath) => existsSync(resolve(process.cwd(), filePath)))
    .map((filePath) => [filePath, readFileSync(resolve(process.cwd(), filePath), 'utf8')] as const),
);

function readStructuredMutationFileContents(rootDir: string, filePath: string): string {
  const targetPath = resolve(rootDir, filePath);
  if (existsSync(targetPath)) {
    return readFileSync(targetPath, 'utf8');
  }

  if (resolve(rootDir) !== resolve(process.cwd())) {
    const result = spawnSync('git', ['-C', rootDir, 'show', `HEAD:${filePath}`], {
      encoding: 'utf8',
      stdio: ['ignore', 'pipe', 'pipe'],
    });
    if ((result.status ?? 1) === 0) {
      return result.stdout ?? '';
    }
  }

  const sourceSnapshot = STRUCTURED_MUTATION_SOURCE_FILE_SNAPSHOTS.get(filePath);
  if (sourceSnapshot !== undefined) {
    return sourceSnapshot;
  }

  return readFileSync(targetPath, 'utf8');
}

function cloneForecastLabResolvedRuntimeDefaults(): ForecastLabResolvedRuntimeDefaults {
  return {
    markov: { ...SHIPPED_PARAMETER_DEFAULT_TARGETS['src/tools/finance/markov-distribution/core.ts'] },
    conformal: { ...SHIPPED_PARAMETER_DEFAULT_TARGETS['src/tools/finance/conformal.ts'] },
    regime: { ...SHIPPED_PARAMETER_DEFAULT_TARGETS['src/tools/finance/regime-calibrator.ts'] },
  } as ForecastLabResolvedRuntimeDefaults;
}

function getResolvedRuntimeDefaultsTarget(
  resolved: ForecastLabResolvedRuntimeDefaults,
  filePath: string,
): ForecastLabMutableParameterDefaults {
  if (filePath === 'src/tools/finance/markov-distribution/core.ts') {
    return resolved.markov as ForecastLabMutableParameterDefaults;
  }
  if (filePath === 'src/tools/finance/conformal.ts') {
    return resolved.conformal as ForecastLabMutableParameterDefaults;
  }
  if (filePath === 'src/tools/finance/regime-calibrator.ts') {
    return resolved.regime as ForecastLabMutableParameterDefaults;
  }

  throw new ForecastLabRunnerError(`Forecast-lab runtime default resolution does not support ${filePath}.`);
}

function getForecastLabRuntimeDefaultsResolutionOrder(
  assetScope: ForecastLabRuntimeAssetScope,
): readonly ForecastLabRuntimeAssetScope[] {
  if (assetScope === 'gold') {
    return ['shared', 'gold'];
  }

  return [assetScope];
}

export function buildForecastLabRuntimeDefaultsActivation(params: {
  readonly profileId: string;
  readonly assetScope: ForecastLabRuntimeAssetScope;
  readonly mutation:
    | ForecastLabMarkovParameterMutationCandidate
    | ForecastLabMarkovParameterMutationReplayPayload;
}): ForecastLabRuntimeDefaultsActivation {
  return {
    profileId: params.profileId,
    assetScope: params.assetScope,
    overrides: params.mutation.edits.map((edit) => ({
      filePath: edit.filePath,
      parameterId: edit.parameterId,
      value: edit.afterValue,
    })),
  };
}

export function resolveForecastLabRuntimeDefaultsForAssetScope(
  assetScope: ForecastLabRuntimeAssetScope,
  activations: readonly ForecastLabRuntimeDefaultsActivation[] = [],
): ForecastLabResolvedRuntimeDefaults {
  const resolved = cloneForecastLabResolvedRuntimeDefaults();

  for (const scope of getForecastLabRuntimeDefaultsResolutionOrder(assetScope)) {
    for (const activation of activations) {
      if (activation.assetScope !== scope) {
        continue;
      }

      for (const override of activation.overrides) {
        const target = getResolvedRuntimeDefaultsTarget(resolved, override.filePath);
        target[override.parameterId] = override.value;
      }
    }
  }

  return resolved;
}

function resolveForecastLabRuntimeAssetScopeForProfile(profile: ForecastLabProfile): ForecastLabRuntimeAssetScope {
  if (profile.id === 'btc-markov-ultra-short-horizon') {
    return 'btc';
  }

  if (profile.id === 'gold-markov-short-horizon') {
    return 'gold';
  }

  if (profile.id === 'sol-markov-short-horizon') {
    return 'sol';
  }

  if (profile.id === 'hype-markov-short-horizon') {
    return 'hype';
  }

  if (profile.id === 'multi-asset-markov-short-horizon') {
    return 'shared';
  }

  throw new ForecastLabRunnerError(
    `Forecast-lab runtime default activation is not configured for profile "${profile.id}".`,
  );
}

function makeRunId(profileId: string, now: Date): string {
  return `forecast-lab-${profileId}-${now.toISOString().replace(/[:.]/g, '-')}`;
}

function makePromotionRunId(profileId: string, now: Date): string {
  return `forecast-lab-promote-${profileId}-${now.toISOString().replace(/[:.]/g, '-')}`;
}

function makeResetRunId(profileId: string, now: Date): string {
  return `forecast-lab-reset-${profileId}-${now.toISOString().replace(/[:.]/g, '-')}`;
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

function getForecastLabActiveStatePath(profileId: string): string {
  if (!SAFE_PROFILE_ID_PATTERN.test(profileId)) {
    throw new ForecastLabRunnerError(`Unsafe forecast-lab profile id for activation state path: ${profileId}`);
  }

  return join(getExperimentsDir({ create: true }), FORECAST_LAB_ACTIVE_STATE_DIR, `${profileId}.json`);
}

function readForecastLabActiveState(path: string, profileId: string): ForecastLabActivePromotionRecord | undefined {
  if (!existsSync(path)) {
    return undefined;
  }

  const parsed: unknown = JSON.parse(readFileSync(path, 'utf8'));
  if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
    throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} must be a JSON object.`);
  }

  const record = parsed as Record<string, unknown>;
  if (record.version !== 1) {
    throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} has unsupported version metadata.`);
  }
  if (record.profileId !== profileId) {
    throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} does not match profile "${profileId}".`);
  }
  if (typeof record.sourceRunId !== 'string' || record.sourceRunId.trim() === '') {
    throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} is missing sourceRunId.`);
  }
  if (typeof record.promotionRunId !== 'string' || record.promotionRunId.trim() === '') {
    throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} is missing promotionRunId.`);
  }
  if (typeof record.activatedAt !== 'string' || record.activatedAt.trim() === '') {
    throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} is missing activatedAt.`);
  }
  if (typeof record.activeStatePath !== 'string' || resolve(record.activeStatePath) !== resolve(path)) {
    throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} has inconsistent activeStatePath metadata.`);
  }
  if (!Array.isArray(record.mutatedFiles) || record.mutatedFiles.some((filePath) => typeof filePath !== 'string')) {
    throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} has invalid mutatedFiles metadata.`);
  }
  if (!Array.isArray(record.patchSummary) || record.patchSummary.some((entry) => typeof entry !== 'string')) {
    throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} has invalid patchSummary metadata.`);
  }

  try {
    validateForecastLabMarkovParameterMutationReplayPayload(record.mutationReplayPayload);
  } catch (error) {
    throw new ForecastLabRunnerError(
      `Forecast-lab active state at ${path} has invalid mutationReplayPayload: ${error instanceof Error ? error.message : String(error)}`,
    );
  }

  const promotion = record.promotion;
  if (!promotion || typeof promotion !== 'object' || Array.isArray(promotion)) {
    throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} is missing promotion metadata.`);
  }
  if ((promotion as Record<string, unknown>).status !== 'activated') {
    throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} must reference an activated promotion.`);
  }

  const previousActive = record.previousActive;
  if (previousActive !== undefined) {
    if (!previousActive || typeof previousActive !== 'object' || Array.isArray(previousActive)) {
      throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} has invalid previousActive metadata.`);
    }

    const previous = previousActive as Record<string, unknown>;
    for (const field of ['profileId', 'sourceRunId', 'promotionRunId', 'activatedAt'] as const) {
      if (typeof previous[field] !== 'string' || previous[field].trim() === '') {
        throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} has invalid previousActive.${field}.`);
      }
    }
    if (!Array.isArray(previous.mutatedFiles) || previous.mutatedFiles.some((filePath) => typeof filePath !== 'string')) {
      throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} has invalid previousActive.mutatedFiles.`);
    }
    if (!Array.isArray(previous.patchSummary) || previous.patchSummary.some((entry) => typeof entry !== 'string')) {
      throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} has invalid previousActive.patchSummary.`);
    }
    if (!previous.promotion || typeof previous.promotion !== 'object' || Array.isArray(previous.promotion)) {
      throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} has invalid previousActive.promotion.`);
    }
    if ((previous.promotion as Record<string, unknown>).status !== 'activated') {
      throw new ForecastLabRunnerError(`Forecast-lab active state at ${path} must keep previousActive.promotion as activated.`);
    }
    try {
      validateForecastLabMarkovParameterMutationReplayPayload(previous.mutationReplayPayload);
    } catch (error) {
      throw new ForecastLabRunnerError(
        `Forecast-lab active state at ${path} has invalid previousActive.mutationReplayPayload: ${error instanceof Error ? error.message : String(error)}`,
      );
    }
  }

  return parsed as ForecastLabActivePromotionRecord;
}

function formatMutationScalarLiteral(value: ForecastLabMutationScalarValue): string {
  return typeof value === 'boolean' ? String(value) : `${value}`;
}

function buildResetMutationFromActiveState(
  profile: ForecastLabProfile,
  activeState: ForecastLabActivePromotionRecord,
  mode: ForecastLabResetMode,
): ForecastLabMarkovParameterMutationCandidate {
  if (!isForecastLabMarkovMutatorProfileId(profile.id)) {
    throw new ForecastLabRunnerError(
      `Forecast-lab reset is only supported for markov-parameter profiles. Profile "${profile.id}" is unsupported.`,
    );
  }

  const currentMutation = replayForecastLabMarkovParameterMutation(activeState.mutationReplayPayload);
  const previousValues = mode === 'last-known-good'
    ? activeState.previousActive
    : undefined;
  if (mode === 'last-known-good' && !previousValues) {
    throw new ForecastLabRunnerError(
      `Forecast-lab reset for "${profile.id}" cannot restore a previous activated baseline because none was recorded.`,
    );
  }

  const previousValueMap = new Map<string, ForecastLabMutationScalarValue>();
  if (previousValues) {
    const previousMutation = replayForecastLabMarkovParameterMutation(previousValues.mutationReplayPayload);
    for (const edit of previousMutation.edits) {
      previousValueMap.set(`${edit.filePath}:${edit.parameterId}`, edit.afterValue);
    }
  }

  const edits = currentMutation.edits
    .map((edit) => {
      const defaultTarget = SHIPPED_PARAMETER_DEFAULT_TARGETS[edit.filePath]?.[edit.parameterId];
      if (defaultTarget === undefined) {
        throw new ForecastLabRunnerError(
          `Forecast-lab reset could not find shipped default for ${edit.parameterId} in ${edit.filePath}.`,
        );
      }

      const targetValue = mode === 'last-known-good'
        ? (previousValueMap.get(`${edit.filePath}:${edit.parameterId}`) ?? defaultTarget)
        : defaultTarget;
      if (targetValue === edit.afterValue) {
        return null;
      }

      return {
        kind: 'search-replace' as const,
        parameterId: edit.parameterId,
        filePath: edit.filePath,
        beforeValue: edit.afterValue,
        afterValue: targetValue,
        search: `  ${edit.parameterId}: ${formatMutationScalarLiteral(edit.afterValue)},`,
        replace: `  ${edit.parameterId}: ${formatMutationScalarLiteral(targetValue)},`,
        expectedReplacements: 1 as const,
      };
    })
    .filter((edit) => edit !== null);

  if (edits.length === 0) {
    throw new ForecastLabRunnerError(
      mode === 'defaults'
        ? `Forecast-lab reset for "${profile.id}" found no live parameter drift from shipped defaults.`
        : `Forecast-lab reset for "${profile.id}" is already aligned with the previously activated baseline.`,
    );
  }

  const targetFiles = [...new Set(edits.map((edit) => edit.filePath))];
  return {
    id: mode === 'defaults' ? 'forecast-lab-reset-defaults' : 'forecast-lab-reset-last-known-good',
    profileId: profile.id,
    mutatorId: 'search-replace',
    specSummary: {
      mutatorId: 'search-replace',
      targetFiles,
      summary: mode === 'defaults'
        ? 'Reset live forecast-lab parameters back to shipped defaults.'
        : 'Restore the previously activated forecast-lab baseline.',
    },
    patchSummary: edits.map(
      (edit) => `${edit.filePath.split('/').at(-1)}: ${edit.parameterId} ${edit.beforeValue} → ${edit.afterValue}`,
    ),
    edits,
  };
}

function assertSafeProfileCommand(command: ForecastLabCommand): void {
  if (UNSAFE_SHELL_COMMAND_PATTERN.test(command.command) || UNSAFE_GIT_COMMAND_PATTERN.test(command.command)) {
    throw new ForecastLabRunnerError(`Unsafe forecast-lab command "${command.id}" is not allowed`);
  }
}

/**
 * Create a forecast-lab command runner using the provided spawn function.
 *
 * **SECURITY NOTE**: The runner uses `shell: true` for the repo-owned profile
 * command strings in profiles.ts. assertSafeProfileCommand() rejects shell
 * metacharacters and high-risk git commands; do not pass user-authored commands
 * to this runner.
 */
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
          env: { ...getEnvironment(), ...(command.env ?? {}) },
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

function parseForecastLabMetrics(output: string): JsonValue | undefined {
  const lines = output
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0);

  for (let index = lines.length - 1; index >= 0; index -= 1) {
    const line = lines[index];
    if (!line.startsWith(FORECAST_LAB_METRICS_PREFIX)) {
      continue;
    }

    const payload = line.slice(FORECAST_LAB_METRICS_PREFIX.length).trim();
    if (!payload) {
      return undefined;
    }

    try {
      return JSON.parse(payload) as JsonValue;
    } catch {
      return undefined;
    }
  }

  return undefined;
}

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
    const metrics = parseForecastLabMetrics(result.stdout);
    results.push(metrics === undefined ? result : { ...result, metrics });
  }

  const commandsWithMetrics = results.filter((result) => result.metrics !== undefined);
  const summaryMetrics = commandsWithMetrics.length === 0
    ? undefined
    : commandsWithMetrics.length === 1
      ? commandsWithMetrics[0].metrics
      : Object.fromEntries(
          commandsWithMetrics.map((result) => [result.id, result.metrics]),
        ) as JsonValue;

  return {
    phase,
    cwd,
    exitCode: results.some((result) => result.exitCode !== 0) ? 1 : 0,
    commands: results,
    ...(summaryMetrics !== undefined ? { metrics: summaryMetrics } : {}),
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
    ...(summary.metrics !== undefined ? { metrics: summary.metrics } : {}),
    commands: summary.commands.map((command) => ({
      id: command.id,
      exitCode: command.exitCode,
      durationMs: command.durationMs,
      timedOut: command.timedOut,
      ...(command.metrics !== undefined ? { metrics: command.metrics } : {}),
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
    return toJsonValue({
      ...candidate,
      mutationMode: structuredMutation.mutationMode,
      mutationId: structuredMutation.mutationId,
      mutationSummary: structuredMutation.mutationSummary,
      lineage: structuredMutation.lineage,
      mutationSpecSummary: structuredMutation.mutationSpecSummary,
      candidateWorkspace: manifest.candidateWorkspace,
      ...(structuredMutation.parentRunId !== undefined ? { parentRunId: structuredMutation.parentRunId } : {}),
      selectedMutator: {
        id: structuredMutation.selectedMutatorId,
        mutatorId: structuredMutation.mutatorId,
      },
      ...(structuredMutation.mutatorRanking
        ? {
            mutatorRanking: {
              enabled: true,
              profileId: structuredMutation.mutatorRanking.profileId,
              totalStructuredRuns: structuredMutation.mutatorRanking.totalStructuredRuns,
              rankedMutators: structuredMutation.mutatorRanking.rankedMutators,
            },
          }
        : {}),
      mutatedFiles: [...structuredMutation.mutatedFiles],
      patchSummary: [...structuredMutation.patchSummary],
    }, 'structured mutation candidate artifact');
  }

  return toJsonValue({
    ...candidate,
    mutation: dryRun ? 'dry-run: no code mutation attempted' : 'skipped by --skip-mutation',
  }, 'candidate artifact');
}

function buildPendingPromotionState(
  manifest: ForecastLabRunManifest,
  manifestPath: string,
  requestedAt: string,
): ForecastLabPromotionState | undefined {
  if (manifest.mutationMode !== 'structured') {
    return undefined;
  }

  return {
    status: 'approval-required',
    source: {
      runId: manifest.runId,
      manifestPath,
    },
    requestedAt,
  };
}

function countOccurrences(haystack: string, needle: string): number {
  return haystack.split(needle).length - 1;
}

interface ForecastLabResolvedMutationPlan {
  readonly dryRun: boolean;
  readonly skipMutation: boolean;
  readonly mutationMode?: 'structured';
  readonly keepWorktree: boolean;
  readonly mutator?: string;
  readonly runRealMutation: boolean;
}

function resolveMutationPlan(options: ForecastLabRunOptions): ForecastLabResolvedMutationPlan {
  const dryRun = options.dryRun === true;
  const skipMutation = options.skipMutation === true;
  const mutationMode = options.mutationMode;
  const keepWorktree = options.keepWorktree === true;
  const mutator = options.mutator?.trim();

  if (mutator !== undefined && mutator.length === 0) {
    throw new ForecastLabRunnerError('Forecast-lab mutator override must not be empty.');
  }

  if (dryRun && skipMutation) {
    throw new ForecastLabRunnerError(
      'Conflicting forecast-lab options: dryRun and skipMutation cannot both be true.',
    );
  }

  if (dryRun || skipMutation) {
    if (mutationMode !== undefined) {
      throw new ForecastLabRunnerError(
        'Mutation mode is only supported for real forecast-lab mutation runs. Remove dryRun/skipMutation or omit mutationMode.',
      );
    }
    if (keepWorktree) {
      throw new ForecastLabRunnerError(
        'keepWorktree is only supported for real forecast-lab mutation runs.',
      );
    }
    if (mutator !== undefined) {
      throw new ForecastLabRunnerError(
        'Mutator overrides are only supported for real forecast-lab mutation runs.',
      );
    }

    return {
      dryRun,
      skipMutation,
      keepWorktree: false,
      runRealMutation: false,
    };
  }

  if (mutationMode === undefined) {
    throw new ForecastLabRunnerError(
      'Real forecast-lab mutation requires an explicit mutationMode. Pass mutationMode: "structured" (CLI: --mutation structured).',
    );
  }

  if (mutationMode !== 'structured') {
    throw new ForecastLabRunnerError(
      `Unsupported forecast-lab mutation mode "${mutationMode}". Only "structured" is currently implemented.`,
    );
  }

  return {
    dryRun: false,
    skipMutation: false,
    mutationMode,
    keepWorktree,
    mutator,
    runRealMutation: true,
  };
}

export function resolveForecastLabMutatorRankingEnabled(
  rankMutators: boolean | undefined,
  forecastingConfig?: ForecastingConfig,
): boolean {
  if (rankMutators !== undefined) {
    return rankMutators;
  }

  const effectiveForecastingConfig = forecastingConfig ?? loadConfig().forecasting;
  return effectiveForecastingConfig?.enableForecastLabMutatorRanking === true;
}

function getStructuredMutationCatalog(
  profile: ForecastLabProfile,
): readonly ForecastLabMarkovParameterMutationCandidate[] {
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

  return catalog;
}

function summarizeStructuredMutationCatalogState(
  candidates: readonly ForecastLabMarkovParameterMutationCandidate[],
  usedMutationIds: ReadonlySet<string> | undefined,
  isApplicable: (candidate: ForecastLabMarkovParameterMutationCandidate) => boolean,
): StructuredMutationCatalogState {
  const appliedCandidateIds: string[] = [];
  const applicableCandidateIds: string[] = [];
  const inapplicableCandidateIds: string[] = [];

  for (const candidate of candidates) {
    if (usedMutationIds?.has(candidate.id)) {
      appliedCandidateIds.push(candidate.id);
      continue;
    }

    if (isApplicable(candidate)) {
      applicableCandidateIds.push(candidate.id);
      continue;
    }

    inapplicableCandidateIds.push(candidate.id);
  }

  return {
    appliedCandidateIds,
    applicableCandidateIds,
    inapplicableCandidateIds,
  };
}

function formatStructuredMutationCatalogStateMessage(
  profileId: string,
  state: StructuredMutationCatalogState,
): string {
  const parts = [
    `No shipped structured mutator remains applicable after replaying the kept parent lineage for profile "${profileId}".`,
  ];

  if (state.appliedCandidateIds.length > 0) {
    parts.push(`Current kept lineage already applied: ${state.appliedCandidateIds.join(', ')}.`);
  }

  if (state.inapplicableCandidateIds.length > 0) {
    parts.push(`Remaining shipped mutators checked and found inapplicable: ${state.inapplicableCandidateIds.join(', ')}.`);
  }

  parts.push(
    'Next actions: keep the current best candidate, add a new shipped structured mutator, or intentionally reset the forecast-lab lineage outside the CLI.',
  );

  return parts.join(' ');
}

function selectStructuredMutation(
  profile: ForecastLabProfile,
  catalog: readonly ForecastLabMarkovParameterMutationCandidate[],
  requestedMutatorId?: string,
  usedMutationIds?: ReadonlySet<string>,
  isApplicable: (candidate: ForecastLabMarkovParameterMutationCandidate) => boolean = () => true,
  ranking?: ForecastLabMutatorRanking,
): ForecastLabMarkovParameterMutationCandidate {
  if (profile.mutation.mode !== 'structured') {
    throw new ForecastLabRunnerError(
      `Real forecast-lab mutation requires a structured profile with a shipped catalog. Profile "${profile.id}" uses "${profile.mutation.mode}".`,
    );
  }

  const allowedMutatorIds = new Set(profile.mutation.allowedMutatorIds);
  const allowedCandidates = catalog.filter((candidate) => allowedMutatorIds.has(candidate.mutatorId));
  const allowedCandidateIds = new Set(allowedCandidates.map((candidate) => candidate.id));

  if (requestedMutatorId) {
    const selected = allowedCandidates.find((candidate) => candidate.id === requestedMutatorId);
    if (!selected) {
      throw new ForecastLabRunnerError(
        `Unknown forecast-lab mutator "${requestedMutatorId}" for profile "${profile.id}". Expected one of: ${catalog.map((candidate) => candidate.id).join(', ')}`,
      );
    }

    if (!isApplicable(selected)) {
      const catalogState = summarizeStructuredMutationCatalogState(allowedCandidates, usedMutationIds, isApplicable);
      const remainingApplicableCandidateIds = catalogState.applicableCandidateIds.filter((candidateId) => candidateId !== selected.id);
      const fallback = remainingApplicableCandidateIds.length > 0
        ? `Try one of the remaining applicable shipped mutators instead: ${remainingApplicableCandidateIds.join(', ')}.`
        : formatStructuredMutationCatalogStateMessage(profile.id, catalogState);
      throw new ForecastLabRunnerError(
        `Forecast-lab mutator "${selected.id}" is not applicable after replaying the kept parent lineage for profile "${profile.id}". ${fallback}`,
      );
    }

    return selected;
  }

  const catalogState = summarizeStructuredMutationCatalogState(allowedCandidates, usedMutationIds, isApplicable);
  const selected = ranking?.rankedCandidates.find((candidate) =>
    allowedCandidateIds.has(candidate.id) && candidate.unused && candidate.applicable
  )?.candidate
    ?? ranking?.rankedCandidates.find((candidate) =>
      allowedCandidateIds.has(candidate.id) && candidate.applicable
    )?.candidate
    ?? allowedCandidates.find((candidate) => !usedMutationIds?.has(candidate.id) && isApplicable(candidate))
    ?? allowedCandidates.find((candidate) => isApplicable(candidate));
  if (!selected) {
    throw new ForecastLabRunnerError(
      formatStructuredMutationCatalogStateMessage(profile.id, catalogState),
    );
  }

  if (!allowedMutatorIds.has(selected.mutatorId)) {
    throw new ForecastLabRunnerError(
      `Forecast-lab mutator "${selected.id}" is not allowed by the structured mutation contract for profile "${profile.id}".`,
    );
  }

  return selected;
}

function applyStructuredMutationEdits(
  rootDir: string,
  updatedFiles: Map<string, string>,
  selectedMutation: ForecastLabMarkovParameterMutationCandidate,
): void {
  for (const edit of selectedMutation.edits) {
    const existingContents = updatedFiles.get(edit.filePath)
      ?? readStructuredMutationFileContents(rootDir, edit.filePath);
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
}

function canApplyStructuredMutation(
  rootDir: string,
  replayedMutations: readonly ForecastLabMarkovParameterMutationCandidate[],
  selectedMutation: ForecastLabMarkovParameterMutationCandidate,
): boolean {
  const updatedFiles = new Map<string, string>();

  try {
    for (const replayedMutation of replayedMutations) {
      applyStructuredMutationEdits(rootDir, updatedFiles, replayedMutation);
    }
    applyStructuredMutationEdits(rootDir, updatedFiles, selectedMutation);
    return true;
  } catch {
    return false;
  }
}

function readStructuredMutationSeed(
  profile: ForecastLabProfile,
  ledgerEntries: readonly ForecastLabLedgerEntry[],
): ForecastLabStructuredMutationSeed | undefined {
  const catalog = getStructuredMutationCatalog(profile);
  const parentEntries = ledgerEntries
    .filter((entry) => entry.profileId === profile.id && entry.decision === 'keep' && entry.mutationMode === 'structured')
    .reverse();

  for (const parentEntry of parentEntries) {
    const replayedMutations: ForecastLabMarkovParameterMutationCandidate[] = [];
    const seenRunIds = new Set<string>();
    let currentRunId: string | undefined = parentEntry.runId;
    let latestManifest: ForecastLabRunManifest = readRunManifest(getExperimentRunManifestPath(parentEntry.runId));

    while (currentRunId) {
      if (seenRunIds.has(currentRunId)) {
        throw new ForecastLabRunnerError(`Forecast-lab mutation lineage for "${profile.id}" contains a cycle at run "${currentRunId}".`);
      }
      seenRunIds.add(currentRunId);

      const manifest: ForecastLabRunManifest = currentRunId === parentEntry.runId
        ? latestManifest
        : readRunManifest(getExperimentRunManifestPath(currentRunId));
      if (manifest.profileId !== profile.id) {
        throw new ForecastLabRunnerError(
          `Forecast-lab parent run "${currentRunId}" belongs to "${manifest.profileId}", expected "${profile.id}".`,
        );
      }
      if (
        manifest.mutationMode !== 'structured' ||
        manifest.lineage === undefined ||
        (manifest.mutationId === undefined && manifest.mutationReplayPayload === undefined)
      ) {
        replayedMutations.length = 0;
        break;
      }
      if (manifest.parentRunId !== manifest.lineage.parentRunId) {
        throw new ForecastLabRunnerError(`Forecast-lab parent run "${currentRunId}" has inconsistent parentRunId metadata.`);
      }
      if (manifest.mutationSummary !== undefined && manifest.mutationSummary !== manifest.mutationSpecSummary?.summary) {
        throw new ForecastLabRunnerError(`Forecast-lab parent run "${currentRunId}" has inconsistent mutation summary metadata.`);
      }

      const replayedMutation = manifest.mutationReplayPayload
        ? replayForecastLabMarkovParameterMutation(manifest.mutationReplayPayload)
        : catalog.find((candidate) => candidate.id === manifest.mutationId);
      if (!replayedMutation) {
        throw new ForecastLabRunnerError(
          `Forecast-lab parent run "${currentRunId}" references unknown mutation "${manifest.mutationId}" for profile "${profile.id}".`,
        );
      }
      if (replayedMutation.profileId !== profile.id) {
        throw new ForecastLabRunnerError(
          `Forecast-lab parent run "${currentRunId}" replays a "${replayedMutation.profileId}" mutation in profile "${profile.id}".`,
        );
      }
      if (manifest.mutationId !== undefined && replayedMutation.id !== manifest.mutationId) {
        throw new ForecastLabRunnerError(`Forecast-lab parent run "${currentRunId}" has inconsistent mutation replay metadata.`);
      }
      replayedMutations.push(replayedMutation);
      currentRunId = manifest.parentRunId;
      latestManifest = manifest;
    }

    if (replayedMutations.length === 0) {
      continue;
    }

    return {
      parentRunId: parentEntry.runId,
      rootRunId: latestManifest.lineage!.rootRunId,
      generation: parentEntry.lineage!.generation + 1,
      replayedMutations: replayedMutations.reverse(),
      usedMutationIds: new Set(replayedMutations.map((mutation) => mutation.id)),
    };
  }

  return undefined;
}

function applyStructuredMutation(
  workspaceRootDir: string,
  profile: ForecastLabProfile,
  selectedMutation: ForecastLabMarkovParameterMutationCandidate,
  runId: string,
  seed?: ForecastLabStructuredMutationSeed,
  ranking?: ForecastLabMutatorRanking,
): ForecastLabStructuredMutationSelection {
  const updatedFiles = new Map<string, string>();

  for (const replayedMutation of seed?.replayedMutations ?? []) {
    applyStructuredMutationEdits(workspaceRootDir, updatedFiles, replayedMutation);
  }

  applyStructuredMutationEdits(workspaceRootDir, updatedFiles, selectedMutation);

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
    parentRunId: seed?.parentRunId,
    mutationId: selectedMutation.id,
    mutationSummary: selectedMutation.specSummary.summary,
    lineage: {
      rootRunId: seed?.rootRunId ?? runId,
      generation: seed?.generation ?? 0,
      ...(seed?.parentRunId !== undefined ? { parentRunId: seed.parentRunId } : {}),
    },
    selectedMutatorId: selectedMutation.id,
    mutatorId: selectedMutation.mutatorId,
    mutatedFiles,
    patchSummary: [...selectedMutation.patchSummary],
    mutationSpecSummary: selectedMutation.specSummary,
    mutationReplayPayload: snapshotForecastLabMarkovParameterMutation(selectedMutation),
    mutatorRanking: ranking,
  };
}

function buildStructuredMutationFileUpdates(
  rootDir: string,
  mutation: ForecastLabMarkovParameterMutationCandidate,
): {
  readonly updatedFiles: Map<string, string>;
  readonly previousContents: Map<string, string>;
} {
  const updatedFiles = new Map<string, string>();
  applyStructuredMutationEdits(rootDir, updatedFiles, mutation);
  return {
    updatedFiles,
    previousContents: new Map(
      [...updatedFiles.keys()].map((filePath) => [filePath, readFileSync(resolve(rootDir, filePath), 'utf8')]),
    ),
  };
}

function applyStructuredMutationRuntimeDefaults(
  assetScope: ForecastLabRuntimeAssetScope,
  mutation: ForecastLabMarkovParameterMutationCandidate,
): readonly ForecastLabRuntimeDefaultsSnapshot[] {
  const snapshots: ForecastLabRuntimeDefaultsSnapshot[] = [];
  const overridesByFilePath = new Map<string, ForecastLabMutableParameterDefaults>();

  try {
    for (const edit of mutation.edits) {
      const target = LIVE_PARAMETER_DEFAULT_TARGETS[edit.filePath];
      if (!target) {
        throw new ForecastLabRunnerError(
          `Forecast-lab activation does not support runtime default updates for ${edit.filePath}.`,
        );
      }

      const previousOverrides = overridesByFilePath.get(edit.filePath)
        ?? target.get(assetScope)
        ?? {};
      const previousValue = previousOverrides[edit.parameterId] ?? SHIPPED_PARAMETER_DEFAULT_TARGETS[edit.filePath]?.[edit.parameterId];
      if (typeof previousValue !== 'boolean' && typeof previousValue !== 'number') {
        throw new ForecastLabRunnerError(
          `Forecast-lab activation could not find runtime parameter "${edit.parameterId}" in ${edit.filePath}.`,
        );
      }

      if (!overridesByFilePath.has(edit.filePath)) {
        snapshots.push({
          assetScope,
          filePath: edit.filePath,
          overrides: target.get(assetScope),
        });
      }

      const nextOverrides = {
        ...previousOverrides,
        [edit.parameterId]: edit.afterValue,
      };
      overridesByFilePath.set(edit.filePath, nextOverrides);
      target.set(assetScope, nextOverrides);
    }
  } catch (error) {
    restoreStructuredMutationRuntimeDefaults(snapshots);
    throw error;
  }

  return snapshots;
}

function restoreStructuredMutationRuntimeDefaults(
  snapshots: readonly ForecastLabRuntimeDefaultsSnapshot[],
): void {
  for (const snapshot of snapshots) {
    const target = LIVE_PARAMETER_DEFAULT_TARGETS[snapshot.filePath];
    if (!target) {
      throw new ForecastLabRunnerError(
        `Forecast-lab activation could not restore runtime defaults for ${snapshot.filePath}.`,
      );
    }
    target.set(snapshot.assetScope, snapshot.overrides);
  }
}

function applyStructuredMutationToLiveSource(
  profile: ForecastLabProfile,
  mutation: ForecastLabMarkovParameterMutationCandidate,
): {
  readonly mutatedFiles: readonly string[];
  readonly previousContents: Map<string, string>;
  readonly runtimeDefaults: readonly ForecastLabRuntimeDefaultsSnapshot[];
  } {
  const rootDir = process.cwd();
  const assetScope = resolveForecastLabRuntimeAssetScopeForProfile(profile);
  if (assetScope === 'gold' || assetScope === 'sol' || assetScope === 'hype') {
    return {
      mutatedFiles: [],
      previousContents: new Map(),
      runtimeDefaults: applyStructuredMutationRuntimeDefaults(assetScope, mutation),
    };
  }

  const { updatedFiles, previousContents } = buildStructuredMutationFileUpdates(rootDir, mutation);
  const mutatedFiles = applyForecastLabCandidateEdits(
    rootDir,
    [...updatedFiles.entries()].map(([path, contents]) => ({ path, contents })),
    {
      allowedPaths: profile.mutation.mutableFiles,
      readOnlyPaths: profile.readOnlyHarnessFiles,
    },
  );

  try {
    return {
      mutatedFiles,
      previousContents,
      runtimeDefaults: applyStructuredMutationRuntimeDefaults(assetScope, mutation),
    };
  } catch (error) {
    applyForecastLabCandidateEdits(
      rootDir,
      [...previousContents.entries()].map(([path, contents]) => ({ path, contents })),
      {
        allowedPaths: profile.mutation.mutableFiles,
        readOnlyPaths: profile.readOnlyHarnessFiles,
      },
    );
    throw error;
  }
}

function cleanupPreparedWorkspaceOnError(
  workspace: ReturnType<typeof prepareForecastLabPromotionWorkspace>,
  error: unknown,
): never {
  try {
    workspace.cleanup();
  } catch (cleanupError) {
    if (error instanceof Error) {
      const cleanupDetail = cleanupError instanceof Error
        ? (cleanupError.stack ?? `${cleanupError.name}: ${cleanupError.message}`)
        : String(cleanupError);
      const baseStack = error.stack ?? `${error.name}: ${error.message}`;
      error.stack = `${baseStack}\nSuppressed forecast-lab workspace cleanup failure: ${cleanupDetail}`;
    }
  }

  throw error;
}

function validatePromotionSourceManifest(
  profile: ForecastLabProfile,
  sourceEntry: ForecastLabLedgerEntry,
  manifest: ForecastLabRunManifest,
): ForecastLabMarkovParameterMutationReplayPayload {
  if (manifest.runId !== sourceEntry.runId) {
    throw new ForecastLabRunnerError(`Forecast-lab promotion source manifest runId mismatch for "${sourceEntry.runId}".`);
  }
  if (manifest.profileId !== profile.id) {
    throw new ForecastLabRunnerError(
      `Forecast-lab promotion source run "${sourceEntry.runId}" belongs to "${manifest.profileId}", expected "${profile.id}".`,
    );
  }
  if (manifest.mutationMode !== 'structured') {
    throw new ForecastLabRunnerError(`Forecast-lab promotion source run "${sourceEntry.runId}" is not a structured mutation run.`);
  }
  if (!manifest.promotion) {
    throw new ForecastLabRunnerError(`Forecast-lab promotion source run "${sourceEntry.runId}" is missing promotion metadata.`);
  }
  if (manifest.promotion.status === 'promoted' || manifest.promotion.status === 'activated') {
    throw new ForecastLabRunnerError(
      `Forecast-lab promotion source run "${sourceEntry.runId}" is already ${manifest.promotion.status}.`,
    );
  }
  if (!manifest.mutationReplayPayload) {
    throw new ForecastLabRunnerError(
      `Forecast-lab promotion source run "${sourceEntry.runId}" is missing its structured mutation replay payload.`,
    );
  }
  if (manifest.lineage === undefined || manifest.candidateWorkspace === undefined || manifest.mutationSpecSummary === undefined) {
    throw new ForecastLabRunnerError(
      `Forecast-lab promotion source run "${sourceEntry.runId}" is missing structured mutation lineage metadata.`,
    );
  }
  if (sourceEntry.mutationId !== undefined && manifest.mutationId !== sourceEntry.mutationId) {
    throw new ForecastLabRunnerError(`Forecast-lab promotion source run "${sourceEntry.runId}" has inconsistent mutationId metadata.`);
  }
  if (
    sourceEntry.mutationSummary !== undefined &&
    manifest.mutationSummary !== undefined &&
    manifest.mutationSummary !== sourceEntry.mutationSummary
  ) {
    throw new ForecastLabRunnerError(`Forecast-lab promotion source run "${sourceEntry.runId}" has inconsistent mutation summary metadata.`);
  }
  if (
    sourceEntry.lineage !== undefined &&
    stableJsonStringify(sourceEntry.lineage) !== stableJsonStringify(manifest.lineage)
  ) {
    throw new ForecastLabRunnerError(`Forecast-lab promotion source run "${sourceEntry.runId}" has inconsistent lineage metadata.`);
  }
  if (
    sourceEntry.mutationSpecSummary !== undefined &&
    stableJsonStringify(sourceEntry.mutationSpecSummary) !== stableJsonStringify(manifest.mutationSpecSummary)
  ) {
    throw new ForecastLabRunnerError(`Forecast-lab promotion source run "${sourceEntry.runId}" has inconsistent mutation spec metadata.`);
  }

  const replayedMutation = replayForecastLabMarkovParameterMutation(manifest.mutationReplayPayload);
  const replaySnapshot = snapshotForecastLabMarkovParameterMutation(replayedMutation);
  if (stableJsonStringify(replaySnapshot) !== stableJsonStringify(manifest.mutationReplayPayload)) {
    throw new ForecastLabRunnerError(
      `Forecast-lab promotion source run "${sourceEntry.runId}" has a replay payload that cannot be losslessly reconstructed.`,
    );
  }

  return replaySnapshot;
}

function repairPromotionSourceManifestIfNeeded(
  profile: ForecastLabProfile,
  sourceEntry: ForecastLabLedgerEntry,
  manifestPath: string,
  manifest: ForecastLabRunManifest,
): ForecastLabRunManifest {
  if (manifest.promotion) {
    return manifest;
  }

  const normalizedSource = {
    runId: sourceEntry.runId,
    manifestPath,
  };
  const activeStatePath = getForecastLabActiveStatePath(profile.id);
  const activeState = readForecastLabActiveState(activeStatePath, profile.id);
  const repairedPromotion = activeState?.sourceRunId === sourceEntry.runId
    ? activeState.promotion
    : sourceEntry.promotion
      ? {
          ...sourceEntry.promotion,
          source: normalizedSource,
        }
      : {
          status: 'approval-required' as const,
          source: normalizedSource,
          requestedAt: sourceEntry.startedAt,
        };
  const repairedManifest: ForecastLabRunManifest = {
    ...manifest,
    promotion: repairedPromotion,
  };
  writeRunManifest(manifestPath, repairedManifest);
  return repairedManifest;
}

type PromotionSourceInvariantSnapshot = {
  readonly runId: string;
  readonly startedAt: string;
  readonly profileId: string;
  readonly targetSubsystem: string;
  readonly baselineCommit?: string;
  readonly candidateBranch: string;
  readonly allowedGlobs: readonly string[];
  readonly routingContext?: ForecastLabRoutingContext;
  readonly effectiveMutationContract?: ForecastLabRunManifest['effectiveMutationContract'];
  readonly mutationMode?: ForecastLabMutationMode;
  readonly parentRunId?: string;
  readonly mutationId?: string;
  readonly mutationSummary?: string;
  readonly lineage?: ForecastLabMutationLineage;
  readonly mutationSpecSummary?: ForecastLabRunManifest['mutationSpecSummary'];
  readonly mutationReplayPayload?: ForecastLabMarkovParameterMutationReplayPayload;
  readonly candidateWorkspace?: ForecastLabRunManifest['candidateWorkspace'];
  readonly promotion?: {
    readonly source: ForecastLabPromotionState['source'];
    readonly requestedAt: string;
  };
  readonly promotionSource?: ForecastLabRunManifest['promotionSource'];
  readonly artifactsPath: string;
};

function snapshotPromotionSourceInvariants(manifest: ForecastLabRunManifest): PromotionSourceInvariantSnapshot {
  return {
    runId: manifest.runId,
    startedAt: manifest.startedAt,
    profileId: manifest.profileId,
    targetSubsystem: manifest.targetSubsystem,
    ...(manifest.baselineCommit !== undefined ? { baselineCommit: manifest.baselineCommit } : {}),
    candidateBranch: manifest.candidateBranch,
    allowedGlobs: [...manifest.allowedGlobs],
    ...(manifest.routingContext !== undefined ? { routingContext: manifest.routingContext } : {}),
    ...(manifest.effectiveMutationContract !== undefined
      ? { effectiveMutationContract: manifest.effectiveMutationContract }
      : {}),
    ...(manifest.mutationMode !== undefined ? { mutationMode: manifest.mutationMode } : {}),
    ...(manifest.parentRunId !== undefined ? { parentRunId: manifest.parentRunId } : {}),
    ...(manifest.mutationId !== undefined ? { mutationId: manifest.mutationId } : {}),
    ...(manifest.mutationSummary !== undefined ? { mutationSummary: manifest.mutationSummary } : {}),
    ...(manifest.lineage !== undefined ? { lineage: manifest.lineage } : {}),
    ...(manifest.mutationSpecSummary !== undefined ? { mutationSpecSummary: manifest.mutationSpecSummary } : {}),
    ...(manifest.mutationReplayPayload !== undefined ? { mutationReplayPayload: manifest.mutationReplayPayload } : {}),
    ...(manifest.candidateWorkspace !== undefined ? { candidateWorkspace: manifest.candidateWorkspace } : {}),
    ...(manifest.promotion
      ? {
          promotion: {
            source: manifest.promotion.source,
            requestedAt: manifest.promotion.requestedAt,
          },
        }
      : {}),
    ...(manifest.promotionSource !== undefined ? { promotionSource: manifest.promotionSource } : {}),
    artifactsPath: manifest.artifactsPath,
  };
}

function revalidatePromotionSourceAfterVerification(
  profile: ForecastLabProfile,
  source: ForecastLabPromotionSourceSelection,
): ForecastLabRunManifest {
  const manifest = readRunManifest(source.manifestPath);
  validatePromotionSourceManifest(profile, source.ledgerEntry, manifest);

  if (
    stableJsonStringify(snapshotPromotionSourceInvariants(manifest))
    !== stableJsonStringify(snapshotPromotionSourceInvariants(source.manifest))
  ) {
    throw new ForecastLabRunnerError(
      `Forecast-lab promotion source run "${source.manifest.runId}" changed while promotion verification was running.`,
    );
  }

  return manifest;
}

function acquirePromotionSourceLock(source: ForecastLabPromotionSourceSelection, promotionRunId: string): () => void {
  const lockPath = `${dirname(source.manifestPath)}/promotion.lock`;
  assertInsideExperiments(lockPath);

  try {
    writeFileSync(lockPath, `${promotionRunId}\n`, { encoding: 'utf8', flag: 'wx' });
  } catch (error) {
    const code = error && typeof error === 'object' && 'code' in error ? (error as NodeJS.ErrnoException).code : undefined;
    if (code === 'EEXIST') {
      throw new ForecastLabRunnerError(
        `Forecast-lab promotion source run "${source.manifest.runId}" is already being promoted by another process.`,
      );
    }
    throw error;
  }

  return () => {
    try {
      unlinkSync(lockPath);
    } catch (error) {
      const code = error && typeof error === 'object' && 'code' in error ? (error as NodeJS.ErrnoException).code : undefined;
      if (code !== 'ENOENT') {
        throw error;
      }
    }
  };
}

function resolvePromotionSource(
  profile: ForecastLabProfile,
  ledgerPath: string,
  requestedRunId?: string,
): ForecastLabPromotionSourceSelection {
  const sourceEntry = requestedRunId
    ? readLedgerEntries(ledgerPath)
      .findLast((entry) => entry.runId === requestedRunId && entry.profileId === profile.id && entry.decision === 'keep' && entry.mutationMode === 'structured')
    : findLatestKeptLedgerEntry(ledgerPath, profile.id, { mutationMode: 'structured' });

  if (!sourceEntry) {
    throw new ForecastLabRunnerError(
      requestedRunId
        ? `Forecast-lab promotion source run "${requestedRunId}" was not found as a kept structured run for "${profile.id}".`
        : `Forecast-lab promotion requires a kept structured run for "${profile.id}".`,
    );
  }

  const manifestPath = sourceEntry.promotion?.source.manifestPath ?? getExperimentRunManifestPath(sourceEntry.runId);
  const manifest = repairPromotionSourceManifestIfNeeded(
    profile,
    sourceEntry,
    manifestPath,
    readRunManifest(manifestPath),
  );
  const replaySnapshot = validatePromotionSourceManifest(profile, sourceEntry, manifest);

  return {
    ledgerEntry: sourceEntry,
    manifestPath,
    manifest,
    replayPayload: replaySnapshot,
  };
}

function stageStructuredMutationReplay(
  workspaceRootDir: string,
  profile: ForecastLabProfile,
  replayPayload: ForecastLabMarkovParameterMutationReplayPayload,
  sourceManifest: ForecastLabRunManifest,
): ForecastLabStructuredMutationSelection {
  const replayedMutation = replayForecastLabMarkovParameterMutation(replayPayload);
  const updatedFiles = new Map<string, string>();
  applyStructuredMutationEdits(workspaceRootDir, updatedFiles, replayedMutation);

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
    parentRunId: sourceManifest.parentRunId,
    mutationId: replayedMutation.id,
    mutationSummary: replayedMutation.specSummary.summary,
    lineage: sourceManifest.lineage!,
    selectedMutatorId: replayedMutation.id,
    mutatorId: replayedMutation.mutatorId,
    mutatedFiles,
    patchSummary: [...replayedMutation.patchSummary],
    mutationSpecSummary: replayedMutation.specSummary,
    mutationReplayPayload: replayPayload,
  };
}

export async function promoteForecastLab(options: ForecastLabPromotionOptions): Promise<ForecastLabPromotionResult> {
  const profile = getForecastLabProfile(options.profileId);
  const now = options.now ?? (() => new Date());
  const startedAt = now().toISOString();
  const runId = options.runId ?? makePromotionRunId(profile.id, new Date(startedAt));
  const progress = options.progress;
  const output = options.output;
  const ledgerPath = options.ledgerPath ?? getExperimentLedgerPath({ create: true });
  const commandRunner = options.commandRunner ?? defaultForecastLabCommandRunner;
  const workspaceMaterializePaths = options.commandRunner ? profile.mutation.mutableFiles : undefined;
  const lightweightWorkspace = options.commandRunner !== undefined;
  const source = resolvePromotionSource(profile, ledgerPath, options.sourceRunId);
  const releaseSourceLock = acquirePromotionSourceLock(source, runId);
  const runDir = getExperimentRunDir(runId, { create: true });
  const manifestPath = getExperimentRunManifestPath(runId, { create: true });
  const baselinePath = `${runDir}/baseline.json`;
  const candidatePath = `${runDir}/candidate.json`;
  const decisionPath = `${runDir}/decision.json`;
  const activationPath = `${runDir}/activation.json`;
  const activeStatePath = getForecastLabActiveStatePath(profile.id);

  for (const path of [runDir, manifestPath, baselinePath, candidatePath, decisionPath, activationPath, activeStatePath]) {
    assertInsideExperiments(path);
  }

  try {
    progress?.(`forecast-lab: promoting kept run ${source.manifest.runId} for ${profile.id} (${runId})`);
    const workspace = prepareForecastLabPromotionWorkspace(runId, {
      materializePaths: workspaceMaterializePaths,
      lightweight: lightweightWorkspace,
    });

    try {
      const structuredMutation = stageStructuredMutationReplay(
        workspace.metadata.rootDir,
        profile,
        source.replayPayload,
        source.manifest,
      );
      const manifest: ForecastLabRunManifest = {
        runId,
        startedAt,
        profileId: profile.id,
        targetSubsystem: profile.targetSubsystem,
        baselineCommit: workspace.baselineCommit,
        candidateBranch: workspace.metadata.branch,
        allowedGlobs: [...profile.allowedGlobs],
        effectiveMutationContract: source.manifest.effectiveMutationContract ?? snapshotEffectiveMutationContract(profile),
        mutationMode: 'structured',
        ...(source.manifest.parentRunId !== undefined ? { parentRunId: source.manifest.parentRunId } : {}),
        mutationId: structuredMutation.mutationId,
        mutationSummary: structuredMutation.mutationSummary,
        lineage: structuredMutation.lineage,
        mutationSpecSummary: structuredMutation.mutationSpecSummary,
        mutationReplayPayload: structuredMutation.mutationReplayPayload,
        candidateWorkspace: workspace.metadata,
        promotionSource: {
          runId: source.manifest.runId,
          manifestPath: source.manifestPath,
        },
        artifactsPath: runDir,
      };

      writeRunManifest(manifestPath, manifest);
      progress?.(`forecast-lab: promotion manifest written to ${manifestPath}`);
      progress?.(`forecast-lab: promotion staging workspace ${workspace.metadata.rootDir}`);
      progress?.(`forecast-lab: replaying kept mutation ${structuredMutation.mutationId} from ${source.manifest.runId}`);
      progress?.(`forecast-lab: promoted files ${structuredMutation.mutatedFiles.join(', ')}`);
      progress?.(`forecast-lab: promoted patch summary ${structuredMutation.patchSummary.join(' | ')}`);

      progress?.('forecast-lab: starting promotion baseline gate');
      const baseline = await runGate('baseline', profile, runId, commandRunner, process.cwd(), progress, output);
      writeJsonArtifact(baselinePath, baseline);
      progress?.(`forecast-lab: promotion baseline results written to ${baselinePath}`);

      progress?.('forecast-lab: starting promotion candidate gate');
      const candidate = await runGate(
        'candidate',
        profile,
        runId,
        commandRunner,
        workspace.metadata.rootDir,
        progress,
        output,
      );
      const candidateArtifact = buildCandidateArtifact(candidate, manifest, structuredMutation, false);
      writeJsonArtifact(candidatePath, candidateArtifact);
      progress?.(`forecast-lab: promotion candidate results written to ${candidatePath}`);

      const decision = decideRun(profile, baseline, candidate);
      if (decision.decision !== 'keep') {
        writeJsonArtifact(decisionPath, {
          ...decision,
          sourceRunId: source.manifest.runId,
          promotionSource: manifest.promotionSource,
          mutationMode: structuredMutation.mutationMode,
          mutationId: structuredMutation.mutationId,
          mutationSummary: structuredMutation.mutationSummary,
          candidateWorkspace: manifest.candidateWorkspace,
          mutatedFiles: [...structuredMutation.mutatedFiles],
          patchSummary: [...structuredMutation.patchSummary],
        });
        progress?.(`forecast-lab: promotion verification failed — ${decision.reason}`);
        throw new ForecastLabRunnerError(
          `Forecast-lab promotion for "${profile.id}" regressed while verifying kept run "${source.manifest.runId}": ${decision.reason}`,
        );
      }

      const refreshedSourceManifest = revalidatePromotionSourceAfterVerification(profile, source);
      const sourcePromotion = refreshedSourceManifest.promotion;
      if (!sourcePromotion || sourcePromotion.status === 'promoted' || sourcePromotion.status === 'activated') {
        throw new ForecastLabRunnerError(
          `Forecast-lab promotion source run "${source.manifest.runId}" is no longer promotable.`,
        );
      }

      const approvedAt = sourcePromotion.status === 'approval-required'
        ? startedAt
        : sourcePromotion.approvedAt;
      const promotedAt = now().toISOString();
      const previouslyActive = readForecastLabActiveState(activeStatePath, profile.id);
      const activation: ForecastLabPromotionActivationRef = {
        runId,
        manifestPath,
        artifactsPath: runDir,
        workspace: workspace.metadata,
      };
      const liveReplayMutation = replayForecastLabMarkovParameterMutation(structuredMutation.mutationReplayPayload);
      const liveActivation = applyStructuredMutationToLiveSource(profile, liveReplayMutation);
      const activatedAt = now().toISOString();
      const promotion: Extract<ForecastLabPromotionState, { status: 'activated' }> = {
        status: 'activated',
        source: sourcePromotion.source,
        requestedAt: sourcePromotion.requestedAt,
        approvedAt,
        promotedAt,
        activatedAt,
        activation,
      };
      const updatedSourceManifest: ForecastLabRunManifest = {
        ...refreshedSourceManifest,
        promotion,
      };

      const activeStateRecord: ForecastLabActivePromotionRecord = {
        version: 1,
        profileId: profile.id,
        sourceRunId: source.manifest.runId,
        promotionRunId: runId,
        activatedAt,
        promotion,
        mutationReplayPayload: structuredMutation.mutationReplayPayload,
        mutatedFiles: [...liveActivation.mutatedFiles],
        patchSummary: [...structuredMutation.patchSummary],
        activeStatePath,
        ...(previouslyActive
          ? {
              previousActive: {
                profileId: previouslyActive.profileId,
                sourceRunId: previouslyActive.sourceRunId,
                promotionRunId: previouslyActive.promotionRunId,
                activatedAt: previouslyActive.activatedAt,
                promotion: previouslyActive.promotion,
                mutationReplayPayload: previouslyActive.mutationReplayPayload,
                mutatedFiles: [...previouslyActive.mutatedFiles],
                patchSummary: [...previouslyActive.patchSummary],
              },
            }
          : {}),
      };

      try {
        writeRunManifest(source.manifestPath, updatedSourceManifest);
        writeJsonArtifact(activeStatePath, activeStateRecord);
        writeJsonArtifact(activationPath, {
          sourceRunId: source.manifest.runId,
          promotion,
          promotionSource: manifest.promotionSource,
          mutationId: structuredMutation.mutationId,
          mutationSummary: structuredMutation.mutationSummary,
          mutationSpecSummary: structuredMutation.mutationSpecSummary,
          mutationReplayPayload: structuredMutation.mutationReplayPayload,
          mutatedFiles: [...liveActivation.mutatedFiles],
          patchSummary: [...structuredMutation.patchSummary],
          allowedGlobs: [...profile.allowedGlobs],
          baselineCommit: workspace.baselineCommit,
          candidateWorkspace: workspace.metadata,
          activeStatePath,
          runtimeDefaultsUpdated: true,
        });
        writeJsonArtifact(decisionPath, {
          ...decision,
          sourceRunId: source.manifest.runId,
          promotion,
          promotionSource: manifest.promotionSource,
          mutationMode: structuredMutation.mutationMode,
          mutationId: structuredMutation.mutationId,
          mutationSummary: structuredMutation.mutationSummary,
          candidateWorkspace: manifest.candidateWorkspace,
          mutatedFiles: [...liveActivation.mutatedFiles],
          patchSummary: [...structuredMutation.patchSummary],
        });
      } catch (error) {
        try {
          writeRunManifest(source.manifestPath, refreshedSourceManifest);
          applyForecastLabCandidateEdits(
            process.cwd(),
            [...liveActivation.previousContents.entries()].map(([path, contents]) => ({ path, contents })),
            {
              allowedPaths: profile.mutation.mutableFiles,
              readOnlyPaths: profile.readOnlyHarnessFiles,
            },
          );
          restoreStructuredMutationRuntimeDefaults(liveActivation.runtimeDefaults);
        } catch (restoreError) {
          if (error instanceof Error) {
            const restoreDetail = restoreError instanceof Error
              ? (restoreError.stack ?? `${restoreError.name}: ${restoreError.message}`)
              : String(restoreError);
            error.stack = `${error.stack ?? `${error.name}: ${error.message}`}\nSuppressed forecast-lab activation rollback failure: ${restoreDetail}`;
          }
        }
        throw error;
      }

      progress?.(`forecast-lab: promotion source manifest updated at ${source.manifestPath}`);
      progress?.(`forecast-lab: live activation recorded at ${activeStatePath}`);
      progress?.(`forecast-lab: activation artifacts written to ${activationPath}`);
      progress?.(`forecast-lab: promotion decision ${decision.decision} — ${decision.reason}`);

      return {
        runId,
        sourceRunId: source.manifest.runId,
        manifest,
        sourceManifest: updatedSourceManifest,
        baseline: toJsonValue(baseline, 'promotion baseline summary'),
        candidate: candidateArtifact,
        decision,
        activation,
        activeStatePath,
      };
    } catch (error) {
      cleanupPreparedWorkspaceOnError(workspace, error);
      throw error;
    }
  } finally {
    releaseSourceLock();
  }
}

export async function resetForecastLab(options: ForecastLabResetOptions): Promise<ForecastLabResetResult> {
  const profile = getForecastLabProfile(options.profileId);
  const now = options.now ?? (() => new Date());
  const startedAt = now().toISOString();
  const runId = options.runId ?? makeResetRunId(profile.id, new Date(startedAt));
  const progress = options.progress;
  const activeStatePath = getForecastLabActiveStatePath(profile.id);
  const activeState = readForecastLabActiveState(activeStatePath, profile.id);

  if (!activeState) {
    throw new ForecastLabRunnerError(`Forecast-lab reset requires an active promoted baseline for "${profile.id}".`);
  }

  const resetMutation = buildResetMutationFromActiveState(profile, activeState, options.mode);
  const runDir = getExperimentRunDir(runId, { create: true });
  const resetArtifactPath = `${runDir}/reset.json`;
  for (const path of [runDir, resetArtifactPath]) {
    assertInsideExperiments(path);
  }

  progress?.(`forecast-lab: resetting live parameters for ${profile.id} (${runId})`);
  const liveReset = applyStructuredMutationToLiveSource(profile, resetMutation);
  const resetAt = now().toISOString();

  try {
    writeJsonArtifact(resetArtifactPath, {
      version: 1,
      runId,
      startedAt,
      resetAt,
      profileId: profile.id,
      mode: options.mode,
      previousActiveStatePath: activeStatePath,
      sourceRunId: activeState.sourceRunId,
      promotionRunId: activeState.promotionRunId,
      mutationReplayPayload: activeState.mutationReplayPayload,
      resetMutation,
      mutatedFiles: [...liveReset.mutatedFiles],
      patchSummary: [...resetMutation.patchSummary],
      ...(options.mode === 'last-known-good'
        ? { restoredActive: activeState.previousActive }
        : {}),
    });

    if (options.mode === 'defaults') {
      if (existsSync(activeStatePath)) {
        unlinkSync(activeStatePath);
      }
    } else {
      const restoredActive = activeState.previousActive;
      if (!restoredActive) {
        throw new ForecastLabRunnerError(
          `Forecast-lab reset for "${profile.id}" could not restore the previous activation metadata.`,
        );
      }

      const restoredActiveState: ForecastLabActivePromotionRecord = {
        version: 1,
        profileId: restoredActive.profileId,
        sourceRunId: restoredActive.sourceRunId,
        promotionRunId: restoredActive.promotionRunId,
        activatedAt: restoredActive.activatedAt,
        promotion: restoredActive.promotion,
        mutationReplayPayload: restoredActive.mutationReplayPayload,
        mutatedFiles: [...restoredActive.mutatedFiles],
        patchSummary: [...restoredActive.patchSummary],
        activeStatePath,
      };
      writeJsonArtifact(activeStatePath, restoredActiveState);
    }
  } catch (error) {
    try {
      restoreStructuredMutationRuntimeDefaults(liveReset.runtimeDefaults);
      applyForecastLabCandidateEdits(
        process.cwd(),
        [...liveReset.previousContents.entries()].map(([path, contents]) => ({ path, contents })),
        {
          allowedPaths: profile.mutation.mutableFiles,
          readOnlyPaths: profile.readOnlyHarnessFiles,
        },
      );
    } catch (restoreError) {
      if (error instanceof Error) {
        const restoreDetail = restoreError instanceof Error
          ? (restoreError.stack ?? `${restoreError.name}: ${restoreError.message}`)
          : String(restoreError);
        error.stack = `${error.stack ?? `${error.name}: ${error.message}`}\nSuppressed forecast-lab reset rollback failure: ${restoreDetail}`;
      }
    }
    throw error;
  }

  progress?.(`forecast-lab: reset artifacts written to ${resetArtifactPath}`);
  progress?.(
    options.mode === 'defaults'
      ? `forecast-lab: ${profile.id} now uses shipped defaults`
      : `forecast-lab: ${profile.id} now uses the previously activated baseline`,
  );

  return {
    runId,
    profileId: profile.id,
    mode: options.mode,
    artifactsPath: runDir,
    resetArtifactPath,
    ...(options.mode === 'last-known-good' ? { activeStatePath } : {}),
  };
}

export async function runForecastLab(options: ForecastLabRunOptions): Promise<ForecastLabRunResult> {
  const profile = getForecastLabProfile(options.profileId);
  const now = options.now ?? (() => new Date());
  const startedAt = now().toISOString();
  const runId = options.runId ?? makeRunId(profile.id, new Date(startedAt));
  const mutationPlan = resolveMutationPlan(options);
  const dryRun = mutationPlan.dryRun;
  const skipMutation = mutationPlan.skipMutation;
  const progress = options.progress;
  const output = options.output;
  const forecastingConfig = loadConfig().forecasting;

  const candidateBranch = makeForecastLabCandidateBranch(runId);
  const baselineCommit = getForecastLabBaselineCommit();
  const runDir = getExperimentRunDir(runId, { create: true });
  const manifestPath = getExperimentRunManifestPath(runId, { create: true });
  const baselinePath = `${runDir}/baseline.json`;
  const candidatePath = `${runDir}/candidate.json`;
  const decisionPath = `${runDir}/decision.json`;
  const ledgerPath = options.ledgerPath ?? getExperimentLedgerPath({ create: true });
  const effectiveMutationContract = snapshotEffectiveMutationContract(profile);
  const mutationCatalog = mutationPlan.runRealMutation ? getStructuredMutationCatalog(profile) : undefined;
  let ledgerEntries: readonly ForecastLabLedgerEntry[] = [];
  let mutationSeed: ForecastLabStructuredMutationSeed | undefined;

  for (const path of [runDir, manifestPath, baselinePath, candidatePath, decisionPath, ledgerPath]) {
    assertInsideExperiments(path);
  }
  if (mutationPlan.runRealMutation) {
    ledgerEntries = readLedgerEntries(ledgerPath);
    mutationSeed = options.forceNoParent ? undefined : readStructuredMutationSeed(profile, ledgerEntries);
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
  if (options.routingContext) {
    manifest.routingContext = options.routingContext;
  }

  writeRunManifest(manifestPath, manifest);
  progress?.(`forecast-lab: started ${profile.id} (${runId})`);
  progress?.(`forecast-lab: manifest written to ${manifestPath}`);

  const commandRunner = options.commandRunner ?? defaultForecastLabCommandRunner;
  const workspaceMaterializePaths = options.commandRunner ? profile.mutation.mutableFiles : undefined;
  const lightweightWorkspace = options.commandRunner !== undefined;
  const baselineCwd = process.cwd();
  const runBaselineGate = async (): Promise<ForecastLabGateSummary> => {
    progress?.('forecast-lab: starting baseline gate');
    const baseline = await runGate('baseline', profile, runId, commandRunner, baselineCwd, progress, output);
    writeJsonArtifact(baselinePath, baseline);
    progress?.(`forecast-lab: baseline results written to ${baselinePath}`);
    return baseline;
  };

  let baseline: ForecastLabGateSummary;

  let candidate: ForecastLabGateSummary;
  let structuredMutation: ForecastLabStructuredMutationSelection | undefined;

  if (mutationPlan.runRealMutation) {
    const executeCandidateRun = async (
      workspace: ReturnType<typeof prepareForecastLabCandidateWorkspace>,
    ) => {
      const mutatorRankingEnabled = resolveForecastLabMutatorRankingEnabled(
        options.rankMutators,
        forecastingConfig,
      );
      const mutatorRanking = mutatorRankingEnabled && mutationPlan.mutator === undefined
        ? rankForecastLabMutators({
            profileId: profile.id,
            catalog: mutationCatalog!,
            ledgerEntries,
            usedMutationIds: mutationSeed?.usedMutationIds,
            isApplicable: (candidate) => canApplyStructuredMutation(
              workspace.metadata.rootDir,
              mutationSeed?.replayedMutations ?? [],
              candidate,
            ),
          })
        : undefined;
      const selectedMutation = selectStructuredMutation(
        profile,
        mutationCatalog!,
        mutationPlan.mutator,
        mutationSeed?.usedMutationIds,
        (candidate) => canApplyStructuredMutation(
          workspace.metadata.rootDir,
          mutationSeed?.replayedMutations ?? [],
          candidate,
        ),
        mutatorRanking,
      );
      const mutation = applyStructuredMutation(
        workspace.metadata.rootDir,
        profile,
        selectedMutation,
        runId,
        mutationSeed,
        mutatorRanking,
      );
      manifest.mutationMode = mutation.mutationMode;
      manifest.mutationId = mutation.mutationId;
      manifest.mutationSummary = mutation.mutationSummary;
      manifest.lineage = mutation.lineage;
      manifest.mutationSpecSummary = mutation.mutationSpecSummary;
      manifest.mutationReplayPayload = mutation.mutationReplayPayload;
      manifest.candidateWorkspace = workspace.metadata;
      if (mutation.parentRunId !== undefined) {
        manifest.parentRunId = mutation.parentRunId;
      } else {
        delete manifest.parentRunId;
      }
      writeRunManifest(manifestPath, manifest);

      progress?.(`forecast-lab: candidate workspace ${workspace.metadata.rootDir}`);
      if (mutationSeed) {
        progress?.(
          `forecast-lab: seeded from kept run ${mutationSeed.parentRunId} (${mutationSeed.replayedMutations.length} replayed mutation${mutationSeed.replayedMutations.length === 1 ? '' : 's'})`,
        );
      }
      if (mutatorRanking) {
        progress?.(formatForecastLabMutatorHealthProgress(mutatorRanking));
      }
      progress?.(`forecast-lab: selected mutator ${mutation.selectedMutatorId} (${mutation.mutatorId})`);
      progress?.(`forecast-lab: mutated files ${mutation.mutatedFiles.join(', ')}`);
      progress?.(`forecast-lab: patch summary ${mutation.patchSummary.join(' | ')}`);
      const baseline = await runBaselineGate();
      progress?.('forecast-lab: starting candidate gate (structured mutation)');

      const gate = await runGate('candidate', profile, runId, commandRunner, workspace.metadata.rootDir, progress, output);
      return { baseline, gate, mutation };
    };

    const candidateRun = mutationPlan.keepWorktree
      ? await (async () => {
        const workspace = prepareForecastLabCandidateWorkspace(runId, {
          materializePaths: workspaceMaterializePaths,
          lightweight: lightweightWorkspace,
        });

        try {
          const result = await executeCandidateRun(workspace);
          progress?.(`forecast-lab: keeping candidate workspace ${workspace.metadata.rootDir}`);
          return result;
        } catch (error) {
          progress?.(`forecast-lab: keeping candidate workspace ${workspace.metadata.rootDir} for debugging after failure`);
          throw error;
        }
      })()
      : await withForecastLabCandidateWorkspace(runId, executeCandidateRun, {
        materializePaths: workspaceMaterializePaths,
        lightweight: lightweightWorkspace,
      });

    baseline = candidateRun.baseline;
    candidate = candidateRun.gate;
    structuredMutation = candidateRun.mutation;
  } else {
    baseline = await runBaselineGate();
    progress?.(`forecast-lab: starting candidate gate (${dryRun ? 'dry-run' : 'skip-mutation'})`);
    candidate = await runGate('candidate', profile, runId, commandRunner, process.cwd(), progress, output);
  }

  const candidateArtifact = buildCandidateArtifact(candidate, manifest, structuredMutation, dryRun);

  writeJsonArtifact(candidatePath, candidateArtifact);
  progress?.(`forecast-lab: candidate results written to ${candidatePath}`);

  const measuredDecision = decideRun(profile, baseline, candidate);
  const decision = skipMutation ? dropSkippedMutation(measuredDecision) : measuredDecision;
  const promotion = !options.diagnosticOnly && decision.decision === 'keep'
    ? buildPendingPromotionState(manifest, manifestPath, now().toISOString())
    : undefined;
  if (promotion) {
    manifest.promotion = promotion;
    writeRunManifest(manifestPath, manifest);
  }
  writeJsonArtifact(decisionPath, structuredMutation
    ? {
        ...decision,
        mutationMode: structuredMutation.mutationMode,
        mutationId: structuredMutation.mutationId,
        mutationSummary: structuredMutation.mutationSummary,
        lineage: structuredMutation.lineage,
        mutationSpecSummary: structuredMutation.mutationSpecSummary,
        candidateWorkspace: manifest.candidateWorkspace,
        ...(structuredMutation.parentRunId !== undefined ? { parentRunId: structuredMutation.parentRunId } : {}),
        selectedMutator: {
          id: structuredMutation.selectedMutatorId,
          mutatorId: structuredMutation.mutatorId,
        },
        mutatedFiles: [...structuredMutation.mutatedFiles],
        patchSummary: [...structuredMutation.patchSummary],
        ...(promotion ? { promotion } : {}),
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
    routingContext: options.routingContext,
    effectiveMutationContract,
    mutationMode: structuredMutation?.mutationMode,
    parentRunId: structuredMutation?.parentRunId,
    mutationId: structuredMutation?.mutationId,
    mutationSummary: structuredMutation?.mutationSummary,
    lineage: structuredMutation?.lineage,
    mutationSpecSummary: structuredMutation?.mutationSpecSummary,
    candidateWorkspace: manifest.candidateWorkspace,
    promotion,
    baselineSummary: summarizeForLedger(baseline),
    candidateSummary: summarizeForLedger(candidate),
    decision: decision.decision,
    reason: decision.reason,
    artifactsPath: runDir,
  };

  if (!options.diagnosticOnly) {
    appendLedgerEntry(ledgerPath, ledgerEntry);
    if (options.routingContext) {
      updateForecastLabRoutingStats(
        profile.id,
        decision.decision,
        startedAt,
        options.routingContext.invocationSource,
        options.routingStatsPath,
      );
    }
    progress?.(`forecast-lab: ledger appended at ${ledgerPath}`);
  } else {
    progress?.(`forecast-lab: diagnostic-only mode, skipping ledger and routing stats`);
  }

  return {
    runId,
    manifest,
    baseline: toJsonValue(baseline, 'run baseline summary'),
    candidate: candidateArtifact,
    decision,
    ledgerEntry,
  };
}
