type Primitive = null | undefined | boolean | number | string | symbol | bigint;
type DeepReadonly<T> = T extends Primitive
  ? T
  : T extends readonly (infer U)[]
    ? readonly DeepReadonly<U>[]
    : { readonly [K in keyof T]: DeepReadonly<T[K]> };

function deepFreeze<T>(value: T): DeepReadonly<T> {
  if (value && typeof value === 'object' && !Object.isFrozen(value)) {
    for (const nested of Object.values(value as Record<string, unknown>)) {
      deepFreeze(nested);
    }
    Object.freeze(value);
  }

  return value as DeepReadonly<T>;
}

export type ForecastLabMutationMode = 'dry-run' | 'structured' | 'llm';

export const FORECAST_LAB_MUTATOR_IDS = deepFreeze([
  'replace-range',
  'search-replace',
  'insert-block',
] as const);

export type ForecastLabMutatorId = (typeof FORECAST_LAB_MUTATOR_IDS)[number];

export interface ForecastLabMutationLineage {
  readonly rootRunId: string;
  readonly parentRunId?: string;
  readonly generation: number;
}

export interface ForecastLabMutationSpecSummary {
  readonly mutatorId: ForecastLabMutatorId;
  readonly targetFiles: readonly string[];
  readonly summary: string;
}

export interface ForecastLabCandidateWorkspaceMetadata {
  readonly kind: 'current-worktree' | 'candidate-worktree';
  readonly rootDir: string;
  readonly branch: string;
}

interface ForecastLabBaseMutationConfig {
  readonly mutableFiles: readonly string[];
  readonly allowMultipleCandidateAttempts: boolean;
}

export interface ForecastLabStructuredMutationConfig extends ForecastLabBaseMutationConfig {
  readonly mode: 'structured';
  readonly allowedMutatorIds: readonly ForecastLabMutatorId[];
}

export interface ForecastLabNonStructuredMutationConfig extends ForecastLabBaseMutationConfig {
  readonly mode: 'dry-run' | 'llm';
}

export type ForecastLabProfileMutationConfig =
  | ForecastLabStructuredMutationConfig
  | ForecastLabNonStructuredMutationConfig;

export class ForecastLabMutationError extends Error {
  override name = 'ForecastLabMutationError';
}

export function isForecastLabMutationMode(mode: string): mode is ForecastLabMutationMode {
  return mode === 'dry-run' || mode === 'structured' || mode === 'llm';
}

export function assertForecastLabMutationMode(mode: string): asserts mode is ForecastLabMutationMode {
  if (!isForecastLabMutationMode(mode)) {
    throw new ForecastLabMutationError(`Unknown forecast-lab mutation mode: ${mode}. Expected one of: dry-run, structured, llm`);
  }
}

export function listForecastLabMutatorIds(): readonly ForecastLabMutatorId[] {
  return FORECAST_LAB_MUTATOR_IDS;
}

export function isForecastLabMutatorId(mutatorId: string): mutatorId is ForecastLabMutatorId {
  return (FORECAST_LAB_MUTATOR_IDS as readonly string[]).includes(mutatorId);
}

export function assertForecastLabMutatorId(mutatorId: string): asserts mutatorId is ForecastLabMutatorId {
  if (!isForecastLabMutatorId(mutatorId)) {
    throw new ForecastLabMutationError(
      `Unknown forecast-lab mutator id: ${mutatorId}. Expected one of: ${FORECAST_LAB_MUTATOR_IDS.join(', ')}`,
    );
  }
}

function assertNonEmptyFiles(field: string, files: readonly string[]): void {
  if (files.length === 0) {
    throw new ForecastLabMutationError(`${field} must contain at least one file`);
  }

  for (const file of files) {
    if (file.trim().length === 0) {
      throw new ForecastLabMutationError(`${field} must not contain empty file paths`);
    }
  }
}

function requireNonEmptyString(record: Record<string, unknown>, field: string): string {
  const value = record[field];

  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new ForecastLabMutationError(`${field} must be a non-empty string`);
  }

  return value;
}

export function validateForecastLabMutationLineage(lineage: unknown): asserts lineage is ForecastLabMutationLineage {
  if (!lineage || typeof lineage !== 'object') {
    throw new ForecastLabMutationError('lineage must be an object');
  }

  const record = lineage as Record<string, unknown>;
  requireNonEmptyString(record, 'rootRunId');

  if (record.parentRunId !== undefined) {
    requireNonEmptyString(record, 'parentRunId');
  }

  if (!Number.isInteger(record.generation) || (record.generation as number) < 0) {
    throw new ForecastLabMutationError('generation must be a non-negative integer');
  }
}

export function validateForecastLabMutationSpecSummary(summary: unknown): asserts summary is ForecastLabMutationSpecSummary {
  if (!summary || typeof summary !== 'object') {
    throw new ForecastLabMutationError('mutationSpecSummary must be an object');
  }

  const record = summary as Record<string, unknown>;
  const mutatorId = requireNonEmptyString(record, 'mutatorId');
  assertForecastLabMutatorId(mutatorId);

  if (!Array.isArray(record.targetFiles) || record.targetFiles.some((file) => typeof file !== 'string')) {
    throw new ForecastLabMutationError('targetFiles must be an array of strings');
  }
  assertNonEmptyFiles('targetFiles', record.targetFiles);
  requireNonEmptyString(record, 'summary');
}

export function validateForecastLabCandidateWorkspaceMetadata(
  metadata: unknown,
): asserts metadata is ForecastLabCandidateWorkspaceMetadata {
  if (!metadata || typeof metadata !== 'object') {
    throw new ForecastLabMutationError('candidateWorkspace must be an object');
  }

  const record = metadata as Record<string, unknown>;

  if (record.kind !== 'current-worktree' && record.kind !== 'candidate-worktree') {
    throw new ForecastLabMutationError('candidateWorkspace.kind must be current-worktree or candidate-worktree');
  }

  requireNonEmptyString(record, 'rootDir');
  requireNonEmptyString(record, 'branch');
}

export function validateForecastLabProfileMutationConfig(
  config: unknown,
): asserts config is ForecastLabProfileMutationConfig {
  if (!config || typeof config !== 'object') {
    throw new ForecastLabMutationError('effectiveMutationContract must be an object');
  }

  const record = config as Record<string, unknown>;

  if (typeof record.allowMultipleCandidateAttempts !== 'boolean') {
    throw new ForecastLabMutationError('allowMultipleCandidateAttempts must be a boolean');
  }

  if (!Array.isArray(record.mutableFiles) || record.mutableFiles.some((file) => typeof file !== 'string')) {
    throw new ForecastLabMutationError('mutableFiles must be an array of strings');
  }
  assertNonEmptyFiles('mutableFiles', record.mutableFiles);

  const mode = requireNonEmptyString(record, 'mode');
  assertForecastLabMutationMode(mode);

  if (mode !== 'structured') {
    if (record.allowedMutatorIds !== undefined) {
      throw new ForecastLabMutationError(`allowedMutatorIds is only supported for structured mutation mode: ${mode}`);
    }
    return;
  }

  if (!Array.isArray(record.allowedMutatorIds) || record.allowedMutatorIds.some((mutatorId) => typeof mutatorId !== 'string')) {
    throw new ForecastLabMutationError('allowedMutatorIds must be an array of strings');
  }

  if (record.allowedMutatorIds.length === 0) {
    throw new ForecastLabMutationError('allowedMutatorIds must contain at least one mutator id');
  }

  for (const mutatorId of record.allowedMutatorIds) {
    assertForecastLabMutatorId(mutatorId);
  }
}

export function defineForecastLabProfileMutationConfig(config: {
  readonly mode: 'structured';
  readonly mutableFiles: readonly string[];
  readonly allowedMutatorIds: readonly string[];
  readonly allowMultipleCandidateAttempts?: boolean;
}): ForecastLabStructuredMutationConfig;
export function defineForecastLabProfileMutationConfig(config: {
  readonly mode: 'dry-run' | 'llm';
  readonly mutableFiles: readonly string[];
  readonly allowMultipleCandidateAttempts?: boolean;
}): ForecastLabNonStructuredMutationConfig;
export function defineForecastLabProfileMutationConfig(config: {
  readonly mode: ForecastLabMutationMode;
  readonly mutableFiles: readonly string[];
  readonly allowedMutatorIds?: readonly string[];
  readonly allowMultipleCandidateAttempts?: boolean;
}): ForecastLabProfileMutationConfig {
  assertNonEmptyFiles('mutableFiles', config.mutableFiles);

  if (config.mode !== 'structured') {
    if (config.allowedMutatorIds !== undefined) {
      throw new ForecastLabMutationError(`allowedMutatorIds is only supported for structured mutation mode: ${config.mode}`);
    }

    return deepFreeze({
      mode: config.mode,
      mutableFiles: [...config.mutableFiles],
      allowMultipleCandidateAttempts: config.allowMultipleCandidateAttempts === true,
    });
  }

  if (config.allowedMutatorIds === undefined || config.allowedMutatorIds.length === 0) {
    throw new ForecastLabMutationError('allowedMutatorIds must contain at least one mutator id');
  }

  const allowedMutatorIds = config.allowedMutatorIds.map((mutatorId) => {
    assertForecastLabMutatorId(mutatorId);
    return mutatorId;
  });

  return deepFreeze({
    mode: 'structured',
    mutableFiles: [...config.mutableFiles],
    allowedMutatorIds,
    allowMultipleCandidateAttempts: config.allowMultipleCandidateAttempts === true,
  });
}
