import {
  assertForecastLabMutatorId,
  validateForecastLabMutationSpecSummary,
} from '../mutation.js';
import type { ForecastLabMutatorId, ForecastLabMutationSpecSummary } from '../mutation.js';
import {
  FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS,
} from '../../../tools/finance/conformal.js';
import {
  FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS,
} from '../../../tools/finance/markov-distribution.js';
import {
  FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS,
} from '../../../tools/finance/regime-calibrator.js';

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

export type ForecastLabMarkovMutatorProfileId =
  | 'multi-asset-markov-short-horizon'
  | 'btc-markov-ultra-short-horizon'
  | 'gold-markov-short-horizon';

export type ForecastLabMutationScalarValue = boolean | number;

export interface ForecastLabMarkovParameterMutationEdit {
  readonly kind: 'search-replace';
  readonly parameterId: string;
  readonly filePath: string;
  readonly beforeValue: ForecastLabMutationScalarValue;
  readonly afterValue: ForecastLabMutationScalarValue;
  readonly search: string;
  readonly replace: string;
  readonly expectedReplacements: 1;
}

export interface ForecastLabMarkovParameterMutationCandidate {
  readonly id: string;
  readonly profileId: ForecastLabMarkovMutatorProfileId;
  readonly mutatorId: ForecastLabMutatorId;
  readonly specSummary: ForecastLabMutationSpecSummary;
  readonly patchSummary: readonly string[];
  readonly edits: readonly ForecastLabMarkovParameterMutationEdit[];
}

export interface ForecastLabMarkovParameterMutationReplayPayload
  extends ForecastLabMarkovParameterMutationCandidate {
  readonly kind: 'markov-parameter-candidate';
}

const MARKOV_FILE = 'src/tools/finance/markov-distribution.ts';
const CONFORMAL_FILE = 'src/tools/finance/conformal.ts';
const REGIME_CALIBRATOR_FILE = 'src/tools/finance/regime-calibrator.ts';

function scalarLiteral(value: ForecastLabMutationScalarValue): string {
  return typeof value === 'boolean' ? String(value) : `${value}`;
}

function buildEdit(params: {
  readonly filePath: string;
  readonly parameterId: string;
  readonly beforeValue: ForecastLabMutationScalarValue;
  readonly afterValue: ForecastLabMutationScalarValue;
}): ForecastLabMarkovParameterMutationEdit {
  return {
    kind: 'search-replace',
    parameterId: params.parameterId,
    filePath: params.filePath,
    beforeValue: params.beforeValue,
    afterValue: params.afterValue,
    search: `  ${params.parameterId}: ${scalarLiteral(params.beforeValue)},`,
    replace: `  ${params.parameterId}: ${scalarLiteral(params.afterValue)},`,
    expectedReplacements: 1,
  };
}

function buildCandidate(params: {
  readonly id: string;
  readonly profileId: ForecastLabMarkovMutatorProfileId;
  readonly summary: string;
  readonly patchSummary: readonly string[];
  readonly edits: readonly ForecastLabMarkovParameterMutationEdit[];
}): ForecastLabMarkovParameterMutationCandidate {
  const targetFiles = [...new Set(params.edits.map((edit) => edit.filePath))];
  return deepFreeze({
    id: params.id,
    profileId: params.profileId,
    mutatorId: 'search-replace',
    specSummary: {
      mutatorId: 'search-replace',
      targetFiles,
      summary: params.summary,
    },
    patchSummary: [...params.patchSummary],
    edits: [...params.edits],
  });
}

function cloneCandidate(
  candidate: ForecastLabMarkovParameterMutationCandidate,
): ForecastLabMarkovParameterMutationCandidate {
  return deepFreeze({
    id: candidate.id,
    profileId: candidate.profileId,
    mutatorId: candidate.mutatorId,
    specSummary: {
      mutatorId: candidate.specSummary.mutatorId,
      targetFiles: [...candidate.specSummary.targetFiles],
      summary: candidate.specSummary.summary,
    },
    patchSummary: [...candidate.patchSummary],
    edits: candidate.edits.map((edit) => ({
      kind: edit.kind,
      parameterId: edit.parameterId,
      filePath: edit.filePath,
      beforeValue: edit.beforeValue,
      afterValue: edit.afterValue,
      search: edit.search,
      replace: edit.replace,
      expectedReplacements: edit.expectedReplacements,
    })),
  });
}

function requireNonEmptyString(record: Record<string, unknown>, field: string): string {
  const value = record[field];
  if (typeof value !== 'string' || value.trim().length === 0) {
    throw new Error(`${field} must be a non-empty string`);
  }
  return value;
}

function validateScalarValue(field: string, value: unknown): asserts value is ForecastLabMutationScalarValue {
  if (typeof value === 'boolean') {
    return;
  }

  if (typeof value === 'number' && Number.isFinite(value)) {
    return;
  }

  throw new Error(`${field} must be a finite number or boolean`);
}

function validateEdit(edit: unknown): asserts edit is ForecastLabMarkovParameterMutationEdit {
  if (!edit || typeof edit !== 'object') {
    throw new Error('mutation edit must be an object');
  }

  const record = edit as Record<string, unknown>;
  if (record.kind !== 'search-replace') {
    throw new Error('mutation edit kind must be "search-replace"');
  }
  requireNonEmptyString(record, 'parameterId');
  requireNonEmptyString(record, 'filePath');
  requireNonEmptyString(record, 'search');
  if (typeof record.replace !== 'string') {
    throw new Error('replace must be a string');
  }
  validateScalarValue('beforeValue', record.beforeValue);
  validateScalarValue('afterValue', record.afterValue);
  if (record.expectedReplacements !== 1) {
    throw new Error('expectedReplacements must be 1');
  }
}

export function validateForecastLabMarkovParameterMutationCandidate(
  candidate: unknown,
): asserts candidate is ForecastLabMarkovParameterMutationCandidate {
  if (!candidate || typeof candidate !== 'object') {
    throw new Error('mutation candidate must be an object');
  }

  const record = candidate as Record<string, unknown>;
  requireNonEmptyString(record, 'id');
  const profileId = requireNonEmptyString(record, 'profileId');
  if (!isForecastLabMarkovMutatorProfileId(profileId)) {
    throw new Error(`Unknown markov mutation profile: ${profileId}`);
  }

  const mutatorId = requireNonEmptyString(record, 'mutatorId');
  assertForecastLabMutatorId(mutatorId);
  validateForecastLabMutationSpecSummary(record.specSummary);
  if (record.specSummary.mutatorId !== mutatorId) {
    throw new Error('specSummary.mutatorId must match mutatorId');
  }

  if (!Array.isArray(record.patchSummary) || record.patchSummary.some((entry) => typeof entry !== 'string')) {
    throw new Error('patchSummary must be an array of strings');
  }

  if (!Array.isArray(record.edits) || record.edits.length === 0) {
    throw new Error('edits must contain at least one edit');
  }
  for (const edit of record.edits) {
    validateEdit(edit);
  }
}

export function validateForecastLabMarkovParameterMutationReplayPayload(
  payload: unknown,
): asserts payload is ForecastLabMarkovParameterMutationReplayPayload {
  if (!payload || typeof payload !== 'object') {
    throw new Error('mutationReplayPayload must be an object');
  }

  const record = payload as Record<string, unknown>;
  if (record.kind !== 'markov-parameter-candidate') {
    throw new Error('mutationReplayPayload.kind must be "markov-parameter-candidate"');
  }
  validateForecastLabMarkovParameterMutationCandidate(record);
}

export function snapshotForecastLabMarkovParameterMutation(
  candidate: ForecastLabMarkovParameterMutationCandidate,
): ForecastLabMarkovParameterMutationReplayPayload {
  return deepFreeze({
    kind: 'markov-parameter-candidate',
    ...cloneCandidate(candidate),
  });
}

export function replayForecastLabMarkovParameterMutation(
  payload: ForecastLabMarkovParameterMutationReplayPayload,
): ForecastLabMarkovParameterMutationCandidate {
  validateForecastLabMarkovParameterMutationReplayPayload(payload);
  return cloneCandidate(payload);
}

function buildCatalog(profileId: ForecastLabMarkovMutatorProfileId): readonly ForecastLabMarkovParameterMutationCandidate[] {
  return deepFreeze([
    buildCandidate({
      id: 'markov-shorter-reactive-window',
      profileId,
      summary: 'Shorten Markov/conformal calibration windows for faster short-horizon adaptation.',
      patchSummary: [
        `markov-distribution.ts: momentumLookback ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback} → 10`,
        `markov-distribution.ts: structuralBreakMinLength ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength} → 36`,
        `conformal.ts: scoreAggregationMinSamples ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples} → 12`,
        `conformal.ts: scoreAggregationCalibrationWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow} → 72`,
        `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → 18`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'momentumLookback',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback,
          afterValue: 10,
        }),
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'structuralBreakMinLength',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength,
          afterValue: 36,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationMinSamples',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples,
          afterValue: 12,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationCalibrationWindow',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow,
          afterValue: 72,
        }),
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'minSamplesPerRegime',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
          afterValue: 18,
        }),
      ],
    }),
    buildCandidate({
      id: 'markov-longer-stability-window',
      profileId,
      summary: 'Lengthen Markov/conformal calibration windows to favor stabler short-horizon fits.',
      patchSummary: [
        `markov-distribution.ts: momentumLookback ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback} → 28`,
        `markov-distribution.ts: structuralBreakMinLength ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength} → 72`,
        `conformal.ts: scoreAggregationMinSamples ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples} → 28`,
        `conformal.ts: scoreAggregationCalibrationWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow} → 144`,
        `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → 36`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'momentumLookback',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback,
          afterValue: 28,
        }),
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'structuralBreakMinLength',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength,
          afterValue: 72,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationMinSamples',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples,
          afterValue: 28,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationCalibrationWindow',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow,
          afterValue: 144,
        }),
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'minSamplesPerRegime',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
          afterValue: 36,
        }),
      ],
    }),
    buildCandidate({
      id: 'markov-faster-decay-reaction',
      profileId,
      summary: 'Lower transition decay and raise adaptive conformal sensitivity for quicker regime resets.',
      patchSummary: [
        `markov-distribution.ts: transitionDecay ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionDecay} → 0.94`,
        `conformal.ts: adaptiveBreakLearningRateMultiplier ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakLearningRateMultiplier} → 1.75`,
        `conformal.ts: adaptiveBreakCooloffWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakCooloffWindow} → 2`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'transitionDecay',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionDecay,
          afterValue: 0.94,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'adaptiveBreakLearningRateMultiplier',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakLearningRateMultiplier,
          afterValue: 1.75,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'adaptiveBreakCooloffWindow',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakCooloffWindow,
          afterValue: 2,
        }),
      ],
    }),
    buildCandidate({
      id: 'markov-slower-decay-persistence',
      profileId,
      summary: 'Raise transition decay and soften adaptive conformal sensitivity for stickier regime persistence.',
      patchSummary: [
        `markov-distribution.ts: transitionDecay ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionDecay} → 0.985`,
        `conformal.ts: adaptiveBreakLearningRateMultiplier ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakLearningRateMultiplier} → 1.25`,
        `conformal.ts: adaptiveBreakCooloffWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakCooloffWindow} → 1`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'transitionDecay',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionDecay,
          afterValue: 0.985,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'adaptiveBreakLearningRateMultiplier',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakLearningRateMultiplier,
          afterValue: 1.25,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'adaptiveBreakCooloffWindow',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakCooloffWindow,
          afterValue: 1,
        }),
      ],
    }),
    buildCandidate({
      id: 'markov-lower-confidence-trend-penalty',
      profileId,
      summary: 'Lower the confidence gate while keeping the trend-only break penalty path active.',
      patchSummary: [
        `markov-distribution.ts: recommendedConfidenceThreshold ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold} → 0.18`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'recommendedConfidenceThreshold',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold,
          afterValue: 0.18,
        }),
      ],
    }),
    buildCandidate({
      id: 'markov-higher-confidence-divergence-weighted',
      profileId,
      summary: 'Raise the confidence gate and enable divergence-weighted break penalties.',
      patchSummary: [
        `markov-distribution.ts: recommendedConfidenceThreshold ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold} → 0.3`,
        `markov-distribution.ts: divergenceWeightedBreakConfidence ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.divergenceWeightedBreakConfidence} → true`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'recommendedConfidenceThreshold',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold,
          afterValue: 0.3,
        }),
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'divergenceWeightedBreakConfidence',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.divergenceWeightedBreakConfidence,
          afterValue: true,
        }),
      ],
    }),
    buildCandidate({
      id: 'markov-calibrator-higher-sample-floor',
      profileId,
      summary: 'Tighten calibrator sample floors and reduce the Platt learning rate.',
      patchSummary: [
        `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → 40`,
        `regime-calibrator.ts: learningRate ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.learningRate} → 0.035`,
      ],
      edits: [
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'minSamplesPerRegime',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
          afterValue: 40,
        }),
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'learningRate',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.learningRate,
          afterValue: 0.035,
        }),
      ],
    }),
    buildCandidate({
      id: 'markov-calibrator-lower-sample-floor',
      profileId,
      summary: 'Relax calibrator sample floors and increase the Platt learning rate.',
      patchSummary: [
        `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → 20`,
        `regime-calibrator.ts: learningRate ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.learningRate} → 0.075`,
      ],
      edits: [
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'minSamplesPerRegime',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
          afterValue: 20,
        }),
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'learningRate',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.learningRate,
          afterValue: 0.075,
        }),
      ],
    }),
  ]);
}

function buildGoldCatalog(): readonly ForecastLabMarkovParameterMutationCandidate[] {
  return deepFreeze([
    buildCandidate({
      id: 'gold-markov-shorter-reactive-window',
      profileId: 'gold-markov-short-horizon',
      summary: 'Tighten GOLD 1d/2d/3d adaptation windows for faster short-horizon recalibration.',
      patchSummary: [
        `markov-distribution.ts: momentumLookback ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback} → 12`,
        `markov-distribution.ts: structuralBreakMinLength ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength} → 40`,
        `conformal.ts: scoreAggregationMinSamples ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples} → 14`,
        `conformal.ts: scoreAggregationCalibrationWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow} → 84`,
        `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → 20`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'momentumLookback',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback,
          afterValue: 12,
        }),
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'structuralBreakMinLength',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength,
          afterValue: 40,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationMinSamples',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples,
          afterValue: 14,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationCalibrationWindow',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow,
          afterValue: 84,
        }),
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'minSamplesPerRegime',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
          afterValue: 20,
        }),
      ],
    }),
    buildCandidate({
      id: 'gold-markov-longer-stability-window',
      profileId: 'gold-markov-short-horizon',
      summary: 'Lengthen GOLD calibration windows to favor stabler 1d/2d/3d fits without relying on 7d/14d.',
      patchSummary: [
        `markov-distribution.ts: momentumLookback ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback} → 24`,
        `markov-distribution.ts: structuralBreakMinLength ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength} → 64`,
        `conformal.ts: scoreAggregationMinSamples ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples} → 24`,
        `conformal.ts: scoreAggregationCalibrationWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow} → 132`,
        `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → 32`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'momentumLookback',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback,
          afterValue: 24,
        }),
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'structuralBreakMinLength',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength,
          afterValue: 64,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationMinSamples',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples,
          afterValue: 24,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationCalibrationWindow',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow,
          afterValue: 132,
        }),
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'minSamplesPerRegime',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
          afterValue: 32,
        }),
      ],
    }),
    buildCandidate({
      id: 'gold-markov-faster-decay-reaction',
      profileId: 'gold-markov-short-horizon',
      summary: 'Lower GOLD transition decay and modestly raise adaptive conformal sensitivity for quicker 1d resets.',
      patchSummary: [
        `markov-distribution.ts: transitionDecay ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionDecay} → 0.95`,
        `conformal.ts: adaptiveBreakLearningRateMultiplier ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakLearningRateMultiplier} → 1.65`,
        `conformal.ts: adaptiveBreakCooloffWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakCooloffWindow} → 1`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'transitionDecay',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionDecay,
          afterValue: 0.95,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'adaptiveBreakLearningRateMultiplier',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakLearningRateMultiplier,
          afterValue: 1.65,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'adaptiveBreakCooloffWindow',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakCooloffWindow,
          afterValue: 1,
        }),
      ],
    }),
    buildCandidate({
      id: 'gold-markov-slower-decay-persistence',
      profileId: 'gold-markov-short-horizon',
      summary: 'Raise GOLD transition decay and soften adaptive break sensitivity for steadier short-horizon persistence.',
      patchSummary: [
        `markov-distribution.ts: transitionDecay ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionDecay} → 0.982`,
        `conformal.ts: adaptiveBreakLearningRateMultiplier ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakLearningRateMultiplier} → 1.15`,
        `conformal.ts: adaptiveBreakCooloffWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakCooloffWindow} → 2`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'transitionDecay',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionDecay,
          afterValue: 0.982,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'adaptiveBreakLearningRateMultiplier',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakLearningRateMultiplier,
          afterValue: 1.15,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'adaptiveBreakCooloffWindow',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakCooloffWindow,
          afterValue: 2,
        }),
      ],
    }),
    buildCandidate({
      id: 'gold-markov-lower-confidence-trend-penalty',
      profileId: 'gold-markov-short-horizon',
      summary: 'Lower the GOLD confidence gate while keeping trend-weighted break penalties conservative.',
      patchSummary: [
        `markov-distribution.ts: recommendedConfidenceThreshold ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold} → 0.2`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'recommendedConfidenceThreshold',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold,
          afterValue: 0.2,
        }),
      ],
    }),
    buildCandidate({
      id: 'gold-markov-higher-confidence-divergence-weighted',
      profileId: 'gold-markov-short-horizon',
      summary: 'Raise the GOLD confidence gate and enable divergence-weighted break penalties.',
      patchSummary: [
        `markov-distribution.ts: recommendedConfidenceThreshold ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold} → 0.28`,
        `markov-distribution.ts: divergenceWeightedBreakConfidence ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.divergenceWeightedBreakConfidence} → true`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'recommendedConfidenceThreshold',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold,
          afterValue: 0.28,
        }),
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'divergenceWeightedBreakConfidence',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.divergenceWeightedBreakConfidence,
          afterValue: true,
        }),
      ],
    }),
    buildCandidate({
      id: 'gold-markov-calibrator-higher-sample-floor',
      profileId: 'gold-markov-short-horizon',
      summary: 'Raise GOLD calibrator sample floors while slightly slowing the Platt update rate.',
      patchSummary: [
        `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → 34`,
        `regime-calibrator.ts: learningRate ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.learningRate} → 0.04`,
      ],
      edits: [
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'minSamplesPerRegime',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
          afterValue: 34,
        }),
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'learningRate',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.learningRate,
          afterValue: 0.04,
        }),
      ],
    }),
    buildCandidate({
      id: 'gold-markov-calibrator-lower-sample-floor',
      profileId: 'gold-markov-short-horizon',
      summary: 'Lower GOLD calibrator sample floors and slightly increase the Platt update rate.',
      patchSummary: [
        `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → 18`,
        `regime-calibrator.ts: learningRate ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.learningRate} → 0.065`,
      ],
      edits: [
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'minSamplesPerRegime',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
          afterValue: 18,
        }),
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'learningRate',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.learningRate,
          afterValue: 0.065,
        }),
      ],
    }),
  ]);
}

const CATALOG_BY_PROFILE = deepFreeze({
  'multi-asset-markov-short-horizon': buildCatalog('multi-asset-markov-short-horizon'),
  'btc-markov-ultra-short-horizon': buildCatalog('btc-markov-ultra-short-horizon'),
  'gold-markov-short-horizon': buildGoldCatalog(),
} as const satisfies Record<
  ForecastLabMarkovMutatorProfileId,
  readonly ForecastLabMarkovParameterMutationCandidate[]
>);

export function isForecastLabMarkovMutatorProfileId(
  profileId: string,
): profileId is ForecastLabMarkovMutatorProfileId {
  return Object.hasOwn(CATALOG_BY_PROFILE, profileId);
}

export function listMarkovParameterMutations(
  profileId: ForecastLabMarkovMutatorProfileId,
): readonly ForecastLabMarkovParameterMutationCandidate[] {
  return CATALOG_BY_PROFILE[profileId];
}
