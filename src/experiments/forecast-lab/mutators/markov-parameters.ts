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
  | 'sol-markov-short-horizon'
  | 'hype-markov-short-horizon'
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

const MARKOV_FILE = 'src/tools/finance/markov-distribution/core.ts';
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
  readonly searchLiteral?: string;
  readonly replaceLiteral?: string;
}): ForecastLabMarkovParameterMutationEdit {
  return {
    kind: 'search-replace',
    parameterId: params.parameterId,
    filePath: params.filePath,
    beforeValue: params.beforeValue,
    afterValue: params.afterValue,
    search: `  ${params.parameterId}: ${params.searchLiteral ?? scalarLiteral(params.beforeValue)},`,
    replace: `  ${params.parameterId}: ${params.replaceLiteral ?? scalarLiteral(params.afterValue)},`,
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
        `markov-distribution.ts: momentumLookback ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback} → 7`,
        `markov-distribution.ts: structuralBreakMinLength ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength} → 24`,
        `conformal.ts: scoreAggregationMinSamples ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples} → 8`,
        `conformal.ts: scoreAggregationCalibrationWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow} → 48`,
        `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → 12`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'momentumLookback',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback,
          afterValue: 7,
        }),
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'structuralBreakMinLength',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength,
          afterValue: 24,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationMinSamples',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples,
          afterValue: 8,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationCalibrationWindow',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow,
          afterValue: 48,
        }),
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'minSamplesPerRegime',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
          afterValue: 12,
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

interface WindowMutationValues {
  readonly transitionMinObservations: number;
  readonly momentumLookback: number;
  readonly momentumAdjustmentScale: number;
  readonly momentumAdjustmentClamp: number;
  readonly structuralBreakMinLength: number;
  readonly scoreAggregationMinSamples: number;
  readonly scoreAggregationCalibrationWindow: number;
  readonly minSamplesPerRegime: number;
}

interface DecayMutationValues {
  readonly transitionDecay: number;
  readonly adaptiveBreakLearningRateMultiplier: number;
  readonly adaptiveBreakCooloffWindow: number;
}

interface CalibratorMutationValues {
  readonly minSamplesPerRegime: number;
  readonly learningRate: number;
  readonly pidLearningRate: number;
  readonly integralDecay: number;
}

interface ConfidenceMutationValues {
  readonly recommendedConfidenceThreshold: number;
  readonly momentumAdjustmentScale: number;
  readonly momentumAdjustmentClamp: number;
  readonly trendPenaltyOnlyBreakConfidence?: boolean;
  readonly divergenceWeightedBreakConfidence?: boolean;
}

interface SpecializedCatalogConfig {
  readonly profileId: ForecastLabMarkovMutatorProfileId;
  readonly assetLabel: string;
  readonly idPrefix?: string;
  readonly shorterReactive: WindowMutationValues;
  readonly longerStability: WindowMutationValues;
  readonly fasterDecay: DecayMutationValues;
  readonly slowerDecay: DecayMutationValues;
  readonly lowerConfidence: ConfidenceMutationValues;
  readonly higherConfidence: ConfidenceMutationValues;
  readonly calibratorHigher: CalibratorMutationValues;
  readonly calibratorLower: CalibratorMutationValues;
}

function buildMutationId(baseId: string, prefix?: string): string {
  return prefix ? `${prefix}-${baseId}` : baseId;
}

function buildWindowCandidate(params: {
  readonly id: string;
  readonly profileId: ForecastLabMarkovMutatorProfileId;
  readonly summary: string;
  readonly values: WindowMutationValues;
}): ForecastLabMarkovParameterMutationCandidate {
  return buildCandidate({
    id: params.id,
    profileId: params.profileId,
    summary: params.summary,
    patchSummary: [
      `markov-distribution.ts: transitionMinObservations ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionMinObservations} → ${params.values.transitionMinObservations}`,
      `markov-distribution.ts: momentumLookback ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback} → ${params.values.momentumLookback}`,
      `markov-distribution.ts: momentumAdjustmentScale ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumAdjustmentScale} → ${params.values.momentumAdjustmentScale}`,
      `markov-distribution.ts: momentumAdjustmentClamp ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumAdjustmentClamp} → ${params.values.momentumAdjustmentClamp}`,
      `markov-distribution.ts: structuralBreakMinLength ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength} → ${params.values.structuralBreakMinLength}`,
      `conformal.ts: scoreAggregationMinSamples ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples} → ${params.values.scoreAggregationMinSamples}`,
      `conformal.ts: scoreAggregationCalibrationWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow} → ${params.values.scoreAggregationCalibrationWindow}`,
      `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → ${params.values.minSamplesPerRegime}`,
    ],
    edits: [
      buildEdit({
        filePath: MARKOV_FILE,
        parameterId: 'transitionMinObservations',
        beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionMinObservations,
        afterValue: params.values.transitionMinObservations,
      }),
      buildEdit({
        filePath: MARKOV_FILE,
        parameterId: 'momentumLookback',
        beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback,
        afterValue: params.values.momentumLookback,
      }),
      buildEdit({
        filePath: MARKOV_FILE,
        parameterId: 'momentumAdjustmentScale',
        beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumAdjustmentScale,
        afterValue: params.values.momentumAdjustmentScale,
      }),
      buildEdit({
        filePath: MARKOV_FILE,
        parameterId: 'momentumAdjustmentClamp',
        beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumAdjustmentClamp,
        afterValue: params.values.momentumAdjustmentClamp,
      }),
      buildEdit({
        filePath: MARKOV_FILE,
        parameterId: 'structuralBreakMinLength',
        beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength,
        afterValue: params.values.structuralBreakMinLength,
      }),
      buildEdit({
        filePath: CONFORMAL_FILE,
        parameterId: 'scoreAggregationMinSamples',
        beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples,
        afterValue: params.values.scoreAggregationMinSamples,
      }),
      buildEdit({
        filePath: CONFORMAL_FILE,
        parameterId: 'scoreAggregationCalibrationWindow',
        beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow,
        afterValue: params.values.scoreAggregationCalibrationWindow,
      }),
      buildEdit({
        filePath: REGIME_CALIBRATOR_FILE,
        parameterId: 'minSamplesPerRegime',
        beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
        afterValue: params.values.minSamplesPerRegime,
      }),
    ],
  });
}

function buildDecayCandidate(params: {
  readonly id: string;
  readonly profileId: ForecastLabMarkovMutatorProfileId;
  readonly summary: string;
  readonly values: DecayMutationValues;
}): ForecastLabMarkovParameterMutationCandidate {
  return buildCandidate({
    id: params.id,
    profileId: params.profileId,
    summary: params.summary,
    patchSummary: [
      `markov-distribution.ts: transitionDecay ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionDecay} → ${params.values.transitionDecay}`,
      `conformal.ts: adaptiveBreakLearningRateMultiplier ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakLearningRateMultiplier} → ${params.values.adaptiveBreakLearningRateMultiplier}`,
      `conformal.ts: adaptiveBreakCooloffWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakCooloffWindow} → ${params.values.adaptiveBreakCooloffWindow}`,
    ],
    edits: [
      buildEdit({
        filePath: MARKOV_FILE,
        parameterId: 'transitionDecay',
        beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.transitionDecay,
        afterValue: params.values.transitionDecay,
      }),
      buildEdit({
        filePath: CONFORMAL_FILE,
        parameterId: 'adaptiveBreakLearningRateMultiplier',
        beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakLearningRateMultiplier,
        afterValue: params.values.adaptiveBreakLearningRateMultiplier,
      }),
      buildEdit({
        filePath: CONFORMAL_FILE,
        parameterId: 'adaptiveBreakCooloffWindow',
        beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.adaptiveBreakCooloffWindow,
        afterValue: params.values.adaptiveBreakCooloffWindow,
      }),
    ],
  });
}

function buildConfidenceCandidate(params: {
  readonly id: string;
  readonly profileId: ForecastLabMarkovMutatorProfileId;
  readonly summary: string;
  readonly values: ConfidenceMutationValues;
}): ForecastLabMarkovParameterMutationCandidate {
  return buildCandidate({
    id: params.id,
    profileId: params.profileId,
    summary: params.summary,
    patchSummary: [
      `markov-distribution.ts: recommendedConfidenceThreshold ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold} → ${params.values.recommendedConfidenceThreshold}`,
      `markov-distribution.ts: momentumAdjustmentScale ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumAdjustmentScale} → ${params.values.momentumAdjustmentScale}`,
      `markov-distribution.ts: momentumAdjustmentClamp ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumAdjustmentClamp} → ${params.values.momentumAdjustmentClamp}`,
      ...(params.values.trendPenaltyOnlyBreakConfidence !== undefined
        ? [
            `markov-distribution.ts: trendPenaltyOnlyBreakConfidence ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.trendPenaltyOnlyBreakConfidence} → ${params.values.trendPenaltyOnlyBreakConfidence}`,
          ]
        : []),
      ...(params.values.divergenceWeightedBreakConfidence !== undefined
        ? [
            `markov-distribution.ts: divergenceWeightedBreakConfidence ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.divergenceWeightedBreakConfidence} → true`,
          ]
        : []),
    ],
    edits: [
      buildEdit({
        filePath: MARKOV_FILE,
        parameterId: 'recommendedConfidenceThreshold',
        beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold,
        afterValue: params.values.recommendedConfidenceThreshold,
      }),
      buildEdit({
        filePath: MARKOV_FILE,
        parameterId: 'momentumAdjustmentScale',
        beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumAdjustmentScale,
        afterValue: params.values.momentumAdjustmentScale,
      }),
      buildEdit({
        filePath: MARKOV_FILE,
        parameterId: 'momentumAdjustmentClamp',
        beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumAdjustmentClamp,
        afterValue: params.values.momentumAdjustmentClamp,
      }),
      ...(params.values.trendPenaltyOnlyBreakConfidence !== undefined
        ? [
            buildEdit({
              filePath: MARKOV_FILE,
              parameterId: 'trendPenaltyOnlyBreakConfidence',
              beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.trendPenaltyOnlyBreakConfidence,
              afterValue: params.values.trendPenaltyOnlyBreakConfidence,
            }),
          ]
        : []),
      ...(params.values.divergenceWeightedBreakConfidence !== undefined
        ? [
            buildEdit({
              filePath: MARKOV_FILE,
              parameterId: 'divergenceWeightedBreakConfidence',
              beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.divergenceWeightedBreakConfidence,
              afterValue: params.values.divergenceWeightedBreakConfidence,
            }),
          ]
        : []),
    ],
  });
}

function buildCalibratorCandidate(params: {
  readonly id: string;
  readonly profileId: ForecastLabMarkovMutatorProfileId;
  readonly summary: string;
  readonly values: CalibratorMutationValues;
}): ForecastLabMarkovParameterMutationCandidate {
  return buildCandidate({
    id: params.id,
    profileId: params.profileId,
    summary: params.summary,
    patchSummary: [
      `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → ${params.values.minSamplesPerRegime}`,
      `regime-calibrator.ts: learningRate ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.learningRate} → ${params.values.learningRate}`,
      `conformal.ts: pidLearningRate ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.pidLearningRate} → ${params.values.pidLearningRate}`,
      `conformal.ts: integralDecay ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.integralDecay} → ${params.values.integralDecay}`,
    ],
    edits: [
      buildEdit({
        filePath: REGIME_CALIBRATOR_FILE,
        parameterId: 'minSamplesPerRegime',
        beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
        afterValue: params.values.minSamplesPerRegime,
      }),
      buildEdit({
        filePath: REGIME_CALIBRATOR_FILE,
        parameterId: 'learningRate',
        beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.learningRate,
        afterValue: params.values.learningRate,
      }),
      buildEdit({
        filePath: CONFORMAL_FILE,
        parameterId: 'pidLearningRate',
        beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.pidLearningRate,
        afterValue: params.values.pidLearningRate,
      }),
      buildEdit({
        filePath: CONFORMAL_FILE,
        parameterId: 'integralDecay',
        beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.integralDecay,
        afterValue: params.values.integralDecay,
        searchLiteral: '1.0',
      }),
    ],
  });
}

function buildSpecializedCatalog(config: SpecializedCatalogConfig): readonly ForecastLabMarkovParameterMutationCandidate[] {
  return deepFreeze([
    buildWindowCandidate({
      id: buildMutationId('markov-shorter-reactive-window', config.idPrefix),
      profileId: config.profileId,
      summary: `Tighten ${config.assetLabel} 1d/2d/3d adaptation windows for faster short-horizon recalibration.`,
      values: config.shorterReactive,
    }),
    buildWindowCandidate({
      id: buildMutationId('markov-longer-stability-window', config.idPrefix),
      profileId: config.profileId,
      summary: `Lengthen ${config.assetLabel} calibration windows to favor stabler 1d/2d/3d fits without relying on 7d/14d.`,
      values: config.longerStability,
    }),
    buildDecayCandidate({
      id: buildMutationId('markov-faster-decay-reaction', config.idPrefix),
      profileId: config.profileId,
      summary: `Lower ${config.assetLabel} transition decay and tune adaptive conformal sensitivity for quicker 1d resets.`,
      values: config.fasterDecay,
    }),
    buildDecayCandidate({
      id: buildMutationId('markov-slower-decay-persistence', config.idPrefix),
      profileId: config.profileId,
      summary: `Raise ${config.assetLabel} transition decay and soften adaptive break sensitivity for steadier short-horizon persistence.`,
      values: config.slowerDecay,
    }),
    buildConfidenceCandidate({
      id: buildMutationId('markov-lower-confidence-trend-penalty', config.idPrefix),
      profileId: config.profileId,
      summary: `Lower the ${config.assetLabel} confidence gate while increasing short-term momentum tilt under conservative break penalties.`,
      values: config.lowerConfidence,
    }),
    buildConfidenceCandidate({
      id: buildMutationId('markov-higher-confidence-divergence-weighted', config.idPrefix),
      profileId: config.profileId,
      summary: `Raise the ${config.assetLabel} confidence gate, soften momentum tilt, and switch to divergence-weighted break penalties.`,
      values: config.higherConfidence,
    }),
    buildCalibratorCandidate({
      id: buildMutationId('markov-calibrator-higher-sample-floor', config.idPrefix),
      profileId: config.profileId,
      summary: `Raise ${config.assetLabel} calibrator sample floors while slowing the Platt/PID update memory.`,
      values: config.calibratorHigher,
    }),
    buildCalibratorCandidate({
      id: buildMutationId('markov-calibrator-lower-sample-floor', config.idPrefix),
      profileId: config.profileId,
      summary: `Lower ${config.assetLabel} calibrator sample floors and speed up the Platt/PID update rate for quicker adaptation.`,
      values: config.calibratorLower,
    }),
  ]);
}

function buildGoldCatalog(): readonly ForecastLabMarkovParameterMutationCandidate[] {
  return buildSpecializedCatalog({
    profileId: 'gold-markov-short-horizon',
    assetLabel: 'GOLD',
    idPrefix: 'gold',
    shorterReactive: {
      transitionMinObservations: 28,
      momentumLookback: 12,
      momentumAdjustmentScale: 0.18,
      momentumAdjustmentClamp: 0.0024,
      structuralBreakMinLength: 40,
      scoreAggregationMinSamples: 14,
      scoreAggregationCalibrationWindow: 84,
      minSamplesPerRegime: 20,
    },
    longerStability: {
      transitionMinObservations: 36,
      momentumLookback: 24,
      momentumAdjustmentScale: 0.14,
      momentumAdjustmentClamp: 0.0018,
      structuralBreakMinLength: 64,
      scoreAggregationMinSamples: 24,
      scoreAggregationCalibrationWindow: 132,
      minSamplesPerRegime: 32,
    },
    fasterDecay: {
      transitionDecay: 0.95,
      adaptiveBreakLearningRateMultiplier: 1.65,
      adaptiveBreakCooloffWindow: 1,
    },
    slowerDecay: {
      transitionDecay: 0.982,
      adaptiveBreakLearningRateMultiplier: 1.15,
      adaptiveBreakCooloffWindow: 2,
    },
    lowerConfidence: {
      recommendedConfidenceThreshold: 0.2,
      momentumAdjustmentScale: 0.2,
      momentumAdjustmentClamp: 0.0025,
    },
    higherConfidence: {
      recommendedConfidenceThreshold: 0.28,
      momentumAdjustmentScale: 0.14,
      momentumAdjustmentClamp: 0.0018,
      trendPenaltyOnlyBreakConfidence: false,
      divergenceWeightedBreakConfidence: true,
    },
    calibratorHigher: {
      minSamplesPerRegime: 34,
      learningRate: 0.04,
      pidLearningRate: 0.042,
      integralDecay: 0.98,
    },
    calibratorLower: {
      minSamplesPerRegime: 14,
      learningRate: 0.065,
      pidLearningRate: 0.06,
      integralDecay: 0.94,
    },
  });
}

function buildSolCatalog(): readonly ForecastLabMarkovParameterMutationCandidate[] {
  return buildSpecializedCatalog({
    profileId: 'sol-markov-short-horizon',
    assetLabel: 'SOL',
    shorterReactive: {
      transitionMinObservations: 31,
      momentumLookback: 9,
      momentumAdjustmentScale: 0.252,
      momentumAdjustmentClamp: 0.00305,
      structuralBreakMinLength: 28,
      scoreAggregationMinSamples: 10,
      scoreAggregationCalibrationWindow: 60,
      minSamplesPerRegime: 14,
    },
    longerStability: {
      transitionMinObservations: 36,
      momentumLookback: 28,
      momentumAdjustmentScale: 0.18,
      momentumAdjustmentClamp: 0.0024,
      structuralBreakMinLength: 72,
      scoreAggregationMinSamples: 24,
      scoreAggregationCalibrationWindow: 144,
      minSamplesPerRegime: 36,
    },
    fasterDecay: {
      transitionDecay: 0.945,
      adaptiveBreakLearningRateMultiplier: 1.85,
      adaptiveBreakCooloffWindow: 2,
    },
    slowerDecay: {
      transitionDecay: 0.99,
      adaptiveBreakLearningRateMultiplier: 1.15,
      adaptiveBreakCooloffWindow: 3,
    },
    lowerConfidence: {
      recommendedConfidenceThreshold: 0.17,
      momentumAdjustmentScale: 0.34,
      momentumAdjustmentClamp: 0.004,
    },
    higherConfidence: {
      recommendedConfidenceThreshold: 0.3,
      momentumAdjustmentScale: 0.16,
      momentumAdjustmentClamp: 0.0021,
      trendPenaltyOnlyBreakConfidence: false,
      divergenceWeightedBreakConfidence: true,
    },
    calibratorHigher: {
      minSamplesPerRegime: 40,
      learningRate: 0.03,
      pidLearningRate: 0.042,
      integralDecay: 0.98,
    },
    calibratorLower: {
      minSamplesPerRegime: 14,
      learningRate: 0.085,
      pidLearningRate: 0.06,
      integralDecay: 0.93,
    },
  });
}

function buildHypeCatalog(): readonly ForecastLabMarkovParameterMutationCandidate[] {
  return buildSpecializedCatalog({
    profileId: 'hype-markov-short-horizon',
    assetLabel: 'HYPE',
    shorterReactive: {
      transitionMinObservations: 22,
      momentumLookback: 8,
      momentumAdjustmentScale: 0.5,
      momentumAdjustmentClamp: 0.006,
      structuralBreakMinLength: 24,
      scoreAggregationMinSamples: 8,
      scoreAggregationCalibrationWindow: 48,
      minSamplesPerRegime: 12,
    },
    longerStability: {
      transitionMinObservations: 34,
      momentumLookback: 24,
      momentumAdjustmentScale: 0.22,
      momentumAdjustmentClamp: 0.0032,
      structuralBreakMinLength: 60,
      scoreAggregationMinSamples: 22,
      scoreAggregationCalibrationWindow: 120,
      minSamplesPerRegime: 28,
    },
    fasterDecay: {
      transitionDecay: 0.925,
      adaptiveBreakLearningRateMultiplier: 2.1,
      adaptiveBreakCooloffWindow: 2,
    },
    slowerDecay: {
      transitionDecay: 0.983,
      adaptiveBreakLearningRateMultiplier: 1.2,
      adaptiveBreakCooloffWindow: 3,
    },
    lowerConfidence: {
      recommendedConfidenceThreshold: 0.15,
      momentumAdjustmentScale: 0.48,
      momentumAdjustmentClamp: 0.0058,
    },
    higherConfidence: {
      recommendedConfidenceThreshold: 0.28,
      momentumAdjustmentScale: 0.18,
      momentumAdjustmentClamp: 0.0026,
      trendPenaltyOnlyBreakConfidence: false,
      divergenceWeightedBreakConfidence: true,
    },
    calibratorHigher: {
      minSamplesPerRegime: 28,
      learningRate: 0.038,
      pidLearningRate: 0.042,
      integralDecay: 0.97,
    },
    calibratorLower: {
      minSamplesPerRegime: 10,
      learningRate: 0.095,
      pidLearningRate: 0.065,
      integralDecay: 0.92,
    },
  });
}

const CATALOG_BY_PROFILE = deepFreeze({
  'multi-asset-markov-short-horizon': buildCatalog('multi-asset-markov-short-horizon'),
  'btc-markov-ultra-short-horizon': buildCatalog('btc-markov-ultra-short-horizon'),
  'sol-markov-short-horizon': buildSolCatalog(),
  'hype-markov-short-horizon': buildHypeCatalog(),
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
