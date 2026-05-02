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
  | 'btc-markov-short-horizon'
  | 'btc-markov-ultra-short-horizon';

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

function buildCatalog(profileId: ForecastLabMarkovMutatorProfileId): readonly ForecastLabMarkovParameterMutationCandidate[] {
  return deepFreeze([
    buildCandidate({
      id: 'markov-shorter-reactive-window',
      profileId,
      summary: 'Shorten Markov/conformal calibration windows for faster short-horizon adaptation.',
      patchSummary: [
        `markov-distribution.ts: momentumLookback ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback} → 14`,
        `markov-distribution.ts: structuralBreakMinLength ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength} → 48`,
        `conformal.ts: scoreAggregationMinSamples ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples} → 16`,
        `conformal.ts: scoreAggregationCalibrationWindow ${FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow} → 96`,
        `regime-calibrator.ts: minSamplesPerRegime ${FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime} → 24`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'momentumLookback',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.momentumLookback,
          afterValue: 14,
        }),
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'structuralBreakMinLength',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.structuralBreakMinLength,
          afterValue: 48,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationMinSamples',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationMinSamples,
          afterValue: 16,
        }),
        buildEdit({
          filePath: CONFORMAL_FILE,
          parameterId: 'scoreAggregationCalibrationWindow',
          beforeValue: FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS.scoreAggregationCalibrationWindow,
          afterValue: 96,
        }),
        buildEdit({
          filePath: REGIME_CALIBRATOR_FILE,
          parameterId: 'minSamplesPerRegime',
          beforeValue: FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS.minSamplesPerRegime,
          afterValue: 24,
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
      summary: 'Lower the confidence gate and enable the trend-only break penalty ablation.',
      patchSummary: [
        `markov-distribution.ts: recommendedConfidenceThreshold ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold} → 0.22`,
        `markov-distribution.ts: trendPenaltyOnlyBreakConfidence ${FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.trendPenaltyOnlyBreakConfidence} → true`,
      ],
      edits: [
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'recommendedConfidenceThreshold',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.recommendedConfidenceThreshold,
          afterValue: 0.22,
        }),
        buildEdit({
          filePath: MARKOV_FILE,
          parameterId: 'trendPenaltyOnlyBreakConfidence',
          beforeValue: FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS.trendPenaltyOnlyBreakConfidence,
          afterValue: true,
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

const CATALOG_BY_PROFILE = deepFreeze({
  'btc-markov-short-horizon': buildCatalog('btc-markov-short-horizon'),
  'btc-markov-ultra-short-horizon': buildCatalog('btc-markov-ultra-short-horizon'),
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
