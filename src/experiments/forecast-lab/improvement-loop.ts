import { fetchBitmexDailyCloses } from '../../tools/finance/bitmex.js';
import { walkForward } from '../../tools/finance/backtest/walk-forward.js';
import { brierScore, ciCoverage, directionalAccuracy, type BacktestStep } from '../../tools/finance/backtest/metrics.js';
import {
  getForecastLabProfile,
  listForecastLabStructuredMutations,
  type ForecastLabKeepDropRule,
  type ForecastLabProfile,
} from './profiles.js';
import type {
  ForecastLabMarkovParameterMutationCandidate,
  ForecastLabMarkovParameterMutationEdit,
  ForecastLabMutationScalarValue,
} from './mutators/markov-parameters.js';
import {
  getForecastLabMarkovRuntimeDefaults,
  setForecastLabMarkovRuntimeDefaults,
} from '../../tools/finance/markov-distribution.js';
import {
  getForecastLabConformalRuntimeDefaults,
  setForecastLabConformalRuntimeDefaults,
} from '../../tools/finance/conformal.js';
import {
  getForecastLabRegimeCalibratorRuntimeDefaults,
  setForecastLabRegimeCalibratorRuntimeDefaults,
} from '../../tools/finance/regime-calibrator.js';
import type { ForecastLabRuntimeAssetScope } from './runner.js';

export interface ForecastLabImprovementHorizonMetrics {
  readonly directionalAccuracy: number;
  readonly brierScore: number;
  readonly ciCoverage: number;
  readonly structuralBreakCount: number;
  readonly abstainCount: number;
}

export interface ForecastLabImprovementMetrics {
  readonly h1: ForecastLabImprovementHorizonMetrics;
  readonly h2: ForecastLabImprovementHorizonMetrics;
  readonly h3: ForecastLabImprovementHorizonMetrics;
  readonly h7: ForecastLabImprovementHorizonMetrics;
  readonly h14: ForecastLabImprovementHorizonMetrics;
}

export interface ForecastLabImprovementEvaluation<TMetrics = ForecastLabImprovementMetrics> {
  readonly objectiveScore: number;
  readonly primaryScore: number;
  readonly keepSatisfied: number;
  readonly keepTotal: number;
  readonly decision: 'keep' | 'drop';
  readonly metrics: TMetrics;
  readonly summary: string;
}

export interface ForecastLabImprovementScoredMutation<TMetrics = ForecastLabImprovementMetrics>
  extends ForecastLabImprovementEvaluation<TMetrics> {
  readonly iteration: number;
  readonly mutation: ForecastLabMarkovParameterMutationCandidate;
}

export interface ForecastLabImprovementSearchResult<TMetrics = ForecastLabImprovementMetrics> {
  readonly seedResults: readonly ForecastLabImprovementScoredMutation<TMetrics>[];
  readonly bestResult: ForecastLabImprovementScoredMutation<TMetrics>;
  readonly history: readonly ForecastLabImprovementScoredMutation<TMetrics>[];
  readonly iterationsRun: number;
}

export interface ForecastLabProfileImprovementLoopResult {
  readonly profileId: string;
  readonly baselineMetrics: ForecastLabImprovementMetrics;
  readonly seedResults: readonly ForecastLabImprovementScoredMutation[];
  readonly bestResult: ForecastLabImprovementScoredMutation;
  readonly history: readonly ForecastLabImprovementScoredMutation[];
  readonly iterationsRun: number;
}

interface ForecastLabImprovementProfileConfig {
  readonly ticker: string;
  readonly assetScope: ForecastLabRuntimeAssetScope;
  readonly historyDays: number;
  readonly warmup: number;
  readonly stride: number;
  readonly horizons: readonly [1, 2, 3, 7, 14];
  readonly primaryWeights: Readonly<Record<'h1' | 'h2' | 'h3', number>>;
}

const SHORT_HORIZON_PROFILE_CONFIG: Record<string, ForecastLabImprovementProfileConfig> = {
  'sol-markov-short-horizon': {
    ticker: 'SOLUSD',
    assetScope: 'sol',
    historyDays: 365,
    warmup: 120,
    stride: 5,
    horizons: [1, 2, 3, 7, 14],
    primaryWeights: { h1: 0.5, h2: 0.3, h3: 0.2 },
  },
  'hype-markov-short-horizon': {
    ticker: 'HYPEUSD',
    assetScope: 'hype',
    historyDays: 365,
    warmup: 120,
    stride: 5,
    horizons: [1, 2, 3, 7, 14],
    primaryWeights: { h1: 0.55, h2: 0.3, h3: 0.15 },
  },
};

const PARAMETER_PRIORITY = [
  'transitionMinObservations',
  'momentumLookback',
  'structuralBreakMinLength',
  'momentumAdjustmentScale',
  'momentumAdjustmentClamp',
  'recommendedConfidenceThreshold',
  'transitionDecay',
  'scoreAggregationMinSamples',
  'scoreAggregationCalibrationWindow',
  'minSamplesPerRegime',
  'learningRate',
  'pidLearningRate',
  'integralDecay',
] as const;

type ImprovementBuildContext = {
  readonly iteration: number;
};

type ImprovementSearchContext = {
  readonly iteration: number;
  readonly stage: 'seed' | 'trial';
};

function countStructuralBreaks(steps: readonly BacktestStep[]): number {
  return steps.filter((step) => (step.originalStructuralBreakDetected ?? step.structuralBreakDetected) === true).length;
}

function countAbstains(steps: readonly BacktestStep[]): number {
  return steps.filter((step) => step.recommendation === 'HOLD').length;
}

function scalarLiteral(value: ForecastLabMutationScalarValue): string {
  return typeof value === 'boolean' ? String(value) : `${value}`;
}

function cloneEdit(
  edit: ForecastLabMarkovParameterMutationEdit,
  afterValue: ForecastLabMutationScalarValue,
): ForecastLabMarkovParameterMutationEdit {
  return {
    ...edit,
    afterValue,
    replace: `  ${edit.parameterId}: ${scalarLiteral(afterValue)},`,
  };
}

function summarizePatch(edits: readonly ForecastLabMarkovParameterMutationEdit[]): string[] {
  return edits.map((edit) => `${edit.filePath}: ${edit.parameterId} ${edit.beforeValue} -> ${edit.afterValue}`);
}

function cloneMutationWithEdits(
  seed: ForecastLabMarkovParameterMutationCandidate,
  id: string,
  summary: string,
  edits: readonly ForecastLabMarkovParameterMutationEdit[],
): ForecastLabMarkovParameterMutationCandidate {
  return {
    ...seed,
    id,
    specSummary: {
      ...seed.specSummary,
      summary,
    },
    patchSummary: summarizePatch(edits),
    edits,
  };
}

function normalizeTrialNumber(
  edit: ForecastLabMarkovParameterMutationEdit,
  value: number,
): number | undefined {
  const baseline = edit.beforeValue;
  const current = edit.afterValue;
  const baselineAndCurrentAreIntegers =
    typeof baseline === 'number'
    && typeof current === 'number'
    && Number.isInteger(baseline)
    && Number.isInteger(current);

  let normalized = baselineAndCurrentAreIntegers ? Math.round(value) : Number(value.toFixed(6));
  if (!Number.isFinite(normalized)) {
    return undefined;
  }
  if (normalized <= 0 && edit.parameterId !== 'recommendedConfidenceThreshold') {
    normalized = baselineAndCurrentAreIntegers ? 1 : 0.000001;
  }
  if (
    typeof baseline === 'number'
    && typeof current === 'number'
    && baseline >= 0
    && baseline <= 1
    && current >= 0
    && current <= 1
  ) {
    normalized = Math.max(0.000001, Math.min(0.999999, normalized));
    normalized = baselineAndCurrentAreIntegers ? Math.round(normalized) : Number(normalized.toFixed(6));
  }
  if (Object.is(normalized, current) || Object.is(normalized, baseline)) {
    return undefined;
  }
  return normalized;
}

function prioritizeEditableParameters(seed: ForecastLabMarkovParameterMutationCandidate): readonly ForecastLabMarkovParameterMutationEdit[] {
  const priority = new Map<string, number>(PARAMETER_PRIORITY.map((name, index) => [name, index]));
  return [...seed.edits]
    .filter((edit) => typeof edit.afterValue === 'number' && typeof edit.beforeValue === 'number')
    .sort((left, right) => {
      const leftPriority = priority.get(left.parameterId) ?? Number.MAX_SAFE_INTEGER;
      const rightPriority = priority.get(right.parameterId) ?? Number.MAX_SAFE_INTEGER;
      if (leftPriority !== rightPriority) {
        return leftPriority - rightPriority;
      }
      return Math.abs(Number(right.afterValue) - Number(right.beforeValue))
        - Math.abs(Number(left.afterValue) - Number(left.beforeValue));
    });
}

export function buildForecastLabImprovementTrials(
  seed: ForecastLabMarkovParameterMutationCandidate,
  options: {
    readonly iteration?: number;
    readonly maxParameterTrials?: number;
  } = {},
): readonly ForecastLabMarkovParameterMutationCandidate[] {
  const iteration = Math.max(1, options.iteration ?? 1);
  const maxParameterTrials = Math.max(1, options.maxParameterTrials ?? 3);
  const stepFactor = 1 / (iteration + 1);
  const prioritized = prioritizeEditableParameters(seed).slice(0, maxParameterTrials);
  const variants: ForecastLabMarkovParameterMutationCandidate[] = [];

  const buildAggregateVariant = (label: 'softer' | 'stronger', direction: -1 | 1) => {
    const edits = seed.edits.map((edit) => {
      if (typeof edit.beforeValue !== 'number' || typeof edit.afterValue !== 'number') {
        return edit;
      }
      const diff = edit.afterValue - edit.beforeValue;
      const candidate = normalizeTrialNumber(edit, edit.afterValue + diff * stepFactor * direction);
      return candidate === undefined ? edit : cloneEdit(edit, candidate);
    });
    if (edits.every((edit, index) => Object.is(edit.afterValue, seed.edits[index]!.afterValue))) {
      return;
    }
    variants.push(cloneMutationWithEdits(
      seed,
      `${seed.id}--iter${iteration}-${label}`,
      `${seed.specSummary.summary} [iter ${iteration} ${label}]`,
      edits,
    ));
  };

  buildAggregateVariant('softer', -1);
  buildAggregateVariant('stronger', 1);

  for (const edit of prioritized) {
    const diff = Number(edit.afterValue) - Number(edit.beforeValue);
    const softerValue = normalizeTrialNumber(edit, Number(edit.afterValue) - diff * stepFactor);
    if (softerValue !== undefined) {
      variants.push(cloneMutationWithEdits(
        seed,
        `${seed.id}--iter${iteration}-${edit.parameterId}-softer`,
        `${seed.specSummary.summary} [iter ${iteration} ${edit.parameterId} softer]`,
        seed.edits.map((candidateEdit) => candidateEdit.parameterId === edit.parameterId ? cloneEdit(candidateEdit, softerValue) : candidateEdit),
      ));
    }

    const strongerValue = normalizeTrialNumber(edit, Number(edit.afterValue) + diff * stepFactor);
    if (strongerValue !== undefined) {
      variants.push(cloneMutationWithEdits(
        seed,
        `${seed.id}--iter${iteration}-${edit.parameterId}-stronger`,
        `${seed.specSummary.summary} [iter ${iteration} ${edit.parameterId} stronger]`,
        seed.edits.map((candidateEdit) => candidateEdit.parameterId === edit.parameterId ? cloneEdit(candidateEdit, strongerValue) : candidateEdit),
      ));
    }
  }

  return variants;
}

export async function runForecastLabImprovementSearch<TMetrics>(params: {
  readonly seedCandidates: readonly ForecastLabMarkovParameterMutationCandidate[];
  readonly maxIterations?: number;
  readonly minObjectiveImprovement?: number;
  readonly buildTrials?: (
    seed: ForecastLabMarkovParameterMutationCandidate,
    context: ImprovementBuildContext,
  ) => readonly ForecastLabMarkovParameterMutationCandidate[];
  readonly evaluate: (
    mutation: ForecastLabMarkovParameterMutationCandidate,
    context: ImprovementSearchContext,
  ) => Promise<ForecastLabImprovementEvaluation<TMetrics>>;
}): Promise<ForecastLabImprovementSearchResult<TMetrics>> {
  if (params.seedCandidates.length === 0) {
    throw new Error('Forecast-lab improvement search requires at least one seed candidate.');
  }

  const maxIterations = Math.max(1, params.maxIterations ?? 2);
  const minObjectiveImprovement = params.minObjectiveImprovement ?? 0.000001;
  const buildTrials = params.buildTrials ?? ((seed, context) => buildForecastLabImprovementTrials(seed, context));
  const seedResults = await Promise.all(
    params.seedCandidates.map(async (mutation) => {
      const evaluation = await params.evaluate(mutation, { iteration: 0, stage: 'seed' });
      return {
        iteration: 0,
        mutation,
        ...evaluation,
      } satisfies ForecastLabImprovementScoredMutation<TMetrics>;
    }),
  );
  seedResults.sort((left, right) => right.objectiveScore - left.objectiveScore || right.primaryScore - left.primaryScore);

  let bestResult = seedResults[0]!;
  const history: ForecastLabImprovementScoredMutation<TMetrics>[] = [];
  const seenIds = new Set(seedResults.map((result) => result.mutation.id));

  for (let iteration = 1; iteration <= maxIterations; iteration += 1) {
    const trials = buildTrials(bestResult.mutation, { iteration }).filter((trial) => !seenIds.has(trial.id));
    if (trials.length === 0) {
      return { seedResults, bestResult, history, iterationsRun: history.length };
    }

    const scoredTrials = await Promise.all(trials.map(async (mutation) => {
      seenIds.add(mutation.id);
      const evaluation = await params.evaluate(mutation, { iteration, stage: 'trial' });
      return {
        iteration,
        mutation,
        ...evaluation,
      } satisfies ForecastLabImprovementScoredMutation<TMetrics>;
    }));
    scoredTrials.sort((left, right) => right.objectiveScore - left.objectiveScore || right.primaryScore - left.primaryScore);
    const bestTrial = scoredTrials[0];

    if (!bestTrial || bestTrial.objectiveScore <= bestResult.objectiveScore + minObjectiveImprovement) {
      return { seedResults, bestResult, history, iterationsRun: history.length };
    }

    bestResult = bestTrial;
    history.push(bestTrial);
  }

  return { seedResults, bestResult, history, iterationsRun: history.length };
}

function getShortHorizonProfileConfig(profileId: string): ForecastLabImprovementProfileConfig {
  const config = SHORT_HORIZON_PROFILE_CONFIG[profileId];
  if (!config) {
    throw new Error(`Forecast-lab iterative improvement loop is not configured for profile "${profileId}".`);
  }
  return config;
}

function snapshotScopeOverrides(assetScope: ForecastLabRuntimeAssetScope) {
  return {
    markov: getForecastLabMarkovRuntimeDefaults(assetScope),
    conformal: getForecastLabConformalRuntimeDefaults(assetScope),
    regime: getForecastLabRegimeCalibratorRuntimeDefaults(assetScope),
  };
}

async function withMutationOverrides<T>(
  assetScope: ForecastLabRuntimeAssetScope,
  mutation: ForecastLabMarkovParameterMutationCandidate | undefined,
  run: () => Promise<T>,
): Promise<T> {
  const previous = snapshotScopeOverrides(assetScope);
  const nextMarkov = mutation?.edits
    .filter((edit) => edit.filePath === 'src/tools/finance/markov-distribution/core.ts')
    .reduce<Record<string, ForecastLabMutationScalarValue>>((acc, edit) => {
      acc[edit.parameterId] = edit.afterValue;
      return acc;
    }, {});
  const nextConformal = mutation?.edits
    .filter((edit) => edit.filePath === 'src/tools/finance/conformal.ts')
    .reduce<Record<string, ForecastLabMutationScalarValue>>((acc, edit) => {
      acc[edit.parameterId] = edit.afterValue;
      return acc;
    }, {});
  const nextRegime = mutation?.edits
    .filter((edit) => edit.filePath === 'src/tools/finance/regime-calibrator.ts')
    .reduce<Record<string, ForecastLabMutationScalarValue>>((acc, edit) => {
      acc[edit.parameterId] = edit.afterValue;
      return acc;
    }, {});

  setForecastLabMarkovRuntimeDefaults(assetScope, nextMarkov && Object.keys(nextMarkov).length > 0 ? nextMarkov : undefined);
  setForecastLabConformalRuntimeDefaults(assetScope, nextConformal && Object.keys(nextConformal).length > 0 ? nextConformal : undefined);
  setForecastLabRegimeCalibratorRuntimeDefaults(assetScope, nextRegime && Object.keys(nextRegime).length > 0 ? nextRegime : undefined);

  try {
    return await run();
  } finally {
    setForecastLabMarkovRuntimeDefaults(assetScope, previous.markov);
    setForecastLabConformalRuntimeDefaults(assetScope, previous.conformal);
    setForecastLabRegimeCalibratorRuntimeDefaults(assetScope, previous.regime);
  }
}

async function benchmarkProfileMetrics(
  profileId: string,
  prices: readonly number[],
  mutation: ForecastLabMarkovParameterMutationCandidate | undefined,
): Promise<ForecastLabImprovementMetrics> {
  const config = getShortHorizonProfileConfig(profileId);

  return withMutationOverrides(config.assetScope, mutation, async () => {
    const results = await Promise.all(config.horizons.map(async (horizon) => {
      const walkForwardResult = await walkForward({
        ticker: config.ticker,
        prices: [...prices],
        horizon,
        warmup: config.warmup,
        stride: config.stride,
      });
      return [
        `h${horizon}`,
        {
          directionalAccuracy: walkForwardResult.steps.length > 0 ? directionalAccuracy(walkForwardResult.steps) : 0,
          brierScore: walkForwardResult.steps.length > 0 ? brierScore(walkForwardResult.steps) : 0,
          ciCoverage: walkForwardResult.steps.length > 0 ? ciCoverage(walkForwardResult.steps) : 0,
          structuralBreakCount: countStructuralBreaks(walkForwardResult.steps),
          abstainCount: countAbstains(walkForwardResult.steps),
        } satisfies ForecastLabImprovementHorizonMetrics,
      ] as const;
    }));

    return Object.fromEntries(results) as unknown as ForecastLabImprovementMetrics;
  });
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
  criterion: ForecastLabKeepDropRule['keepWhen']['all'][number],
  metrics: readonly { readonly name: string; readonly candidate: number; readonly delta: number }[],
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

function scoreCandidateAgainstProfile(
  profile: ForecastLabProfile,
  baselineMetrics: ForecastLabImprovementMetrics,
  candidateMetrics: ForecastLabImprovementMetrics,
): Omit<ForecastLabImprovementEvaluation, 'metrics'> {
  const metricRoot = {
    baseline: {
      exitCode: 0,
      metrics: baselineMetrics,
    },
    candidate: {
      exitCode: 0,
      metrics: candidateMetrics,
    },
  };

  const metricEvaluations = profile.minimumMetrics.map((metric) => {
    const baseline = getPathNumber(metricRoot, metric.baselinePath) ?? 0;
    const candidate = getPathNumber(metricRoot, metric.candidatePath) ?? 0;
    return {
      name: metric.name,
      candidate,
      delta: candidate - baseline,
    };
  });

  const keepSatisfied = profile.keepDropRule.keepWhen.all.filter((criterion) => evaluateCriterion(criterion, metricEvaluations)).length;
  const keepTotal = profile.keepDropRule.keepWhen.all.length;
  const dropTriggered = profile.keepDropRule.dropWhen.any.some((criterion) => evaluateCriterion(criterion, metricEvaluations));
  const decision = !dropTriggered && keepSatisfied === keepTotal ? 'keep' : 'drop';
  const config = getShortHorizonProfileConfig(profile.id);
  const primaryScore =
    candidateMetrics.h1.directionalAccuracy * config.primaryWeights.h1
    + candidateMetrics.h2.directionalAccuracy * config.primaryWeights.h2
    + candidateMetrics.h3.directionalAccuracy * config.primaryWeights.h3;
  const objectiveScore = (dropTriggered ? -100 : 0) + keepSatisfied * 10 + primaryScore;

  return {
    objectiveScore,
    primaryScore,
    keepSatisfied,
    keepTotal,
    decision,
    summary: decision === 'keep'
      ? `keep-ready (${keepSatisfied}/${keepTotal} gate checks satisfied)`
      : `${keepSatisfied}/${keepTotal} keep checks satisfied`,
  };
}

export async function runForecastLabProfileImprovementLoop(options: {
  readonly profileId: string;
  readonly seedMutatorId?: string;
  readonly maxIterations?: number;
  readonly progress?: (message: string) => void;
}): Promise<ForecastLabProfileImprovementLoopResult> {
  const profile = getForecastLabProfile(options.profileId);
  const config = getShortHorizonProfileConfig(profile.id);
  const catalog = listForecastLabStructuredMutations(profile.id);
  const seedCandidates = options.seedMutatorId
    ? catalog.filter((candidate) => candidate.id === options.seedMutatorId)
    : catalog;
  if (seedCandidates.length === 0) {
    throw new Error(`No shipped structured mutators found for profile "${profile.id}" and requested seed "${options.seedMutatorId ?? 'auto'}".`);
  }

  options.progress?.(`forecast-lab loop: benchmarking shipped baseline for ${profile.id}`);
  const prices = await fetchBitmexDailyCloses(config.ticker, config.historyDays);
  const baselineMetrics = await benchmarkProfileMetrics(profile.id, prices, undefined);

  const result = await runForecastLabImprovementSearch({
    seedCandidates,
    maxIterations: options.maxIterations ?? 2,
    evaluate: async (mutation, context) => {
      options.progress?.(`forecast-lab loop: ${context.stage} ${mutation.id}`);
      const metrics = await benchmarkProfileMetrics(profile.id, prices, mutation);
      return {
        ...scoreCandidateAgainstProfile(profile, baselineMetrics, metrics),
        metrics,
      };
    },
  });

  return {
    profileId: profile.id,
    baselineMetrics,
    seedResults: result.seedResults,
    bestResult: result.bestResult,
    history: result.history,
    iterationsRun: result.iterationsRun,
  };
}
