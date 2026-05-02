import type { ForecastLabDecision } from './types.js';
import {
  defineForecastLabProfileMutationConfig,
  type ForecastLabMutatorId,
  type ForecastLabProfileMutationConfig,
} from './mutation.js';
import {
  isForecastLabMarkovMutatorProfileId,
  listMarkovParameterMutations,
  type ForecastLabMarkovParameterMutationCandidate,
} from './mutators/markov-parameters.js';

export type ForecastLabProfileId =
  | 'multi-asset-markov-short-horizon'
  | 'btc-markov-ultra-short-horizon'
  | 'btc-arbiter-replay'
  | 'polymarket-selection-sanity';

export type ForecastLabProfileAliasId = 'btc-markov-short-horizon';
export type ForecastLabAnyProfileId = ForecastLabProfileId | ForecastLabProfileAliasId;

export type ForecastLabTargetSubsystem =
  | 'markov-distribution'
  | 'forecast-arbiter'
  | 'polymarket-selection';

export type ForecastLabMetricDirection = 'higher-is-better' | 'lower-is-better' | 'target-is-better';

export type ForecastLabRuleOperator =
  | 'candidate-delta-gte'
  | 'candidate-delta-lte'
  | 'candidate-value-gte'
  | 'candidate-value-lte';

export interface ForecastLabCommand {
  readonly id: string;
  readonly command: string;
  readonly env?: Readonly<Record<string, string>>;
  readonly timeoutMs?: number;
}

export interface ForecastLabMetric {
  readonly name: string;
  readonly baselinePath: string;
  readonly candidatePath: string;
  readonly direction: ForecastLabMetricDirection;
  readonly required: true;
}

export interface ForecastLabRuleCriterion {
  readonly metric: string;
  readonly operator: ForecastLabRuleOperator;
  readonly value: number;
  readonly reason: string;
}

export interface ForecastLabKeepDropRule {
  readonly defaultDecision: ForecastLabDecision;
  readonly keepWhen: {
    readonly all: readonly ForecastLabRuleCriterion[];
  };
  readonly dropWhen: {
    readonly any: readonly ForecastLabRuleCriterion[];
  };
}

export interface ForecastLabProfileRoutingKeywordGroup {
  readonly label: string;
  readonly terms: readonly string[];
  readonly weight?: number;
}

export interface ForecastLabProfileRoutingMetadata {
  readonly summary: string;
  readonly keywordGroups: readonly ForecastLabProfileRoutingKeywordGroup[];
}

export interface ForecastLabProfile {
  readonly id: ForecastLabProfileId;
  readonly targetSubsystem: ForecastLabTargetSubsystem;
  readonly routing: ForecastLabProfileRoutingMetadata;
  readonly allowedGlobs: readonly string[];
  readonly mutation: ForecastLabProfileMutationConfig;
  readonly readOnlyHarnessFiles: readonly string[];
  readonly baselineCommands: readonly ForecastLabCommand[];
  readonly candidateCommands: readonly ForecastLabCommand[];
  readonly minimumMetrics: readonly ForecastLabMetric[];
  readonly keepDropRule: ForecastLabKeepDropRule;
}

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

export const FORECAST_LAB_READ_ONLY_HARNESS_FILES = deepFreeze([
  'src/tools/finance/backtest/walk-forward.ts',
  'src/tools/finance/backtest/arbiter-replay-runner.ts',
] as const);

const MARKOV_ALLOWED_MUTATOR_IDS = deepFreeze([
  'search-replace',
] as const satisfies readonly ForecastLabMutatorId[]);

const MARKOV_MUTABLE_FILES = deepFreeze([
  'src/tools/finance/markov-distribution.ts',
  'src/tools/finance/conformal.ts',
  'src/tools/finance/regime-calibrator.ts',
] as const);

const ARBITER_MUTABLE_FILES = deepFreeze([
  'src/tools/finance/forecast-arbitrator.ts',
  'src/tools/finance/forecast-hooks.ts',
] as const);

const POLYMARKET_MUTABLE_FILES = deepFreeze([
  'src/tools/finance/polymarket-forecast.ts',
  'src/tools/finance/polymarket.ts',
] as const);

const NO_STRUCTURED_MUTATIONS = deepFreeze([] as const satisfies readonly ForecastLabMarkovParameterMutationCandidate[]);

const WALK_FORWARD_SHORT_HORIZON_COMMANDS = deepFreeze([
  {
    id: 'walk-forward-short-horizon',
    command: 'bun test src/tools/finance/backtest/walk-forward-short-horizon.test.ts --timeout 480000',
    env: { RUN_INTEGRATION: '1' },
    timeoutMs: 480_000,
  },
] as const satisfies readonly ForecastLabCommand[]);

const WALK_FORWARD_BTC_ULTRA_SHORT_HORIZON_COMMANDS = deepFreeze([
  {
    id: 'walk-forward-btc-ultra-short-horizon',
    command: 'bun test src/tools/finance/backtest/walk-forward-btc-ultra-short-horizon.test.ts --timeout 360000',
    env: { RUN_INTEGRATION: '1' },
    timeoutMs: 360_000,
  },
] as const satisfies readonly ForecastLabCommand[]);

const ARBITER_REPLAY_COMMANDS = deepFreeze([
  {
    id: 'arbiter-replay-runner',
    command: 'bun test src/tools/finance/backtest/arbiter-replay-runner.test.ts',
    timeoutMs: 120_000,
  },
] as const satisfies readonly ForecastLabCommand[]);

const POLYMARKET_SELECTION_COMMANDS = deepFreeze([
  {
    id: 'polymarket-forecast-selection',
    command: 'bun test src/tools/finance/polymarket-forecast.test.ts',
    timeoutMs: 120_000,
  },
] as const satisfies readonly ForecastLabCommand[]);

const MULTI_ASSET_MARKOV_SHORT_HORIZON_ROUTING = deepFreeze({
  summary: 'Improve multi-asset short-horizon Markov mechanics.',
  keywordGroups: [
    { label: 'multi-asset context', terms: ['multi-asset', 'multi asset', 'cross-asset', 'cross asset'] },
    { label: 'short-horizon mechanics', terms: ['short-horizon', 'short horizon', 'mechanics'] },
    { label: 'markov context', terms: ['markov'] },
    {
      label: 'profile id',
      terms: ['multi-asset-markov-short-horizon', 'btc-markov-short-horizon'],
      weight: 3,
    },
  ],
} as const satisfies ForecastLabProfileRoutingMetadata);

const BTC_MARKOV_ULTRA_SHORT_HORIZON_ROUTING = deepFreeze({
  summary: 'Improve BTC ultra-short-horizon Markov behavior for 1d/2d/3d forecasts.',
  keywordGroups: [
    { label: 'btc asset', terms: ['btc', 'bitcoin'] },
    {
      label: 'ultra-short horizons',
      terms: ['1d/2d/3d', '1d', '2d', '3d', 'ultra-short-horizon', 'ultra short horizon', 'ultra-short'],
    },
    { label: 'markov context', terms: ['markov', 'short-horizon', 'short horizon'] },
    {
      label: 'profile id',
      terms: ['btc-markov-ultra-short-horizon'],
      weight: 3,
    },
  ],
} as const satisfies ForecastLabProfileRoutingMetadata);

const BTC_ARBITER_REPLAY_ROUTING = deepFreeze({
  summary: 'Improve BTC forecast arbiter replay behavior.',
  keywordGroups: [
    { label: 'arbiter context', terms: ['arbiter', 'arbitrator'] },
    { label: 'replay context', terms: ['replay', 'replays'] },
    {
      label: 'profile id',
      terms: ['btc-arbiter-replay'],
      weight: 3,
    },
  ],
} as const satisfies ForecastLabProfileRoutingMetadata);

const POLYMARKET_SELECTION_ROUTING = deepFreeze({
  summary: 'Improve Polymarket forecast selection sanity behavior.',
  keywordGroups: [
    { label: 'polymarket context', terms: ['polymarket'] },
    { label: 'selection sanity', terms: ['selection', 'sanity'] },
    {
      label: 'profile id',
      terms: ['polymarket-selection-sanity'],
      weight: 3,
    },
  ],
} as const satisfies ForecastLabProfileRoutingMetadata);

const PROFILES_BY_ID = deepFreeze({
  'multi-asset-markov-short-horizon': {
    id: 'multi-asset-markov-short-horizon',
    targetSubsystem: 'markov-distribution',
    routing: MULTI_ASSET_MARKOV_SHORT_HORIZON_ROUTING,
    allowedGlobs: MARKOV_MUTABLE_FILES,
    mutation: defineForecastLabProfileMutationConfig({
      mode: 'structured',
      mutableFiles: MARKOV_MUTABLE_FILES,
      allowedMutatorIds: MARKOV_ALLOWED_MUTATOR_IDS,
    }),
    readOnlyHarnessFiles: ['src/tools/finance/backtest/walk-forward.ts'],
    baselineCommands: WALK_FORWARD_SHORT_HORIZON_COMMANDS,
    candidateCommands: WALK_FORWARD_SHORT_HORIZON_COMMANDS,
    minimumMetrics: [
      // The referenced Bun test is a status gate; it prints tables but does not export JSON metrics.
      {
        name: 'walkForwardShortHorizonTestExitCode',
        baselinePath: 'baseline.exitCode',
        candidatePath: 'candidate.exitCode',
        direction: 'lower-is-better',
        required: true,
      },
    ],
    keepDropRule: {
      defaultDecision: 'drop',
      keepWhen: {
        all: [
          {
            metric: 'walkForwardShortHorizonTestExitCode',
            operator: 'candidate-value-lte',
            value: 0,
            reason: 'candidate walk-forward short-horizon test command must pass',
          },
          {
            metric: 'walkForwardShortHorizonTestExitCode',
            operator: 'candidate-delta-lte',
            value: 0,
            reason: 'candidate walk-forward short-horizon test command must not regress versus baseline',
          },
        ],
      },
      dropWhen: {
        any: [
          {
            metric: 'walkForwardShortHorizonTestExitCode',
            operator: 'candidate-value-gte',
            value: 1,
            reason: 'drop when the walk-forward short-horizon test command fails',
          },
        ],
      },
    },
  },
  'btc-markov-ultra-short-horizon': {
    id: 'btc-markov-ultra-short-horizon',
    targetSubsystem: 'markov-distribution',
    routing: BTC_MARKOV_ULTRA_SHORT_HORIZON_ROUTING,
    allowedGlobs: MARKOV_MUTABLE_FILES,
    mutation: defineForecastLabProfileMutationConfig({
      mode: 'structured',
      mutableFiles: MARKOV_MUTABLE_FILES,
      allowedMutatorIds: MARKOV_ALLOWED_MUTATOR_IDS,
    }),
    readOnlyHarnessFiles: ['src/tools/finance/backtest/walk-forward.ts'],
    baselineCommands: WALK_FORWARD_BTC_ULTRA_SHORT_HORIZON_COMMANDS,
    candidateCommands: WALK_FORWARD_BTC_ULTRA_SHORT_HORIZON_COMMANDS,
    minimumMetrics: [
      {
        name: 'walkForwardBtcUltraShortHorizonTestExitCode',
        baselinePath: 'baseline.exitCode',
        candidatePath: 'candidate.exitCode',
        direction: 'lower-is-better',
        required: true,
      },
    ],
    keepDropRule: {
      defaultDecision: 'drop',
      keepWhen: {
        all: [
          {
            metric: 'walkForwardBtcUltraShortHorizonTestExitCode',
            operator: 'candidate-value-lte',
            value: 0,
            reason: 'candidate BTC ultra-short-horizon test command must pass',
          },
          {
            metric: 'walkForwardBtcUltraShortHorizonTestExitCode',
            operator: 'candidate-delta-lte',
            value: 0,
            reason: 'candidate BTC ultra-short-horizon test command must not regress versus baseline',
          },
        ],
      },
      dropWhen: {
        any: [
          {
            metric: 'walkForwardBtcUltraShortHorizonTestExitCode',
            operator: 'candidate-value-gte',
            value: 1,
            reason: 'drop when the BTC ultra-short-horizon test command fails',
          },
        ],
      },
    },
  },
  'btc-arbiter-replay': {
    id: 'btc-arbiter-replay',
    targetSubsystem: 'forecast-arbiter',
    routing: BTC_ARBITER_REPLAY_ROUTING,
    allowedGlobs: ARBITER_MUTABLE_FILES,
    mutation: defineForecastLabProfileMutationConfig({
      mode: 'dry-run',
      mutableFiles: ARBITER_MUTABLE_FILES,
    }),
    readOnlyHarnessFiles: ['src/tools/finance/backtest/arbiter-replay-runner.ts'],
    baselineCommands: ARBITER_REPLAY_COMMANDS,
    candidateCommands: ARBITER_REPLAY_COMMANDS,
    minimumMetrics: [
      // The referenced Bun test exercises the replay harness but does not export JSON metrics.
      {
        name: 'arbiterReplayRunnerTestExitCode',
        baselinePath: 'baseline.exitCode',
        candidatePath: 'candidate.exitCode',
        direction: 'lower-is-better',
        required: true,
      },
    ],
    keepDropRule: {
      defaultDecision: 'drop',
      keepWhen: {
        all: [
          {
            metric: 'arbiterReplayRunnerTestExitCode',
            operator: 'candidate-value-lte',
            value: 0,
            reason: 'candidate arbiter replay test command must pass',
          },
          {
            metric: 'arbiterReplayRunnerTestExitCode',
            operator: 'candidate-delta-lte',
            value: 0,
            reason: 'candidate arbiter replay test command must not regress versus baseline',
          },
        ],
      },
      dropWhen: {
        any: [
          {
            metric: 'arbiterReplayRunnerTestExitCode',
            operator: 'candidate-value-gte',
            value: 1,
            reason: 'drop when the arbiter replay test command fails',
          },
        ],
      },
    },
  },
  'polymarket-selection-sanity': {
    id: 'polymarket-selection-sanity',
    targetSubsystem: 'polymarket-selection',
    routing: POLYMARKET_SELECTION_ROUTING,
    allowedGlobs: POLYMARKET_MUTABLE_FILES,
    mutation: defineForecastLabProfileMutationConfig({
      mode: 'dry-run',
      mutableFiles: POLYMARKET_MUTABLE_FILES,
    }),
    readOnlyHarnessFiles: [],
    baselineCommands: POLYMARKET_SELECTION_COMMANDS,
    candidateCommands: POLYMARKET_SELECTION_COMMANDS,
    minimumMetrics: [
      // The referenced Bun test is a sanity gate; it does not export selection counters.
      {
        name: 'polymarketForecastTestExitCode',
        baselinePath: 'baseline.exitCode',
        candidatePath: 'candidate.exitCode',
        direction: 'lower-is-better',
        required: true,
      },
    ],
    keepDropRule: {
      defaultDecision: 'drop',
      keepWhen: {
        all: [
          {
            metric: 'polymarketForecastTestExitCode',
            operator: 'candidate-value-lte',
            value: 0,
            reason: 'candidate Polymarket forecast sanity tests must pass',
          },
          {
            metric: 'polymarketForecastTestExitCode',
            operator: 'candidate-delta-lte',
            value: 0,
            reason: 'candidate Polymarket forecast sanity tests must not regress versus baseline',
          },
        ],
      },
      dropWhen: {
        any: [
          {
            metric: 'polymarketForecastTestExitCode',
            operator: 'candidate-value-gte',
            value: 1,
            reason: 'drop when the Polymarket forecast sanity test command fails',
          },
        ],
      },
    },
  },
} as const satisfies Record<ForecastLabProfileId, ForecastLabProfile>);

const PROFILE_ALIASES = deepFreeze({
  'btc-markov-short-horizon': 'multi-asset-markov-short-horizon',
} as const satisfies Record<ForecastLabProfileAliasId, ForecastLabProfileId>);

export const FORECAST_LAB_PROFILES = deepFreeze([
  PROFILES_BY_ID['multi-asset-markov-short-horizon'],
  PROFILES_BY_ID['btc-markov-ultra-short-horizon'],
  PROFILES_BY_ID['btc-arbiter-replay'],
  PROFILES_BY_ID['polymarket-selection-sanity'],
] as const satisfies readonly ForecastLabProfile[]);

export class ForecastLabProfileError extends Error {
  override name = 'ForecastLabProfileError';
}

export function listForecastLabProfiles(): readonly ForecastLabProfile[] {
  return FORECAST_LAB_PROFILES;
}

export function normalizeForecastLabProfileId(profileId: string): ForecastLabProfileId | undefined {
  if (Object.hasOwn(PROFILES_BY_ID, profileId)) {
    return profileId as ForecastLabProfileId;
  }

  if (Object.hasOwn(PROFILE_ALIASES, profileId)) {
    return PROFILE_ALIASES[profileId as ForecastLabProfileAliasId];
  }

  return undefined;
}

export function isForecastLabProfileId(profileId: string): profileId is ForecastLabAnyProfileId {
  return normalizeForecastLabProfileId(profileId) !== undefined;
}

export function isCanonicalForecastLabProfileId(profileId: string): profileId is ForecastLabProfileId {
  return Object.hasOwn(PROFILES_BY_ID, profileId);
}

export function assertForecastLabProfileId(profileId: string): asserts profileId is ForecastLabAnyProfileId {
  if (!isForecastLabProfileId(profileId)) {
    throw new ForecastLabProfileError(`Unknown forecast-lab profile id: ${profileId}`);
  }
}

export function getForecastLabProfile(profileId: string): ForecastLabProfile {
  const normalized = normalizeForecastLabProfileId(profileId);
  if (!normalized) {
    throw new ForecastLabProfileError(`Unknown forecast-lab profile id: ${profileId}`);
  }
  return PROFILES_BY_ID[normalized];
}

export function getForecastLabProfileRoutingMetadata(
  profileId: string,
): ForecastLabProfileRoutingMetadata {
  return getForecastLabProfile(profileId).routing;
}

export function listForecastLabStructuredMutations(
  profileId: ForecastLabProfileId,
): readonly ForecastLabMarkovParameterMutationCandidate[] {
  if (!isForecastLabMarkovMutatorProfileId(profileId)) {
    return NO_STRUCTURED_MUTATIONS;
  }

  return listMarkovParameterMutations(profileId);
}
