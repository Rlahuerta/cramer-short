import type { ForecastLabDecision } from './types.js';
import {
  defineForecastLabProfileMutationConfig,
  type ForecastLabMutatorId,
  type ForecastLabProfileMutationConfig,
} from './mutation.js';

export type ForecastLabProfileId =
  | 'btc-markov-short-horizon'
  | 'btc-markov-ultra-short-horizon'
  | 'btc-arbiter-replay'
  | 'polymarket-selection-sanity';

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

export interface ForecastLabProfile {
  readonly id: ForecastLabProfileId;
  readonly targetSubsystem: ForecastLabTargetSubsystem;
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

const DEFAULT_STRUCTURED_MUTATOR_IDS = deepFreeze([
  'replace-range',
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

const PROFILES_BY_ID = deepFreeze({
  'btc-markov-short-horizon': {
    id: 'btc-markov-short-horizon',
    targetSubsystem: 'markov-distribution',
    allowedGlobs: MARKOV_MUTABLE_FILES,
    mutation: defineForecastLabProfileMutationConfig({
      mode: 'structured',
      mutableFiles: MARKOV_MUTABLE_FILES,
      allowedMutatorIds: DEFAULT_STRUCTURED_MUTATOR_IDS,
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
    allowedGlobs: MARKOV_MUTABLE_FILES,
    mutation: defineForecastLabProfileMutationConfig({
      mode: 'structured',
      mutableFiles: MARKOV_MUTABLE_FILES,
      allowedMutatorIds: DEFAULT_STRUCTURED_MUTATOR_IDS,
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
    allowedGlobs: ARBITER_MUTABLE_FILES,
    mutation: defineForecastLabProfileMutationConfig({
      mode: 'structured',
      mutableFiles: ARBITER_MUTABLE_FILES,
      allowedMutatorIds: DEFAULT_STRUCTURED_MUTATOR_IDS,
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
    allowedGlobs: POLYMARKET_MUTABLE_FILES,
    mutation: defineForecastLabProfileMutationConfig({
      mode: 'structured',
      mutableFiles: POLYMARKET_MUTABLE_FILES,
      allowedMutatorIds: DEFAULT_STRUCTURED_MUTATOR_IDS,
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

export const FORECAST_LAB_PROFILES = deepFreeze([
  PROFILES_BY_ID['btc-markov-short-horizon'],
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

export function isForecastLabProfileId(profileId: string): profileId is ForecastLabProfileId {
  return Object.hasOwn(PROFILES_BY_ID, profileId);
}

export function assertForecastLabProfileId(profileId: string): asserts profileId is ForecastLabProfileId {
  if (!isForecastLabProfileId(profileId)) {
    throw new ForecastLabProfileError(`Unknown forecast-lab profile id: ${profileId}`);
  }
}

export function getForecastLabProfile(profileId: string): ForecastLabProfile {
  assertForecastLabProfileId(profileId);
  return PROFILES_BY_ID[profileId];
}
