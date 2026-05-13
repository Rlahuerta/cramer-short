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
  | 'gold-markov-short-horizon'
  | 'sol-markov-short-horizon'
  | 'hype-markov-short-horizon'
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
  readonly requiredTerms?: readonly string[];
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
  'src/tools/finance/markov-distribution/core.ts',
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
    env: {
      RUN_INTEGRATION: '1',
      FORECAST_LAB_OUTPUT_METRICS: '1',
    },
    timeoutMs: 360_000,
  },
] as const satisfies readonly ForecastLabCommand[]);

const WALK_FORWARD_GOLD_SHORT_HORIZON_COMMANDS = deepFreeze([
  {
    id: 'walk-forward-gold-short-horizon',
    command: 'bun test src/tools/finance/backtest/walk-forward-gold-short-horizon.test.ts --timeout 480000',
    env: {
      RUN_INTEGRATION: '1',
      FORECAST_LAB_OUTPUT_METRICS: '1',
    },
    timeoutMs: 480_000,
  },
] as const satisfies readonly ForecastLabCommand[]);

const WALK_FORWARD_SOL_SHORT_HORIZON_COMMANDS = deepFreeze([
  {
    id: 'walk-forward-sol-short-horizon',
    command: 'bun test src/tools/finance/backtest/walk-forward-sol-short-horizon.test.ts --timeout 480000',
    env: {
      RUN_INTEGRATION: '1',
      FORECAST_LAB_OUTPUT_METRICS: '1',
    },
    timeoutMs: 480_000,
  },
] as const satisfies readonly ForecastLabCommand[]);

const WALK_FORWARD_HYPE_SHORT_HORIZON_COMMANDS = deepFreeze([
  {
    id: 'walk-forward-hype-short-horizon',
    command: 'bun test src/tools/finance/backtest/walk-forward-hype-short-horizon.test.ts --timeout 480000',
    env: {
      RUN_INTEGRATION: '1',
      FORECAST_LAB_OUTPUT_METRICS: '1',
    },
    timeoutMs: 480_000,
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

const GOLD_MARKOV_SHORT_HORIZON_ROUTING = deepFreeze({
  summary: 'Improve GOLD short-horizon Markov behavior for 1d/2d/3d forecasts with 7d/14d guardrails.',
  requiredTerms: ['markov', 'gold-markov-short-horizon'],
  keywordGroups: [
    { label: 'gold asset', terms: ['gold', 'gld', 'xauusd'] },
    {
      label: 'review horizons',
      terms: ['1d/2d/3d', '1d', '2d', '3d', '7d', '14d'],
    },
    { label: 'markov context', terms: ['markov', 'short-horizon', 'short horizon'] },
    {
      label: 'profile id',
      terms: ['gold-markov-short-horizon'],
      weight: 3,
    },
  ],
} as const satisfies ForecastLabProfileRoutingMetadata);

const SOL_MARKOV_SHORT_HORIZON_ROUTING = deepFreeze({
  summary: 'Improve SOL short-horizon Markov behavior for 1d/2d/3d forecasts with 7d/14d guardrails.',
  requiredTerms: ['markov'],
  keywordGroups: [
    { label: 'sol asset', terms: ['sol', 'solana', 'solusd', 'solusdt', 'sol-usd'] },
    {
      label: 'review horizons',
      terms: ['1d/2d/3d', '1d', '2d', '3d', '7d', '14d'],
    },
    { label: 'markov context', terms: ['markov', 'short-horizon', 'short horizon'] },
    {
      label: 'profile id',
      terms: ['sol-markov-short-horizon'],
      weight: 3,
    },
  ],
} as const satisfies ForecastLabProfileRoutingMetadata);

const HYPE_MARKOV_SHORT_HORIZON_ROUTING = deepFreeze({
  summary: 'Improve HYPE short-horizon Markov behavior for 1d/2d/3d forecasts with 7d/14d guardrails.',
  requiredTerms: ['markov'],
  keywordGroups: [
    { label: 'hype asset', terms: ['hype', 'hyperliquid', 'hypeusd', 'hypeusdt', 'hype-usd'] },
    {
      label: 'review horizons',
      terms: ['1d/2d/3d', '1d', '2d', '3d', '7d', '14d'],
    },
    { label: 'markov context', terms: ['markov', 'short-horizon', 'short horizon'] },
    {
      label: 'profile id',
      terms: ['hype-markov-short-horizon'],
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
      {
        name: 'btcUltraShortH1DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h1.directionalAccuracy',
        candidatePath: 'candidate.metrics.h1.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'btcUltraShortH1BrierScore',
        baselinePath: 'baseline.metrics.h1.brierScore',
        candidatePath: 'candidate.metrics.h1.brierScore',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'btcUltraShortH1RerunRate',
        baselinePath: 'baseline.metrics.h1.rerunRate',
        candidatePath: 'candidate.metrics.h1.rerunRate',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'btcUltraShortH2DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h2.directionalAccuracy',
        candidatePath: 'candidate.metrics.h2.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'btcUltraShortH3DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h3.directionalAccuracy',
        candidatePath: 'candidate.metrics.h3.directionalAccuracy',
        direction: 'higher-is-better',
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
          {
            metric: 'btcUltraShortH1DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: 0.015,
            reason: 'candidate BTC 1d directional accuracy must improve by at least 1.5 percentage points',
          },
          {
            metric: 'btcUltraShortH1BrierScore',
            operator: 'candidate-delta-lte',
            value: 0,
            reason: 'candidate BTC 1d Brier score must not worsen',
          },
          {
            metric: 'btcUltraShortH1RerunRate',
            operator: 'candidate-value-lte',
            value: 0.80,
            reason: 'candidate BTC 1d rerun rate must stay bounded at or below 80%',
          },
          {
            metric: 'btcUltraShortH1RerunRate',
            operator: 'candidate-delta-lte',
            value: 0.05,
            reason: 'candidate BTC 1d rerun rate must not rise by more than 5 percentage points',
          },
          {
            metric: 'btcUltraShortH2DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: -0.02,
            reason: 'candidate BTC 2d directional accuracy must not regress by more than 2 percentage points',
          },
          {
            metric: 'btcUltraShortH3DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: -0.02,
            reason: 'candidate BTC 3d directional accuracy must not regress by more than 2 percentage points',
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
          {
            metric: 'btcUltraShortH1DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.005,
            reason: 'drop when BTC 1d directional accuracy regresses by more than 0.5 percentage points',
          },
          {
            metric: 'btcUltraShortH2DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.05,
            reason: 'drop when BTC 2d directional accuracy regresses by more than 5 percentage points',
          },
          {
            metric: 'btcUltraShortH3DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.05,
            reason: 'drop when BTC 3d directional accuracy regresses by more than 5 percentage points',
          },
          {
            metric: 'btcUltraShortH1RerunRate',
            operator: 'candidate-value-gte',
            value: 0.85,
            reason: 'drop when BTC 1d rerun rate rises above 85%',
          },
        ],
      },
    },
  },
  'gold-markov-short-horizon': {
    id: 'gold-markov-short-horizon',
    targetSubsystem: 'markov-distribution',
    routing: GOLD_MARKOV_SHORT_HORIZON_ROUTING,
    allowedGlobs: MARKOV_MUTABLE_FILES,
    mutation: defineForecastLabProfileMutationConfig({
      mode: 'structured',
      mutableFiles: MARKOV_MUTABLE_FILES,
      allowedMutatorIds: MARKOV_ALLOWED_MUTATOR_IDS,
    }),
    readOnlyHarnessFiles: ['src/tools/finance/backtest/walk-forward.ts'],
    baselineCommands: WALK_FORWARD_GOLD_SHORT_HORIZON_COMMANDS,
    candidateCommands: WALK_FORWARD_GOLD_SHORT_HORIZON_COMMANDS,
    minimumMetrics: [
      {
        name: 'walkForwardGoldShortHorizonTestExitCode',
        baselinePath: 'baseline.exitCode',
        candidatePath: 'candidate.exitCode',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'goldShortH1DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h1.directionalAccuracy',
        candidatePath: 'candidate.metrics.h1.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'goldShortH2DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h2.directionalAccuracy',
        candidatePath: 'candidate.metrics.h2.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'goldShortH3DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h3.directionalAccuracy',
        candidatePath: 'candidate.metrics.h3.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'goldShortH1BrierScore',
        baselinePath: 'baseline.metrics.h1.brierScore',
        candidatePath: 'candidate.metrics.h1.brierScore',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'goldShortH2BrierScore',
        baselinePath: 'baseline.metrics.h2.brierScore',
        candidatePath: 'candidate.metrics.h2.brierScore',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'goldShortH3BrierScore',
        baselinePath: 'baseline.metrics.h3.brierScore',
        candidatePath: 'candidate.metrics.h3.brierScore',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'goldShortH7DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h7.directionalAccuracy',
        candidatePath: 'candidate.metrics.h7.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'goldShortH14DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h14.directionalAccuracy',
        candidatePath: 'candidate.metrics.h14.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
    ],
    keepDropRule: {
      defaultDecision: 'drop',
      keepWhen: {
        all: [
          {
            metric: 'walkForwardGoldShortHorizonTestExitCode',
            operator: 'candidate-value-lte',
            value: 0,
            reason: 'candidate GOLD short-horizon test command must pass',
          },
          {
            metric: 'walkForwardGoldShortHorizonTestExitCode',
            operator: 'candidate-delta-lte',
            value: 0,
            reason: 'candidate GOLD short-horizon test command must not regress versus baseline',
          },
          {
            metric: 'goldShortH1DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: 0.01,
            reason: 'candidate GOLD 1d directional accuracy must improve by at least 1 percentage point',
          },
          {
            metric: 'goldShortH2DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: -0.01,
            reason: 'candidate GOLD 2d directional accuracy must stay within 1 percentage point of baseline',
          },
          {
            metric: 'goldShortH3DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: -0.015,
            reason: 'candidate GOLD 3d directional accuracy must stay within 1.5 percentage points of baseline',
          },
          {
            metric: 'goldShortH1BrierScore',
            operator: 'candidate-delta-lte',
            value: 0,
            reason: 'candidate GOLD 1d Brier score must not worsen',
          },
          {
            metric: 'goldShortH2BrierScore',
            operator: 'candidate-delta-lte',
            value: 0.005,
            reason: 'candidate GOLD 2d Brier score must stay within the +0.005 short-horizon band',
          },
          {
            metric: 'goldShortH3BrierScore',
            operator: 'candidate-delta-lte',
            value: 0.005,
            reason: 'candidate GOLD 3d Brier score must stay within the +0.005 short-horizon band',
          },
        ],
      },
      dropWhen: {
        any: [
          {
            metric: 'walkForwardGoldShortHorizonTestExitCode',
            operator: 'candidate-value-gte',
            value: 1,
            reason: 'drop when the GOLD short-horizon test command fails',
          },
          {
            metric: 'goldShortH1DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.0025,
            reason: 'drop when GOLD 1d directional accuracy regresses by more than 0.25 percentage points',
          },
          {
            metric: 'goldShortH2DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.04,
            reason: 'drop when GOLD 2d directional accuracy regresses by more than 4 percentage points',
          },
          {
            metric: 'goldShortH3DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.04,
            reason: 'drop when GOLD 3d directional accuracy regresses by more than 4 percentage points',
          },
          {
            metric: 'goldShortH1BrierScore',
            operator: 'candidate-delta-gte',
            value: 0.01,
            reason: 'drop when GOLD 1d Brier score regresses by more than 0.01',
          },
          {
            metric: 'goldShortH2BrierScore',
            operator: 'candidate-delta-gte',
            value: 0.012,
            reason: 'drop when GOLD 2d Brier score regresses by more than 0.012',
          },
          {
            metric: 'goldShortH3BrierScore',
            operator: 'candidate-delta-gte',
            value: 0.012,
            reason: 'drop when GOLD 3d Brier score regresses by more than 0.012',
          },
          {
            metric: 'goldShortH7DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.08,
            reason: 'drop when GOLD 7d directional accuracy breaks the safety guardrail by more than 8 percentage points',
          },
          {
            metric: 'goldShortH14DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.08,
            reason: 'drop when GOLD 14d directional accuracy breaks the safety guardrail by more than 8 percentage points',
          },
        ],
      },
    },
  },
  'sol-markov-short-horizon': {
    id: 'sol-markov-short-horizon',
    targetSubsystem: 'markov-distribution',
    routing: SOL_MARKOV_SHORT_HORIZON_ROUTING,
    allowedGlobs: MARKOV_MUTABLE_FILES,
    mutation: defineForecastLabProfileMutationConfig({
      mode: 'structured',
      mutableFiles: MARKOV_MUTABLE_FILES,
      allowedMutatorIds: MARKOV_ALLOWED_MUTATOR_IDS,
    }),
    readOnlyHarnessFiles: ['src/tools/finance/backtest/walk-forward.ts'],
    baselineCommands: WALK_FORWARD_SOL_SHORT_HORIZON_COMMANDS,
    candidateCommands: WALK_FORWARD_SOL_SHORT_HORIZON_COMMANDS,
    minimumMetrics: [
      {
        name: 'walkForwardSolShortHorizonTestExitCode',
        baselinePath: 'baseline.exitCode',
        candidatePath: 'candidate.exitCode',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'solShortH1DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h1.directionalAccuracy',
        candidatePath: 'candidate.metrics.h1.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'solShortH2DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h2.directionalAccuracy',
        candidatePath: 'candidate.metrics.h2.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'solShortH3DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h3.directionalAccuracy',
        candidatePath: 'candidate.metrics.h3.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'solShortH1BrierScore',
        baselinePath: 'baseline.metrics.h1.brierScore',
        candidatePath: 'candidate.metrics.h1.brierScore',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'solShortH2BrierScore',
        baselinePath: 'baseline.metrics.h2.brierScore',
        candidatePath: 'candidate.metrics.h2.brierScore',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'solShortH3BrierScore',
        baselinePath: 'baseline.metrics.h3.brierScore',
        candidatePath: 'candidate.metrics.h3.brierScore',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'solShortH7DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h7.directionalAccuracy',
        candidatePath: 'candidate.metrics.h7.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'solShortH14DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h14.directionalAccuracy',
        candidatePath: 'candidate.metrics.h14.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
    ],
    keepDropRule: {
      defaultDecision: 'drop',
      keepWhen: {
        all: [
          {
            metric: 'walkForwardSolShortHorizonTestExitCode',
            operator: 'candidate-value-lte',
            value: 0,
            reason: 'candidate SOL short-horizon test command must pass',
          },
          {
            metric: 'walkForwardSolShortHorizonTestExitCode',
            operator: 'candidate-delta-lte',
            value: 0,
            reason: 'candidate SOL short-horizon test command must not regress versus baseline',
          },
          {
            metric: 'solShortH1DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: 0.02,
            reason: 'candidate SOL 1d directional accuracy must improve by at least 2 percentage points',
          },
          {
            metric: 'solShortH2DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: -0.015,
            reason: 'candidate SOL 2d directional accuracy must stay within 1.5 percentage points of baseline',
          },
          {
            metric: 'solShortH3DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: -0.015,
            reason: 'candidate SOL 3d directional accuracy must stay within 1.5 percentage points of baseline',
          },
          {
            metric: 'solShortH1BrierScore',
            operator: 'candidate-delta-lte',
            value: 0,
            reason: 'candidate SOL 1d Brier score must not worsen',
          },
          {
            metric: 'solShortH2BrierScore',
            operator: 'candidate-delta-lte',
            value: 0.004,
            reason: 'candidate SOL 2d Brier score must stay within the +0.004 short-horizon band',
          },
          {
            metric: 'solShortH3BrierScore',
            operator: 'candidate-delta-lte',
            value: 0.004,
            reason: 'candidate SOL 3d Brier score must stay within the +0.004 short-horizon band',
          },
        ],
      },
      dropWhen: {
        any: [
          {
            metric: 'walkForwardSolShortHorizonTestExitCode',
            operator: 'candidate-value-gte',
            value: 1,
            reason: 'drop when the SOL short-horizon test command fails',
          },
          {
            metric: 'solShortH1DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.01,
            reason: 'drop when SOL 1d directional accuracy regresses by more than 1 percentage point',
          },
          {
            metric: 'solShortH2DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.05,
            reason: 'drop when SOL 2d directional accuracy regresses by more than 5 percentage points',
          },
          {
            metric: 'solShortH3DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.05,
            reason: 'drop when SOL 3d directional accuracy regresses by more than 5 percentage points',
          },
          {
            metric: 'solShortH1BrierScore',
            operator: 'candidate-delta-gte',
            value: 0.01,
            reason: 'drop when SOL 1d Brier score regresses by more than 0.01',
          },
          {
            metric: 'solShortH2BrierScore',
            operator: 'candidate-delta-gte',
            value: 0.012,
            reason: 'drop when SOL 2d Brier score regresses by more than 0.012',
          },
          {
            metric: 'solShortH3BrierScore',
            operator: 'candidate-delta-gte',
            value: 0.012,
            reason: 'drop when SOL 3d Brier score regresses by more than 0.012',
          },
          {
            metric: 'solShortH7DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.08,
            reason: 'drop when SOL 7d directional accuracy breaks the safety guardrail by more than 8 percentage points',
          },
          {
            metric: 'solShortH14DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.08,
            reason: 'drop when SOL 14d directional accuracy breaks the safety guardrail by more than 8 percentage points',
          },
        ],
      },
    },
  },
  'hype-markov-short-horizon': {
    id: 'hype-markov-short-horizon',
    targetSubsystem: 'markov-distribution',
    routing: HYPE_MARKOV_SHORT_HORIZON_ROUTING,
    allowedGlobs: MARKOV_MUTABLE_FILES,
    mutation: defineForecastLabProfileMutationConfig({
      mode: 'structured',
      mutableFiles: MARKOV_MUTABLE_FILES,
      allowedMutatorIds: MARKOV_ALLOWED_MUTATOR_IDS,
    }),
    readOnlyHarnessFiles: ['src/tools/finance/backtest/walk-forward.ts'],
    baselineCommands: WALK_FORWARD_HYPE_SHORT_HORIZON_COMMANDS,
    candidateCommands: WALK_FORWARD_HYPE_SHORT_HORIZON_COMMANDS,
    minimumMetrics: [
      {
        name: 'walkForwardHypeShortHorizonTestExitCode',
        baselinePath: 'baseline.exitCode',
        candidatePath: 'candidate.exitCode',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'hypeShortH1DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h1.directionalAccuracy',
        candidatePath: 'candidate.metrics.h1.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'hypeShortH2DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h2.directionalAccuracy',
        candidatePath: 'candidate.metrics.h2.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'hypeShortH3DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h3.directionalAccuracy',
        candidatePath: 'candidate.metrics.h3.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'hypeShortH1BrierScore',
        baselinePath: 'baseline.metrics.h1.brierScore',
        candidatePath: 'candidate.metrics.h1.brierScore',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'hypeShortH2BrierScore',
        baselinePath: 'baseline.metrics.h2.brierScore',
        candidatePath: 'candidate.metrics.h2.brierScore',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'hypeShortH3BrierScore',
        baselinePath: 'baseline.metrics.h3.brierScore',
        candidatePath: 'candidate.metrics.h3.brierScore',
        direction: 'lower-is-better',
        required: true,
      },
      {
        name: 'hypeShortH7DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h7.directionalAccuracy',
        candidatePath: 'candidate.metrics.h7.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
      {
        name: 'hypeShortH14DirectionalAccuracy',
        baselinePath: 'baseline.metrics.h14.directionalAccuracy',
        candidatePath: 'candidate.metrics.h14.directionalAccuracy',
        direction: 'higher-is-better',
        required: true,
      },
    ],
    keepDropRule: {
      defaultDecision: 'drop',
      keepWhen: {
        all: [
          {
            metric: 'walkForwardHypeShortHorizonTestExitCode',
            operator: 'candidate-value-lte',
            value: 0,
            reason: 'candidate HYPE short-horizon test command must pass',
          },
          {
            metric: 'walkForwardHypeShortHorizonTestExitCode',
            operator: 'candidate-delta-lte',
            value: 0,
            reason: 'candidate HYPE short-horizon test command must not regress versus baseline',
          },
          {
            metric: 'hypeShortH1DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: 0.015,
            reason: 'candidate HYPE 1d directional accuracy must improve by at least 1.5 percentage points',
          },
          {
            metric: 'hypeShortH2DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: -0.01,
            reason: 'candidate HYPE 2d directional accuracy must stay within 1 percentage point of baseline',
          },
          {
            metric: 'hypeShortH3DirectionalAccuracy',
            operator: 'candidate-delta-gte',
            value: -0.015,
            reason: 'candidate HYPE 3d directional accuracy must stay within 1.5 percentage points of baseline',
          },
          {
            metric: 'hypeShortH1BrierScore',
            operator: 'candidate-delta-lte',
            value: 0,
            reason: 'candidate HYPE 1d Brier score must not worsen',
          },
          {
            metric: 'hypeShortH2BrierScore',
            operator: 'candidate-delta-lte',
            value: 0.005,
            reason: 'candidate HYPE 2d Brier score must stay within the +0.005 short-horizon band',
          },
          {
            metric: 'hypeShortH3BrierScore',
            operator: 'candidate-delta-lte',
            value: 0.005,
            reason: 'candidate HYPE 3d Brier score must stay within the +0.005 short-horizon band',
          },
        ],
      },
      dropWhen: {
        any: [
          {
            metric: 'walkForwardHypeShortHorizonTestExitCode',
            operator: 'candidate-value-gte',
            value: 1,
            reason: 'drop when the HYPE short-horizon test command fails',
          },
          {
            metric: 'hypeShortH1DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.005,
            reason: 'drop when HYPE 1d directional accuracy regresses by more than 0.5 percentage points',
          },
          {
            metric: 'hypeShortH2DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.04,
            reason: 'drop when HYPE 2d directional accuracy regresses by more than 4 percentage points',
          },
          {
            metric: 'hypeShortH3DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.04,
            reason: 'drop when HYPE 3d directional accuracy regresses by more than 4 percentage points',
          },
          {
            metric: 'hypeShortH1BrierScore',
            operator: 'candidate-delta-gte',
            value: 0.01,
            reason: 'drop when HYPE 1d Brier score regresses by more than 0.01',
          },
          {
            metric: 'hypeShortH2BrierScore',
            operator: 'candidate-delta-gte',
            value: 0.012,
            reason: 'drop when HYPE 2d Brier score regresses by more than 0.012',
          },
          {
            metric: 'hypeShortH3BrierScore',
            operator: 'candidate-delta-gte',
            value: 0.012,
            reason: 'drop when HYPE 3d Brier score regresses by more than 0.012',
          },
          {
            metric: 'hypeShortH7DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.08,
            reason: 'drop when HYPE 7d directional accuracy breaks the safety guardrail by more than 8 percentage points',
          },
          {
            metric: 'hypeShortH14DirectionalAccuracy',
            operator: 'candidate-delta-lte',
            value: -0.08,
            reason: 'drop when HYPE 14d directional accuracy breaks the safety guardrail by more than 8 percentage points',
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
  PROFILES_BY_ID['gold-markov-short-horizon'],
  PROFILES_BY_ID['sol-markov-short-horizon'],
  PROFILES_BY_ID['hype-markov-short-horizon'],
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
