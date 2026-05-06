import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { EventEmitter } from 'node:events';
import { existsSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { spawnSync } from 'node:child_process';
import { join, resolve, sep } from 'node:path';
import { PassThrough } from 'node:stream';
import { runForecastLabCommand } from '../../cli-forecast-lab.js';
import { getExperimentRunDir, getExperimentRunManifestPath } from '../../utils/paths.js';
import {
  getForecastLabCandidateWorktreePath,
  makeForecastLabCandidateBranch,
  getForecastLabPromotionWorktreePath,
  makeForecastLabPromotionBranch,
} from './git.js';
import { appendLedgerEntry, readLedgerEntries, readRunManifest, writeRunManifest } from './ledger.js';
import { getForecastLabProfile, listForecastLabStructuredMutations } from './profiles.js';
import type { ForecastLabCommandRunner } from './runner.js';
import {
  buildForecastLabRuntimeDefaultsActivation,
  ForecastLabRunnerError,
  createForecastLabCommandRunner,
  defaultForecastLabCommandRunner,
  resolveForecastLabRuntimeDefaultsForAssetScope,
  resolveForecastLabMutatorRankingEnabled,
  runForecastLab,
  promoteForecastLab,
  resetForecastLab,
} from './runner.js';
import {
  snapshotForecastLabMarkovParameterMutation,
  type ForecastLabMarkovParameterMutationCandidate,
} from './mutators/markov-parameters.js';
import {
  FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS,
  getForecastLabMarkovRuntimeDefaults,
  resolveForecastLabMarkovParameterDefaults,
  setForecastLabMarkovRuntimeDefaults,
} from '../../tools/finance/markov-distribution.js';
import {
  FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS,
  getForecastLabConformalRuntimeDefaults,
  resolveForecastLabConformalParameterDefaults,
  setForecastLabConformalRuntimeDefaults,
} from '../../tools/finance/conformal.js';
import {
  FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS,
  getForecastLabRegimeCalibratorRuntimeDefaults,
  resolveForecastLabRegimeCalibratorDefaults,
  setForecastLabRegimeCalibratorRuntimeDefaults,
} from '../../tools/finance/regime-calibrator.js';
import type { ForecastLabRuntimeAssetScope } from '../../tools/finance/forecast-lab-runtime-defaults.js';

const TEST_LEDGER_DIR = join('.cramer-short', 'experiments', '__runner_test__');
const TEST_LEDGER_PATH = join(TEST_LEDGER_DIR, 'forecast-results.tsv');
const TEST_ROUTING_STATS_PATH = join('.cramer-short', '__runner_test__', 'forecast-lab-routing-stats.json');
const RUN_IDS = [
  'runner-test-dry-run',
  'runner-test-unknown',
  'runner-test-failed-candidate',
  'runner-test-structured',
  'runner-test-structured-dirty-live-checkout',
  'runner-test-structured-parent-root',
  'runner-test-structured-parent-child',
  'runner-test-structured-auto-parent',
  'runner-test-structured-auto-child',
  'runner-test-structured-payload-parent',
  'runner-test-structured-payload-child',
  'runner-test-structured-keep-worktree',
  'runner-test-ranked-history-long-keep',
  'runner-test-ranked-history-short-drop',
  'runner-test-ranked-history-seed-parent',
  'runner-test-ranked-auto-child',
  'runner-test-ranked-override-child',
  'runner-test-btc-structured-parent-1',
  'runner-test-btc-structured-parent-2',
  'runner-test-btc-structured-parent-3',
  'runner-test-btc-structured-exhausted',
  'runner-test-skip-mutation',
  'runner-test-outside-ledger',
  'runner-test-promote-source',
  'runner-test-promote-verify',
  'runner-test-promote-legacy-source',
  'runner-test-promote-legacy-verify',
  'runner-test-promote-concurrent-source',
  'runner-test-promote-concurrent-verify-a',
  'runner-test-promote-concurrent-verify-b',
  'runner-test-promote-stale-source',
  'runner-test-promote-stale-verify',
  'runner-test-promote-missing-payload-source',
  'runner-test-promote-missing-payload-verify',
  'runner-test-promote-regression-source',
  'runner-test-promote-regression-verify',
  'runner-test-gold-metric-gate',
  'runner-test-gold-guardrail-reject',
  'runner-test-btc-promote-source',
  'runner-test-btc-promote-verify',
  'runner-test-gold-promote-source-a',
  'runner-test-gold-promote-verify-a',
  'runner-test-gold-promote-source-b',
  'runner-test-gold-promote-verify-b',
  'runner-test-gold-reset-defaults',
  'runner-test-gold-reset-last-known-good',
  'runner-test-reset-source-a',
  'runner-test-reset-promote-a',
  'runner-test-reset-source-b',
  'runner-test-reset-promote-b',
  'runner-test-reset-defaults',
  'runner-test-reset-last-known-good',
];

function cleanup(): void {
  rmSync(TEST_LEDGER_DIR, { recursive: true, force: true });
  rmSync(TEST_ROUTING_STATS_PATH, { force: true });
  rmSync(join('.cramer-short', 'experiments', 'active-promotions'), { recursive: true, force: true });
  for (const runId of RUN_IDS) {
    const worktreePath = getForecastLabCandidateWorktreePath(runId);
    if (existsSync(worktreePath)) {
      spawnSync('git', ['worktree', 'remove', '--force', worktreePath], { stdio: 'ignore' });
    }
    spawnSync('git', ['branch', '-D', makeForecastLabCandidateBranch(runId)], { stdio: 'ignore' });
    const promotionWorktreePath = getForecastLabPromotionWorktreePath(runId);
    if (existsSync(promotionWorktreePath)) {
      spawnSync('git', ['worktree', 'remove', '--force', promotionWorktreePath], { stdio: 'ignore' });
    }
    spawnSync('git', ['branch', '-D', makeForecastLabPromotionBranch(runId)], { stdio: 'ignore' });
    rmSync(getExperimentRunDir(runId), { recursive: true, force: true });
  }
}

function passingRunner(calls: string[]): ForecastLabCommandRunner {
  return async (command, context) => {
    calls.push(`${context.phase}:${command.id}`);
    return {
      id: command.id,
      command: command.command,
      exitCode: 0,
      stdout: `${context.phase} ok`,
      stderr: '',
      durationMs: 1,
      timedOut: false,
    };
  };
}

function buildBtcUltraShortMetrics(params?: {
  h1DirectionalAccuracy?: number;
  h1BrierScore?: number;
  h1RerunRate?: number;
  h2DirectionalAccuracy?: number;
  h3DirectionalAccuracy?: number;
}): Record<string, Record<string, number>> {
  return {
    h1: {
      directionalAccuracy: params?.h1DirectionalAccuracy ?? 0.62,
      brierScore: params?.h1BrierScore ?? 0.252,
      ciCoverage: 0.972,
      rerunRate: params?.h1RerunRate ?? 0.75,
    },
    h2: {
      directionalAccuracy: params?.h2DirectionalAccuracy ?? 0.52,
      brierScore: 0.254,
      ciCoverage: 0.989,
      rerunRate: 0,
    },
    h3: {
      directionalAccuracy: params?.h3DirectionalAccuracy ?? 0.56,
      brierScore: 0.261,
      ciCoverage: 0.978,
      rerunRate: 0.30,
    },
  };
}

function btcMetricsRunner(calls: string[], metricsByPhase?: {
  baseline?: Record<string, Record<string, number>>;
  candidate?: Record<string, Record<string, number>>;
}): ForecastLabCommandRunner {
  return async (command, context) => {
    calls.push(`${context.phase}:${command.id}`);
    const metrics = metricsByPhase?.[context.phase] ?? buildBtcUltraShortMetrics();
    return {
      id: command.id,
      command: command.command,
      exitCode: 0,
      stdout: `${context.phase} ok\nFORECAST_LAB_METRICS ${JSON.stringify(metrics)}\n`,
      stderr: '',
      durationMs: 1,
      timedOut: false,
    };
  };
}

function buildGoldShortMetrics(params?: {
  h1DirectionalAccuracy?: number;
  h2DirectionalAccuracy?: number;
  h3DirectionalAccuracy?: number;
  h1BrierScore?: number;
  h2BrierScore?: number;
  h3BrierScore?: number;
  h7DirectionalAccuracy?: number;
  h14DirectionalAccuracy?: number;
}): Record<string, Record<string, number>> {
  return {
    h1: {
      directionalAccuracy: params?.h1DirectionalAccuracy ?? 0.58,
      brierScore: params?.h1BrierScore ?? 0.248,
      ciCoverage: 0.964,
    },
    h2: {
      directionalAccuracy: params?.h2DirectionalAccuracy ?? 0.55,
      brierScore: params?.h2BrierScore ?? 0.251,
      ciCoverage: 0.972,
    },
    h3: {
      directionalAccuracy: params?.h3DirectionalAccuracy ?? 0.57,
      brierScore: params?.h3BrierScore ?? 0.256,
      ciCoverage: 0.975,
    },
    h7: {
      directionalAccuracy: params?.h7DirectionalAccuracy ?? 0.64,
      brierScore: 0.244,
      ciCoverage: 0.983,
    },
    h14: {
      directionalAccuracy: params?.h14DirectionalAccuracy ?? 0.68,
      brierScore: 0.238,
      ciCoverage: 0.989,
    },
  };
}

function goldMetricsRunner(calls: string[], metricsByPhase?: {
  baseline?: Record<string, Record<string, number>>;
  candidate?: Record<string, Record<string, number>>;
}): ForecastLabCommandRunner {
  return async (command, context) => {
    calls.push(`${context.phase}:${command.id}`);
    const metrics = metricsByPhase?.[context.phase] ?? buildGoldShortMetrics();
    return {
      id: command.id,
      command: command.command,
      exitCode: 0,
      stdout: `${context.phase} ok\nFORECAST_LAB_METRICS ${JSON.stringify(metrics)}\n`,
      stderr: '',
      durationMs: 1,
      timedOut: false,
    };
  };
}

function keptBtcMetricsRunner(calls: string[] = []): ForecastLabCommandRunner {
  return btcMetricsRunner(calls, {
    baseline: buildBtcUltraShortMetrics({
      h1DirectionalAccuracy: 0.62,
      h1BrierScore: 0.252,
      h1RerunRate: 0.75,
      h2DirectionalAccuracy: 0.52,
      h3DirectionalAccuracy: 0.56,
    }),
    candidate: buildBtcUltraShortMetrics({
      h1DirectionalAccuracy: 0.64,
      h1BrierScore: 0.247,
      h1RerunRate: 0.78,
      h2DirectionalAccuracy: 0.51,
      h3DirectionalAccuracy: 0.55,
    }),
  });
}

function keptGoldMetricsRunner(calls: string[] = []): ForecastLabCommandRunner {
  return goldMetricsRunner(calls, {
    baseline: buildGoldShortMetrics({
      h1DirectionalAccuracy: 0.58,
      h2DirectionalAccuracy: 0.55,
      h3DirectionalAccuracy: 0.57,
      h1BrierScore: 0.248,
      h2BrierScore: 0.251,
      h3BrierScore: 0.256,
      h7DirectionalAccuracy: 0.64,
      h14DirectionalAccuracy: 0.68,
    }),
    candidate: buildGoldShortMetrics({
      h1DirectionalAccuracy: 0.595,
      h2DirectionalAccuracy: 0.545,
      h3DirectionalAccuracy: 0.56,
      h1BrierScore: 0.245,
      h2BrierScore: 0.255,
      h3BrierScore: 0.26,
      h7DirectionalAccuracy: 0.6,
      h14DirectionalAccuracy: 0.65,
    }),
  });
}

const LIVE_MUTABLE_FILES = [
  'src/tools/finance/markov-distribution.ts',
  'src/tools/finance/conformal.ts',
  'src/tools/finance/regime-calibrator.ts',
] as const;

function snapshotLiveMutableFiles(): Map<(typeof LIVE_MUTABLE_FILES)[number], string> {
  return new Map(
    LIVE_MUTABLE_FILES.map((filePath) => [filePath, readFileSync(filePath, 'utf8')]),
  );
}

function restoreLiveMutableFiles(snapshot: Map<(typeof LIVE_MUTABLE_FILES)[number], string>): void {
  for (const [filePath, contents] of snapshot.entries()) {
    writeFileSync(filePath, contents, 'utf8');
  }
}

function snapshotRuntimeDefaults() {
  const shipped = {
    markov: { ...FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS },
    conformal: { ...FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS },
    regime: { ...FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS },
  };

  return {
    ...shipped,
    shipped,
    active: {
      shared: {
        markov: getForecastLabMarkovRuntimeDefaults('shared'),
        conformal: getForecastLabConformalRuntimeDefaults('shared'),
        regime: getForecastLabRegimeCalibratorRuntimeDefaults('shared'),
      },
      btc: {
        markov: getForecastLabMarkovRuntimeDefaults('btc'),
        conformal: getForecastLabConformalRuntimeDefaults('btc'),
        regime: getForecastLabRegimeCalibratorRuntimeDefaults('btc'),
      },
      gold: {
        markov: getForecastLabMarkovRuntimeDefaults('gold'),
        conformal: getForecastLabConformalRuntimeDefaults('gold'),
        regime: getForecastLabRegimeCalibratorRuntimeDefaults('gold'),
      },
    },
  };
}

function restoreRuntimeDefaults(snapshot: ReturnType<typeof snapshotRuntimeDefaults>): void {
  Object.assign(FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS as Record<string, unknown>, snapshot.shipped.markov);
  Object.assign(FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS as Record<string, unknown>, snapshot.shipped.conformal);
  Object.assign(FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS as Record<string, unknown>, snapshot.shipped.regime);
  setForecastLabMarkovRuntimeDefaults('shared', snapshot.active.shared.markov);
  setForecastLabConformalRuntimeDefaults('shared', snapshot.active.shared.conformal);
  setForecastLabRegimeCalibratorRuntimeDefaults('shared', snapshot.active.shared.regime);
  setForecastLabMarkovRuntimeDefaults('btc', snapshot.active.btc.markov);
  setForecastLabConformalRuntimeDefaults('btc', snapshot.active.btc.conformal);
  setForecastLabRegimeCalibratorRuntimeDefaults('btc', snapshot.active.btc.regime);
  setForecastLabMarkovRuntimeDefaults('gold', snapshot.active.gold.markov);
  setForecastLabConformalRuntimeDefaults('gold', snapshot.active.gold.conformal);
  setForecastLabRegimeCalibratorRuntimeDefaults('gold', snapshot.active.gold.regime);
}

function readHeadTrackedFile(filePath: (typeof LIVE_MUTABLE_FILES)[number]): string {
  const result = spawnSync('git', ['show', `HEAD:${filePath}`], {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'pipe'],
  });
  if (result.status !== 0) {
    throw new Error(`Failed to read ${filePath} from HEAD: ${result.stderr || result.stdout || `exit ${result.status}`}`);
  }
  return result.stdout;
}

const ORIGINAL_LIVE_FILES = snapshotLiveMutableFiles();
const ORIGINAL_RUNTIME_DEFAULTS = snapshotRuntimeDefaults();
const SHIPPED_LIVE_FILES = new Map(
  LIVE_MUTABLE_FILES.map((filePath) => [filePath, readHeadTrackedFile(filePath)]),
);
const SHIPPED_RUNTIME_DEFAULTS = snapshotRuntimeDefaults();

function getStructuredMutationFixture(
  profileId: 'multi-asset-markov-short-horizon' | 'btc-markov-ultra-short-horizon' | 'gold-markov-short-horizon',
  mutationId: string,
): ForecastLabMarkovParameterMutationCandidate {
  const mutation = listForecastLabStructuredMutations(profileId).find((candidate) => candidate.id === mutationId);
  if (!mutation) {
    throw new Error(`Missing test mutation fixture: ${profileId}/${mutationId}`);
  }
  return mutation;
}

function getStructuredMutationEdit(
  mutation: ForecastLabMarkovParameterMutationCandidate,
  parameterId: string,
) {
  const edit = mutation.edits.find((candidateEdit) => candidateEdit.parameterId === parameterId);
  if (!edit) {
    throw new Error(`Missing edit "${parameterId}" for mutation ${mutation.id}`);
  }
  return edit;
}

function getStructuredMutationLine(
  mutation: ForecastLabMarkovParameterMutationCandidate,
  parameterId: string,
): string {
  const edit = getStructuredMutationEdit(mutation, parameterId);
  return `  ${parameterId}: ${edit.afterValue},`;
}

function getStructuredMutationNumericAfterValue(
  mutation: ForecastLabMarkovParameterMutationCandidate,
  parameterId: string,
): number {
  const value = getStructuredMutationEdit(mutation, parameterId).afterValue;
  if (typeof value !== 'number') {
    throw new Error(`Expected numeric afterValue for ${mutation.id}/${parameterId}`);
  }
  return value;
}

function resolveEffectiveRuntimeDefaults(assetScope: ForecastLabRuntimeAssetScope) {
  return {
    markov: resolveForecastLabMarkovParameterDefaults(assetScope),
    conformal: resolveForecastLabConformalParameterDefaults(assetScope),
    regime: resolveForecastLabRegimeCalibratorDefaults(assetScope),
  };
}

function expectResolvedDefaultsToMatchShipped(
  resolved: ReturnType<typeof resolveForecastLabRuntimeDefaultsForAssetScope>,
): void {
  expect(resolved.markov.momentumLookback).toBe(SHIPPED_RUNTIME_DEFAULTS.markov.momentumLookback);
  expect(resolved.conformal.scoreAggregationCalibrationWindow)
    .toBe(SHIPPED_RUNTIME_DEFAULTS.conformal.scoreAggregationCalibrationWindow);
  expect(resolved.regime.minSamplesPerRegime).toBe(SHIPPED_RUNTIME_DEFAULTS.regime.minSamplesPerRegime);
}

function expectResolvedDefaultsToMatchMutation(
  resolved: ReturnType<typeof resolveForecastLabRuntimeDefaultsForAssetScope>,
  mutation: ForecastLabMarkovParameterMutationCandidate,
): void {
  expect(resolved.markov.momentumLookback).toBe(getStructuredMutationNumericAfterValue(mutation, 'momentumLookback'));
  expect(resolved.conformal.scoreAggregationCalibrationWindow)
    .toBe(getStructuredMutationNumericAfterValue(mutation, 'scoreAggregationCalibrationWindow'));
  expect(resolved.regime.minSamplesPerRegime)
    .toBe(getStructuredMutationNumericAfterValue(mutation, 'minSamplesPerRegime'));
}

const MULTI_ASSET_SHORTER_REACTIVE_WINDOW = getStructuredMutationFixture(
  'multi-asset-markov-short-horizon',
  'markov-shorter-reactive-window',
);
const MULTI_ASSET_FASTER_DECAY_REACTION = getStructuredMutationFixture(
  'multi-asset-markov-short-horizon',
  'markov-faster-decay-reaction',
);
const MULTI_ASSET_LONGER_STABILITY_WINDOW = getStructuredMutationFixture(
  'multi-asset-markov-short-horizon',
  'markov-longer-stability-window',
);
const BTC_SHORTER_REACTIVE_WINDOW = getStructuredMutationFixture(
  'btc-markov-ultra-short-horizon',
  'markov-shorter-reactive-window',
);
const GOLD_SHORTER_REACTIVE_WINDOW = getStructuredMutationFixture(
  'gold-markov-short-horizon',
  'gold-markov-shorter-reactive-window',
);
const GOLD_FASTER_DECAY_REACTION = getStructuredMutationFixture(
  'gold-markov-short-horizon',
  'gold-markov-faster-decay-reaction',
);

function appendStructuredMutationHistory(params: {
  readonly profileId?: string;
  readonly runId: string;
  readonly mutationId: string;
  readonly decision: 'keep' | 'drop';
  readonly startedAt: string;
  readonly parentRunId?: string;
}): void {
  const profile = getForecastLabProfile(params.profileId ?? 'multi-asset-markov-short-horizon');
  const mutation = listForecastLabStructuredMutations(profile.id).find((candidate) => candidate.id === params.mutationId);

  if (!mutation) {
    throw new Error(`Unknown test mutation: ${params.mutationId}`);
  }

  const parentManifest = params.parentRunId
    ? readRunManifest(getExperimentRunManifestPath(params.parentRunId))
    : undefined;
  const lineage = {
    rootRunId: parentManifest?.lineage?.rootRunId ?? params.runId,
    ...(params.parentRunId ? { parentRunId: params.parentRunId } : {}),
    generation: (parentManifest?.lineage?.generation ?? -1) + 1,
  } as const;

  const artifactsPath = getExperimentRunDir(params.runId, { create: true });
  const promotion = params.decision === 'keep'
    ? {
        status: 'approval-required' as const,
        source: {
          runId: params.runId,
          manifestPath: getExperimentRunManifestPath(params.runId),
        },
        requestedAt: params.startedAt,
      }
    : undefined;
  writeRunManifest(getExperimentRunManifestPath(params.runId, { create: true }), {
    runId: params.runId,
    startedAt: params.startedAt,
    profileId: profile.id,
    targetSubsystem: profile.targetSubsystem,
    candidateBranch: makeForecastLabCandidateBranch(params.runId),
    allowedGlobs: [...profile.allowedGlobs],
    mutationMode: 'structured',
    ...(params.parentRunId ? { parentRunId: params.parentRunId } : {}),
    mutationId: mutation.id,
    mutationSummary: mutation.specSummary.summary,
    lineage,
    mutationSpecSummary: mutation.specSummary,
    candidateWorkspace: {
      kind: 'candidate-worktree',
      rootDir: resolve('.cramer-short', 'experiments', 'worktrees', params.runId),
      branch: makeForecastLabCandidateBranch(params.runId),
    },
    ...(promotion ? { promotion } : {}),
    artifactsPath,
  });

  const candidateExitCode = params.decision === 'keep' ? 0 : 1;
  appendLedgerEntry(TEST_LEDGER_PATH, {
    runId: params.runId,
    startedAt: params.startedAt,
    profileId: profile.id,
    targetSubsystem: profile.targetSubsystem,
    candidateBranch: makeForecastLabCandidateBranch(params.runId),
    allowedGlobs: [...profile.allowedGlobs],
    mutationMode: 'structured',
    ...(params.parentRunId ? { parentRunId: params.parentRunId } : {}),
    mutationId: mutation.id,
    mutationSummary: mutation.specSummary.summary,
    lineage,
    mutationSpecSummary: mutation.specSummary,
    candidateWorkspace: {
      kind: 'candidate-worktree',
      rootDir: resolve('.cramer-short', 'experiments', 'worktrees', params.runId),
      branch: makeForecastLabCandidateBranch(params.runId),
    },
    ...(promotion ? { promotion } : {}),
    baselineSummary: {
      exitCode: 0,
      commands: [
        {
          id: 'walk-forward-short-horizon',
          exitCode: 0,
          durationMs: 1,
          timedOut: false,
        },
      ],
    },
    candidateSummary: {
      exitCode: candidateExitCode,
      commands: [
        {
          id: 'walk-forward-short-horizon',
          exitCode: candidateExitCode,
          durationMs: 1,
          timedOut: false,
        },
      ],
    },
    decision: params.decision,
    reason: params.decision === 'keep' ? 'fixture keep' : 'fixture drop',
    artifactsPath,
  });
}

class FakeSpawnedChild extends EventEmitter {
  readonly stdout = new PassThrough();
  readonly stderr = new PassThrough();
  readonly killSignals: Array<number | NodeJS.Signals | undefined> = [];

  kill(signal?: number | NodeJS.Signals): boolean {
    this.killSignals.push(signal);
    return true;
  }
}

beforeEach(() => {
  cleanup();
  restoreLiveMutableFiles(SHIPPED_LIVE_FILES);
  restoreRuntimeDefaults(SHIPPED_RUNTIME_DEFAULTS);
});

afterEach(() => {
  cleanup();
  restoreLiveMutableFiles(ORIGINAL_LIVE_FILES);
  restoreRuntimeDefaults(ORIGINAL_RUNTIME_DEFAULTS);
});

describe('forecast-lab runner', () => {
  it('defaults mutator ranking off when the rollout flag is absent', () => {
    expect(resolveForecastLabMutatorRankingEnabled(undefined, undefined)).toBe(false);
  });

  it('enables mutator ranking from config when no explicit runner override is provided', () => {
    expect(resolveForecastLabMutatorRankingEnabled(undefined, {
      enableForecastLabMutatorRanking: true,
    })).toBe(true);
  });

  it('keeps explicit runner mutator-ranking overrides ahead of config', () => {
    expect(resolveForecastLabMutatorRankingEnabled(false, {
      enableForecastLabMutatorRanking: true,
    })).toBe(false);
    expect(resolveForecastLabMutatorRankingEnabled(true, {
      enableForecastLabMutatorRanking: false,
    })).toBe(true);
  });

  it('dry-run completes baseline, candidate, decision, and ledger write with an injected command runner', async () => {
    const calls: string[] = [];
    const progress: string[] = [];
    const output: string[] = [];
    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      dryRun: true,
      runId: 'runner-test-dry-run',
      now: () => new Date('2026-05-02T00:00:00.000Z'),
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner(calls),
      progress: (message) => progress.push(message),
      output: (chunk) => output.push(chunk),
    });

    expect(calls).toEqual([
      'baseline:walk-forward-short-horizon',
      'candidate:walk-forward-short-horizon',
    ]);
    expect(result.decision.decision).toBe('keep');
    expect(result.manifest.profileId).toBe('multi-asset-markov-short-horizon');
    expect(result.manifest.baselineCommit).toMatch(/^[0-9a-f]{40}$/);
    expect(result.manifest.candidateWorkspace).toBeUndefined();
    expect(result.manifest.promotion).toBeUndefined();
    expect(result.manifest.effectiveMutationContract).toEqual({
      mode: 'structured',
      mutableFiles: [
        'src/tools/finance/markov-distribution.ts',
        'src/tools/finance/conformal.ts',
        'src/tools/finance/regime-calibrator.ts',
      ],
      allowedMutatorIds: ['search-replace'],
      allowMultipleCandidateAttempts: false,
    });

    const runDir = getExperimentRunDir('runner-test-dry-run');
    for (const fileName of ['manifest.json', 'baseline.json', 'candidate.json', 'decision.json']) {
      expect(existsSync(join(runDir, fileName))).toBe(true);
      expect(resolve(join(runDir, fileName)).startsWith(resolve('.cramer-short', 'experiments') + sep)).toBe(true);
    }

    const candidate = JSON.parse(readFileSync(join(runDir, 'candidate.json'), 'utf8')) as Record<string, unknown>;
    expect(candidate.mutation).toBe('dry-run: no code mutation attempted');
    expect(result.candidate as unknown).toEqual(candidate);
    const manifest = JSON.parse(readFileSync(join(runDir, 'manifest.json'), 'utf8')) as Record<string, unknown>;
    expect(manifest.baselineCommit).toMatch(/^[0-9a-f]{40}$/);
    expect(manifest).not.toHaveProperty('candidateWorkspace');
    expect(manifest).not.toHaveProperty('promotion');
    expect(manifest.effectiveMutationContract).toEqual(result.manifest.effectiveMutationContract);

    const entries = readLedgerEntries(TEST_LEDGER_PATH);
    expect(entries).toHaveLength(1);
    expect(entries[0]).toMatchObject({
      runId: 'runner-test-dry-run',
      decision: 'keep',
      artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-dry-run'),
      effectiveMutationContract: result.manifest.effectiveMutationContract,
    });
    expect(entries[0]).not.toHaveProperty('candidateWorkspace');
    expect(entries[0]).not.toHaveProperty('promotion');
    expect(progress).toEqual([
      'forecast-lab: started multi-asset-markov-short-horizon (runner-test-dry-run)',
      `forecast-lab: manifest written to ${join('.cramer-short', 'experiments', 'runs', 'runner-test-dry-run', 'manifest.json')}`,
      'forecast-lab: starting baseline gate',
      'baseline: running walk-forward-short-horizon — bun test src/tools/finance/backtest/walk-forward-short-horizon.test.ts --timeout 480000',
      'baseline: completed walk-forward-short-horizon (exit 0, 1ms)',
      `forecast-lab: baseline results written to ${join('.cramer-short', 'experiments', 'runs', 'runner-test-dry-run', 'baseline.json')}`,
      'forecast-lab: starting candidate gate (dry-run)',
      'candidate: running walk-forward-short-horizon — bun test src/tools/finance/backtest/walk-forward-short-horizon.test.ts --timeout 480000',
      'candidate: completed walk-forward-short-horizon (exit 0, 1ms)',
      `forecast-lab: candidate results written to ${join('.cramer-short', 'experiments', 'runs', 'runner-test-dry-run', 'candidate.json')}`,
      'forecast-lab: decision keep — candidate walk-forward short-horizon test command must pass; candidate walk-forward short-horizon test command must not regress versus baseline',
      `forecast-lab: ledger appended at ${TEST_LEDGER_PATH}`,
    ]);
    expect(output).toEqual([]);
  });

  it('rejects unknown profiles before running commands', async () => {
    const calls: string[] = [];

    await expect(runForecastLab({
      profileId: 'unknown-profile',
      dryRun: true,
      runId: 'runner-test-unknown',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner(calls),
    })).rejects.toThrow(/Unknown forecast-lab profile id/);

    expect(calls).toEqual([]);
    expect(existsSync(getExperimentRunDir('runner-test-unknown'))).toBe(false);
  });

  it('drops a failed candidate from command exit codes', async () => {
    const result = await runForecastLab({
      profileId: 'polymarket-selection-sanity',
      dryRun: true,
      runId: 'runner-test-failed-candidate',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: async (command, context) => ({
        id: command.id,
        command: command.command,
        exitCode: context.phase === 'candidate' ? 1 : 0,
        stdout: '',
        stderr: context.phase === 'candidate' ? 'failed' : '',
        durationMs: 1,
        timedOut: false,
      }),
    });

    expect(result.decision.decision).toBe('drop');
    expect(result.decision.reason).toMatch(/drop when the Polymarket forecast sanity test command fails/);
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).toMatchObject({
      runId: 'runner-test-failed-candidate',
      decision: 'drop',
    });
  });

  it('uses parsed BTC harness metrics for the ultra-short-horizon keep/drop decision', async () => {
    const calls: string[] = [];
    const baselineMetrics = buildBtcUltraShortMetrics({
      h1DirectionalAccuracy: 0.62,
      h1BrierScore: 0.252,
      h1RerunRate: 0.75,
      h2DirectionalAccuracy: 0.52,
      h3DirectionalAccuracy: 0.56,
    });
    const candidateMetrics = buildBtcUltraShortMetrics({
      h1DirectionalAccuracy: 0.64,
      h1BrierScore: 0.247,
      h1RerunRate: 0.78,
      h2DirectionalAccuracy: 0.51,
      h3DirectionalAccuracy: 0.55,
    });

    const result = await runForecastLab({
      profileId: 'btc-markov-ultra-short-horizon',
      dryRun: true,
      runId: 'runner-test-btc-metric-gate',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: btcMetricsRunner(calls, {
        baseline: baselineMetrics,
        candidate: candidateMetrics,
      }),
    });

    expect(calls).toEqual([
      'baseline:walk-forward-btc-ultra-short-horizon',
      'candidate:walk-forward-btc-ultra-short-horizon',
    ]);
    expect(result.decision.decision).toBe('keep');
    expect(result.decision.reason).toContain('candidate BTC 1d directional accuracy must improve');
    expect(result.decision.metrics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'walkForwardBtcUltraShortHorizonTestExitCode',
          baseline: 0,
          candidate: 0,
          delta: 0,
        }),
        expect.objectContaining({
          name: 'btcUltraShortH1DirectionalAccuracy',
          baseline: 0.62,
          candidate: 0.64,
          delta: 0.020000000000000018,
        }),
      ]),
    );

    const runDir = getExperimentRunDir('runner-test-btc-metric-gate');
    const baseline = JSON.parse(readFileSync(join(runDir, 'baseline.json'), 'utf8')) as Record<string, unknown>;
    const candidate = JSON.parse(readFileSync(join(runDir, 'candidate.json'), 'utf8')) as Record<string, unknown>;
    expect(baseline.metrics).toEqual(baselineMetrics);
    expect(candidate.metrics).toEqual(candidateMetrics);
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).toMatchObject({
      runId: 'runner-test-btc-metric-gate',
      decision: 'keep',
      baselineSummary: expect.objectContaining({
        metrics: baselineMetrics,
      }),
      candidateSummary: expect.objectContaining({
        metrics: candidateMetrics,
      }),
    });
  });

  it('uses parsed GOLD harness metrics for the 1d-first keep/drop decision', async () => {
    const calls: string[] = [];
    const baselineMetrics = buildGoldShortMetrics({
      h1DirectionalAccuracy: 0.58,
      h2DirectionalAccuracy: 0.55,
      h3DirectionalAccuracy: 0.57,
      h1BrierScore: 0.248,
      h2BrierScore: 0.251,
      h3BrierScore: 0.256,
      h7DirectionalAccuracy: 0.64,
      h14DirectionalAccuracy: 0.68,
    });
    const candidateMetrics = buildGoldShortMetrics({
      h1DirectionalAccuracy: 0.595,
      h2DirectionalAccuracy: 0.545,
      h3DirectionalAccuracy: 0.56,
      h1BrierScore: 0.245,
      h2BrierScore: 0.255,
      h3BrierScore: 0.26,
      h7DirectionalAccuracy: 0.6,
      h14DirectionalAccuracy: 0.65,
    });

    const result = await runForecastLab({
      profileId: 'gold-markov-short-horizon',
      dryRun: true,
      runId: 'runner-test-gold-metric-gate',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: goldMetricsRunner(calls, {
        baseline: baselineMetrics,
        candidate: candidateMetrics,
      }),
    });

    expect(calls).toEqual([
      'baseline:walk-forward-gold-short-horizon',
      'candidate:walk-forward-gold-short-horizon',
    ]);
    expect(result.decision.decision).toBe('keep');
    expect(result.decision.reason).toContain('candidate GOLD 1d directional accuracy must improve');
    expect(result.decision.metrics).toEqual(
      expect.arrayContaining([
        expect.objectContaining({
          name: 'goldShortH1DirectionalAccuracy',
          baseline: 0.58,
          candidate: 0.595,
          delta: 0.015000000000000013,
        }),
        expect.objectContaining({
          name: 'goldShortH7DirectionalAccuracy',
          baseline: 0.64,
          candidate: 0.6,
          delta: -0.040000000000000036,
        }),
      ]),
    );
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).toMatchObject({
      runId: 'runner-test-gold-metric-gate',
      decision: 'keep',
      baselineSummary: expect.objectContaining({
        metrics: baselineMetrics,
      }),
      candidateSummary: expect.objectContaining({
        metrics: candidateMetrics,
      }),
    });
  });

  it('does not keep GOLD candidates on 7d/14d guardrails alone when 1d regresses', async () => {
    const result = await runForecastLab({
      profileId: 'gold-markov-short-horizon',
      dryRun: true,
      runId: 'runner-test-gold-guardrail-reject',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: goldMetricsRunner([], {
        baseline: buildGoldShortMetrics({
          h1DirectionalAccuracy: 0.58,
          h2DirectionalAccuracy: 0.55,
          h3DirectionalAccuracy: 0.57,
          h1BrierScore: 0.248,
          h2BrierScore: 0.251,
          h3BrierScore: 0.256,
          h7DirectionalAccuracy: 0.64,
          h14DirectionalAccuracy: 0.68,
        }),
        candidate: buildGoldShortMetrics({
          h1DirectionalAccuracy: 0.572,
          h2DirectionalAccuracy: 0.558,
          h3DirectionalAccuracy: 0.575,
          h1BrierScore: 0.246,
          h2BrierScore: 0.25,
          h3BrierScore: 0.255,
          h7DirectionalAccuracy: 0.9,
          h14DirectionalAccuracy: 0.93,
        }),
      }),
    });

    expect(result.decision).toMatchObject({
      decision: 'drop',
      reason: 'drop when GOLD 1d directional accuracy regresses by more than 0.25 percentage points',
    });
  });

  it('documents that mutating shipped shared defaults would leak a GOLD activation into BTC', () => {
    const runtimeDefaults = snapshotRuntimeDefaults();

    try {
      const goldMomentumLookback = getStructuredMutationNumericAfterValue(
        MULTI_ASSET_LONGER_STABILITY_WINDOW,
        'momentumLookback',
      );
      const goldCalibrationWindow = getStructuredMutationNumericAfterValue(
        MULTI_ASSET_LONGER_STABILITY_WINDOW,
        'scoreAggregationCalibrationWindow',
      );
      const goldMinSamplesPerRegime = getStructuredMutationNumericAfterValue(
        MULTI_ASSET_LONGER_STABILITY_WINDOW,
        'minSamplesPerRegime',
      );

      Object.assign(FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS as Record<string, unknown>, {
        momentumLookback: goldMomentumLookback,
      });
      Object.assign(FORECAST_LAB_CONFORMAL_PARAMETER_DEFAULTS as Record<string, unknown>, {
        scoreAggregationCalibrationWindow: goldCalibrationWindow,
      });
      Object.assign(FORECAST_LAB_REGIME_CALIBRATOR_DEFAULTS as Record<string, unknown>, {
        minSamplesPerRegime: goldMinSamplesPerRegime,
      });

      const currentBtcEffectiveDefaults = snapshotRuntimeDefaults();
      expect(currentBtcEffectiveDefaults.markov.momentumLookback).toBe(goldMomentumLookback);
      expect(currentBtcEffectiveDefaults.conformal.scoreAggregationCalibrationWindow).toBe(goldCalibrationWindow);
      expect(currentBtcEffectiveDefaults.regime.minSamplesPerRegime).toBe(goldMinSamplesPerRegime);
      expect(currentBtcEffectiveDefaults.markov.momentumLookback).not.toBe(runtimeDefaults.markov.momentumLookback);
    } finally {
      restoreRuntimeDefaults(runtimeDefaults);
    }
  });

  it('resolves shipped defaults, shared multi-asset overrides, BTC overrides, and GOLD overrides by asset scope', () => {
    const btcActivation = buildForecastLabRuntimeDefaultsActivation({
      profileId: 'btc-markov-ultra-short-horizon',
      assetScope: 'btc',
      mutation: BTC_SHORTER_REACTIVE_WINDOW,
    });
    const sharedActivation = buildForecastLabRuntimeDefaultsActivation({
      profileId: 'multi-asset-markov-short-horizon',
      assetScope: 'shared',
      mutation: MULTI_ASSET_LONGER_STABILITY_WINDOW,
    });
    const goldActivation = buildForecastLabRuntimeDefaultsActivation({
      profileId: 'gold-markov-short-horizon',
      assetScope: 'gold',
      mutation: GOLD_SHORTER_REACTIVE_WINDOW,
    });

    const baseSharedDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('shared');
    const shippedFallbackGoldDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('gold', [btcActivation]);
    const sharedActiveDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('shared', [btcActivation, sharedActivation]);
    const goldInheritedSharedDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('gold', [btcActivation, sharedActivation]);
    const btcActiveDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('btc', [btcActivation, sharedActivation, goldActivation]);
    const goldActiveDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('gold', [btcActivation, sharedActivation, goldActivation]);

    expectResolvedDefaultsToMatchShipped(baseSharedDefaults);
    expectResolvedDefaultsToMatchShipped(shippedFallbackGoldDefaults);
    expectResolvedDefaultsToMatchMutation(sharedActiveDefaults, MULTI_ASSET_LONGER_STABILITY_WINDOW);
    expectResolvedDefaultsToMatchMutation(goldInheritedSharedDefaults, MULTI_ASSET_LONGER_STABILITY_WINDOW);
    expectResolvedDefaultsToMatchMutation(btcActiveDefaults, BTC_SHORTER_REACTIVE_WINDOW);
    expectResolvedDefaultsToMatchMutation(goldActiveDefaults, GOLD_SHORTER_REACTIVE_WINDOW);
  });

  it('keeps shared and GOLD activation, reset, and restore isolated from BTC effective runtime defaults', () => {
    const btcActivation = buildForecastLabRuntimeDefaultsActivation({
      profileId: 'btc-markov-ultra-short-horizon',
      assetScope: 'btc',
      mutation: BTC_SHORTER_REACTIVE_WINDOW,
    });
    const sharedActivation = buildForecastLabRuntimeDefaultsActivation({
      profileId: 'multi-asset-markov-short-horizon',
      assetScope: 'shared',
      mutation: MULTI_ASSET_LONGER_STABILITY_WINDOW,
    });
    const goldActivation = buildForecastLabRuntimeDefaultsActivation({
      profileId: 'gold-markov-short-horizon',
      assetScope: 'gold',
      mutation: GOLD_SHORTER_REACTIVE_WINDOW,
    });

    const sharedAndBtcActive = [btcActivation, sharedActivation] as const;
    const allActive = [btcActivation, sharedActivation, goldActivation] as const;
    const afterGoldActivationBtcDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('btc', allActive);
    const afterGoldActivationSharedDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('shared', allActive);
    const afterGoldActivationGoldDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('gold', allActive);
    const afterGoldResetBtcDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('btc', sharedAndBtcActive);
    const afterGoldResetSharedDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('shared', sharedAndBtcActive);
    const afterGoldResetGoldDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('gold', sharedAndBtcActive);
    const afterGoldRestoreBtcDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('btc', allActive);
    const afterGoldRestoreSharedDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('shared', allActive);
    const afterGoldRestoreGoldDefaults = resolveForecastLabRuntimeDefaultsForAssetScope('gold', allActive);

    expectResolvedDefaultsToMatchMutation(afterGoldActivationBtcDefaults, BTC_SHORTER_REACTIVE_WINDOW);
    expectResolvedDefaultsToMatchMutation(afterGoldActivationSharedDefaults, MULTI_ASSET_LONGER_STABILITY_WINDOW);
    expectResolvedDefaultsToMatchMutation(afterGoldActivationGoldDefaults, GOLD_SHORTER_REACTIVE_WINDOW);
    expectResolvedDefaultsToMatchMutation(afterGoldResetBtcDefaults, BTC_SHORTER_REACTIVE_WINDOW);
    expectResolvedDefaultsToMatchMutation(afterGoldResetSharedDefaults, MULTI_ASSET_LONGER_STABILITY_WINDOW);
    expectResolvedDefaultsToMatchMutation(afterGoldResetGoldDefaults, MULTI_ASSET_LONGER_STABILITY_WINDOW);
    expectResolvedDefaultsToMatchMutation(afterGoldRestoreBtcDefaults, BTC_SHORTER_REACTIVE_WINDOW);
    expectResolvedDefaultsToMatchMutation(afterGoldRestoreSharedDefaults, MULTI_ASSET_LONGER_STABILITY_WINDOW);
    expectResolvedDefaultsToMatchMutation(afterGoldRestoreGoldDefaults, GOLD_SHORTER_REACTIVE_WINDOW);
  });

  it('fails closed for the BTC ultra-short-horizon profile when parsed harness metrics are missing', async () => {
    const result = await runForecastLab({
      profileId: 'btc-markov-ultra-short-horizon',
      dryRun: true,
      runId: 'runner-test-btc-metric-missing',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    expect(result.decision).toMatchObject({
      decision: 'drop',
      reason: 'missing required metric: btcUltraShortH1DirectionalAccuracy',
    });
  });

  it('persists routing context to manifests and ledgers while updating per-profile routing stats', async () => {
    const autoRoutedResult = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      dryRun: true,
      runId: 'runner-test-dry-run',
      now: () => new Date('2026-05-02T00:00:00.000Z'),
      ledgerPath: TEST_LEDGER_PATH,
      routingStatsPath: TEST_ROUTING_STATS_PATH,
      routingContext: {
        originatingQuery: 'Improve the short-horizon Markov calibration.',
        selectedProfileId: 'multi-asset-markov-short-horizon',
        routerReason: 'Matched improvement intent and Markov short-horizon routing keywords.',
        invocationSource: 'auto-routed',
      },
      commandRunner: passingRunner([]),
    });

    const manualResult = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      dryRun: true,
      runId: 'runner-test-failed-candidate',
      now: () => new Date('2026-05-03T00:00:00.000Z'),
      ledgerPath: TEST_LEDGER_PATH,
      routingStatsPath: TEST_ROUTING_STATS_PATH,
      routingContext: {
        originatingQuery: 'Run the Markov short-horizon skill manually.',
        selectedProfileId: 'multi-asset-markov-short-horizon',
        routerReason: 'User explicitly requested the shipped Markov short-horizon skill.',
        invocationSource: 'manual-request',
      },
      commandRunner: async (command, context) => ({
        id: command.id,
        command: command.command,
        exitCode: context.phase === 'candidate' ? 1 : 0,
        stdout: '',
        stderr: context.phase === 'candidate' ? 'failed' : '',
        durationMs: 1,
        timedOut: false,
      }),
    });

    expect(autoRoutedResult.manifest.routingContext).toEqual({
      originatingQuery: 'Improve the short-horizon Markov calibration.',
      selectedProfileId: 'multi-asset-markov-short-horizon',
      routerReason: 'Matched improvement intent and Markov short-horizon routing keywords.',
      invocationSource: 'auto-routed',
    });
    expect(manualResult.ledgerEntry.routingContext).toEqual({
      originatingQuery: 'Run the Markov short-horizon skill manually.',
      selectedProfileId: 'multi-asset-markov-short-horizon',
      routerReason: 'User explicitly requested the shipped Markov short-horizon skill.',
      invocationSource: 'manual-request',
    });

    expect(
      JSON.parse(readFileSync(join(getExperimentRunDir('runner-test-dry-run'), 'manifest.json'), 'utf8')),
    ).toMatchObject({
      routingContext: autoRoutedResult.manifest.routingContext,
    });

    expect(readLedgerEntries(TEST_LEDGER_PATH)).toEqual([
      expect.objectContaining({
        runId: 'runner-test-dry-run',
        routingContext: autoRoutedResult.manifest.routingContext,
        decision: 'keep',
      }),
      expect.objectContaining({
        runId: 'runner-test-failed-candidate',
        routingContext: manualResult.manifest.routingContext,
        decision: 'drop',
      }),
    ]);

    expect(JSON.parse(readFileSync(TEST_ROUTING_STATS_PATH, 'utf8'))).toEqual({
      profiles: {
        'multi-asset-markov-short-horizon': {
          autoRoutedRuns: 1,
          droppedRuns: 1,
          keptRuns: 1,
          lastDecision: 'drop',
          lastRunAt: '2026-05-03T00:00:00.000Z',
          manualRequestedRuns: 1,
          totalRuns: 2,
        },
      },
      version: 1,
    });
  });

  it('runs one real structured mutation inside an isolated candidate workspace and records its metadata', async () => {
    const calls: string[] = [];
    const progress: string[] = [];
    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      runId: 'runner-test-structured',
      ledgerPath: TEST_LEDGER_PATH,
      progress: (message) => progress.push(message),
      commandRunner: async (command, context) => {
        calls.push(`${context.phase}:${command.id}:${context.cwd ?? ''}`);

        if (context.phase === 'candidate') {
          const candidateRoot = context.cwd!;
          expect(candidateRoot).toBe(resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured'));
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'momentumLookback'),
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/conformal.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow'),
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/regime-calibrator.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'minSamplesPerRegime'),
          );
        } else {
          expect(context.cwd).toBe(process.cwd());
        }

        return {
          id: command.id,
          command: command.command,
          exitCode: 0,
          stdout: `${context.phase} ok`,
          stderr: '',
          durationMs: 1,
          timedOut: false,
        };
      },
    });

    expect(calls).toEqual([
      `baseline:walk-forward-short-horizon:${process.cwd()}`,
      `candidate:walk-forward-short-horizon:${resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured')}`,
    ]);
    expect(result.decision.decision).toBe('keep');
    expect(result.manifest.mutationMode).toBe('structured');
    expect(result.manifest.parentRunId).toBeUndefined();
    expect(result.manifest.mutationId).toBe('markov-shorter-reactive-window');
    expect(result.manifest.mutationSummary).toBe(MULTI_ASSET_SHORTER_REACTIVE_WINDOW.specSummary.summary);
    expect(result.manifest.lineage).toEqual({
      rootRunId: 'runner-test-structured',
      generation: 0,
    });
    expect(result.manifest.mutationSpecSummary).toEqual({
      mutatorId: 'search-replace',
      targetFiles: [...MULTI_ASSET_SHORTER_REACTIVE_WINDOW.specSummary.targetFiles],
      summary: MULTI_ASSET_SHORTER_REACTIVE_WINDOW.specSummary.summary,
    });
    expect(result.manifest.mutationReplayPayload).toMatchObject({
      kind: 'markov-parameter-candidate',
      id: 'markov-shorter-reactive-window',
      profileId: 'multi-asset-markov-short-horizon',
    });
    expect(result.manifest.promotion).toMatchObject({
      status: 'approval-required',
      source: {
        runId: 'runner-test-structured',
        manifestPath: getExperimentRunManifestPath('runner-test-structured'),
      },
    });
    expect(result.manifest.promotion?.requestedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);
    const candidateWorkspace = result.manifest.candidateWorkspace;
    expect(candidateWorkspace).toEqual({
      kind: 'candidate-worktree',
      rootDir: resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured'),
      branch: 'topic/forecast-lab-runner-test-structured',
    });
    expect(candidateWorkspace).toBeDefined();
    expect(existsSync(candidateWorkspace!.rootDir)).toBe(false);

    const runDir = getExperimentRunDir('runner-test-structured');
    const candidate = JSON.parse(readFileSync(join(runDir, 'candidate.json'), 'utf8')) as Record<string, unknown>;
    expect(candidate.mutationMode).toBe('structured');
    expect(candidate.mutatedFiles).toEqual([
      'src/tools/finance/markov-distribution.ts',
      'src/tools/finance/conformal.ts',
      'src/tools/finance/regime-calibrator.ts',
    ]);
    expect(result.candidate as unknown).toEqual(candidate);
    expect(candidate.selectedMutator).toEqual({
      id: 'markov-shorter-reactive-window',
      mutatorId: 'search-replace',
    });
    expect(candidate.patchSummary).toEqual([...MULTI_ASSET_SHORTER_REACTIVE_WINDOW.patchSummary]);

    const decision = JSON.parse(readFileSync(join(runDir, 'decision.json'), 'utf8')) as Record<string, unknown>;
    expect(decision.mutationMode).toBe('structured');
    expect(decision.candidateWorkspace).toEqual(candidateWorkspace);
    expect(decision.promotion).toEqual(result.manifest.promotion);

    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).toMatchObject({
      runId: 'runner-test-structured',
      decision: 'keep',
      mutationMode: 'structured',
      mutationId: 'markov-shorter-reactive-window',
      mutationSummary: MULTI_ASSET_SHORTER_REACTIVE_WINDOW.specSummary.summary,
      lineage: {
        rootRunId: 'runner-test-structured',
        generation: 0,
      },
      mutationSpecSummary: result.manifest.mutationSpecSummary,
      candidateWorkspace,
      promotion: result.manifest.promotion,
    });
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).not.toHaveProperty('parentRunId');
    expect(progress).toContain(
      `forecast-lab: candidate workspace ${resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured')}`,
    );
    expect(progress).toContain('forecast-lab: selected mutator markov-shorter-reactive-window (search-replace)');
    expect(progress).toContain(
      'forecast-lab: mutated files src/tools/finance/markov-distribution.ts, src/tools/finance/conformal.ts, src/tools/finance/regime-calibrator.ts',
    );
    expect(progress).toContain(
      `forecast-lab: patch summary ${MULTI_ASSET_SHORTER_REACTIVE_WINDOW.patchSummary.join(' | ')}`,
    );
    expect(progress.indexOf('forecast-lab: selected mutator markov-shorter-reactive-window (search-replace)'))
      .toBeLessThan(progress.indexOf('forecast-lab: starting baseline gate'));
  });

  it('selects structured mutations against the clean candidate workspace instead of a dirty live checkout', async () => {
    const liveCheckoutPath = join(process.cwd(), 'src/tools/finance/markov-distribution.ts');
    const originalContents = readFileSync(liveCheckoutPath, 'utf8');
    const shippedMomentumLookback = SHIPPED_RUNTIME_DEFAULTS.markov.momentumLookback;
    expect(originalContents).toContain(`  momentumLookback: ${shippedMomentumLookback},`);

    writeFileSync(
      liveCheckoutPath,
      originalContents.replace(
        `  momentumLookback: ${shippedMomentumLookback},`,
        '  momentumLookback: 999,',
      ),
    );

    try {
      const result = await runForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        mutationMode: 'structured',
        runId: 'runner-test-structured-dirty-live-checkout',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: async (command, context) => {
          if (context.phase === 'candidate') {
            const candidateRoot = context.cwd!;
            expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
              getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'momentumLookback'),
            );
            expect(readFileSync(liveCheckoutPath, 'utf8')).toContain('  momentumLookback: 999,');
          }

          return {
            id: command.id,
            command: command.command,
            exitCode: 0,
            stdout: `${context.phase} ok`,
            stderr: '',
            durationMs: 1,
            timedOut: false,
          };
        },
      });

      expect(result.manifest.mutationId).toBe('markov-shorter-reactive-window');
    } finally {
      writeFileSync(liveCheckoutPath, originalContents);
    }
  });

  it('seeds the next structured mutation from the last kept structured run lineage', async () => {
    await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      runId: 'runner-test-structured-parent-root',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    const progress: string[] = [];
    const calls: string[] = [];
    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      mutator: 'markov-faster-decay-reaction',
      runId: 'runner-test-structured-parent-child',
      ledgerPath: TEST_LEDGER_PATH,
      progress: (message) => progress.push(message),
      commandRunner: async (command, context) => {
        calls.push(`${context.phase}:${command.id}:${context.cwd ?? ''}`);
        if (context.phase === 'candidate') {
          const candidateRoot = context.cwd!;
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'momentumLookback'),
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_FASTER_DECAY_REACTION, 'transitionDecay'),
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/conformal.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow'),
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/conformal.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_FASTER_DECAY_REACTION, 'adaptiveBreakLearningRateMultiplier'),
          );
        }

        return {
          id: command.id,
          command: command.command,
          exitCode: 0,
          stdout: `${context.phase} ok`,
          stderr: '',
          durationMs: 1,
          timedOut: false,
        };
      },
    });

    expect(calls).toEqual([
      `baseline:walk-forward-short-horizon:${process.cwd()}`,
      `candidate:walk-forward-short-horizon:${resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured-parent-child')}`,
    ]);
    expect(result.manifest.parentRunId).toBe('runner-test-structured-parent-root');
    expect(result.manifest.mutationId).toBe('markov-faster-decay-reaction');
    expect(result.manifest.mutationSummary).toBe(MULTI_ASSET_FASTER_DECAY_REACTION.specSummary.summary);
    expect(result.manifest.lineage).toEqual({
      rootRunId: 'runner-test-structured-parent-root',
      parentRunId: 'runner-test-structured-parent-root',
      generation: 1,
    });
    expect(progress).toContain(
      'forecast-lab: seeded from kept run runner-test-structured-parent-root (1 replayed mutation)',
    );
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).toMatchObject({
      runId: 'runner-test-structured-parent-child',
      parentRunId: 'runner-test-structured-parent-root',
      mutationId: 'markov-faster-decay-reaction',
      mutationSummary: MULTI_ASSET_FASTER_DECAY_REACTION.specSummary.summary,
      lineage: {
        rootRunId: 'runner-test-structured-parent-root',
        parentRunId: 'runner-test-structured-parent-root',
        generation: 1,
      },
    });
  });

  it('auto-selects the first applicable unused structured mutation after replay', async () => {
    await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      runId: 'runner-test-structured-auto-parent',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    const progress: string[] = [];
    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      runId: 'runner-test-structured-auto-child',
      ledgerPath: TEST_LEDGER_PATH,
      progress: (message) => progress.push(message),
      commandRunner: async (command, context) => {
        if (context.phase === 'candidate') {
          const candidateRoot = context.cwd!;
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'momentumLookback'),
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_FASTER_DECAY_REACTION, 'transitionDecay'),
          );
        }

        return {
          id: command.id,
          command: command.command,
          exitCode: 0,
          stdout: `${context.phase} ok`,
          stderr: '',
          durationMs: 1,
          timedOut: false,
        };
      },
    });

    expect(result.manifest.parentRunId).toBe('runner-test-structured-auto-parent');
    expect(result.manifest.mutationId).toBe('markov-faster-decay-reaction');
    expect(progress).toContain('forecast-lab: selected mutator markov-faster-decay-reaction (search-replace)');
  });

  it('prefers better-ranked kept structured mutations when ranking is enabled', async () => {
    appendStructuredMutationHistory({
      runId: 'runner-test-ranked-history-long-keep',
      mutationId: 'markov-longer-stability-window',
      decision: 'keep',
      startedAt: '2026-05-01T00:00:00.000Z',
    });
    appendStructuredMutationHistory({
      runId: 'runner-test-ranked-history-short-drop',
      mutationId: 'markov-shorter-reactive-window',
      decision: 'drop',
      startedAt: '2026-05-02T00:00:00.000Z',
    });
    appendStructuredMutationHistory({
      runId: 'runner-test-ranked-history-seed-parent',
      mutationId: 'markov-faster-decay-reaction',
      decision: 'keep',
      startedAt: '2026-05-03T00:00:00.000Z',
    });

    const progress: string[] = [];
    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      rankMutators: true,
      runId: 'runner-test-ranked-auto-child',
      ledgerPath: TEST_LEDGER_PATH,
      progress: (message) => progress.push(message),
      commandRunner: async (command, context) => {
        if (context.phase === 'candidate') {
          const candidateRoot = context.cwd!;
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_FASTER_DECAY_REACTION, 'transitionDecay'),
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_LONGER_STABILITY_WINDOW, 'momentumLookback'),
          );
        }

        return {
          id: command.id,
          command: command.command,
          exitCode: 0,
          stdout: `${context.phase} ok`,
          stderr: '',
          durationMs: 1,
          timedOut: false,
        };
      },
    });

    expect(result.manifest.parentRunId).toBe('runner-test-ranked-history-seed-parent');
    expect(result.manifest.mutationId).toBe('markov-longer-stability-window');
    const mutatorRanking = (result.candidate as Record<string, unknown>).mutatorRanking as {
      enabled: boolean;
      profileId: string;
      totalStructuredRuns: number;
      rankedMutators: Array<Record<string, unknown>>;
    };
    expect(mutatorRanking.enabled).toBe(true);
    expect(mutatorRanking.profileId).toBe('multi-asset-markov-short-horizon');
    expect(mutatorRanking.totalStructuredRuns).toBe(3);
    expect(mutatorRanking.rankedMutators[0]).toMatchObject({
      id: 'markov-longer-stability-window',
      health: 'healthy',
    });
    expect(mutatorRanking.rankedMutators).toEqual(expect.arrayContaining([
      expect.objectContaining({
        id: 'markov-shorter-reactive-window',
        health: 'underperforming',
        regressedRuns: 1,
      }),
    ]));
    expect(progress.join('\n')).toContain('markov-shorter-reactive-window');
  });

  it('keeps explicit mutator overrides ahead of ranked ordering', async () => {
    appendStructuredMutationHistory({
      runId: 'runner-test-ranked-history-long-keep',
      mutationId: 'markov-longer-stability-window',
      decision: 'keep',
      startedAt: '2026-05-01T00:00:00.000Z',
    });
    appendStructuredMutationHistory({
      runId: 'runner-test-ranked-history-short-drop',
      mutationId: 'markov-shorter-reactive-window',
      decision: 'drop',
      startedAt: '2026-05-02T00:00:00.000Z',
    });
    appendStructuredMutationHistory({
      runId: 'runner-test-ranked-history-seed-parent',
      mutationId: 'markov-faster-decay-reaction',
      decision: 'keep',
      startedAt: '2026-05-03T00:00:00.000Z',
    });

    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      rankMutators: true,
      mutator: 'markov-shorter-reactive-window',
      runId: 'runner-test-ranked-override-child',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    expect(result.manifest.parentRunId).toBe('runner-test-ranked-history-seed-parent');
    expect(result.manifest.mutationId).toBe('markov-shorter-reactive-window');
  });

  it('replays the persisted mutation payload even when the catalog entry is no longer available by mutationId', async () => {
    const parent = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      mutator: 'markov-shorter-reactive-window',
      runId: 'runner-test-structured-payload-parent',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    expect(parent.manifest.mutationReplayPayload).toBeDefined();
    writeRunManifest(getExperimentRunManifestPath('runner-test-structured-payload-parent'), {
      ...parent.manifest,
      mutationId: 'archived-shorter-reactive-window',
      mutationReplayPayload: {
        ...parent.manifest.mutationReplayPayload!,
        id: 'archived-shorter-reactive-window',
      },
    });

    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      mutator: 'markov-faster-decay-reaction',
      runId: 'runner-test-structured-payload-child',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: async (command, context) => {
        if (context.phase === 'candidate') {
          const candidateRoot = context.cwd!;
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'momentumLookback'),
          );
          expect(readFileSync(join(candidateRoot, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
            getStructuredMutationLine(MULTI_ASSET_FASTER_DECAY_REACTION, 'transitionDecay'),
          );
        }

        return {
          id: command.id,
          command: command.command,
          exitCode: 0,
          stdout: `${context.phase} ok`,
          stderr: '',
          durationMs: 1,
          timedOut: false,
        };
      },
    });

    expect(result.manifest.parentRunId).toBe('runner-test-structured-payload-parent');
    expect(result.manifest.mutationId).toBe('markov-faster-decay-reaction');
  });

  it('promotes the latest kept structured run by replaying its persisted payload in an isolated staging workspace', async () => {
    const liveFiles = snapshotLiveMutableFiles();
    const runtimeDefaults = snapshotRuntimeDefaults();

    try {
      const source = await runForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        mutationMode: 'structured',
        mutator: 'markov-shorter-reactive-window',
        runId: 'runner-test-promote-source',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      });
      const progress: string[] = [];
      const calls: string[] = [];
      const promotionWorktreePath = getForecastLabPromotionWorktreePath('runner-test-promote-verify');
      const result = await promoteForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        runId: 'runner-test-promote-verify',
        ledgerPath: TEST_LEDGER_PATH,
        progress: (message) => progress.push(message),
        commandRunner: async (command, context) => {
          calls.push(`${context.phase}:${command.id}:${context.cwd ?? ''}`);

          if (context.phase === 'candidate') {
            expect(context.cwd).toBe(promotionWorktreePath);
            expect(readFileSync(join(promotionWorktreePath, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
              getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'momentumLookback'),
            );
            expect(readFileSync(join(promotionWorktreePath, 'src/tools/finance/conformal.ts'), 'utf8')).toContain(
              getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow'),
            );
          } else {
            expect(context.cwd).toBe(process.cwd());
          }

          return {
            id: command.id,
            command: command.command,
            exitCode: 0,
            stdout: `${context.phase} ok`,
            stderr: '',
            durationMs: 1,
            timedOut: false,
          };
        },
      });

      expect(calls).toEqual([
        `baseline:walk-forward-short-horizon:${process.cwd()}`,
        `candidate:walk-forward-short-horizon:${promotionWorktreePath}`,
      ]);
      expect(result.sourceRunId).toBe('runner-test-promote-source');
      expect(result.activation).toEqual({
        runId: 'runner-test-promote-verify',
        manifestPath: getExperimentRunManifestPath('runner-test-promote-verify'),
        artifactsPath: getExperimentRunDir('runner-test-promote-verify'),
        workspace: {
          kind: 'candidate-worktree',
          rootDir: promotionWorktreePath,
          branch: 'topic/forecast-lab-promote-runner-test-promote-verify',
        },
      });
      expect(result.activeStatePath).toBe(join('.cramer-short', 'experiments', 'active-promotions', 'multi-asset-markov-short-horizon.json'));
      if (!source.manifest.promotion || source.manifest.promotion.status !== 'approval-required') {
        throw new Error('expected the promotion source fixture to start in approval-required state');
      }
      if (!result.sourceManifest.promotion || result.sourceManifest.promotion.status !== 'activated') {
        throw new Error('expected an activated forecast-lab source manifest state');
      }
      const activatedState = result.sourceManifest.promotion;
      expect(activatedState).toEqual({
        status: 'activated',
        source: {
          runId: 'runner-test-promote-source',
          manifestPath: getExperimentRunManifestPath('runner-test-promote-source'),
        },
        requestedAt: source.manifest.promotion.requestedAt,
        approvedAt: activatedState.approvedAt,
        promotedAt: activatedState.promotedAt,
        activatedAt: activatedState.activatedAt,
        activation: result.activation,
      });
      expect(readRunManifest(getExperimentRunManifestPath('runner-test-promote-source')).promotion).toEqual(
        activatedState,
      );
      expect(readLedgerEntries(TEST_LEDGER_PATH)).toHaveLength(1);

      const activationArtifact = JSON.parse(
        readFileSync(join(getExperimentRunDir('runner-test-promote-verify'), 'activation.json'), 'utf8'),
      ) as Record<string, unknown>;
      expect(activationArtifact.promotion).toEqual(activatedState);
      expect(activationArtifact.mutationReplayPayload).toEqual(source.manifest.mutationReplayPayload);
      expect(activationArtifact.activeStatePath).toBe(result.activeStatePath);
      expect(readFileSync('src/tools/finance/markov-distribution.ts', 'utf8'))
        .toContain(getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'momentumLookback'));
      expect(readFileSync('src/tools/finance/conformal.ts', 'utf8'))
        .toContain(getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow'));
      expect(readFileSync('src/tools/finance/regime-calibrator.ts', 'utf8'))
        .toContain(getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'minSamplesPerRegime'));
      const sharedEffectiveDefaults = resolveEffectiveRuntimeDefaults('shared');
      const goldEffectiveDefaults = resolveEffectiveRuntimeDefaults('gold');
      const btcEffectiveDefaults = resolveEffectiveRuntimeDefaults('btc');
      expectResolvedDefaultsToMatchMutation(sharedEffectiveDefaults, MULTI_ASSET_SHORTER_REACTIVE_WINDOW);
      expectResolvedDefaultsToMatchMutation(goldEffectiveDefaults, MULTI_ASSET_SHORTER_REACTIVE_WINDOW);
      expectResolvedDefaultsToMatchShipped(btcEffectiveDefaults);
      const activeState = JSON.parse(readFileSync(result.activeStatePath, 'utf8')) as Record<string, unknown>;
      expect(activeState.profileId).toBe('multi-asset-markov-short-horizon');
      expect(activeState.sourceRunId).toBe('runner-test-promote-source');
      expect(activeState.promotionRunId).toBe('runner-test-promote-verify');
      expect(activeState.promotion).toEqual(activatedState);
      expect(progress).toContain(`forecast-lab: promotion staging workspace ${promotionWorktreePath}`);
      expect(progress).toContain(`forecast-lab: live activation recorded at ${result.activeStatePath}`);
      expect(progress).toContain(
        `forecast-lab: activation artifacts written to ${join(getExperimentRunDir('runner-test-promote-verify'), 'activation.json')}`,
      );
    } finally {
      restoreLiveMutableFiles(liveFiles);
      restoreRuntimeDefaults(runtimeDefaults);
    }
  });

  it('keeps BTC live activation and active state intact when GOLD is promoted beside it', async () => {
    const liveFiles = snapshotLiveMutableFiles();
    const runtimeDefaults = snapshotRuntimeDefaults();

    try {
      await runForecastLab({
        profileId: 'btc-markov-ultra-short-horizon',
        mutationMode: 'structured',
        mutator: 'markov-shorter-reactive-window',
        runId: 'runner-test-btc-promote-source',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptBtcMetricsRunner(),
      });
      const btcPromotion = await promoteForecastLab({
        profileId: 'btc-markov-ultra-short-horizon',
        runId: 'runner-test-btc-promote-verify',
        sourceRunId: 'runner-test-btc-promote-source',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptBtcMetricsRunner(),
      });

      await runForecastLab({
        profileId: 'gold-markov-short-horizon',
        mutationMode: 'structured',
        mutator: 'gold-markov-shorter-reactive-window',
        runId: 'runner-test-gold-promote-source-a',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptGoldMetricsRunner(),
      });
      const goldPromotion = await promoteForecastLab({
        profileId: 'gold-markov-short-horizon',
        runId: 'runner-test-gold-promote-verify-a',
        sourceRunId: 'runner-test-gold-promote-source-a',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptGoldMetricsRunner(),
      });

      expect(goldPromotion.activeStatePath).not.toBe(btcPromotion.activeStatePath);
      expect(readFileSync('src/tools/finance/markov-distribution.ts', 'utf8'))
        .toContain(getStructuredMutationLine(BTC_SHORTER_REACTIVE_WINDOW, 'momentumLookback'));
      expect(readFileSync('src/tools/finance/markov-distribution.ts', 'utf8'))
        .not.toContain(getStructuredMutationLine(GOLD_SHORTER_REACTIVE_WINDOW, 'momentumLookback'));
      expect(readFileSync('src/tools/finance/conformal.ts', 'utf8'))
        .toContain(getStructuredMutationLine(BTC_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow'));
      expect(readFileSync('src/tools/finance/conformal.ts', 'utf8'))
        .not.toContain(getStructuredMutationLine(GOLD_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow'));
      expect(readFileSync('src/tools/finance/regime-calibrator.ts', 'utf8'))
        .toContain(getStructuredMutationLine(BTC_SHORTER_REACTIVE_WINDOW, 'minSamplesPerRegime'));
      expect(readFileSync('src/tools/finance/regime-calibrator.ts', 'utf8'))
        .not.toContain(getStructuredMutationLine(GOLD_SHORTER_REACTIVE_WINDOW, 'minSamplesPerRegime'));

      expectResolvedDefaultsToMatchMutation(resolveEffectiveRuntimeDefaults('btc'), BTC_SHORTER_REACTIVE_WINDOW);
      expectResolvedDefaultsToMatchMutation(resolveEffectiveRuntimeDefaults('gold'), GOLD_SHORTER_REACTIVE_WINDOW);
      expectResolvedDefaultsToMatchShipped(resolveEffectiveRuntimeDefaults('shared'));

      expect(JSON.parse(readFileSync(btcPromotion.activeStatePath, 'utf8'))).toMatchObject({
        profileId: 'btc-markov-ultra-short-horizon',
        sourceRunId: btcPromotion.sourceRunId,
        promotionRunId: btcPromotion.runId,
      });
      expect(JSON.parse(readFileSync(goldPromotion.activeStatePath, 'utf8'))).toMatchObject({
        profileId: 'gold-markov-short-horizon',
        sourceRunId: goldPromotion.sourceRunId,
        promotionRunId: goldPromotion.runId,
        mutatedFiles: [],
      });
    } finally {
      restoreLiveMutableFiles(liveFiles);
      restoreRuntimeDefaults(runtimeDefaults);
    }
  }, 20_000);

  it('repairs missing promotion metadata on a legacy kept source manifest before promotion', async () => {
    const liveFiles = snapshotLiveMutableFiles();
    const runtimeDefaults = snapshotRuntimeDefaults();

    try {
      await runForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        mutationMode: 'structured',
        mutator: 'markov-shorter-reactive-window',
        runId: 'runner-test-promote-legacy-source',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      });

      const sourceManifestPath = getExperimentRunManifestPath('runner-test-promote-legacy-source');
      const sourceManifest = readRunManifest(sourceManifestPath);
      const { promotion: _omittedPromotion, ...legacyManifest } = sourceManifest;
      writeRunManifest(sourceManifestPath, {
        ...legacyManifest,
      });

      const result = await promoteForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        runId: 'runner-test-promote-legacy-verify',
        sourceRunId: 'runner-test-promote-legacy-source',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      });

      expect(result.sourceRunId).toBe('runner-test-promote-legacy-source');
      const repairedManifest = readRunManifest(sourceManifestPath);
      expect(repairedManifest.promotion?.status).toBe('activated');
      expect(repairedManifest.promotion?.source).toEqual({
        runId: 'runner-test-promote-legacy-source',
        manifestPath: sourceManifestPath,
      });
    } finally {
      restoreLiveMutableFiles(liveFiles);
      restoreRuntimeDefaults(runtimeDefaults);
    }
  });

  it('fails closed when live source drift blocks activation after verification passes', async () => {
    const liveFiles = snapshotLiveMutableFiles();
    const runtimeDefaults = snapshotRuntimeDefaults();

    try {
      const source = await runForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        mutationMode: 'structured',
        mutator: 'markov-shorter-reactive-window',
        runId: 'runner-test-promote-source',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      });

      const liveMarkovPath = 'src/tools/finance/markov-distribution.ts';
      const originalMarkov = readFileSync(liveMarkovPath, 'utf8');
      writeFileSync(
        liveMarkovPath,
        originalMarkov.replace(
          `  momentumLookback: ${runtimeDefaults.markov.momentumLookback},`,
          '  momentumLookback: 999,',
        ),
        'utf8',
      );

      await expect(promoteForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        runId: 'runner-test-promote-verify',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      })).rejects.toThrow(/expected 1 match\(es\) for momentumLookback/);

      expect(readRunManifest(getExperimentRunManifestPath('runner-test-promote-source')).promotion).toEqual(
        source.manifest.promotion,
      );
      expect(existsSync(join('.cramer-short', 'experiments', 'active-promotions', 'multi-asset-markov-short-horizon.json'))).toBe(
        false,
      );
      expectResolvedDefaultsToMatchShipped(resolveEffectiveRuntimeDefaults('shared'));
      expectResolvedDefaultsToMatchShipped(resolveEffectiveRuntimeDefaults('gold'));
      expectResolvedDefaultsToMatchShipped(resolveEffectiveRuntimeDefaults('btc'));
    } finally {
      restoreLiveMutableFiles(liveFiles);
      restoreRuntimeDefaults(runtimeDefaults);
    }
  });

  it('resets the live profile back to shipped defaults and removes the active baseline record', async () => {
    const liveFiles = snapshotLiveMutableFiles();
    const runtimeDefaults = snapshotRuntimeDefaults();

    try {
      await runForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        mutationMode: 'structured',
        mutator: 'markov-shorter-reactive-window',
        runId: 'runner-test-reset-source-a',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      });
      await promoteForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        runId: 'runner-test-reset-promote-a',
        sourceRunId: 'runner-test-reset-source-a',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      });

      const result = await resetForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        mode: 'defaults',
        runId: 'runner-test-reset-defaults',
      });

      expect(result.mode).toBe('defaults');
      expect(result.activeStatePath).toBeUndefined();
      expect(readFileSync('src/tools/finance/markov-distribution.ts', 'utf8'))
        .toContain(`  momentumLookback: ${runtimeDefaults.markov.momentumLookback},`);
      expect(readFileSync('src/tools/finance/conformal.ts', 'utf8'))
        .toContain(`  scoreAggregationCalibrationWindow: ${runtimeDefaults.conformal.scoreAggregationCalibrationWindow},`);
      expect(readFileSync('src/tools/finance/regime-calibrator.ts', 'utf8'))
        .toContain(`  minSamplesPerRegime: ${runtimeDefaults.regime.minSamplesPerRegime},`);
      expectResolvedDefaultsToMatchShipped(resolveEffectiveRuntimeDefaults('shared'));
      expectResolvedDefaultsToMatchShipped(resolveEffectiveRuntimeDefaults('gold'));
      expectResolvedDefaultsToMatchShipped(resolveEffectiveRuntimeDefaults('btc'));
      expect(existsSync(join('.cramer-short', 'experiments', 'active-promotions', 'multi-asset-markov-short-horizon.json'))).toBe(false);
      const resetArtifact = JSON.parse(readFileSync(result.resetArtifactPath, 'utf8')) as Record<string, unknown>;
      expect(resetArtifact.mode).toBe('defaults');
      expect(resetArtifact.profileId).toBe('multi-asset-markov-short-horizon');
    } finally {
      restoreLiveMutableFiles(liveFiles);
      restoreRuntimeDefaults(runtimeDefaults);
    }
  });

  it('resets GOLD to shipped defaults without disturbing BTC live activation or active state', async () => {
    const liveFiles = snapshotLiveMutableFiles();
    const runtimeDefaults = snapshotRuntimeDefaults();

    try {
      await runForecastLab({
        profileId: 'btc-markov-ultra-short-horizon',
        mutationMode: 'structured',
        mutator: 'markov-shorter-reactive-window',
        runId: 'runner-test-btc-promote-source',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptBtcMetricsRunner(),
      });
      const btcPromotion = await promoteForecastLab({
        profileId: 'btc-markov-ultra-short-horizon',
        runId: 'runner-test-btc-promote-verify',
        sourceRunId: 'runner-test-btc-promote-source',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptBtcMetricsRunner(),
      });

      await runForecastLab({
        profileId: 'gold-markov-short-horizon',
        mutationMode: 'structured',
        mutator: 'gold-markov-shorter-reactive-window',
        runId: 'runner-test-gold-promote-source-a',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptGoldMetricsRunner(),
      });
      const goldPromotion = await promoteForecastLab({
        profileId: 'gold-markov-short-horizon',
        runId: 'runner-test-gold-promote-verify-a',
        sourceRunId: 'runner-test-gold-promote-source-a',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptGoldMetricsRunner(),
      });

      const result = await resetForecastLab({
        profileId: 'gold-markov-short-horizon',
        mode: 'defaults',
        runId: 'runner-test-gold-reset-defaults',
      });

      expect(result.mode).toBe('defaults');
      expect(result.activeStatePath).toBeUndefined();
      expect(readFileSync('src/tools/finance/markov-distribution.ts', 'utf8'))
        .toContain(getStructuredMutationLine(BTC_SHORTER_REACTIVE_WINDOW, 'momentumLookback'));
      expect(readFileSync('src/tools/finance/markov-distribution.ts', 'utf8'))
        .not.toContain(getStructuredMutationLine(GOLD_SHORTER_REACTIVE_WINDOW, 'momentumLookback'));
      expect(readFileSync('src/tools/finance/conformal.ts', 'utf8'))
        .toContain(getStructuredMutationLine(BTC_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow'));
      expect(readFileSync('src/tools/finance/conformal.ts', 'utf8'))
        .not.toContain(getStructuredMutationLine(GOLD_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow'));
      expect(readFileSync('src/tools/finance/regime-calibrator.ts', 'utf8'))
        .toContain(getStructuredMutationLine(BTC_SHORTER_REACTIVE_WINDOW, 'minSamplesPerRegime'));
      expect(readFileSync('src/tools/finance/regime-calibrator.ts', 'utf8'))
        .not.toContain(getStructuredMutationLine(GOLD_SHORTER_REACTIVE_WINDOW, 'minSamplesPerRegime'));

      expectResolvedDefaultsToMatchMutation(resolveEffectiveRuntimeDefaults('btc'), BTC_SHORTER_REACTIVE_WINDOW);
      expectResolvedDefaultsToMatchShipped(resolveEffectiveRuntimeDefaults('gold'));
      expectResolvedDefaultsToMatchShipped(resolveEffectiveRuntimeDefaults('shared'));
      expect(JSON.parse(readFileSync(btcPromotion.activeStatePath, 'utf8'))).toMatchObject({
        profileId: 'btc-markov-ultra-short-horizon',
        sourceRunId: btcPromotion.sourceRunId,
        promotionRunId: btcPromotion.runId,
      });
      const goldResetArtifact = JSON.parse(readFileSync(result.resetArtifactPath, 'utf8')) as Record<string, unknown>;
      expect(goldResetArtifact.mutatedFiles).toEqual([]);
      expect(existsSync(goldPromotion.activeStatePath)).toBe(false);
    } finally {
      restoreLiveMutableFiles(liveFiles);
      restoreRuntimeDefaults(runtimeDefaults);
    }
  }, 20_000);

  it('resets the live profile back to the previously activated baseline when requested', async () => {
    const liveFiles = snapshotLiveMutableFiles();
    const runtimeDefaults = snapshotRuntimeDefaults();

    try {
      await runForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        mutationMode: 'structured',
        mutator: 'markov-shorter-reactive-window',
        runId: 'runner-test-reset-source-a',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      });
      const firstPromotion = await promoteForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        runId: 'runner-test-reset-promote-a',
        sourceRunId: 'runner-test-reset-source-a',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      });

      await runForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        mutationMode: 'structured',
        mutator: 'markov-faster-decay-reaction',
        runId: 'runner-test-reset-source-b',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      });
      await promoteForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        runId: 'runner-test-reset-promote-b',
        sourceRunId: 'runner-test-reset-source-b',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      });

      const result = await resetForecastLab({
        profileId: 'multi-asset-markov-short-horizon',
        mode: 'last-known-good',
        runId: 'runner-test-reset-last-known-good',
      });

      expect(result.mode).toBe('last-known-good');
      expect(result.activeStatePath).toBe(join('.cramer-short', 'experiments', 'active-promotions', 'multi-asset-markov-short-horizon.json'));
      expect(readFileSync('src/tools/finance/markov-distribution.ts', 'utf8'))
        .toContain(getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'momentumLookback'));
      expect(readFileSync('src/tools/finance/markov-distribution.ts', 'utf8'))
        .toContain(`  transitionDecay: ${SHIPPED_RUNTIME_DEFAULTS.markov.transitionDecay},`);
      expect(readFileSync('src/tools/finance/conformal.ts', 'utf8'))
        .toContain(`  adaptiveBreakLearningRateMultiplier: ${SHIPPED_RUNTIME_DEFAULTS.conformal.adaptiveBreakLearningRateMultiplier},`);
      expect(readFileSync('src/tools/finance/regime-calibrator.ts', 'utf8'))
        .toContain(getStructuredMutationLine(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'minSamplesPerRegime'));
      const sharedEffectiveDefaults = resolveEffectiveRuntimeDefaults('shared');
      const goldEffectiveDefaults = resolveEffectiveRuntimeDefaults('gold');
      const btcEffectiveDefaults = resolveEffectiveRuntimeDefaults('btc');
      expect(sharedEffectiveDefaults.markov.momentumLookback)
        .toBe(getStructuredMutationNumericAfterValue(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'momentumLookback'));
      expect(sharedEffectiveDefaults.markov.transitionDecay).toBe(0.97);
      expect(sharedEffectiveDefaults.conformal.adaptiveBreakLearningRateMultiplier).toBe(1.5);
      expect(sharedEffectiveDefaults.regime.minSamplesPerRegime)
        .toBe(getStructuredMutationNumericAfterValue(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'minSamplesPerRegime'));
      expect(goldEffectiveDefaults.markov.momentumLookback)
        .toBe(getStructuredMutationNumericAfterValue(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'momentumLookback'));
      expect(goldEffectiveDefaults.markov.transitionDecay).toBe(0.97);
      expect(goldEffectiveDefaults.conformal.adaptiveBreakLearningRateMultiplier).toBe(1.5);
      expect(goldEffectiveDefaults.regime.minSamplesPerRegime)
        .toBe(getStructuredMutationNumericAfterValue(MULTI_ASSET_SHORTER_REACTIVE_WINDOW, 'minSamplesPerRegime'));
      expectResolvedDefaultsToMatchShipped(btcEffectiveDefaults);
      const activeState = JSON.parse(readFileSync(result.activeStatePath!, 'utf8')) as Record<string, unknown>;
      expect(activeState.sourceRunId).toBe(firstPromotion.sourceRunId);
      expect(activeState.promotionRunId).toBe(firstPromotion.runId);
      const resetArtifact = JSON.parse(readFileSync(result.resetArtifactPath, 'utf8')) as Record<string, unknown>;
      expect(resetArtifact.mode).toBe('last-known-good');
      expect(resetArtifact.restoredActive).toMatchObject({
        sourceRunId: firstPromotion.sourceRunId,
        promotionRunId: firstPromotion.runId,
      });
    } finally {
      restoreLiveMutableFiles(liveFiles);
      restoreRuntimeDefaults(runtimeDefaults);
    }
  }, 15_000);

  it('resets GOLD to the previous GOLD activation without disturbing BTC live activation or active state', async () => {
    const liveFiles = snapshotLiveMutableFiles();
    const runtimeDefaults = snapshotRuntimeDefaults();

    try {
      await runForecastLab({
        profileId: 'btc-markov-ultra-short-horizon',
        mutationMode: 'structured',
        mutator: 'markov-shorter-reactive-window',
        runId: 'runner-test-btc-promote-source',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptBtcMetricsRunner(),
      });
      const btcPromotion = await promoteForecastLab({
        profileId: 'btc-markov-ultra-short-horizon',
        runId: 'runner-test-btc-promote-verify',
        sourceRunId: 'runner-test-btc-promote-source',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptBtcMetricsRunner(),
      });

      await runForecastLab({
        profileId: 'gold-markov-short-horizon',
        mutationMode: 'structured',
        mutator: 'gold-markov-shorter-reactive-window',
        runId: 'runner-test-gold-promote-source-a',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptGoldMetricsRunner(),
      });
      const firstGoldPromotion = await promoteForecastLab({
        profileId: 'gold-markov-short-horizon',
        runId: 'runner-test-gold-promote-verify-a',
        sourceRunId: 'runner-test-gold-promote-source-a',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptGoldMetricsRunner(),
      });

      await runForecastLab({
        profileId: 'gold-markov-short-horizon',
        mutationMode: 'structured',
        mutator: 'gold-markov-faster-decay-reaction',
        runId: 'runner-test-gold-promote-source-b',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptGoldMetricsRunner(),
      });
      await promoteForecastLab({
        profileId: 'gold-markov-short-horizon',
        runId: 'runner-test-gold-promote-verify-b',
        sourceRunId: 'runner-test-gold-promote-source-b',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: keptGoldMetricsRunner(),
      });

      const result = await resetForecastLab({
        profileId: 'gold-markov-short-horizon',
        mode: 'last-known-good',
        runId: 'runner-test-gold-reset-last-known-good',
      });

      expect(result.mode).toBe('last-known-good');
      expect(result.activeStatePath).toBe(firstGoldPromotion.activeStatePath);
      expect(readFileSync('src/tools/finance/markov-distribution.ts', 'utf8'))
        .toContain(getStructuredMutationLine(BTC_SHORTER_REACTIVE_WINDOW, 'momentumLookback'));
      expect(readFileSync('src/tools/finance/markov-distribution.ts', 'utf8'))
        .not.toContain(getStructuredMutationLine(GOLD_SHORTER_REACTIVE_WINDOW, 'momentumLookback'));
      expect(readFileSync('src/tools/finance/markov-distribution.ts', 'utf8'))
        .not.toContain(getStructuredMutationLine(GOLD_FASTER_DECAY_REACTION, 'transitionDecay'));
      expect(readFileSync('src/tools/finance/conformal.ts', 'utf8'))
        .toContain(getStructuredMutationLine(BTC_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow'));
      expect(readFileSync('src/tools/finance/conformal.ts', 'utf8'))
        .not.toContain(getStructuredMutationLine(GOLD_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow'));
      expect(readFileSync('src/tools/finance/conformal.ts', 'utf8'))
        .not.toContain(getStructuredMutationLine(GOLD_FASTER_DECAY_REACTION, 'adaptiveBreakLearningRateMultiplier'));
      expect(readFileSync('src/tools/finance/regime-calibrator.ts', 'utf8'))
        .toContain(getStructuredMutationLine(BTC_SHORTER_REACTIVE_WINDOW, 'minSamplesPerRegime'));
      expect(readFileSync('src/tools/finance/regime-calibrator.ts', 'utf8'))
        .not.toContain(getStructuredMutationLine(GOLD_SHORTER_REACTIVE_WINDOW, 'minSamplesPerRegime'));

      const goldEffectiveDefaults = resolveEffectiveRuntimeDefaults('gold');
      expectResolvedDefaultsToMatchMutation(goldEffectiveDefaults, GOLD_SHORTER_REACTIVE_WINDOW);
      expect(goldEffectiveDefaults.markov.transitionDecay).toBe(SHIPPED_RUNTIME_DEFAULTS.markov.transitionDecay);
      expect(goldEffectiveDefaults.conformal.adaptiveBreakLearningRateMultiplier)
        .toBe(SHIPPED_RUNTIME_DEFAULTS.conformal.adaptiveBreakLearningRateMultiplier);
      expectResolvedDefaultsToMatchMutation(resolveEffectiveRuntimeDefaults('btc'), BTC_SHORTER_REACTIVE_WINDOW);
      expectResolvedDefaultsToMatchShipped(resolveEffectiveRuntimeDefaults('shared'));
      expect(JSON.parse(readFileSync(btcPromotion.activeStatePath, 'utf8'))).toMatchObject({
        profileId: 'btc-markov-ultra-short-horizon',
        sourceRunId: btcPromotion.sourceRunId,
        promotionRunId: btcPromotion.runId,
      });
      expect(JSON.parse(readFileSync(result.activeStatePath!, 'utf8'))).toMatchObject({
        profileId: 'gold-markov-short-horizon',
        sourceRunId: firstGoldPromotion.sourceRunId,
        promotionRunId: firstGoldPromotion.runId,
        mutatedFiles: [],
      });
      const goldResetArtifact = JSON.parse(readFileSync(result.resetArtifactPath, 'utf8')) as Record<string, unknown>;
      expect(goldResetArtifact.mutatedFiles).toEqual([]);
    } finally {
      restoreLiveMutableFiles(liveFiles);
      restoreRuntimeDefaults(runtimeDefaults);
    }
  }, 20_000);

  it('fails closed when another promotion attempt is already verifying the same source run', async () => {
    await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      mutator: 'markov-shorter-reactive-window',
      runId: 'runner-test-promote-concurrent-source',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    let releaseBaselineGate: (() => void) | undefined;
    let baselineGateStarted!: () => void;
    const baselineGateStartedPromise = new Promise<void>((resolve) => {
      baselineGateStarted = resolve;
    });

    const firstPromotion = promoteForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      runId: 'runner-test-promote-concurrent-verify-a',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: async (command, context) => {
        if (context.phase === 'baseline') {
          baselineGateStarted();
          await new Promise<void>((resolve) => {
            releaseBaselineGate = resolve;
          });
        }

        return {
          id: command.id,
          command: command.command,
          exitCode: 0,
          stdout: `${context.phase} ok`,
          stderr: '',
          durationMs: 1,
          timedOut: false,
        };
      },
    });

    await baselineGateStartedPromise;

    await expect(promoteForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      runId: 'runner-test-promote-concurrent-verify-b',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    })).rejects.toThrow(/already being promoted by another process/);

    expect(existsSync(getExperimentRunDir('runner-test-promote-concurrent-verify-b'))).toBe(false);

    releaseBaselineGate?.();
    const result = await firstPromotion;
    expect(result.sourceManifest.promotion?.status).toBe('activated');
  });

  it('re-validates the source manifest after verification and fails when source metadata changes mid-promotion', async () => {
    await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      mutator: 'markov-shorter-reactive-window',
      runId: 'runner-test-promote-stale-source',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    const sourceManifestPath = getExperimentRunManifestPath('runner-test-promote-stale-source');
    let mutatedSourceManifest = false;

    await expect(promoteForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      runId: 'runner-test-promote-stale-verify',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: async (command, context) => {
        if (context.phase === 'candidate' && !mutatedSourceManifest) {
          const manifest = readRunManifest(sourceManifestPath);
          if (!manifest.effectiveMutationContract || manifest.effectiveMutationContract.mode !== 'structured') {
            throw new Error('expected structured effective mutation contract');
          }

          writeRunManifest(sourceManifestPath, {
            ...manifest,
            effectiveMutationContract: {
              ...manifest.effectiveMutationContract,
              allowMultipleCandidateAttempts: true,
            },
          });
          mutatedSourceManifest = true;
        }

        return {
          id: command.id,
          command: command.command,
          exitCode: 0,
          stdout: `${context.phase} ok`,
          stderr: '',
          durationMs: 1,
          timedOut: false,
        };
      },
    })).rejects.toThrow(/changed while promotion verification was running/);

    expect(mutatedSourceManifest).toBe(true);
    expect(existsSync(getForecastLabPromotionWorktreePath('runner-test-promote-stale-verify'))).toBe(false);
    expect(readRunManifest(sourceManifestPath).effectiveMutationContract).toMatchObject({
      allowMultipleCandidateAttempts: true,
    });
  });

  it('fails closed when the kept structured source run is missing its replay payload', async () => {
    const source = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      mutator: 'markov-shorter-reactive-window',
      runId: 'runner-test-promote-missing-payload-source',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });
    const sourceManifestPath = getExperimentRunManifestPath('runner-test-promote-missing-payload-source');
    const { mutationReplayPayload: _removedReplayPayload, ...manifestWithoutReplayPayload } = source.manifest;

    writeRunManifest(sourceManifestPath, manifestWithoutReplayPayload);

    await expect(promoteForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      runId: 'runner-test-promote-missing-payload-verify',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    })).rejects.toThrow(/missing its structured mutation replay payload/);

    expect(existsSync(getForecastLabPromotionWorktreePath('runner-test-promote-missing-payload-verify'))).toBe(false);
    expect(readRunManifest(sourceManifestPath).promotion).toEqual(source.manifest.promotion);
  });

  it('fails closed and cleans up the promotion staging workspace when verification regresses', async () => {
    const source = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      mutator: 'markov-shorter-reactive-window',
      runId: 'runner-test-promote-regression-source',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });
    const promotionRunId = 'runner-test-promote-regression-verify';

    await expect(promoteForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      runId: promotionRunId,
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: async (command, context) => ({
        id: command.id,
        command: command.command,
        exitCode: context.phase === 'candidate' ? 1 : 0,
        stdout: '',
        stderr: context.phase === 'candidate' ? 'promotion failed' : '',
        durationMs: 1,
        timedOut: false,
      }),
    })).rejects.toThrow(/regressed while verifying kept run "runner-test-promote-regression-source"/);

    expect(existsSync(getForecastLabPromotionWorktreePath(promotionRunId))).toBe(false);
    expect(readRunManifest(getExperimentRunManifestPath('runner-test-promote-regression-source')).promotion).toEqual(
      source.manifest.promotion,
    );
    expect(
      JSON.parse(readFileSync(join(getExperimentRunDir(promotionRunId), 'decision.json'), 'utf8')),
    ).toMatchObject({
      sourceRunId: 'runner-test-promote-regression-source',
      decision: 'drop',
    });
  });

  it('preserves the candidate workspace when keepWorktree is enabled for a structured mutation run', async () => {
    const worktreePath = getForecastLabCandidateWorktreePath('runner-test-structured-keep-worktree');
    const progress: string[] = [];

    const result = await runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      mutationMode: 'structured',
      keepWorktree: true,
      mutator: 'markov-longer-stability-window',
      runId: 'runner-test-structured-keep-worktree',
      ledgerPath: TEST_LEDGER_PATH,
      progress: (message) => progress.push(message),
      commandRunner: passingRunner([]),
    });

    expect(result.decision.decision).toBe('keep');
    expect(result.manifest.candidateWorkspace?.rootDir).toBe(worktreePath);
    expect(existsSync(worktreePath)).toBe(true);
    expect(readFileSync(join(worktreePath, 'src/tools/finance/markov-distribution.ts'), 'utf8')).toContain(
      '  momentumLookback: 28,',
    );
    expect(progress).toContain(`forecast-lab: keeping candidate workspace ${worktreePath}`);

    const candidate = JSON.parse(
      readFileSync(join(getExperimentRunDir('runner-test-structured-keep-worktree'), 'candidate.json'), 'utf8'),
    ) as Record<string, unknown>;
    expect(candidate.selectedMutator).toEqual({
      id: 'markov-longer-stability-window',
      mutatorId: 'search-replace',
    });
  });

  it('explains the next valid actions when a structured mutation catalog is exhausted', async () => {
    const btcProfileId = 'btc-markov-ultra-short-horizon';

    appendStructuredMutationHistory({
      profileId: btcProfileId,
      runId: 'runner-test-btc-structured-parent-1',
      mutationId: 'markov-shorter-reactive-window',
      decision: 'keep',
      startedAt: '2026-05-01T00:00:00.000Z',
    });
    appendStructuredMutationHistory({
      profileId: btcProfileId,
      runId: 'runner-test-btc-structured-parent-2',
      mutationId: 'markov-faster-decay-reaction',
      decision: 'keep',
      startedAt: '2026-05-02T00:00:00.000Z',
      parentRunId: 'runner-test-btc-structured-parent-1',
    });
    appendStructuredMutationHistory({
      profileId: btcProfileId,
      runId: 'runner-test-btc-structured-parent-3',
      mutationId: 'markov-lower-confidence-trend-penalty',
      decision: 'keep',
      startedAt: '2026-05-03T00:00:00.000Z',
      parentRunId: 'runner-test-btc-structured-parent-2',
    });

    let error: unknown;
    try {
      await runForecastLab({
        profileId: btcProfileId,
        mutationMode: 'structured',
        runId: 'runner-test-btc-structured-exhausted',
        ledgerPath: TEST_LEDGER_PATH,
        commandRunner: passingRunner([]),
      });
    } catch (caught) {
      error = caught;
    }

    expect(error).toBeInstanceOf(ForecastLabRunnerError);
    expect((error as Error).message).toMatch(
      /Current kept lineage already applied: markov-shorter-reactive-window, markov-faster-decay-reaction, markov-lower-confidence-trend-penalty\./,
    );
    expect((error as Error).message).toMatch(
      /Remaining shipped mutators checked and found inapplicable: markov-longer-stability-window, markov-slower-decay-persistence, markov-higher-confidence-divergence-weighted, markov-calibrator-higher-sample-floor, markov-calibrator-lower-sample-floor\./,
    );
    expect((error as Error).message).toMatch(
      /Next actions: keep the current best candidate, add a new shipped structured mutator, or intentionally reset the forecast-lab lineage outside the CLI\./,
    );
  });

  it('rejects ambiguous real mutation runs with no explicit mutation mode', async () => {
    await expect(runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      runId: 'runner-test-skip-mutation',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    })).rejects.toThrow(/explicit mutationMode/i);
  });

  it('fails loudly when real mutation is requested for a profile without a shipped structured catalog', async () => {
    await expect(runForecastLab({
      profileId: 'btc-arbiter-replay',
      mutationMode: 'structured',
      runId: 'runner-test-skip-mutation',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    })).rejects.toThrow(/requires a structured profile with a shipped catalog/i);
  });

  it('drops --skip-mutation runs because no candidate code change exists', async () => {
    const result = await runForecastLab({
      profileId: 'btc-arbiter-replay',
      skipMutation: true,
      runId: 'runner-test-skip-mutation',
      ledgerPath: TEST_LEDGER_PATH,
      commandRunner: passingRunner([]),
    });

    expect(result.decision.decision).toBe('drop');
    expect(result.decision.reason).toContain('mutation skipped by --skip-mutation');
    expect(result.manifest.candidateWorkspace).toBeUndefined();
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).toMatchObject({
      runId: 'runner-test-skip-mutation',
      decision: 'drop',
      effectiveMutationContract: {
        mode: 'dry-run',
        mutableFiles: ['src/tools/finance/forecast-arbitrator.ts', 'src/tools/finance/forecast-hooks.ts'],
        allowMultipleCandidateAttempts: false,
      },
    });
    expect(readLedgerEntries(TEST_LEDGER_PATH).at(-1)).not.toHaveProperty('candidateWorkspace');
  });

  it('returns a failed result when the child emits a spawn error and clears timeout cleanup', async () => {
    const child = new FakeSpawnedChild();
    const output: string[] = [];
    const runner = createForecastLabCommandRunner((() => {
      queueMicrotask(() => {
        child.stderr.write('spawn stderr\n');
        child.emit('error', new Error('bad cwd'));
        child.emit('close');
      });
      return child as unknown as ReturnType<typeof import('node:child_process').spawn>;
    }) as typeof import('node:child_process').spawn);

    const result = await runner(
      {
        id: 'spawn-error-command',
        command: 'bun --version',
        timeoutMs: 10,
      },
      {
        phase: 'candidate',
        profile: {} as never,
        runId: 'runner-test-spawn-error',
        output: (chunk) => output.push(chunk),
      },
    );

    expect(result).toMatchObject({
      id: 'spawn-error-command',
      command: 'bun --version',
      exitCode: 1,
      stdout: '',
      timedOut: false,
    });
    expect(result.stderr).toBe('spawn stderr\nFailed to start command "spawn-error-command": bad cwd');
    expect(output).toEqual(['spawn stderr\n']);

    await new Promise((resolve) => setTimeout(resolve, 25));
    expect(child.killSignals).toEqual([]);
  });

  it('rejects unsafe profile commands before shell execution', async () => {
    await expect(defaultForecastLabCommandRunner(
      {
        id: 'unsafe-command',
        command: 'bun test src/tools/finance/polymarket-forecast.test.ts; git commit -m bad',
      },
      {
        phase: 'baseline',
        profile: {} as never,
        runId: 'runner-test-unsafe-command',
      },
    )).rejects.toThrow(/Unsafe forecast-lab command/);
  });

  it('refuses broad path writes outside .cramer-short/experiments', async () => {
    await expect(runForecastLab({
      profileId: 'multi-asset-markov-short-horizon',
      dryRun: true,
      runId: 'runner-test-outside-ledger',
      ledgerPath: join('.cramer-short', 'forecast-results.tsv'),
      commandRunner: passingRunner([]),
    })).rejects.toThrow(/outside \.cramer-short\/experiments/);
  });
});

describe('forecast-lab CLI module', () => {
  it('lists profiles without launching the TUI', async () => {
    const output: string[] = [];

    await runForecastLabCommand(['list'], { log: (message) => output.push(message) });

    expect(output.join('\n')).toContain('Forecast-lab profiles:');
    expect(output.join('\n')).toContain('multi-asset-markov-short-horizon');
  });

  it('prints useful usage for missing commands', async () => {
    const output: string[] = [];

    await runForecastLabCommand([], { log: (message) => output.push(message) });

    expect(output.join('\n')).not.toContain('  cramer-short lab run <profileId>\n');
    expect(output.join('\n')).toContain('cramer-short lab run <profileId> --dry-run');
    expect(output.join('\n')).toContain('--mutation structured');
  });

  it('prints useful usage for missing run profile ids', async () => {
    const output: string[] = [];
    const errors: string[] = [];
    let exitCode = 0;

    await runForecastLabCommand(['run'], {
      log: (message) => output.push(message),
      error: (message) => errors.push(message),
      exit: (code) => {
        exitCode = code;
      },
    });

    expect(exitCode).toBe(1);
    expect(errors.join('\n')).toContain('Missing forecast-lab profile id.');
    expect(output.join('\n')).toContain('cramer-short lab run <profileId> --dry-run');
  });

  it('rejects unknown run flags instead of defaulting into a real mutation run', async () => {
    const output: string[] = [];
    const errors: string[] = [];
    let exitCode = 0;
    let runLabCalls = 0;

    await runForecastLabCommand(['run', 'multi-asset-markov-short-horizon', '--dryrun'], {
      log: (message) => output.push(message),
      error: (message) => errors.push(message),
      exit: (code) => {
        exitCode = code;
      },
      runLab: async () => {
        runLabCalls += 1;
        throw new Error('runLab should not be called');
      },
    });

    expect(exitCode).toBe(1);
    expect(runLabCalls).toBe(0);
    expect(errors.join('\n')).toContain('Unknown forecast-lab flag: "--dryrun"');
    expect(output.join('\n')).not.toContain('Running forecast-lab profile');
  });

  it('rejects conflicting no-mutation flags', async () => {
    const errors: string[] = [];
    let exitCode = 0;
    let runLabCalls = 0;

    await runForecastLabCommand(['run', 'multi-asset-markov-short-horizon', '--dry-run', '--skip-mutation'], {
      error: (message) => errors.push(message),
      exit: (code) => {
        exitCode = code;
      },
      runLab: async () => {
        runLabCalls += 1;
        throw new Error('runLab should not be called');
      },
    });

    expect(exitCode).toBe(1);
    expect(runLabCalls).toBe(0);
    expect(errors.join('\n')).toContain(
      'Conflicting forecast-lab flags: --dry-run and --skip-mutation cannot be used together.',
    );
  });

  it('requires an explicit mutation mode before running a real mutation', async () => {
    const errors: string[] = [];
    let exitCode = 0;
    let runLabCalls = 0;

    await runForecastLabCommand(['run', 'multi-asset-markov-short-horizon'], {
      error: (message) => errors.push(message),
      exit: (code) => {
        exitCode = code;
      },
      runLab: async () => {
        runLabCalls += 1;
        throw new Error('runLab should not be called');
      },
    });

    expect(exitCode).toBe(1);
    expect(runLabCalls).toBe(0);
    expect(errors.join('\n')).toContain('Real forecast-lab mutation requires an explicit flag: --mutation structured.');
  });

  it('passes explicit structured mutation controls through to the runner', async () => {
    const output: string[] = [];
    const runLabCalls: Array<Record<string, unknown>> = [];

    await runForecastLabCommand([
      'run',
      'multi-asset-markov-short-horizon',
      '--mutation',
      'structured',
      '--mutator',
      'markov-longer-stability-window',
      '--keep-worktree',
    ], {
      log: (message) => output.push(message),
      runLab: async (options) => {
        runLabCalls.push(options as unknown as Record<string, unknown>);
        return {
          runId: 'runner-test-explicit-structured',
          manifest: {
            runId: 'runner-test-explicit-structured',
            startedAt: '2026-05-02T00:00:00.000Z',
            profileId: 'multi-asset-markov-short-horizon',
            targetSubsystem: 'markov-distribution',
            baselineCommit: '0123456789abcdef0123456789abcdef01234567',
            candidateBranch: 'topic/forecast-lab-runner-test-explicit-structured',
            allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
            artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-explicit-structured'),
          },
          baseline: { exitCode: 0 },
          candidate: { exitCode: 0 },
          decision: {
            decision: 'keep',
            reason: 'candidate passed',
            metrics: [],
          },
          ledgerEntry: {
            runId: 'runner-test-explicit-structured',
            startedAt: '2026-05-02T00:00:00.000Z',
            profileId: 'multi-asset-markov-short-horizon',
            targetSubsystem: 'markov-distribution',
            candidateBranch: 'topic/forecast-lab-runner-test-explicit-structured',
            allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
            baselineSummary: { exitCode: 0 },
            candidateSummary: { exitCode: 0 },
            decision: 'keep',
            reason: 'candidate passed',
            artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-explicit-structured'),
          },
        };
      },
    });

    expect(runLabCalls).toEqual([
      expect.objectContaining({
        profileId: 'multi-asset-markov-short-horizon',
        dryRun: false,
        skipMutation: false,
        mutationMode: 'structured',
        keepWorktree: true,
        mutator: 'markov-longer-stability-window',
        progress: expect.any(Function),
        output: expect.any(Function),
      }),
    ]);
    expect(output.join('\n')).toContain('Running forecast-lab profile "multi-asset-markov-short-horizon" with structured mutation...');
  });

  it('rejects mutator overrides without an explicit structured mutation mode', async () => {
    const errors: string[] = [];
    let exitCode = 0;
    let runLabCalls = 0;

    await runForecastLabCommand(['run', 'multi-asset-markov-short-horizon', '--mutator', 'markov-longer-stability-window'], {
      error: (message) => errors.push(message),
      exit: (code) => {
        exitCode = code;
      },
      runLab: async () => {
        runLabCalls += 1;
        throw new Error('runLab should not be called');
      },
    });

    expect(exitCode).toBe(1);
    expect(runLabCalls).toBe(0);
    expect(errors.join('\n')).toContain('--keep-worktree and --mutator require --mutation structured');
  });

  it('prints evolution and parameter summaries after a successful run', async () => {
    const output: string[] = [];
    const writes: string[] = [];

    await runForecastLabCommand(['run', 'btc-markov-ultra-short-horizon', '--mutation', 'structured'], {
      log: (message) => output.push(message),
      write: (chunk) => writes.push(chunk),
      runLab: async () => ({
        runId: 'runner-test-structured-summary',
        manifest: {
          runId: 'runner-test-structured-summary',
          startedAt: '2026-05-02T00:00:00.000Z',
          profileId: 'btc-markov-ultra-short-horizon',
          targetSubsystem: 'markov-distribution',
          baselineCommit: '0123456789abcdef0123456789abcdef01234567',
          candidateBranch: 'topic/forecast-lab-runner-test-structured-summary',
          allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
          mutationMode: 'structured',
          mutationId: 'markov-shorter-reactive-window',
          mutationSummary: BTC_SHORTER_REACTIVE_WINDOW.specSummary.summary,
          mutationReplayPayload: snapshotForecastLabMarkovParameterMutation(BTC_SHORTER_REACTIVE_WINDOW),
          candidateWorkspace: {
            kind: 'candidate-worktree',
            rootDir: resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured-summary'),
            branch: 'topic/forecast-lab-runner-test-structured-summary',
          },
          artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-structured-summary'),
        },
        baseline: { exitCode: 0 },
        candidate: { exitCode: 0 },
        decision: {
          decision: 'keep',
          reason: 'candidate BTC ultra-short-horizon test command must pass',
          metrics: [
            {
              name: 'walkForwardBtcUltraShortHorizonTestExitCode',
              baseline: 0,
              candidate: 0,
              delta: 0,
            },
          ],
        },
        ledgerEntry: {
          runId: 'runner-test-structured-summary',
          startedAt: '2026-05-02T00:00:00.000Z',
          profileId: 'btc-markov-ultra-short-horizon',
          targetSubsystem: 'markov-distribution',
          candidateBranch: 'topic/forecast-lab-runner-test-structured-summary',
          allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
          mutationMode: 'structured',
          mutationId: 'markov-shorter-reactive-window',
          mutationSummary: BTC_SHORTER_REACTIVE_WINDOW.specSummary.summary,
          candidateWorkspace: {
            kind: 'candidate-worktree',
            rootDir: resolve('.cramer-short', 'experiments', 'worktrees', 'runner-test-structured-summary'),
            branch: 'topic/forecast-lab-runner-test-structured-summary',
          },
          baselineSummary: { exitCode: 0 },
          candidateSummary: { exitCode: 0 },
          decision: 'keep',
          reason: 'candidate BTC ultra-short-horizon test command must pass',
          artifactsPath: join('.cramer-short', 'experiments', 'runs', 'runner-test-structured-summary'),
        },
      }),
    });

    const text = output.join('\n');
    expect(text).toContain('Running forecast-lab profile "btc-markov-ultra-short-horizon" with structured mutation...');
    expect(text).toContain('Evolution summary:');
    expect(text).toContain('baseline exitCode: 0');
    expect(text).toContain('candidate exitCode: 0');
    expect(text).toContain(`Mutation summary: ${BTC_SHORTER_REACTIVE_WINDOW.specSummary.summary}`);
    expect(text).toContain('mutation id: markov-shorter-reactive-window');
    expect(text).toContain('Previous parameters (baseline defaults):');
    expect(text).toContain('New parameters (candidate mutation):');
    expect(text).toContain(
      `markov-distribution.ts: momentumLookback = ${getStructuredMutationEdit(BTC_SHORTER_REACTIVE_WINDOW, 'momentumLookback').beforeValue}`,
    );
    expect(text).toContain(
      `markov-distribution.ts: momentumLookback = ${getStructuredMutationEdit(BTC_SHORTER_REACTIVE_WINDOW, 'momentumLookback').afterValue}`,
    );
    expect(text).toContain(
      `conformal.ts: scoreAggregationCalibrationWindow = ${getStructuredMutationEdit(BTC_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow').beforeValue}`,
    );
    expect(text).toContain(
      `conformal.ts: scoreAggregationCalibrationWindow = ${getStructuredMutationEdit(BTC_SHORTER_REACTIVE_WINDOW, 'scoreAggregationCalibrationWindow').afterValue}`,
    );
    expect(text).toContain('forecast-lab keep: candidate BTC ultra-short-horizon test command must pass');
    expect(writes).toEqual([]);
  });

  it('diagnostic-only mode prevents promotion state, ledger entries, and routing stats', async () => {
    const profile = getForecastLabProfile('btc-markov-ultra-short-horizon');
    const tempRoot = join('.cramer-short', 'experiments', 'diagnostic-only-test');
    const ledgerPath = join(tempRoot, 'ledger.jsonl');
    const routingStatsPath = join(tempRoot, 'routing-stats.json');

    // Clean up any existing test artifacts
    rmSync(tempRoot, { recursive: true, force: true });

    const commandRunner = async (command: any, context: any) => ({
      id: command.id,
      command: command.command,
      exitCode: 0,
      stdout: context.phase === 'baseline' ? 'baseline OK' : 'candidate OK',
      stderr: '',
      durationMs: 1,
      timedOut: false,
    });

    const result = await runForecastLab({
      profileId: profile.id,
      mutationMode: 'structured',
      // Let auto-selection pick an applicable mutator
      forceNoParent: true,
      diagnosticOnly: true,
      ledgerPath,
      routingContext: {
        originatingQuery: 'diagnostic test',
        selectedProfileId: profile.id,
        routerReason: 'test',
        invocationSource: 'manual-request',
      },
      routingStatsPath,
      commandRunner: commandRunner as any,
    });

    // Verify no promotion state was set
    expect(result.manifest.promotion).toBeUndefined();

    // Verify no ledger file was created
    expect(existsSync(ledgerPath)).toBe(false);

    // Verify no routing stats were created
    expect(existsSync(routingStatsPath)).toBe(false);

    // Clean up
    rmSync(tempRoot, { recursive: true, force: true });
  });

  it('normal mode (diagnosticOnly=false) creates promotion state, ledger entries, and routing stats', async () => {
    const profile = getForecastLabProfile('btc-markov-ultra-short-horizon');
    const tempRoot = join('.cramer-short', 'experiments', 'normal-mode-test');
    const ledgerPath = join(tempRoot, 'ledger.jsonl');
    const routingStatsPath = join(tempRoot, 'routing-stats.json');

    // Clean up any existing test artifacts
    rmSync(tempRoot, { recursive: true, force: true });

    const commandRunner = async (command: any, context: any) => ({
      id: command.id,
      command: command.command,
      exitCode: 0,
      stdout: context.phase === 'baseline' ? 'baseline OK' : 'candidate OK',
      stderr: '',
      durationMs: 1,
      timedOut: false,
    });

    const result = await runForecastLab({
      profileId: profile.id,
      mutationMode: 'structured',
      // Let auto-selection pick an applicable mutator
      forceNoParent: true,
      diagnosticOnly: false,
      ledgerPath,
      routingContext: {
        originatingQuery: 'normal test',
        selectedProfileId: profile.id,
        routerReason: 'test',
        invocationSource: 'manual-request',
      },
      routingStatsPath,
      commandRunner: commandRunner as any,
    });

    // Verify promotion state was set for kept runs
    if (result.decision.decision === 'keep') {
      expect(result.manifest.promotion).toBeDefined();
      expect(result.manifest.promotion?.status).toBe('approval-required');
    }

    // Verify ledger file was created
    expect(existsSync(ledgerPath)).toBe(true);

    // Verify routing stats were created
    expect(existsSync(routingStatsPath)).toBe(true);

    // Clean up
    rmSync(tempRoot, { recursive: true, force: true });
  });
});
