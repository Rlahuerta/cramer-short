import { describe, expect, it } from 'bun:test';
import type { ForecastLabProfile, ForecastLabProfileId } from './profiles.js';
import { listForecastLabMutatorIds } from './mutation.js';
import {
  FORECAST_LAB_PROFILES,
  FORECAST_LAB_READ_ONLY_HARNESS_FILES,
  assertForecastLabProfileId,
  getForecastLabProfile,
  isCanonicalForecastLabProfileId,
  isForecastLabProfileId,
  listForecastLabProfiles,
  listForecastLabStructuredMutations,
  normalizeForecastLabProfileId,
} from './profiles.js';

const EXPECTED_PROFILE_IDS: readonly ForecastLabProfileId[] = [
  'multi-asset-markov-short-horizon',
  'btc-markov-ultra-short-horizon',
  'btc-arbiter-replay',
  'polymarket-selection-sanity',
];

const COMMAND_STATUS_METRIC_PATHS = {
  baselinePath: 'baseline.exitCode',
  candidatePath: 'candidate.exitCode',
} as const;

const EXPECTED_COMMAND_STATUS_PROFILES = {
  'multi-asset-markov-short-horizon': {
    commands: [
      {
        id: 'walk-forward-short-horizon',
        command: 'bun test src/tools/finance/backtest/walk-forward-short-horizon.test.ts --timeout 480000',
        env: { RUN_INTEGRATION: '1' },
        timeoutMs: 480_000,
      },
    ],
    metricName: 'walkForwardShortHorizonTestExitCode',
  },
  'btc-markov-ultra-short-horizon': {
    commands: [
      {
        id: 'walk-forward-btc-ultra-short-horizon',
        command: 'bun test src/tools/finance/backtest/walk-forward-btc-ultra-short-horizon.test.ts --timeout 360000',
        env: { RUN_INTEGRATION: '1' },
        timeoutMs: 360_000,
      },
    ],
    metricName: 'walkForwardBtcUltraShortHorizonTestExitCode',
  },
  'btc-arbiter-replay': {
    commands: [
      {
        id: 'arbiter-replay-runner',
        command: 'bun test src/tools/finance/backtest/arbiter-replay-runner.test.ts',
        timeoutMs: 120_000,
      },
    ],
    metricName: 'arbiterReplayRunnerTestExitCode',
  },
  'polymarket-selection-sanity': {
    commands: [
      {
        id: 'polymarket-forecast-selection',
        command: 'bun test src/tools/finance/polymarket-forecast.test.ts',
        timeoutMs: 120_000,
      },
    ],
    metricName: 'polymarketForecastTestExitCode',
  },
} as const satisfies Record<ForecastLabProfileId, {
  readonly commands: readonly {
    readonly id: string;
    readonly command: string;
    readonly env?: Readonly<Record<string, string>>;
    readonly timeoutMs?: number;
  }[];
  readonly metricName: string;
}>;

const EXPECTED_READ_ONLY_HARNESSES = {
  'multi-asset-markov-short-horizon': ['src/tools/finance/backtest/walk-forward.ts'],
  'btc-markov-ultra-short-horizon': ['src/tools/finance/backtest/walk-forward.ts'],
  'btc-arbiter-replay': ['src/tools/finance/backtest/arbiter-replay-runner.ts'],
  'polymarket-selection-sanity': [],
} as const satisfies Record<ForecastLabProfileId, readonly string[]>;

const EXPECTED_MUTATION_CONFIGS = {
  'multi-asset-markov-short-horizon': {
    mode: 'structured',
    mutableFiles: [
      'src/tools/finance/markov-distribution.ts',
      'src/tools/finance/conformal.ts',
      'src/tools/finance/regime-calibrator.ts',
    ],
    allowedMutatorIds: ['search-replace'],
    allowMultipleCandidateAttempts: false,
  },
  'btc-markov-ultra-short-horizon': {
    mode: 'structured',
    mutableFiles: [
      'src/tools/finance/markov-distribution.ts',
      'src/tools/finance/conformal.ts',
      'src/tools/finance/regime-calibrator.ts',
    ],
    allowedMutatorIds: ['search-replace'],
    allowMultipleCandidateAttempts: false,
  },
  'btc-arbiter-replay': {
    mode: 'dry-run',
    mutableFiles: [
      'src/tools/finance/forecast-arbitrator.ts',
      'src/tools/finance/forecast-hooks.ts',
    ],
    allowMultipleCandidateAttempts: false,
  },
  'polymarket-selection-sanity': {
    mode: 'dry-run',
    mutableFiles: [
      'src/tools/finance/polymarket-forecast.ts',
      'src/tools/finance/polymarket.ts',
    ],
    allowMultipleCandidateAttempts: false,
  },
} as const satisfies Record<ForecastLabProfileId, {
  readonly mode: 'structured' | 'dry-run';
  readonly mutableFiles: readonly string[];
  readonly allowedMutatorIds?: readonly string[];
  readonly allowMultipleCandidateAttempts: false;
}>;

function comparableCommands(profile: ForecastLabProfile) {
  return {
    baseline: profile.baselineCommands.map((command) => ({
      id: command.id,
      command: command.command,
      env: command.env ?? {},
      timeoutMs: command.timeoutMs ?? null,
    })),
    candidate: profile.candidateCommands.map((command) => ({
      id: command.id,
      command: command.command,
      env: command.env ?? {},
      timeoutMs: command.timeoutMs ?? null,
    })),
  };
}

describe('forecast-lab profiles', () => {
  it('lists typed profile definitions for the initial Phase 3 set', () => {
    const profiles: readonly ForecastLabProfile[] = listForecastLabProfiles();

    expect(FORECAST_LAB_PROFILES === profiles).toBe(true);
    expect([...profiles.map((profile) => profile.id)]).toEqual([...EXPECTED_PROFILE_IDS]);

    for (const profile of profiles) {
      expect(typeof profile.targetSubsystem).toBe('string');
      expect(profile.allowedGlobs.length).toBeGreaterThan(0);
      expect(profile.mutation.mutableFiles.length).toBeGreaterThan(0);
      if (profile.mutation.mode === 'structured') {
        expect(profile.mutation.allowedMutatorIds.length).toBeGreaterThan(0);
      } else {
        expect('allowedMutatorIds' in profile.mutation).toBe(false);
      }
      expect(Array.isArray(profile.readOnlyHarnessFiles)).toBe(true);
      expect(profile.baselineCommands.length).toBeGreaterThan(0);
      expect(profile.candidateCommands.length).toBeGreaterThan(0);
      expect(profile.minimumMetrics.length).toBeGreaterThan(0);
      expect(profile.keepDropRule.defaultDecision).toBe('drop');
      expect(profile.keepDropRule.keepWhen.all.length).toBeGreaterThan(0);
      expect(profile.keepDropRule.dropWhen.any.length).toBeGreaterThan(0);
    }
  });

  it('gets known profiles and rejects unknown profile ids', () => {
    expect(getForecastLabProfile('multi-asset-markov-short-horizon').targetSubsystem).toBe('markov-distribution');
    expect(getForecastLabProfile('btc-markov-ultra-short-horizon').targetSubsystem).toBe('markov-distribution');
    expect(isForecastLabProfileId('btc-arbiter-replay')).toBe(true);
    expect(isForecastLabProfileId('btc-markov-short-horizon')).toBe(true);
    expect(isCanonicalForecastLabProfileId('btc-markov-short-horizon')).toBe(false);
    expect(normalizeForecastLabProfileId('btc-markov-short-horizon')).toBe('multi-asset-markov-short-horizon');
    expect(getForecastLabProfile('btc-markov-short-horizon').id).toBe('multi-asset-markov-short-horizon');
    expect(isForecastLabProfileId('unknown-profile')).toBe(false);
    expect(() => getForecastLabProfile('unknown-profile')).toThrow(/Unknown forecast-lab profile id/);
    expect(() => assertForecastLabProfileId('unknown-profile')).toThrow(/Unknown forecast-lab profile id/);

    let id: string = 'polymarket-selection-sanity';
    assertForecastLabProfileId(id);
    const narrowed = normalizeForecastLabProfileId(id);
    expect(narrowed).toBe('polymarket-selection-sanity');
  });

  it('is immutable enough for callers to reuse safely', () => {
    const profiles = listForecastLabProfiles();
    const markov = getForecastLabProfile('multi-asset-markov-short-horizon');

    expect(Object.isFrozen(profiles)).toBe(true);
    expect(Object.isFrozen(markov)).toBe(true);
    expect(Object.isFrozen(markov.allowedGlobs)).toBe(true);
    expect(Object.isFrozen(markov.mutation)).toBe(true);
    expect(Object.isFrozen(markov.mutation.mutableFiles)).toBe(true);
    if (markov.mutation.mode === 'structured') {
      expect(Object.isFrozen(markov.mutation.allowedMutatorIds)).toBe(true);
    }
    expect(Object.isFrozen(markov.readOnlyHarnessFiles)).toBe(true);
    expect(Object.isFrozen(markov.baselineCommands)).toBe(true);
    expect(Object.isFrozen(markov.baselineCommands[0])).toBe(true);
    expect(Object.isFrozen(markov.baselineCommands[0]?.env)).toBe(true);
    expect(Object.isFrozen(markov.keepDropRule.keepWhen.all[0])).toBe(true);

    expect(() => {
      (markov.allowedGlobs as unknown as string[]).push('src/tools/finance/backtest/walk-forward.ts');
    }).toThrow();
    expect(markov.allowedGlobs).not.toContain('src/tools/finance/backtest/walk-forward.ts');

    expect(() => {
      (markov.readOnlyHarnessFiles as unknown as string[]).push('src/tools/finance/backtest/arbiter-replay-runner.ts');
    }).toThrow();
    expect(markov.readOnlyHarnessFiles).toEqual(['src/tools/finance/backtest/walk-forward.ts']);

    expect(() => {
      if (markov.mutation.mode !== 'structured') {
        throw new Error('expected structured mutation config');
      }
      (markov.mutation.allowedMutatorIds as unknown as string[]).push('insert-block');
    }).toThrow();
    if (markov.mutation.mode === 'structured') {
      expect(markov.mutation.allowedMutatorIds).toEqual(['search-replace']);
    }
  });

  it('includes only applicable fixed read-only harness references', () => {
    expect(FORECAST_LAB_READ_ONLY_HARNESS_FILES).toEqual([
      'src/tools/finance/backtest/walk-forward.ts',
      'src/tools/finance/backtest/arbiter-replay-runner.ts',
    ]);

    for (const profileId of EXPECTED_PROFILE_IDS) {
      const profile = getForecastLabProfile(profileId);

      expect(profile.readOnlyHarnessFiles).toEqual(EXPECTED_READ_ONLY_HARNESSES[profileId]);
    }
  });

  it('does not reference unrelated fixed harnesses', () => {
    const commandHarnessPairs = [
      {
        commandNeedle: 'walk-forward-short-horizon.test.ts',
        harness: 'src/tools/finance/backtest/walk-forward.ts',
      },
      {
        commandNeedle: 'walk-forward-btc-ultra-short-horizon.test.ts',
        harness: 'src/tools/finance/backtest/walk-forward.ts',
      },
      {
        commandNeedle: 'arbiter-replay-runner.test.ts',
        harness: 'src/tools/finance/backtest/arbiter-replay-runner.ts',
      },
    ] as const;

    for (const profile of listForecastLabProfiles()) {
      const commands = profile.baselineCommands.map((command) => command.command);
      const expectedHarnesses = new Set(
        commandHarnessPairs
          .filter(({ commandNeedle }) =>
            commands.some((command) => command.includes(commandNeedle)))
          .map(({ harness }) => harness),
      );

      for (const harness of FORECAST_LAB_READ_ONLY_HARNESS_FILES) {
        expect(profile.readOnlyHarnessFiles.includes(harness)).toBe(expectedHarnesses.has(harness));
      }
    }
  });

  it('does not allow harness, test, or docs edits through allowed globs', () => {
    for (const profile of listForecastLabProfiles()) {
      for (const allowedGlob of profile.allowedGlobs) {
        expect(allowedGlob.startsWith('src/tools/finance/')).toBe(true);
        expect(allowedGlob).not.toContain('/backtest/');
        expect(allowedGlob.endsWith('.test.ts')).toBe(false);
        expect(allowedGlob.startsWith('docs/')).toBe(false);
      }

      for (const mutableFile of profile.mutation.mutableFiles) {
        expect(mutableFile.startsWith('src/tools/finance/')).toBe(true);
        expect(mutableFile).not.toContain('/backtest/');
        expect(mutableFile.endsWith('.test.ts')).toBe(false);
        expect(mutableFile.startsWith('docs/')).toBe(false);
      }

      for (const harness of FORECAST_LAB_READ_ONLY_HARNESS_FILES) {
        expect(profile.allowedGlobs).not.toContain(harness);
        expect(profile.mutation.mutableFiles).not.toContain(harness);
      }

      for (const harness of profile.readOnlyHarnessFiles) {
        expect((FORECAST_LAB_READ_ONLY_HARNESS_FILES as readonly string[]).includes(harness)).toBe(true);
        expect(profile.allowedGlobs).not.toContain(harness);
        expect(profile.mutation.mutableFiles).not.toContain(harness);
      }
    }
  });

  it('keeps profile edit surfaces and mutation surfaces compatible without requiring identity', () => {
    for (const profile of listForecastLabProfiles()) {
      const allowedGlobs = new Set(profile.allowedGlobs);

      for (const mutableFile of profile.mutation.mutableFiles) {
        expect(allowedGlobs.has(mutableFile)).toBe(true);
      }
    }
  });

  it('declares bounded immutable mutation configs for each profile', () => {
    const knownMutators = new Set(listForecastLabMutatorIds());

    for (const profileId of EXPECTED_PROFILE_IDS) {
      const profile = getForecastLabProfile(profileId);
      const expected = EXPECTED_MUTATION_CONFIGS[profileId];

      expect(profile.mutation).toEqual(expected);
      if (profile.mutation.mode === 'structured') {
        for (const mutatorId of profile.mutation.allowedMutatorIds) {
          expect(knownMutators.has(mutatorId)).toBe(true);
        }
      }
    }
  });

  it('keeps structured mutator contracts aligned with shipped structured catalogs', () => {
    for (const profile of listForecastLabProfiles()) {
      const catalogMutatorIds = [...new Set(
        listForecastLabStructuredMutations(profile.id).map((candidate) => candidate.mutatorId),
      )];

      if (profile.mutation.mode !== 'structured') {
        expect(catalogMutatorIds).toEqual([]);
        expect('allowedMutatorIds' in profile.mutation).toBe(false);
        continue;
      }

      expect(catalogMutatorIds.length).toBeGreaterThan(0);
      expect(profile.mutation.allowedMutatorIds).toEqual(catalogMutatorIds);
    }
  });

  it('keeps candidate and baseline command sets in parity', () => {
    for (const profile of listForecastLabProfiles()) {
      const commands = comparableCommands(profile);
      expect(commands.candidate).toEqual(commands.baseline);
      expect(new Set(commands.baseline.map((command) => command.id)).size).toBe(commands.baseline.length);
    }
  });

  it('uses structured keep/drop rules over declared minimum metrics', () => {
    for (const profile of listForecastLabProfiles()) {
      const metricNames = new Set(profile.minimumMetrics.map((metric) => metric.name));

      for (const metric of profile.minimumMetrics) {
        expect(metric.required).toBe(true);
        expect(metric.baselinePath).toContain('baseline.');
        expect(metric.candidatePath).toContain('candidate.');
      }

      for (const criterion of [
        ...profile.keepDropRule.keepWhen.all,
        ...profile.keepDropRule.dropWhen.any,
      ]) {
        expect(metricNames.has(criterion.metric)).toBe(true);
        expect(criterion.reason.length).toBeGreaterThan(0);
        expect(Number.isFinite(criterion.value)).toBe(true);
      }
    }
  });

  it('uses only command-status metric paths for profiles backed by Bun test commands', () => {
    for (const profile of listForecastLabProfiles()) {
      const commands = [...profile.baselineCommands, ...profile.candidateCommands];
      const statusOnlyCommands = commands.every((command) => command.command.startsWith('bun test '));

      if (!statusOnlyCommands) {
        continue;
      }

      for (const metric of profile.minimumMetrics) {
        expect(metric).toMatchObject({
          ...COMMAND_STATUS_METRIC_PATHS,
          direction: 'lower-is-better',
          required: true,
        });
      }
    }
  });

  it('gates current Phase 3 profiles on command status rather than unexported harness metrics', () => {
    for (const profileId of EXPECTED_PROFILE_IDS) {
      const profile = getForecastLabProfile(profileId);
      const expected = EXPECTED_COMMAND_STATUS_PROFILES[profileId];
      const metricNames = profile.minimumMetrics.map((metric) => metric.name);

      expect(profile.baselineCommands).toEqual(profile.candidateCommands);
      expect(profile.baselineCommands).toEqual(expected.commands);
      expect(metricNames).toEqual([expected.metricName]);
      expect(metricNames).not.toContain('directionalAccuracy');
      expect(metricNames).not.toContain('brierScore');
      expect(metricNames).not.toContain('ciCoverage');
      expect(metricNames).not.toContain('abstainRate');
      expect(metricNames).not.toContain('selectedRelevantMarketRate');
      expect(metricNames).not.toContain('invalidSelectedMarketCount');
      expect(metricNames).not.toContain('replayCaptureRows');
      expect(profile.minimumMetrics[0]).toMatchObject({
        ...COMMAND_STATUS_METRIC_PATHS,
        direction: 'lower-is-better',
        required: true,
      });
      expect(profile.keepDropRule.keepWhen.all.map((criterion) => criterion.metric)).toEqual([
        expected.metricName,
        expected.metricName,
      ]);
      expect(profile.keepDropRule.dropWhen.any.map((criterion) => criterion.metric)).toEqual([
        expected.metricName,
      ]);
    }
  });
});
