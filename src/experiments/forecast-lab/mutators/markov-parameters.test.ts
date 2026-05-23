import { describe, expect, it } from 'bun:test';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';
import { getForecastLabProfile, listForecastLabStructuredMutations } from '../profiles.js';
import {
  isForecastLabMarkovMutatorProfileId,
  listMarkovParameterMutations,
} from './markov-parameters.js';

const REPO_ROOT = process.cwd();
const MARKOV_PROFILE_IDS = [
  'multi-asset-markov-short-horizon',
  'btc-markov-ultra-short-horizon',
  'sol-markov-short-horizon',
  'hype-markov-short-horizon',
  'gold-markov-short-horizon',
] as const;

function toAfterValueMap(profileId: (typeof MARKOV_PROFILE_IDS)[number]) {
  return Object.fromEntries(
    listMarkovParameterMutations(profileId).map((candidate) => [
      candidate.id,
      Object.fromEntries(candidate.edits.map((edit) => [edit.parameterId, edit.afterValue])),
    ]),
  );
}

describe('forecast-lab markov parameter mutators', () => {
  it('only exposes bounded deterministic mutators for the shipped markov profiles', () => {
    expect(isForecastLabMarkovMutatorProfileId('multi-asset-markov-short-horizon')).toBe(true);
    expect(isForecastLabMarkovMutatorProfileId('btc-markov-ultra-short-horizon')).toBe(true);
    expect(isForecastLabMarkovMutatorProfileId('sol-markov-short-horizon')).toBe(true);
    expect(isForecastLabMarkovMutatorProfileId('hype-markov-short-horizon')).toBe(true);
    expect(isForecastLabMarkovMutatorProfileId('gold-markov-short-horizon')).toBe(true);
    expect(isForecastLabMarkovMutatorProfileId('btc-arbiter-replay')).toBe(false);

    for (const profileId of MARKOV_PROFILE_IDS) {
      const first = listMarkovParameterMutations(profileId);
      const second = listMarkovParameterMutations(profileId);

      expect(first).toBe(second);
      expect(listForecastLabStructuredMutations(profileId)).toBe(first);
      expect(first.length).toBe(8);
      expect(Object.isFrozen(first)).toBe(true);
        expect([...first.map((candidate) => candidate.id)]).toEqual(
          profileId === 'gold-markov-short-horizon'
            ? [
                'gold-markov-shorter-reactive-window',
                'gold-markov-longer-stability-window',
                'gold-markov-faster-decay-reaction',
                'gold-markov-slower-decay-persistence',
                'gold-markov-lower-confidence-trend-penalty',
                'gold-markov-higher-confidence-divergence-weighted',
                'gold-markov-calibrator-higher-sample-floor',
                'gold-markov-calibrator-lower-sample-floor',
              ]
            : [
                'markov-shorter-reactive-window',
                'markov-longer-stability-window',
                'markov-faster-decay-reaction',
                'markov-slower-decay-persistence',
                'markov-lower-confidence-trend-penalty',
                'markov-higher-confidence-divergence-weighted',
                'markov-calibrator-higher-sample-floor',
                'markov-calibrator-lower-sample-floor',
              ],
        );
      }

    const nonMarkov = listForecastLabStructuredMutations('btc-arbiter-replay');
    expect(nonMarkov).toEqual([]);
    expect(Object.isFrozen(nonMarkov)).toBe(true);
  });

  it('matches the markov profile mutator contract exactly', () => {
    for (const profileId of MARKOV_PROFILE_IDS) {
      const profile = getForecastLabProfile(profileId);
      expect(profile.mutation.mode).toBe('structured');

      if (profile.mutation.mode !== 'structured') {
        continue;
      }

      const catalogMutatorIds = [...new Set(
        listMarkovParameterMutations(profileId).map((candidate) => candidate.mutatorId),
      )];
      expect(profile.mutation.allowedMutatorIds).toEqual(catalogMutatorIds);
    }
  });

  it('ships SOL and HYPE-specific catalogs instead of reusing the shared or GOLD parameter values', () => {
    const shared = toAfterValueMap('multi-asset-markov-short-horizon');
    const gold = toAfterValueMap('gold-markov-short-horizon');

    expect(toAfterValueMap('sol-markov-short-horizon')).toEqual({
      'markov-shorter-reactive-window': {
        momentumLookback: 9,
        structuralBreakMinLength: 28,
        scoreAggregationMinSamples: 10,
        scoreAggregationCalibrationWindow: 60,
        minSamplesPerRegime: 14,
        transitionMinObservations: 31,
        momentumAdjustmentScale: 0.252,
        momentumAdjustmentClamp: 0.00305,
      },
      'markov-longer-stability-window': {
        momentumLookback: 28,
        structuralBreakMinLength: 72,
        scoreAggregationMinSamples: 24,
        scoreAggregationCalibrationWindow: 144,
        minSamplesPerRegime: 36,
        transitionMinObservations: 36,
        momentumAdjustmentScale: 0.18,
        momentumAdjustmentClamp: 0.0024,
      },
      'markov-faster-decay-reaction': {
        transitionDecay: 0.945,
        adaptiveBreakLearningRateMultiplier: 1.85,
        adaptiveBreakCooloffWindow: 2,
      },
      'markov-slower-decay-persistence': {
        transitionDecay: 0.99,
        adaptiveBreakLearningRateMultiplier: 1.15,
        adaptiveBreakCooloffWindow: 3,
      },
      'markov-lower-confidence-trend-penalty': {
        recommendedConfidenceThreshold: 0.17,
        momentumAdjustmentScale: 0.34,
        momentumAdjustmentClamp: 0.004,
      },
      'markov-higher-confidence-divergence-weighted': {
        recommendedConfidenceThreshold: 0.3,
        momentumAdjustmentScale: 0.16,
        momentumAdjustmentClamp: 0.0021,
        trendPenaltyOnlyBreakConfidence: false,
        divergenceWeightedBreakConfidence: true,
      },
      'markov-calibrator-higher-sample-floor': {
        minSamplesPerRegime: 40,
        learningRate: 0.03,
        pidLearningRate: 0.042,
        integralDecay: 0.98,
      },
      'markov-calibrator-lower-sample-floor': {
        minSamplesPerRegime: 14,
        learningRate: 0.085,
        pidLearningRate: 0.06,
        integralDecay: 0.93,
      },
    });

    expect(toAfterValueMap('hype-markov-short-horizon')).toEqual({
      'markov-shorter-reactive-window': {
        momentumLookback: 8,
        structuralBreakMinLength: 24,
        scoreAggregationMinSamples: 8,
        scoreAggregationCalibrationWindow: 48,
        minSamplesPerRegime: 12,
        transitionMinObservations: 22,
        momentumAdjustmentScale: 0.5,
        momentumAdjustmentClamp: 0.006,
      },
      'markov-longer-stability-window': {
        momentumLookback: 24,
        structuralBreakMinLength: 60,
        scoreAggregationMinSamples: 22,
        scoreAggregationCalibrationWindow: 120,
        minSamplesPerRegime: 28,
        transitionMinObservations: 34,
        momentumAdjustmentScale: 0.22,
        momentumAdjustmentClamp: 0.0032,
      },
      'markov-faster-decay-reaction': {
        transitionDecay: 0.925,
        adaptiveBreakLearningRateMultiplier: 2.1,
        adaptiveBreakCooloffWindow: 2,
      },
      'markov-slower-decay-persistence': {
        transitionDecay: 0.983,
        adaptiveBreakLearningRateMultiplier: 1.2,
        adaptiveBreakCooloffWindow: 3,
      },
      'markov-lower-confidence-trend-penalty': {
        recommendedConfidenceThreshold: 0.15,
        momentumAdjustmentScale: 0.48,
        momentumAdjustmentClamp: 0.0058,
      },
      'markov-higher-confidence-divergence-weighted': {
        recommendedConfidenceThreshold: 0.28,
        momentumAdjustmentScale: 0.18,
        momentumAdjustmentClamp: 0.0026,
        trendPenaltyOnlyBreakConfidence: false,
        divergenceWeightedBreakConfidence: true,
      },
      'markov-calibrator-higher-sample-floor': {
        minSamplesPerRegime: 28,
        learningRate: 0.038,
        pidLearningRate: 0.042,
        integralDecay: 0.97,
      },
      'markov-calibrator-lower-sample-floor': {
        minSamplesPerRegime: 10,
        learningRate: 0.095,
        pidLearningRate: 0.065,
        integralDecay: 0.92,
      },
    });

    expect(toAfterValueMap('sol-markov-short-horizon')).not.toEqual(shared);
    expect(toAfterValueMap('hype-markov-short-horizon')).not.toEqual(shared);
    expect(toAfterValueMap('sol-markov-short-horizon')).not.toEqual(gold);
    expect(toAfterValueMap('hype-markov-short-horizon')).not.toEqual(gold);
  });

  it('retunes SOL and HYPE candidates with forecast-driving Markov and conformal parameters', () => {
    for (const profileId of ['sol-markov-short-horizon', 'hype-markov-short-horizon'] as const) {
      const byId = Object.fromEntries(
        listMarkovParameterMutations(profileId).map((candidate) => [candidate.id, candidate]),
      );

      expect(byId['markov-shorter-reactive-window'].edits.map((edit) => edit.parameterId)).toEqual(
        expect.arrayContaining(['transitionMinObservations', 'momentumAdjustmentScale', 'momentumAdjustmentClamp']),
      );
      expect(byId['markov-longer-stability-window'].edits.map((edit) => edit.parameterId)).toEqual(
        expect.arrayContaining(['transitionMinObservations', 'momentumAdjustmentScale', 'momentumAdjustmentClamp']),
      );
      expect(byId['markov-higher-confidence-divergence-weighted'].edits.map((edit) => edit.parameterId)).toEqual(
        expect.arrayContaining(['trendPenaltyOnlyBreakConfidence', 'divergenceWeightedBreakConfidence']),
      );
      expect(byId['markov-calibrator-higher-sample-floor'].edits.map((edit) => edit.parameterId)).toEqual(
        expect.arrayContaining(['pidLearningRate', 'integralDecay']),
      );
      expect(byId['markov-calibrator-lower-sample-floor'].edits.map((edit) => edit.parameterId)).toEqual(
        expect.arrayContaining(['pidLearningRate', 'integralDecay']),
      );
    }
  });

  it('produces machine-readable specs and human-readable summaries over allowed finance files only', () => {
    for (const profileId of MARKOV_PROFILE_IDS) {
      const seenIds = new Set<string>();

      for (const candidate of listMarkovParameterMutations(profileId)) {
        expect(seenIds.has(candidate.id)).toBe(false);
        seenIds.add(candidate.id);

        expect(candidate.profileId).toBe(profileId);
        expect(candidate.mutatorId).toBe('search-replace');
        expect(candidate.specSummary.mutatorId).toBe('search-replace');
        expect(candidate.specSummary.summary.length).toBeGreaterThan(0);
        expect(candidate.patchSummary.length).toBeGreaterThan(0);
        expect(candidate.specSummary.targetFiles).toEqual([...new Set(candidate.edits.map((edit) => edit.filePath))]);
        expect(Object.isFrozen(candidate)).toBe(true);
        expect(Object.isFrozen(candidate.patchSummary)).toBe(true);
        expect(Object.isFrozen(candidate.edits)).toBe(true);

        for (const edit of candidate.edits) {
          expect(edit.kind).toBe('search-replace');
          expect(edit.filePath.startsWith('src/tools/finance/')).toBe(true);
          expect(edit.filePath.includes('/backtest/')).toBe(false);
          expect(edit.filePath.endsWith('.test.ts')).toBe(false);
          expect(edit.filePath.startsWith('docs/')).toBe(false);
          expect(edit.search).not.toBe(edit.replace);
          expect(edit.expectedReplacements).toBe(1);
        }
      }
    }
  });

  it('anchors every search-replace spec to the current source text exactly once', () => {
    const fileCache = new Map<string, string>();

    for (const profileId of MARKOV_PROFILE_IDS) {
      for (const candidate of listMarkovParameterMutations(profileId)) {
        for (const edit of candidate.edits) {
          let fileContents = fileCache.get(edit.filePath);
          if (fileContents === undefined) {
            fileContents = readFileSync(join(REPO_ROOT, edit.filePath), 'utf8');
            fileCache.set(edit.filePath, fileContents);
          }

          const matchCount = fileContents.split(edit.search).length - 1;
          const replaceCount = fileContents.split(edit.replace).length - 1;

          if (matchCount !== edit.expectedReplacements) {
            throw new Error(
              `${profileId}/${candidate.id}/${edit.parameterId} expected ${edit.expectedReplacements} anchor in ${edit.filePath}, found ${matchCount}`,
            );
          }

          const patchedContents = fileContents.replace(edit.search, edit.replace);

          expect(patchedContents.split(edit.search).length - 1).toBe(0);
          expect(patchedContents.split(edit.replace).length - 1).toBe(replaceCount + 1);
        }
      }
    }
  });
});
