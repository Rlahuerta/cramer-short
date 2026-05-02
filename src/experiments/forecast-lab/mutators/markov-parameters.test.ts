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
  'btc-markov-short-horizon',
  'btc-markov-ultra-short-horizon',
] as const;

describe('forecast-lab markov parameter mutators', () => {
  it('only exposes bounded deterministic mutators for the first markov profiles', () => {
    expect(isForecastLabMarkovMutatorProfileId('btc-markov-short-horizon')).toBe(true);
    expect(isForecastLabMarkovMutatorProfileId('btc-markov-ultra-short-horizon')).toBe(true);
    expect(isForecastLabMarkovMutatorProfileId('btc-arbiter-replay')).toBe(false);

    for (const profileId of MARKOV_PROFILE_IDS) {
      const first = listMarkovParameterMutations(profileId);
      const second = listMarkovParameterMutations(profileId);

      expect(first).toBe(second);
      expect(listForecastLabStructuredMutations(profileId)).toBe(first);
      expect(first.length).toBe(8);
      expect(Object.isFrozen(first)).toBe(true);
      expect([...first.map((candidate) => candidate.id)]).toEqual([
        'markov-shorter-reactive-window',
        'markov-longer-stability-window',
        'markov-faster-decay-reaction',
        'markov-slower-decay-persistence',
        'markov-lower-confidence-trend-penalty',
        'markov-higher-confidence-divergence-weighted',
        'markov-calibrator-higher-sample-floor',
        'markov-calibrator-lower-sample-floor',
      ]);
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
    for (const profileId of MARKOV_PROFILE_IDS) {
      for (const candidate of listMarkovParameterMutations(profileId)) {
        for (const edit of candidate.edits) {
          const fileContents = readFileSync(join(REPO_ROOT, edit.filePath), 'utf8');
          const matchCount = fileContents.split(edit.search).length - 1;

          expect(matchCount).toBe(edit.expectedReplacements);
          expect(fileContents.includes(edit.replace)).toBe(false);
        }
      }
    }
  });
});
