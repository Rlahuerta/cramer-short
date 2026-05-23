import { describe, expect, it, mock } from 'bun:test';
import { listMarkovParameterMutations } from './mutators/markov-parameters.js';
import {
  buildForecastLabImprovementTrials,
  runForecastLabImprovementSearch,
  type ForecastLabImprovementEvaluation,
} from './improvement-loop.js';

function findMutation(profileId: 'sol-markov-short-horizon' | 'hype-markov-short-horizon', id: string) {
  const mutation = listMarkovParameterMutations(profileId).find((candidate) => candidate.id === id);
  expect(mutation).toBeDefined();
  return mutation!;
}

describe('forecast-lab improvement loop', () => {
  it('builds bounded aggregate and single-parameter trial variants from a shipped mutator seed', () => {
    const seed = findMutation('sol-markov-short-horizon', 'markov-shorter-reactive-window');
    const trials = buildForecastLabImprovementTrials(seed, {
      iteration: 1,
      maxParameterTrials: 2,
    });

    expect(trials.map((trial) => trial.id)).toEqual([
      'markov-shorter-reactive-window--iter1-softer',
      'markov-shorter-reactive-window--iter1-stronger',
      'markov-shorter-reactive-window--iter1-transitionMinObservations-stronger',
    ]);

    for (const trial of trials) {
      expect(trial.profileId).toBe(seed.profileId);
      expect(trial.mutatorId).toBe(seed.mutatorId);
      expect(trial.specSummary.targetFiles).toEqual(seed.specSummary.targetFiles);
      expect(trial.patchSummary.length).toBeGreaterThan(0);

      for (const [index, edit] of trial.edits.entries()) {
        expect(edit.beforeValue).toBe(seed.edits[index]!.beforeValue);
        expect(edit.search).toBe(seed.edits[index]!.search);
        expect(edit.replace).not.toBe(edit.search);
      }
    }
  });

  it('scores shipped seeds, iterates on the best one, and stops when no better trial appears', async () => {
    const seedA = findMutation('hype-markov-short-horizon', 'markov-shorter-reactive-window');
    const seedB = findMutation('hype-markov-short-horizon', 'markov-lower-confidence-trend-penalty');
    const improved = {
      ...seedB,
      id: 'trial-improved',
      patchSummary: ['trial-improved'],
    };
    const regressed = {
      ...seedB,
      id: 'trial-regressed',
      patchSummary: ['trial-regressed'],
    };

    const evaluate = mock(async (mutation: typeof seedA): Promise<ForecastLabImprovementEvaluation<{ label: string }>> => {
      switch (mutation.id) {
        case seedA.id:
          return { objectiveScore: 10, primaryScore: 0.53, keepSatisfied: 4, keepTotal: 8, decision: 'drop', metrics: { label: 'seed-a' }, summary: 'seed-a' };
        case seedB.id:
          return { objectiveScore: 12, primaryScore: 0.54, keepSatisfied: 5, keepTotal: 8, decision: 'drop', metrics: { label: 'seed-b' }, summary: 'seed-b' };
        case improved.id:
          return { objectiveScore: 14, primaryScore: 0.56, keepSatisfied: 6, keepTotal: 8, decision: 'drop', metrics: { label: 'improved' }, summary: 'improved' };
        default:
          return { objectiveScore: 11, primaryScore: 0.51, keepSatisfied: 4, keepTotal: 8, decision: 'drop', metrics: { label: 'other' }, summary: 'other' };
      }
    });

    const result = await runForecastLabImprovementSearch({
      seedCandidates: [seedA, seedB],
      maxIterations: 3,
      buildTrials: (seed, context) => {
        if (context.iteration === 1) {
          return [improved, regressed];
        }
        return [];
      },
      evaluate,
    });

    expect(result.seedResults.map((entry) => entry.mutation.id)).toEqual([seedB.id, seedA.id]);
    expect(result.bestResult.mutation.id).toBe(improved.id);
    expect(result.history.map((entry) => entry.mutation.id)).toEqual([improved.id]);
    expect(result.iterationsRun).toBe(1);
    expect(evaluate).toHaveBeenCalledTimes(4);
  });
});
