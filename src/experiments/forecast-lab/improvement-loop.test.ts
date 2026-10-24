import { describe, expect, it, mock } from 'bun:test';
import { listMarkovParameterMutations } from './mutators/markov-parameters.js';
import {
  buildForecastLabImprovementTrials,
  runForecastLabImprovementSearch,
  scoreCandidateAgainstProfile,
  type ForecastLabImprovementEvaluation,
  type ForecastLabImprovementHorizonMetrics,
  type ForecastLabImprovementMetrics,
} from './improvement-loop.js';
import { getForecastLabProfile } from './profiles.js';

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

  it('never evaluates two candidates concurrently (evaluators share a process-global override map)', async () => {
    const seedA = findMutation('hype-markov-short-horizon', 'markov-shorter-reactive-window');
    const seedB = findMutation('hype-markov-short-horizon', 'markov-lower-confidence-trend-penalty');
    const trials = [1, 2, 3].map((n) => ({ ...seedB, id: `trial-${n}`, patchSummary: [`trial-${n}`] }));
    let inFlight = 0;
    let maxInFlight = 0;

    await runForecastLabImprovementSearch({
      seedCandidates: [seedA, seedB],
      maxIterations: 1,
      buildTrials: () => trials,
      evaluate: async (mutation) => {
        inFlight += 1;
        maxInFlight = Math.max(maxInFlight, inFlight);
        await new Promise((resolve) => setTimeout(resolve, 1));
        inFlight -= 1;
        const objectiveScore = mutation.id === 'trial-3' ? 20 : 10;
        return { objectiveScore, primaryScore: 0.5, keepSatisfied: 4, keepTotal: 8, decision: 'drop', metrics: {}, summary: mutation.id };
      },
    });

    expect(maxInFlight).toBe(1);
  });
});

describe('forecast-lab improvement loop scoring', () => {
  const horizon = (overrides: Partial<ForecastLabImprovementHorizonMetrics> = {}): ForecastLabImprovementHorizonMetrics => ({
    directionalAccuracy: 0.5,
    brierScore: 0.25,
    ciCoverage: 0.9,
    structuralBreakCount: 0,
    abstainCount: 10,
    abstainRate: 0.2,
    stepCount: 50,
    errorCount: 0,
    ...overrides,
  });
  const metrics = (overrides: Partial<Record<keyof ForecastLabImprovementMetrics, Partial<ForecastLabImprovementHorizonMetrics>>> = {}): ForecastLabImprovementMetrics => ({
    h1: horizon(overrides.h1),
    h2: horizon(overrides.h2),
    h3: horizon(overrides.h3),
    h7: horizon(overrides.h7),
    h14: horizon(overrides.h14),
  });
  const profile = getForecastLabProfile('sol-markov-short-horizon');
  const improvedH1 = { directionalAccuracy: 0.53 };

  it('keeps a clean candidate that clears every gate', () => {
    expect(scoreCandidateAgainstProfile(profile, metrics(), metrics({ h1: improvedH1 })).decision).toBe('keep');
  });

  it('drops a candidate whose walk-forward raised errors, however good its metrics look', () => {
    const scored = scoreCandidateAgainstProfile(profile, metrics(), metrics({ h1: improvedH1, h14: { errorCount: 3 } }));
    expect(scored.decision).toBe('drop');
    expect(scored.objectiveScore).toBeLessThan(0);
  });

  it('drops a candidate with an empty horizon instead of scoring its zeroed metrics', () => {
    const scored = scoreCandidateAgainstProfile(profile, metrics(), metrics({ h1: improvedH1, h2: { stepCount: 0 } }));
    expect(scored.decision).toBe('drop');
  });

  it('drops on a missing required metric instead of defaulting it to 0', () => {
    const candidate = metrics({ h1: improvedH1 });
    const { abstainRate: _dropped, ...h1WithoutAbstainRate } = candidate.h1;
    const scored = scoreCandidateAgainstProfile(
      profile,
      metrics(),
      { ...candidate, h1: h1WithoutAbstainRate as ForecastLabImprovementHorizonMetrics },
    );
    expect(scored.decision).toBe('drop');
    expect(scored.summary).toContain('missing required metric: solShortH1AbstainRate');
  });
});
