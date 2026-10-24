import { describe, expect, it } from 'bun:test';
import { rankForecastLabMutators } from './mutator-ranker.js';
import { listForecastLabStructuredMutations } from './profiles.js';
import type { ForecastLabLedgerEntry } from './types.js';

const PROFILE_ID = 'multi-asset-markov-short-horizon';
const catalog = listForecastLabStructuredMutations(PROFILE_ID);
const droppedId = catalog[catalog.length - 1]!.id;

function droppedRun(runId: string): ForecastLabLedgerEntry {
  const passing = { exitCode: 0, commands: [] };
  return {
    runId,
    startedAt: '2026-05-01T00:00:00.000Z',
    profileId: PROFILE_ID,
    targetSubsystem: 'markov-distribution',
    candidateBranch: `forecast-lab/${runId}`,
    allowedGlobs: [],
    mutationMode: 'structured',
    mutationId: droppedId,
    baselineSummary: passing,
    candidateSummary: passing,
    decision: 'drop',
    reason: 'default drop: no keep rule satisfied',
    artifactsPath: `/tmp/${runId}`,
  };
}

describe('rankForecastLabMutators', () => {
  it('ranks an untested mutator ahead of one already dropped, so a dropped mutator is not re-picked forever', () => {
    const ranking = rankForecastLabMutators({
      profileId: PROFILE_ID,
      catalog,
      ledgerEntries: [droppedRun('drop-1')],
    });

    expect(ranking.rankedCandidates[0]!.id).not.toBe(droppedId);
    const dropped = ranking.rankedMutators.find((mutator) => mutator.id === droppedId)!;
    expect(dropped.regressedRuns).toBe(0);
    expect(dropped.score).toBeLessThan(0);
  });

  it('lets a dropped mutator compete on its history when the profile allows repeat candidate attempts', () => {
    const ranking = rankForecastLabMutators({
      profileId: PROFILE_ID,
      catalog,
      ledgerEntries: [droppedRun('drop-1')],
      allowMultipleCandidateAttempts: true,
    });

    expect(ranking.rankedCandidates[0]!.id).toBe(droppedId);
  });
});
