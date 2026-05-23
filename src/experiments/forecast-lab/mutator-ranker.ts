import type { ForecastLabLedgerEntry } from './types.js';
import type { ForecastLabMarkovParameterMutationCandidate } from './mutators/markov-parameters.js';

export type ForecastLabMutatorHealth = 'healthy' | 'mixed' | 'underperforming' | 'untested';

export interface ForecastLabRankedMutator {
  readonly id: string;
  readonly mutatorId: string;
  readonly applicable: boolean;
  readonly unused: boolean;
  readonly attempts: number;
  readonly keptRuns: number;
  readonly droppedRuns: number;
  readonly regressedRuns: number;
  readonly keepRate: number;
  readonly dropRate: number;
  readonly regressionRate: number;
  readonly applicabilityFrequency: number;
  readonly score: number;
  readonly health: ForecastLabMutatorHealth;
}

export interface ForecastLabMutatorRanking {
  readonly profileId: string;
  readonly totalStructuredRuns: number;
  readonly rankedMutators: readonly ForecastLabRankedMutator[];
  readonly rankedCandidates: readonly (ForecastLabRankedMutator & {
    readonly candidate: ForecastLabMarkovParameterMutationCandidate;
  })[];
}

interface ForecastLabMutatorRankerParams {
  readonly profileId: string;
  readonly catalog: readonly ForecastLabMarkovParameterMutationCandidate[];
  readonly ledgerEntries: readonly ForecastLabLedgerEntry[];
  readonly usedMutationIds?: ReadonlySet<string>;
  readonly isApplicable?: (candidate: ForecastLabMarkovParameterMutationCandidate) => boolean;
}

interface ForecastLabHistoricalMutatorStats {
  attempts: number;
  keptRuns: number;
  droppedRuns: number;
  regressedRuns: number;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value);
}

function getExitCode(value: unknown): number | undefined {
  if (!isRecord(value) || typeof value.exitCode !== 'number' || !Number.isFinite(value.exitCode)) {
    return undefined;
  }

  return value.exitCode;
}

function getCommandExitCodes(value: unknown): Map<string, number> {
  const exitCodes = new Map<string, number>();
  if (!isRecord(value) || !Array.isArray(value.commands)) {
    return exitCodes;
  }

  for (const command of value.commands) {
    if (!isRecord(command)) {
      continue;
    }
    const { id, exitCode } = command;
    if (typeof id === 'string' && typeof exitCode === 'number' && Number.isFinite(exitCode)) {
      exitCodes.set(id, exitCode);
    }
  }

  return exitCodes;
}

function didRunRegress(entry: ForecastLabLedgerEntry): boolean {
  const baselineExitCode = getExitCode(entry.baselineSummary);
  const candidateExitCode = getExitCode(entry.candidateSummary);
  if (
    baselineExitCode !== undefined &&
    candidateExitCode !== undefined &&
    candidateExitCode > baselineExitCode
  ) {
    return true;
  }

  const baselineCommandExitCodes = getCommandExitCodes(entry.baselineSummary);
  const candidateCommandExitCodes = getCommandExitCodes(entry.candidateSummary);
  for (const [commandId, candidateCommandExitCode] of candidateCommandExitCodes.entries()) {
    const baselineCommandExitCode = baselineCommandExitCodes.get(commandId);
    if (
      baselineCommandExitCode !== undefined &&
      candidateCommandExitCode > baselineCommandExitCode
    ) {
      return true;
    }
  }

  return false;
}

function roundMetric(value: number): number {
  return Number(value.toFixed(4));
}

function compareDescending(left: number, right: number): number {
  return right - left;
}

function compareAscending(left: number, right: number): number {
  return left - right;
}

function classifyMutatorHealth(params: {
  readonly attempts: number;
  readonly keptRuns: number;
  readonly regressedRuns: number;
  readonly score: number;
}): ForecastLabMutatorHealth {
  if (params.attempts === 0) {
    return 'untested';
  }

  if (params.score < 0 || params.regressedRuns * 2 > params.attempts) {
    return 'underperforming';
  }

  if (params.keptRuns * 2 >= params.attempts && params.regressedRuns === 0) {
    return 'healthy';
  }

  return 'mixed';
}

export function rankForecastLabMutators(
  params: ForecastLabMutatorRankerParams,
): ForecastLabMutatorRanking {
  const statsByMutationId = new Map<string, ForecastLabHistoricalMutatorStats>();
  const relevantEntries = params.ledgerEntries.filter((entry) =>
    entry.profileId === params.profileId &&
    entry.mutationMode === 'structured' &&
    typeof entry.mutationId === 'string'
  );
  const totalStructuredRuns = relevantEntries.length;
  const knownMutationIds = new Set(params.catalog.map((candidate) => candidate.id));

  for (const entry of relevantEntries) {
    if (!entry.mutationId || !knownMutationIds.has(entry.mutationId)) {
      continue;
    }

    const stats = statsByMutationId.get(entry.mutationId) ?? {
      attempts: 0,
      keptRuns: 0,
      droppedRuns: 0,
      regressedRuns: 0,
    };
    stats.attempts += 1;
    if (entry.decision === 'keep') {
      stats.keptRuns += 1;
    } else {
      stats.droppedRuns += 1;
    }
    if (didRunRegress(entry)) {
      stats.regressedRuns += 1;
    }
    statsByMutationId.set(entry.mutationId, stats);
  }

  const rankedCandidates = params.catalog
    .map((candidate, catalogIndex) => {
      const stats = statsByMutationId.get(candidate.id) ?? {
        attempts: 0,
        keptRuns: 0,
        droppedRuns: 0,
        regressedRuns: 0,
      };
      const applicable = params.isApplicable?.(candidate) ?? true;
      const unused = !params.usedMutationIds?.has(candidate.id);
      const keepRate = stats.attempts === 0 ? 0 : stats.keptRuns / stats.attempts;
      const dropRate = stats.attempts === 0 ? 0 : stats.droppedRuns / stats.attempts;
      const regressionRate = stats.attempts === 0 ? 0 : stats.regressedRuns / stats.attempts;
      const applicabilityFrequency = totalStructuredRuns === 0 ? 0 : stats.attempts / totalStructuredRuns;
      const score = keepRate - regressionRate + (applicabilityFrequency * 0.25);

      return {
        candidate,
        catalogIndex,
        id: candidate.id,
        mutatorId: candidate.mutatorId,
        applicable,
        unused,
        attempts: stats.attempts,
        keptRuns: stats.keptRuns,
        droppedRuns: stats.droppedRuns,
        regressedRuns: stats.regressedRuns,
        keepRate: roundMetric(keepRate),
        dropRate: roundMetric(dropRate),
        regressionRate: roundMetric(regressionRate),
        applicabilityFrequency: roundMetric(applicabilityFrequency),
        score: roundMetric(score),
        health: classifyMutatorHealth({
          attempts: stats.attempts,
          keptRuns: stats.keptRuns,
          regressedRuns: stats.regressedRuns,
          score,
        }),
      };
    })
    .sort((left, right) =>
      compareDescending(Number(left.applicable && left.unused), Number(right.applicable && right.unused)) ||
      compareDescending(Number(left.applicable), Number(right.applicable)) ||
      compareDescending(left.score, right.score) ||
      compareDescending(left.keepRate, right.keepRate) ||
      compareAscending(left.regressionRate, right.regressionRate) ||
      compareDescending(left.applicabilityFrequency, right.applicabilityFrequency) ||
      compareDescending(left.attempts, right.attempts) ||
      compareAscending(left.catalogIndex, right.catalogIndex) ||
      left.id.localeCompare(right.id)
    );

  return {
    profileId: params.profileId,
    totalStructuredRuns,
    rankedCandidates,
    rankedMutators: rankedCandidates.map(({ candidate: _candidate, catalogIndex: _catalogIndex, ...mutator }) => mutator),
  };
}

function formatMutatorHealth(mutator: ForecastLabRankedMutator): string {
  return `${mutator.id} score=${mutator.score} keep=${mutator.keptRuns}/${mutator.attempts} regressions=${mutator.regressedRuns}/${mutator.attempts}`;
}

export function formatForecastLabMutatorHealthProgress(
  ranking: ForecastLabMutatorRanking,
): string {
  const bestCandidate = ranking.rankedMutators.find((mutator) => mutator.applicable && mutator.unused)
    ?? ranking.rankedMutators.find((mutator) => mutator.applicable)
    ?? ranking.rankedMutators[0];
  const underperformers = ranking.rankedMutators.filter((mutator) => mutator.health === 'underperforming');

  if (!bestCandidate) {
    return 'forecast-lab: mutator health ranking enabled with no shipped candidates';
  }

  return underperformers.length === 0
    ? `forecast-lab: mutator health best ${formatMutatorHealth(bestCandidate)}`
    : `forecast-lab: mutator health best ${formatMutatorHealth(bestCandidate)}; underperforming ${underperformers.map(formatMutatorHealth).join(', ')}`;
}
