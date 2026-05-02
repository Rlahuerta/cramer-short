import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, resolve, sep } from 'node:path';
import { cramerShortPath, getCramerShortDir } from '../../utils/paths.js';
import { stableJsonStringify } from './ledger.js';
import type {
  ForecastLabDecision,
  ForecastLabProfileRoutingStats,
  ForecastLabRoutingInvocationSource,
  ForecastLabRoutingStats,
} from './types.js';

export class ForecastLabRouterMemoryError extends Error {
  override name = 'ForecastLabRouterMemoryError';
}

function assertInsideCramerShort(path: string): void {
  const root = resolve(getCramerShortDir());
  const resolved = resolve(path);

  if (resolved !== root && !resolved.startsWith(root + sep)) {
    throw new ForecastLabRouterMemoryError(`Refusing to write outside .cramer-short: ${path}`);
  }
}

function makeEmptyRoutingStats(): ForecastLabRoutingStats {
  return {
    version: 1,
    profiles: {},
  };
}

function validateNonNegativeInteger(field: string, value: unknown): number {
  if (!Number.isInteger(value) || (value as number) < 0) {
    throw new ForecastLabRouterMemoryError(`${field} must be a non-negative integer`);
  }

  return value as number;
}

function validateProfileRoutingStats(
  profileId: string,
  stats: unknown,
): ForecastLabProfileRoutingStats {
  if (!stats || typeof stats !== 'object') {
    throw new ForecastLabRouterMemoryError(`routing stats for "${profileId}" must be an object`);
  }

  const record = stats as Record<string, unknown>;
  const lastDecision = record.lastDecision;
  if (lastDecision !== 'keep' && lastDecision !== 'drop') {
    throw new ForecastLabRouterMemoryError(`routing stats for "${profileId}" must record a valid lastDecision`);
  }

  if (typeof record.lastRunAt !== 'string' || record.lastRunAt.trim() === '') {
    throw new ForecastLabRouterMemoryError(`routing stats for "${profileId}" must record lastRunAt`);
  }

  return {
    totalRuns: validateNonNegativeInteger(`${profileId}.totalRuns`, record.totalRuns),
    keptRuns: validateNonNegativeInteger(`${profileId}.keptRuns`, record.keptRuns),
    droppedRuns: validateNonNegativeInteger(`${profileId}.droppedRuns`, record.droppedRuns),
    autoRoutedRuns: validateNonNegativeInteger(`${profileId}.autoRoutedRuns`, record.autoRoutedRuns),
    manualRequestedRuns: validateNonNegativeInteger(`${profileId}.manualRequestedRuns`, record.manualRequestedRuns),
    lastDecision,
    lastRunAt: record.lastRunAt,
  };
}

export function validateForecastLabRoutingStats(stats: unknown): asserts stats is ForecastLabRoutingStats {
  if (!stats || typeof stats !== 'object') {
    throw new ForecastLabRouterMemoryError('routing stats must be an object');
  }

  const record = stats as Record<string, unknown>;
  if (record.version !== 1) {
    throw new ForecastLabRouterMemoryError('routing stats version must be 1');
  }

  if (!record.profiles || typeof record.profiles !== 'object' || Array.isArray(record.profiles)) {
    throw new ForecastLabRouterMemoryError('routing stats profiles must be an object');
  }

  const profiles = record.profiles as Record<string, unknown>;
  for (const [profileId, profileStats] of Object.entries(profiles)) {
    validateProfileRoutingStats(profileId, profileStats);
  }
}

export function getForecastLabRoutingStatsPath(): string {
  return cramerShortPath('forecast-lab-routing-stats.json');
}

export function readForecastLabRoutingStats(path = getForecastLabRoutingStatsPath()): ForecastLabRoutingStats {
  if (!existsSync(path)) {
    return makeEmptyRoutingStats();
  }

  const stats: unknown = JSON.parse(readFileSync(path, 'utf8'));
  validateForecastLabRoutingStats(stats);
  return stats;
}

export function updateForecastLabRoutingStats(
  profileId: string,
  decision: ForecastLabDecision,
  startedAt: string,
  invocationSource: ForecastLabRoutingInvocationSource,
  path = getForecastLabRoutingStatsPath(),
): ForecastLabRoutingStats {
  assertInsideCramerShort(path);

  const currentStats = readForecastLabRoutingStats(path);
  const previous = currentStats.profiles[profileId];
  const nextProfileStats: ForecastLabProfileRoutingStats = {
    totalRuns: (previous?.totalRuns ?? 0) + 1,
    keptRuns: (previous?.keptRuns ?? 0) + (decision === 'keep' ? 1 : 0),
    droppedRuns: (previous?.droppedRuns ?? 0) + (decision === 'drop' ? 1 : 0),
    autoRoutedRuns: (previous?.autoRoutedRuns ?? 0) + (invocationSource === 'auto-routed' ? 1 : 0),
    manualRequestedRuns: (previous?.manualRequestedRuns ?? 0) + (invocationSource === 'manual-request' ? 1 : 0),
    lastDecision: decision,
    lastRunAt: startedAt,
  };
  const nextStats: ForecastLabRoutingStats = {
    version: 1,
    profiles: {
      ...currentStats.profiles,
      [profileId]: nextProfileStats,
    },
  };

  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, `${stableJsonStringify(nextStats)}\n`, 'utf8');
  return nextStats;
}
