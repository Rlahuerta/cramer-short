import type { MarkovDistributionPoint } from '../../tools/finance/markov-distribution.js';
import type { ToolCallRecord } from '../scratchpad.js';
import {
  inferDistributionTicker,
  inferMarkovQueryHorizon,
} from '../query-router.js';
import {
  isFiniteNumber,
  isFinitePositiveNumber,
  matchesTickerAndOptionalHorizon,
  parseToolCallData,
} from './tool-call-utils.js';

function inferRequestedBucketCount(query: string): number | null {
  const match = query.match(/\b(\d+)\s*(?:parts|buckets|bins|segments)\b/i);
  if (!match) return null;
  const parsed = Number.parseInt(match[1]!, 10);
  return Number.isInteger(parsed) && parsed >= 2 ? parsed : null;
}

function interpolateSurvivalProbability(
  distribution: MarkovDistributionPoint[],
  price: number,
): number | null {
  if (distribution.length === 0) return null;

  const sorted = [...distribution].sort((a, b) => a.price - b.price);
  if (price <= sorted[0]!.price) return sorted[0]!.probability;
  if (price >= sorted[sorted.length - 1]!.price) return sorted[sorted.length - 1]!.probability;

  for (let i = 0; i < sorted.length - 1; i += 1) {
    const left = sorted[i]!;
    const right = sorted[i + 1]!;
    if (price < left.price || price > right.price) continue;
    if (right.price === left.price) return left.probability;

    const weight = (price - left.price) / (right.price - left.price);
    return left.probability + ((right.probability - left.probability) * weight);
  }

  return sorted[sorted.length - 1]!.probability;
}

function estimateBucketProbabilityPct(
  distribution: MarkovDistributionPoint[],
  lower: number | null,
  upper: number | null,
): number | null {
  const lowerSurvival = lower === null ? 1 : interpolateSurvivalProbability(distribution, lower);
  const upperSurvival = upper === null ? 0 : interpolateSurvivalProbability(distribution, upper);
  if (lowerSurvival === null || upperSurvival === null) return null;
  return Math.max(0, Math.min(100, (lowerSurvival - upperSurvival) * 100));
}

function formatDensityPrice(value: number): string {
  return `$${value.toLocaleString('en-US', {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })}`;
}

function formatDensityRange(lower: number | null, upper: number | null): string {
  if (lower === null && upper === null) return 'N/A';
  if (lower === null && upper !== null) return `< ${formatDensityPrice(upper)}`;
  if (lower !== null && upper === null) return `> ${formatDensityPrice(lower)}`;
  return `${formatDensityPrice(lower!)}–${formatDensityPrice(upper!)}`;
}

function buildDensityThresholds(
  minPrice: number,
  maxPrice: number,
  bucketCount: number,
): number[] {
  const thresholdCount = bucketCount - 1;
  if (thresholdCount <= 0) return [];

  const useLogSpacing = minPrice > 0;
  return Array.from({ length: thresholdCount }, (_, index) => {
    const weight = (index + 1) / bucketCount;
    if (useLogSpacing) {
      const logPrice = Math.log(minPrice) + ((Math.log(maxPrice) - Math.log(minPrice)) * weight);
      return Math.exp(logPrice);
    }
    return minPrice + ((maxPrice - minPrice) * weight);
  });
}

function buildCanonicalDensityTable(query: string, toolCalls: ToolCallRecord[]): string | null {
  const requestedBucketCount = inferRequestedBucketCount(query);
  if (requestedBucketCount === null) return null;

  const desiredTicker = inferDistributionTicker(query);
  const desiredHorizon = inferMarkovQueryHorizon(query);

  for (let i = toolCalls.length - 1; i >= 0; i -= 1) {
    const call = toolCalls[i];
    if (call.tool !== 'markov_distribution') continue;
    if (!matchesTickerAndOptionalHorizon(call.args, desiredTicker, 'horizon', desiredHorizon)) continue;

    const data = parseToolCallData(call);
    if (!data || data['_tool'] !== 'markov_distribution' || data['status'] !== 'ok') continue;
    if (!Array.isArray(data['distribution'])) continue;

    const distribution = data['distribution']
      .filter((point): point is MarkovDistributionPoint => (
        !!point
        && typeof point === 'object'
        && isFinitePositiveNumber((point as Record<string, unknown>)['price'])
        && isFiniteNumber((point as Record<string, unknown>)['probability'])
      ))
      .map((point) => point as MarkovDistributionPoint)
      .sort((a, b) => a.price - b.price);

    if (distribution.length < 2) continue;

    const minPrice = distribution[0]!.price;
    const maxPrice = distribution[distribution.length - 1]!.price;
    if (!(minPrice > 0) || !(maxPrice > minPrice)) continue;

    const thresholds = buildDensityThresholds(minPrice, maxPrice, requestedBucketCount);

    const rows: string[] = [];
    for (let bucketIndex = 0; bucketIndex < requestedBucketCount; bucketIndex += 1) {
      const lower = bucketIndex === 0 ? null : thresholds[bucketIndex - 1]!;
      const upper = bucketIndex === requestedBucketCount - 1 ? null : thresholds[bucketIndex]!;
      const probabilityPct = estimateBucketProbabilityPct(distribution, lower, upper);
      if (probabilityPct === null) continue;

      rows.push(
        `| ${bucketIndex + 1} | ${formatDensityRange(lower, upper)} | ${probabilityPct.toFixed(2)}% |`,
      );
    }

    if (rows.length < requestedBucketCount) continue;

    return [
      `## ${requestedBucketCount}-Part Density Probability Table`,
      '',
      'Canonical scenario breakdown (P(bucket) = probability mass in each price range):',
      '',
      '| Bucket | Price Range | P(bucket) |',
      '|--------|-------------|-----------|',
      ...rows,
    ].join('\n');
  }

  return null;
}

export function ensureStructuredDensityTable(
  answer: string,
  query: string,
  toolCalls: ToolCallRecord[],
): string {
  if (inferRequestedBucketCount(query) === null) return answer;
  const canonicalTable = buildCanonicalDensityTable(query, toolCalls);
  if (!canonicalTable) return answer;

  const densitySectionPattern = /##\s+.*density probability table[\s\S]*?(?=\n---\n|\n##\s|\n###\s|$)/i;
  if (densitySectionPattern.test(answer)) {
    return answer.replace(densitySectionPattern, canonicalTable);
  }

  return `${canonicalTable}\n\n---\n\n${answer}`;
}
