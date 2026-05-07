import { mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import {
  readArbiterReplayBundles,
  DEFAULT_ARBITER_REPLAY_BUNDLES_PATH,
  type ArbiterReplayBundle,
} from '../arbiter-replay.js';
import {
  evaluateReplayLabelEligibility,
  labelReplayBundle,
  type ReplayPriceHistory,
} from '../arbiter-replay-labeler.js';
import { cramerShortPath } from '../../../utils/paths.js';
import { assertNoCanonicalPathCollision } from './path-collision-guard.js';

export const DEFAULT_ARBITER_REPLAY_LABELED_PATH = cramerShortPath('arbiter-replay-bundles-labeled.jsonl');

export interface ReplayLabelTickerCounts {
  total: number;
  alreadyLabeled: number;
  newlyLabeled: number;
  skippedByMissingHistory: number;
  pending: number;
}

export interface ReplayLabelRunSummary {
  total: number;
  alreadyLabeled: number;
  newlyLabeled: number;
  skippedByMissingHistory: number;
  /** Number of bundles that have history but are not yet eligible (e.g. horizon not reached). */
  pending: number;
  /** Map of eligibility reason string → number of bundles blocked by that reason. */
  pendingReasons: Record<string, number>;
  perTickerCounts: Record<string, ReplayLabelTickerCounts>;
}

export interface ReplayLabelRunResult {
  summary: ReplayLabelRunSummary;
  labeledAt: string;
  /** All bundles in the same order as the input, with newly labeled ones updated. */
  bundles: ArbiterReplayBundle[];
}

export type HistoryProvider = (ticker: string, bundle: ArbiterReplayBundle) => ReplayPriceHistory | null;

export function runReplayLabelPass(params: {
  bundles: ArbiterReplayBundle[];
  getHistory: HistoryProvider;
  labeledAt?: string;
}): ReplayLabelRunResult {
  const labeledAt = params.labeledAt ?? new Date().toISOString();

  const summary: ReplayLabelRunSummary = {
    total: params.bundles.length,
    alreadyLabeled: 0,
    newlyLabeled: 0,
    skippedByMissingHistory: 0,
    pending: 0,
    pendingReasons: {},
    perTickerCounts: {},
  };

  const outputBundles: ArbiterReplayBundle[] = [];

  for (const bundle of params.bundles) {
    const ticker = bundle.ticker;

    if (!summary.perTickerCounts[ticker]) {
      summary.perTickerCounts[ticker] = {
        total: 0,
        alreadyLabeled: 0,
        newlyLabeled: 0,
        skippedByMissingHistory: 0,
        pending: 0,
      };
    }
    const tc = summary.perTickerCounts[ticker]!;
    tc.total++;

    if (bundle.labels?.forecast !== undefined) {
      summary.alreadyLabeled++;
      tc.alreadyLabeled++;
      outputBundles.push(bundle);
      continue;
    }

    const history = params.getHistory(ticker, bundle);
    if (!history) {
      summary.skippedByMissingHistory++;
      tc.skippedByMissingHistory++;
      outputBundles.push(bundle);
      continue;
    }

    const eligibility = evaluateReplayLabelEligibility(bundle, history);
    if (!eligibility.ready) {
      summary.pending++;
      tc.pending++;
      for (const reason of eligibility.pendingReasons) {
        summary.pendingReasons[reason] = (summary.pendingReasons[reason] ?? 0) + 1;
      }
      outputBundles.push(bundle);
      continue;
    }

    const labeled = labelReplayBundle(bundle, history, labeledAt);
    summary.newlyLabeled++;
    tc.newlyLabeled++;
    outputBundles.push(labeled);
  }

  return { summary, labeledAt, bundles: outputBundles };
}

function writeBundlesJsonl(bundles: ArbiterReplayBundle[], filePath: string): void {
  mkdirSync(dirname(filePath), { recursive: true });
  const content = bundles.length > 0
    ? `${bundles.map((b) => JSON.stringify(b)).join('\n')}\n`
    : '';
  writeFileSync(filePath, content, 'utf-8');
}

export function runReplayLabelPassFromFile(params: {
  inputPath?: string;
  outputPath?: string;
  getHistory: HistoryProvider;
  labeledAt?: string;
}): ReplayLabelRunResult {
  const inputPath = params.inputPath ?? DEFAULT_ARBITER_REPLAY_BUNDLES_PATH;
  const outputPath = params.outputPath ?? DEFAULT_ARBITER_REPLAY_LABELED_PATH;

  assertNoCanonicalPathCollision(
    'replay-label-runner',
    'inputPath and outputPath must differ to prevent in-place mutation.',
    { inputPath, outputPath },
  );

  const bundles = readArbiterReplayBundles(inputPath);
  const result = runReplayLabelPass({ bundles, getHistory: params.getHistory, labeledAt: params.labeledAt });
  writeBundlesJsonl(result.bundles, outputPath);
  return result;
}
