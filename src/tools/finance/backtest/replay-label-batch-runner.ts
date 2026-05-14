import { MS_PER_DAY } from '../../../utils/time.js';
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import {
  readArbiterReplayBundles,
  DEFAULT_ARBITER_REPLAY_BUNDLES_PATH,
  type ArbiterReplayBundle,
  type ArbiterReplayPolymarketMarket,
} from '../arbiter-replay.js';
import type { ReplayPriceHistory } from '../arbiter-replay-labeler.js';
import {
  DEFAULT_ARBITER_REPLAY_LABELED_PATH,
  runReplayLabelPass,
  type ReplayLabelRunResult,
  type ReplayLabelRunSummary,
} from './replay-label-runner.js';
import { assertNoCanonicalPathCollision } from './path-collision-guard.js';

const DAY_MS = MS_PER_DAY;

export interface ReplayTickerHistoryRequest {
  ticker: string;
  windowStartAt: string;
  windowEndAt: string;
  bundles: ArbiterReplayBundle[];
}

export type ReplayTickerHistoryLoader = (
  request: ReplayTickerHistoryRequest,
) => Promise<ReplayPriceHistory | null>;

export interface ReplayLabelBatchHistoryRequestAudit {
  ticker: string;
  windowStartAt: string;
  windowEndAt: string;
  bundleCount: number;
  historyFound: boolean;
  pointCount: number;
}

export interface ReplayLabelBatchReport {
  formatVersion: 'replay-label-batch-report.v1';
  inputPath: string;
  outputPath: string;
  labeledAt: string;
  summary: ReplayLabelRunSummary;
  historyRequests: ReplayLabelBatchHistoryRequestAudit[];
}

export function toReplayLabelBatchReportPath(outputPath: string): string {
  return outputPath.endsWith('.jsonl')
    ? `${outputPath.slice(0, -'.jsonl'.length)}.report.json`
    : `${outputPath}.report.json`;
}

export const DEFAULT_ARBITER_REPLAY_LABELED_REPORT_PATH = toReplayLabelBatchReportPath(
  DEFAULT_ARBITER_REPLAY_LABELED_PATH,
);

function forecastTargetTimeMs(bundle: ArbiterReplayBundle): number {
  return Date.parse(bundle.capturedAt) + bundle.horizonDays * DAY_MS;
}

function marketTargetTimeMs(
  market: ArbiterReplayPolymarketMarket,
  bundle: ArbiterReplayBundle,
): number {
  const marketEndMs = Date.parse(market.endDate);
  return Number.isFinite(marketEndMs) ? marketEndMs : forecastTargetTimeMs(bundle);
}

function buildTickerHistoryRequests(bundles: ArbiterReplayBundle[]): ReplayTickerHistoryRequest[] {
  const bundlesByTicker = new Map<string, ArbiterReplayBundle[]>();

  for (const bundle of bundles) {
    if (bundle.labels?.forecast !== undefined) continue;
    const group = bundlesByTicker.get(bundle.ticker);
    if (group) {
      group.push(bundle);
    } else {
      bundlesByTicker.set(bundle.ticker, [bundle]);
    }
  }

  const requests: ReplayTickerHistoryRequest[] = [];
  for (const [ticker, tickerBundles] of bundlesByTicker) {
    let startMs = Number.POSITIVE_INFINITY;
    let endMs = Number.NEGATIVE_INFINITY;

    for (const bundle of tickerBundles) {
      const capturedMs = Date.parse(bundle.capturedAt);
      if (!Number.isFinite(capturedMs)) continue;

      startMs = Math.min(startMs, capturedMs);
      endMs = Math.max(endMs, forecastTargetTimeMs(bundle));

      for (const market of bundle.polymarket?.selectedMarkets ?? []) {
        endMs = Math.max(endMs, marketTargetTimeMs(market, bundle));
      }
    }

    if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) continue;

    requests.push({
      ticker,
      windowStartAt: new Date(startMs).toISOString(),
      windowEndAt: new Date(endMs).toISOString(),
      bundles: tickerBundles,
    });
  }

  return requests;
}

function writeBundlesJsonl(bundles: ArbiterReplayBundle[], filePath: string): void {
  mkdirSync(dirname(filePath), { recursive: true });
  const content = bundles.length > 0
    ? `${bundles.map((bundle) => JSON.stringify(bundle)).join('\n')}\n`
    : '';
  writeFileSync(filePath, content, 'utf-8');
}

function writeReplayLabelBatchReport(report: ReplayLabelBatchReport, filePath: string): void {
  mkdirSync(dirname(filePath), { recursive: true });
  writeFileSync(filePath, `${JSON.stringify(report, null, 2)}\n`, 'utf-8');
}

export async function runReplayLabelBatch(params: {
  bundles: ArbiterReplayBundle[];
  loadHistory: ReplayTickerHistoryLoader;
  labeledAt?: string;
}): Promise<ReplayLabelRunResult> {
  const historyRequests = buildTickerHistoryRequests(params.bundles);
  const historiesByTicker = new Map<string, ReplayPriceHistory | null>();

  await Promise.all(historyRequests.map(async (request) => {
    historiesByTicker.set(request.ticker, await params.loadHistory(request));
  }));

  return runReplayLabelPass({
    bundles: params.bundles,
    labeledAt: params.labeledAt,
    getHistory: (ticker) => historiesByTicker.get(ticker) ?? null,
  });
}

export async function runReplayLabelBatchFromFile(params: {
  inputPath?: string;
  outputPath?: string;
  reportPath?: string;
  loadHistory: ReplayTickerHistoryLoader;
  labeledAt?: string;
}): Promise<ReplayLabelRunResult> {
  const inputPath = params.inputPath ?? DEFAULT_ARBITER_REPLAY_BUNDLES_PATH;
  const outputPath = params.outputPath ?? DEFAULT_ARBITER_REPLAY_LABELED_PATH;
  const reportPath = params.reportPath ?? toReplayLabelBatchReportPath(outputPath);

  assertNoCanonicalPathCollision(
    'replay-label-batch-runner',
    'inputPath and outputPath must differ to prevent in-place mutation.',
    { inputPath, outputPath },
  );
  assertNoCanonicalPathCollision(
    'replay-label-batch-runner',
    'reportPath must differ from both inputPath and outputPath.',
    { reportPath, inputPath, outputPath },
  );

  const bundles = readArbiterReplayBundles(inputPath);
  const historyRequestAudits = new Map<string, ReplayLabelBatchHistoryRequestAudit>();
  const result = await runReplayLabelBatch({
    bundles,
    loadHistory: async (request) => {
      const history = await params.loadHistory(request);
      historyRequestAudits.set(request.ticker, {
        ticker: request.ticker,
        windowStartAt: request.windowStartAt,
        windowEndAt: request.windowEndAt,
        bundleCount: request.bundles.length,
        historyFound: history !== null,
        pointCount: history?.points.length ?? 0,
      });
      return history;
    },
    labeledAt: params.labeledAt,
  });
  writeBundlesJsonl(result.bundles, outputPath);
  writeReplayLabelBatchReport({
    formatVersion: 'replay-label-batch-report.v1',
    inputPath,
    outputPath,
    labeledAt: result.labeledAt,
    summary: result.summary,
    historyRequests: buildTickerHistoryRequests(bundles)
      .map((request) => historyRequestAudits.get(request.ticker))
      .filter((request): request is ReplayLabelBatchHistoryRequestAudit => request !== undefined),
  }, reportPath);
  return result;
}
