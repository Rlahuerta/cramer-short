import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import {
  DEFAULT_ARBITER_REPLAY_BUNDLES_PATH,
  DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH,
  DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH,
  readArbiterReplayBundles,
  type ArbiterReplayBundle,
} from '../arbiter-replay.js';
import { DEFAULT_ARBITER_REPLAY_LABELED_PATH } from './replay-label-runner.js';
import { toReplayLabelBatchReportPath } from './replay-label-batch-runner.js';
import { toReplayLabelBenchmarkReportPath } from './replay-label-benchmark-pipeline.js';
import { assertNoCanonicalPathCollision } from './path-collision-guard.js';

export interface ReplayLabelPromotionReceipt {
  formatVersion: 'replay-label-promotion-receipt.v1';
  promotedAt: string;
  receiptPath: string;
  source: {
    stagedLabeledPath: string;
    stagedLabelReportPath: string;
    stagedBenchmarkReportPath: string;
  };
  target: {
    promotedLabeledPath: string;
    promotedLabelReportPath: string;
    promotedBenchmarkReportPath: string;
  };
  bundleCount: number;
  labeledBundleCount: number;
}

export interface ReplayLabelPromotionResult {
  receipt: ReplayLabelPromotionReceipt;
  bundles: ArbiterReplayBundle[];
}

export function toReplayLabelPromotionReceiptPath(promotedLabeledPath: string): string {
  return promotedLabeledPath.endsWith('.jsonl')
    ? `${promotedLabeledPath.slice(0, -'.jsonl'.length)}.promotion.receipt.json`
    : `${promotedLabeledPath}.promotion.receipt.json`;
}

export const DEFAULT_ARBITER_REPLAY_LABELED_CACHE_PROMOTION_RECEIPT_PATH =
  toReplayLabelPromotionReceiptPath(DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH);
export { DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH };

function assertRequiredArtifactExists(path: string, label: string): void {
  if (!existsSync(path)) {
    throw new Error(`replay-label promotion: missing staged ${label} at ${path}`);
  }
}

function countNonEmptyJsonlRows(path: string): number {
  return readFileSync(path, 'utf-8')
    .split(/\r?\n/)
    .filter((line) => line.trim().length > 0)
    .length;
}

function assertReadableBundles(path: string, label: string): ArbiterReplayBundle[] {
  const bundles = readArbiterReplayBundles(path);
  const expectedRowCount = countNonEmptyJsonlRows(path);
  if (bundles.length !== expectedRowCount) {
    throw new Error(
      `replay-label promotion: ${label} must parse cleanly via readArbiterReplayBundles (${bundles.length}/${expectedRowCount} rows parsed)`,
    );
  }
  return bundles;
}

function copyArtifact(sourcePath: string, targetPath: string): void {
  mkdirSync(dirname(targetPath), { recursive: true });
  copyFileSync(sourcePath, targetPath);
}

function assertNoRawReplayBundleCollision(label: string, path: string): void {
  assertNoCanonicalPathCollision(
    'replay-label-promotion',
    `${label} must differ from the raw replay capture bundle path.`,
    {
      [label]: path,
      rawReplayCaptureBundlePath: DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH,
    },
  );
  assertNoCanonicalPathCollision(
    'replay-label-promotion',
    `${label} must differ from the legacy raw replay bundle path.`,
    {
      [label]: path,
      rawReplayBundlePath: DEFAULT_ARBITER_REPLAY_BUNDLES_PATH,
    },
  );
}

export function promoteReplayLabelArtifacts(params: {
  stagedLabeledPath?: string;
  stagedLabelReportPath?: string;
  stagedBenchmarkReportPath?: string;
  promotedLabeledPath?: string;
  promotedLabelReportPath?: string;
  promotedBenchmarkReportPath?: string;
  receiptPath?: string;
  promotedAt?: string;
} = {}): ReplayLabelPromotionResult {
  const stagedLabeledPath = params.stagedLabeledPath ?? DEFAULT_ARBITER_REPLAY_LABELED_PATH;
  const stagedLabelReportPath = params.stagedLabelReportPath ?? toReplayLabelBatchReportPath(stagedLabeledPath);
  const stagedBenchmarkReportPath = params.stagedBenchmarkReportPath
    ?? toReplayLabelBenchmarkReportPath(stagedLabeledPath);
  const promotedLabeledPath = params.promotedLabeledPath ?? DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH;
  const promotedLabelReportPath = params.promotedLabelReportPath
    ?? toReplayLabelBatchReportPath(promotedLabeledPath);
  const promotedBenchmarkReportPath = params.promotedBenchmarkReportPath
    ?? toReplayLabelBenchmarkReportPath(promotedLabeledPath);
  const receiptPath = params.receiptPath ?? toReplayLabelPromotionReceiptPath(promotedLabeledPath);
  const promotedAt = params.promotedAt ?? new Date().toISOString();

  assertNoRawReplayBundleCollision('stagedLabeledPath', stagedLabeledPath);
  assertNoRawReplayBundleCollision('stagedLabelReportPath', stagedLabelReportPath);
  assertNoRawReplayBundleCollision('stagedBenchmarkReportPath', stagedBenchmarkReportPath);
  assertNoRawReplayBundleCollision('promotedLabeledPath', promotedLabeledPath);
  assertNoRawReplayBundleCollision('promotedLabelReportPath', promotedLabelReportPath);
  assertNoRawReplayBundleCollision('promotedBenchmarkReportPath', promotedBenchmarkReportPath);
  assertNoRawReplayBundleCollision('receiptPath', receiptPath);
  assertNoCanonicalPathCollision(
    'replay-label-promotion',
    'staged artifacts, promoted artifacts, and the receipt path must all differ.',
    {
      stagedLabeledPath,
      stagedLabelReportPath,
      stagedBenchmarkReportPath,
      promotedLabeledPath,
      promotedLabelReportPath,
      promotedBenchmarkReportPath,
      receiptPath,
    },
  );

  assertRequiredArtifactExists(stagedLabeledPath, 'labeled JSONL');
  assertRequiredArtifactExists(stagedLabelReportPath, 'label report');
  assertRequiredArtifactExists(stagedBenchmarkReportPath, 'benchmark report');

  assertReadableBundles(stagedLabeledPath, 'staged labeled JSONL');
  copyArtifact(stagedLabeledPath, promotedLabeledPath);
  copyArtifact(stagedLabelReportPath, promotedLabelReportPath);
  copyArtifact(stagedBenchmarkReportPath, promotedBenchmarkReportPath);

  const promotedBundles = assertReadableBundles(promotedLabeledPath, 'promoted labeled JSONL');
  const receipt: ReplayLabelPromotionReceipt = {
    formatVersion: 'replay-label-promotion-receipt.v1',
    promotedAt,
    receiptPath,
    source: {
      stagedLabeledPath,
      stagedLabelReportPath,
      stagedBenchmarkReportPath,
    },
    target: {
      promotedLabeledPath,
      promotedLabelReportPath,
      promotedBenchmarkReportPath,
    },
    bundleCount: promotedBundles.length,
    labeledBundleCount: promotedBundles.filter((bundle) => bundle.labels?.forecast !== undefined).length,
  };

  mkdirSync(dirname(receiptPath), { recursive: true });
  writeFileSync(receiptPath, `${JSON.stringify(receipt, null, 2)}\n`, 'utf-8');
  return {
    receipt,
    bundles: promotedBundles,
  };
}
