/**
 * Cramer-Short replay-label command: headless benchmark pipeline runner.
 *
 * Usage:
 *   cramer-short replay-label run [options]
 *   cramer-short replay-label readiness [options]
 *   cramer-short replay-label promote [options]
 *   cramer-short replay-label help
 *
 * Run options:
 *   --input PATH            Input bundles path (default: cache bundles path)
 *   --output PATH           Labeled output path (default: labeled path)
 *   --label-report PATH     Label batch report path (derived from --output by default)
 *   --benchmark-report PATH Benchmark report path (derived from --output by default)
 *   --loader MODE           History loader mode (default: fixture)
 *                           Modes: fixture, local:<path>
 *
 * Promote options:
 *   --input PATH            Staged labeled JSONL path (default: staged labeled path)
 *   --output PATH           Promoted labeled cache path (default: promoted labeled cache path)
 *   --label-report PATH     Staged label report path (derived from --input by default)
 *   --benchmark-report PATH Staged benchmark report path (derived from --input by default)
 *   --receipt PATH          Promotion receipt path (derived from --output by default)
 *
 * Readiness options:
 *   --input PATH            Benchmark artifact/report path (default: staged benchmark report path)
 *   --min-labeled MAP       Per-horizon minimum labeled bundles, e.g. 1d=30,2d=25,3d=20
 *   --min-traded MAP        Per-horizon minimum traded rows, e.g. 1d=15,2d=12,3d=10
 *   --max-abstain MAP       Per-horizon maximum abstain rate, e.g. 1d=0.45,2d=0.5,3d=0.55
 *   --max-brier MAP         Per-horizon maximum brier score, e.g. 1d=0.24,2d=0.25,3d=0.26
 *   --allow-not-ready       Do not require benchmark slices to be marked ready
 */
import {
  runReplayLabelBenchmarkPipelineFromFile,
  toReplayLabelBenchmarkReportPath,
  type ReplayLabelBenchmarkPipelineResult,
} from './tools/finance/backtest/replay-label-benchmark-pipeline.js';
import {
  DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH,
  promoteReplayLabelArtifacts,
  toReplayLabelPromotionReceiptPath,
  type ReplayLabelPromotionResult,
} from './tools/finance/backtest/replay-label-promotion.js';
import { DEFAULT_ARBITER_REPLAY_LABELED_PATH } from './tools/finance/backtest/replay-label-runner.js';
import {
  toReplayLabelBatchReportPath,
  type ReplayTickerHistoryLoader,
} from './tools/finance/backtest/replay-label-batch-runner.js';
import {
  runReplayLabelReadinessFromFile,
  type ReplayLabelReadinessReport,
  type ReplayLabelReadinessThresholdOverrides,
} from './tools/finance/backtest/replay-label-readiness.js';
import { DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH } from './tools/finance/arbiter-replay.js';
import { createReplayPriceHistoryProvider } from './tools/finance/backtest/replay-price-history-adapter.js';

// ─── loader seam ─────────────────────────────────────────────────────────────

/**
 * Create a ReplayTickerHistoryLoader from a loader mode string.
 *
 * Supported modes:
 *   "fixture"        — uses the bundled backtest-prices.json fixture (no network)
 *   "local:<path>"   — reads a custom ReplayFixturePriceStore JSON at <path>
 *
 * Both modes are purely local; no live API calls are made.
 */
export function createReplayHistoryLoader(mode: string): ReplayTickerHistoryLoader {
  if (mode === 'fixture') {
    const provider = createReplayPriceHistoryProvider();
    return async (request) => {
      const bundle = request.bundles[0] ?? { ticker: request.ticker } as Parameters<typeof provider>[1];
      return provider(request.ticker, bundle);
    };
  }

  if (mode.startsWith('local:')) {
    const fixturePath = mode.slice('local:'.length);
    if (!fixturePath) {
      throw new Error(
        'replay-label: --loader local: requires a path, e.g. --loader local:/path/to/fixture.json',
      );
    }
    // createReplayPriceHistoryProvider reads the file immediately — throws now if not found
    const provider = createReplayPriceHistoryProvider({ fixturePath });
    return async (request) => {
      const bundle = request.bundles[0] ?? { ticker: request.ticker } as Parameters<typeof provider>[1];
      return provider(request.ticker, bundle);
    };
  }

  throw new Error(
    `replay-label: unknown --loader mode "${mode}". Supported modes: fixture, local:<path>`,
  );
}

// ─── arg parsing ─────────────────────────────────────────────────────────────

interface ParsedRunArgs {
  inputPath: string;
  outputPath: string;
  labelReportPath: string | undefined;
  benchmarkReportPath: string | undefined;
  loaderMode: string;
}

interface ParsedPromoteArgs {
  stagedLabeledPath: string;
  stagedLabelReportPath: string | undefined;
  stagedBenchmarkReportPath: string | undefined;
  promotedLabeledPath: string;
  receiptPath: string | undefined;
}

interface ParsedReadinessArgs {
  inputPath: string;
  thresholds: ReplayLabelReadinessThresholdOverrides;
}

function parseHorizonMetricMap(flagName: string, value: string): Partial<Record<'1d' | '2d' | '3d', number>> {
  const result: Partial<Record<'1d' | '2d' | '3d', number>> = {};

  for (const entry of value.split(',')) {
    const trimmed = entry.trim();
    if (!trimmed) continue;
    const [horizonKey, rawMetric] = trimmed.split('=', 2);
    if (horizonKey !== '1d' && horizonKey !== '2d' && horizonKey !== '3d') {
      throw new Error(`replay-label readiness: ${flagName} only accepts 1d,2d,3d horizon keys`);
    }
    const metric = Number(rawMetric);
    if (!Number.isFinite(metric)) {
      throw new Error(`replay-label readiness: ${flagName} requires numeric values, received "${trimmed}"`);
    }
    result[horizonKey] = metric;
  }

  if (Object.keys(result).length === 0) {
    throw new Error(`replay-label readiness: ${flagName} requires a map like 1d=30,2d=25,3d=20`);
  }

  return result;
}

function parseRunArgs(argv: string[]): ParsedRunArgs {
  let inputPath = DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH;
  let outputPath = DEFAULT_ARBITER_REPLAY_LABELED_PATH;
  let labelReportPath: string | undefined;
  let benchmarkReportPath: string | undefined;
  let loaderMode = 'fixture';

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];

    if (arg === '--input') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label run: --input requires a path');
      inputPath = val;
      continue;
    }

    if (arg === '--output') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label run: --output requires a path');
      outputPath = val;
      continue;
    }

    if (arg === '--label-report') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label run: --label-report requires a path');
      labelReportPath = val;
      continue;
    }

    if (arg === '--benchmark-report') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label run: --benchmark-report requires a path');
      benchmarkReportPath = val;
      continue;
    }

    if (arg === '--loader') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) {
        throw new Error('replay-label run: --loader requires a mode (fixture | local:<path>)');
      }
      loaderMode = val;
      continue;
    }

    throw new Error(`replay-label run: unknown flag "${arg}"`);
  }

  return { inputPath, outputPath, labelReportPath, benchmarkReportPath, loaderMode };
}

function parsePromoteArgs(argv: string[]): ParsedPromoteArgs {
  let stagedLabeledPath = DEFAULT_ARBITER_REPLAY_LABELED_PATH;
  let stagedLabelReportPath: string | undefined;
  let stagedBenchmarkReportPath: string | undefined;
  let promotedLabeledPath = DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH;
  let receiptPath: string | undefined;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];

    if (arg === '--input') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label promote: --input requires a path');
      stagedLabeledPath = val;
      continue;
    }

    if (arg === '--label-report') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) {
        throw new Error('replay-label promote: --label-report requires a path');
      }
      stagedLabelReportPath = val;
      continue;
    }

    if (arg === '--benchmark-report') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) {
        throw new Error('replay-label promote: --benchmark-report requires a path');
      }
      stagedBenchmarkReportPath = val;
      continue;
    }

    if (arg === '--output') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label promote: --output requires a path');
      promotedLabeledPath = val;
      continue;
    }

    if (arg === '--receipt') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label promote: --receipt requires a path');
      receiptPath = val;
      continue;
    }

    throw new Error(`replay-label promote: unknown flag "${arg}"`);
  }

  return {
    stagedLabeledPath,
    stagedLabelReportPath,
    stagedBenchmarkReportPath,
    promotedLabeledPath,
    receiptPath,
  };
}

function parseReadinessArgs(argv: string[]): ParsedReadinessArgs {
  let inputPath = toReplayLabelBenchmarkReportPath(DEFAULT_ARBITER_REPLAY_LABELED_PATH);
  const thresholds: ReplayLabelReadinessThresholdOverrides = {};

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];

    if (arg === '--input') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label readiness: --input requires a path');
      inputPath = val;
      continue;
    }

    if (arg === '--min-labeled') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label readiness: --min-labeled requires a map');
      thresholds.minLabeledBundleCountByHorizon = parseHorizonMetricMap('--min-labeled', val);
      continue;
    }

    if (arg === '--min-traded') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label readiness: --min-traded requires a map');
      thresholds.minTradedRowCountByHorizon = parseHorizonMetricMap('--min-traded', val);
      continue;
    }

    if (arg === '--max-abstain') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label readiness: --max-abstain requires a map');
      thresholds.maxAbstainRateByHorizon = parseHorizonMetricMap('--max-abstain', val);
      continue;
    }

    if (arg === '--max-brier') {
      const val = argv[++i];
      if (!val || val.startsWith('--')) throw new Error('replay-label readiness: --max-brier requires a map');
      thresholds.maxBrierScoreByHorizon = parseHorizonMetricMap('--max-brier', val);
      continue;
    }

    if (arg === '--allow-not-ready') {
      thresholds.requireReadySlice = false;
      continue;
    }

    throw new Error(`replay-label readiness: unknown flag "${arg}"`);
  }

  return { inputPath, thresholds };
}

// ─── output types ─────────────────────────────────────────────────────────────

export interface ReplayLabelRunSummaryOutput {
  status: 'ok';
  inputPath: string;
  outputPath: string;
  labelReportPath: string;
  benchmarkReportPath: string;
  labelingSummary: {
    total: number;
    alreadyLabeled: number;
    newlyLabeled: number;
    skippedByMissingHistory: number;
    pending: number;
  };
}

export interface ReplayLabelPromoteSummaryOutput {
  status: 'ok';
  stagedLabeledPath: string;
  stagedLabelReportPath: string;
  stagedBenchmarkReportPath: string;
  promotedLabeledPath: string;
  promotedLabelReportPath: string;
  promotedBenchmarkReportPath: string;
  receiptPath: string;
  promotedAt: string;
  bundleCount: number;
  labeledBundleCount: number;
}

export interface ReplayLabelReadinessSummaryOutput {
  status: 'ok';
  inputPath: string;
  report: ReplayLabelReadinessReport;
}

// ─── command options ──────────────────────────────────────────────────────────

export interface ReplayLabelCommandOptions {
  log?: (message: string) => void;
  error?: (message: string) => void;
  exit?: (code: number) => void;
  /** Override pipeline for testing. */
  runPipeline?: (params: {
    inputPath?: string;
    outputPath?: string;
    labelReportPath?: string;
    benchmarkReportPath?: string;
    loadHistory: ReplayTickerHistoryLoader;
    labeledAt?: string;
    benchmarkGeneratedAt?: string;
  }) => Promise<ReplayLabelBenchmarkPipelineResult>;
  /** Override loader factory for testing. */
  loaderFactory?: (mode: string) => ReplayTickerHistoryLoader;
  /** Override promotion for testing. */
  promoteArtifacts?: (params: {
    stagedLabeledPath?: string;
    stagedLabelReportPath?: string;
    stagedBenchmarkReportPath?: string;
    promotedLabeledPath?: string;
    receiptPath?: string;
    promotedAt?: string;
  }) => ReplayLabelPromotionResult;
  /** Override readiness evaluation for testing. */
  runReadiness?: (params: {
    inputPath: string;
    thresholds?: ReplayLabelReadinessThresholdOverrides;
    generatedAt?: string;
  }) => ReplayLabelReadinessReport;
}

// ─── usage ────────────────────────────────────────────────────────────────────

function printUsage(log: (m: string) => void): void {
  log(
    [
      'Usage:',
      '  cramer-short replay-label run [options]',
      '  cramer-short replay-label readiness [options]',
      '  cramer-short replay-label promote [options]',
      '',
      'Runs the replay-label benchmark pipeline from the raw replay cache.',
      'Promotion is explicit only; nothing is promoted until replay-label promote is invoked.',
      'Readiness reports whether the staged replay-label benchmark is eligible or should hold.',
      '',
      'Options:',
      `  --input PATH            Input bundles path (default: ${DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH})`,
      `  --output PATH           Labeled output path (default: ${DEFAULT_ARBITER_REPLAY_LABELED_PATH})`,
      '  --label-report PATH     Label batch report path (derived from --output by default)',
      '  --benchmark-report PATH Benchmark report path (derived from --output by default)',
      '  --loader MODE           History loader mode (default: fixture)',
      '                          Modes: fixture, local:<path>',
      '',
      'Promote options:',
      `  --input PATH            Staged labeled JSONL path (default: ${DEFAULT_ARBITER_REPLAY_LABELED_PATH})`,
      `  --output PATH           Promoted labeled cache path (default: ${DEFAULT_ARBITER_REPLAY_LABELED_CACHE_BUNDLES_PATH})`,
      '  --label-report PATH     Staged label report path (derived from --input by default)',
      '  --benchmark-report PATH Staged benchmark report path (derived from --input by default)',
      '  --receipt PATH          Promotion receipt path (derived from --output by default)',
      '',
      'Readiness options:',
      `  --input PATH            Benchmark artifact/report path (default: ${toReplayLabelBenchmarkReportPath(DEFAULT_ARBITER_REPLAY_LABELED_PATH)})`,
      '  --min-labeled MAP       Override min labeled bundles, e.g. 1d=30,2d=25,3d=20',
      '  --min-traded MAP        Override min traded rows, e.g. 1d=15,2d=12,3d=10',
      '  --max-abstain MAP       Override max abstain rate, e.g. 1d=0.45,2d=0.5,3d=0.55',
      '  --max-brier MAP         Override max brier score, e.g. 1d=0.24,2d=0.25,3d=0.26',
      '  --allow-not-ready       Ignore the benchmark slice ready flag',
    ].join('\n'),
  );
}

// ─── command entry ────────────────────────────────────────────────────────────

export async function runReplayLabelCommand(
  argv: string[],
  options: ReplayLabelCommandOptions = {},
): Promise<void> {
  const log = options.log ?? console.log;
  const error = options.error ?? console.error;
  const exit = options.exit ?? ((code) => process.exit(code));

  const [subCmd, ...rest] = argv;

  if (!subCmd || subCmd === 'help' || subCmd === '--help') {
    printUsage(log);
    return;
  }

  if (subCmd === 'run') {
    let parsed: ParsedRunArgs;
    try {
      parsed = parseRunArgs(rest);
    } catch (err) {
      error(err instanceof Error ? err.message : String(err));
      printUsage(log);
      exit(1);
      return;
    }

    let loadHistory: ReplayTickerHistoryLoader;
    try {
      const loaderFactory = options.loaderFactory ?? createReplayHistoryLoader;
      loadHistory = loaderFactory(parsed.loaderMode);
    } catch (err) {
      error(err instanceof Error ? err.message : String(err));
      exit(1);
      return;
    }

    const pipeline = options.runPipeline ?? runReplayLabelBenchmarkPipelineFromFile;
    const labelReportPath = parsed.labelReportPath ?? toReplayLabelBatchReportPath(parsed.outputPath);
    const benchmarkReportPath =
      parsed.benchmarkReportPath ?? toReplayLabelBenchmarkReportPath(parsed.outputPath);

    try {
      const result = await pipeline({
        inputPath: parsed.inputPath,
        outputPath: parsed.outputPath,
        labelReportPath,
        benchmarkReportPath,
        loadHistory,
      });

      const summary: ReplayLabelRunSummaryOutput = {
        status: 'ok',
        inputPath: parsed.inputPath,
        outputPath: parsed.outputPath,
        labelReportPath,
        benchmarkReportPath,
        labelingSummary: {
          total: result.labeling.summary.total,
          alreadyLabeled: result.labeling.summary.alreadyLabeled,
          newlyLabeled: result.labeling.summary.newlyLabeled,
          skippedByMissingHistory: result.labeling.summary.skippedByMissingHistory,
          pending: result.labeling.summary.pending,
        },
      };

      log(JSON.stringify(summary, null, 2));
    } catch (err) {
      error(err instanceof Error ? err.message : String(err));
      exit(1);
    }

    return;
  }

  if (subCmd === 'promote') {
    let parsed: ParsedPromoteArgs;
    try {
      parsed = parsePromoteArgs(rest);
    } catch (err) {
      error(err instanceof Error ? err.message : String(err));
      printUsage(log);
      exit(1);
      return;
    }

    const stagedLabelReportPath = parsed.stagedLabelReportPath
      ?? toReplayLabelBatchReportPath(parsed.stagedLabeledPath);
    const stagedBenchmarkReportPath = parsed.stagedBenchmarkReportPath
      ?? toReplayLabelBenchmarkReportPath(parsed.stagedLabeledPath);
    const receiptPath = parsed.receiptPath ?? toReplayLabelPromotionReceiptPath(parsed.promotedLabeledPath);
    const promoteArtifacts = options.promoteArtifacts ?? promoteReplayLabelArtifacts;

    try {
      const result = promoteArtifacts({
        stagedLabeledPath: parsed.stagedLabeledPath,
        stagedLabelReportPath,
        stagedBenchmarkReportPath,
        promotedLabeledPath: parsed.promotedLabeledPath,
        receiptPath,
      });

      const summary: ReplayLabelPromoteSummaryOutput = {
        status: 'ok',
        stagedLabeledPath: result.receipt.source.stagedLabeledPath,
        stagedLabelReportPath: result.receipt.source.stagedLabelReportPath,
        stagedBenchmarkReportPath: result.receipt.source.stagedBenchmarkReportPath,
        promotedLabeledPath: result.receipt.target.promotedLabeledPath,
        promotedLabelReportPath: result.receipt.target.promotedLabelReportPath,
        promotedBenchmarkReportPath: result.receipt.target.promotedBenchmarkReportPath,
        receiptPath: result.receipt.receiptPath,
        promotedAt: result.receipt.promotedAt,
        bundleCount: result.receipt.bundleCount,
        labeledBundleCount: result.receipt.labeledBundleCount,
      };

      log(JSON.stringify(summary, null, 2));
    } catch (err) {
      error(err instanceof Error ? err.message : String(err));
      exit(1);
    }

    return;
  }

  if (subCmd === 'readiness') {
    let parsed: ParsedReadinessArgs;
    try {
      parsed = parseReadinessArgs(rest);
    } catch (err) {
      error(err instanceof Error ? err.message : String(err));
      printUsage(log);
      exit(1);
      return;
    }

    const runReadiness = options.runReadiness ?? runReplayLabelReadinessFromFile;

    try {
      const report = runReadiness({
        inputPath: parsed.inputPath,
        thresholds: parsed.thresholds,
      });

      const summary: ReplayLabelReadinessSummaryOutput = {
        status: 'ok',
        inputPath: parsed.inputPath,
        report,
      };

      log(JSON.stringify(summary, null, 2));
    } catch (err) {
      error(err instanceof Error ? err.message : String(err));
      exit(1);
    }

    return;
  }

  error(`replay-label: Unknown subcommand "${subCmd}"`);
  printUsage(log);
  exit(1);
}
