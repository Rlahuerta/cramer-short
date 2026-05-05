import {
  DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH,
  readArbiterReplayBundles,
  type ArbiterReplayBundle,
} from '../arbiter-replay.js';
import {
  runArbiterReplay,
  type ArbiterReplayEvaluator,
} from './arbiter-replay-runner.js';

const SHORT_HORIZON_DAYS = [1, 2, 3] as const;

type ShortHorizonDays = typeof SHORT_HORIZON_DAYS[number];
type ShortHorizonKey = `${ShortHorizonDays}d`;

export interface ShortHorizonReplayBenchmarkSlice {
  horizonDays: ShortHorizonDays;
  bundleCount: number;
  labeledBundleCount: number;
  unlabeledBundleCount: number;
  tradedRowCount: number;
  abstainRate: number;
  directionalAccuracy: number | null;
  brierScore: number | null;
  evaluatorName: string;
  ready: boolean;
  pendingReasons: string[];
}

export interface ShortHorizonReplayBenchmarkReport {
  formatVersion: 'polymarket-short-horizon-benchmark.v1';
  generatedAt: string;
  sourcePath: string;
  totalBundleCount: number;
  shortHorizonBundleCount: number;
  shortHorizonLabeledBundleCount: number;
  horizons: Record<ShortHorizonKey, ShortHorizonReplayBenchmarkSlice>;
}

function horizonKey(horizonDays: ShortHorizonDays): ShortHorizonKey {
  return `${horizonDays}d`;
}

function summarizeHorizon(
  horizonDays: ShortHorizonDays,
  bundles: ArbiterReplayBundle[],
  evaluator?: ArbiterReplayEvaluator,
): ShortHorizonReplayBenchmarkSlice {
  const horizonBundles = bundles.filter((bundle) => bundle.horizonDays === horizonDays);
  const bundleCount = horizonBundles.length;
  const replay = runArbiterReplay({
    bundles: horizonBundles,
    ...(evaluator ? { baselineEvaluator: evaluator } : {}),
  });
  const pendingReasons: string[] = [];

  if (bundleCount === 0) {
    pendingReasons.push(`No ${horizonDays}d replay bundles were found in the benchmark source.`);
  }
  if (bundleCount > 0 && replay.labeledBundleCount === 0) {
    pendingReasons.push(`No labeled ${horizonDays}d replay bundles were found; unlabeled bundles are not accuracy proof.`);
  }
  if (replay.labeledBundleCount > 0 && replay.baseline.tradedRows === 0) {
    pendingReasons.push(`Evaluator abstained on every labeled ${horizonDays}d replay bundle, so directional accuracy is undefined.`);
  }

  return {
    horizonDays,
    bundleCount,
    labeledBundleCount: replay.labeledBundleCount,
    unlabeledBundleCount: bundleCount - replay.labeledBundleCount,
    tradedRowCount: replay.baseline.tradedRows,
    abstainRate: replay.baseline.abstainRate,
    directionalAccuracy: replay.baseline.directionalAccuracy,
    brierScore: replay.baseline.brierScore,
    evaluatorName: replay.baseline.name,
    ready: pendingReasons.length === 0,
    pendingReasons,
  };
}

export function runShortHorizonReplayBenchmark(params: {
  bundles: ArbiterReplayBundle[];
  evaluator?: ArbiterReplayEvaluator;
  generatedAt?: string;
  sourcePath?: string;
}): ShortHorizonReplayBenchmarkReport {
  const horizons = Object.fromEntries(
    SHORT_HORIZON_DAYS.map((horizonDays) => [
      horizonKey(horizonDays),
      summarizeHorizon(horizonDays, params.bundles, params.evaluator),
    ]),
  ) as Record<ShortHorizonKey, ShortHorizonReplayBenchmarkSlice>;
  const shortHorizonBundleCount = SHORT_HORIZON_DAYS.reduce(
    (sum, horizonDays) => sum + horizons[horizonKey(horizonDays)].bundleCount,
    0,
  );
  const shortHorizonLabeledBundleCount = SHORT_HORIZON_DAYS.reduce(
    (sum, horizonDays) => sum + horizons[horizonKey(horizonDays)].labeledBundleCount,
    0,
  );

  return {
    formatVersion: 'polymarket-short-horizon-benchmark.v1',
    generatedAt: params.generatedAt ?? new Date().toISOString(),
    sourcePath: params.sourcePath ?? DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH,
    totalBundleCount: params.bundles.length,
    shortHorizonBundleCount,
    shortHorizonLabeledBundleCount,
    horizons,
  };
}

export function runShortHorizonReplayBenchmarkFromFile(params: {
  bundlePath?: string;
  evaluator?: ArbiterReplayEvaluator;
  generatedAt?: string;
} = {}): ShortHorizonReplayBenchmarkReport {
  const bundlePath = params.bundlePath ?? DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH;
  return runShortHorizonReplayBenchmark({
    bundles: readArbiterReplayBundles(bundlePath),
    evaluator: params.evaluator,
    generatedAt: params.generatedAt,
    sourcePath: bundlePath,
  });
}

export function formatShortHorizonReplayBenchmarkReport(
  report: ShortHorizonReplayBenchmarkReport,
): string {
  return JSON.stringify(report, null, 2);
}

function parseCliArgs(argv: string[]): { bundlePath: string } {
  let bundlePath = DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH;

  for (let i = 0; i < argv.length; i++) {
    const arg = argv[i];
    if (arg === '--bundle-path') {
      const value = argv[i + 1];
      if (!value) {
        throw new Error('Missing value for --bundle-path');
      }
      if (value.startsWith('-')) {
        throw new Error(`Invalid value for --bundle-path: expected a path, received flag ${value}`);
      }
      bundlePath = value;
      i++;
      continue;
    }
    if (arg === '--help' || arg === '-h') {
      console.log('Usage: bun run src/tools/finance/backtest/polymarket-short-horizon-benchmark.ts [--bundle-path PATH]');
      process.exit(0);
    }
    throw new Error(`Unknown argument: ${arg}`);
  }

  return { bundlePath };
}

if (import.meta.main) {
  try {
    const args = parseCliArgs(process.argv.slice(2));
    const report = runShortHorizonReplayBenchmarkFromFile({
      bundlePath: args.bundlePath,
    });
    console.log(formatShortHorizonReplayBenchmarkReport(report));
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error(message);
    process.exit(1);
  }
}
