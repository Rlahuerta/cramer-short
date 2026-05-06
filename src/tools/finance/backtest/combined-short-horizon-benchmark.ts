import { resolveAssetIntent } from '../asset-resolver.js';
import { readArbiterReplayBundles, type ArbiterReplayBundle } from '../arbiter-replay.js';
import { arbitrateForecast, type ForecastArbiterInput, type ForecastArbiterResult, type ForecastMarketSemantics } from '../forecast-arbitrator.js';
import { getBtcShortHorizonLivePolicy } from '../markov-distribution.js';
import { runArbiterReplay, type ArbiterReplayEvaluator } from './arbiter-replay-runner.js';
import { brierScore, ciCoverage, directionalAccuracy } from './metrics.js';
import { walkForward } from './walk-forward.js';

export const COMBINED_SHORT_HORIZON_DAYS = [1, 2, 3, 7, 14] as const;

type CombinedShortHorizonDays = typeof COMBINED_SHORT_HORIZON_DAYS[number];
type CombinedShortHorizonKey = `${CombinedShortHorizonDays}d`;
type BenchmarkSliceKind = 'markov-only' | 'polymarket-only' | 'combined/arbitrated';
type BenchmarkDirection = ForecastArbiterResult['preferredDirection'];

export interface CombinedShortHorizonBenchmarkSlice {
  slice: BenchmarkSliceKind;
  horizonDays: CombinedShortHorizonDays;
  observationCount: number;
  labeledObservationCount: number | null;
  tradedCount: number | null;
  abstainCount: number | null;
  directionalAccuracy: number | null;
  brierScore: number | null;
  ciCoverage: number | null;
  structuralBreakCount: number | null;
  evaluatorName: string | null;
  ready: boolean;
  pendingReasons: string[];
}

export interface CombinedShortHorizonBenchmarkHorizonReport {
  markovOnly: CombinedShortHorizonBenchmarkSlice;
  polymarketOnly: CombinedShortHorizonBenchmarkSlice;
  combinedArbitrated: CombinedShortHorizonBenchmarkSlice;
}

export interface CombinedShortHorizonBenchmarkReport {
  formatVersion: 'combined-short-horizon-benchmark.v1';
  generatedAt: string;
  ticker: string;
  resolvedTicker: string;
  replaySourcePath: string | null;
  horizons: Record<CombinedShortHorizonKey, CombinedShortHorizonBenchmarkHorizonReport>;
}

function horizonKey(horizonDays: CombinedShortHorizonDays): CombinedShortHorizonKey {
  return `${horizonDays}d`;
}

function isFiniteNumber(value: unknown): value is number {
  return typeof value === 'number' && Number.isFinite(value);
}

function directionFromReturn(value: number | undefined, threshold = 0.001): BenchmarkDirection {
  if (!isFiniteNumber(value) || Math.abs(value) < threshold) return 'neutral';
  return value > 0 ? 'long' : 'short';
}

function confidenceFromScore(score: number | undefined): ForecastArbiterResult['confidence'] {
  if (isFiniteNumber(score) && score >= 0.75) return 'high';
  if (isFiniteNumber(score) && score >= 0.6) return 'medium';
  return 'low';
}

function summarizeSemantics(markets: NonNullable<ForecastArbiterInput['polymarket']>['markets']) {
  const counts: Record<ForecastMarketSemantics, number> = {
    terminal: 0,
    barrier_touch: 0,
    range: 0,
    path_dependent: 0,
    unknown: 0,
  };

  for (const market of markets ?? []) {
    const semantics = market.semantics ?? 'unknown';
    counts[semantics] += 1;
  }

  const primary = (Object.entries(counts).sort((a, b) => b[1] - a[1])[0]?.[0] as ForecastMarketSemantics | undefined) ?? 'unknown';
  const barrierPrices = (markets ?? [])
    .filter((market) => market.semantics === 'barrier_touch' && isFiniteNumber(market.price))
    .map((market) => market.price!) as number[];

  return {
    primary,
    counts,
    barrierPrices,
  };
}

function buildReplayResult(params: {
  input: ForecastArbiterInput;
  preferredDirection: BenchmarkDirection;
  confidence: ForecastArbiterResult['confidence'];
  shouldEnterNow: boolean;
  name: string;
}): ForecastArbiterResult {
  const markets = params.input.polymarket?.markets ?? [];
  const semanticSummary = summarizeSemantics(markets);
  const markovDirection = directionFromReturn(params.input.markov?.forecast_return);
  const polymarketDirection = directionFromReturn(params.input.polymarket?.forecast_return);
  const whaleDirection = params.input.whale?.direction ?? 'neutral';
  const divergent =
    markovDirection !== 'neutral'
    && polymarketDirection !== 'neutral'
    && markovDirection !== polymarketDirection;

  return {
    ticker: params.input.ticker,
    horizonDays: params.input.horizon_days,
    currentPrice: params.input.current_price ?? null,
    leverage: params.input.leverage ?? 1,
    verdict: params.shouldEnterNow
      ? params.preferredDirection === 'long'
        ? 'LONG'
        : params.preferredDirection === 'short'
          ? 'SHORT'
          : 'NO_TRADE'
      : 'NO_TRADE',
    preferredDirection: params.preferredDirection,
    confidence: params.confidence,
    shouldEnterNow: params.shouldEnterNow,
    semanticSummary: {
      primaryPolymarketSemantics: semanticSummary.primary,
      counts: semanticSummary.counts,
      barrierPrices: semanticSummary.barrierPrices,
      reconciliation: `${params.name} replay benchmark slice.`,
    },
    disagreement: {
      markovDirection,
      polymarketDirection,
      whaleDirection,
      isDivergent: divergent,
      summary: `${params.name} replay benchmark disagreement snapshot.`,
    },
    leverageAssessment: {
      long: { directionalEdgePct: 0, riskAdjustedScore: 0, leveragePnlPct: 0, rr: null, notes: [] },
      short: { directionalEdgePct: 0, riskAdjustedScore: 0, leveragePnlPct: 0, rr: null, notes: [] },
      warning: null,
    },
    conditionalPlan: {
      longTrigger: null,
      shortTrigger: null,
      invalidation: null,
    },
    policy: {
      level: params.shouldEnterNow ? 'full' : markets.length > 0 ? 'context-only' : 'abstain',
      horizonEligible: true,
      tradeEligible: params.shouldEnterNow,
      reasons: [],
    },
    rationale: [],
    rawEvidence: {
      markov: params.input.markov ?? null,
      polymarket: params.input.polymarket ?? null,
      whale: params.input.whale ?? null,
    },
  };
}

function makePolymarketOnlyEvaluator(): ArbiterReplayEvaluator {
  return {
    name: 'polymarket-only',
    evaluate(input) {
      const direction = directionFromReturn(input.polymarket?.forecast_return);
      const confidenceScore = input.polymarket?.confidence
        ?? (isFiniteNumber(input.polymarket?.quality_score) ? input.polymarket!.quality_score! / 100 : undefined);
      const shouldEnterNow = direction !== 'neutral' && (input.polymarket?.markets?.length ?? 0) > 0;

      return buildReplayResult({
        input,
        preferredDirection: shouldEnterNow ? direction : 'neutral',
        confidence: confidenceFromScore(confidenceScore),
        shouldEnterNow,
        name: 'polymarket-only',
      });
    },
  };
}

function summarizeReplaySlice(params: {
  horizonDays: CombinedShortHorizonDays;
  bundles: ArbiterReplayBundle[];
  evaluator: ArbiterReplayEvaluator;
  slice: BenchmarkSliceKind;
  structuralBreakCount: number | null;
}): CombinedShortHorizonBenchmarkSlice {
  const replay = runArbiterReplay({
    bundles: params.bundles,
    baselineEvaluator: params.evaluator,
  });
  const pendingReasons: string[] = [];

  if (params.bundles.length === 0) {
    pendingReasons.push(`No ${params.horizonDays}d replay bundles were found in the benchmark source.`);
  }
  if (params.bundles.length > 0 && replay.labeledBundleCount === 0) {
    pendingReasons.push(`No labeled ${params.horizonDays}d replay bundles were found; unlabeled bundles are not accuracy proof.`);
  }
  if (replay.labeledBundleCount > 0 && replay.baseline.tradedRows === 0) {
    pendingReasons.push(`Evaluator abstained on every labeled ${params.horizonDays}d replay bundle, so directional accuracy is undefined.`);
  }

  return {
    slice: params.slice,
    horizonDays: params.horizonDays,
    observationCount: params.bundles.length,
    labeledObservationCount: replay.labeledBundleCount,
    tradedCount: replay.baseline.tradedRows,
    abstainCount: replay.baseline.totalRows - replay.baseline.tradedRows,
    directionalAccuracy: replay.baseline.directionalAccuracy,
    brierScore: replay.baseline.brierScore,
    ciCoverage: null,
    structuralBreakCount: params.structuralBreakCount,
    evaluatorName: replay.baseline.name,
    ready: pendingReasons.length === 0,
    pendingReasons,
  };
}

async function summarizeMarkovSlice(params: {
  ticker: string;
  prices: number[];
  horizonDays: CombinedShortHorizonDays;
  warmup: number;
  stride: number;
  btcBreakDivergenceThreshold?: number;
  postBreakShortWindow?: boolean;
  postBreakWindowSize?: number;
}): Promise<CombinedShortHorizonBenchmarkSlice> {
  const pendingReasons: string[] = [];
  if (params.prices.length < params.warmup + params.horizonDays + 1) {
    pendingReasons.push(`Not enough prices to compute a ${params.horizonDays}d Markov walk-forward slice.`);
    return {
      slice: 'markov-only',
      horizonDays: params.horizonDays,
      observationCount: 0,
      labeledObservationCount: null,
      tradedCount: null,
      abstainCount: 0,
      directionalAccuracy: null,
      brierScore: null,
      ciCoverage: null,
      structuralBreakCount: 0,
      evaluatorName: null,
      ready: false,
      pendingReasons,
    };
  }

  const result = await walkForward({
    ticker: params.ticker,
    prices: params.prices,
    horizon: params.horizonDays,
    warmup: params.warmup,
    stride: params.stride,
    ...(params.btcBreakDivergenceThreshold !== undefined
      ? { btcBreakDivergenceThreshold: params.btcBreakDivergenceThreshold }
      : {}),
    ...(params.postBreakShortWindow !== undefined
      ? { postBreakShortWindow: params.postBreakShortWindow }
      : {}),
    ...(params.postBreakWindowSize !== undefined
      ? { postBreakWindowSize: params.postBreakWindowSize }
      : {}),
  });

  if (result.errors.length > 0) {
    pendingReasons.push(`${result.errors.length} walk-forward errors were recorded for ${params.horizonDays}d.`);
  }
  if (result.steps.length === 0) {
    pendingReasons.push(`No Markov walk-forward steps were produced for ${params.horizonDays}d.`);
  }

  const structuralBreakCount = result.steps.filter((step) => (step.originalStructuralBreakDetected ?? step.structuralBreakDetected) === true).length;
  const abstainCount = result.steps.filter((step) => step.recommendation === 'HOLD').length;

  return {
    slice: 'markov-only',
    horizonDays: params.horizonDays,
    observationCount: result.steps.length,
    labeledObservationCount: null,
    tradedCount: null,
    abstainCount,
    directionalAccuracy: result.steps.length > 0 ? directionalAccuracy(result.steps) : null,
    brierScore: result.steps.length > 0 ? brierScore(result.steps) : null,
    ciCoverage: result.steps.length > 0 ? ciCoverage(result.steps) : null,
    structuralBreakCount,
    evaluatorName: null,
    ready: pendingReasons.length === 0,
    pendingReasons,
  };
}

function resolveBenchmarkTicker(ticker: string): string {
  return resolveAssetIntent(`${ticker} short-horizon benchmark`, ticker).resolvedTicker
    ?? ticker.trim().toUpperCase();
}

export async function runCombinedShortHorizonBenchmark(params: {
  ticker: string;
  prices: number[];
  replayBundles: ArbiterReplayBundle[];
  stride?: number;
  warmup?: number;
  generatedAt?: string;
  replaySourcePath?: string;
}): Promise<CombinedShortHorizonBenchmarkReport> {
  const resolvedTicker = resolveBenchmarkTicker(params.ticker);
  const stride = params.stride ?? 5;
  const horizons = {} as Record<CombinedShortHorizonKey, CombinedShortHorizonBenchmarkHorizonReport>;
  const polymarketEvaluator = makePolymarketOnlyEvaluator();

  for (const horizonDays of COMBINED_SHORT_HORIZON_DAYS) {
    const btcPolicy = getBtcShortHorizonLivePolicy(resolvedTicker, horizonDays);
    const horizonBundles = params.replayBundles.filter((bundle) =>
      bundle.ticker.trim().toUpperCase() === resolvedTicker.trim().toUpperCase() && bundle.horizonDays === horizonDays,
    );
    const bundleStructuralBreakCount = horizonBundles.filter((bundle) => bundle.markov?.structural_break === true).length;

    horizons[horizonKey(horizonDays)] = {
      markovOnly: await summarizeMarkovSlice({
        ticker: resolvedTicker,
        prices: params.prices,
        horizonDays,
        warmup: btcPolicy?.historyDays ?? params.warmup ?? 120,
        stride,
        ...(btcPolicy?.breakDivergenceThreshold !== undefined
          ? { btcBreakDivergenceThreshold: btcPolicy.breakDivergenceThreshold }
          : {}),
        ...(btcPolicy?.rerunOnBreak !== undefined
          ? { postBreakShortWindow: btcPolicy.rerunOnBreak }
          : {}),
        ...(btcPolicy?.rerunWindowDays !== undefined
          ? { postBreakWindowSize: btcPolicy.rerunWindowDays }
          : {}),
      }),
      polymarketOnly: summarizeReplaySlice({
        horizonDays,
        bundles: horizonBundles,
        evaluator: polymarketEvaluator,
        slice: 'polymarket-only',
        structuralBreakCount: null,
      }),
      combinedArbitrated: summarizeReplaySlice({
        horizonDays,
        bundles: horizonBundles,
        evaluator: {
          name: 'combined/arbitrated',
          evaluate: arbitrateForecast,
        },
        slice: 'combined/arbitrated',
        structuralBreakCount: bundleStructuralBreakCount,
      }),
    };
  }

  return {
    formatVersion: 'combined-short-horizon-benchmark.v1',
    generatedAt: params.generatedAt ?? new Date().toISOString(),
    ticker: params.ticker,
    resolvedTicker,
    replaySourcePath: params.replaySourcePath ?? null,
    horizons,
  };
}

export async function runCombinedShortHorizonBenchmarkFromFile(params: {
  ticker: string;
  prices: number[];
  bundlePath: string;
  stride?: number;
  warmup?: number;
  generatedAt?: string;
}): Promise<CombinedShortHorizonBenchmarkReport> {
  return runCombinedShortHorizonBenchmark({
    ticker: params.ticker,
    prices: params.prices,
    replayBundles: readArbiterReplayBundles(params.bundlePath),
    stride: params.stride,
    warmup: params.warmup,
    generatedAt: params.generatedAt,
    replaySourcePath: params.bundlePath,
  });
}

export function formatCombinedShortHorizonBenchmarkReport(
  report: CombinedShortHorizonBenchmarkReport,
): string {
  return JSON.stringify(report, null, 2);
}
