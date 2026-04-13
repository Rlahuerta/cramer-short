import { mkdirSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { walkForward, type WalkForwardResult } from './walk-forward.js';
import {
  brierScore,
  bootstrapDirectionalCI,
  ciCoverage,
  computeRCCurve,
  directionalAccuracy,
  type BacktestStep,
} from './metrics.js';
import { effectiveBreakContext } from './break-policy-ablation.js';

interface FixtureTickerData {
  closes: number[];
  dates: string[];
}

interface FixtureData {
  tickers: Record<string, FixtureTickerData>;
}

interface MetricBlock {
  directionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
  avgConfidence: number;
  directionalCi: { lower: number; median: number; upper: number; nResamples: number };
}

interface RCPoint {
  threshold: number;
  accuracy: number;
  coverage: number;
  n: number;
}

interface ArmSummary {
  label: string;
  totalSteps: number;
  breakSteps: number;
  breakTrendingSteps: number;
  breakChopSteps: number;
  overall: MetricBlock;
  breakContext: MetricBlock;
  nonBreak: MetricBlock;
  breakTrending: MetricBlock;
  breakChop: MetricBlock;
  overallRC: RCPoint[];
  breakContextRC: RCPoint[];
}

interface DeltaSummary {
  changedStepCount: number;
  changedBreakChopCount: number;
  changedBreakTrendingCount: number;
  overallRC020AccuracyDelta: number;
  overallRC020CoverageDelta: number;
  overallRC030AccuracyDelta: number;
  overallRC030CoverageDelta: number;
  breakContextRC020AccuracyDelta: number;
  breakContextRC020CoverageDelta: number;
  breakContextRC030AccuracyDelta: number;
  breakContextRC030CoverageDelta: number;
}

interface StepDiff {
  ticker: string;
  horizon: number;
  t: number;
  regime: string | undefined;
  breakDetected: boolean;
  baselineConfidence: number;
  experimentConfidence: number;
  confidenceDelta: number;
}

export interface Phase4ComparisonArtifact {
  generatedAt: string;
  universe: {
    tickers: string[];
    horizons: number[];
    warmup: number;
    stride: number;
  };
  baseline: ArmSummary;
  experiment: ArmSummary;
  delta: DeltaSummary;
  perStep: StepDiff[];
}

export const TICKERS = ['SPY', 'QQQ', 'GLD', 'VOO', 'NVDA', 'MSFT'] as const;
export const HORIZONS = [7, 14, 30] as const;
export const WARMUP = 120;
export const STRIDE = 5;
export const THRESHOLDS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5] as const;

function averageConfidence(steps: BacktestStep[]): number {
  return steps.length > 0
    ? steps.reduce((sum, step) => sum + step.confidence, 0) / steps.length
    : 0;
}

function summarizePartition(steps: BacktestStep[]): MetricBlock {
  return {
    directionalAccuracy: steps.length > 0 ? directionalAccuracy(steps) : 0,
    brierScore: steps.length > 0 ? brierScore(steps) : 0,
    ciCoverage: steps.length > 0 ? ciCoverage(steps) : 0,
    avgConfidence: averageConfidence(steps),
    directionalCi: bootstrapDirectionalCI(steps, 500, 12345),
  };
}

function isTrending(step: BacktestStep): boolean {
  return step.regime === 'bull' || step.regime === 'bear';
}

function isChop(step: BacktestStep): boolean {
  return step.regime === 'sideways';
}

function summarizeArm(label: string, steps: BacktestStep[]): ArmSummary {
  const breakContext = steps.filter(step => effectiveBreakContext(step));
  const nonBreak = steps.filter(step => !effectiveBreakContext(step));
  const breakTrending = breakContext.filter(isTrending);
  const breakChop = breakContext.filter(isChop);

  return {
    label,
    totalSteps: steps.length,
    breakSteps: breakContext.length,
    breakTrendingSteps: breakTrending.length,
    breakChopSteps: breakChop.length,
    overall: summarizePartition(steps),
    breakContext: summarizePartition(breakContext),
    nonBreak: summarizePartition(nonBreak),
    breakTrending: summarizePartition(breakTrending),
    breakChop: summarizePartition(breakChop),
    overallRC: computeRCCurve(steps, [...THRESHOLDS]),
    breakContextRC: computeRCCurve(breakContext, [...THRESHOLDS]),
  };
}

function findPoint(points: RCPoint[], threshold: number): RCPoint {
  const found = points.find(point => point.threshold === threshold);
  if (!found) throw new Error(`Missing threshold ${threshold}`);
  return found;
}

function computeStepDiffs(
  baseline: Array<BacktestStep & { ticker: string; horizon: number }>,
  experiment: Array<BacktestStep & { ticker: string; horizon: number }>,
): StepDiff[] {
  if (baseline.length !== experiment.length) {
    throw new Error(`Step length mismatch: baseline=${baseline.length}, experiment=${experiment.length}`);
  }

  const diffs: StepDiff[] = [];
  for (let i = 0; i < baseline.length; i++) {
    const base = baseline[i];
    const exp = experiment[i];
    if (base.ticker !== exp.ticker || base.horizon !== exp.horizon || base.t !== exp.t) {
      throw new Error(
        `Step alignment mismatch at index ${i}: ${base.ticker}/${base.horizon}/${base.t} vs ${exp.ticker}/${exp.horizon}/${exp.t}`,
      );
    }

    const confidenceDelta = exp.confidence - base.confidence;
    if (Math.abs(confidenceDelta) <= 1e-9) continue;

    diffs.push({
      ticker: base.ticker,
      horizon: base.horizon,
      t: base.t,
      regime: base.regime,
      breakDetected: effectiveBreakContext(base),
      baselineConfidence: base.confidence,
      experimentConfidence: exp.confidence,
      confidenceDelta,
    });
  }

  return diffs;
}

async function loadArmSteps(options: {
  fixturePath: string;
  trendPenaltyOnlyBreakConfidence?: boolean;
}): Promise<Array<BacktestStep & { ticker: string; horizon: number }>> {
  const fixture = JSON.parse(readFileSync(options.fixturePath, 'utf-8')) as FixtureData;
  const allSteps: Array<BacktestStep & { ticker: string; horizon: number }> = [];

  for (const ticker of TICKERS) {
    const tickerData = fixture.tickers[ticker];
    if (!tickerData || tickerData.closes.length === 0) {
      throw new Error(`Missing fixture data for ${ticker}`);
    }

    for (const horizon of HORIZONS) {
      const result: WalkForwardResult = await walkForward({
        ticker,
        prices: tickerData.closes,
        horizon,
        warmup: WARMUP,
        stride: STRIDE,
        trendPenaltyOnlyBreakConfidence: options.trendPenaltyOnlyBreakConfidence,
      });

      if (result.errors.length > 0) {
        throw new Error(`Walk-forward produced errors for ${ticker} ${horizon}d`);
      }

      allSteps.push(...result.steps.map(step => ({ ...step, ticker, horizon })));
    }
  }

  return allSteps;
}

export async function runComparison(
  fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json'),
): Promise<Phase4ComparisonArtifact> {
  const baselineSteps = await loadArmSteps({ fixturePath });
  const experimentSteps = await loadArmSteps({
    fixturePath,
    trendPenaltyOnlyBreakConfidence: true,
  });

  const perStep = computeStepDiffs(baselineSteps, experimentSteps);
  const baseline = summarizeArm('baseline', baselineSteps);
  const experiment = summarizeArm('trendPenaltyOnlyBreakConfidence=true', experimentSteps);

  const overall020Base = findPoint(baseline.overallRC, 0.2);
  const overall020Exp = findPoint(experiment.overallRC, 0.2);
  const overall030Base = findPoint(baseline.overallRC, 0.3);
  const overall030Exp = findPoint(experiment.overallRC, 0.3);
  const break020Base = findPoint(baseline.breakContextRC, 0.2);
  const break020Exp = findPoint(experiment.breakContextRC, 0.2);
  const break030Base = findPoint(baseline.breakContextRC, 0.3);
  const break030Exp = findPoint(experiment.breakContextRC, 0.3);

  const changedBreakChopCount = perStep.filter(step => step.regime === 'sideways').length;
  const changedBreakTrendingCount = perStep.filter(step => step.regime === 'bull' || step.regime === 'bear').length;

  return {
    generatedAt: new Date().toISOString(),
    universe: {
      tickers: [...TICKERS],
      horizons: [...HORIZONS],
      warmup: WARMUP,
      stride: STRIDE,
    },
    baseline,
    experiment,
    delta: {
      changedStepCount: perStep.length,
      changedBreakChopCount,
      changedBreakTrendingCount,
      overallRC020AccuracyDelta: overall020Exp.accuracy - overall020Base.accuracy,
      overallRC020CoverageDelta: overall020Exp.coverage - overall020Base.coverage,
      overallRC030AccuracyDelta: overall030Exp.accuracy - overall030Base.accuracy,
      overallRC030CoverageDelta: overall030Exp.coverage - overall030Base.coverage,
      breakContextRC020AccuracyDelta: break020Exp.accuracy - break020Base.accuracy,
      breakContextRC020CoverageDelta: break020Exp.coverage - break020Base.coverage,
      breakContextRC030AccuracyDelta: break030Exp.accuracy - break030Base.accuracy,
      breakContextRC030CoverageDelta: break030Exp.coverage - break030Base.coverage,
    },
    perStep,
  };
}

function formatPct(value: number): string {
  return `${(value * 100).toFixed(1)}%`;
}

function printSummary(artifact: Phase4ComparisonArtifact): void {
  const base020 = findPoint(artifact.baseline.overallRC, 0.2);
  const exp020 = findPoint(artifact.experiment.overallRC, 0.2);
  const base030 = findPoint(artifact.baseline.overallRC, 0.3);
  const exp030 = findPoint(artifact.experiment.overallRC, 0.3);
  const breakBase020 = findPoint(artifact.baseline.breakContextRC, 0.2);
  const breakExp020 = findPoint(artifact.experiment.breakContextRC, 0.2);
  const breakBase030 = findPoint(artifact.baseline.breakContextRC, 0.3);
  const breakExp030 = findPoint(artifact.experiment.breakContextRC, 0.3);

  console.log('Phase 4: trendPenaltyOnlyBreakConfidence Comparison');
  console.log('═══════════════════════════════════════════════════');
  console.log(`Universe: ${artifact.universe.tickers.join(', ')} | horizons=${artifact.universe.horizons.join(', ')}d`);
  console.log(
    `Total steps: ${artifact.baseline.totalSteps} | break-context: ${artifact.baseline.breakSteps} | break+chop: ${artifact.baseline.breakChopSteps} | break+trending: ${artifact.baseline.breakTrendingSteps}`,
  );
  console.log('');

  console.log('Overall:');
  console.log(`  baseline   rc@0.2=${formatPct(base020.accuracy)}/${formatPct(base020.coverage)}  rc@0.3=${formatPct(base030.accuracy)}/${formatPct(base030.coverage)}`);
  console.log(`  experiment rc@0.2=${formatPct(exp020.accuracy)}/${formatPct(exp020.coverage)}  rc@0.3=${formatPct(exp030.accuracy)}/${formatPct(exp030.coverage)}`);
  console.log('');

  console.log('Break-context:');
  console.log(`  baseline   rc@0.2=${formatPct(breakBase020.accuracy)}/${formatPct(breakBase020.coverage)}  rc@0.3=${formatPct(breakBase030.accuracy)}/${formatPct(breakBase030.coverage)}`);
  console.log(`  experiment rc@0.2=${formatPct(breakExp020.accuracy)}/${formatPct(breakExp020.coverage)}  rc@0.3=${formatPct(breakExp030.accuracy)}/${formatPct(breakExp030.coverage)}`);
  console.log('');

  console.log(
    `Changed steps: ${artifact.delta.changedStepCount} (break+chop: ${artifact.delta.changedBreakChopCount}, break+trending: ${artifact.delta.changedBreakTrendingCount})`,
  );
}

export async function main(): Promise<void> {
  const artifact = await runComparison();
  const artifactDir = join(process.cwd(), '.sisyphus', 'artifacts');
  const artifactPath = join(artifactDir, 'phase4-trend-penalty-comparison.json');

  mkdirSync(artifactDir, { recursive: true });
  writeFileSync(artifactPath, JSON.stringify(artifact, null, 2));

  printSummary(artifact);
  console.log('');
  console.log(`Saved artifact to ${artifactPath}`);
}

if (import.meta.main) {
  main().catch(error => {
    console.error('Fatal:', error);
    process.exit(1);
  });
}
