import { mkdirSync, readFileSync, writeFileSync } from 'fs';
import { join } from 'path';
import { walkForward, type WalkForwardResult } from './walk-forward.js';
import {
  brierScore,
  bootstrapDirectionalCI,
  bucketByConfidence,
  bucketByDivergence,
  bucketByEnsembleConsensus,
  bucketByMoveDirection,
  bucketByMoveMagnitude,
  bucketByRecommendation,
  bucketByRegime,
  bucketByTrendVsChop,
  ciCoverage,
  computeRCCurve,
  directionalAccuracy,
  selectivePUpAccuracy,
  selectiveRawPUpAccuracy,
  type BacktestStep,
} from './metrics.js';

interface FixtureTickerData {
  type: string;
  closes: number[];
  dates: string[];
  count: number;
  synthetic?: boolean;
}

interface FixtureData {
  generatedAt: string;
  startDate: string;
  endDate: string;
  tickers: Record<string, FixtureTickerData>;
}

interface MarketEvent {
  label: string;
  startDate: string;
  endDate: string;
  category: 'macro' | 'geopolitical' | 'earnings' | 'tariff' | 'market-stress' | 'other';
  notes: string;
  tickers?: string[];
}

interface BreakStepRecord {
  ticker: string;
  horizon: number;
  t: number;
  date: string;
  divergence: number;
  predictedProb: number;
  actualBinary: number;
  actualReturn: number;
  ciLower: number;
  ciUpper: number;
  realizedPrice: number;
  structuralBreakDetected: boolean;
  structuralBreakRerunTriggered?: boolean;
  originalStructuralBreakDetected?: boolean;
  originalStructuralBreakDivergence?: number | null;
  recommendation: BacktestStep['recommendation'];
  confidence: number;
}

interface BreakCluster {
  startDate: string;
  endDate: string;
  stepCount: number;
  meanDivergence: number;
  tickers: string[];
  horizons: number[];
  matchedEvents: string[];
  breakWindowAccuracy: number;
  breakWindowBrier: number;
}

interface PerBucketSummary {
  totalSteps: number;
  breakSteps: number;
  breakDirectionalAccuracy: number;
  nonBreakDirectionalAccuracy: number;
  breakBrierScore: number;
  nonBreakBrierScore: number;
  breakCiCoverage: number;
  nonBreakCiCoverage: number;
  errorCount: number;
}

interface WindowSummary {
  totalSteps: number;
  directionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
}

interface SelectiveSummaryPoint {
  threshold: number;
  accuracy: number;
  coverage: number;
  n: number;
}

interface SelectivePUpSummaryPoint {
  threshold: number;
  accuracy: number;
  coverage: number;
  selected: number;
  total: number;
}

interface ConditionalSubsetSummary {
  label: string;
  totalSteps: number;
  directionalAccuracy: number;
  brierScore: number;
  ciCoverage: number;
  rcCurve: SelectiveSummaryPoint[];
}

interface AnalysisArtifact {
  generatedAt: string;
  tickers: string[];
  horizons: number[];
  summary: {
    totalSteps: number;
    breakSteps: number;
    breakClusters: number;
    eventMatchedClusters: number;
    breakDirectionalAccuracy: number;
    nonBreakDirectionalAccuracy: number;
    breakBrierScore: number;
    nonBreakBrierScore: number;
    breakCiCoverage: number;
    nonBreakCiCoverage: number;
    breakDirectionalCi: { lower: number; median: number; upper: number; nResamples: number };
    nonBreakDirectionalCi: { lower: number; median: number; upper: number; nResamples: number };
    unmatchedEventRate: number;
    errorCount: number;
  };
  perTicker: Record<string, PerBucketSummary>;
  perHorizon: Record<string, PerBucketSummary>;
  byRegime: Array<{
    label: string;
    breakSteps: number;
    nonBreakSteps: number;
    breakDirectionalAccuracy: number;
    nonBreakDirectionalAccuracy: number;
    breakBrierScore: number;
    nonBreakBrierScore: number;
  }>;
  byConfidence: Array<{
    label: string;
    breakSteps: number;
    nonBreakSteps: number;
    breakDirectionalAccuracy: number;
    nonBreakDirectionalAccuracy: number;
    breakBrierScore: number;
    nonBreakBrierScore: number;
  }>;
  byDivergence: Array<{
    label: string;
    count: number;
    directionalAccuracy: number;
    brierScore: number;
    ciCoverage: number;
  }>;
  byMoveMagnitude: Array<{
    label: string;
    breakSteps: number;
    nonBreakSteps: number;
    breakDirectionalAccuracy: number;
    nonBreakDirectionalAccuracy: number;
    breakBrierScore: number;
    nonBreakBrierScore: number;
  }>;
  byMoveDirection: Array<{
    label: string;
    breakSteps: number;
    nonBreakSteps: number;
    breakDirectionalAccuracy: number;
    nonBreakDirectionalAccuracy: number;
    breakBrierScore: number;
    nonBreakBrierScore: number;
  }>;
  byRecommendation: Array<{
    label: string;
    breakSteps: number;
    nonBreakSteps: number;
    breakDirectionalAccuracy: number;
    nonBreakDirectionalAccuracy: number;
    breakBrierScore: number;
    nonBreakBrierScore: number;
  }>;
  byTrendVsChop: Array<{
    label: string;
    breakSteps: number;
    nonBreakSteps: number;
    breakDirectionalAccuracy: number;
    nonBreakDirectionalAccuracy: number;
    breakBrierScore: number;
    nonBreakBrierScore: number;
  }>;
  byEnsembleConsensus: Array<{
    label: string;
    breakSteps: number;
    nonBreakSteps: number;
    breakDirectionalAccuracy: number;
    nonBreakDirectionalAccuracy: number;
    breakBrierScore: number;
    nonBreakBrierScore: number;
  }>;
  byRerunTriggered: {
    rerunTriggered: PerBucketSummary;
    noRerunTriggerWithinBreaks: PerBucketSummary;
  };
  selectiveAccuracy: {
    breakRC: SelectiveSummaryPoint[];
    nonBreakRC: SelectiveSummaryPoint[];
    breakPUpRC: SelectivePUpSummaryPoint[];
    nonBreakPUpRC: SelectivePUpSummaryPoint[];
    breakRawPUpRC: SelectivePUpSummaryPoint[];
    nonBreakRawPUpRC: SelectivePUpSummaryPoint[];
  };
  conditionalBreakContexts: {
    breakTrending: ConditionalSubsetSummary;
    breakChop: ConditionalSubsetSummary;
    breakLargeMove: ConditionalSubsetSummary;
  };
  prePostBreakWindow: {
    preBreak: WindowSummary;
    atBreak: WindowSummary;
    postBreak: WindowSummary;
  };
  clusters: BreakCluster[];
}

const TICKERS = ['SPY', 'QQQ', 'GLD', 'VOO', 'NVDA', 'MSFT'] as const;
const HORIZONS = [7, 14, 30] as const;
const WARMUP = 120;
const STRIDE = 5;
const CLUSTER_GAP_DAYS = 15;
const MAX_CLUSTER_DURATION_DAYS = 60;

function dateDiffDays(a: string, b: string): number {
  const aMs = Date.parse(`${a}T00:00:00Z`);
  const bMs = Date.parse(`${b}T00:00:00Z`);
  return Math.round((bMs - aMs) / 86_400_000);
}

function rangesOverlap(startA: string, endA: string, startB: string, endB: string): boolean {
  return startA <= endB && endA >= startB;
}

function safeDirectionalAccuracy(steps: BacktestStep[]): number {
  return steps.length > 0 ? directionalAccuracy(steps) : 0;
}

function safeBrierScore(steps: BacktestStep[]): number {
  return steps.length > 0 ? brierScore(steps) : 0;
}

function safeCiCoverage(steps: BacktestStep[]): number {
  return steps.length > 0 ? ciCoverage(steps) : 0;
}

function summarizePartition(steps: BacktestStep[], errors: number): PerBucketSummary {
  const breakSteps = steps.filter(step =>
    step.structuralBreakDetected === true
    || step.originalStructuralBreakDetected === true,
  );
  const nonBreakSteps = steps.filter(step =>
    step.structuralBreakDetected !== true
    && step.originalStructuralBreakDetected !== true,
  );
  return {
    totalSteps: steps.length,
    breakSteps: breakSteps.length,
    breakDirectionalAccuracy: safeDirectionalAccuracy(breakSteps),
    nonBreakDirectionalAccuracy: safeDirectionalAccuracy(nonBreakSteps),
    breakBrierScore: safeBrierScore(breakSteps),
    nonBreakBrierScore: safeBrierScore(nonBreakSteps),
    breakCiCoverage: safeCiCoverage(breakSteps),
    nonBreakCiCoverage: safeCiCoverage(nonBreakSteps),
    errorCount: errors,
  };
}

function summarizeWindow(steps: BacktestStep[]): WindowSummary {
  return {
    totalSteps: steps.length,
    directionalAccuracy: safeDirectionalAccuracy(steps),
    brierScore: safeBrierScore(steps),
    ciCoverage: safeCiCoverage(steps),
  };
}

function summarizeConditionalSubset(
  label: string,
  steps: BacktestStep[],
): ConditionalSubsetSummary {
  return {
    label,
    totalSteps: steps.length,
    directionalAccuracy: safeDirectionalAccuracy(steps),
    brierScore: safeBrierScore(steps),
    ciCoverage: safeCiCoverage(steps),
    rcCurve: computeRCCurve(steps),
  };
}

function toSummaryRows(
  breakRows: Array<{ label: string; count: number; directionalAccuracy: number; brierScore: number }>,
  nonBreakRows: Array<{ label: string; count: number; directionalAccuracy: number; brierScore: number }>,
): Array<{
  label: string;
  breakSteps: number;
  nonBreakSteps: number;
  breakDirectionalAccuracy: number;
  nonBreakDirectionalAccuracy: number;
  breakBrierScore: number;
  nonBreakBrierScore: number;
}> {
  const labels = Array.from(new Set([...breakRows.map(row => row.label), ...nonBreakRows.map(row => row.label)]));
  return labels.map(label => {
    const breakRow = breakRows.find(row => row.label === label);
    const nonBreakRow = nonBreakRows.find(row => row.label === label);
    return {
      label,
      breakSteps: breakRow?.count ?? 0,
      nonBreakSteps: nonBreakRow?.count ?? 0,
      breakDirectionalAccuracy: breakRow?.directionalAccuracy ?? 0,
      nonBreakDirectionalAccuracy: nonBreakRow?.directionalAccuracy ?? 0,
      breakBrierScore: breakRow?.brierScore ?? 0,
      nonBreakBrierScore: nonBreakRow?.brierScore ?? 0,
    };
  });
}

function collectWindowSteps(
  stepsByTicker: Map<string, BacktestStep[]>,
  fixture: FixtureData,
  breakRecords: BreakStepRecord[],
  dayOffsetStart: number,
  dayOffsetEnd: number,
): BacktestStep[] {
  const seen = new Set<string>();
  const windowSteps: BacktestStep[] = [];

  for (const record of breakRecords) {
    const tickerSteps = stepsByTicker.get(record.ticker) ?? [];
    const tickerDates = fixture.tickers[record.ticker]?.dates ?? [];
    const breakDate = tickerDates[record.t] ?? record.date;
    for (const step of tickerSteps) {
      const stepDate = tickerDates[step.t];
      if (!stepDate) continue;
      const deltaDays = dateDiffDays(breakDate, stepDate);
      if (deltaDays < dayOffsetStart || deltaDays > dayOffsetEnd) continue;
      const key = `${record.ticker}:${step.t}`;
      if (seen.has(key)) continue;
      seen.add(key);
      windowSteps.push(step);
    }
  }

  return windowSteps;
}

function clusterBreakSteps(records: BreakStepRecord[]): BreakCluster[] {
  if (records.length === 0) return [];
  const sorted = [...records].sort((a, b) => a.date.localeCompare(b.date));
  const groups: BreakStepRecord[][] = [];
  let current: BreakStepRecord[] = [sorted[0]];

  for (let i = 1; i < sorted.length; i++) {
    const prev = current[current.length - 1];
    const next = sorted[i];
    const clusterDuration = dateDiffDays(current[0].date, next.date);
    if (dateDiffDays(prev.date, next.date) <= CLUSTER_GAP_DAYS && clusterDuration <= MAX_CLUSTER_DURATION_DAYS) {
      current.push(next);
    } else {
      groups.push(current);
      current = [next];
    }
  }
  groups.push(current);

  return groups.map(group => {
    const tickers = Array.from(new Set(group.map(item => item.ticker))).sort();
    const horizons = Array.from(new Set(group.map(item => item.horizon))).sort((a, b) => a - b);
    const meanDivergence = group.reduce((sum, item) => sum + item.divergence, 0) / group.length;
    const clusterSteps: BacktestStep[] = group.map(item => ({
      t: item.t,
      predictedProb: item.predictedProb,
      actualBinary: item.actualBinary,
      predictedReturn: 0,
      actualReturn: item.actualReturn,
      ciLower: item.ciLower,
      ciUpper: item.ciUpper,
      realizedPrice: item.realizedPrice,
      recommendation: item.recommendation,
      gofPasses: null,
      confidence: item.confidence,
      structuralBreakDetected: true,
    }));

    return {
      startDate: group[0].date,
      endDate: group[group.length - 1].date,
      stepCount: group.length,
      meanDivergence,
      tickers,
      horizons,
      matchedEvents: [],
      breakWindowAccuracy: safeDirectionalAccuracy(clusterSteps),
      breakWindowBrier: safeBrierScore(clusterSteps),
    } satisfies BreakCluster;
  });
}

function eventMatchesCluster(event: MarketEvent, cluster: BreakCluster): boolean {
  if (!rangesOverlap(cluster.startDate, cluster.endDate, event.startDate, event.endDate)) {
    return false;
  }
  if (event.category !== 'earnings') {
    return true;
  }
  if (!event.tickers || event.tickers.length === 0) {
    return false;
  }
  return event.tickers.some(ticker => cluster.tickers.includes(ticker));
}

async function main(): Promise<void> {
  const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
  const eventsPath = join(import.meta.dir, 'market-events.json');
  const artifactDir = join(process.cwd(), '.sisyphus', 'artifacts');
  const artifactPath = join(artifactDir, 'markov-structural-break-analysis.json');

  const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8')) as FixtureData;
  const events = JSON.parse(readFileSync(eventsPath, 'utf-8')) as MarketEvent[];

  const allSteps: BacktestStep[] = [];
  const stepsByTicker = new Map<string, BacktestStep[]>();
  const breakRecords: BreakStepRecord[] = [];
  const perTicker: Record<string, PerBucketSummary> = {};
  const perHorizonSteps = new Map<number, BacktestStep[]>();
  const perHorizonErrors = new Map<number, number>();
  let totalErrors = 0;

  console.log('Markov Structural-Break Analysis');
  console.log('═══════════════════════════════════════════════════');
  console.log(`Tickers: ${TICKERS.join(', ')}`);
  console.log(`Horizons: ${HORIZONS.map(h => `${h}d`).join(', ')}`);

  for (const ticker of TICKERS) {
    const tickerData = fixture.tickers[ticker];
    if (!tickerData || tickerData.closes.length === 0 || tickerData.dates.length !== tickerData.closes.length) {
      throw new Error(`Invalid fixture data for ${ticker}`);
    }

    const tickerSteps: BacktestStep[] = [];
    let tickerErrors = 0;

    for (const horizon of HORIZONS) {
      const result: WalkForwardResult = await walkForward({
        ticker,
        prices: tickerData.closes,
        horizon,
        warmup: WARMUP,
        stride: STRIDE,
      });

      tickerErrors += result.errors.length;
      totalErrors += result.errors.length;
      tickerSteps.push(...result.steps);
      allSteps.push(...result.steps);

      const existing = perHorizonSteps.get(horizon) ?? [];
      existing.push(...result.steps);
      perHorizonSteps.set(horizon, existing);
      perHorizonErrors.set(horizon, (perHorizonErrors.get(horizon) ?? 0) + result.errors.length);

      for (const step of result.steps) {
        if (step.structuralBreakDetected !== true && step.originalStructuralBreakDetected !== true) continue;
        const stepDate = tickerData.dates[step.t];
        if (!stepDate) continue;
        breakRecords.push({
          ticker,
          horizon,
          t: step.t,
          date: stepDate,
          divergence: step.originalStructuralBreakDivergence ?? step.structuralBreakDivergence ?? 0,
          predictedProb: step.predictedProb,
          actualBinary: step.actualBinary,
          actualReturn: step.actualReturn,
          ciLower: step.ciLower,
          ciUpper: step.ciUpper,
          realizedPrice: step.realizedPrice,
          structuralBreakDetected: step.structuralBreakDetected === true,
          structuralBreakRerunTriggered: step.structuralBreakRerunTriggered,
          originalStructuralBreakDetected: step.originalStructuralBreakDetected,
          originalStructuralBreakDivergence: step.originalStructuralBreakDivergence,
          recommendation: step.recommendation,
          confidence: step.confidence,
        });
      }
    }

    stepsByTicker.set(ticker, tickerSteps);
    perTicker[ticker] = summarizePartition(tickerSteps, tickerErrors);
  }

  const clusters = clusterBreakSteps(breakRecords).map(cluster => {
    const matchedEvents = events
      .filter(event => eventMatchesCluster(event, cluster))
      .map(event => event.label);
    return { ...cluster, matchedEvents };
  });

  const perHorizon: Record<string, PerBucketSummary> = {};
  for (const horizon of HORIZONS) {
    perHorizon[`${horizon}`] = summarizePartition(
      perHorizonSteps.get(horizon) ?? [],
      perHorizonErrors.get(horizon) ?? 0,
    );
  }

  const breakSteps = allSteps.filter(step =>
    step.structuralBreakDetected === true
    || step.originalStructuralBreakDetected === true,
  );
  const nonBreakSteps = allSteps.filter(step =>
    step.structuralBreakDetected !== true
    && step.originalStructuralBreakDetected !== true,
  );
  const matchedClusters = clusters.filter(cluster => cluster.matchedEvents.length > 0).length;
  const unmatchedClusters = clusters.length - matchedClusters;
  const breakDirectionalCi = bootstrapDirectionalCI(breakSteps, 1000, 12345);
  const nonBreakDirectionalCi = bootstrapDirectionalCI(nonBreakSteps, 1000, 54321);

  const byRegime = toSummaryRows(
    bucketByRegime(breakSteps).filter(row => row.count > 0),
    bucketByRegime(nonBreakSteps).filter(row => row.count > 0),
  );
  const byConfidence = toSummaryRows(
    bucketByConfidence(breakSteps).filter(row => row.count > 0),
    bucketByConfidence(nonBreakSteps).filter(row => row.count > 0),
  );
  const byDivergence = bucketByDivergence(breakSteps)
    .filter(row => row.count > 0)
    .map(row => ({
      label: row.label,
      count: row.count,
      directionalAccuracy: row.directionalAccuracy,
      brierScore: row.brierScore,
      ciCoverage: row.ciCoverage,
    }));
  const byMoveMagnitude = toSummaryRows(
    bucketByMoveMagnitude(breakSteps).filter(row => row.count > 0),
    bucketByMoveMagnitude(nonBreakSteps).filter(row => row.count > 0),
  );
  const byMoveDirection = toSummaryRows(
    bucketByMoveDirection(breakSteps).filter(row => row.count > 0),
    bucketByMoveDirection(nonBreakSteps).filter(row => row.count > 0),
  );
  const byRecommendation = toSummaryRows(
    bucketByRecommendation(breakSteps).filter(row => row.count > 0),
    bucketByRecommendation(nonBreakSteps).filter(row => row.count > 0),
  );
  const byTrendVsChop = toSummaryRows(
    bucketByTrendVsChop(breakSteps).filter(row => row.count > 0),
    bucketByTrendVsChop(nonBreakSteps).filter(row => row.count > 0),
  );
  const byEnsembleConsensus = toSummaryRows(
    bucketByEnsembleConsensus(breakSteps).filter(row => row.count > 0),
    bucketByEnsembleConsensus(nonBreakSteps).filter(row => row.count > 0),
  );

  const rerunTriggeredSteps = breakSteps.filter(step => step.structuralBreakRerunTriggered === true);
  const noRerunTriggerWithinBreaks = breakSteps.filter(step => step.structuralBreakRerunTriggered !== true);
  const breakTrendingSteps = breakSteps.filter(step => step.regime === 'bull' || step.regime === 'bear');
  const breakChopSteps = breakSteps.filter(step => step.regime === 'sideways');
  const breakLargeMoveSteps = breakSteps.filter(step => Math.abs(step.actualReturn) >= 0.10);

  const rcThresholds = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5];
  const selectiveAccuracy = {
    breakRC: computeRCCurve(breakSteps, rcThresholds),
    nonBreakRC: computeRCCurve(nonBreakSteps, rcThresholds),
    breakPUpRC: rcThresholds.map(threshold => ({ threshold, ...selectivePUpAccuracy(breakSteps, threshold) })),
    nonBreakPUpRC: rcThresholds.map(threshold => ({ threshold, ...selectivePUpAccuracy(nonBreakSteps, threshold) })),
    breakRawPUpRC: rcThresholds.map(threshold => ({ threshold, ...selectiveRawPUpAccuracy(breakSteps, threshold) })),
    nonBreakRawPUpRC: rcThresholds.map(threshold => ({ threshold, ...selectiveRawPUpAccuracy(nonBreakSteps, threshold) })),
  };

  const preBreakSteps = collectWindowSteps(stepsByTicker, fixture, breakRecords, -10, -5);
  const atBreakSteps = collectWindowSteps(stepsByTicker, fixture, breakRecords, 0, 4);
  const postBreakSteps = collectWindowSteps(stepsByTicker, fixture, breakRecords, 5, 10);

  const artifact: AnalysisArtifact = {
    generatedAt: new Date().toISOString(),
    tickers: [...TICKERS],
    horizons: [...HORIZONS],
    summary: {
      totalSteps: allSteps.length,
      breakSteps: breakSteps.length,
      breakClusters: clusters.length,
      eventMatchedClusters: matchedClusters,
      breakDirectionalAccuracy: safeDirectionalAccuracy(breakSteps),
      nonBreakDirectionalAccuracy: safeDirectionalAccuracy(nonBreakSteps),
      breakBrierScore: safeBrierScore(breakSteps),
      nonBreakBrierScore: safeBrierScore(nonBreakSteps),
      breakCiCoverage: safeCiCoverage(breakSteps),
      nonBreakCiCoverage: safeCiCoverage(nonBreakSteps),
      breakDirectionalCi,
      nonBreakDirectionalCi,
      unmatchedEventRate: clusters.length > 0 ? unmatchedClusters / clusters.length : 0,
      errorCount: totalErrors,
    },
    perTicker,
    perHorizon,
    byRegime,
    byConfidence,
    byDivergence,
    byMoveMagnitude,
    byMoveDirection,
    byRecommendation,
    byTrendVsChop,
    byEnsembleConsensus,
    byRerunTriggered: {
      rerunTriggered: summarizePartition(rerunTriggeredSteps, 0),
      noRerunTriggerWithinBreaks: summarizePartition(noRerunTriggerWithinBreaks, 0),
    },
    selectiveAccuracy,
    conditionalBreakContexts: {
      breakTrending: summarizeConditionalSubset('break+trending', breakTrendingSteps),
      breakChop: summarizeConditionalSubset('break+chop', breakChopSteps),
      breakLargeMove: summarizeConditionalSubset('break+large-move', breakLargeMoveSteps),
    },
    prePostBreakWindow: {
      preBreak: summarizeWindow(preBreakSteps),
      atBreak: summarizeWindow(atBreakSteps),
      postBreak: summarizeWindow(postBreakSteps),
    },
    clusters,
  };

  console.log(`Total steps: ${artifact.summary.totalSteps}  |  Break-context steps: ${artifact.summary.breakSteps}`);
  console.log(`Break clusters: ${artifact.summary.breakClusters}  |  Event-matched: ${artifact.summary.eventMatchedClusters}`);
  console.log(`Break-context directional accuracy: ${(artifact.summary.breakDirectionalAccuracy * 100).toFixed(1)}%`);
  console.log(`Break-context directional uncertainty band: ${(artifact.summary.breakDirectionalCi.lower * 100).toFixed(1)}% – ${(artifact.summary.breakDirectionalCi.upper * 100).toFixed(1)}%`);
  console.log(`Non-break directional accuracy: ${(artifact.summary.nonBreakDirectionalAccuracy * 100).toFixed(1)}%`);
  console.log(`Non-break directional uncertainty band: ${(artifact.summary.nonBreakDirectionalCi.lower * 100).toFixed(1)}% – ${(artifact.summary.nonBreakDirectionalCi.upper * 100).toFixed(1)}%`);
  console.log(`Break-context Brier: ${artifact.summary.breakBrierScore.toFixed(3)}`);
  console.log(`Non-break Brier: ${artifact.summary.nonBreakBrierScore.toFixed(3)}`);
  console.log(`Unmatched event rate: ${(artifact.summary.unmatchedEventRate * 100).toFixed(1)}%`);
  console.log('');
  console.log('Rerun-trigger diagnosis:');
  console.log(`  rerun-triggered break-context dir=${(artifact.byRerunTriggered.rerunTriggered.breakDirectionalAccuracy * 100).toFixed(1)}% | n=${artifact.byRerunTriggered.rerunTriggered.breakSteps}`);
  console.log(`  non-rerun break-context dir=${(artifact.byRerunTriggered.noRerunTriggerWithinBreaks.breakDirectionalAccuracy * 100).toFixed(1)}% | n=${artifact.byRerunTriggered.noRerunTriggerWithinBreaks.breakSteps}`);
  console.log('');
  console.log('Selective accuracy by break-context status (recommendation-based confidence filter):');
  for (const point of artifact.selectiveAccuracy.breakRC) {
    const nonBreak = artifact.selectiveAccuracy.nonBreakRC.find(item => item.threshold === point.threshold);
    console.log(`  conf>=${point.threshold.toFixed(1)} | break-context acc=${(point.accuracy * 100).toFixed(1).padStart(5)}% cov=${(point.coverage * 100).toFixed(1).padStart(5)}% n=${String(point.n).padStart(4)} | non-break acc=${((nonBreak?.accuracy ?? 0) * 100).toFixed(1).padStart(5)}% cov=${((nonBreak?.coverage ?? 0) * 100).toFixed(1).padStart(5)}% n=${String(nonBreak?.n ?? 0).padStart(4)}`);
  }
  console.log('');
  console.log('Selective accuracy by break-context status (P(up)-based, same confidence filter):');
  for (const point of artifact.selectiveAccuracy.breakPUpRC) {
    const nonBreak = artifact.selectiveAccuracy.nonBreakPUpRC.find(item => item.threshold === point.threshold);
    console.log(`  conf>=${point.threshold.toFixed(1)} | break-context acc=${(point.accuracy * 100).toFixed(1).padStart(5)}% cov=${(point.coverage * 100).toFixed(1).padStart(5)}% sel=${String(point.selected).padStart(4)} | non-break acc=${((nonBreak?.accuracy ?? 0) * 100).toFixed(1).padStart(5)}% cov=${((nonBreak?.coverage ?? 0) * 100).toFixed(1).padStart(5)}% sel=${String(nonBreak?.selected ?? 0).padStart(4)}`);
  }
  console.log('');
  console.log('Pre / at / post break-window directional accuracy:');
  console.log(`  pre-break : ${(artifact.prePostBreakWindow.preBreak.directionalAccuracy * 100).toFixed(1)}% | steps=${artifact.prePostBreakWindow.preBreak.totalSteps}`);
  console.log(`  at-break  : ${(artifact.prePostBreakWindow.atBreak.directionalAccuracy * 100).toFixed(1)}% | steps=${artifact.prePostBreakWindow.atBreak.totalSteps}`);
  console.log(`  post-break: ${(artifact.prePostBreakWindow.postBreak.directionalAccuracy * 100).toFixed(1)}% | steps=${artifact.prePostBreakWindow.postBreak.totalSteps}`);
  console.log('');
  console.log('Per-ticker summary:');
  for (const ticker of TICKERS) {
    const row = perTicker[ticker];
    console.log(
      `  ${ticker.padEnd(5)} | steps=${String(row.totalSteps).padStart(3)} | break=${String(row.breakSteps).padStart(3)} | break dir=${(row.breakDirectionalAccuracy * 100).toFixed(1).padStart(5)}% | non-break dir=${(row.nonBreakDirectionalAccuracy * 100).toFixed(1).padStart(5)}% | errors=${row.errorCount}`,
    );
  }
  console.log('');
  console.log('By divergence bucket:');
  for (const row of artifact.byDivergence) {
    console.log(`  ${row.label.padEnd(9)} | n=${String(row.count).padStart(4)} | dir=${(row.directionalAccuracy * 100).toFixed(1).padStart(5)}% | brier=${row.brierScore.toFixed(3)} | ci=${(row.ciCoverage * 100).toFixed(1).padStart(5)}%`);
  }
  console.log('');
  console.log('By realized move direction (descriptive, break vs non-break):');
  for (const row of artifact.byMoveDirection) {
    console.log(`  ${row.label.padEnd(6)} | break n=${String(row.breakSteps).padStart(4)} dir=${(row.breakDirectionalAccuracy * 100).toFixed(1).padStart(5)}% | non-break n=${String(row.nonBreakSteps).padStart(4)} dir=${(row.nonBreakDirectionalAccuracy * 100).toFixed(1).padStart(5)}%`);
  }
  console.log('');
  console.log('By realized move magnitude (descriptive, break vs non-break):');
  for (const row of artifact.byMoveMagnitude) {
    console.log(`  ${row.label.padEnd(7)} | break n=${String(row.breakSteps).padStart(4)} dir=${(row.breakDirectionalAccuracy * 100).toFixed(1).padStart(5)}% | non-break n=${String(row.nonBreakSteps).padStart(4)} dir=${(row.nonBreakDirectionalAccuracy * 100).toFixed(1).padStart(5)}%`);
  }
  console.log('');
  console.log('By trend vs chop (break vs non-break):');
  for (const row of artifact.byTrendVsChop) {
    console.log(`  ${row.label.padEnd(8)} | break n=${String(row.breakSteps).padStart(4)} dir=${(row.breakDirectionalAccuracy * 100).toFixed(1).padStart(5)}% | non-break n=${String(row.nonBreakSteps).padStart(4)} dir=${(row.nonBreakDirectionalAccuracy * 100).toFixed(1).padStart(5)}%`);
  }
  console.log('');
  console.log('By recommendation (break vs non-break):');
  for (const row of artifact.byRecommendation) {
    console.log(`  ${row.label.padEnd(4)} | break n=${String(row.breakSteps).padStart(4)} dir=${(row.breakDirectionalAccuracy * 100).toFixed(1).padStart(5)}% | non-break n=${String(row.nonBreakSteps).padStart(4)} dir=${(row.nonBreakDirectionalAccuracy * 100).toFixed(1).padStart(5)}%`);
  }
  console.log('');
  console.log('By ensemble consensus (break vs non-break):');
  for (const row of artifact.byEnsembleConsensus) {
    console.log(`  ${row.label.padEnd(7)} | break n=${String(row.breakSteps).padStart(4)} dir=${(row.breakDirectionalAccuracy * 100).toFixed(1).padStart(5)}% | non-break n=${String(row.nonBreakSteps).padStart(4)} dir=${(row.nonBreakDirectionalAccuracy * 100).toFixed(1).padStart(5)}%`);
  }
  console.log('');
  console.log('Conditional break contexts:');
  for (const row of [
    artifact.conditionalBreakContexts.breakTrending,
    artifact.conditionalBreakContexts.breakChop,
    artifact.conditionalBreakContexts.breakLargeMove,
  ]) {
    console.log(`  ${row.label.padEnd(17)} | n=${String(row.totalSteps).padStart(4)} | dir=${(row.directionalAccuracy * 100).toFixed(1).padStart(5)}% | brier=${row.brierScore.toFixed(3)} | ci=${(row.ciCoverage * 100).toFixed(1).padStart(5)}%`);
  }

  mkdirSync(artifactDir, { recursive: true });
  writeFileSync(artifactPath, JSON.stringify(artifact, null, 2));
  console.log('');
  console.log(`Saved artifact to ${artifactPath}`);
}

main().catch(err => {
  console.error('Fatal:', err);
  process.exit(1);
});
