import { describe, expect, it } from 'bun:test';
import {
  formatFailureAnalysisReport,
  rankFailureBuckets,
  type RankedFailureBucket,
} from './btc-failure-analysis.js';
import { type BacktestStep } from './metrics.js';

function makeStep(overrides: Partial<BacktestStep> = {}): BacktestStep {
  return {
    t: 0,
    predictedProb: 0.5,
    actualBinary: 1,
    predictedReturn: 0.02,
    actualReturn: 0.03,
    ciLower: 90,
    ciUpper: 110,
    realizedPrice: 100,
    recommendation: 'BUY',
    gofPasses: null,
    confidence: 0.5,
    probabilitySource: 'calibrated',
    decisionSource: 'default',
    ...overrides,
  };
}

function makeAllCorrectSteps(total: number): BacktestStep[] {
  return Array.from({ length: total }, (_, index) =>
    makeStep({
      t: index,
      regime: 'bull',
      predictedProb: 0.9,
      actualBinary: 1,
      predictedReturn: 0.03,
      actualReturn: 0.05,
      recommendation: 'BUY',
      confidence: 0.8,
    }),
  );
}

function makeActionableSteps(): BacktestStep[] {
  const bullishBuys = Array.from({ length: 7 }, (_, index) =>
    makeStep({
      t: index,
      regime: 'bull',
      predictedProb: 0.9,
      actualBinary: 1,
      predictedReturn: 0.03,
      actualReturn: 0.05,
      recommendation: 'BUY',
      confidence: 0.8,
    }),
  );
  const bullishSells = Array.from({ length: 7 }, (_, index) =>
    makeStep({
      t: 10 + index,
      regime: 'bull',
      predictedProb: 0.1,
      actualBinary: 0,
      predictedReturn: -0.03,
      actualReturn: -0.05,
      recommendation: 'SELL',
      confidence: 0.8,
    }),
  );
  const bearishWrongBuys = Array.from({ length: 3 }, (_, index) =>
    makeStep({
      t: 20 + index,
      regime: 'bear',
      predictedProb: 0.9,
      actualBinary: 0,
      predictedReturn: 0.03,
      actualReturn: -0.05,
      recommendation: 'BUY',
      confidence: 0.8,
    }),
  );
  const bearishWrongSells = Array.from({ length: 3 }, (_, index) =>
    makeStep({
      t: 30 + index,
      regime: 'bear',
      predictedProb: 0.1,
      actualBinary: 1,
      predictedReturn: -0.03,
      actualReturn: 0.05,
      recommendation: 'SELL',
      confidence: 0.8,
    }),
  );

  return [
    ...bullishBuys,
    ...bullishSells,
    ...bearishWrongBuys,
    ...bearishWrongSells,
  ];
}

function makeDiffuseWeaknessSteps(): BacktestStep[] {
  const strongUp = Array.from({ length: 5 }, (_, index) =>
    makeStep({
      t: index,
      regime: 'bull',
      predictedProb: 0.9,
      actualBinary: 1,
      predictedReturn: 0.03,
      actualReturn: 0.05,
      recommendation: 'BUY',
      confidence: 0.8,
      anchorQuality: 'good',
    }),
  );
  const strongDown = Array.from({ length: 5 }, (_, index) =>
    makeStep({
      t: 10 + index,
      regime: 'bull',
      predictedProb: 0.1,
      actualBinary: 0,
      predictedReturn: -0.03,
      actualReturn: -0.05,
      recommendation: 'SELL',
      confidence: 0.8,
      anchorQuality: 'good',
    }),
  );
  const weakSideways = [
    ...Array.from({ length: 3 }, (_, index) =>
      makeStep({
        t: 20 + index,
        regime: 'sideways',
        predictedProb: 0.9,
        actualBinary: 0,
        predictedReturn: 0.03,
        actualReturn: -0.05,
        recommendation: 'BUY',
        confidence: 0.8,
        anchorQuality: 'good',
      }),
    ),
    ...Array.from({ length: 2 }, (_, index) =>
      makeStep({
        t: 30 + index,
        regime: 'sideways',
        predictedProb: 0.1,
        actualBinary: 1,
        predictedReturn: -0.03,
        actualReturn: 0.05,
        recommendation: 'SELL',
        confidence: 0.8,
        anchorQuality: 'good',
      }),
    ),
  ];
  const weakAnchorNone = [
    ...Array.from({ length: 3 }, (_, index) =>
      makeStep({
        t: 40 + index,
        regime: 'bull',
        predictedProb: 0.9,
        actualBinary: 0,
        predictedReturn: 0.03,
        actualReturn: -0.05,
        recommendation: 'BUY',
        confidence: 0.8,
        anchorQuality: 'none',
      }),
    ),
    ...Array.from({ length: 2 }, (_, index) =>
      makeStep({
        t: 50 + index,
        regime: 'bull',
        predictedProb: 0.1,
        actualBinary: 1,
        predictedReturn: -0.03,
        actualReturn: 0.05,
        recommendation: 'SELL',
        confidence: 0.8,
        anchorQuality: 'none',
      }),
    ),
  ];

  return [...strongUp, ...strongDown, ...weakSideways, ...weakAnchorNone];
}

function makeBroadWeakSliceSteps(): BacktestStep[] {
  const strongUp = Array.from({ length: 6 }, (_, index) =>
    makeStep({
      t: index,
      regime: 'bull',
      predictedProb: 0.9,
      actualBinary: 1,
      predictedReturn: 0.03,
      actualReturn: 0.05,
      recommendation: 'BUY',
      confidence: 0.8,
      anchorQuality: 'good',
    }),
  );
  const strongDown = Array.from({ length: 6 }, (_, index) =>
    makeStep({
      t: 10 + index,
      regime: 'bull',
      predictedProb: 0.1,
      actualBinary: 0,
      predictedReturn: -0.03,
      actualReturn: -0.05,
      recommendation: 'SELL',
      confidence: 0.8,
      anchorQuality: 'good',
    }),
  );
  const broadWeakUp = Array.from({ length: 4 }, (_, index) =>
    makeStep({
      t: 20 + index,
      regime: 'bull',
      predictedProb: 0.1,
      actualBinary: 1,
      predictedReturn: -0.03,
      actualReturn: 0.05,
      recommendation: 'SELL',
      confidence: 0.8,
      anchorQuality: 'none',
    }),
  );
  const broadWeakDown = Array.from({ length: 4 }, (_, index) =>
    makeStep({
      t: 30 + index,
      regime: 'bull',
      predictedProb: 0.9,
      actualBinary: 0,
      predictedReturn: 0.03,
      actualReturn: -0.05,
      recommendation: 'BUY',
      confidence: 0.8,
      anchorQuality: 'none',
    }),
  );

  return [...strongUp, ...strongDown, ...broadWeakUp, ...broadWeakDown];
}

function makeMaskedComparableFamilySteps(): BacktestStep[] {
  const strongUp = Array.from({ length: 13 }, (_, index) =>
    makeStep({
      t: index,
      regime: 'bull',
      predictedProb: 0.9,
      actualBinary: 1,
      predictedReturn: 0.03,
      actualReturn: 0.05,
      recommendation: 'BUY',
      confidence: 0.8,
      anchorQuality: 'good',
      structuralBreakDetected: false,
    }),
  );
  const broadAnchorCorrect = makeStep({
    t: 20,
    regime: 'bull',
    predictedProb: 0.9,
    actualBinary: 1,
    predictedReturn: 0.03,
    actualReturn: 0.05,
    recommendation: 'BUY',
    confidence: 0.8,
    anchorQuality: 'none',
    structuralBreakDetected: false,
  });
  const weakDown = Array.from({ length: 7 }, (_, index) =>
    makeStep({
      t: 30 + index,
      regime: 'bull',
      predictedProb: 0.9,
      actualBinary: 0,
      predictedReturn: 0.03,
      actualReturn: -0.05,
      recommendation: 'BUY',
      confidence: 0.8,
      anchorQuality: 'none',
      structuralBreakDetected: index < 5,
    }),
  );

  return [...strongUp, broadAnchorCorrect, ...weakDown];
}

function makeDerivedOverlapSteps(): BacktestStep[] {
  const strongUp = Array.from({ length: 7 }, (_, index) =>
    makeStep({
      t: index,
      regime: 'bull',
      predictedProb: 0.9,
      actualBinary: 1,
      predictedReturn: 0.03,
      actualReturn: 0.05,
      recommendation: 'BUY',
      confidence: 0.8,
      anchorQuality: 'good',
    }),
  );
  const strongDown = Array.from({ length: 7 }, (_, index) =>
    makeStep({
      t: 10 + index,
      regime: 'bull',
      predictedProb: 0.1,
      actualBinary: 0,
      predictedReturn: -0.03,
      actualReturn: -0.05,
      recommendation: 'SELL',
      confidence: 0.8,
      anchorQuality: 'good',
    }),
  );
  const weakChopDown = Array.from({ length: 3 }, (_, index) =>
    makeStep({
      t: 20 + index,
      regime: 'sideways',
      predictedProb: 0.9,
      actualBinary: 0,
      predictedReturn: 0.03,
      actualReturn: -0.05,
      recommendation: 'BUY',
      confidence: 0.8,
      anchorQuality: 'good',
    }),
  );
  const weakChopUp = Array.from({ length: 3 }, (_, index) =>
    makeStep({
      t: 30 + index,
      regime: 'sideways',
      predictedProb: 0.1,
      actualBinary: 1,
      predictedReturn: -0.03,
      actualReturn: 0.05,
      recommendation: 'SELL',
      confidence: 0.8,
      anchorQuality: 'good',
    }),
  );

  return [...strongUp, ...strongDown, ...weakChopDown, ...weakChopUp];
}

function makeMultiWeakSteps(): BacktestStep[] {
  const strongSteps = Array.from({ length: 15 }, (_, index) =>
    makeStep({
      t: index,
      regime: 'bull',
      predictedProb: index < 8 ? 0.9 : 0.1,
      actualBinary: index < 8 ? 1 : 0,
      predictedReturn: index < 8 ? 0.03 : -0.03,
      actualReturn: index < 8 ? 0.05 : -0.05,
      recommendation: index < 8 ? 'BUY' : 'SELL',
      confidence: 0.85,
      anchorQuality: 'good',
      validationMetric: 'daily_return',
      structuralBreakDetected: false,
      hmmConverged: true,
      ensembleConsensus: 0.8,
    }),
  );
  const weakSteps = Array.from({ length: 5 }, (_, index) =>
    makeStep({
      t: 100 + index,
      regime: 'bear',
      predictedProb: 0.9,
      actualBinary: 0,
      predictedReturn: 0.03,
      actualReturn: -0.05,
      recommendation: 'BUY',
      confidence: 0.1,
      anchorQuality: 'none',
      validationMetric: 'horizon_return',
      structuralBreakDetected: true,
      hmmConverged: false,
      ensembleConsensus: 0.2,
    }),
  );

  return [...strongSteps, ...weakSteps];
}

function makeTwoOfTenCorrectSteps(): BacktestStep[] {
  const strongSteps = Array.from({ length: 10 }, (_, index) =>
    makeStep({
      t: index,
      regime: 'bull',
      predictedProb: 0.9,
      actualBinary: 1,
      predictedReturn: 0.03,
      actualReturn: 0.05,
      recommendation: 'BUY',
      confidence: 0.8,
    }),
  );

  const weakBearCorrect = Array.from({ length: 2 }, (_, index) =>
    makeStep({
      t: 20 + index,
      regime: 'bear',
      predictedProb: 0.1,
      actualBinary: 0,
      predictedReturn: -0.03,
      actualReturn: -0.05,
      recommendation: 'SELL',
      confidence: 0.8,
    }),
  );

  const weakBearWrong = Array.from({ length: 8 }, (_, index) =>
    makeStep({
      t: 30 + index,
      regime: 'bear',
      predictedProb: 0.9,
      actualBinary: 0,
      predictedReturn: 0.03,
      actualReturn: -0.05,
      recommendation: 'BUY',
      confidence: 0.8,
    }),
  );

  return [...strongSteps, ...weakBearCorrect, ...weakBearWrong];
}

function findBucket(
  report: ReturnType<typeof rankFailureBuckets>,
  dimension: RankedFailureBucket['dimension'],
  bucket: string,
): RankedFailureBucket | undefined {
  return report.rankedBuckets.find(entry => entry.dimension === dimension && entry.bucket === bucket);
}

describe('rankFailureBuckets', () => {
  it('returns insufficient-data for empty steps', () => {
    const report = rankFailureBuckets([], 7);

    expect(report.verdict).toBe('insufficient-data');
    expect(report.rankedBuckets).toHaveLength(0);
    expect(report.topCandidate).toBeNull();
  });

  it('returns insufficient-data when no buckets meet minBucketN after enough total steps', () => {
    const steps = makeAllCorrectSteps(20);

    const report = rankFailureBuckets(steps, 7, { minBucketN: 21 });

    expect(report.verdict).toBe('insufficient-data');
    expect(report.rankedBuckets).toHaveLength(0);
  });

  it('returns diffuse when no bucket underperforms the aggregate', () => {
    const report = rankFailureBuckets(makeAllCorrectSteps(20), 7);

    expect(report.verdict).toBe('diffuse');
    expect(report.rankedBuckets).toHaveLength(0);
    expect(report.topCandidate).toBeNull();
  });

  it('marks one clear weak bucket as actionable', () => {
    const report = rankFailureBuckets(makeActionableSteps(), 7);

    expect(report.verdict).toBe('actionable');
    expect(report.topCandidate).not.toBeNull();
    expect(report.topCandidate?.dimension).toBe('regime');
    expect(report.topCandidate?.bucket).toBe('bear');
    expect(report.topCandidate?.passesWilsonScreen).toBe(true);
    expect(report.topCandidate?.weaknessScore).toBeGreaterThan(0.05);
  });

  it('returns diffuse when weakness is split across comparable independent families', () => {
    const report = rankFailureBuckets(makeDiffuseWeaknessSteps(), 14);

    expect(report.verdict).toBe('diffuse');
    expect(report.topCandidate).not.toBeNull();
    expect(report.verdictReason).toContain('split across independent families');
  });

  it('keeps the verdict diffuse even when topK hides comparable families', () => {
    const report = rankFailureBuckets(makeDiffuseWeaknessSteps(), 14, { topK: 1 });

    expect(report.verdict).toBe('diffuse');
    expect(report.rankedBuckets).toHaveLength(1);
    expect(report.verdictReason).toContain('split across independent families');
  });

  it('returns diffuse when the top weakness is too broad to be sparse', () => {
    const report = rankFailureBuckets(makeBroadWeakSliceSteps(), 14);

    expect(report.verdict).toBe('diffuse');
    expect(report.topCandidate).not.toBeNull();
    expect(report.topCandidate?.fraction).toBeGreaterThan(0.35);
    expect(report.verdictReason).toContain('too broad');
  });

  it('respects topK when multiple weak buckets exist', () => {
    const report = rankFailureBuckets(makeMultiWeakSteps(), 14, { topK: 3 });

    expect(report.rankedBuckets).toHaveLength(3);
  });

  it('collapses derived overlap before verdicting actionable slices', () => {
    const report = rankFailureBuckets(makeDerivedOverlapSteps(), 14);

    expect(report.verdict).toBe('actionable');
    expect(report.rankedBuckets.some(bucket => bucket.dimension === 'regime' && bucket.bucket === 'sideways')).toBe(true);
    expect(report.rankedBuckets.some(bucket => bucket.dimension === 'trendVsChop')).toBe(false);
  });

  it('finds later comparable families even when a broad family ranks second', () => {
    const report = rankFailureBuckets(makeMaskedComparableFamilySteps(), 14, { topK: 2 });

    expect(report.verdict).toBe('diffuse');
    expect(report.rankedBuckets).toHaveLength(2);
    expect(report.verdictReason).toContain('structuralBreak=true');
  });

  it('produces sane Wilson confidence bounds for a weak bucket', () => {
    const report = rankFailureBuckets(makeTwoOfTenCorrectSteps(), 14);
    const bucket = findBucket(report, 'regime', 'bear');

    expect(bucket).toBeDefined();
    expect(bucket!.accuracyCI.lower).toBeGreaterThanOrEqual(0);
    expect(bucket!.accuracyCI.upper).toBeLessThanOrEqual(1);
    expect(bucket!.accuracyCI.lower).toBeLessThan(bucket!.accuracyCI.upper);
  });
});

describe('formatFailureAnalysisReport', () => {
  it('returns readable lines for a populated report', () => {
    const report = rankFailureBuckets(makeActionableSteps(), 7);
    const lines = formatFailureAnalysisReport(report);

    expect(lines.length).toBeGreaterThan(3);
    expect(lines[0]).toContain('BTC-USD 7d failure-slice analysis');
    expect(lines.some(line => line.includes('verdict=actionable'))).toBe(true);
    expect(lines.some(line => line.includes('regime:bear'))).toBe(true);
  });

  it('surfaces insufficient-data explicitly in report output', () => {
    const report = rankFailureBuckets([], 7);
    const lines = formatFailureAnalysisReport(report);

    expect(lines.some(line => line.includes('insufficient data'))).toBe(true);
  });
});
