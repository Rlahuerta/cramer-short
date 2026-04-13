import { describe, expect, it } from 'bun:test';
import type { BacktestStep } from './metrics.js';
import {
  applyPolicyToSteps,
  CANDIDATE_POLICIES,
  computePolicyDeltas,
  effectiveBreakContext,
  effectiveDivergence,
  generateAblationReport,
  reconstructPreBreakConfidence,
  simulateAllPolicies,
  simulatePolicy,
} from './break-policy-ablation.js';

function makeStep(overrides: Partial<BacktestStep> = {}): BacktestStep {
  return {
    t: 0,
    predictedProb: 0.55,
    actualBinary: 1,
    predictedReturn: 0.02,
    actualReturn: 0.03,
    ciLower: 95,
    ciUpper: 105,
    realizedPrice: 100,
    recommendation: 'BUY',
    gofPasses: null,
    confidence: 0.30,
    probabilitySource: 'calibrated',
    decisionSource: 'default',
    ...overrides,
  };
}

describe('break-policy-ablation helpers', () => {
  it('detects effective break context using original or final flag', () => {
    expect(effectiveBreakContext(makeStep({ structuralBreakDetected: true }))).toBe(true);
    expect(effectiveBreakContext(makeStep({ originalStructuralBreakDetected: true, structuralBreakDetected: false }))).toBe(true);
    expect(effectiveBreakContext(makeStep({ structuralBreakDetected: false }))).toBe(false);
  });

  it('prefers original divergence when present', () => {
    expect(effectiveDivergence(makeStep({ originalStructuralBreakDivergence: 0.12, structuralBreakDivergence: 0.08 }))).toBe(0.12);
    expect(effectiveDivergence(makeStep({ structuralBreakDivergence: 0.08 }))).toBe(0.08);
    expect(effectiveDivergence(makeStep())).toBeNull();
  });

  it('reconstructs pre-break confidence by dividing out the baseline penalty', () => {
    expect(reconstructPreBreakConfidence(makeStep({ structuralBreakDetected: true, confidence: 0.36 }))).toBeCloseTo(0.6, 6);
    expect(reconstructPreBreakConfidence(makeStep({ structuralBreakDetected: false, confidence: 0.36 }))).toBeCloseTo(0.36, 6);
  });

  it('does not divide out the baseline penalty when the live trend-only policy was already active', () => {
    expect(reconstructPreBreakConfidence(makeStep({
      structuralBreakDetected: true,
      regime: 'sideways',
      trendPenaltyOnlyBreakConfidenceActive: true,
      confidence: 0.60,
    }))).toBeCloseTo(0.60, 6);
  });

  it('still reconstructs the pre-break baseline for flagged break+trending steps', () => {
    expect(reconstructPreBreakConfidence(makeStep({
      structuralBreakDetected: true,
      regime: 'bull',
      trendPenaltyOnlyBreakConfidenceActive: true,
      confidence: 0.36,
    }))).toBeCloseTo(0.60, 6);
  });
});

describe('Phase 3 candidate policies', () => {
  const breakStep = makeStep({
    structuralBreakDetected: true,
    confidence: 0.36,
    regime: 'bull',
    actualReturn: 0.12,
    structuralBreakDivergence: 0.22,
  });
  const chopBreakStep = makeStep({
    structuralBreakDetected: true,
    confidence: 0.36,
    regime: 'sideways',
    actualReturn: 0.01,
    structuralBreakDivergence: 0.06,
  });

  it('contains the expected policy set', () => {
    expect(CANDIDATE_POLICIES.map(policy => policy.name)).toEqual([
      'baseline',
      'reduced_penalty_075',
      'no_break_penalty',
      'aggressive_penalty_040',
      'cap_029',
      'trend_penalty_only',
      'chop_penalty_only',
      'divergence_relief_bucketed',
    ]);
  });

  it('applies the baseline and reduced-penalty policies correctly', () => {
    const baseline = CANDIDATE_POLICIES.find(policy => policy.name === 'baseline');
    const reduced = CANDIDATE_POLICIES.find(policy => policy.name === 'reduced_penalty_075');
    expect(baseline?.apply(breakStep)).toBeCloseTo(0.36, 6);
    expect(reduced?.apply(breakStep)).toBeCloseTo(0.45, 6);
  });

  it('applies conditional break policies correctly', () => {
    const trendOnly = CANDIDATE_POLICIES.find(policy => policy.name === 'trend_penalty_only');
    const chopOnly = CANDIDATE_POLICIES.find(policy => policy.name === 'chop_penalty_only');
    expect(trendOnly?.apply(breakStep)).toBeCloseTo(0.36, 6);
    expect(trendOnly?.apply(chopBreakStep)).toBeCloseTo(0.6, 6);
    expect(chopOnly?.apply(breakStep)).toBeCloseTo(0.6, 6);
    expect(chopOnly?.apply(chopBreakStep)).toBeCloseTo(0.36, 6);
  });

  it('applies divergence-scaled relief correctly', () => {
    const divergenceRelief = CANDIDATE_POLICIES.find(policy => policy.name === 'divergence_relief_bucketed');
    expect(divergenceRelief?.apply(breakStep)).toBeCloseTo(0.54, 6);
    expect(divergenceRelief?.apply(chopBreakStep)).toBeCloseTo(0.36, 6);
  });
});

describe('Phase 3 simulation engine', () => {
  const steps: BacktestStep[] = [
    makeStep({ structuralBreakDetected: true, confidence: 0.36, regime: 'bull', actualReturn: 0.04, recommendation: 'BUY', predictedProb: 0.60, rawPredictedProb: 0.62 }),
    makeStep({ structuralBreakDetected: true, confidence: 0.18, regime: 'sideways', actualReturn: 0.01, recommendation: 'HOLD', predictedProb: 0.52, rawPredictedProb: 0.55 }),
    makeStep({ structuralBreakDetected: false, confidence: 0.42, regime: 'bull', actualReturn: 0.05, recommendation: 'BUY', predictedProb: 0.65, rawPredictedProb: 0.67 }),
    makeStep({ structuralBreakDetected: false, confidence: 0.28, regime: 'bear', actualReturn: -0.04, recommendation: 'SELL', actualBinary: 0, predictedProb: 0.35, rawPredictedProb: 0.32 }),
  ];

  it('replaces confidence only, preserving step count and non-break values', () => {
    const reduced = CANDIDATE_POLICIES.find(policy => policy.name === 'reduced_penalty_075');
    const adjusted = applyPolicyToSteps(steps, reduced!);
    expect(adjusted).toHaveLength(steps.length);
    expect(adjusted[0].confidence).toBeCloseTo(0.45, 6);
    expect(adjusted[2].confidence).toBeCloseTo(steps[2].confidence, 6);
  });

  it('simulates one policy and exposes RC curves for break and non-break contexts', () => {
    const result = simulatePolicy(steps, CANDIDATE_POLICIES[0], [0.0, 0.2, 0.3]);
    expect(result.name).toBe('baseline');
    expect(result.breakContext.count).toBe(2);
    expect(result.nonBreak.count).toBe(2);
    expect(result.recommendationRC.breakContext).toHaveLength(3);
    expect(result.pUpRC.breakContext).toHaveLength(3);
    expect(result.conditionalBreakContexts.breakTrending.count).toBe(1);
    expect(result.conditionalBreakContexts.breakChop.count).toBe(1);
  });

  it('simulates all policies and computes pairwise deltas vs baseline', () => {
    const results = simulateAllPolicies(steps, CANDIDATE_POLICIES, [0.0, 0.2, 0.3]);
    const report = generateAblationReport(steps, results, [0.0, 0.2, 0.3]);
    const deltas = computePolicyDeltas(results);
    expect(results).toHaveLength(CANDIDATE_POLICIES.length);
    expect(report.policies).toHaveLength(CANDIDATE_POLICIES.length);
    expect(report.pairwiseVsBaseline).toHaveLength(CANDIDATE_POLICIES.length);
    expect(deltas.find(delta => delta.name === 'baseline')?.breakAccuracyDelta).toBeCloseTo(0, 10);
  });
});
