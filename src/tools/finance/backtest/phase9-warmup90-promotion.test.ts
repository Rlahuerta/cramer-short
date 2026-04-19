import { describe, expect, it } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import {
  evaluatePhase9Thresholds,
  runPromotion,
  type Phase9Artifact,
} from './phase9-warmup90-promotion.js';

describe('phase9-warmup90-promotion', () => {
  describe('evaluatePhase9Thresholds (pure unit tests)', () => {
    it('passes when all deltas are within guardrails and 14d evidence expectation met', () => {
      const result = evaluatePhase9Thresholds({
        candidate: {
          overall: {
            directionalAccuracy: 0.66,
            brierScore: 0.22,
            ciCoverage: 0.94,
            avgConfidence: 0.30,
            directionalCi: { lower: 0.63, median: 0.66, upper: 0.69, nResamples: 500 },
          },
          deltaVsBaseline: {
            overallDirectionalAccuracy: 0.002,
            overallBrier: -0.001,
            overallCiCoverage: 0.005,
            breakContextDirectionalAccuracy: 0.014,
            nonBreakDirectionalAccuracy: 0.0,
            breakChopRC020Accuracy: 0.016,
            breakChopRC020Coverage: 0.01,
            breakContextRC020Accuracy: 0.02,
            breakContextRC020Coverage: 0.015,
            breakContextRC030Accuracy: 0.024,
            breakContextRC030Coverage: 0.019,
          },
          perHorizonDelta: [
            { horizon: 7, overallDirectionalAccuracy: 0.013, breakContextDirectionalAccuracy: 0.014, nonBreakDirectionalAccuracy: 0.0, brier: -0.004, ciCoverage: 0.002, breakContextRC020Accuracy: 0.022, breakContextRC030Accuracy: 0.019 },
            { horizon: 14, overallDirectionalAccuracy: 0.017, breakContextDirectionalAccuracy: 0.018, nonBreakDirectionalAccuracy: 0.0, brier: -0.009, ciCoverage: 0.003, breakContextRC020Accuracy: 0.032, breakContextRC030Accuracy: 0.051 },
            { horizon: 30, overallDirectionalAccuracy: 0.010, breakContextDirectionalAccuracy: 0.011, nonBreakDirectionalAccuracy: 0.0, brier: -0.003, ciCoverage: 0.005, breakContextRC020Accuracy: 0.013, breakContextRC030Accuracy: 0.011 },
          ],
        },
        baselinePerHorizon: [
          { horizon: 7, steps: 100, breakSteps: 50, rc020: { threshold: 0.2, accuracy: 0.60, coverage: 0.50, n: 50 }, rc030: { threshold: 0.3, accuracy: 0.58, coverage: 0.30, n: 30 } } as any,
        ],
      });

      expect(result.passes).toBe(true);
      expect(result.failureReasons).toHaveLength(0);
    });

    it('fails when overall directional accuracy regresses beyond guardrail', () => {
      const result = evaluatePhase9Thresholds({
        candidate: {
          overall: {
            directionalAccuracy: 0.55,
            brierScore: 0.20,
            ciCoverage: 0.93,
            avgConfidence: 0.30,
            directionalCi: { lower: 0.50, median: 0.55, upper: 0.60, nResamples: 500 },
          },
          deltaVsBaseline: {
            overallDirectionalAccuracy: -0.01,
            overallBrier: 0.0,
            overallCiCoverage: 0.0,
            breakContextDirectionalAccuracy: 0.0,
            nonBreakDirectionalAccuracy: 0.0,
            breakChopRC020Accuracy: 0.0,
            breakChopRC020Coverage: 0.0,
            breakContextRC020Accuracy: 0.0,
            breakContextRC020Coverage: 0.0,
            breakContextRC030Accuracy: 0.0,
            breakContextRC030Coverage: 0.0,
          },
          perHorizonDelta: [
            { horizon: 7, overallDirectionalAccuracy: -0.01, breakContextDirectionalAccuracy: 0.0, nonBreakDirectionalAccuracy: -0.01, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
            { horizon: 14, overallDirectionalAccuracy: -0.01, breakContextDirectionalAccuracy: 0.01, nonBreakDirectionalAccuracy: -0.01, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
            { horizon: 30, overallDirectionalAccuracy: -0.01, breakContextDirectionalAccuracy: 0.0, nonBreakDirectionalAccuracy: -0.01, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
          ],
        },
        baselinePerHorizon: [],
      });

      expect(result.passes).toBe(false);
      expect(result.failureReasons.some(r => r.includes('overall directional accuracy'))).toBe(true);
    });

    it('fails when CI coverage drops below minimum', () => {
      const result = evaluatePhase9Thresholds({
        candidate: {
          overall: {
            directionalAccuracy: 0.66,
            brierScore: 0.22,
            ciCoverage: 0.88,
            avgConfidence: 0.30,
            directionalCi: { lower: 0.55, median: 0.60, upper: 0.65, nResamples: 500 },
          },
          deltaVsBaseline: {
            overallDirectionalAccuracy: 0.0,
            overallBrier: 0.0,
            overallCiCoverage: -0.06,
            breakContextDirectionalAccuracy: 0.0,
            nonBreakDirectionalAccuracy: 0.0,
            breakChopRC020Accuracy: 0.0,
            breakChopRC020Coverage: 0.0,
            breakContextRC020Accuracy: 0.0,
            breakContextRC020Coverage: 0.0,
            breakContextRC030Accuracy: 0.0,
            breakContextRC030Coverage: 0.0,
          },
          perHorizonDelta: [
            { horizon: 7, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.0, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: -0.06, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
            { horizon: 14, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.0, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
            { horizon: 30, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.01, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
          ],
        },
        baselinePerHorizon: [],
      });

      expect(result.passes).toBe(false);
      expect(result.failureReasons.some(r => r.includes('CI coverage'))).toBe(true);
    });

    it('fails when 14d break-context gain is below evidence expectation', () => {
      const result = evaluatePhase9Thresholds({
        candidate: {
          overall: {
            directionalAccuracy: 0.66,
            brierScore: 0.22,
            ciCoverage: 0.94,
            avgConfidence: 0.30,
            directionalCi: { lower: 0.55, median: 0.60, upper: 0.65, nResamples: 500 },
          },
          deltaVsBaseline: {
            overallDirectionalAccuracy: 0.0,
            overallBrier: 0.0,
            overallCiCoverage: 0.0,
            breakContextDirectionalAccuracy: 0.0,
            nonBreakDirectionalAccuracy: 0.0,
            breakChopRC020Accuracy: 0.0,
            breakChopRC020Coverage: 0.0,
            breakContextRC020Accuracy: 0.0,
            breakContextRC020Coverage: 0.0,
            breakContextRC030Accuracy: 0.0,
            breakContextRC030Coverage: 0.0,
          },
          perHorizonDelta: [
            { horizon: 7, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.0, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
            { horizon: 14, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.002, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
            { horizon: 30, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.0, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
          ],
        },
        baselinePerHorizon: [],
      });

      expect(result.passes).toBe(false);
      expect(result.failureReasons.some(r => r.includes('14d'))).toBe(true);
    });

    it('fails when per-horizon break-context regresses beyond max loss', () => {
      const result = evaluatePhase9Thresholds({
        candidate: {
          overall: {
            directionalAccuracy: 0.66,
            brierScore: 0.22,
            ciCoverage: 0.94,
            avgConfidence: 0.30,
            directionalCi: { lower: 0.55, median: 0.60, upper: 0.65, nResamples: 500 },
          },
          deltaVsBaseline: {
            overallDirectionalAccuracy: 0.0,
            overallBrier: 0.0,
            overallCiCoverage: 0.0,
            breakContextDirectionalAccuracy: 0.0,
            nonBreakDirectionalAccuracy: 0.0,
            breakChopRC020Accuracy: 0.0,
            breakChopRC020Coverage: 0.0,
            breakContextRC020Accuracy: 0.0,
            breakContextRC020Coverage: 0.0,
            breakContextRC030Accuracy: 0.0,
            breakContextRC030Coverage: 0.0,
          },
          perHorizonDelta: [
            { horizon: 7, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: -0.02, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
            { horizon: 14, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.01, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
            { horizon: 30, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.0, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
          ],
        },
        baselinePerHorizon: [],
      });

      expect(result.passes).toBe(false);
      expect(result.failureReasons.some(r => r.includes('horizon guardrail') && r.includes('7d'))).toBe(true);
    });

    it('fails when break+sideways RC@0.2 accuracy regresses', () => {
      const result = evaluatePhase9Thresholds({
        candidate: {
          overall: {
            directionalAccuracy: 0.66,
            brierScore: 0.22,
            ciCoverage: 0.94,
            avgConfidence: 0.30,
            directionalCi: { lower: 0.63, median: 0.66, upper: 0.69, nResamples: 500 },
          },
          deltaVsBaseline: {
            overallDirectionalAccuracy: 0.001,
            overallBrier: 0.0,
            overallCiCoverage: 0.0,
            breakContextDirectionalAccuracy: 0.01,
            nonBreakDirectionalAccuracy: 0.0,
            breakChopRC020Accuracy: -0.015,
            breakChopRC020Coverage: 0.01,
            breakContextRC020Accuracy: 0.01,
            breakContextRC020Coverage: 0.01,
            breakContextRC030Accuracy: 0.01,
            breakContextRC030Coverage: 0.01,
          },
          perHorizonDelta: [
            { horizon: 7, overallDirectionalAccuracy: 0.001, breakContextDirectionalAccuracy: 0.014, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.01, breakContextRC030Accuracy: 0.01 },
            { horizon: 14, overallDirectionalAccuracy: 0.001, breakContextDirectionalAccuracy: 0.018, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.01, breakContextRC030Accuracy: 0.01 },
            { horizon: 30, overallDirectionalAccuracy: 0.001, breakContextDirectionalAccuracy: 0.011, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.01, breakContextRC030Accuracy: 0.01 },
          ],
        },
        baselinePerHorizon: [],
      });

      expect(result.passes).toBe(false);
      expect(result.failureReasons.some(r => r.includes('break+sideways RC@0.2 accuracy'))).toBe(true);
    });
  });

  integrationIt('runs the full promotion and produces a valid artifact', async () => {
    const artifact = await runPromotion();

    expect(artifact.generatedAt).toBeTruthy();
    expect(artifact.baseline.label).toBe('phase4-control');
    expect(artifact.baseline.warmup).toBe(120);
    expect(artifact.universe.tickers).toEqual(['SPY', 'QQQ', 'GLD', 'VOO', 'NVDA', 'MSFT']);
    expect(artifact.universe.horizons).toEqual([7, 14, 30]);

    const c = artifact.candidate;
    expect(c.candidateId).toBe('warmup-90');
    expect(c.warmup).toBe(90);
    expect(c.perHorizon).toHaveLength(3);
    expect(c.perHorizonDelta).toHaveLength(3);
    expect(c.deltaVsBaseline).toBeDefined();
    expect(typeof c.passesThresholds).toBe('boolean');
    if (!c.passesThresholds) {
      expect(c.failureReasons.length).toBeGreaterThan(0);
    }

    expect(artifact.verdict).toBeDefined();
    expect(typeof artifact.verdict.promoted).toBe('boolean');
    if (!artifact.verdict.promoted) {
      expect(artifact.verdict.reason).toBeTruthy();
    }
  }, 480_000);
});