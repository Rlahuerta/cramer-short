import { describe, expect, it } from 'bun:test';
import { integrationIt } from '@/utils/test-guards.js';
import {
  evaluatePhase8Thresholds,
  selectWinner,
  runComparison,
  type Phase8Artifact,
} from './phase8-warmup-window-comparison.js';

describe('phase8-warmup-window-comparison', () => {
  describe('evaluatePhase8Thresholds (pure unit tests)', () => {
    it('passes when all deltas are within guardrails and evidence expectations met', () => {
      const result = evaluatePhase8Thresholds({
        candidateId: 'warmup-90',
        warmup: 90,
        candidate: {
          overall: {
            directionalAccuracy: 0.60,
            brierScore: 0.18,
            ciCoverage: 0.92,
            avgConfidence: 0.35,
            directionalCi: { lower: 0.55, median: 0.60, upper: 0.65, nResamples: 500 },
          },
          deltaVsBaseline: {
            overallDirectionalAccuracy: 0.002,
            overallBrier: -0.001,
            overallCiCoverage: 0.005,
            breakContextDirectionalAccuracy: 0.01,
            nonBreakDirectionalAccuracy: 0.001,
            breakChopRC020Accuracy: 0.005,
            breakChopRC020Coverage: 0.01,
            breakContextRC020Accuracy: 0.008,
            breakContextRC020Coverage: 0.01,
            breakContextRC030Accuracy: 0.005,
            breakContextRC030Coverage: 0.008,
          },
          perHorizonDelta: [
            { horizon: 7, overallDirectionalAccuracy: 0.001, breakContextDirectionalAccuracy: 0.002, nonBreakDirectionalAccuracy: 0.001, brier: -0.001, ciCoverage: 0.002, breakContextRC020Accuracy: 0.003, breakContextRC030Accuracy: 0.002 },
            { horizon: 14, overallDirectionalAccuracy: 0.005, breakContextDirectionalAccuracy: 0.01, nonBreakDirectionalAccuracy: 0.002, brier: -0.002, ciCoverage: 0.003, breakContextRC020Accuracy: 0.008, breakContextRC030Accuracy: 0.005 },
            { horizon: 30, overallDirectionalAccuracy: -0.001, breakContextDirectionalAccuracy: 0.003, nonBreakDirectionalAccuracy: -0.001, brier: 0.001, ciCoverage: -0.002, breakContextRC020Accuracy: 0.002, breakContextRC030Accuracy: 0.001 },
          ],
        },
        baselinePerHorizon: [
          { horizon: 7, steps: 100, breakSteps: 50, rc020: { threshold: 0.2, accuracy: 0.60, coverage: 0.50, n: 50 }, rc030: { threshold: 0.3, accuracy: 0.58, coverage: 0.30, n: 30 } } as any,
          { horizon: 14, steps: 100, breakSteps: 50, rc020: { threshold: 0.2, accuracy: 0.60, coverage: 0.50, n: 50 }, rc030: { threshold: 0.3, accuracy: 0.58, coverage: 0.30, n: 30 } } as any,
          { horizon: 30, steps: 100, breakSteps: 50, rc020: { threshold: 0.2, accuracy: 0.60, coverage: 0.50, n: 50 }, rc030: { threshold: 0.3, accuracy: 0.58, coverage: 0.30, n: 30 } } as any,
        ],
      });

      expect(result.passes).toBe(true);
      expect(result.failureReasons).toHaveLength(0);
    });

    it('fails when overall directional accuracy regresses beyond guardrail', () => {
      const result = evaluatePhase8Thresholds({
        candidateId: 'warmup-90',
        warmup: 90,
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
      const result = evaluatePhase8Thresholds({
        candidateId: 'warmup-60',
        warmup: 60,
        candidate: {
          overall: {
            directionalAccuracy: 0.60,
            brierScore: 0.18,
            ciCoverage: 0.88,
            avgConfidence: 0.35,
            directionalCi: { lower: 0.55, median: 0.60, upper: 0.65, nResamples: 500 },
          },
          deltaVsBaseline: {
            overallDirectionalAccuracy: 0.0,
            overallBrier: 0.0,
            overallCiCoverage: -0.03,
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
            { horizon: 7, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.0, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: -0.03, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
            { horizon: 14, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.0, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
            { horizon: 30, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.01, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
          ],
        },
        baselinePerHorizon: [],
      });

      expect(result.passes).toBe(false);
      expect(result.failureReasons.some(r => r.includes('CI coverage'))).toBe(true);
    });

    it('fails warmup=90 when 14d break-context gain is below evidence expectation', () => {
      const result = evaluatePhase8Thresholds({
        candidateId: 'warmup-90',
        warmup: 90,
        candidate: {
          overall: {
            directionalAccuracy: 0.60,
            brierScore: 0.18,
            ciCoverage: 0.93,
            avgConfidence: 0.35,
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

    it('fails warmup=60 when 30d break-context gain is below evidence expectation', () => {
      const result = evaluatePhase8Thresholds({
        candidateId: 'warmup-60',
        warmup: 60,
        candidate: {
          overall: {
            directionalAccuracy: 0.60,
            brierScore: 0.18,
            ciCoverage: 0.93,
            avgConfidence: 0.35,
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
            { horizon: 14, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.0, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
            { horizon: 30, overallDirectionalAccuracy: 0.0, breakContextDirectionalAccuracy: 0.002, nonBreakDirectionalAccuracy: 0.0, brier: 0.0, ciCoverage: 0.0, breakContextRC020Accuracy: 0.0, breakContextRC030Accuracy: 0.0 },
          ],
        },
        baselinePerHorizon: [],
      });

      expect(result.passes).toBe(false);
      expect(result.failureReasons.some(r => r.includes('30d'))).toBe(true);
    });

    it('fails when per-horizon break-context regresses beyond max loss', () => {
      const result = evaluatePhase8Thresholds({
        candidateId: 'warmup-90',
        warmup: 90,
        candidate: {
          overall: {
            directionalAccuracy: 0.60,
            brierScore: 0.18,
            ciCoverage: 0.93,
            avgConfidence: 0.35,
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
  });

  describe('selectWinner (pure unit tests)', () => {
    it('selects the single passing candidate', () => {
      const candidates = [
        {
          candidateId: 'warmup-90',
          warmup: 90,
          passesThresholds: true,
          perHorizonDelta: [
            { horizon: 7, breakContextDirectionalAccuracy: 0.002 },
            { horizon: 14, breakContextDirectionalAccuracy: 0.01 },
            { horizon: 30, breakContextDirectionalAccuracy: 0.003 },
          ],
        },
        {
          candidateId: 'warmup-60',
          warmup: 60,
          passesThresholds: false,
          failureReasons: ['CI coverage too low'],
          perHorizonDelta: [
            { horizon: 7, breakContextDirectionalAccuracy: -0.01 },
            { horizon: 14, breakContextDirectionalAccuracy: 0.0 },
            { horizon: 30, breakContextDirectionalAccuracy: 0.006 },
          ],
        },
      ] as any;

      const winner = selectWinner(candidates);
      expect(winner.candidateId).toBe('warmup-90');
    });

    it('returns null with reason when no candidate passes', () => {
      const candidates = [
        {
          candidateId: 'warmup-90',
          warmup: 90,
          passesThresholds: false,
          failureReasons: ['regression'],
          perHorizonDelta: [],
        },
        {
          candidateId: 'warmup-60',
          warmup: 60,
          passesThresholds: false,
          failureReasons: ['regression'],
          perHorizonDelta: [],
        },
      ] as any;

      const winner = selectWinner(candidates);
      expect(winner.candidateId).toBeNull();
      expect(winner.reason).toContain('No candidate passed');
    });

    it('picks the candidate with highest sum of per-horizon break-context gains when both pass', () => {
      const candidates = [
        {
          candidateId: 'warmup-90',
          warmup: 90,
          passesThresholds: true,
          perHorizonDelta: [
            { horizon: 7, breakContextDirectionalAccuracy: 0.001 },
            { horizon: 14, breakContextDirectionalAccuracy: 0.01 },
            { horizon: 30, breakContextDirectionalAccuracy: 0.002 },
          ],
        },
        {
          candidateId: 'warmup-60',
          warmup: 60,
          passesThresholds: true,
          perHorizonDelta: [
            { horizon: 7, breakContextDirectionalAccuracy: 0.0 },
            { horizon: 14, breakContextDirectionalAccuracy: 0.002 },
            { horizon: 30, breakContextDirectionalAccuracy: 0.015 },
          ],
        },
      ] as any;

      const winner = selectWinner(candidates);
      // warmup-60 sum = 0.017, warmup-90 sum = 0.013
      expect(winner.candidateId).toBe('warmup-60');
    });
  });

  integrationIt('runs the full comparison and produces a valid artifact', async () => {
    const artifact = await runComparison();

    expect(artifact.generatedAt).toBeTruthy();
    expect(artifact.baseline.label).toBe('phase4-control');
    expect(artifact.baseline.warmup).toBe(120);
    expect(artifact.universe.tickers).toEqual(['SPY', 'QQQ', 'GLD', 'VOO', 'NVDA', 'MSFT']);
    expect(artifact.universe.horizons).toEqual([7, 14, 30]);
    expect(artifact.candidates).toHaveLength(2);
    expect(artifact.candidates[0].candidateId).toBe('warmup-90');
    expect(artifact.candidates[0].warmup).toBe(90);
    expect(artifact.candidates[1].candidateId).toBe('warmup-60');
    expect(artifact.candidates[1].warmup).toBe(60);

    for (const c of artifact.candidates) {
      expect(c.perHorizon).toHaveLength(3);
      expect(c.perHorizonDelta).toHaveLength(3);
      expect(c.deltaVsBaseline).toBeDefined();
      expect(typeof c.passesThresholds).toBe('boolean');
      if (!c.passesThresholds) {
        expect(c.failureReasons.length).toBeGreaterThan(0);
      }
    }

    expect(artifact.winner).toBeDefined();
    if (artifact.winner.candidateId) {
      expect(['warmup-90', 'warmup-60']).toContain(artifact.winner.candidateId);
    } else {
      expect(artifact.winner.reason).toBeTruthy();
    }
  }, 480_000);
});