import { describe, expect, it } from 'bun:test';
import {
  buildCoefficientClusteringDiagnostics,
  clusterCoefficientSegments,
  fitSegmentCoefficients,
  segmentReturnWindows,
} from './coefficient-clustering.js';

function generateArSeries(phi: number, sigma: number, length: number): number[] {
  const series: number[] = [];
  let prev = 0.004;
  for (let i = 0; i < length; i++) {
    const noise = sigma * Math.sin((i + 1) * 1.73);
    const next = phi * prev + noise;
    series.push(next);
    prev = next;
  }
  return series;
}

describe('segmentReturnWindows', () => {
  it('creates fixed windows and includes the trailing window', () => {
    const values = Array.from({ length: 120 }, (_, index) => Math.sin(index / 10) * 0.01);
    const windows = segmentReturnWindows(values, 40, 30);
    expect(windows.length).toBe(4);
    expect(windows[0]).toMatchObject({ start: 0, end: 40 });
    expect(windows.at(-1)).toMatchObject({ start: 80, end: 120 });
  });
});

describe('fitSegmentCoefficients', () => {
  it('separates momentum-like and mean-reverting segments', () => {
    const momentum = fitSegmentCoefficients(generateArSeries(0.82, 0.003, 48));
    const meanReverting = fitSegmentCoefficients(generateArSeries(-0.7, 0.003, 48));

    expect(momentum.coefficients[1]).toBeGreaterThan(0.2);
    expect(meanReverting.coefficients[1]).toBeLessThan(-0.15);
  });
});

describe('clusterCoefficientSegments', () => {
  it('is deterministic across repeated runs on the same segments', () => {
    const regimeA = generateArSeries(0.8, 0.002, 48);
    const regimeB = generateArSeries(-0.65, 0.002, 48);
    const segments = [
      { start: 0, end: 48, ...fitSegmentCoefficients(regimeA) },
      { start: 48, end: 96, ...fitSegmentCoefficients(regimeA.map((value) => value * 0.9)) },
      { start: 96, end: 144, ...fitSegmentCoefficients(regimeB) },
      { start: 144, end: 192, ...fitSegmentCoefficients(regimeB.map((value) => value * 1.1)) },
    ];

    const first = clusterCoefficientSegments(segments, 2, 8);
    const second = clusterCoefficientSegments(segments, 2, 8);

    expect(first.assignments.map((assignment) => assignment.cluster)).toEqual(
      second.assignments.map((assignment) => assignment.cluster),
    );
    for (const assignment of first.assignments) {
      const probabilitySum = assignment.probabilities.reduce((sum, value) => sum + value, 0);
      expect(probabilitySum).toBeCloseTo(1, 8);
    }
  });
});

describe('buildCoefficientClusteringDiagnostics', () => {
  it('finds multiple occupied clusters on a mixed-regime series', () => {
    const mixed = [
      ...generateArSeries(0.82, 0.002, 64),
      ...generateArSeries(-0.75, 0.0025, 64),
      ...generateArSeries(0.15, 0.006, 64),
    ];

    const diagnostics = buildCoefficientClusteringDiagnostics(mixed, {
      windowSize: 32,
      stride: 16,
      clusterCount: 3,
    });

    expect(diagnostics).not.toBeNull();
    expect(diagnostics?.segmentCount).toBeGreaterThanOrEqual(10);
    expect(diagnostics?.occupiedClusters).toBeGreaterThanOrEqual(2);
    expect(diagnostics?.currentProbabilities.length).toBe(3);
  });
});
