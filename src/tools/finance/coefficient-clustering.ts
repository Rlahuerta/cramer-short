export interface CoefficientSegment {
  start: number;
  end: number;
  coefficients: number[];
  residualVariance: number;
}

export interface CoefficientClusterAssignment {
  cluster: number;
  probabilities: number[];
  distance: number;
}

export interface CoefficientClusteringDiagnostics {
  segmentCount: number;
  occupiedClusters: number;
  currentCluster: number;
  currentProbabilities: number[];
  currentCoefficients: number[];
  centroids: number[][];
  assignments: CoefficientClusterAssignment[];
}

export interface CoefficientClusteringOptions {
  windowSize?: number;
  stride?: number;
  clusterCount?: number;
  ridge?: number;
  iterations?: number;
}

function buildFeatureRow(values: number[], index: number): number[] {
  const lag1 = values[index - 1];
  const lag2 = values[index - 2];
  const lag3 = values[index - 3];
  const mean3 = (lag1 + lag2 + lag3) / 3;
  return [1, lag1, lag2, lag3, mean3, Math.abs(lag1)];
}

function solveLinearSystem(matrix: number[][], rhs: number[]): number[] {
  const n = matrix.length;
  const aug = matrix.map((row, i) => [...row, rhs[i]]);
  for (let col = 0; col < n; col++) {
    let pivot = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(aug[row][col]) > Math.abs(aug[pivot][col])) pivot = row;
    }
    if (Math.abs(aug[pivot][col]) < 1e-12) continue;
    if (pivot !== col) [aug[col], aug[pivot]] = [aug[pivot], aug[col]];
    const scale = aug[col][col];
    for (let j = col; j <= n; j++) aug[col][j] /= scale;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = aug[row][col];
      if (Math.abs(factor) < 1e-12) continue;
      for (let j = col; j <= n; j++) aug[row][j] -= factor * aug[col][j];
    }
  }
  return aug.map((row) => row[n]);
}

function euclideanDistanceSquared(a: number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < a.length; i++) sum += (a[i] - b[i]) ** 2;
  return sum;
}

function softmax(values: number[]): number[] {
  const max = Math.max(...values);
  const exps = values.map((value) => Math.exp(value - max));
  const total = exps.reduce((sum, value) => sum + value, 0);
  return exps.map((value) => value / total);
}

export function segmentReturnWindows(
  returns: number[],
  windowSize = 48,
  stride = 24,
): Array<{ start: number; end: number; values: number[] }> {
  const segments: Array<{ start: number; end: number; values: number[] }> = [];
  if (windowSize < 8 || stride < 1 || returns.length < windowSize) return segments;
  for (let start = 0; start + windowSize <= returns.length; start += stride) {
    const end = start + windowSize;
    segments.push({ start, end, values: returns.slice(start, end) });
  }
  if (segments.length === 0 || segments.at(-1)!.end !== returns.length) {
    segments.push({
      start: Math.max(0, returns.length - windowSize),
      end: returns.length,
      values: returns.slice(-windowSize),
    });
  }
  return segments;
}

export function fitSegmentCoefficients(
  returns: number[],
  ridge = 1e-6,
): { coefficients: number[]; residualVariance: number } {
  if (returns.length < 12) {
    throw new Error('need at least 12 returns to fit segment coefficients');
  }
  const X: number[][] = [];
  const y: number[] = [];
  for (let i = 3; i < returns.length; i++) {
    X.push(buildFeatureRow(returns, i));
    y.push(returns[i]);
  }
  const p = X[0].length;
  const xtx = Array.from({ length: p }, () => Array(p).fill(0));
  const xty = Array(p).fill(0);
  for (let row = 0; row < X.length; row++) {
    for (let i = 0; i < p; i++) {
      xty[i] += X[row][i] * y[row];
      for (let j = 0; j < p; j++) xtx[i][j] += X[row][i] * X[row][j];
    }
  }
  for (let i = 0; i < p; i++) xtx[i][i] += ridge;
  const coefficients = solveLinearSystem(xtx, xty);
  let sse = 0;
  for (let row = 0; row < X.length; row++) {
    const fitted = coefficients.reduce((sum, beta, index) => sum + beta * X[row][index], 0);
    sse += (y[row] - fitted) ** 2;
  }
  return {
    coefficients,
    residualVariance: sse / Math.max(1, X.length - p),
  };
}

export function clusterCoefficientSegments(
  segments: CoefficientSegment[],
  clusterCount = 3,
  iterations = 10,
): { centroids: number[][]; assignments: CoefficientClusterAssignment[]; occupiedClusters: number } {
  if (segments.length === 0) {
    return { centroids: [], assignments: [], occupiedClusters: 0 };
  }
  const vectors = segments.map((segment) => [...segment.coefficients, segment.residualVariance]);
  const k = Math.max(1, Math.min(clusterCount, vectors.length));
  const sorted = [...vectors].sort((a, b) => a[1] - b[1]);
  let centroids = Array.from({ length: k }, (_, index) => {
    const pos = Math.round((index * (sorted.length - 1)) / Math.max(1, k - 1));
    return [...sorted[pos]];
  });

  let assignments: CoefficientClusterAssignment[] = [];
  for (let iter = 0; iter < iterations; iter++) {
    assignments = vectors.map((vector) => {
      const distances = centroids.map((centroid) => euclideanDistanceSquared(vector, centroid));
      const cluster = distances.indexOf(Math.min(...distances));
      const temperature = Math.max(1e-6, distances.reduce((sum, value) => sum + value, 0) / Math.max(1, distances.length));
      const probabilities = softmax(distances.map((distance) => -distance / temperature));
      return {
        cluster,
        probabilities,
        distance: Math.sqrt(distances[cluster]),
      };
    });

    const nextCentroids = centroids.map((centroid) => Array(centroid.length).fill(0));
    const counts = Array(k).fill(0);
    assignments.forEach((assignment, index) => {
      counts[assignment.cluster]++;
      for (let j = 0; j < vectors[index].length; j++) {
        nextCentroids[assignment.cluster][j] += vectors[index][j];
      }
    });
    centroids = nextCentroids.map((centroid, index) =>
      counts[index] > 0
        ? centroid.map((value) => value / counts[index])
        : centroids[index],
    );
  }

  const occupiedClusters = new Set(assignments.map((assignment) => assignment.cluster)).size;
  return { centroids, assignments, occupiedClusters };
}

export function buildCoefficientClusteringDiagnostics(
  returns: number[],
  options: CoefficientClusteringOptions = {},
): CoefficientClusteringDiagnostics | null {
  const windowSize = options.windowSize ?? 48;
  const stride = options.stride ?? 24;
  const clusterCount = options.clusterCount ?? 3;
  const ridge = options.ridge ?? 1e-6;
  const iterations = options.iterations ?? 10;

  const windows = segmentReturnWindows(returns, windowSize, stride);
  if (windows.length < 2) return null;

  const segments = windows.map((window) => {
    const fitted = fitSegmentCoefficients(window.values, ridge);
    return {
      start: window.start,
      end: window.end,
      coefficients: fitted.coefficients,
      residualVariance: fitted.residualVariance,
    };
  });
  const clustered = clusterCoefficientSegments(segments, clusterCount, iterations);
  const currentAssignment = clustered.assignments.at(-1);
  const currentSegment = segments.at(-1);
  if (!currentAssignment || !currentSegment) return null;

  return {
    segmentCount: segments.length,
    occupiedClusters: clustered.occupiedClusters,
    currentCluster: currentAssignment.cluster,
    currentProbabilities: currentAssignment.probabilities,
    currentCoefficients: currentSegment.coefficients,
    centroids: clustered.centroids,
    assignments: clustered.assignments,
  };
}
