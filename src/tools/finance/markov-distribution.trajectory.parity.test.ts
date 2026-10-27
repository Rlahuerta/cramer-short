/**
 * TS↔Python parity tests for trajectory / Monte Carlo engine.
 *
 * Verifies core stats helpers (log_normal_survival, compute_horizon_drift_vol)
 * and trajectory generation consistency.
 */
import { describe, test, expect } from 'bun:test';
import { runPython, enablePythonParity, pythonParityAvailable } from '../../utils/finance/python-parity.js';

enablePythonParity();
// Skip (with a warning) when the cramer-research conda env is absent; REQUIRE_PARITY=1 forces a run.
const describeParity = describe.skipIf(!pythonParityAvailable());
import {
  computeTrajectory,
  estimateRegimeStats,
  logNormalSurvival,
  studentTSurvival,
} from './markov-distribution/confidence-intervals.js';
import type { RegimeState } from './markov-distribution/core.js';

const PYTHON_PARITY_TIMEOUT_MS = 30_000;
const SURVIVAL_PARITY_DIGITS = 6;

describeParity('TS/Python parity — log_normal_survival', () => {
  test('at-the-money, 21-day horizon', async () => {
    const ts = logNormalSurvival(100, 100, 0.10, 0.25);
    const py = await runPython(`
from research.models.trajectory import log_normal_survival
print(log_normal_survival(100, 100, 0.10, 0.25))
`);
    expect(ts).toBeCloseTo(parseFloat(py), SURVIVAL_PARITY_DIGITS);
    // Positive log-space drift lifts ATM survival above a coin flip.
    expect(ts).toBeGreaterThan(0.5);
    expect(ts).toBeLessThan(0.7);
  }, PYTHON_PARITY_TIMEOUT_MS);

  test('deep OTM, negative drift', async () => {
    const ts = logNormalSurvival(100, 120, -0.05, 0.30);
    const py = await runPython(`
from research.models.trajectory import log_normal_survival
print(log_normal_survival(100, 120, -0.05, 0.30))
`);
    expect(ts).toBeCloseTo(parseFloat(py), SURVIVAL_PARITY_DIGITS);
    expect(ts).toBeLessThan(0.3);
  }, PYTHON_PARITY_TIMEOUT_MS);
});

describeParity('TS/Python parity — student_t_survival', () => {
  test('sigma_n is the t standard deviation (variance-matched scale)', async () => {
    // Python takes (target, current); TS takes (current, target).
    const ts = studentTSurvival(100, 153.46, 0, 0.15275, 5);
    const py = await runPython(`
from research.models.trajectory import student_t_survival
print(student_t_survival(153.46, 100, 0.0, 0.15275, 5))
`);
    expect(ts).toBeCloseTo(parseFloat(py), SURVIVAL_PARITY_DIGITS);
    expect(ts).toBeCloseTo(0.0076, 4);
  }, PYTHON_PARITY_TIMEOUT_MS);
});

describeParity('TS/Python parity — estimate_regime_stats', () => {
  test('consistent across TS and Python', async () => {
    const returns = [
      0.02, 0.01, 0.03, 0.015, 0.025, 0.018,
      -0.02, -0.015, -0.025, -0.01, -0.03, -0.018,
      0.001, -0.002, 0.0, 0.003, -0.001, 0.002,
    ];
    const states: RegimeState[] = [
      'bull', 'bull', 'bull', 'bull', 'bull', 'bull',
      'bear', 'bear', 'bear', 'bear', 'bear', 'bear',
      'sideways', 'sideways', 'sideways', 'sideways', 'sideways', 'sideways',
    ];
    const tsStats = estimateRegimeStats(returns, states, 0.05);

    const py = await runPython(`
from research.models.markov import estimate_regime_stats
import json

returns = ${JSON.stringify(returns)}
states = ${JSON.stringify(states)}
result = estimate_regime_stats(returns, states, 0.05)
print(json.dumps(result, sort_keys=True))
`);
    const pyStats = JSON.parse(py);
    for (const state of ['bull', 'bear', 'sideways'] as const) {
      expect(tsStats[state].meanReturn).toBeCloseTo(pyStats[state].meanReturn, 12);
      expect(tsStats[state].stdReturn).toBeCloseTo(pyStats[state].stdReturn, 12);
    }
  }, PYTHON_PARITY_TIMEOUT_MS);
});

describeParity('TS/Python parity — trajectory CI width guard', () => {
  test('low-vol downtrend: widths never narrow, the upper bound is not pinned to its early peak, and both languages agree', async () => {
    const stats = {
      bull: { meanReturn: 0.01, stdReturn: 0.01 },
      bear: { meanReturn: -0.004, stdReturn: 0.01 },
      sideways: { meanReturn: 0, stdReturn: 0.01 },
    };
    const identity = [[1, 0, 0], [0, 1, 0], [0, 0, 1]];
    const ts = computeTrajectory(
      100, 30, identity, stats, 'bear', 0, undefined, 2000, 5,
      undefined, undefined, undefined, undefined, 1.0, 7,
    ).map((p) => [p.lowerBound, p.upperBound]);

    const py = await runPython(`
from research.models.trajectory import RegimeStats, compute_trajectory
import json
import numpy as np

stats = {
    "bull": RegimeStats(mean_return=0.01, std_return=0.01),
    "bear": RegimeStats(mean_return=-0.004, std_return=0.01),
    "sideways": RegimeStats(mean_return=0.0, std_return=0.01),
}
traj = compute_trajectory(100.0, 30, np.eye(3), stats, "bear", n_samples=2000, random_state=7)
print(json.dumps([[p.lower_bound, p.upper_bound] for p in traj]))
`);
    const pyBands = JSON.parse(py) as number[][];

    for (const bands of [ts, pyBands]) {
      const widths = bands.map(([lo, hi]) => hi - lo);
      for (let i = 1; i < widths.length; i++) expect(widths[i]).toBeGreaterThanOrEqual(widths[i - 1] - 1e-9);
      // Width grows every day, but the true upper quantile peaks early (~101.7) and then falls.
      expect(bands[29][1]).toBeLessThan(Math.max(...bands.map(([, hi]) => hi)) - 2);
    }
    // The RNGs differ, so compare the sampled day-30 bands within Monte Carlo tolerance.
    expect(Math.abs(ts[29][0] - pyBands[29][0])).toBeLessThan(1.5);
    expect(Math.abs(ts[29][1] - pyBands[29][1])).toBeLessThan(1.5);
  }, 120_000);
});
