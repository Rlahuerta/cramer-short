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
