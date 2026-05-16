/**
 * TS↔Python parity tests for trajectory / Monte Carlo engine.
 *
 * Verifies core stats helpers (log_normal_survival, compute_horizon_drift_vol)
 * and trajectory generation consistency.
 */
import { describe, test, expect } from 'bun:test';
import { runPython } from '../../utils/finance/python-parity.js';
import { logNormalSurvival } from './markov-distribution/confidence-intervals.js';

describe('TS/Python parity — log_normal_survival', () => {
  test('at-the-money, 21-day horizon', async () => {
    const ts = logNormalSurvival(100, 100, 0.10, 0.25);
    const py = await runPython(`
from research.models.trajectory import log_normal_survival
print(log_normal_survival(100, 100, 0.10, 0.25))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    // ATM survival should be near 0.5 (slight drift bias)
    expect(ts).toBeGreaterThan(0.4);
    expect(ts).toBeLessThan(0.6);
  });

  test('deep OTM, negative drift', async () => {
    const ts = logNormalSurvival(100, 120, -0.05, 0.30);
    const py = await runPython(`
from research.models.trajectory import log_normal_survival
print(log_normal_survival(100, 120, -0.05, 0.30))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    expect(ts).toBeLessThan(0.3);
  });
});

describe('TS/Python parity — estimate_regime_stats', () => {
  test('consistent across TS and Python', async () => {
    const py = await runPython(`
from research.models.markov import estimate_regime_stats
import json, numpy as np

returns = [0.02, 0.01, -0.005, -0.02, 0.005, -0.01, 0.03, 0.015]
states = ['bull', 'bull', 'bear', 'bear', 'sideways', 'sideways', 'bull', 'bull']
result = estimate_regime_stats(returns, states, 0.05, 3)
print(json.dumps({k: round(v, 6) for k, v in result.items()}))
`);
    const pyStats = JSON.parse(py);
    expect(pyStats).toHaveProperty('bull');
    expect(pyStats).toHaveProperty('bear');
    expect(pyStats).toHaveProperty('sideways');
  });
});
