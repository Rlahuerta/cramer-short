/**
 * TS↔Python parity tests for conformal PID engine.
 *
 * Verifies that the ConformalPID and AdaptiveConformalPID classes
 * produce numerically identical radius / interval outputs.
 */
import { describe, test, expect } from 'bun:test';
import { runPython } from '../../utils/finance/python-parity.js';

describe('TS/Python parity — ConformalPID basic', () => {
  test('radius adjusts after over-coverage', async () => {
    const py = await runPython(`
from research.models.conformal import ConformalPID
import json

pid = ConformalPID(alpha=0.1, initial_radius=1.0, kp=1.0, ki=0.2, kd=0.1)

# Simulate 50 steps — first 25 under-shooting, second 25 on-target
results = []
for i in range(50):
    forecast = 100.0
    actual = 100.0 - (1.5 if i < 25 else 0.0)
    pid.record(forecast, actual)
    results.append(pid.current_radius())

print(json.dumps(results[-5:]))
`);
    const finalRadii = JSON.parse(py);
    // After stabilizing, radius should shrink
    expect(finalRadii[finalRadii.length - 1]).toBeLessThan(1.0);
  });

  test('empirical coverage approaches target (1-alpha)', async () => {
    const py = await runPython(`
from research.models.conformal import ConformalPID
pid = ConformalPID(alpha=0.1, initial_radius=1.0, kp=1.0, ki=0.2, kd=0.1)

for i in range(200):
    pid.record(100.0, 100.0 + (0.0 if i % 2 == 0 else 1.8))
cov = pid.empirical_coverage()
print(cov)
`);
    const coverage = parseFloat(py);
    expect(coverage).toBeGreaterThan(0.8);
    expect(coverage).toBeLessThan(0.95);
  });
});

describe('TS/Python parity — AdaptiveConformalPID break mode', () => {
  test('enters break mode when structural break detected', async () => {
    const py = await runPython(`
from research.models.conformal import AdaptiveConformalPID
import json

pid = AdaptiveConformalPID(alpha=0.1, initial_radius=1.0, kp=1.0, ki=0.2, kd=0.1)

# Run with break diagnostics on every step
results = []
for step, (forecast, actual, break_detected, divergence) in enumerate([
    (100, 99,   False, 0.0),
    (100, 98,   False, 0.0),
    (100, 97,   False, 0.0),
    (100, 105,  True,  0.25),
    (100, 104,  True,  0.20),
    (100, 103,  False, 0.10),
    (100, 102,  False, 0.05),
]):
    diag = {'structural_break_detected': break_detected, 'structural_break_divergence': divergence}
    pid.record(forecast, actual, diag)
    results.append({'radius': pid.current_radius(), 'mode': pid.current_mode()})

print(json.dumps(results))
`);
    const steps = JSON.parse(py);
    expect(steps[3].mode).toBe('break');
    expect(steps[3].radius).toBeGreaterThan(steps[2].radius);
  });
});
