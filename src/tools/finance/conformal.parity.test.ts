/**
 * TS↔Python parity tests for conformal PID engine.
 *
 * Verifies that the ConformalPID and AdaptiveConformalPID classes
 * produce numerically identical radius / interval outputs.
 */
import { describe, test, expect } from 'bun:test';
import { runPython } from '../../utils/finance/python-parity.js';
import { AdaptiveConformalPID, ConformalPID } from './conformal.js';

const PYTHON_PARITY_TIMEOUT_MS = 30_000;

function expectNumberArraysClose(actual: number[], expected: number[], digits = 12): void {
  expect(actual).toHaveLength(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i], digits);
  }
}

describe('TS/Python parity — ConformalPID basic', () => {
  test('radius adjusts after over-coverage', async () => {
    const tsPid = new ConformalPID({ alpha: 0.1, initialRadius: 1.0, kp: 1.0, ki: 0.2, kd: 0.1 });
    const tsRadii: number[] = [];
    for (let i = 0; i < 50; i++) {
      tsPid.record(100.0, 100.0);
      tsRadii.push(tsPid.currentRadius());
    }

    const py = await runPython(`
from research.models.conformal import ConformalPID
import json

pid = ConformalPID(alpha=0.1, initial_radius=1.0, kp=1.0, ki=0.2, kd=0.1)

results = []
for i in range(50):
    pid.record(100.0, 100.0)
    results.append(pid.current_radius())

print(json.dumps(results))
`);
    const pyRadii = JSON.parse(py);
    expectNumberArraysClose(tsRadii, pyRadii);
    expect(tsRadii[tsRadii.length - 1]).toBeLessThan(tsRadii[0]);
    expect(tsRadii[tsRadii.length - 1]).toBeGreaterThanOrEqual(0);
  }, PYTHON_PARITY_TIMEOUT_MS);

  test('empirical coverage approaches target (1-alpha)', async () => {
    const tsPid = new ConformalPID({ alpha: 0.1, initialRadius: 1.0, kp: 1.0, ki: 0.2, kd: 0.1 });
    for (let i = 0; i < 200; i++) {
      tsPid.record(100.0, 100.0 + (i % 2 === 0 ? 0.0 : 1.8));
    }
    const tsCoverage = tsPid.empiricalCoverage();

    const py = await runPython(`
from research.models.conformal import ConformalPID
pid = ConformalPID(alpha=0.1, initial_radius=1.0, kp=1.0, ki=0.2, kd=0.1)

for i in range(200):
    pid.record(100.0, 100.0 + (0.0 if i % 2 == 0 else 1.8))
cov = pid.empirical_coverage()
print(cov)
`);
    const coverage = parseFloat(py);
    expect(tsCoverage).toBeDefined();
    expect(tsCoverage!).toBeCloseTo(coverage, 12);
    expect(tsCoverage!).toBeGreaterThan(0.8);
    expect(tsCoverage!).toBeLessThan(0.95);
  }, PYTHON_PARITY_TIMEOUT_MS);
});

describe('TS/Python parity — AdaptiveConformalPID break mode', () => {
  test('enters break mode when structural break detected', async () => {
    const events: [number, number, boolean, number][] = [
      [100, 99, false, 0.0],
      [100, 98, false, 0.0],
      [100, 97, false, 0.0],
      [100, 105, true, 0.25],
      [100, 104, true, 0.20],
      [100, 103, false, 0.10],
      [100, 102, false, 0.05],
    ];
    const tsPid = new AdaptiveConformalPID({
      alpha: 0.1,
      initialRadius: 1.0,
      kp: 1.0,
      ki: 0.2,
      kd: 0.1,
      enabled: true,
    });
    const tsSteps = events.map(([forecast, actual, structuralBreak]) => {
      tsPid.record(forecast, actual, { structuralBreak });
      return { radius: tsPid.currentRadius(), mode: tsPid.currentMode() };
    });

    const py = await runPython(`
from research.models.conformal import AdaptiveConformalPID
import json

pid = AdaptiveConformalPID(alpha=0.1, initial_radius=1.0, kp=1.0, ki=0.2, kd=0.1, enabled=True)

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
    diag = {'structural_break': break_detected, 'structural_break_divergence': divergence}
    pid.record(forecast, actual, diag)
    results.append({'radius': pid.current_radius(), 'mode': pid.current_mode()})

print(json.dumps(results))
`);
    const pySteps = JSON.parse(py);
    expect(tsSteps.map(step => step.mode)).toEqual(pySteps.map((step: { mode: string }) => step.mode));
    expectNumberArraysClose(
      tsSteps.map(step => step.radius),
      pySteps.map((step: { radius: number }) => step.radius),
    );
    expect(tsSteps[3].mode).toBe('break');
    expect(tsSteps[3].radius).toBeGreaterThan(tsSteps[2].radius);
  }, PYTHON_PARITY_TIMEOUT_MS);
});
