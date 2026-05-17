/**
 * TS↔Python parity tests for Polymarket weighted ensemble engine.
 *
 * Verifies that pure-math ensemble functions produce numerically
 * identical outputs across TypeScript and Python.
 */
import { describe, test, expect } from 'bun:test';
import { runPython } from '../../utils/finance/python-parity.js';
import { adjustYesBias, adjustYesBiasV2, depthDecayHaircut, YES_BIAS_MULTIPLIER } from './ensemble.js';

const PYTHON_PARITY_TIMEOUT_MS = 20_000;

describe('TS/Python parity — adjust_yes_bias', () => {
  test('p=0.60 → 0.95x (basic multiplier)', async () => {
    const ts = adjustYesBias(0.60, 0.035);
    const py = await runPython(`
from research.utils.calibration import adjust_yes_bias
print(adjust_yes_bias(0.60, 0.035))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
  }, PYTHON_PARITY_TIMEOUT_MS);

  test('p=0.90, p=0.10, p=0.50', async () => {
    for (const p of [0.90, 0.10, 0.50]) {
      const ts = adjustYesBias(p);
      const py = await runPython(`
from research.utils.calibration import adjust_yes_bias
print(adjust_yes_bias(${p}))
`);
      expect(ts).toBeCloseTo(parseFloat(py), 10);
    }
  }, PYTHON_PARITY_TIMEOUT_MS);
});

describe('TS/Python parity — adjust_yes_bias_v2 (U-shaped)', () => {
  test('extreme longshots pulled toward center', async () => {
    const ts = adjustYesBiasV2(0.02);
    const py = await runPython(`
from research.utils.calibration import adjust_yes_bias_v2
print(adjust_yes_bias_v2(0.02))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
// Strong longshot discount should reduce overstated tail probabilities.
expect(ts).toBeLessThan(0.02);
  }, PYTHON_PARITY_TIMEOUT_MS);

  test('near-favourites also corrected', async () => {
    const ts = adjustYesBiasV2(0.98);
    const py = await runPython(`
from research.utils.calibration import adjust_yes_bias_v2
print(adjust_yes_bias_v2(0.98))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    expect(ts).toBeLessThan(0.98);
  }, PYTHON_PARITY_TIMEOUT_MS);

  test('neutral region (0.4-0.6) left alone', async () => {
    const ts = adjustYesBiasV2(0.45);
    const py = await runPython(`
from research.utils.calibration import adjust_yes_bias_v2
print(adjust_yes_bias_v2(0.45))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    expect(ts).toBeCloseTo(0.45, 3);
  }, PYTHON_PARITY_TIMEOUT_MS);
});

describe('TS/Python parity — depth_decay_haircut', () => {
  test('same-day expiry → floor weight', async () => {
    const ts = depthDecayHaircut(0);
    const py = await runPython(`
from research.models.ensemble import depth_decay_haircut
print(depth_decay_haircut(0))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    expect(ts).toBe(0.5);
  }, PYTHON_PARITY_TIMEOUT_MS);

  test('90-day depth → no haircut beyond the 30-day reference horizon', async () => {
    const ts = depthDecayHaircut(90);
    const py = await runPython(`
from research.models.ensemble import depth_decay_haircut
print(depth_decay_haircut(90))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    expect(ts).toBe(1.0);
  }, PYTHON_PARITY_TIMEOUT_MS);

  test('undefined depth → no penalty', async () => {
    const ts = depthDecayHaircut(undefined);
    const py = await runPython(`
from research.models.ensemble import depth_decay_haircut
print(depth_decay_haircut(None))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    expect(ts).toBe(1.0);
  }, PYTHON_PARITY_TIMEOUT_MS);
});

describe('TS/Python parity — YES_BIAS_MULTIPLIER', () => {
  test('constant matches Python', async () => {
    const py = await runPython(`
from research.utils.calibration import YES_BIAS_MULTIPLIER
print(YES_BIAS_MULTIPLIER)
`);
    expect(YES_BIAS_MULTIPLIER).toBeCloseTo(parseFloat(py), 10);
  }, PYTHON_PARITY_TIMEOUT_MS);
});
