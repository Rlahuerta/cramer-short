/**
 * TS↔Python parity tests for Polymarket weighted ensemble engine.
 *
 * Verifies that pure-math ensemble functions produce numerically
 * identical outputs across TypeScript and Python.
 */
import { describe, test, expect } from 'bun:test';
import { runPython } from '../../utils/finance/python-parity.js';
import { adjustYesBias, adjustYesBiasV2, depthDecayHaircut, YES_BIAS_MULTIPLIER } from './ensemble.js';

describe('TS/Python parity — adjust_yes_bias', () => {
  test('p=0.60 → 0.95x (basic multiplier)', async () => {
    const ts = adjustYesBias(0.60, 0.035);
    const py = await runPython(`
from research.utils.calibration import adjust_yes_bias
print(adjust_yes_bias(0.60, 0.035))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
  });

  test('p=0.90, p=0.10, p=0.50', async () => {
    for (const p of [0.90, 0.10, 0.50]) {
      const ts = adjustYesBias(p);
      const py = await runPython(`
from research.utils.calibration import adjust_yes_bias
print(adjust_yes_bias(${p}))
`);
      expect(ts).toBeCloseTo(parseFloat(py), 10);
    }
  });
});

describe('TS/Python parity — adjust_yes_bias_v2 (U-shaped)', () => {
  test('extreme longshots pulled toward center', async () => {
    const ts = adjustYesBiasV2(0.02);
    const py = await runPython(`
from research.utils.calibration import adjust_yes_bias_v2
print(adjust_yes_bias_v2(0.02))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    // Longshot correction should push toward 0.5
    expect(ts).toBeGreaterThan(0.02);
  });

  test('near-favourites also corrected', async () => {
    const ts = adjustYesBiasV2(0.98);
    const py = await runPython(`
from research.utils.calibration import adjust_yes_bias_v2
print(adjust_yes_bias_v2(0.98))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    expect(ts).toBeLessThan(0.98);
  });

  test('neutral region (0.4-0.6) left alone', async () => {
    const ts = adjustYesBiasV2(0.45);
    const py = await runPython(`
from research.utils.calibration import adjust_yes_bias_v2
print(adjust_yes_bias_v2(0.45))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    expect(ts).toBeCloseTo(0.45, 3);
  });
});

describe('TS/Python parity — depth_decay_haircut', () => {
  test('same-day expiry → full weight', async () => {
    const ts = depthDecayHaircut(0);
    const py = await runPython(`
from research.models.ensemble import depth_decay_haircut
print(depth_decay_haircut(0))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
  });

  test('90-day depth → decays toward zero', async () => {
    const ts = depthDecayHaircut(90);
    const py = await runPython(`
from research.models.ensemble import depth_decay_haircut
print(depth_decay_haircut(90))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    expect(ts).toBeLessThan(0.5);
    expect(ts).toBeGreaterThan(0.0);
  });

  test('null depth → no penalty', async () => {
    const ts = depthDecayHaircut(null as unknown as number | null); // null case
    const py = await runPython(`
from research.models.ensemble import depth_decay_haircut
print(depth_decay_haircut(None))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 10);
    expect(ts).toBeGreaterThan(0.5);
  });
});

describe('TS/Python parity — YES_BIAS_MULTIPLIER', () => {
  test('constant matches Python', async () => {
    const py = await runPython(`
from research.utils.calibration import YES_BIAS_MULTIPLIER
print(YES_BIAS_MULTIPLIER)
`);
    expect(YES_BIAS_MULTIPLIER).toBeCloseTo(parseFloat(py), 10);
  });
});
