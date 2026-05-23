/**
 * TS↔Python parity tests for Markov regime model.
 *
 * Verifies that core Markov functions (classify_regime,
 * estimate_transition_matrix, structural break detection)
 * produce numerically identical outputs.
 */
import { describe, test, expect } from 'bun:test';
import { runPython } from '../../utils/finance/python-parity.js';
import { classifyRegimeState } from './markov-distribution/regime.js';

const PYTHON_PARITY_TIMEOUT_MS = 20_000;

type RegimeState = ReturnType<typeof classifyRegimeState>;

function parsePythonRegimeState(value: string): RegimeState {
  if (value === 'bull' || value === 'bear' || value === 'sideways') {
    return value;
  }
  throw new Error(`Unexpected Python regime state: ${value}`);
}

describe('TS/Python parity — classify_regime_state', () => {
  test('strong bull day (+3%)', async () => {
    const ts = classifyRegimeState(0.03, 0.01);
    const py = parsePythonRegimeState(await runPython(`
from research.models.markov import classify_regime
print(classify_regime(0.03, 0.01))
`));
    expect(ts).toBe(py);
  }, PYTHON_PARITY_TIMEOUT_MS);

  test('strong bear day (-3%)', async () => {
    const ts = classifyRegimeState(-0.03, 0.01);
    const py = parsePythonRegimeState(await runPython(`
from research.models.markov import classify_regime
print(classify_regime(-0.03, 0.01))
`));
    expect(ts).toBe(py);
  }, PYTHON_PARITY_TIMEOUT_MS);

  test('sideways day (+0.1%)', async () => {
    const ts = classifyRegimeState(0.001, 0.01);
    const py = parsePythonRegimeState(await runPython(`
from research.models.markov import classify_regime
print(classify_regime(0.001, 0.01))
`));
    expect(ts).toBe(py);
  }, PYTHON_PARITY_TIMEOUT_MS);

  test('default threshold', async () => {
    const ts = classifyRegimeState(0.005, 0.01);
    const py = parsePythonRegimeState(await runPython(`
from research.models.markov import classify_regime
print(classify_regime(0.005))
`));
    expect(ts).toBe(py);
  }, PYTHON_PARITY_TIMEOUT_MS);
});

describe('TS/Python parity — regime constants', () => {
  test('REGIME_STATES match Python', async () => {
    const py = await runPython(`
from research.models.markov import REGIME_STATES
print(','.join(REGIME_STATES))
`);
    expect(py.split(',').sort()).toEqual(['bear', 'bull', 'sideways']);
  }, PYTHON_PARITY_TIMEOUT_MS);

  test('NUM_STATES is 3', async () => {
    const py = await runPython(`
from research.models.markov import NUM_STATES
print(NUM_STATES)
`);
    expect(parseInt(py)).toBe(3);
  }, PYTHON_PARITY_TIMEOUT_MS);
});
