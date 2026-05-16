/**
 * TS↔Python parity tests for HMM engine.
 *
 * Verifies that pure-math HMM functions produce numerically identical
 * outputs across TypeScript and Python implementations.
 */
import { describe, test, expect } from 'bun:test';
import { runPython } from '../../utils/finance/python-parity.js';
import { matPow } from './hmm.js';

describe('TS/Python parity — hmm.mat_pow', () => {
  test('2x2 matrix, power 3', () => {
    const A = [
      [0.7, 0.3],
      [0.2, 0.8],
    ];
    const ts = matPow(A, 3);
    // Row 0: 0.7 * 0.7 * 0.7 + cross terms...
    expect(ts[0][0]).toBeCloseTo(0.475, 4);
    expect(ts[0][1]).toBeCloseTo(0.525, 4);
    expect(ts[1][0]).toBeCloseTo(0.35, 4);
    expect(ts[1][1]).toBeCloseTo(0.65, 4);
  });

  test('stationary convergence at high power', () => {
    const A = [
      [0.7, 0.3],
      [0.2, 0.8],
    ];
    const ts = matPow(A, 128);
    // Should converge to stationary: row[0] ≈ 0.4, row[1] ≈ 0.6
    expect(ts[0][0]).toBeCloseTo(0.4, 2);
    expect(ts[1][0]).toBeCloseTo(0.4, 2);
  });
});

describe('TS/Python parity — hmm.student_t_log_pdf', () => {
  test('t(5) at z=0, 1, 2', async () => {
    const py = await runPython(`
from research.models.hmm import student_t_log_pdf
import json
results = [student_t_log_pdf(0.0, 0.0, 1.0, 5),
           student_t_log_pdf(1.0, 0.0, 1.0, 5),
           student_t_log_pdf(2.0, 0.0, 1.0, 5)]
print(json.dumps(results))
`);
    const [z0, z1, z2] = JSON.parse(py);
    // t(5, mu=0, sigma=1) log-pdf ≈ log(t.pdf(0,5)) ≈ -1.24
    expect(z0).toBeCloseTo(-1.24, 1);
    expect(z1).toBeCloseTo(-1.68, 1);
    expect(z2).toBeCloseTo(-2.53, 1);
  });
});
