/**
 * TS / Python parity tests for RND integration.
 *
 * These tests spawn the Python implementation and compare
 * outputs to the TypeScript mirror for identical inputs.
 */

import { describe, test, expect } from 'bun:test';
import { transformQToP, fitLognormalFromStrikes, lognormalToRegimeProbabilities, nudgeTransitionMatrix } from './rnd-integration.js';

function runPython(script: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const proc = Bun.spawn({
      cmd: ['python3', '-c', script],
      cwd: '/home/hephaestus/NAS/Repositories/dexter',
      env: { ...process.env, PYTHONPATH: '/home/hephaestus/NAS/Repositories/dexter' },
      stdout: 'pipe',
      stderr: 'pipe',
    });
    const stdout: string[] = [];
    const stderr: string[] = [];
    proc.stdout.pipeTo(new WritableStream({ write(chunk) { stdout.push(new TextDecoder().decode(chunk)); } }));
    proc.stderr.pipeTo(new WritableStream({ write(chunk) { stderr.push(new TextDecoder().decode(chunk)); } }));
    proc.exited.then((code) => {
      if (code !== 0) {
        reject(new Error(`Python exited ${code}: ${stderr.join('')}`));
      } else {
        resolve(stdout.join('').trim());
      }
    });
  });
}

describe('TS/Python parity — transform_q_to_p', () => {
  test('identity case', async () => {
    const ts = transformQToP(0.3, 0.05, 0.05, 0.5, 30);
    const py = await runPython(`
from research.models.rnd import transform_q_to_p
print(transform_q_to_p(0.3, 0.05, 0.05, 0.5, 30))
    `);
    expect(ts).toBeCloseTo(parseFloat(py), 6);
  });

  test('bullish shift', async () => {
    const ts = transformQToP(0.3, 0.40, 0.05, 0.5, 30);
    const py = await runPython(`
from research.models.rnd import transform_q_to_p
print(transform_q_to_p(0.3, 0.40, 0.05, 0.5, 30))
    `);
    expect(ts).toBeCloseTo(parseFloat(py), 6);
  });

  test('extreme probability', async () => {
    const ts = transformQToP(0.0001, 0.40, 0.05, 0.5, 30);
    const py = await runPython(`
from research.models.rnd import transform_q_to_p
print(transform_q_to_p(0.0001, 0.40, 0.05, 0.5, 30))
    `);
    expect(ts).toBeCloseTo(parseFloat(py), 6);
  });

  test('mpr cap parity (default 1.5)', async () => {
    const ts = transformQToP(0.30, 3.0, 0.05, 0.3, 30);
    const py = await runPython(`
from research.models.rnd import transform_q_to_p
print(transform_q_to_p(0.30, 3.0, 0.05, 0.3, 30))
    `);
    expect(ts).toBeCloseTo(parseFloat(py), 6);
  });

  test('explicit mpr cap parity', async () => {
    const ts = transformQToP(0.30, 0.50, 0.05, 0.30, 30, 0.1);
    const py = await runPython(`
from research.models.rnd import transform_q_to_p
print(transform_q_to_p(0.30, 0.50, 0.05, 0.30, 30, 0.1))
    `);
    expect(ts).toBeCloseTo(parseFloat(py), 6);
  });
});

describe('TS/Python parity — lognormal fit + regime mapping', () => {
  test('recovers same parameters', async () => {
    const strikes = [40000, 45000, 50000, 55000, 60000];
    const yesPrices = [0.85, 0.65, 0.45, 0.25, 0.10];

    const tsFit = fitLognormalFromStrikes(strikes, yesPrices, 50000);
    const tsRegime = lognormalToRegimeProbabilities(tsFit.muLn, tsFit.sigmaLn, 50000);

    const py = await runPython(`
import json
from research.models.rnd import fit_lognormal_from_strikes, lognormal_to_regime_probabilities
mu, sigma = fit_lognormal_from_strikes([40000, 45000, 50000, 55000, 60000], [0.85, 0.65, 0.45, 0.25, 0.10], 50000)
regimes = lognormal_to_regime_probabilities(mu, sigma, 50000)
print(json.dumps({"mu": mu, "sigma": sigma, "regimes": regimes}))
    `);

    const parsed = JSON.parse(py);
    expect(tsFit.muLn).toBeCloseTo(parsed.mu, 2);
    expect(tsFit.sigmaLn).toBeCloseTo(parsed.sigma, 2);
    expect(tsRegime.bull).toBeCloseTo(parsed.regimes.bull, 2);
    expect(tsRegime.bear).toBeCloseTo(parsed.regimes.bear, 2);
    expect(tsRegime.sideways).toBeCloseTo(parsed.regimes.sideways, 2);
  });
});

describe('TS/Python parity — nudge_transition_matrix', () => {
  test('same nudge for identical inputs', async () => {
    const P = [
      [0.7, 0.2, 0.1],
      [0.2, 0.6, 0.2],
      [0.1, 0.2, 0.7],
    ];
    const target = { bull: 0.60, bear: 0.25, sideways: 0.15 };

    const tsNudged = nudgeTransitionMatrix(P, 'bull', target, 7, 80);

    const py = await runPython(`
import json
import numpy as np
from research.models.rnd import nudge_transition_matrix
P = np.array([[0.7, 0.2, 0.1], [0.2, 0.6, 0.2], [0.1, 0.2, 0.7]])
result = nudge_transition_matrix(P, "bull", {"bull": 0.60, "bear": 0.25, "sideways": 0.15}, 7, 80)
print(json.dumps(result.tolist()))
    `);

    const pyNudged = JSON.parse(py) as number[][];
    for (let i = 0; i < 3; i++) {
      for (let j = 0; j < 3; j++) {
        expect(tsNudged[i][j]).toBeCloseTo(pyNudged[i][j], 6);
      }
    }
  });
});
