/**
 * TS↔Python parity tests for jump-diffusion helpers (Idea 2).
 *
 * Verifies that pure-math functions produce numerically identical
 * outputs across the two implementations.
 */
import { describe, test, expect } from 'bun:test';
import { runPython } from '../../utils/python-parity.js';
import {
  JUMP_DEFAULTS,
  buildJumpEventSpec,
  jumpDriftCompensator,
  polymarketProbToHazard,
} from './jump-diffusion.js';

describe('TS/Python parity — polymarket_prob_to_hazard', () => {
  test('p=0.3, days=30', async () => {
    const ts = polymarketProbToHazard(0.3, 30);
    const py = await runPython(`
from research.models.jump_diffusion import polymarket_prob_to_hazard
print(polymarket_prob_to_hazard(0.3, 30))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 12);
  });

  test('p=0.85, days=14', async () => {
    const ts = polymarketProbToHazard(0.85, 14);
    const py = await runPython(`
from research.models.jump_diffusion import polymarket_prob_to_hazard
print(polymarket_prob_to_hazard(0.85, 14))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 12);
  });
});

describe('TS/Python parity — JUMP_DEFAULTS', () => {
  test('all asset classes match exactly', async () => {
    const py = await runPython(`
from research.models.jump_diffusion import JUMP_DEFAULTS
import json
print(json.dumps({k: dict(v) for k, v in JUMP_DEFAULTS.items()}))
`);
    const pyDefaults = JSON.parse(py);
    for (const cls of ['etf', 'equity', 'crypto', 'commodity']) {
      expect(pyDefaults[cls].mean_log_jump).toBeCloseTo(JUMP_DEFAULTS[cls as 'etf'].meanLogJump, 14);
      expect(pyDefaults[cls].std_log_jump).toBeCloseTo(JUMP_DEFAULTS[cls as 'etf'].stdLogJump, 14);
    }
  });
});

describe('TS/Python parity — jump_drift_compensator', () => {
  test('multi-event compensator', async () => {
    const events = [
      { id: 'a', dailyIntensity: 0.005, meanLogJump: -0.05, stdLogJump: 0.03 },
      { id: 'b', dailyIntensity: 0.01,  meanLogJump: -0.08, stdLogJump: 0.05 },
    ];
    const ts = jumpDriftCompensator(events);
    const py = await runPython(`
from research.models.jump_diffusion import JumpEventSpec, jump_drift_compensator
events = [
  JumpEventSpec(id='a', daily_intensity=0.005, mean_log_jump=-0.05, std_log_jump=0.03),
  JumpEventSpec(id='b', daily_intensity=0.01,  mean_log_jump=-0.08, std_log_jump=0.05),
]
print(jump_drift_compensator(events))
`);
    expect(ts).toBeCloseTo(parseFloat(py), 14);
  });
});

describe('TS/Python parity — build_jump_event_spec', () => {
  test('equity prior end-to-end composition', async () => {
    const ts = buildJumpEventSpec(0.3, 30, 0.10, 0.05, 0.20, JUMP_DEFAULTS.equity, 'mkt');
    const py = await runPython(`
from research.models.jump_diffusion import JUMP_DEFAULTS, build_jump_event_spec
spec = build_jump_event_spec(
  raw=0.3, horizon_days=30, historical_drift_annual=0.10,
  risk_free_rate=0.05, volatility_annual=0.20,
  prior=JUMP_DEFAULTS['equity'], id='mkt',
)
print(spec.daily_intensity)
`);
    expect(ts.dailyIntensity).toBeCloseTo(parseFloat(py), 7);
  });
});
