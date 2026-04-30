import { describe, expect, it } from 'bun:test';
import { BrierReplayCalibrator } from './brier-replay-calibrator.js';

function brierScore(rows: Array<{ probability: number; actual: number }>): number {
  if (rows.length === 0) return 0;
  return rows.reduce((sum, row) => sum + (row.probability - row.actual) ** 2, 0) / rows.length;
}

function runReplay(
  calibrator: BrierReplayCalibrator,
  rows: Array<{ raw: number; actual: number }>,
  burnInFraction = 0.5,
) {
  const rawHoldout: Array<{ probability: number; actual: number }> = [];
  const calibratedHoldout: Array<{ probability: number; actual: number }> = [];
  const rawMid: Array<{ probability: number; actual: number }> = [];
  const calibratedMid: Array<{ probability: number; actual: number }> = [];
  const holdoutStart = Math.floor(rows.length * burnInFraction);

  rows.forEach((row, index) => {
    const calibrated = calibrator.predict(row.raw);
    if (index >= holdoutStart) {
      rawHoldout.push({ probability: row.raw, actual: row.actual });
      calibratedHoldout.push({ probability: calibrated, actual: row.actual });
      if (row.raw >= 0.4 && row.raw <= 0.6) {
        rawMid.push({ probability: row.raw, actual: row.actual });
        calibratedMid.push({ probability: calibrated, actual: row.actual });
      }
    }
    calibrator.record(row.raw, row.actual);
  });

  return {
    rawHoldoutBrier: brierScore(rawHoldout),
    calibratedHoldoutBrier: brierScore(calibratedHoldout),
    rawMidBrier: brierScore(rawMid),
    calibratedMidBrier: brierScore(calibratedMid),
  };
}

describe('BrierReplayCalibrator', () => {
  it('improves holdout Brier on an overconfident mid-confidence replay', () => {
    const replay = Array.from({ length: 120 }, (_, index) => {
      const raw = index % 4 < 2 ? 0.58 : 0.42;
      const actual = index % 4 < 2 ? (index % 6 === 0 ? 1 : 0) : (index % 6 === 0 ? 0 : 1);
      return { raw, actual };
    });
    const calibrator = new BrierReplayCalibrator({
      learningRate: 0.1,
      midConfidenceWeight: 4,
      maxSlope: 3,
    });

    const result = runReplay(calibrator, replay, 0.5);

    expect(result.calibratedHoldoutBrier).toBeLessThan(result.rawHoldoutBrier);
    expect(result.calibratedMidBrier).toBeLessThan(result.rawMidBrier);
    expect(calibrator.state().slope).toBeLessThan(1);
  });

  it('stays close to identity on already calibrated replay data', () => {
    const replay = Array.from({ length: 120 }, (_, index) => {
      const raw = index % 2 === 0 ? 0.7 : 0.3;
      const actual = index % 10 < 7 ? 1 : 0;
      return index % 2 === 0 ? { raw, actual } : { raw, actual: 1 - actual };
    });
    const calibrator = new BrierReplayCalibrator({
      learningRate: 0.05,
      midConfidenceWeight: 2,
      maxSlope: 2,
    });

    const result = runReplay(calibrator, replay, 0.5);
    const state = calibrator.state();

    expect(Math.abs(state.bias)).toBeLessThan(0.2);
    expect(Math.abs(state.slope - 1)).toBeLessThan(0.2);
    expect(result.calibratedHoldoutBrier).toBeLessThanOrEqual(result.rawHoldoutBrier + 0.01);
  });
});
