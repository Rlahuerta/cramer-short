import { describe, expect, it } from 'bun:test';
import { classifyVolumeRegime, VolumeRegimeDetector } from './volume-regime.js';

describe('classifyVolumeRegime', () => {
  it('returns an empty list for empty volumes', () => {
    expect(classifyVolumeRegime([])).toEqual([]);
  });

  it('returns [] for empty input even with an invalid lookback (parity with Python/xiphos)', () => {
    expect(classifyVolumeRegime([], 1)).toEqual([]);
  });

  it('defaults to normal when there is not enough lookback history', () => {
    expect(classifyVolumeRegime([100, 200, 50], 3)).toEqual(['normal', 'normal', 'normal']);
  });

  it('classifies high, low, and normal using the previous rolling median', () => {
    expect(classifyVolumeRegime([100, 100, 100, 200, 50, 100], 3, 1.5)).toEqual([
      'normal',
      'normal',
      'normal',
      'high',
      'low',
      'normal',
    ]);
  });

  it('treats NaN current values as normal', () => {
    expect(classifyVolumeRegime([100, 100, 100, Number.NaN, 200], 3, 1.5)).toEqual([
      'normal',
      'normal',
      'normal',
      'normal',
      'normal',
    ]);
  });

  it('leaves points normal when the rolling median is non-positive or NaN', () => {
    expect(classifyVolumeRegime([0, 0, 0, 100, Number.NaN, 100, 100], 3, 1.5)).toEqual([
      'normal',
      'normal',
      'normal',
      'normal',
      'normal',
      'normal',
      'normal',
    ]);
  });

  it('rejects invalid lookback and threshold values', () => {
    expect(() => classifyVolumeRegime([100, 200], 1)).toThrow('lookback must be >= 2');
    expect(() => classifyVolumeRegime([100, 200], 2, 0)).toThrow('threshold_multiplier must be positive');
  });
});

describe('VolumeRegimeDetector', () => {
  it('maps short labels to volume environment names', () => {
    const detector = new VolumeRegimeDetector(3, 1.5);

    const result = detector.fit([100, 100, 100, 200, 50]);

    expect(detector.fitted).toBe(true);
    expect(result.environment_sequence).toEqual([
      'normal_volume',
      'normal_volume',
      'normal_volume',
      'high_volume',
      'low_volume',
    ]);
    expect(result.current_environment).toBe('low_volume');
  });

  it('returns normal_volume as current environment for empty input', () => {
    const detector = new VolumeRegimeDetector();

    expect(detector.fit([])).toEqual({ environment_sequence: [], current_environment: 'normal_volume' });
  });
});
