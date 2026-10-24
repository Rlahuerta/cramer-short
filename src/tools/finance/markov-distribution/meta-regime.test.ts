import { describe, expect, it } from 'bun:test';
import { MetaRegimeDetector } from './meta-regime.js';

describe('MetaRegimeDetector', () => {
  it('classifies rolling volatility at the threshold as low uncertainty', () => {
    const detector = new MetaRegimeDetector(2, 0);

    const result = detector.fit([0, 0, 0.1]);

    expect(detector.fitted).toBe(true);
    expect(result.threshold_used).toBe(0);
    expect(result.environment_sequence[1]).toBe('low_uncertainty');
  });

  it('classifies rolling volatility above the threshold as high uncertainty', () => {
    const detector = new MetaRegimeDetector(2, 0);

    const result = detector.fit([0, 0, 0.1]);

    expect(result.environment_sequence[2]).toBe('high_uncertainty');
    expect(result.current_environment).toBe('high_uncertainty');
  });

  it('uses the fallback threshold when data is insufficient for rolling volatility', () => {
    const detector = new MetaRegimeDetector(20, 0.75);

    const result = detector.fit([0.03]);

    expect(detector.threshold).toBe(0.02);
    expect(result.threshold_used).toBe(0.02);
    expect(result.environment_sequence).toEqual(['low_uncertainty']);
    expect(result.current_environment).toBe('low_uncertainty');
  });

  it('uses the fallback threshold for empty data', () => {
    const detector = new MetaRegimeDetector(20, 0.75);

    const result = detector.fit([]);

    expect(detector.threshold).toBe(0.02);
    expect(result.threshold_used).toBe(0.02);
    expect(result.environment_sequence).toEqual([]);
    expect(result.current_environment).toBeUndefined();
  });
});
