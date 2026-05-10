import { describe, expect, it } from 'bun:test';
import {
  createForecastLabAssetScopedRuntimeDefaults,
  resolveForecastLabRuntimeAssetScopeForTicker,
} from './forecast-lab-runtime-defaults.js';

describe('forecast-lab runtime defaults', () => {
  it('maps BTC, GOLD, SOL, and HYPE tickers into the expected runtime scopes', () => {
    expect(resolveForecastLabRuntimeAssetScopeForTicker('BTC')).toBe('btc');
    expect(resolveForecastLabRuntimeAssetScopeForTicker('BTC-USD')).toBe('btc');
    expect(resolveForecastLabRuntimeAssetScopeForTicker('GLD')).toBe('gold');
    expect(resolveForecastLabRuntimeAssetScopeForTicker('SOL')).toBe('sol');
    expect(resolveForecastLabRuntimeAssetScopeForTicker('SOL-USD')).toBe('sol');
    expect(resolveForecastLabRuntimeAssetScopeForTicker('SOLUSD')).toBe('sol');
    expect(resolveForecastLabRuntimeAssetScopeForTicker('SOLUSDT')).toBe('sol');
    expect(resolveForecastLabRuntimeAssetScopeForTicker('HYPE')).toBe('hype');
    expect(resolveForecastLabRuntimeAssetScopeForTicker('HYPE-USD')).toBe('hype');
    expect(resolveForecastLabRuntimeAssetScopeForTicker('HYPEUSD')).toBe('hype');
    expect(resolveForecastLabRuntimeAssetScopeForTicker('HYPEUSDT')).toBe('hype');
    expect(resolveForecastLabRuntimeAssetScopeForTicker('QQQ')).toBe('shared');
  });

  it('keeps SOL and HYPE runtime overrides isolated from shared overrides while GOLD still inherits shared', () => {
    const defaults = createForecastLabAssetScopedRuntimeDefaults({
      momentumLookback: 10,
      transitionDecay: 0.97,
    });

    defaults.set('shared', {
      momentumLookback: 20,
    });
    defaults.set('gold', {
      transitionDecay: 0.95,
    });
    defaults.set('sol', {
      momentumLookback: 7,
    });
    defaults.set('hype', {
      transitionDecay: 0.94,
    });

    expect(defaults.resolve('shared')).toEqual({
      momentumLookback: 20,
      transitionDecay: 0.97,
    });
    expect(defaults.resolve('gold')).toEqual({
      momentumLookback: 20,
      transitionDecay: 0.95,
    });
    expect(defaults.resolve('sol')).toEqual({
      momentumLookback: 7,
      transitionDecay: 0.97,
    });
    expect(defaults.resolve('hype')).toEqual({
      momentumLookback: 10,
      transitionDecay: 0.94,
    });
  });

  it('does not leak BTC or GOLD overrides into SOL/HYPE scopes, and vice versa', () => {
    const defaults = createForecastLabAssetScopedRuntimeDefaults({
      momentumLookback: 10,
      transitionDecay: 0.97,
    });

    defaults.set('btc', {
      momentumLookback: 5,
    });
    defaults.set('gold', {
      transitionDecay: 0.95,
    });
    defaults.set('sol', {
      momentumLookback: 7,
      transitionDecay: 0.93,
    });
    defaults.set('hype', {
      momentumLookback: 8,
      transitionDecay: 0.92,
    });

    expect(defaults.resolve('btc')).toEqual({
      momentumLookback: 5,
      transitionDecay: 0.97,
    });
    expect(defaults.resolve('gold')).toEqual({
      momentumLookback: 10,
      transitionDecay: 0.95,
    });
    expect(defaults.resolve('sol')).toEqual({
      momentumLookback: 7,
      transitionDecay: 0.93,
    });
    expect(defaults.resolve('hype')).toEqual({
      momentumLookback: 8,
      transitionDecay: 0.92,
    });
  });
});
