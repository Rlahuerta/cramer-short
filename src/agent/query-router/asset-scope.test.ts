/**
 * TDD tests for the multi-asset scope guard.
 *
 * Used by the forecast-tool ticker normalizer to stay conservative: when a
 * query (without an explicit "X only" override) references more than one
 * distinct tradable asset, the normalizer must not rewrite a ticker (it could
 * clobber a legitimate comparison). Exclusion clauses ("no GOLD, ETH…") are
 * ignored so they don't create false multi-asset signals.
 */
import { describe, it, expect } from 'bun:test';
import { referencesMultipleDistinctAssets } from './asset-scope.js';

describe('referencesMultipleDistinctAssets', () => {
  it('is false for a single crypto asset', () => {
    expect(referencesMultipleDistinctAssets('Give me a BTC 24h trajectory briefing with trade plan.')).toBe(false);
  });

  it('is false for a single commodity asset', () => {
    expect(referencesMultipleDistinctAssets('GOLD trajectory forecast next 5 trading days with a trade plan.')).toBe(false);
  });

  it('is false for a single equity asset', () => {
    expect(referencesMultipleDistinctAssets('NVDA earnings outlook for next week')).toBe(false);
  });

  it('is true for two distinct crypto assets', () => {
    expect(referencesMultipleDistinctAssets('Compare BTC and ETH over the next 7 days.')).toBe(true);
  });

  it('is true for two distinct equities', () => {
    expect(referencesMultipleDistinctAssets('NVDA vs AMD comparison')).toBe(true);
  });

  it('is true for two distinct commodities named in lowercase', () => {
    expect(referencesMultipleDistinctAssets('gold vs silver over the next week')).toBe(true);
  });

  it('ignores assets mentioned only inside an exclusion clause', () => {
    expect(referencesMultipleDistinctAssets('BTC briefing. No GOLD, ETH, SOL context.')).toBe(false);
  });

  it('treats an asset and its proxy as the same asset (GOLD vs GLD)', () => {
    expect(referencesMultipleDistinctAssets('gold via GLD proxy, 7 day forecast')).toBe(false);
  });

  it('is false for a macro query with no assets', () => {
    expect(referencesMultipleDistinctAssets('How is the market doing today?')).toBe(false);
  });
});
