import { describe, expect, it } from 'bun:test';

import { assertAssetConsistency, resolveAssetIntent, resolveTickerSearchIdentity } from './asset-resolver.js';

describe('resolveAssetIntent', () => {
  it('routes commodity gold queries to GLD proxy', () => {
    const result = resolveAssetIntent('gold price forecast next month');
    expect(result.resolvedTicker).toBe('GLD');
    expect(result.assetClass).toBe('commodity_gold');
  });

  it('routes bare GOLD to GLD proxy by default', () => {
    const result = resolveAssetIntent('GOLD 30-day forecast', 'GOLD');
    expect(result.resolvedTicker).toBe('GLD');
    expect(result.assetClass).toBe('commodity_gold');
  });

  it('routes explicit Barrick context to GOLD equity', () => {
    const result = resolveAssetIntent('Barrick Gold stock forecast', 'GOLD');
    expect(result.resolvedTicker).toBe('GOLD');
    expect(result.assetClass).toBe('gold_miner');
  });

  it('routes silver commodity queries to SLV proxy', () => {
    const result = resolveAssetIntent('silver price outlook');
    expect(result.resolvedTicker).toBe('SLV');
    expect(result.assetClass).toBe('commodity_silver');
  });

  it('passes through non-commodity explicit tickers', () => {
    const result = resolveAssetIntent('AAPL forecast', 'AAPL');
    expect(result.resolvedTicker).toBe('AAPL');
    expect(result.assetClass).toBe('ticker');
  });

  it('keeps explicit non-GOLD miner tickers ahead of Barrick wording heuristics', () => {
    const result = resolveAssetIntent('GDX gold miner ETF forecast', 'GDX');
    expect(result.resolvedTicker).toBe('GDX');
    expect(result.assetClass).toBe('ticker');
  });
});

describe('assertAssetConsistency', () => {
  it('rejects commodity gold routed to Barrick ticker', () => {
    const intent = resolveAssetIntent('gold forecast', 'GOLD');
    expect(() => assertAssetConsistency(intent, 'get_stock_price', 'GOLD')).toThrow();
  });

  it('rejects Barrick equity routed to GLD proxy', () => {
    const intent = resolveAssetIntent('Barrick Gold earnings', 'GOLD');
    expect(() => assertAssetConsistency(intent, 'get_stock_price', 'GLD')).toThrow();
  });

  it('rejects commodity silver routed to SILVER pseudo-ticker', () => {
    const intent = resolveAssetIntent('silver forecast', 'SILVER');
    expect(() => assertAssetConsistency(intent, 'get_stock_price', 'SILVER')).toThrow();
  });
});

describe('resolveTickerSearchIdentity', () => {
  it('keeps explicit GOLD ticker tied to Barrick Gold search identity', () => {
    const result = resolveTickerSearchIdentity('GOLD');
    expect(result.canonicalTicker).toBe('GOLD');
    expect(result.searchQuery).toBe('Barrick Gold');
    expect(result.canonicalNames).toEqual(['barrick gold', 'barrick']);
    expect(result.strictQuestionMatch).toBe(true);
  });

  it('maps GLD to commodity gold search identity', () => {
    const result = resolveTickerSearchIdentity('GLD');
    expect(result.canonicalTicker).toBe('GLD');
    expect(result.searchQuery).toBe('gold');
    expect(result.canonicalNames).toEqual(['gold', 'gld']);
    expect(result.strictQuestionMatch).toBe(false);
  });

  it('maps SLV to commodity silver search identity', () => {
    const result = resolveTickerSearchIdentity('SLV');
    expect(result.canonicalTicker).toBe('SLV');
    expect(result.searchQuery).toBe('silver');
    expect(result.canonicalNames).toEqual(['silver', 'slv']);
    expect(result.strictQuestionMatch).toBe(false);
  });
});
