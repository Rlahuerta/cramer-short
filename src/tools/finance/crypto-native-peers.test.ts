import { describe, expect, it } from 'bun:test';
import {
  CryptoPeerLoaderUnavailable,
  loadCryptoPeerReturns,
} from './crypto-native-peers.js';

describe('loadCryptoPeerReturns', () => {
  it('loads and aligns ETH/SOL/MSTR/COIN returns to a shared BTC axis', () => {
    const peers = ['ETH-USD', 'SOL-USD', 'MSTR', 'COIN'] as const;
    const result = loadCryptoPeerReturns([...peers]);

    expect(result.dates.length).toBeGreaterThan(700);
    expect(Object.keys(result.returns).sort()).toEqual([...peers].sort());

    for (const peer of peers) {
      expect(result.returns[peer]).toHaveLength(result.dates.length);
      expect(result.returns[peer].every(Number.isFinite)).toBe(true);
    }
  });

  it('supports subset peer requests', () => {
    const result = loadCryptoPeerReturns(['ETH-USD']);
    expect(Object.keys(result.returns)).toEqual(['ETH-USD']);
    expect(result.returns['ETH-USD']).toHaveLength(result.dates.length);
  });

  it('forward-fills equity peers onto the BTC daily axis', () => {
    const result = loadCryptoPeerReturns(['MSTR', 'COIN']);
    expect(result.returns.MSTR).toContain(0);
    expect(result.returns.COIN).toContain(0);
  });

  it('throws when the anchor ticker is unavailable', () => {
    expect(() => loadCryptoPeerReturns(['ETH-USD'], 'DOGE-USD')).toThrow(CryptoPeerLoaderUnavailable);
  });
});
