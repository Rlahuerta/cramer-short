/**
 * R5 Sprint 2 Idea #1 — Crypto-native peer-asset bias (STUB).
 *
 * Source: docs/forecast-improvement-ideas-round5-2026-04-29.md
 *
 * Status: interface-only stub.  The Round-4 cross-asset Lasso bias uses
 * SPY / GLD / QQQ as peer assets; this module reserves the contract for
 * crypto-native peers (ETH, SOL, MSTR, COIN) once daily-close fixtures
 * are added under `references/` or `exports/`.
 *
 * Once fixtures exist, drop a real implementation that:
 *   1. Loads peer daily closes aligned to BTC's date axis.
 *   2. Returns a `crossAssetReturns` map suitable for `WalkForwardConfig`.
 *
 * For now this throws on use so callers fail loudly instead of silently
 * applying wrong defaults.
 */

export type CryptoPeerTicker = 'ETH-USD' | 'SOL-USD' | 'MSTR' | 'COIN';

export interface CryptoPeerLoaderResult {
  /** Map of peer ticker → aligned daily log-returns vector. */
  returns: Record<string, number[]>;
  /** Date axis used for alignment (ISO YYYY-MM-DD), aligned to the BTC anchor. */
  dates: string[];
}

export class CryptoPeerLoaderUnavailable extends Error {
  constructor(reason: string) {
    super(`crypto peer loader unavailable: ${reason}`);
    this.name = 'CryptoPeerLoaderUnavailable';
  }
}

/**
 * Reserved API. Throws until fixtures are added.
 *
 * @param _peers   — list of crypto-native peer tickers to load.
 * @param _anchor  — ticker whose date axis the peers must align to (default 'BTC-USD').
 */
export function loadCryptoPeerReturns(
  _peers: CryptoPeerTicker[],
  _anchor: string = 'BTC-USD',
): CryptoPeerLoaderResult {
  throw new CryptoPeerLoaderUnavailable(
    'no daily-close fixtures for ETH/SOL/MSTR/COIN under references/ or exports/.\n' +
    '  Add fixtures and replace this stub with a real loader before enabling crypto peer bias.',
  );
}
