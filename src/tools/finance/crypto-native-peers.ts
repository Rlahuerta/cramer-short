import { readFileSync } from 'node:fs';

/**
 * R5 Sprint 2 Idea #1 — Crypto-native peer-asset bias loader.
 *
 * Uses fixture-backed daily closes for ETH / SOL / MSTR / COIN and aligns them
 * to the BTC daily axis. Equity peers do not trade on weekends, so their closes
 * are forward-filled after their first available date to keep the aligned return
 * vectors usable for the walk-forward cross-asset Lasso path.
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

interface FixtureTickerData {
  dates: string[];
  closes: number[];
}

interface PriceFixture {
  tickers: Record<string, FixtureTickerData>;
}

const ANCHOR_FIXTURE_PATH = new URL('./fixtures/backtest-prices.json', import.meta.url);
const PEER_FIXTURE_PATH = new URL('./fixtures/crypto-peer-prices.json', import.meta.url);

function readFixture(path: URL, label: string): PriceFixture {
  try {
    return JSON.parse(readFileSync(path, 'utf8')) as PriceFixture;
  } catch (error) {
    throw new CryptoPeerLoaderUnavailable(`failed to read ${label} fixture: ${String(error)}`);
  }
}

function loadTickerData(
  fixture: PriceFixture,
  ticker: string,
  label: string,
): FixtureTickerData {
  const item = fixture.tickers?.[ticker];
  if (!item || !Array.isArray(item.dates) || !Array.isArray(item.closes) || item.dates.length !== item.closes.length) {
    throw new CryptoPeerLoaderUnavailable(`missing ${ticker} in ${label} fixture`);
  }
  return item;
}

function computeLogReturns(closes: number[]): number[] {
  const returns: number[] = [];
  for (let i = 1; i < closes.length; i++) {
    returns.push(Math.log(closes[i] / closes[i - 1]));
  }
  return returns;
}

function alignPeerClosesToAnchorDates(
  anchorDates: string[],
  peer: FixtureTickerData,
  sharedStartDate: string,
): { dates: string[]; closes: number[] } {
  const alignedDates = anchorDates.filter((date) => date >= sharedStartDate);
  if (alignedDates.length < 2) {
    throw new CryptoPeerLoaderUnavailable(`insufficient overlap after aligning ${sharedStartDate}`);
  }

  let peerIndex = 0;
  let lastClose: number | undefined;
  const alignedCloses: number[] = [];

  for (const date of alignedDates) {
    while (peerIndex < peer.dates.length && peer.dates[peerIndex] <= date) {
      lastClose = peer.closes[peerIndex];
      peerIndex++;
    }
    if (lastClose === undefined) {
      throw new CryptoPeerLoaderUnavailable(`cannot align peer series before first close on ${date}`);
    }
    alignedCloses.push(lastClose);
  }

  return { dates: alignedDates, closes: alignedCloses };
}

/**
 * Load crypto-native peer returns aligned to a BTC-like anchor date axis.
 *
 * Returned `dates` align with the return endpoints, so `dates.length === returns[ticker].length`.
 *
 * @param peers   — list of crypto-native peer tickers to load.
 * @param anchor  — ticker whose date axis the peers must align to (default 'BTC-USD').
 */
export function loadCryptoPeerReturns(
  peers: CryptoPeerTicker[],
  anchor: string = 'BTC-USD',
): CryptoPeerLoaderResult {
  if (peers.length === 0) {
    throw new CryptoPeerLoaderUnavailable('peer list is empty');
  }

  const anchorFixture = readFixture(ANCHOR_FIXTURE_PATH, 'anchor');
  const peerFixture = readFixture(PEER_FIXTURE_PATH, 'peer');
  const anchorData = loadTickerData(anchorFixture, anchor, 'anchor');
  const peerDataByTicker = Object.fromEntries(
    peers.map((peer) => [peer, loadTickerData(peerFixture, peer, 'peer')]),
  ) as Record<CryptoPeerTicker, FixtureTickerData>;
  const sharedStartDate = [anchorData.dates[0], ...peers.map((peer) => peerDataByTicker[peer].dates[0])]
    .sort()
    .at(-1);
  if (!sharedStartDate) {
    throw new CryptoPeerLoaderUnavailable('cannot determine shared start date');
  }

  const returns: Record<string, number[]> = {};
  let alignedReturnDates: string[] | undefined;

  for (const peer of peers) {
    const aligned = alignPeerClosesToAnchorDates(anchorData.dates, peerDataByTicker[peer], sharedStartDate);
    const alignedReturns = computeLogReturns(aligned.closes);
    const returnDates = aligned.dates.slice(1);

    if (!alignedReturnDates) {
      alignedReturnDates = returnDates;
    } else if (
      alignedReturnDates.length !== returnDates.length
      || alignedReturnDates.some((date, index) => date !== returnDates[index])
    ) {
      throw new CryptoPeerLoaderUnavailable(`aligned dates for ${peer} do not match the shared anchor axis`);
    }

    returns[peer] = alignedReturns;
  }

  if (!alignedReturnDates || alignedReturnDates.length === 0) {
    throw new CryptoPeerLoaderUnavailable('no aligned return dates available');
  }

  return {
    returns,
    dates: alignedReturnDates,
  };
}
