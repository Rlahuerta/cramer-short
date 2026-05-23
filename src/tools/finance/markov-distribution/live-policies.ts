import { resolveTickerSearchIdentity } from '../asset-resolver.js';

export function normalizeHistoricalPriceTicker(ticker: string): string {
  const upper = ticker.trim().toUpperCase();
  switch (upper) {
    case 'OIL':
    case 'WTICOUSD':
    case 'CRUDE':
      // USO is a futures ETF with contango decay — not a direct spot-crude proxy.
      return 'USO';
    default:
      return upper;
  }
}

const BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS = 252;
const BTC_SHORT_HORIZON_LIVE_RERUN_WINDOW_DAYS = 60;
const BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT = 0.15;
const GOLD_SHORT_HORIZON_LIVE_HISTORY_DAYS = 252;
const GOLD_ULTRA_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT = 0.12;
const GOLD_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT = 0.15;

export function isBtcTickerSymbol(ticker: string): boolean {
  const upper = ticker.trim().toUpperCase();
  return upper === 'BTC' || upper === 'BTC-USD';
}

export interface BtcShortHorizonLivePolicy {
  historyDays: number;
  breakDivergenceThreshold: number;
  rerunOnBreak: boolean;
  rerunWindowDays?: number;
}

export function getBtcShortHorizonLivePolicy(
  ticker: string,
  horizon: number,
): BtcShortHorizonLivePolicy | null {
  if (!isBtcTickerSymbol(ticker) || horizon < 1 || horizon > 14) return null;

  if (horizon === 1) {
    return {
      historyDays: BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
      breakDivergenceThreshold: 0.10,
      rerunOnBreak: true,
      rerunWindowDays: BTC_SHORT_HORIZON_LIVE_RERUN_WINDOW_DAYS,
    };
  }

  if (horizon === 2) {
    return {
      historyDays: BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
      breakDivergenceThreshold: BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
      rerunOnBreak: true,
      rerunWindowDays: 120,
    };
  }

  if (horizon === 3) {
    return {
      historyDays: BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
      breakDivergenceThreshold: BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
      rerunOnBreak: true,
      rerunWindowDays: 45,
    };
  }

  return {
    historyDays: BTC_SHORT_HORIZON_LIVE_HISTORY_DAYS,
    breakDivergenceThreshold: horizon === 14 ? 0.08 : BTC_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
    rerunOnBreak: false,
  };
}

export interface GoldShortHorizonLivePolicy {
  historyDays: number;
  breakDivergenceThreshold: number;
  rerunOnBreak: false;
}

export function getGoldShortHorizonLivePolicy(
  ticker: string,
  horizon: number,
): GoldShortHorizonLivePolicy | null {
  if (horizon < 1 || horizon > 14) return null;
  if (resolveTickerSearchIdentity(ticker).canonicalTicker !== 'GLD') return null;

  return {
    historyDays: GOLD_SHORT_HORIZON_LIVE_HISTORY_DAYS,
    breakDivergenceThreshold: horizon <= 3
      ? GOLD_ULTRA_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT
      : GOLD_SHORT_HORIZON_LIVE_BREAK_THRESHOLD_DEFAULT,
    rerunOnBreak: false,
  };
}
