// ---------------------------------------------------------------------------
// Asset-class parameter profiles (Idea N)
// ---------------------------------------------------------------------------

/**
 * Per-asset-class parameter overrides. Different asset types have fundamentally
 * different volatility, mean-reversion, and regime-switching characteristics.
 * Using identical parameters for SPY (vol ~1.2%) and BTC (vol ~4%) is incorrect.
 */
export interface AssetProfile {
  /** Asset class identifier */
  type: 'etf' | 'equity' | 'crypto' | 'commodity';
  /** Calibration kappa multiplier (1.0 = default). Higher = more conservative. */
  kappaMultiplier: number;
  /** HMM weight multiplier (1.0 = default). Lower = trust HMM less. */
  hmmWeightMultiplier: number;
  /** Student-t degrees of freedom (lower = fatter tails) */
  studentTNu: number;
  /** Transition matrix decay rate */
  decayRate: number;
  /** Maximum absolute daily drift (caps regime meanReturn to prevent shock contamination) */
  maxDailyDrift?: number;
}

const ASSET_PROFILES: Record<AssetProfile['type'], AssetProfile> = {
  etf: {
    type: 'etf',
    kappaMultiplier: 0.85,     // ETFs are more predictable → trust model more
    hmmWeightMultiplier: 1.1,  // HMM works well on smoother series
    studentTNu: 5,
    decayRate: 0.97,
    maxDailyDrift: 0.008,      // ~2% annualized cap
  },
  equity: {
    type: 'equity',
    kappaMultiplier: 1.0,      // baseline
    hmmWeightMultiplier: 0.9,  // slightly less HMM trust (more idiosyncratic)
    studentTNu: 4,             // fatter tails than ETFs
    decayRate: 0.96,
    maxDailyDrift: 0.012,      // ~3% annualized cap
  },
  crypto: {
    type: 'crypto',
    kappaMultiplier: 1.3,      // crypto is noisier → more shrinkage toward base rate
    hmmWeightMultiplier: 0.5,  // HMM less reliable on crypto noise
    studentTNu: 3,             // fattest tails
    decayRate: 0.94,
    maxDailyDrift: 0.025,      // crypto can legitimately drift more
  },
  commodity: {
    type: 'commodity',
    kappaMultiplier: 1.1,      // commodities are driven by supply shocks → slightly more conservative
    hmmWeightMultiplier: 0.7,  // regime switching is real but noisy (geopolitics)
    studentTNu: 4,             // fat tails from supply shocks
    decayRate: 0.95,
    maxDailyDrift: 0.010,      // ~2.5% annualized; prevents geopolitical shock drift contamination
  },
};

/**
 * Map ticker symbol to asset profile. Falls back to 'equity' for unknown tickers.
 * Common ETFs and crypto tickers are recognized by pattern.
 */
export function getAssetProfile(ticker: string): AssetProfile {
  const t = ticker.toUpperCase();
  // Crypto detection
  if (t.includes('BTC') || t.includes('ETH') || t.includes('SOL') ||
      t.includes('DOGE') || t.includes('XRP') || t.endsWith('-USD') ||
      t.endsWith('USDT') || t.includes('CRYPTO')) {
    return ASSET_PROFILES.crypto;
  }
  // Commodity futures detection (CME/NYMEX/COMEX tickers + common names)
  const commodityTickers = new Set([
    'CL', 'NG', 'HO', 'RB',          // energy: crude, nat gas, heating oil, gasoline
    'GC', 'SI', 'HG', 'PL', 'PA',    // metals: gold, silver, copper, platinum, palladium
    'ZC', 'ZW', 'ZS', 'ZM', 'ZL',    // grains: corn, wheat, soybeans, soybean meal/oil
    'CT', 'KC', 'SB', 'CC', 'OJ',    // softs: cotton, coffee, sugar, cocoa, OJ
    'LE', 'HE', 'GF',                 // livestock: live cattle, lean hogs, feeder cattle
    'WTICOUSD', 'BRENTUSD',           // spot oil aliases
    'SILVER', 'COPPER',               // common names for precious/base metals
    'XAUUSD', 'XAGUSD',              // forex-style spot metal symbols
    'NATGAS', 'CRUDE', 'OIL',        // informal energy names
  ]);
  if (commodityTickers.has(t)) return ASSET_PROFILES.commodity;
  // Commodity ETFs — use commodity profile (they track commodity prices)
  const commodityEtfs = new Set([
    'USO', 'UNG', 'DBO', 'GSG', 'DJP', 'PDBC',  // broad/energy commodity ETFs
    'GLD', 'SLV', 'IAU', 'SGOL', 'PPLT',         // precious metal ETFs
    'CPER', 'JJC',                                  // copper ETFs
    'DBA', 'WEAT', 'CORN', 'SOYB',                // agriculture ETFs
  ]);
  if (commodityEtfs.has(t)) return ASSET_PROFILES.commodity;
  // ETF detection (common US ETFs and patterns)
  const etfTickers = new Set([
    'SPY', 'QQQ', 'IWM', 'DIA', 'VOO', 'VTI', 'VXUS', 'EFA', 'EEM',
    'TLT', 'IEF', 'SHY', 'LQD', 'HYG',
    'XLF', 'XLK', 'XLE', 'XLV', 'XLI', 'XLP', 'XLY', 'XLU', 'XLB',
    'ARKK', 'ARKG', 'ARKW', 'SOXL', 'TQQQ', 'SQQQ', 'SPXL', 'VGK',
    'IEMG', 'AGG', 'BND', 'SCHD', 'VYM', 'JEPI', 'VNQ', 'XLRE',
  ]);
  if (etfTickers.has(t)) return ASSET_PROFILES.etf;

  return ASSET_PROFILES.equity;
}
