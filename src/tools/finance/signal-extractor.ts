/**
 * Asset signal extractor.
 *
 * Rule-based (no LLM, no API calls) module that maps a query containing
 * asset identifiers (tickers, keywords) to a prioritised list of signal
 * categories. Each signal carries a normalised Polymarket search phrase,
 * fallback query variants, and a weight for the log-odds probability combiner.
 */

import {
  extractExclusiveAssetOverride,
  resolveAssetIntent,
  resolveTickerSearchIdentity,
  type ResolvedAssetClass,
} from './asset-resolver.js';

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export type AssetType =
  | 'tech_semiconductor'
  | 'tech_software'
  | 'tech_general'
  | 'healthcare'
  | 'financials'
  | 'energy'
  | 'consumer'
  | 'crypto'
  | 'commodity'
  | 'oil'
  | 'macro'
  | 'defense'
  | 'cybersecurity'
  | 'materials'
  | 'industrial'
  | 'small_cap';

export interface SignalCategory {
  name: string;
  /** Primary Polymarket search phrase (normalised: company name, no year/quarter tokens). */
  searchPhrase: string;
  /** Fallback phrases tried in order when searchPhrase returns 0 results. */
  queryVariants?: string[];
  weight: number;
  category: string;
}

export interface ExtractSignalOptions {
  horizonDays?: number;
  preferShortHorizonCryptoSignals?: boolean;
}

type SignalTemplate = {
  name: string;
  tpl: string;
  variantTpls: string[];
  weight: number;
  category: string;
};

// ---------------------------------------------------------------------------
// Ticker → company name (Polymarket uses full names, not exchange symbols)
// ---------------------------------------------------------------------------

/** Maps exchange ticker to the name used in Polymarket question titles. */
export const TICKER_TO_COMPANY_NAME: Record<string, string> = {
  // Semiconductors
  NVDA: 'NVIDIA', AMD: 'AMD', TSM: 'TSMC', TSMC: 'TSMC',
  INTC: 'Intel', QCOM: 'Qualcomm', AVGO: 'Broadcom',
  MU: 'Micron', AMAT: 'Applied Materials', LRCX: 'Lam Research',
  KLAC: 'KLA', MRVL: 'Marvell', ARM: 'Arm', ASML: 'ASML', ON: 'ON Semiconductor',
  // Software / Cloud
  MSFT: 'Microsoft', GOOGL: 'Google', GOOG: 'Google', META: 'Meta',
  AMZN: 'Amazon', ORCL: 'Oracle', CRM: 'Salesforce', NOW: 'ServiceNow',
  SNOW: 'Snowflake', PLTR: 'Palantir', ADBE: 'Adobe', SAP: 'SAP',
  TEAM: 'Atlassian', NET: 'Cloudflare', DDOG: 'Datadog',
  ZS: 'Zscaler', CRWD: 'CrowdStrike',
  // Tech general
  AAPL: 'Apple', DELL: 'Dell', HPQ: 'HP',
  // Healthcare
  PFE: 'Pfizer', MRNA: 'Moderna', LLY: 'Eli Lilly', JNJ: 'Johnson & Johnson',
  ABBV: 'AbbVie', BMY: 'Bristol Myers Squibb', MRK: 'Merck', GILD: 'Gilead',
  REGN: 'Regeneron', BIIB: 'Biogen', AMGN: 'Amgen', AZN: 'AstraZeneca',
  NVO: 'Novo Nordisk', RHHBY: 'Roche', SNY: 'Sanofi',
  // Financials
  JPM: 'JPMorgan', GS: 'Goldman Sachs', BAC: 'Bank of America', WFC: 'Wells Fargo',
  MS: 'Morgan Stanley', C: 'Citigroup', BLK: 'BlackRock', V: 'Visa',
  MA: 'Mastercard', AXP: 'American Express', SCHW: 'Charles Schwab', BRK: 'Berkshire Hathaway',
  // Energy
  XOM: 'ExxonMobil', CVX: 'Chevron', COP: 'ConocoPhillips', SLB: 'SLB',
  EOG: 'EOG Resources', OXY: 'Occidental', PSX: 'Phillips 66', VLO: 'Valero',
  MPC: 'Marathon Petroleum', BP: 'BP', SHEL: 'Shell',
  // Consumer
  WMT: 'Walmart', COST: 'Costco', TGT: 'Target', MCD: "McDonald's",
  SBUX: 'Starbucks', NKE: 'Nike', DIS: 'Disney', NFLX: 'Netflix',
  TSLA: 'Tesla', HD: 'Home Depot', LOW: "Lowe's",
  BABA: 'Alibaba', PG: 'Procter & Gamble', KO: 'Coca-Cola', PEP: 'PepsiCo',
  // Sector ETFs
  SLX: 'VanEck Steel', XME: 'SPDR Metals Mining', XLB: 'Materials Select',
  GDX: 'VanEck Gold Miners', GDXJ: 'VanEck Junior Gold Miners',
  XLI: 'Industrials Select', IWM: 'Russell 2000',
  KRE: 'KBW Regional Banking', KBE: 'SPDR Bank', XLF: 'Financial Select',
  // Tech ETFs
  QQQ: 'Nasdaq 100', QQQM: 'Nasdaq 100', XLK: 'Technology Select', VGT: 'Vanguard tech',
  // Broad-market ETFs
  SPY: 'S&P 500', VOO: 'S&P 500', IVV: 'S&P 500', VTI: 'US stock market', DIA: 'Dow Jones',
  // Crypto
  BTC: 'Bitcoin', ETH: 'Ethereum', SOL: 'Solana',
  // Commodities — proxy tickers map to commodity names; explicit GOLD ticker is Barrick Gold equity
  GOLD: 'Barrick Gold', SILVER: 'silver', COPPER: 'copper', PLATINUM: 'platinum',
  PALLADIUM: 'palladium', OIL: 'oil', CRUDE: 'oil', NATGAS: 'natural gas',
  WHEAT: 'wheat', CORN: 'corn', SOYBEAN: 'soybeans', COFFEE: 'coffee', SUGAR: 'sugar',
  GLD: 'gold', IAU: 'gold', SLV: 'silver', XAUUSD: 'gold', XAGUSD: 'silver',
  USO: 'oil', UNG: 'natural gas',
};

// ---------------------------------------------------------------------------
// Signal keywords — used by scoreMarketRelevance to filter irrelevant results
// ---------------------------------------------------------------------------

/** Keywords expected in relevant Polymarket market titles per signal category. */
export const SIGNAL_KEYWORDS: Record<string, string[]> = {
  earnings:     ['earnings', 'EPS', 'revenue', 'quarterly', 'results', 'beat', 'miss', 'profit', 'guidance'],
  macro_rates:  ['Fed', 'FOMC', 'rate', 'interest', 'Federal Reserve', 'cut', 'hike'],
  macro_growth: ['recession', 'GDP', 'growth', 'downturn', 'economic', 'contraction'],
  regulatory:   ['regulation', 'ban', 'law', 'policy', 'SEC', 'antitrust', 'fine', 'penalty'],
  fda_approval: ['FDA', 'approval', 'drug', 'trial', 'phase', 'clearance'],
  commodity:    ['oil', 'OPEC', 'price', 'barrel', 'energy', 'supply', 'crude', 'production',
                 'gold', 'silver', 'copper', 'metal', 'ounce', 'commodity', 'natural gas', 'wheat', 'corn'],
  oil_supply:   ['opec', 'production', 'cut', 'supply', 'inventory', 'spr', 'strategic petroleum', 'output'],
  geopolitical: ['war', 'conflict', 'sanction', 'Middle East', 'Russia', 'China', 'Ukraine'],
  trade_policy: ['tariff', 'trade', 'import', 'export', 'duty'],
  etf_product:     ['ETF', 'fund', 'approval', 'launch', 'spot'],
  btc_price_target: ['Bitcoin', 'BTC', 'price target', 'price level', 'reach', 'exceed'],
  supply_chain: ['supply', 'disruption', 'shortage', 'TSMC', 'chip', 'wafer'],
};

const COMMODITY_CRYPTO_MARKER_RE = /\b(bitcoin|btc|ethereum|eth|solana|sol|crypto|cryptocurrency)\b/i;

// ---------------------------------------------------------------------------
// Sector map (top ~80 tickers → AssetType)
// ---------------------------------------------------------------------------

const SECTOR_MAP: Record<string, AssetType> = {
  // Tech — Semiconductors
  NVDA: 'tech_semiconductor', AMD: 'tech_semiconductor', TSM: 'tech_semiconductor',
  TSMC: 'tech_semiconductor', INTC: 'tech_semiconductor', QCOM: 'tech_semiconductor',
  AVGO: 'tech_semiconductor', MU: 'tech_semiconductor', AMAT: 'tech_semiconductor',
  LRCX: 'tech_semiconductor', KLAC: 'tech_semiconductor', MRVL: 'tech_semiconductor',
  ARM: 'tech_semiconductor', ASML: 'tech_semiconductor', ON: 'tech_semiconductor',
  // Tech — Software / Cloud
  MSFT: 'tech_software', GOOGL: 'tech_software', GOOG: 'tech_software',
  META: 'tech_software', AMZN: 'tech_software', ORCL: 'tech_software',
  CRM: 'tech_software', NOW: 'tech_software', SNOW: 'tech_software',
  PLTR: 'tech_software', ADBE: 'tech_software', SAP: 'tech_software',
  TEAM: 'tech_software', NET: 'tech_software', DDOG: 'tech_software',
  ZS: 'cybersecurity', CRWD: 'cybersecurity', PANW: 'cybersecurity',
  FTNT: 'cybersecurity', S: 'cybersecurity',
  // Defense
  LMT: 'defense', RTX: 'defense', NOC: 'defense', GD: 'defense',
  LHX: 'defense', BA: 'defense', HII: 'defense', LDOS: 'defense',
  // Tech — General (hardware / devices)
  AAPL: 'tech_general', DELL: 'tech_general', HPQ: 'tech_general',
  // Healthcare
  PFE: 'healthcare', MRNA: 'healthcare', LLY: 'healthcare', JNJ: 'healthcare',
  ABBV: 'healthcare', BMY: 'healthcare', MRK: 'healthcare', GILD: 'healthcare',
  REGN: 'healthcare', BIIB: 'healthcare', AMGN: 'healthcare', AZN: 'healthcare',
  NVO: 'healthcare', RHHBY: 'healthcare', SNY: 'healthcare',
  // Financials
  JPM: 'financials', GS: 'financials', BAC: 'financials', WFC: 'financials',
  MS: 'financials', C: 'financials', BLK: 'financials', V: 'financials',
  MA: 'financials', AXP: 'financials', SCHW: 'financials', BRK: 'financials',
  // Energy
  XOM: 'energy', CVX: 'energy', COP: 'energy', SLB: 'energy',
  EOG: 'energy', OXY: 'energy', PSX: 'energy', VLO: 'energy',
  MPC: 'energy', BP: 'energy', SHEL: 'energy',
  // Consumer
  WMT: 'consumer', COST: 'consumer', TGT: 'consumer', MCD: 'consumer',
  SBUX: 'consumer', NKE: 'consumer', DIS: 'consumer', NFLX: 'consumer',
  TSLA: 'consumer', HD: 'consumer', LOW: 'consumer',
  BABA: 'consumer', PG: 'consumer', KO: 'consumer', PEP: 'consumer',
  // Materials / Metals ETFs
  SLX: 'materials', XME: 'materials', XLB: 'materials', GDX: 'materials', GDXJ: 'materials',
  PICK: 'materials', REMX: 'materials',
  // Commodity proxy ETFs route through commodity signals (not equity)
  GLD: 'commodity', IAU: 'commodity', SLV: 'commodity', USO: 'commodity',
  // Barrick Gold routes through materials (equity, not commodity)
  GOLD: 'materials',
  // Industrials ETFs
  XLI: 'industrial', VIS: 'industrial', PAVE: 'industrial', GWX: 'industrial',
  // Financials ETFs (KRE, KBE, XLF, etc.)
  KRE: 'financials', KBE: 'financials', XLF: 'financials', IAI: 'financials', KBWB: 'financials',
  // Small-cap ETFs
  IWM: 'small_cap', VBK: 'small_cap', IJR: 'small_cap', SLY: 'small_cap',
  // Defense ETFs
  ITA: 'defense',
  // Tech ETFs — QQQ/XLK are tech-heavy, not generic macro
  QQQ: 'tech_general', QQQM: 'tech_general', XLK: 'tech_general',
  VGT: 'tech_general', IGM: 'tech_general',
  // Broad-market ETFs — macro signals are correct
  SPY: 'macro', VOO: 'macro', VTI: 'macro', IVV: 'macro', DIA: 'macro',
  SCHB: 'macro', VT: 'macro',
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const CRYPTO_KEYWORD_RE = /\b(?:btc|bitcoin|eth|ethereum|sol|solana|crypto|defi|nft|bnb|xrp|ada|avax|doge)\b/i;
const BTC_KEYWORD_RE = /\b(?:btc|bitcoin)\b/i;
const ETH_KEYWORD_RE = /\b(?:eth|ethereum)\b/i;
const SOL_KEYWORD_RE = /\b(?:sol|solana)\b/i;

const MACRO_KEYWORDS = [
  'fed', 'fomc', 'rate cut', 'rate hike', 'cpi', 'ppi', 'gdp',
  'recession', 'inflation', 'employment', 'jobs', 'payroll',
  'tariff', 'trade war',
];

/**
 * Maps commodity keyword (lowercase) → internal ticker symbol used in
 * signal templates and TICKER_TO_COMPANY_NAME. Ordered longest-first so
 * "natural gas" is matched before "gas".
 */
const COMMODITY_KEYWORD_MAP: Array<[keyword: string, ticker: string]> = [
  ['natural gas', 'NATGAS'],
  ['xauusd',       'GOLD'],
  ['xagusd',       'SILVER'],
  ['gold',         'GOLD'],
  ['silver',       'SILVER'],
  ['copper',      'COPPER'],
  ['platinum',    'PLATINUM'],
  ['palladium',   'PALLADIUM'],
  ['crude oil',   'OIL'],
  ['crude',       'OIL'],
  ['oil price',   'OIL'],
  ['wheat',       'WHEAT'],
  ['corn',        'CORN'],
  ['soybean',     'SOYBEAN'],
  ['coffee',      'COFFEE'],
  ['sugar',       'SUGAR'],
];

function substituteTemplates(phrase: string, ticker: string): string {
  return phrase.replace(/\{ticker\}/g, ticker);
}

/**
 * Converts a raw signal phrase into a Polymarket-friendly query string by:
 * 1. Replacing the ticker symbol with the company's common name
 * 2. Stripping year (2020–2035) and quarter (Q1–Q4) tokens
 * 3. Collapsing whitespace and truncating to ≤ 4 words
 *
 * Short phrases with company names perform far better against the Gamma API's
 * keyword text-matching than long ticker+year compound strings.
 */
export function normalizeForPolymarket(phrase: string, ticker: string | null): string {
  let result = phrase;

  // Replace ticker symbol with company name
  if (ticker) {
    const name = TICKER_TO_COMPANY_NAME[ticker.toUpperCase()];
    if (name) {
      result = result.replace(new RegExp(`\\b${ticker}\\b`, 'gi'), name);
    }
  }

  // Strip 4-digit year tokens (2020–2035)
  result = result.replace(/\b20[2-3]\d\b/g, '');

  // Strip quarter tokens Q1–Q4
  result = result.replace(/\bQ[1-4]\b/gi, '');

  // Collapse multiple spaces and trim
  result = result.replace(/\s+/g, ' ').trim();

  // Truncate to 4 words max (Gamma API keyword search works best with short phrases)
  const words = result.split(' ').filter(Boolean);
  return words.length > 4 ? words.slice(0, 4).join(' ') : result;
}

/**
 * Scores how relevant a Polymarket market question is to a signal category.
 * Returns a value 0–1 (fraction of category keywords present in the question).
 * Score 0 means no category keywords matched — the market should be filtered out.
 * Unknown categories return 1 (no filtering applied).
 */
export function scoreMarketRelevance(question: string, category: string): number {
  const keywords = SIGNAL_KEYWORDS[category];
  if (!keywords || keywords.length === 0) return 1;
  const lower = question.toLowerCase();

  if (category === 'commodity' && COMMODITY_CRYPTO_MARKER_RE.test(question)) {
    return 0;
  }

  const matches = keywords.filter((kw) => lower.includes(kw.toLowerCase())).length;
  return matches / keywords.length;
}

// ---------------------------------------------------------------------------
// Detection
// ---------------------------------------------------------------------------

const BARRICK_MINER_RE = /\bbarrick\b|\bgold\s+(?:stock|equity|shares|company|earnings|revenue|miner|mining)\b|\$gold\b/i;

export function detectAssetType(query: string): { type: AssetType; ticker: string | null } {
  const lower = query.toLowerCase();
  const exclusiveTicker = extractExclusiveAssetOverride(query);
  if (exclusiveTicker) {
    const exclusiveIntent = resolveAssetIntent(query, exclusiveTicker);
    if (exclusiveIntent.assetClass === 'commodity_gold') return { type: 'commodity', ticker: 'GOLD' };
    if (exclusiveIntent.assetClass === 'commodity_silver') return { type: 'commodity', ticker: 'SILVER' };
    if (exclusiveIntent.assetClass === 'commodity_oil') return { type: 'commodity', ticker: 'OIL' };
    if (exclusiveIntent.assetClass === 'gold_miner') return { type: 'materials', ticker: 'GOLD' };

    const normalizedExclusiveTicker = exclusiveTicker.replace(/-USD$/, '');
    if (normalizedExclusiveTicker === 'BTC' || normalizedExclusiveTicker === 'ETH' || normalizedExclusiveTicker === 'SOL') {
      return { type: 'crypto', ticker: normalizedExclusiveTicker };
    }

    const type: AssetType = SECTOR_MAP[normalizedExclusiveTicker] ?? 'tech_general';
    return { type, ticker: normalizedExclusiveTicker };
  }

  // 0. Barrick Gold / gold-miner disambiguation (must precede commodity gold match)
  if (BARRICK_MINER_RE.test(query)) {
    return { type: 'materials', ticker: 'GOLD' };
  }

  // 1. Crypto keywords (check before ticker scan to avoid false positives)
  if (CRYPTO_KEYWORD_RE.test(query)) {
    if (BTC_KEYWORD_RE.test(query)) return { type: 'crypto', ticker: 'BTC' };
    if (ETH_KEYWORD_RE.test(query)) return { type: 'crypto', ticker: 'ETH' };
    if (SOL_KEYWORD_RE.test(query)) return { type: 'crypto', ticker: 'SOL' };
    return { type: 'crypto', ticker: null };
  }

  // 2. Explicit $TICKER prefix (dollar sign signals intent clearly)
  const dollarMatch = query.match(/\$([A-Z]{1,5})\b/);
  if (dollarMatch) {
    const ticker = dollarMatch[1];
    // $GOLD = Barrick Gold equity, not commodity gold
    if (ticker === 'GOLD') return { type: 'materials', ticker: 'GOLD' };
    const type: AssetType = SECTOR_MAP[ticker] ?? 'tech_general';
    return { type, ticker };
  }

  // 3. Known tickers from SECTOR_MAP
  for (const ticker of Object.keys(SECTOR_MAP)) {
    if (new RegExp(`\\b${ticker}\\b`).test(query)) {
      return { type: SECTOR_MAP[ticker], ticker };
    }
  }

  // 4. Commodity keywords (check before macro to avoid "oil price" → macro)
  for (const [keyword, commodityTicker] of COMMODITY_KEYWORD_MAP) {
    if (lower.includes(keyword)) {
      return { type: 'commodity', ticker: commodityTicker };
    }
  }

  // 5. Macro keywords
  if (MACRO_KEYWORDS.some((kw) => lower.includes(kw))) {
    return { type: 'macro', ticker: null };
  }

  return { type: 'macro', ticker: null };
}

// ---------------------------------------------------------------------------
// Signal maps — weights MUST sum to 1.0 for each asset type
// variantTpls: fallback templates tried in order when the primary returns 0 results
// ---------------------------------------------------------------------------

const SIGNAL_MAPS: Record<AssetType, SignalTemplate[]> = {
  tech_semiconductor: [
    { name: 'Earnings',          tpl: '{ticker} earnings',    variantTpls: ['{ticker}', 'semiconductor earnings'], weight: 0.35, category: 'earnings' },
    { name: 'Export Controls',   tpl: 'chip export controls', variantTpls: ['semiconductor export', 'chip export'], weight: 0.20, category: 'regulatory' },
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',         variantTpls: ['Federal Reserve rate', 'FOMC'],       weight: 0.20, category: 'macro_rates' },
    { name: 'US Recession',      tpl: 'US recession',         variantTpls: ['recession', 'economic recession'],    weight: 0.15, category: 'macro_growth' },
    { name: 'Supply Chain',      tpl: 'TSMC supply',          variantTpls: ['chip supply', 'semiconductor supply'], weight: 0.10, category: 'supply_chain' },
  ],
  tech_software: [
    { name: 'Earnings',          tpl: '{ticker} earnings',    variantTpls: ['{ticker}', 'tech earnings'],           weight: 0.35, category: 'earnings' },
    { name: 'AI Regulation',     tpl: 'AI regulation',        variantTpls: ['artificial intelligence regulation', 'AI policy'], weight: 0.20, category: 'regulatory' },
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',         variantTpls: ['Federal Reserve rate', 'FOMC'],        weight: 0.20, category: 'macro_rates' },
    { name: 'US Recession',      tpl: 'US recession',         variantTpls: ['recession', 'economic recession'],     weight: 0.15, category: 'macro_growth' },
    { name: 'Antitrust',         tpl: 'tech antitrust',       variantTpls: ['antitrust', '{ticker} antitrust'],     weight: 0.10, category: 'regulatory' },
  ],
  tech_general: [
    { name: 'Earnings',          tpl: '{ticker} earnings',    variantTpls: ['{ticker}', 'tech earnings'],          weight: 0.35, category: 'earnings' },
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',         variantTpls: ['Federal Reserve rate', 'FOMC'],       weight: 0.25, category: 'macro_rates' },
    { name: 'US Recession',      tpl: 'US recession',         variantTpls: ['recession', 'economic recession'],    weight: 0.20, category: 'macro_growth' },
    { name: 'Trade / Tariffs',   tpl: 'tariff trade war',     variantTpls: ['tariff', 'trade war'],                weight: 0.20, category: 'trade_policy' },
  ],
  healthcare: [
    { name: 'FDA Approval',        tpl: '{ticker} FDA approval', variantTpls: ['{ticker} drug', 'FDA approval'],          weight: 0.40, category: 'fda_approval' },
    { name: 'Earnings',            tpl: '{ticker} earnings',     variantTpls: ['{ticker}', 'pharma earnings'],             weight: 0.25, category: 'earnings' },
    { name: 'Drug Pricing Policy', tpl: 'drug pricing',          variantTpls: ['drug price regulation', 'Medicare drug'],  weight: 0.20, category: 'regulatory' },
    { name: 'Fed Rate Decision',   tpl: 'Fed rate cut',          variantTpls: ['Federal Reserve rate', 'FOMC'],            weight: 0.15, category: 'macro_rates' },
  ],
  financials: [
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',         variantTpls: ['Federal Reserve rate', 'FOMC'],       weight: 0.35, category: 'macro_rates' },
    { name: 'Earnings',          tpl: '{ticker} earnings',    variantTpls: ['{ticker}', 'bank earnings'],           weight: 0.30, category: 'earnings' },
    { name: 'US Recession',      tpl: 'US recession',         variantTpls: ['recession', 'economic recession'],    weight: 0.25, category: 'macro_growth' },
    { name: 'Bank Regulation',   tpl: 'bank regulation',      variantTpls: ['banking regulation', 'bank capital'], weight: 0.10, category: 'regulatory' },
  ],
  energy: [
    { name: 'Oil Price / OPEC',  tpl: 'OPEC oil production',  variantTpls: ['oil price', 'OPEC'],                  weight: 0.35, category: 'commodity' },
    { name: 'Earnings',          tpl: '{ticker} earnings',    variantTpls: ['{ticker}', 'oil earnings'],            weight: 0.25, category: 'earnings' },
    { name: 'Geopolitical',      tpl: 'Middle East conflict', variantTpls: ['oil geopolitical', 'Middle East'],    weight: 0.25, category: 'geopolitical' },
    { name: 'US Recession',      tpl: 'US recession',         variantTpls: ['recession', 'economic recession'],    weight: 0.15, category: 'macro_growth' },
  ],
  consumer: [
    { name: 'Earnings',          tpl: '{ticker} earnings',    variantTpls: ['{ticker}', 'consumer earnings'],      weight: 0.35, category: 'earnings' },
    { name: 'US Recession',      tpl: 'US recession',         variantTpls: ['recession', 'economic recession'],    weight: 0.30, category: 'macro_growth' },
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',         variantTpls: ['Federal Reserve rate', 'FOMC'],       weight: 0.20, category: 'macro_rates' },
    { name: 'Trade / Tariffs',   tpl: 'tariff trade war',     variantTpls: ['tariff', 'trade war'],                weight: 0.15, category: 'trade_policy' },
  ],
  crypto: [
    { name: 'SEC / Regulation',  tpl: 'crypto regulation',             variantTpls: ['SEC crypto', 'cryptocurrency regulation'],               weight: 0.30, category: 'regulatory' },
    { name: 'ETF / Product',     tpl: '{ticker} ETF',                  variantTpls: ['Bitcoin ETF', 'crypto ETF'],                               weight: 0.25, category: 'etf_product' },
    { name: 'BTC Price Target',  tpl: 'Bitcoin price target',          variantTpls: ['Bitcoin reach', 'Bitcoin exceed', 'BTC price level'],      weight: 0.20, category: 'btc_price_target' },
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',                 variantTpls: ['Federal Reserve rate', 'FOMC'],                            weight: 0.15, category: 'macro_rates' },
    { name: 'US Recession',      tpl: 'US recession',                  variantTpls: ['recession', 'economic recession'],                         weight: 0.10, category: 'macro_growth' },
  ],
  commodity: [
    { name: 'Price Level',       tpl: '{ticker} price',        variantTpls: ['{ticker}', 'commodity price'],           weight: 0.45, category: 'commodity' },
    { name: 'Geopolitical',      tpl: 'geopolitical conflict', variantTpls: ['geopolitical', 'Middle East conflict'],   weight: 0.25, category: 'geopolitical' },
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',          variantTpls: ['Federal Reserve rate', 'FOMC'],           weight: 0.20, category: 'macro_rates' },
    { name: 'US Recession',      tpl: 'US recession',          variantTpls: ['recession', 'economic recession'],        weight: 0.10, category: 'macro_growth' },
  ],
  oil: [
    { name: 'Price Level',       tpl: '{ticker} price',        variantTpls: ['{ticker}', 'commodity price', 'crude oil', 'WTI', 'Brent oil'], weight: 0.35, category: 'commodity' },
    { name: 'OPEC / Supply',     tpl: 'OPEC oil production',   variantTpls: ['OPEC', 'oil supply', 'oil inventory', 'SPR release'],      weight: 0.25, category: 'oil_supply' },
    { name: 'Geopolitical',      tpl: 'geopolitical conflict', variantTpls: ['geopolitical', 'Middle East conflict', 'Iran sanctions'],  weight: 0.20, category: 'geopolitical' },
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',          variantTpls: ['Federal Reserve rate', 'FOMC'],                            weight: 0.10, category: 'macro_rates' },
    { name: 'US Recession',      tpl: 'US recession',          variantTpls: ['recession', 'economic recession'],                         weight: 0.10, category: 'macro_growth' },
  ],
  macro: [
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',          variantTpls: ['Federal Reserve rate', 'FOMC'],         weight: 0.35, category: 'macro_rates' },
    { name: 'US Recession',      tpl: 'US recession',          variantTpls: ['recession', 'economic recession'],      weight: 0.35, category: 'macro_growth' },
    { name: 'Trade / Tariffs',   tpl: 'tariff trade war',      variantTpls: ['tariff', 'trade war'],                  weight: 0.20, category: 'trade_policy' },
    { name: 'Geopolitical',      tpl: 'geopolitical conflict', variantTpls: ['geopolitical', 'conflict war'],         weight: 0.10, category: 'geopolitical' },
  ],
  defense: [
    { name: 'Military Conflict', tpl: 'military conflict war', variantTpls: ['defense spending', 'NATO budget'],      weight: 0.40, category: 'geopolitical' },
    { name: 'Defense Budget',    tpl: 'US defense spending',   variantTpls: ['Pentagon budget', 'military budget'],   weight: 0.30, category: 'government_budget' },
    { name: 'Earnings',          tpl: '{ticker} earnings',     variantTpls: ['{ticker}', 'defense earnings'],         weight: 0.20, category: 'earnings' },
    { name: 'US Recession',      tpl: 'US recession',          variantTpls: ['recession', 'economic recession'],      weight: 0.10, category: 'macro_growth' },
  ],
  cybersecurity: [
    { name: 'Cyberattack',       tpl: 'cyberattack infrastructure', variantTpls: ['ransomware', 'nation-state hacker'], weight: 0.40, category: 'geopolitical' },
    { name: 'Regulation / CISA', tpl: 'cybersecurity regulation',   variantTpls: ['CISA', 'cyber executive order'],     weight: 0.25, category: 'regulatory' },
    { name: 'Earnings',          tpl: '{ticker} earnings',          variantTpls: ['{ticker}', 'cybersecurity earnings'], weight: 0.25, category: 'earnings' },
    { name: 'US Recession',      tpl: 'US recession',               variantTpls: ['recession', 'economic recession'],   weight: 0.10, category: 'macro_growth' },
  ],
  // US tariffs PROTECT domestic steel/metals producers → deltaYes is POSITIVE for tariff_increase
  materials: [
    { name: 'Trade / Tariffs',   tpl: 'steel tariff',          variantTpls: ['metal tariff', 'trade war steel'],          weight: 0.40, category: 'tariff_increase' },
    { name: 'US Recession',      tpl: 'US recession',          variantTpls: ['recession', 'economic recession'],           weight: 0.25, category: 'macro_growth' },
    { name: 'Commodity Prices',  tpl: 'steel price',           variantTpls: ['metal price', 'copper price'],               weight: 0.20, category: 'commodity' },
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',          variantTpls: ['Federal Reserve rate', 'FOMC'],              weight: 0.15, category: 'macro_rates' },
  ],
  // Industrials: supply-chain sensitive, highly cyclical
  industrial: [
    { name: 'Trade / Tariffs',   tpl: 'tariff trade war',      variantTpls: ['tariff', 'trade war'],                       weight: 0.30, category: 'tariff_increase' },
    { name: 'US Recession',      tpl: 'US recession',          variantTpls: ['recession', 'economic recession'],           weight: 0.30, category: 'macro_growth' },
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',          variantTpls: ['Federal Reserve rate', 'FOMC'],              weight: 0.25, category: 'macro_rates' },
    { name: 'Geopolitical',      tpl: 'geopolitical conflict', variantTpls: ['trade disruption', 'supply chain'],          weight: 0.15, category: 'geopolitical' },
  ],
  // Small-cap: rate-sensitive (floating-rate debt), more domestic-focused
  small_cap: [
    { name: 'US Recession',      tpl: 'US recession',          variantTpls: ['recession', 'economic recession'],           weight: 0.35, category: 'macro_growth' },
    { name: 'Fed Rate Decision', tpl: 'Fed rate cut',          variantTpls: ['Federal Reserve rate', 'FOMC'],              weight: 0.30, category: 'macro_rates' },
    { name: 'Trade / Tariffs',   tpl: 'tariff trade war',      variantTpls: ['tariff', 'trade war'],                       weight: 0.20, category: 'tariff_increase' },
    { name: 'Geopolitical',      tpl: 'geopolitical conflict', variantTpls: ['geopolitical', 'conflict war'],              weight: 0.15, category: 'geopolitical' },
  ],
};

function normalizeSignalPhrases(rawPhrases: string[], ticker: string | null): string[] {
  const phrases: string[] = [];
  const seen = new Set<string>();

  for (const rawPhrase of rawPhrases) {
    const phrase = normalizeForPolymarket(rawPhrase, ticker);
    if (!phrase || seen.has(phrase)) continue;
    seen.add(phrase);
    phrases.push(phrase);
  }

  return phrases;
}

function buildSignalsFromTemplates(
  templates: SignalTemplate[],
  effectiveTicker: string,
  extraNames: string[],
): SignalCategory[] {
  return templates.map((t) => {
    const rawPhrase = substituteTemplates(t.tpl, effectiveTicker);
    const searchPhrase = normalizeForPolymarket(rawPhrase, effectiveTicker);
    const queryVariants = normalizeSignalPhrases(
      t.variantTpls.map((vTpl) => substituteTemplates(vTpl, effectiveTicker)),
      effectiveTicker,
    );
    for (const name of extraNames) {
      const variant = normalizeForPolymarket(substituteTemplates(t.tpl, name), name);
      if (variant !== searchPhrase && !queryVariants.includes(variant)) {
        queryVariants.push(variant);
      }
    }
    return { name: t.name, searchPhrase, queryVariants, weight: t.weight, category: t.category };
  });
}

function buildShortHorizonCryptoSignals(horizonDays: number): SignalCategory[] {
  const targetDate = new Date(Date.now() + horizonDays * 86_400_000);
  const targetMonthDay = targetDate.toLocaleString('en-US', { month: 'short', day: 'numeric' }).replace(',', '');
  const targetWeekday = targetDate.toLocaleString('en-US', { weekday: 'long' });
  const relativePhrase = horizonDays === 1 ? 'tomorrow' : `in ${horizonDays} days`;
  const templates: SignalTemplate[] = [
    {
      name: 'BTC Terminal Threshold',
      tpl: 'Bitcoin above',
      variantTpls: [
        'Bitcoin below',
        `Bitcoin above ${targetMonthDay}`,
        `Bitcoin below ${targetMonthDay}`,
        'Bitcoin reach',
        'Bitcoin exceed',
      ],
      weight: 0.34,
      category: 'btc_price_target',
    },
    {
      name: 'BTC Near-Expiry Price',
      tpl: 'Bitcoin price',
      variantTpls: [
        `Bitcoin ${targetMonthDay}`,
        `Bitcoin price ${targetMonthDay}`,
        `Bitcoin ${targetWeekday}`,
        `Bitcoin ${relativePhrase}`,
      ],
      weight: 0.24,
      category: 'btc_price_target',
    },
    {
      name: 'BTC Narrow Phrase',
      tpl: `Bitcoin ${relativePhrase}`,
      variantTpls: [
        `Bitcoin ${targetMonthDay}`,
        `Bitcoin ${targetWeekday}`,
        'Bitcoin price target',
        'BTC price level',
      ],
      weight: 0.16,
      category: 'btc_price_target',
    },
    {
      name: 'ETF / Product',
      tpl: 'Bitcoin ETF',
      variantTpls: ['crypto ETF', 'spot Bitcoin ETF'],
      weight: 0.10,
      category: 'etf_product',
    },
    {
      name: 'SEC / Regulation',
      tpl: 'crypto regulation',
      variantTpls: ['SEC crypto', 'cryptocurrency regulation'],
      weight: 0.08,
      category: 'regulatory',
    },
    {
      name: 'Fed Rate Decision',
      tpl: 'Fed rate cut',
      variantTpls: ['Federal Reserve rate', 'FOMC'],
      weight: 0.05,
      category: 'macro_rates',
    },
    {
      name: 'US Recession',
      tpl: 'US recession',
      variantTpls: ['recession', 'economic recession'],
      weight: 0.03,
      category: 'macro_growth',
    },
  ];

  return templates.map((template) => {
    const phrases = normalizeSignalPhrases([template.tpl, ...template.variantTpls], 'BTC');
    const [searchPhrase = 'Bitcoin price', ...queryVariants] = phrases;
    return {
      name: template.name,
      searchPhrase,
      queryVariants,
      weight: template.weight,
      category: template.category,
    };
  });
}

function shouldUseShortHorizonCryptoSignalMap(
  signalType: AssetType,
  options?: ExtractSignalOptions,
): boolean {
  return signalType === 'crypto'
    && options?.preferShortHorizonCryptoSignals === true
    && (options.horizonDays ?? Number.POSITIVE_INFINITY) <= 3;
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Returns prioritised signal categories for the asset(s) found in `query`.
 * Uses the asset-intent resolver to disambiguate commodity vs equity intent
 * (e.g., "gold" → GLD commodity proxy, "Barrick GOLD" → materials equity).
 * Template placeholders are substituted with the resolved ticker/display name,
 * then each phrase is normalised for Polymarket's keyword search API.
 */
export function extractSignals(query: string, options?: ExtractSignalOptions): SignalCategory[] {
  const { type, ticker } = detectAssetType(query);
  const resolved = resolveAssetIntent(query, ticker);
  const signalType = resolveSignalType(resolved.assetClass, type);
  if (shouldUseShortHorizonCryptoSignalMap(signalType, options)) {
    return buildShortHorizonCryptoSignals(options?.horizonDays ?? 3);
  }
  const templates = SIGNAL_MAPS[signalType];
  // For commodity proxies, use the proxy label (GLD/SLV/USO) as the ticker in templates
  // so Polymarket search phrases resolve correctly. For equity, use original ticker.
  const effectiveTicker = resolved.proxyLabel ?? resolved.resolvedTicker ?? ticker ?? 'asset';

  // Pull canonical commodity names (e.g. USO → ['oil', 'uso']) for richer query variants
  const searchIdentity = resolveTickerSearchIdentity(resolved.resolvedTicker ?? ticker ?? 'asset');
  const extraNames = searchIdentity.canonicalNames.filter(
    (n) => n !== effectiveTicker.toLowerCase() && n !== (ticker ?? '').toLowerCase(),
  );

  return buildSignalsFromTemplates(templates, effectiveTicker, extraNames);
}

function resolveSignalType(assetClass: ResolvedAssetClass | null, fallbackType: AssetType): AssetType {
  if (assetClass === 'commodity_gold' || assetClass === 'commodity_silver') return 'commodity';
  if (assetClass === 'commodity_oil') return 'oil';
  if (assetClass === 'gold_miner') return 'materials';
  return fallbackType;
}
