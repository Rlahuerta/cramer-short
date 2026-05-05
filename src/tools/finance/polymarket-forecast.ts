/**
 * Polymarket Forecast Tool
 *
 * Prediction-market-weighted ensemble price forecast for any asset.
 * Combines Polymarket probability signals with optional news sentiment,
 * fundamental analyst targets, and options skew into a single forecast.
 *
 * Research basis:
 *   Reichenbach & Walther (2025) · Cordoba et al. (2024) · Tsang & Yang (2026)
 */
import { DynamicStructuredTool } from '@langchain/core/tools';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { polymarketBreaker } from '../../utils/circuit-breaker.js';
import {
  fetchPolymarketAnchorMarketsWithQueries,
  fetchPolymarketMarkets,
  type PolymarketMarketResult,
} from './polymarket.js';
import { findSnapshotInWindow, readSnapshotRecords, DEFAULT_POLYMARKET_SNAPSHOTS_PATH } from './polymarket-snapshots.js';
import { extractSignals, scoreMarketRelevance } from './signal-extractor.js';
import { resolveTickerSearchIdentity } from './asset-resolver.js';
import { lookupImpact, inferAssetClass } from './impact-map.js';
import { runEnsemble, computePolymarketSignal, computeEnsemble, computeConditionalReturn, adjustYesBias, type MarketInput } from '../../utils/ensemble.js';
import {
  buildPriceDistributionChart,
  extractPriceThresholds as extractChartPriceThresholds,
} from './price-distribution-chart.js';
import type { PolymarketSnapshotRecord } from './polymarket-snapshots.js';
import {
  applyCryptoTerminalAnchorFallback,
  buildPolymarketAnchorQueryVariants,
  evaluateAnchorTrust,
  extractPriceThresholds as extractAnchorPriceThresholds,
  type PriceThreshold,
} from './markov-distribution.js';
import {
  appendReplayCachePolymarketCapture,
  createRawPolymarketReplayRow,
  freezePolymarketReplayBlock,
  type ArbiterReplayBundle,
  type RawPolymarketReplayRow,
} from './arbiter-replay.js';
import {
  predictWithBrierReplayState,
  type BrierReplayCalibratorState,
} from './brier-replay-calibrator.js';

type RawMarket = {
  marketId?: string;
  assetId?: string;
  question: string;
  probability: number;
  volume24h: number;
  ageDays: number | undefined;
  signalCategory: string;
  endDate?: string | null;
  active?: boolean;
  closed?: boolean;
  enableOrderBook?: boolean;
  priceSpikeDetected: boolean;
  transitoryMove: boolean;
};

type ForecastHistoryReader = typeof readSnapshotRecords;
type ForecastMarketFetcher = typeof fetchPolymarketMarkets;
type ForecastAnchorMarketFetcher = typeof fetchPolymarketAnchorMarketsWithQueries;

type ForecastToolDependencies = {
  fetchMarkets?: ForecastMarketFetcher;
  fetchAnchorMarketsWithQueries?: ForecastAnchorMarketFetcher;
  readRecords?: ForecastHistoryReader;
  recordReplayPolymarketCapture?: (capture: {
    rawRow: RawPolymarketReplayRow;
    polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
  }) => void;
};

type HistoryFlags = {
  priceSpikeDetected: boolean;
  transitoryMove: boolean;
  warnings: string[];
};

const DAY_MS = 86_400_000;
const TWO_HOURS_MS = 2 * 3_600_000;
const FOUR_HOURS_MS = 4 * 3_600_000;
const TWENTY_FOUR_HOURS_MS = 24 * 3_600_000;
const FORTY_EIGHT_HOURS_MS = 48 * 3_600_000;
const LIVE_BRIER_REPLAY_FLAG = 'POLYMARKET_BRIER_REPLAY_CALIBRATOR_ENABLED';
const LIVE_BRIER_REPLAY_STATE: BrierReplayCalibratorState = {
  bias: 0.016276026205423927,
  slope: 1 / 3,
};
const LIVE_BRIER_REPLAY_MIN_PROBABILITY = 0.4;
const LIVE_BRIER_REPLAY_MAX_PROBABILITY = 0.6;

function isLiveBrierReplayCalibratorEnabled(): boolean {
  return /^(1|true|yes|on)$/i.test(process.env[LIVE_BRIER_REPLAY_FLAG] ?? '');
}

function isUsablePolymarketProbability(probability: number): boolean {
  return Number.isFinite(probability) && probability >= 0 && probability <= 1;
}

function applyLiveBrierReplayCalibration(probability: number): number {
  if (
    !isUsablePolymarketProbability(probability)
    || probability < LIVE_BRIER_REPLAY_MIN_PROBABILITY
    || probability > LIVE_BRIER_REPLAY_MAX_PROBABILITY
  ) {
    return probability;
  }
  return predictWithBrierReplayState(probability, LIVE_BRIER_REPLAY_STATE);
}

function toRawMarket(
  market: PolymarketMarketResult,
  signalCategory: string,
  history: HistoryFlags,
): RawMarket | null {
  if (!isUsablePolymarketProbability(market.probability)) {
    return null;
  }

  return {
    marketId: market.marketId,
    assetId: market.assetId,
    question: market.question,
    probability: market.probability,
    volume24h: market.volume24h,
    ageDays: market.ageDays,
    endDate: market.endDate,
    signalCategory,
    active: market.active,
    closed: market.closed,
    enableOrderBook: market.enableOrderBook,
    priceSpikeDetected: history.priceSpikeDetected,
    transitoryMove: history.transitoryMove,
  };
}

function isLiveCalibrationApplicable(probability: number): boolean {
  return isUsablePolymarketProbability(probability)
    && probability >= LIVE_BRIER_REPLAY_MIN_PROBABILITY
    && probability <= LIVE_BRIER_REPLAY_MAX_PROBABILITY;
}

export function evaluateMarketHistory(
  market: Pick<PolymarketMarketResult, 'marketId' | 'probability' | 'volume24h'>,
  records: PolymarketSnapshotRecord[],
  nowMs: number,
): {
  priceSpikeDetected: boolean;
  transitoryMove: boolean;
  warnings: string[];
} {
  if (!market.marketId) {
    return {
      priceSpikeDetected: false,
      transitoryMove: false,
      warnings: [],
    };
  }
  const warnings: string[] = [];

  const spikeSnapshot = findSnapshotInWindow(
    records,
    market.marketId,
    nowMs - FOUR_HOURS_MS,
    nowMs - TWO_HOURS_MS,
  );

  const persistenceSnapshot = findSnapshotInWindow(
    records,
    market.marketId,
    nowMs - FORTY_EIGHT_HOURS_MS,
    nowMs - TWENTY_FOUR_HOURS_MS,
  );

  let priceSpikeDetected = false;
  let transitoryMove = false;

  if (spikeSnapshot) {
    const absDelta = Math.abs(market.probability - spikeSnapshot.probability);
    priceSpikeDetected = absDelta > 0.08 && market.volume24h < 100_000;
  } else {
    warnings.push(`Spike detection unavailable: no prior snapshot found for market ${market.marketId}`);
  }

  if (!persistenceSnapshot) {
    warnings.push(
      `Persistence test unavailable: no prior snapshot found for market ${market.marketId} in 24-48h window`,
    );
  }

  if (spikeSnapshot && persistenceSnapshot) {
    const originalMove = spikeSnapshot.probability - persistenceSnapshot.probability;
    const originalMoveMagnitude = Math.abs(originalMove);
    const movedTowardBaseline =
      Math.abs(market.probability - persistenceSnapshot.probability)
      < Math.abs(spikeSnapshot.probability - persistenceSnapshot.probability);
    const reversalAmount = Math.abs(spikeSnapshot.probability - market.probability);
    transitoryMove =
      originalMoveMagnitude > 0.10
      && movedTowardBaseline
      && reversalAmount > originalMoveMagnitude * 0.5;
  }

  return {
    priceSpikeDetected,
    transitoryMove,
    warnings,
  };
}

export function evaluateHistoryFlags(
  market: PolymarketMarketResult,
  nowMs: number,
  snapshotFilePath?: string,
): HistoryFlags {
  return evaluateHistoryFlagsWithReader(market, nowMs, snapshotFilePath, readSnapshotRecords);
}

function evaluateHistoryFlagsWithReader(
  market: PolymarketMarketResult,
  nowMs: number,
  snapshotFilePath: string | undefined,
  readRecords: ForecastHistoryReader,
): HistoryFlags {
  if (!market.marketId) {
    return {
      priceSpikeDetected: false,
      transitoryMove: false,
      warnings: [],
    };
  }

  try {
    const records = readRecords(snapshotFilePath, market.marketId);
    return evaluateMarketHistory(market, records, nowMs);
  } catch {
    return {
      priceSpikeDetected: false,
      transitoryMove: false,
      warnings: ['Snapshot history unavailable due to filesystem error'],
    };
  }
}

function daysUntilEndDate(endDate: string | null | undefined): number | null {
  if (!endDate) return null;
  const target = new Date(endDate);
  if (Number.isNaN(target.getTime())) return null;
  return (target.getTime() - Date.now()) / 86_400_000;
}

function isResolutionAlignedToHorizon(daysUntilResolution: number | null, horizonDays: number): boolean {
  if (daysUntilResolution === null) return false;
  return Math.abs(daysUntilResolution - horizonDays) <= Math.max(1.5, horizonDays * 0.35);
}

function shouldFilterResolutionMismatch(assetClass: string, horizonDays: number): boolean {
  return assetClass === 'crypto' && horizonDays <= 3;
}

function shouldUseShortHorizonCryptoAnchorRetrieval(assetClass: string, horizonDays: number): boolean {
  return assetClass === 'crypto' && horizonDays <= 3;
}

function resolveAnchorSignalCategory(
  question: string,
  signals: ReturnType<typeof extractSignals>,
): string {
  let bestCategory = signals[0]?.category ?? 'btc_price_target';
  let bestScore = -1;

  for (const signal of signals) {
    const score = scoreMarketRelevance(question, signal.category);
    if (score > bestScore) {
      bestScore = score;
      bestCategory = signal.category;
    }
  }

  return bestScore > 0
    ? bestCategory
    : signals.find((signal) => signal.category === 'btc_price_target')?.category ?? bestCategory;
}

type AnchorCandidateMarket = {
  question: string;
  probability: number;
  volume?: number;
  createdAt?: number;
  endDate?: string | null;
};

type ShortHorizonAnchorSelection = {
  selectedMarkets: RawMarket[];
  selectedThresholds: PriceThreshold[];
  skippedResolutionMismatches: RawMarket[];
};

function toAnchorCandidateMarket(market: RawMarket, referenceTimeMs: number): AnchorCandidateMarket {
  return {
    question: market.question,
    probability: market.probability,
    volume: market.volume24h,
    createdAt: market.ageDays != null ? referenceTimeMs - market.ageDays * DAY_MS : undefined,
    endDate: market.endDate ?? null,
  };
}

function buildAnchorSelectionKey(
  anchor: Pick<PriceThreshold, 'price' | 'rawProbability' | 'endDate'>,
): string {
  return `${anchor.price}|${anchor.rawProbability}|${anchor.endDate ?? ''}`;
}

function evaluateShortHorizonAnchorTrust(
  market: RawMarket,
  horizonDays: number,
): ReturnType<typeof evaluateAnchorTrust> {
  const daysToResolution = daysUntilEndDate(market.endDate);
  return evaluateAnchorTrust({
    hasVolume: market.volume24h > 0,
    isYoung: (market.ageDays ?? Number.POSITIVE_INFINITY) < 2,
    isShortHorizonCrypto: true,
    isLongHorizonCrypto: false,
    isNearTargetResolution: daysToResolution !== null && Math.abs(daysToResolution - horizonDays) <= 2,
  });
}

function selectShortHorizonCryptoAnchorMarkets(
  markets: RawMarket[],
  ticker: string,
  horizonDays: number,
  referenceTimeMs: number,
): ShortHorizonAnchorSelection {
  const candidates = markets.map((market) => ({
    market,
    candidate: toAnchorCandidateMarket(market, referenceTimeMs),
  }));
  const strictCandidates = candidates.filter(({ market }) => {
    const daysToResolution = daysUntilEndDate(market.endDate);
    return daysToResolution === null || isResolutionAlignedToHorizon(daysToResolution, horizonDays);
  });
  const trustedStrictCandidates = strictCandidates.filter(
    ({ market }) => evaluateShortHorizonAnchorTrust(market, horizonDays).trustScore === 'high',
  );
  const strictThresholdSource = trustedStrictCandidates.length > 0 ? trustedStrictCandidates : strictCandidates;
  const strictThresholds = extractAnchorPriceThresholds(
    strictThresholdSource.map(({ candidate }) => candidate),
    { ticker, horizonDays, referenceTimeMs },
  );
  const selectedThresholds = applyCryptoTerminalAnchorFallback(
    candidates.map(({ candidate }) => candidate),
    strictThresholds,
    ticker,
    horizonDays,
    referenceTimeMs,
  );

  const marketByKey = new Map<string, RawMarket>();
  for (const { market, candidate } of candidates) {
    const [threshold] = extractAnchorPriceThresholds([candidate], { ticker, horizonDays, referenceTimeMs });
    if (!threshold) continue;
    const key = buildAnchorSelectionKey(threshold);
    if (!marketByKey.has(key)) {
      marketByKey.set(key, market);
    }
  }

  const selectedMarkets = selectedThresholds
    .map((threshold) => marketByKey.get(buildAnchorSelectionKey(threshold)))
    .filter((market): market is RawMarket => market != null);
  const selectedIds = new Set(selectedMarkets.map((market) => market.marketId ?? market.question));
  const skippedResolutionMismatches = candidates
    .filter(({ market }) => {
      const daysToResolution = daysUntilEndDate(market.endDate);
      return daysToResolution !== null
        && !isResolutionAlignedToHorizon(daysToResolution, horizonDays)
        && !selectedIds.has(market.marketId ?? market.question);
    })
    .map(({ market }) => market);

  return {
    selectedMarkets,
    selectedThresholds,
    skippedResolutionMismatches,
  };
}

function isThresholdChartAlignedToHorizon(markets: RawMarket[], horizonDays: number): boolean {
  const thresholds = extractChartPriceThresholds(markets);
  if (thresholds.length < 2) return false;

  const datedThresholdMarkets = markets.filter((market) => {
    if (!market.endDate) return false;
    return extractChartPriceThresholds([{ question: market.question, probability: market.probability }]).length > 0;
  });
  if (datedThresholdMarkets.length < 2) return false;

  const alignedMarkets = datedThresholdMarkets.filter((market) => {
    const days = daysUntilEndDate(market.endDate);
    return isResolutionAlignedToHorizon(days, horizonDays);
  });

  return alignedMarkets.length >= 2;
}

// ---------------------------------------------------------------------------
// Description (injected into system prompt)
// ---------------------------------------------------------------------------

export const POLYMARKET_FORECAST_DESCRIPTION = `
Generates a prediction-market-weighted ensemble price forecast for any asset over a 1–365 day horizon.

Combines Polymarket crowd probabilities with optional auxiliary signals (news sentiment, fundamental
analyst targets, options skew) into a single calibrated forecast with a 95% confidence interval and
a quality grade (A–D).

Polymarket hosts markets from 1 day to 12 months out — all are valid inputs. Signal quality is
highest for liquid markets resolving in 1–90 days; longer-dated or low-volume markets are
automatically down-weighted by the quality scoring engine.

## What This Tool Does

1. Extracts relevant Polymarket search signals for the asset (earnings, macro, geopolitical, etc.)
2. Fetches live prediction-market probabilities from Polymarket
3. Maps each market to an asset-return impact using a pre-built δ(YES)/δ(NO) lookup table
4. Blends the Polymarket signal with any auxiliary signals you provide (sentiment, fundamentals, skew)
5. Outputs: forecast price, 95% CI, return percentage, per-signal breakdown, grade, and warnings

## When to Use

- User asks "Where will NVDA trade in a week / month / quarter?"
- User wants a forecast incorporating prediction-market data at any horizon (days to months)
- User asks "What does the market imply for [TICKER] by end of year?"
- You have already fetched sentiment or fundamentals and want to incorporate them into a price forecast
- User asks for a probability-weighted scenario analysis combining multiple market signals

## When NOT to Use

- Real-time stock price — use \`get_market_data\`
- Fundamental company analysis — use \`get_financials\`
- Multi-year DCF valuation (> 2 years) — use the DCF skill instead
- News summarisation — use \`web_search\`

## Signal Quality by Horizon

| Horizon | Polymarket signal strength | Notes |
|---------|--------------------------|-------|
| 1–30 days | ★★★ Strong | Many active markets, high volume, best accuracy |
| 30–90 days | ★★ Moderate | Fewer markets, still actionable |
| 90–365 days | ★ Weaker | Longer-dated markets tend to be less liquid; quality weights auto-adjust |

## Input Tips

- **ALWAYS pass \`current_price\`** (fetch with \`get_market_data\` first). Without it the 95% CI
  is shown as percentages only (relative to base 100), NOT in dollar terms. Call order:
  get_market_data(ticker) → polymarket_forecast(ticker, current_price=<fetched price>).
- **Pass \`sentiment_score\`** (from \`social_sentiment\`) if you have already called that tool —
  it improves forecast quality at no extra cost.
- **Pass \`fundamental_return\`** (analyst 1-year target implied return from \`get_financials\`) if
  available — use the decimal form, e.g. \`0.15\` for a +15% target.
- **Pass \`options_skew\`** if you have options data — use −1 (bearish), 0 (neutral), +1 (bullish).
- The tool fetches Polymarket data itself; you do **not** need to call \`polymarket_search\` first.
- For commodity proxies (USO for oil, GLD for gold, SLV for silver), the tool searches both the ETF ticker and the underlying commodity name (e.g. "WTI", "crude", "OPEC" for oil) to find the broadest set of relevant prediction markets.

## Interpreting the Output

| Grade | Score | Meaning |
|-------|-------|---------|
| A | 80–100 | High conviction — ≥5 liquid markets, multiple corroborating signals |
| B | 60–79  | Moderate conviction — useful directional signal |
| C | 40–59  | Low conviction — treat as indicative, not actionable alone |
| D | 0–39   | Speculative — few or no liquid markets, high uncertainty |

The 95% CI reflects both market-probability variance and a 20% model-uncertainty buffer.
A wide CI (σ > 5%) typically signals Grade C/D and limited predictive power.

## Composability Note

For richer analysis, call \`get_financials\` and \`social_sentiment\` first, then pass their results
to this tool via \`fundamental_return\` and \`sentiment_score\`. This turns two separate lookups into
a unified forecast with a higher quality grade.

\`\`\`
get_financials(NVDA) → fundamental_return = 0.18
social_sentiment(NVDA) → sentiment_score = 0.6
polymarket_forecast(NVDA, current_price=135.50, fundamental_return=0.18, sentiment_score=0.6)
\`\`\`
`.trim();

// ---------------------------------------------------------------------------
// Schema
// ---------------------------------------------------------------------------

const schema = z.object({
  ticker: z.string().describe('Asset ticker or name, e.g. "NVDA", "BTC", "GLD"'),
  horizon_days: z.number().int().min(1).max(365).default(7)
    .describe('Forecast horizon in days (1–365). Default: 7. Polymarket has markets from 1 day to 12 months — all are valid. Signal quality is highest for 1–90 day horizons.'),
  current_price: z.number().optional()
    .describe('Current asset price. If omitted, tool uses a placeholder and notes it.'),
  sentiment_score: z.number().min(-1).max(1).optional()
    .describe('News/social sentiment: -1 bearish, 0 neutral, +1 bullish. Pass if already retrieved.'),
  markov_return: z.number().optional()
    .describe('Markov-chain expected return over the forecast horizon as a decimal, pre-shrunk by Markov weight. Pass from a prior successful markov_distribution result when available.'),
  fundamental_return: z.number().optional()
    .describe('Analyst 1-year price target implied return as decimal (e.g. 0.15 for +15%). Pass if known.'),
  options_skew: z.number().min(-1).max(1).optional()
    .describe('Options skew signal: -1 bearish, 0 neutral, +1 bullish. Pass if available.'),
});

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/** Map a signal category to a human-readable theme label. */
function catToLabel(category: string): string {
  const map: Record<string, string> = {
    macro_rates:                'Fed / Rates',
    macro_growth:               'Growth / Recession',
    trade_policy:               'Trade Policy',
    tariff_increase:            'Tariffs',
    tariff_relief:              'Tariff Relief',
    geopolitical:               'Geopolitical',
    geopolitical_conflict:      'Conflict Risk',
    earnings:                   'Earnings',
    earnings_beat:              'Earnings Beat',
    earnings_miss:              'Earnings Risk',
    commodity:                  'Commodity',
    oil_spike:                  'Oil / Energy',
    supply_chain:               'Supply Chain',
    government_budget:          'Govt Budget',
    regulatory:                 'Regulation',
    fda_approval:               'FDA Approval',
    fda_rejection:              'FDA Risk',
    crypto_regulation_positive: 'Crypto Reg',
    crypto_regulation_negative: 'Crypto Reg Risk',
    btc_price_target:           'BTC Price Target',
    election_market_friendly:   'Election',
    etf_product:                'ETF Product',
    recession:                  'Recession',
    macro_data_strong:          'Strong Macro Data',
    macro_data_weak:            'Weak Macro Data',
    fed_rate_cut:               'Fed Rate Cut',
    fed_rate_hike:              'Fed Rate Hike',
  };
  return map[category] ?? category.replace(/_/g, ' ').replace(/\b\w/g, (c) => c.toUpperCase());
}

/** Truncate string to maxLen, appending '…' if cut. */
function truncCol(s: string, maxLen: number): string {
  return s.length > maxLen ? s.slice(0, maxLen - 1) + '…' : s;
}


function categoryToTier(category: string): 'macro' | 'geopolitical' | 'electoral' {
  const lower = category.toLowerCase();
  if (lower.includes('macro') || lower.includes('fed') || lower.includes('rate') ||
      lower.includes('gdp') || lower.includes('cpi')) return 'macro';
  if (lower.includes('election') || lower.includes('vote') || lower.includes('president')) return 'electoral';
  return 'geopolitical';
}

function sign(n: number): string {
  return n >= 0 ? '+' : '';
}

function pct(n: number, decimals = 2): string {
  return `${sign(n)}${(n * 100).toFixed(decimals)}`;
}

function sentimentLabel(score: number): string {
  if (score >= 0.5) return 'very bullish';
  if (score >= 0.1) return 'bullish';
  if (score <= -0.5) return 'very bearish';
  if (score <= -0.1) return 'bearish';
  return 'neutral';
}

function optionsLabel(skew: number): string {
  if (skew >= 0.5) return 'bullish skew';
  if (skew >= 0.1) return 'mildly bullish';
  if (skew <= -0.5) return 'bearish skew';
  if (skew <= -0.1) return 'mildly bearish';
  return 'neutral skew';
}

// ---------------------------------------------------------------------------
// Tool
// ---------------------------------------------------------------------------

export function createPolymarketForecastTool(dependencies: ForecastToolDependencies = {}) {
  const fetchMarkets = dependencies.fetchMarkets ?? fetchPolymarketMarkets;
  const fetchAnchorMarketsWithQueries = dependencies.fetchAnchorMarketsWithQueries ?? fetchPolymarketAnchorMarketsWithQueries;
  const readRecords = dependencies.readRecords ?? readSnapshotRecords;
  const recordReplayPolymarketCapture = dependencies.recordReplayPolymarketCapture
    ?? ((capture: {
      rawRow: RawPolymarketReplayRow;
      polymarket: NonNullable<ArbiterReplayBundle['polymarket']>;
    }) => {
      appendReplayCachePolymarketCapture(capture);
    });

  return new DynamicStructuredTool({
    name: 'polymarket_forecast',
    description:
      'Generate a prediction-market-weighted ensemble price forecast for an asset over any horizon (1–365 days), ' +
      'combining Polymarket probabilities (markets span 1 day to 12 months) with optional sentiment, Markov, fundamental, and options signals.',
    schema,
    func: async (input) => {
      if (polymarketBreaker.isOpen()) {
        return formatToolResult({
          error: 'Polymarket API is temporarily unavailable (circuit open). Try again in a few minutes.',
        });
      }

      try {
        const ticker = input.ticker.trim().toUpperCase();
        const horizonDays = input.horizon_days ?? 7;
        const currentPrice = input.current_price;
        const basePrice = currentPrice ?? 100;
        const liveBrierReplayEnabled = isLiveBrierReplayCalibratorEnabled();
        const searchIdentity = resolveTickerSearchIdentity(ticker);
        const assetClass = inferAssetClass(searchIdentity.canonicalTicker);
        const historyWarnings: string[] = [];
        const nowMs = Date.now();
        const replayCapturedAt = new Date(nowMs).toISOString();
        const useShortHorizonCryptoAnchorRetrieval = shouldUseShortHorizonCryptoAnchorRetrieval(assetClass, horizonDays);

        // Step 1: Extract signals for this ticker (up to 5)
        const signals = extractSignals(searchIdentity.canonicalTicker).slice(0, 5);
        const genericReplayQuerySet = [...new Set(
          signals.flatMap((sig) => [sig.searchPhrase, ...(sig.queryVariants ?? [])]),
        )];
        const anchorReplayQuerySet = useShortHorizonCryptoAnchorRetrieval
          ? buildPolymarketAnchorQueryVariants(searchIdentity.canonicalTicker, { horizonDays })
          : [];
        const rawMarkets: RawMarket[] = [];
        const skippedResolutionMismatches: RawMarket[] = [];
        let shortHorizonAnchorThresholds: PriceThreshold[] = [];
        const allSnapshotRecords = readRecords(undefined);
        let replayQuerySet = genericReplayQuerySet;
        // Reader reused across all markets — avoids creating a closure per iteration
        const marketReader = (_filePath: string | undefined, marketId?: string) =>
          allSnapshotRecords.filter((r) => !marketId || r.marketId === marketId);
        const addStructuredMarkets = (
          markets: Array<PolymarketMarketResult & { signalCategory: string }>,
          options?: { skipResolutionMismatchFilter?: boolean },
        ) => {
          const seen = new Set<string>();
          for (const market of markets) {
            if (seen.has(market.question)) continue;
            seen.add(market.question);
            const history = evaluateHistoryFlagsWithReader(market, nowMs, undefined, marketReader);
            historyWarnings.push(...history.warnings);
            const rawMarket = toRawMarket(market, market.signalCategory, history);
            if (!rawMarket) continue;
            if (
              shouldFilterResolutionMismatch(assetClass, horizonDays)
              && !options?.skipResolutionMismatchFilter
            ) {
              const daysToResolution = daysUntilEndDate(rawMarket.endDate);
              if (daysToResolution !== null && !isResolutionAlignedToHorizon(daysToResolution, horizonDays)) {
                skippedResolutionMismatches.push(rawMarket);
                continue;
              }
            }
            rawMarkets.push(rawMarket);
          }
        };
        const fetchGenericStructuredMarkets = async () => {
          const allResults = await Promise.allSettled(
            signals.map((sig) => {
              const phrases = [sig.searchPhrase, ...(sig.queryVariants ?? [])];
              return Promise.allSettled(
                phrases.map((phrase) => fetchMarkets(phrase, 5, { snapshotFilePath: DEFAULT_POLYMARKET_SNAPSHOTS_PATH })),
              ).then((settledVariants) =>
                settledVariants
                  .filter((r): r is PromiseFulfilledResult<PolymarketMarketResult[]> => r.status === 'fulfilled')
                  .flatMap((r) => r.value)
                  .filter((m) => scoreMarketRelevance(m.question, sig.category) > 0)
                  .map((m) => ({ ...m, signalCategory: sig.category })),
              );
            }),
          );

          return allResults
            .filter((result): result is PromiseFulfilledResult<Array<PolymarketMarketResult & { signalCategory: string }>> => result.status === 'fulfilled')
            .flatMap((result) => result.value);
        };

        if (useShortHorizonCryptoAnchorRetrieval) {
          const anchorFrontQueries = anchorReplayQuerySet.slice(0, 6);
          const anchorRetryQueries = anchorReplayQuerySet.slice(anchorFrontQueries.length);
          replayQuerySet = anchorFrontQueries;

          let anchorMarkets = await fetchAnchorMarketsWithQueries(anchorFrontQueries, 12, {
            ticker: searchIdentity.canonicalTicker,
            horizonDays,
          });

          if (anchorMarkets.length === 0 && anchorRetryQueries.length > 0) {
            replayQuerySet = [...anchorFrontQueries, ...anchorRetryQueries];
            anchorMarkets = await fetchAnchorMarketsWithQueries(anchorRetryQueries, 12, {
              ticker: searchIdentity.canonicalTicker,
              horizonDays,
            });
          }

          addStructuredMarkets(
            anchorMarkets.map((market) => ({
              ...market,
              signalCategory: resolveAnchorSignalCategory(market.question, signals),
            })),
            { skipResolutionMismatchFilter: true },
          );

          const shortHorizonSelection = selectShortHorizonCryptoAnchorMarkets(
            rawMarkets,
            searchIdentity.canonicalTicker,
            horizonDays,
            nowMs,
          );
          rawMarkets.splice(0, rawMarkets.length, ...shortHorizonSelection.selectedMarkets);
          shortHorizonAnchorThresholds = shortHorizonSelection.selectedThresholds;
          skippedResolutionMismatches.push(...shortHorizonSelection.skippedResolutionMismatches);

          if (rawMarkets.length === 0) {
            replayQuerySet = [...new Set([...anchorReplayQuerySet, ...genericReplayQuerySet])];
            addStructuredMarkets(await fetchGenericStructuredMarkets());
          }
        } else {
          addStructuredMarkets(await fetchGenericStructuredMarkets());
        }

        if (skippedResolutionMismatches.length > 0) {
          historyWarnings.push(
            `Skipped ${skippedResolutionMismatches.length} Polymarket market${skippedResolutionMismatches.length === 1 ? '' : 's'} because ${skippedResolutionMismatches.length === 1 ? 'its resolution date does' : 'their resolution dates do'} not align with the requested ${horizonDays}-day horizon.`,
          );
        }

        if (recordReplayPolymarketCapture) {
          const rawRow = createRawPolymarketReplayRow({
            capturedAt: replayCapturedAt,
            ticker: searchIdentity.canonicalTicker,
            horizonDays,
            currentPrice: currentPrice ?? null,
            querySet: replayQuerySet,
            selectedMarkets: rawMarkets,
            warnings: historyWarnings,
          });
          const polymarket = freezePolymarketReplayBlock({
            querySet: replayQuerySet,
            selectedMarkets: rawMarkets.map((market) => ({
              ...market,
              relevanceScore: scoreMarketRelevance(market.question, market.signalCategory),
            })),
            warnings: historyWarnings,
          });

          recordReplayPolymarketCapture({ rawRow, polymarket });
        }

        // Step 3: Build MarketInput array
        let calibratedMarketCount = 0;
        const markets: MarketInput[] = rawMarkets.map((m) => {
          const mImpact = lookupImpact(m.signalCategory, assetClass);
          const shouldApplyLiveCalibration = liveBrierReplayEnabled && isLiveCalibrationApplicable(m.probability);
          const probability = shouldApplyLiveCalibration
            ? applyLiveBrierReplayCalibration(m.probability)
            : m.probability;
          const daysToExpiry = daysUntilEndDate(m.endDate);
          if (shouldApplyLiveCalibration && probability !== m.probability) calibratedMarketCount++;
          return {
            question: m.question,
            probability,
            volume24hUsd: m.volume24h,
            ageDays: m.ageDays,
            daysToExpiry: daysToExpiry === null ? undefined : daysToExpiry,
            priceSpikeDetected: m.priceSpikeDetected,
            transitoryMove: m.transitoryMove,
            signalTier: categoryToTier(m.signalCategory),
            deltaYes: mImpact.deltaYes,
            deltaNo: mImpact.deltaNo,
          };
        });
        const liveCalibrationApplied = liveBrierReplayEnabled && calibratedMarketCount > 0;

        // Step 4: Run ensemble — also capture intermediate values for display
        const otherSignals = {
          sentimentScore: input.sentiment_score,
          markovReturn: input.markov_return,
          fundamentalReturn: input.fundamental_return,
          optionsSkew: input.options_skew,
          horizonDays,
        };
        const ensembleOptions = { adaptiveWeighting: true } as const;

        const { signal: pmSignal, avgQuality, warnings: pmWarnings } = computePolymarketSignal(markets);
        const { weights } = computeEnsemble(pmSignal, avgQuality, otherSignals, ensembleOptions);
        const result = runEnsemble(basePrice, markets, otherSignals, ensembleOptions);

        // Step 5: Format output
        const returnPct = (result.forecastReturn * 100).toFixed(2);
        const sigmaPct = (result.sigma * 100).toFixed(2);
        const ciLow = result.ciLow95;
        const ciHigh = result.ciHigh95;
        const pmPct = pct(result.pmSignal);
        const pmWeightPct = (result.pmNormalizedWeight * 100).toFixed(1);
        const avgQualityStr = result.avgMarketQuality.toFixed(3);
        let thresholdChartWarning: string | null = null;
        const displayLabel = searchIdentity.canonicalNames[0]?.toUpperCase() ?? ticker;

        const lines: string[] = [
          `📊 Polymarket Forecast: ${displayLabel} (${ticker})  |  Horizon: ${horizonDays} days  |  Grade: ${result.qualityGrade} (${result.qualityScore}/100)`,
        ];

        if (currentPrice === undefined) {
          lines.push('⚠️  No current price provided — price shown relative to base 100');
        }

        if (horizonDays > 90) {
          lines.push(`⚠️  Horizon ${horizonDays}d > 90 days: Polymarket signal accuracy decreases for longer horizons. Wider CI expected. Consider supplementing with DCF skill for multi-month forecasts.`);
        } else if (horizonDays > 14) {
          lines.push(`ℹ️  Horizon ${horizonDays}d: Polymarket markets exist at this range but signal quality is moderate. 95% CI is wider than short-term forecasts.`);
        }
        if (liveCalibrationApplied) {
          lines.push(`ℹ️  Live Brier replay calibration is active: compressed ${calibratedMarketCount} mid-confidence market ${calibratedMarketCount === 1 ? 'quote' : 'quotes'} toward neutral before blending.`);
        }

        lines.push('');
        lines.push(`Current price:   ${currentPrice !== undefined ? '$' + basePrice : 'not provided — CI shown as %'}`);
        lines.push(`Forecast price:  ${currentPrice !== undefined ? '$' + result.forecastPrice.toFixed(2) : '(base 100) ' + result.forecastPrice.toFixed(2)}  (${sign(result.forecastReturn)}${returnPct}%)`);
        if (currentPrice !== undefined) {
          lines.push(`95% CI:          [$${ciLow.toFixed(2)} – $${ciHigh.toFixed(2)}]  (σ = ${sigmaPct}%)`);
        } else {
          const ciLowPct = ((result.ciLow95 / basePrice - 1) * 100).toFixed(2);
          const ciHighPct = ((result.ciHigh95 / basePrice - 1) * 100).toFixed(2);
          const ciHighSign = parseFloat(ciHighPct) >= 0 ? '+' : '';
          lines.push(`95% CI:          [${ciLowPct}% – ${ciHighSign}${ciHighPct}%]  (σ = ${sigmaPct}%)  ← % relative to current price`);
        }
        lines.push('');

        // ── Polymarket Signal Summary (grouped by theme) ───────────────────────────
        const numThemes = rawMarkets.reduce((s, m) => { s.add(m.signalCategory); return s; }, new Set<string>()).size;
        lines.push(`── Polymarket Signal Summary  (w̄ = ${avgQualityStr} · ${markets.length} markets · ${numThemes} themes) ─`);

        if (markets.length === 0) {
          lines.push('  [No Polymarket markets found for this asset]');
        } else {
          type ThemeRow = {
            category: string;
            label: string;
            netCondReturn: number;
            topQuestion: string;
            topProb: number;
            absContrib: number;
          };

          const byCategory = new Map<string, { question: string; probability: number; condReturn: number }[]>();
          for (const m of rawMarkets) {
            const mImpact = lookupImpact(m.signalCategory, assetClass);
            const condReturn = computeConditionalReturn(adjustYesBias(m.probability), mImpact.deltaYes, mImpact.deltaNo);
            if (!byCategory.has(m.signalCategory)) byCategory.set(m.signalCategory, []);
            byCategory.get(m.signalCategory)!.push({ question: m.question, probability: m.probability, condReturn });
          }

          const rows: ThemeRow[] = [];
          for (const [cat, entries] of byCategory) {
            const net = entries.reduce((s, e) => s + e.condReturn, 0) / entries.length;
            const top = entries.reduce((best, e) => Math.abs(e.condReturn) >= Math.abs(best.condReturn) ? e : best);
            rows.push({ category: cat, label: catToLabel(cat), netCondReturn: net, topQuestion: top.question, topProb: top.probability, absContrib: Math.abs(net) });
          }

          const totalAbs = rows.reduce((s, r) => s + r.absContrib, 0) || 1;
          rows.sort((a, b) => b.absContrib - a.absContrib);

          const W_THEME = 22;
          const W_DIR   = 13;
          const W_SIG   = 48;

          const header = `  ${'Theme'.padEnd(W_THEME)}  ${'Direction'.padEnd(W_DIR)}  ${'Key Signal'.padEnd(W_SIG)}  Contribution`;
          const divider = `  ${'─'.repeat(W_THEME + W_DIR + W_SIG + 18)}`;
          lines.push(header);
          lines.push(divider);

          let bullish = 0;
          let bearish = 0;
          let neutral = 0;
          for (const row of rows) {
            const dir = row.netCondReturn > 0.0005 ? '↑ Bullish' : row.netCondReturn < -0.0005 ? '↓ Bearish' : '→ Neutral';
            if (dir.startsWith('↑')) bullish++;
            else if (dir.startsWith('↓')) bearish++;
            else neutral++;

            const probPct = `${(row.topProb * 100).toFixed(0)}% YES`;
            const keySignal = truncCol(`${row.topQuestion}: ${probPct}`, W_SIG);
            const contrib = `${((row.absContrib / totalAbs) * 100).toFixed(0)}%`;

            lines.push(
              `  ${truncCol(row.label, W_THEME).padEnd(W_THEME)}  ${dir.padEnd(W_DIR)}  ${keySignal.padEnd(W_SIG)}  ${contrib.padStart(5)}`,
            );
          }

          lines.push(divider);

          const netLean = result.pmSignal > 0.005 ? ' (bullish lean)' : result.pmSignal < -0.005 ? ' (bearish lean)' : '';
          lines.push(`  Consensus: ${bullish} bullish · ${bearish} bearish · ${neutral} neutral    Net signal: ${pmPct}%${netLean}`);
          lines.push(`  Polymarket drives ${pmWeightPct}% of this forecast  (remainder from sentiment / fundamentals / options)`);
        }

        lines.push('');
        lines.push('── Other Signals ──────────────────────────────────────────────────────────');

        const wSent = weights['sentiment'];
        const wMarkov = weights['markov'];
        const wFund = weights['fundamental'];
        const wOpt = weights['options'];

        if (input.sentiment_score !== undefined) {
          const sentContrib = pct(input.sentiment_score * 0.04);
          lines.push(`  News sentiment:     ${sentimentLabel(input.sentiment_score)} → ${sentContrib}%  (weight: ${((wSent ?? 0) * 100).toFixed(1)}%)`);
        } else {
          lines.push('  News sentiment:     [signal omitted — not provided]');
        }

        if (input.fundamental_return !== undefined) {
          const fundContrib = pct(input.fundamental_return * (horizonDays / 365));
          lines.push(`  Fundamentals:       ${fundContrib}%  (weight: ${((wFund ?? 0) * 100).toFixed(1)}%)`);
        } else {
          lines.push('  Fundamentals:       [signal omitted — not provided]');
        }

        if (input.markov_return !== undefined) {
          const markovContrib = pct(input.markov_return);
          lines.push(`  Markov chain:       ${markovContrib}%  (weight: ${((wMarkov ?? 0) * 100).toFixed(1)}%)`);
        } else {
          lines.push('  Markov chain:       [signal omitted — not provided]');
        }

        if (input.options_skew !== undefined) {
          const optContrib = pct(input.options_skew * 0.03);
          lines.push(`  Options skew:       ${optionsLabel(input.options_skew)} → ${optContrib}%  (weight: ${((wOpt ?? 0) * 100).toFixed(1)}%)`);
        } else {
          lines.push('  Options skew:       [signal omitted — not provided]');
        }

        const thresholds = shortHorizonAnchorThresholds.length > 0
          ? shortHorizonAnchorThresholds
          : extractChartPriceThresholds(rawMarkets);
        const thresholdChartAligned = shortHorizonAnchorThresholds.length > 0
          ? true
          : isThresholdChartAlignedToHorizon(rawMarkets, horizonDays);
        if (thresholds.length >= 2 && thresholdChartAligned) {
          const chart = buildPriceDistributionChart(thresholds, currentPrice, ticker);
          if (chart) {
            lines.push('');
            lines.push('── Price Distribution (from threshold markets) ────────────────────────────');
            lines.push(chart);
          }
        } else if (thresholds.length >= 2) {
          thresholdChartWarning = 'Threshold-style markets were omitted from the distribution chart because their resolution dates do not align with the requested forecast horizon.';
        }

        lines.push('');
        lines.push('── Warnings ───────────────────────────────────────────────────────────────');

        const allWarnings = [
          ...historyWarnings,
          ...(result.warnings ?? []),
          ...pmWarnings.filter((w) => !result.warnings?.includes(w)),
          ...(thresholdChartWarning ? [thresholdChartWarning] : []),
        ];
        const uniqueWarnings = [...new Set(allWarnings)];
        if (uniqueWarnings.length === 0) {
          lines.push('  None');
        } else {
          for (const w of uniqueWarnings) {
            lines.push(`  ⚠ ${w}`);
          }
        }

        lines.push('');
        lines.push('── Research basis: Reichenbach & Walther (2025) · Cordoba et al. (2024) · Tsang & Yang (2026)');

        return formatToolResult({
          result: lines.join('\n'),
          ...(liveCalibrationApplied ? { forecastReturn: result.forecastReturn } : {}),
        });
      } catch (error) {
        const message = error instanceof Error ? error.message : String(error);
        throw new Error(`[polymarket_forecast] ${message}`);
      }
    },
  });
}

export const polymarketForecastTool = createPolymarketForecastTool();
