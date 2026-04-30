type KalshiEventType = 'fomc' | 'cpi' | 'nfp' | 'other';
type FetchLike = (input: string | URL | Request, init?: RequestInit) => Promise<Response>;

export interface KalshiVolSignal {
  /** ISO timestamp of the event (UTC). */
  eventAt: string;
  /** Short label, e.g. "FOMC-2026-06" or "CPI-2026-05". */
  eventId: string;
  /** Implied probability of the selected high-impact macro outcome. */
  probability: number;
  /** Suggested volatility multiplier for the days leading into the event. */
  intensityBoost: number;
  /** Classified event bucket for downstream covariate routing. */
  eventType: KalshiEventType;
  /** Original title for diagnostics / auditability. */
  sourceTitle: string;
}

export interface KalshiFetchOptions {
  /** ISO YYYY-MM-DD inclusive lower bound for events to consider. */
  fromDate: string;
  /** ISO YYYY-MM-DD inclusive upper bound for events to consider. */
  toDate: string;
  /** Override the API base URL (for testing). Default: https://trading-api.kalshi.com/trade-api/v2 */
  baseUrl?: string;
  /** Override the API key. Falls back to process.env.KALSHI_API_KEY. */
  apiKey?: string;
  /** Injectable fetch for tests. Defaults to global fetch. */
  fetchImpl?: FetchLike;
}

export interface KalshiVolatilityCovariate {
  dates: string[];
  values: number[];
  activeSignals: number;
  peakValue: number;
}

interface KalshiMarketRecord {
  ticker?: string;
  title?: string;
  subtitle?: string;
  status?: string;
  expiration_time?: string;
  close_time?: string;
  settlement_time?: string;
  yes_ask?: number;
  last_price?: number;
}

const MS_PER_DAY = 24 * 60 * 60 * 1000;

const EVENT_PATTERNS: Array<{ type: KalshiEventType; regex: RegExp; weight: number }> = [
  { type: 'fomc', regex: /\b(fomc|fed|federal reserve|rate decision|rate hike|rate cut)\b/i, weight: 1.5 },
  { type: 'cpi', regex: /\b(cpi|inflation|consumer price index)\b/i, weight: 1.3 },
  { type: 'nfp', regex: /\b(nonfarm payroll|non-farm payroll|payrolls|jobs report|nfp)\b/i, weight: 1.2 },
];

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function parseDate(value: string): number {
  const millis = Date.parse(value);
  if (!Number.isFinite(millis)) throw new Error(`invalid Kalshi date: ${value}`);
  return millis;
}

function normalizeProbability(market: KalshiMarketRecord): number | null {
  const raw = market.yes_ask ?? market.last_price;
  if (raw === undefined || !Number.isFinite(raw)) return null;
  const probability = raw > 1 ? raw / 100 : raw;
  return probability >= 0 && probability <= 1 ? probability : null;
}

function classifyEventType(title: string): { type: KalshiEventType; weight: number } | null {
  for (const pattern of EVENT_PATTERNS) {
    if (pattern.regex.test(title)) return { type: pattern.type, weight: pattern.weight };
  }
  return null;
}

function computeIntensityBoost(probability: number, weight: number, eventAt: string, fromDate: string): number {
  const daysUntilEvent = Math.max(0, (parseDate(eventAt) - parseDate(`${fromDate}T00:00:00Z`)) / MS_PER_DAY);
  const urgency = clamp(1 - daysUntilEvent / 14, 0.25, 1);
  return clamp(0.2 + probability * weight * urgency, 0.1, 2.5);
}

export function extractKalshiVolSignalsFromPayload(
  payload: unknown,
  opts: Pick<KalshiFetchOptions, 'fromDate' | 'toDate'>,
): KalshiVolSignal[] {
  if (!payload || typeof payload !== 'object' || !Array.isArray((payload as { markets?: unknown[] }).markets)) {
    throw new Error('Kalshi response missing markets array');
  }
  const fromMs = parseDate(`${opts.fromDate}T00:00:00Z`);
  const toMs = parseDate(`${opts.toDate}T23:59:59Z`);

  return (payload as { markets: KalshiMarketRecord[] }).markets.flatMap((market) => {
    const title = market.title?.trim();
    const eventAt = market.expiration_time ?? market.close_time ?? market.settlement_time;
    if (!title || !eventAt) return [];
    const classified = classifyEventType(title);
    if (!classified) return [];
    const probability = normalizeProbability(market);
    if (probability === null) return [];
    const eventMs = parseDate(eventAt);
    if (eventMs < fromMs || eventMs > toMs) return [];
    return [{
      eventAt,
      eventId: market.ticker ?? `${classified.type}-${eventAt.slice(0, 10)}`,
      probability,
      intensityBoost: computeIntensityBoost(probability, classified.weight, eventAt, opts.fromDate),
      eventType: classified.type,
      sourceTitle: title,
    }];
  });
}

export class KalshiUnconfiguredError extends Error {
  constructor() {
    super(
      'Kalshi macro signal loader is not configured.\n' +
      '  Set KALSHI_API_KEY in the environment or pass apiKey explicitly to\n' +
      '  fetchKalshiVolSignals().',
    );
    this.name = 'KalshiUnconfiguredError';
  }
}

export async function fetchKalshiVolSignals(opts: KalshiFetchOptions): Promise<KalshiVolSignal[]> {
  const apiKey = opts.apiKey ?? process.env.KALSHI_API_KEY;
  if (!apiKey) throw new KalshiUnconfiguredError();

  const fetchImpl = opts.fetchImpl ?? fetch;
  const baseUrl = (opts.baseUrl ?? 'https://trading-api.kalshi.com/trade-api/v2').replace(/\/$/, '');
  const url = `${baseUrl}/markets?status=open&limit=200`;
  const response = await fetchImpl(url, {
    headers: {
      Authorization: `Bearer ${apiKey}`,
      Accept: 'application/json',
    },
  });
  if (!response.ok) {
    throw new Error(`Kalshi request failed: ${response.status} ${response.statusText}`);
  }
  const payload = await response.json();
  return extractKalshiVolSignalsFromPayload(payload, opts);
}

export function buildKalshiVolatilityCovariate(
  dates: string[],
  signals: KalshiVolSignal[],
  lookaheadDays = 5,
): KalshiVolatilityCovariate {
  const values = dates.map((date) => {
    const dateMs = parseDate(`${date}T00:00:00Z`);
    let total = 0;
    for (const signal of signals) {
      const daysAhead = (parseDate(signal.eventAt) - dateMs) / MS_PER_DAY;
      if (daysAhead < 0 || daysAhead > lookaheadDays) continue;
      const decay = 1 - daysAhead / Math.max(1, lookaheadDays);
      total += signal.intensityBoost * decay;
    }
    return Number(total.toFixed(6));
  });
  return {
    dates: [...dates],
    values,
    activeSignals: signals.length,
    peakValue: values.length > 0 ? Math.max(...values) : 0,
  };
}
