/**
 * Polymarket CLOB API client + microstructure helpers.
 *
 * Implements the velocity / spike / spread fetches outlined in
 * docs/polymarket-prediction-improvements-research-2026-07.md §3.3 and §3.4.
 *
 * Pure-math helpers (`computePriceVelocityPpH`, `computeMaxHourlyJump`,
 * `parseClobPriceHistory`) are exported so they can be unit-tested without
 * live HTTP. The fetcher (`fetchClobSpread`, `fetchClobPriceHistory`) is
 * intentionally tiny — it is exercised in integration tests only.
 */

const CLOB_BASE = 'https://clob.polymarket.com';

export interface ClobPricePoint {
  tSec: number;
  p: number;
}

interface RawHistoryPoint {
  t?: unknown;
  p?: unknown;
}

interface RawHistoryResponse {
  history?: unknown;
}

/** Parse the `/prices-history` JSON shape into a sorted, finite, in-range series. */
export function parseClobPriceHistory(raw: unknown): ClobPricePoint[] {
  if (raw === null || typeof raw !== 'object') return [];
  const hist = (raw as RawHistoryResponse).history;
  if (!Array.isArray(hist)) return [];
  const out: ClobPricePoint[] = [];
  for (const item of hist as RawHistoryPoint[]) {
    if (item === null || typeof item !== 'object') continue;
    const t = Number(item.t);
    const p = Number(item.p);
    if (!Number.isFinite(t) || !Number.isFinite(p)) continue;
    if (p < 0 || p > 1) continue;
    out.push({ tSec: t, p });
  }
  out.sort((a, b) => a.tSec - b.tSec);
  return out;
}

/**
 * Linear-regression slope over the last `lookbackHours` of price history,
 * expressed in **percentage points per hour** (so a 0.01 → 0.04 ramp over
 * 3 hours yields ≈ 1.0, not 0.01).
 */
export function computePriceVelocityPpH(
  history: readonly ClobPricePoint[],
  lookbackHours = 6,
): number {
  if (history.length < 2) return 0;
  const newestSec = history[history.length - 1]!.tSec;
  const cutoff = newestSec - lookbackHours * 3600;
  const window = history.filter((pt) => pt.tSec >= cutoff);
  if (window.length < 2) return 0;
  // OLS slope of p (in pp = p * 100) vs. t (in hours).
  const n = window.length;
  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;
  for (const pt of window) {
    const x = pt.tSec / 3600;
    const y = pt.p * 100;
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
  }
  const denom = n * sumXX - sumX * sumX;
  if (denom === 0) return 0;
  return (n * sumXY - sumX * sumY) / denom;
}

/**
 * Maximum absolute hour-over-hour price change within the last `windowHours`.
 *
 * Used to detect whale prints / sharp regime shifts. Returned as a raw delta
 * in [0, 1] (i.e. 0.13 = 13 percentage-points).
 */
export function computeMaxHourlyJump(
  history: readonly ClobPricePoint[],
  windowHours = 24,
): number {
  if (history.length < 2) return 0;
  const newestSec = history[history.length - 1]!.tSec;
  const cutoff = newestSec - windowHours * 3600;
  let maxAbs = 0;
  for (let i = 1; i < history.length; i += 1) {
    const cur = history[i]!;
    const prev = history[i - 1]!;
    if (cur.tSec < cutoff) continue;
    const d = Math.abs(cur.p - prev.p);
    if (d > maxAbs) maxAbs = d;
  }
  return maxAbs;
}

// ---------------------------------------------------------------------------
// HTTP fetchers (kept thin; integration-tested only).
// ---------------------------------------------------------------------------

export interface ClobSpreadResponse {
  /** Spread in dollars on the YES token (e.g. 0.025 = 2.5pp). */
  spread: number;
}

export async function fetchClobSpread(tokenId: string): Promise<number | null> {
  try {
    const r = await fetch(`${CLOB_BASE}/spread/${encodeURIComponent(tokenId)}`);
    if (!r.ok) return null;
    const j = (await r.json()) as Partial<ClobSpreadResponse>;
    const s = Number(j?.spread);
    return Number.isFinite(s) && s >= 0 && s <= 1 ? s : null;
  } catch {
    return null;
  }
}

export async function fetchClobPriceHistory(
  market: string,
  interval: '1h' | '6h' | '1d' = '1h',
): Promise<ClobPricePoint[]> {
  try {
    const r = await fetch(
      `${CLOB_BASE}/prices-history?market=${encodeURIComponent(market)}&interval=${interval}`,
    );
    if (!r.ok) return [];
    return parseClobPriceHistory(await r.json());
  } catch {
    return [];
  }
}
