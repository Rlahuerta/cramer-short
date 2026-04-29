/**
 * R5 Sprint 2 Idea #13 — Kalshi macro vol-event signals (STUB).
 *
 * Source: docs/forecast-improvement-ideas-round5-2026-04-29.md
 *
 * Status: REST scaffolding only.  The plan calls for pulling
 * macro/policy event probabilities from Kalshi (FOMC moves, CPI prints,
 * election outcomes) and feeding them as a Hawkes-style intensity boost
 * into the trajectory MC.
 *
 * Once an API key is configured (KALSHI_API_KEY), implement
 * `fetchKalshiVolSignals` to:
 *   1. Pull the active markets matching a small whitelist of macro
 *      event types (FOMC rate decision, CPI MoM, NFP, etc.).
 *   2. Convert market price → implied probability → signed vol-event
 *      intensity boost (e.g. high-impact event in next 7d ⇒ +0.5σ
 *      Hawkes background intensity for the next 5 days).
 *
 * For now the function throws with a helpful diagnostic so misconfigured
 * pipelines surface the gap immediately instead of silently degrading.
 */

export interface KalshiVolSignal {
  /** ISO timestamp of the event (UTC). */
  eventAt: string;
  /** Short label, e.g. "FOMC-2026-06" or "CPI-2026-05". */
  eventId: string;
  /** Implied probability of the high-vol outcome. */
  probability: number;
  /** Suggested Hawkes intensity multiplier for the days surrounding the event. */
  intensityBoost: number;
}

export interface KalshiFetchOptions {
  /** ISO YYYY-MM-DD inclusive lower bound for events to consider. */
  fromDate: string;
  /** ISO YYYY-MM-DD inclusive upper bound for events to consider. */
  toDate: string;
  /** Override the API base URL (for testing).  Default: https://trading-api.kalshi.com */
  baseUrl?: string;
  /** Override the API key.  Falls back to process.env.KALSHI_API_KEY. */
  apiKey?: string;
}

export class KalshiUnconfiguredError extends Error {
  constructor() {
    super(
      'Kalshi macro signal loader is not configured.\n' +
      '  Set KALSHI_API_KEY in the environment and replace this stub with a\n' +
      '  real implementation under src/tools/finance/kalshi-vol-signals.ts.',
    );
    this.name = 'KalshiUnconfiguredError';
  }
}

/**
 * Reserved API. Throws until live integration is implemented.
 */
export async function fetchKalshiVolSignals(_opts: KalshiFetchOptions): Promise<KalshiVolSignal[]> {
  const apiKey = _opts.apiKey ?? process.env.KALSHI_API_KEY;
  if (!apiKey) throw new KalshiUnconfiguredError();
  throw new KalshiUnconfiguredError();
}
