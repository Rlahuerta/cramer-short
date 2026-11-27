/**
 * Appends today's Deribit BTC and ETH option books to .cramer-short/cache/options/deribit.jsonl (relative to cwd),
 * one row per currency. Forward data for H5 (option-implied densities as the anchor): there is no free history,
 * so the test needs months of daily captures. Public API, no key.
 *
 * Usage: bun scripts/capture-deribit-options.ts
 */
import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';
import { type RetryOptions, withRetry } from '../src/utils/retry.js';

const OUTPUT_PATH = '.cramer-short/cache/options/deribit.jsonl';
const BOOK_SUMMARY_URL = 'https://www.deribit.com/api/v2/public/get_book_summary_by_currency';

type DeribitSummary = Record<string, unknown>;

export function toDeribitCaptureRow(currency: string, now: Date, summaries: DeribitSummary[]) {
  return {
    capturedAt: now.toISOString(),
    currency,
    options: summaries.map((s) => ({
      instrument: s.instrument_name,
      markIv: s.mark_iv,
      markPrice: s.mark_price,
      bid: s.bid_price,
      ask: s.ask_price,
      openInterest: s.open_interest,
      underlyingPrice: s.underlying_price,
      underlyingIndex: s.underlying_index,
      volumeUsd: s.volume_usd,
    })),
  };
}

/** About 2.5 minutes of backoff: rides out a short Deribit outage (the first capture hit a one-off HTTP 503). */
const RETRY: RetryOptions = { maxAttempts: 6, baseDelayMs: 5_000, maxDelayMs: 60_000 };

/** Fetches each currency's book on its own, retrying transient errors, so one failing currency doesn't lose the other. */
export async function captureDeribitOptions(fetchImpl: typeof fetch = fetch, retry: RetryOptions = RETRY) {
  const rows: ReturnType<typeof toDeribitCaptureRow>[] = [];
  const failures: string[] = [];
  for (const currency of ['BTC', 'ETH']) {
    try {
      // Retry everything except a definite client error: Bun reports refused connections and DNS failures as a plain
      // Error with a code (ConnectionRefused), not Node's TypeError, so a type check misses them.
      const isClientError = (err: unknown) => /HTTP 4(?!29)\d\d/.test(err instanceof Error ? err.message : String(err));
      const result = await withRetry(async () => {
        const res = await fetchImpl(`${BOOK_SUMMARY_URL}?currency=${currency}&kind=option`);
        if (!res.ok) throw new Error(`Deribit ${currency} options: HTTP ${res.status}`);
        return ((await res.json()) as { result: DeribitSummary[] }).result;
      }, { ...retry, shouldRetry: (err) => !isClientError(err) });
      rows.push(toDeribitCaptureRow(currency, new Date(), result));
    } catch (err) {
      failures.push(`${currency}: ${err instanceof Error ? err.message : String(err)}`);
    }
  }
  return { rows, failures };
}

if (import.meta.main) {
  const { rows, failures } = await captureDeribitOptions();
  mkdirSync(dirname(OUTPUT_PATH), { recursive: true });
  for (const row of rows) {
    appendFileSync(OUTPUT_PATH, `${JSON.stringify(row)}\n`);
    console.log(`${row.currency}: ${row.options.length} options -> ${OUTPUT_PATH}`);
  }
  for (const failure of failures) console.error(failure);
  if (failures.length > 0) process.exit(1);
}
