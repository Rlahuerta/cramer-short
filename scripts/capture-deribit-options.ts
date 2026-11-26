/**
 * Appends today's Deribit BTC and ETH option books to .cramer-short/cache/options/deribit.jsonl (relative to cwd),
 * one row per currency. Forward data for H5 (option-implied densities as the anchor): there is no free history,
 * so the test needs months of daily captures. Public API, no key.
 *
 * Usage: bun scripts/capture-deribit-options.ts
 */
import { appendFileSync, mkdirSync } from 'node:fs';
import { dirname } from 'node:path';

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

if (import.meta.main) {
  mkdirSync(dirname(OUTPUT_PATH), { recursive: true });
  for (const currency of ['BTC', 'ETH']) {
    const res = await fetch(`${BOOK_SUMMARY_URL}?currency=${currency}&kind=option`);
    if (!res.ok) throw new Error(`Deribit ${currency} options: HTTP ${res.status}`);
    const { result } = (await res.json()) as { result: DeribitSummary[] };
    appendFileSync(OUTPUT_PATH, `${JSON.stringify(toDeribitCaptureRow(currency, new Date(), result))}\n`);
    console.log(`${currency}: ${result.length} options -> ${OUTPUT_PATH}`);
  }
}
