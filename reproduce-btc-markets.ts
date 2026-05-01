#!/usr/bin/env bun
/**
 * Reproduce Polymarket forecast market selection for BTC with horizon_days=1
 * Uses the same internal logic as src/tools/finance/polymarket-forecast.ts
 */

import { resolveTickerSearchIdentity } from './src/tools/finance/asset-resolver.js';
import { extractSignals, scoreMarketRelevance } from './src/tools/finance/signal-extractor.js';
import { fetchPolymarketMarkets, type PolymarketMarketResult } from './src/tools/finance/polymarket.js';
import { DEFAULT_POLYMARKET_SNAPSHOTS_PATH } from './src/tools/finance/polymarket-snapshots.js';

interface RawMarket {
  marketId?: string;
  question: string;
  probability: number;
  volume24h: number;
  signalCategory: string;
  endDate?: string | null;
}

function isUsablePolymarketProbability(probability: number): boolean {
  return Number.isFinite(probability) && probability >= 0 && probability <= 1;
}

function daysUntilEndDate(endDate: string | null | undefined): number | null {
  if (!endDate) return null;
  const target = new Date(endDate);
  if (Number.isNaN(target.getTime())) return null;
  return (target.getTime() - Date.now()) / 86_400_000;
}

function isAlignedToHorizon(daysToExpiry: number | null, horizonDays: number): boolean {
  if (daysToExpiry === null) return false;
  return Math.abs(daysToExpiry - horizonDays) <= Math.max(1.5, horizonDays * 0.35);
}

async function main() {
  const ticker = 'BTC';
  const horizonDays = 1;
  
  console.log(`\n🔍 Reproducing Polymarket forecast for ${ticker} with horizon=${horizonDays} days\n`);

  // Step 1: Resolve ticker search identity
  const searchIdentity = resolveTickerSearchIdentity(ticker);
  console.log(`✓ Resolved: ${ticker} → canonical: ${searchIdentity.canonicalTicker}`);

  // Step 2: Extract signals
  const signals = extractSignals(searchIdentity.canonicalTicker).slice(0, 5);
  console.log(`✓ Extracted ${signals.length} signals:`);
  signals.forEach((sig, i) => {
    console.log(`  ${i + 1}. ${sig.category}: "${sig.searchPhrase}"`);
    if (sig.queryVariants?.length) {
      console.log(`     Variants: ${sig.queryVariants.join(', ')}`);
    }
  });

  // Step 3: Fetch markets for each signal
  const allResults = await Promise.allSettled(
    signals.map((sig) => {
      const phrases = [sig.searchPhrase, ...(sig.queryVariants ?? [])];
      return Promise.allSettled(
        phrases.map((phrase) =>
          fetchPolymarketMarkets(phrase, 5, { snapshotFilePath: DEFAULT_POLYMARKET_SNAPSHOTS_PATH })
        )
      ).then((settledVariants) =>
        settledVariants
          .filter((r): r is PromiseFulfilledResult<PolymarketMarketResult[]> => r.status === 'fulfilled')
          .flatMap((r) => r.value)
          .filter((m) => scoreMarketRelevance(m.question, sig.category) > 0)
          .map((m) => ({ ...m, signalCategory: sig.category }))
      );
    })
  );

  // Step 4: Deduplicate and convert to raw markets
  const seen = new Set<string>();
  const rawMarkets: RawMarket[] = [];

  for (let i = 0; i < signals.length; i++) {
    const result = allResults[i];
    if (result.status !== 'fulfilled') continue;

    for (const m of result.value) {
      if (!isUsablePolymarketProbability(m.probability)) continue;
      if (seen.has(m.question)) continue;

      seen.add(m.question);
      rawMarkets.push({
        marketId: m.marketId,
        question: m.question,
        probability: m.probability,
        volume24h: m.volume24h,
        signalCategory: m.signalCategory,
        endDate: m.endDate,
      });
    }
  }

  console.log(`\n✓ Selected ${rawMarkets.length} unique markets after deduplication\n`);

  // Step 5: Build table
  const tableData: Array<{
    signal: string;
    question: string;
    prob: string;
    volume: string;
    endDate: string;
    daysToExpiry: string;
    isAligned: string;
  }> = [];

  let farFromHorizonCount = 0;

  for (const market of rawMarkets) {
    const daysToExpiry = daysUntilEndDate(market.endDate);
    const aligned = isAlignedToHorizon(daysToExpiry, horizonDays);

    if (!aligned) {
      farFromHorizonCount++;
    }

    tableData.push({
      signal: market.signalCategory,
      question: market.question.length > 60 ? market.question.slice(0, 57) + '...' : market.question,
      prob: (market.probability * 100).toFixed(1) + '%',
      volume: market.volume24h >= 1_000_000
        ? (market.volume24h / 1_000_000).toFixed(2) + 'M'
        : market.volume24h >= 1_000
          ? (market.volume24h / 1_000).toFixed(1) + 'K'
          : market.volume24h.toFixed(0),
      endDate: market.endDate ? new Date(market.endDate).toISOString().split('T')[0] : 'N/A',
      daysToExpiry: daysToExpiry !== null ? daysToExpiry.toFixed(2) : 'N/A',
      isAligned: aligned ? '✓' : '✗',
    });
  }

  // Print table
  if (tableData.length === 0) {
    console.log('No markets found.');
  } else {
    console.log('┌─────────────────┬──────────────────────────────────────────┬──────┬────────┬────────────┬──────────┬───────┐');
    console.log('│ Signal Category │ Question                                 │ Prob │ Volume │ End Date   │ DaysLeft │ Aligned│');
    console.log('├─────────────────┼──────────────────────────────────────────┼──────┼────────┼────────────┼──────────┼───────┤');

    tableData.forEach((row) => {
      const signal = row.signal.padEnd(15);
      const question = row.question.padEnd(40);
      const prob = row.prob.padStart(5);
      const volume = row.volume.padStart(6);
      const endDate = row.endDate.padEnd(10);
      const daysLeft = row.daysToExpiry.padStart(8);
      const aligned = row.isAligned.padEnd(6);

      console.log(`│ ${signal} │ ${question} │${prob} │ ${volume} │ ${endDate} │ ${daysLeft} │ ${aligned}│`);
    });

    console.log('└─────────────────┴──────────────────────────────────────────┴──────┴────────┴────────────┴──────────┴───────┘');
  }

  // Summary
  console.log(`\n📊 Summary:`);
  console.log(`  Total selected markets: ${rawMarkets.length}`);
  console.log(`  Aligned to 1d horizon: ${rawMarkets.length - farFromHorizonCount}`);
  console.log(`  Far from 1d horizon: ${farFromHorizonCount}`);
  console.log(`  Alignment threshold: abs(daysToExpiry - 1) <= max(1.5, 0.35) = 1.5 days`);

  // List the far ones
  if (farFromHorizonCount > 0) {
    console.log(`\n⚠️  Markets far from 1d horizon:`);
    for (const market of rawMarkets) {
      const daysToExpiry = daysUntilEndDate(market.endDate);
      if (!isAlignedToHorizon(daysToExpiry, horizonDays)) {
        console.log(`  - [${market.signalCategory}] ${market.question.slice(0, 70)}`);
        console.log(`    Days to expiry: ${daysToExpiry?.toFixed(2) ?? 'N/A'}`);
      }
    }
  }

  console.log();
}

main().catch(console.error);
