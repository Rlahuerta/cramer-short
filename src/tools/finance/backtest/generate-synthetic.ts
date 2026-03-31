/**
 * Generate synthetic but realistic price series for backtest tickers
 * where live API data is unavailable.
 *
 * Uses Geometric Brownian Motion calibrated to real market parameters.
 * Deterministic via seed → fully reproducible.
 */

import { readFileSync, writeFileSync } from 'fs';
import { join } from 'path';

function generateGBM(
  start: number,
  days: number,
  annualDrift: number,
  annualVol: number,
  seed: number,
): number[] {
  const dt = 1 / 252;
  const drift = annualDrift * dt;
  const vol = annualVol * Math.sqrt(dt);
  const prices = [start];

  // Mulberry32 seeded PRNG for reproducibility
  let s = seed;
  function rand(): number {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  }
  function normalRand(): number {
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
  }

  for (let i = 1; i < days; i++) {
    const prev = prices[i - 1];
    const ret = drift + vol * normalRand();
    prices.push(Math.round(prev * Math.exp(ret) * 100) / 100);
  }
  return prices;
}

// Realistic market parameters (calibrated to 2024-2025 behavior)
const SYNTHETIC_TICKERS: Record<string, { start: number; drift: number; vol: number; seed: number; type: string }> = {
  SPY:       { start: 472,   drift: 0.12, vol: 0.15, seed: 42,  type: 'etf' },
  GLD:       { start: 188,   drift: 0.08, vol: 0.12, seed: 137, type: 'etf' },
  QQQ:       { start: 405,   drift: 0.15, vol: 0.20, seed: 271, type: 'etf' },
  'BTC-USD': { start: 42500, drift: 0.30, vol: 0.55, seed: 314, type: 'crypto' },
};

const DAYS = 502; // ~2 years of trading days

// Read existing fixture
const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'));

// Generate dates (trading days, Mon-Fri)
function generateTradingDates(count: number): string[] {
  const dates: string[] = [];
  const d = new Date('2024-01-02');
  while (dates.length < count) {
    const dow = d.getDay();
    if (dow !== 0 && dow !== 6) {
      dates.push(d.toISOString().slice(0, 10));
    }
    d.setDate(d.getDate() + 1);
  }
  return dates;
}

const tradingDates = generateTradingDates(DAYS);

// Fill in missing tickers with synthetic data
for (const [ticker, params] of Object.entries(SYNTHETIC_TICKERS)) {
  const existing = fixture.tickers[ticker];
  if (existing && existing.count > 0) {
    console.log(`  ✓ ${ticker}: already has ${existing.count} real bars (keeping)`);
    continue;
  }

  const closes = generateGBM(params.start, DAYS, params.drift, params.vol, params.seed);
  fixture.tickers[ticker] = {
    type: params.type,
    closes,
    dates: tradingDates,
    count: closes.length,
    synthetic: true,
  };
  console.log(`  ★ ${ticker}: generated ${closes.length} synthetic bars (GBM, seed=${params.seed})`);
}

// Mark fixture metadata
fixture.syntheticNote = 'Some tickers use GBM-synthetic data (marked synthetic:true). Real API data used where available.';

writeFileSync(fixturePath, JSON.stringify(fixture, null, 2));
console.log(`\nSaved to ${fixturePath}`);
console.log('Tickers:', Object.entries(fixture.tickers).map(([k, v]: [string, any]) =>
  `${k}(${v.count}${v.synthetic ? '*' : ''})`).join(', '));
