import { readFileSync } from 'fs';
import { walkForward } from './src/tools/finance/backtest/walk-forward.js';

const fixture = JSON.parse(readFileSync('./src/tools/finance/fixtures/backtest-prices.json', 'utf-8'));
const prices = fixture.tickers['BTC-USD'].closes;

walkForward({
  ticker: 'BTC-USD',
  prices,
  horizon: 14,
  warmup: 120,
  stride: 10,
  matureBullCalibration: true,
}).then(res => {
  const steps = res.steps.filter(s => s.matureBullCalibrationActive);
  console.log("Steps active:", steps.length);
  // Can't see regimeRunLength from step, but it's ok
});
