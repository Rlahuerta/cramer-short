import { describe, beforeAll, expect } from 'bun:test';
import { readFileSync } from 'fs';
import { join } from 'path';
import { integrationIt } from '@/utils/test-guards.js';
import { walkForward } from './walk-forward.js';
import { abstainRate, brierScore, ciCoverage, directionalAccuracy, type BacktestStep } from './metrics.js';
import { xiphosBtcWalkForwardConfig } from './xiphos-btc-preset.js';

/**
 * Verifies the xiphos BTC-tuned Markov profile (vbparam_1d) does not regress —
 * and generally improves — BTC forecast performance vs. cramer-short's current
 * defaults on the shared BTC fixture. Both runs are seeded and share the xiphos
 * warmup (so they score the same forecast origins), which isolates the parameter
 * effect from Monte Carlo noise and from comparing different evaluation windows.
 */
const TICKER = 'BTC-USD';
const STRIDE = 5;
const TIMEOUT = 360_000;

let prices: number[];

function metrics(steps: BacktestStep[]) {
  return {
    brier: brierScore(steps),
    dirAcc: directionalAccuracy(steps),
    ciCov: ciCoverage(steps),
    abstain: abstainRate(steps),
    n: steps.length,
  };
}

describe('xiphos BTC profile walk-forward benchmark', () => {
  beforeAll(() => {
    const fixturePath = join(import.meta.dir, '..', 'fixtures', 'backtest-prices.json');
    const fixture = JSON.parse(readFileSync(fixturePath, 'utf-8'));
    prices = fixture.tickers[TICKER].closes;
  });

  for (const horizon of [1, 7] as const) {
    integrationIt(
      `does not regress Brier/directional accuracy at ${horizon}d`,
      async () => {
        const xiphosConfig = xiphosBtcWalkForwardConfig();
        const base = await walkForward({
          ticker: TICKER,
          prices,
          horizon,
          stride: STRIDE,
          warmup: xiphosConfig.warmup,
          randomState: 42,
        });
        const xiphos = await walkForward({
          ticker: TICKER,
          prices,
          horizon,
          stride: STRIDE,
          ...xiphosConfig,
        });

        const b = metrics(base.steps);
        const x = metrics(xiphos.steps);
        console.log(
          `[xiphos-btc ${horizon}d] baseline brier=${b.brier.toFixed(4)} dirAcc=${b.dirAcc.toFixed(4)} | ` +
            `xiphos brier=${x.brier.toFixed(4)} dirAcc=${x.dirAcc.toFixed(4)} (n=${x.n}, abstain ${b.abstain.toFixed(4)} → ${x.abstain.toFixed(4)})`,
        );

        expect(base.errors.length).toBe(0);
        expect(xiphos.errors.length).toBe(0);
        expect(xiphos.steps.map((step) => step.t)).toEqual(base.steps.map((step) => step.t));
        // No meaningful regression (tolerances). Directional accuracy is scored on traded
        // steps only, so the abstain rate is bounded too.
        expect(x.brier).toBeLessThanOrEqual(b.brier + 0.01);
        expect(x.dirAcc).toBeGreaterThanOrEqual(b.dirAcc - 0.02);
        expect(x.abstain).toBeLessThanOrEqual(b.abstain + 0.05);
      },
      TIMEOUT,
    );
  }
});
