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
 * defaults on the shared BTC fixture. Both runs are seeded so the comparison
 * isolates the parameter effect from Monte Carlo noise.
 */
const TICKER = 'BTC-USD';
const STRIDE = 5;
const TIMEOUT = 360_000;

let prices: number[];

function metrics(steps: BacktestStep[]) {
  return {
    brier: brierScore(steps),
    dirAcc: directionalAccuracy(steps),
    abstain: abstainRate(steps),
    // Correct calls over all steps (HOLD counts as a miss) = dirAcc × (1 − abstain).
    hitsPerStep: directionalAccuracy(steps) * (1 - abstainRate(steps)),
    ciCov: ciCoverage(steps),
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
        const base = await walkForward({ ticker: TICKER, prices, horizon, stride: STRIDE, randomState: 42 });
        const xiphos = await walkForward({
          ticker: TICKER,
          prices,
          horizon,
          stride: STRIDE,
          ...xiphosBtcWalkForwardConfig(),
        });

        // The xiphos profile warms up earlier (54d vs the default 120d), so its run
        // also forecasts dates the baseline never sees. Score both arms over the
        // baseline's window so the comparison isolates the parameter effect rather
        // than the extra early steps — with traded-only directional accuracy those
        // ~14 steps alone are enough to swing the 7d hit rate past the tolerance.
        const windowStart = base.steps[0]!.t;
        const b = metrics(base.steps);
        const x = metrics(xiphos.steps.filter(s => s.t >= windowStart));
        console.log(
          `[xiphos-btc ${horizon}d] baseline n=${b.n} brier=${b.brier.toFixed(4)} dirAcc=${b.dirAcc.toFixed(4)} ` +
            `abstain=${b.abstain.toFixed(4)} hits/step=${b.hitsPerStep.toFixed(4)} | xiphos n=${x.n} brier=${x.brier.toFixed(4)} ` +
            `dirAcc=${x.dirAcc.toFixed(4)} abstain=${x.abstain.toFixed(4)} hits/step=${x.hitsPerStep.toFixed(4)}`,
        );

        expect(base.errors.length).toBe(0);
        expect(xiphos.errors.length).toBe(0);
        // No meaningful regression (tolerances); on the fixture the xiphos profile improves both.
        expect(x.brier).toBeLessThanOrEqual(b.brier + 0.01);
        expect(x.dirAcc).toBeGreaterThanOrEqual(b.dirAcc - 0.02);
        // dirAcc scores traded steps only, so abstaining more can inflate it. Check
        // the extra abstentions don't cost correct calls: hits per step must hold up.
        expect(x.hitsPerStep).toBeGreaterThanOrEqual(b.hitsPerStep - 0.02);
      },
      TIMEOUT,
    );
  }
});
