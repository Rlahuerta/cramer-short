/**
 * Stressed scenario price generators for the Markov distribution model.
 *
 * Each scenario produces ~400 daily close prices designed to test model
 * behavior under extreme or unusual market conditions.
 *
 * Uses Mulberry32 seeded PRNG for full reproducibility.
 */

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

export interface StressScenario {
  name: string;
  description: string;
  prices: number[];
  expectedBehavior: string;
}

// ---------------------------------------------------------------------------
// Seeded PRNG (Mulberry32)
// ---------------------------------------------------------------------------

function mulberry32(seed: number): () => number {
  let s = seed;
  return () => {
    s = (s + 0x6D2B79F5) | 0;
    let t = Math.imul(s ^ (s >>> 15), 1 | s);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeNormal(rand: () => number): () => number {
  return () => {
    const u1 = rand();
    const u2 = rand();
    return Math.sqrt(-2 * Math.log(Math.max(u1, 1e-10))) * Math.cos(2 * Math.PI * u2);
  };
}

// ---------------------------------------------------------------------------
// Individual scenario generators
// ---------------------------------------------------------------------------

const N = 400;

/**
 * 1. Crash: Normal uptrend for 200 days, then −30% crash over 10 days,
 *    then gradual recovery.
 */
function generateCrash(basePrice: number, seed: number): StressScenario {
  const rand = mulberry32(seed);
  const normal = makeNormal(rand);
  const prices = [basePrice];
  const dailyVol = 0.015;

  for (let i = 1; i < N; i++) {
    const prev = prices[i - 1];
    let ret: number;

    if (i < 200) {
      // Gentle uptrend
      ret = 0.0005 + dailyVol * normal();
    } else if (i < 210) {
      // Crash: −30% over 10 days → ~−3.5% per day
      ret = -0.035 + dailyVol * 0.5 * normal();
    } else {
      // Recovery: slow grind up with elevated vol
      ret = 0.001 + dailyVol * 1.5 * normal();
    }

    prices.push(Math.round(prev * Math.exp(ret) * 100) / 100);
  }

  return {
    name: 'crash',
    description: 'Normal uptrend → sudden −30% crash → recovery',
    prices,
    expectedBehavior: 'Model should not predict BUY during crash (days 200-210). CIs should widen after crash.',
  };
}

/**
 * 2. V-Recovery: Extended bear period then sharp V-shaped recovery.
 */
function generateVRecovery(basePrice: number, seed: number): StressScenario {
  const rand = mulberry32(seed);
  const normal = makeNormal(rand);
  const prices = [basePrice];
  const dailyVol = 0.012;

  for (let i = 1; i < N; i++) {
    const prev = prices[i - 1];
    let ret: number;

    if (i < 250) {
      // Long bear market: steady decline
      ret = -0.0012 + dailyVol * normal();
    } else if (i < 280) {
      // Sharp recovery: aggressive rally
      ret = 0.025 + dailyVol * 1.5 * normal();
    } else {
      // Stabilization
      ret = 0.0003 + dailyVol * normal();
    }

    prices.push(Math.round(prev * Math.exp(ret) * 100) / 100);
  }

  return {
    name: 'v-recovery',
    description: 'Extended bear market → sharp V-shaped recovery → stabilization',
    prices,
    expectedBehavior: 'Model should eventually shift from SELL/HOLD to BUY during recovery.',
  };
}

/**
 * 3. Sideways Chop: High-frequency mean-reverting noise with zero trend.
 */
function generateSidewaysChop(basePrice: number, seed: number): StressScenario {
  const rand = mulberry32(seed);
  const normal = makeNormal(rand);
  const prices = [basePrice];
  const dailyVol = 0.02;
  const meanRevStrength = 0.05;

  for (let i = 1; i < N; i++) {
    const prev = prices[i - 1];
    // Mean-revert toward basePrice
    const drift = meanRevStrength * (Math.log(basePrice) - Math.log(prev)) / N;
    const ret = drift + dailyVol * normal();
    prices.push(Math.round(prev * Math.exp(ret) * 100) / 100);
  }

  return {
    name: 'sideways-chop',
    description: 'Mean-reverting noise around a flat level — no trend',
    prices,
    expectedBehavior: 'Model should predominantly recommend HOLD. Directional accuracy is less meaningful.',
  };
}

/**
 * 4. Volatility Spike: Low-vol trending period suddenly transitions to 3× vol.
 */
function generateVolSpike(basePrice: number, seed: number): StressScenario {
  const rand = mulberry32(seed);
  const normal = makeNormal(rand);
  const prices = [basePrice];
  const baseVol = 0.008;

  for (let i = 1; i < N; i++) {
    const prev = prices[i - 1];
    let vol: number;
    let drift: number;

    if (i < 250) {
      // Calm trending market
      vol = baseVol;
      drift = 0.0004;
    } else {
      // Volatility explosion — 3× vol, no clear trend
      vol = baseVol * 3;
      drift = 0;
    }

    const ret = drift + vol * normal();
    prices.push(Math.round(prev * Math.exp(ret) * 100) / 100);
  }

  return {
    name: 'volatility-spike',
    description: 'Low-volatility uptrend → sudden 3× volatility explosion',
    prices,
    expectedBehavior: 'CIs should widen significantly after volatility regime change.',
  };
}

/**
 * 5. Regime Flip: Clear bull→bear transition at midpoint.
 */
function generateRegimeFlip(basePrice: number, seed: number): StressScenario {
  const rand = mulberry32(seed);
  const normal = makeNormal(rand);
  const prices = [basePrice];
  const dailyVol = 0.013;

  for (let i = 1; i < N; i++) {
    const prev = prices[i - 1];
    let drift: number;

    if (i < 200) {
      // Bull market
      drift = 0.001;
    } else {
      // Bear market
      drift = -0.001;
    }

    const ret = drift + dailyVol * normal();
    prices.push(Math.round(prev * Math.exp(ret) * 100) / 100);
  }

  return {
    name: 'regime-flip',
    description: 'Bull market first half → bear market second half',
    prices,
    expectedBehavior: 'Model should transition from BUY to SELL/HOLD after regime change.',
  };
}

/**
 * 6. Persistent Bear: Steady downtrend throughout.
 *    Tests if model over-predicts UP.
 */
function generatePersistentBear(basePrice: number, seed: number): StressScenario {
  const rand = mulberry32(seed);
  const normal = makeNormal(rand);
  const prices = [basePrice];
  const dailyVol = 0.014;

  for (let i = 1; i < N; i++) {
    const prev = prices[i - 1];
    const ret = -0.0008 + dailyVol * normal();
    prices.push(Math.round(prev * Math.exp(ret) * 100) / 100);
  }

  return {
    name: 'persistent-bear',
    description: 'Steady downtrend for the entire series',
    prices,
    expectedBehavior: 'Model should eventually output SELL or HOLD — not continuously BUY.',
  };
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/**
 * Generate all stress scenarios with deterministic seeds.
 * Each scenario has 400 data points.
 */
export function generateStressScenarios(basePrice = 100): StressScenario[] {
  return [
    generateCrash(basePrice, 1001),
    generateVRecovery(basePrice, 2002),
    generateSidewaysChop(basePrice, 3003),
    generateVolSpike(basePrice, 4004),
    generateRegimeFlip(basePrice, 5005),
    generatePersistentBear(basePrice, 6006),
  ];
}
