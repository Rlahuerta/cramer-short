import { describe, expect, test } from "bun:test";
import { HawkesIntensity, fitHawkesMLE, simulateHawkes } from "./hawkes.js";

// Deterministic Box-Muller PRNG (mirrors conformal.test.ts)
function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

describe("HawkesIntensity — instantaneous intensity", () => {
  test("with no history, intensity equals the baseline μ", () => {
    const h = new HawkesIntensity({ mu: 0.5, alpha: 0.4, beta: 1.0 });
    expect(h.intensity(10, [])).toBeCloseTo(0.5, 10);
  });

  test("after a single jump at t=0, intensity at t=0 equals μ + α", () => {
    const h = new HawkesIntensity({ mu: 0.5, alpha: 0.4, beta: 1.0 });
    expect(h.intensity(0, [0])).toBeCloseTo(0.9, 10);
  });

  test("intensity decays exponentially with rate β toward μ", () => {
    const h = new HawkesIntensity({ mu: 0.5, alpha: 0.4, beta: 1.0 });
    // at Δt=1 with single past jump: 0.5 + 0.4·e^-1
    expect(h.intensity(1, [0])).toBeCloseTo(0.5 + 0.4 * Math.exp(-1), 10);
    // far future: → μ
    expect(h.intensity(1000, [0])).toBeCloseTo(0.5, 8);
  });

  test("excitation is additive across past jumps", () => {
    const h = new HawkesIntensity({ mu: 0.0, alpha: 1.0, beta: 1.0 });
    // jumps at t=0 and t=1, observed at t=2
    const expected = Math.exp(-2) + Math.exp(-1);
    expect(h.intensity(2, [0, 1])).toBeCloseTo(expected, 10);
  });

  test("ignores future jumps (only t_i < t contribute)", () => {
    const h = new HawkesIntensity({ mu: 0.1, alpha: 0.5, beta: 1.0 });
    // Past jump at t=0.5 contributes; future jump at t=2 must not.
    expect(h.intensity(1, [0.5, 2.0])).toBeCloseTo(
      0.1 + 0.5 * Math.exp(-0.5),
      10,
    );
  });
});

describe("HawkesIntensity — branching ratio + stability", () => {
  test("branchingRatio() = α/β", () => {
    expect(new HawkesIntensity({ mu: 1, alpha: 0.5, beta: 1 }).branchingRatio()).toBeCloseTo(0.5);
    expect(new HawkesIntensity({ mu: 1, alpha: 0.9, beta: 1 }).branchingRatio()).toBeCloseTo(0.9);
  });

  test("isStable() requires α < β (branching ratio < 1)", () => {
    expect(new HawkesIntensity({ mu: 1, alpha: 0.5, beta: 1 }).isStable()).toBe(true);
    expect(new HawkesIntensity({ mu: 1, alpha: 1.0, beta: 1 }).isStable()).toBe(false);
    expect(new HawkesIntensity({ mu: 1, alpha: 1.5, beta: 1 }).isStable()).toBe(false);
  });

  test("constructor rejects non-finite μ, non-positive β or negative α", () => {
    expect(() => new HawkesIntensity({ mu: NaN, alpha: 0.1, beta: 1 })).toThrow();
    expect(() => new HawkesIntensity({ mu: 1, alpha: -0.1, beta: 1 })).toThrow();
    expect(() => new HawkesIntensity({ mu: 1, alpha: 0.1, beta: 0 })).toThrow();
  });
});

describe("HawkesIntensity — log-likelihood", () => {
  test("empty event stream over horizon T gives logL = -μ·T", () => {
    const h = new HawkesIntensity({ mu: 0.5, alpha: 0.3, beta: 1.0 });
    expect(h.logLikelihood([], 10)).toBeCloseTo(-5, 10);
  });

  test("single event at t=0 over horizon T", () => {
    const h = new HawkesIntensity({ mu: 1.0, alpha: 0.0, beta: 1.0 });
    // Σ log λ(t_i) − ∫₀ᵀ λ(s) ds = log 1 − 1·T = -T
    expect(h.logLikelihood([0], 5)).toBeCloseTo(-5, 8);
  });

  test("compensator includes the excitation integral", () => {
    const h2 = new HawkesIntensity({ mu: 0.1, alpha: 0.5, beta: 1.0 });
    const ll = h2.logLikelihood([0, 1], 2);
    // analytical:
    // logλ(0)=log(0.1), logλ(1)=log(0.1 + 0.5·e^-1)
    // ∫λ = 0.1·2 + 0.5·(1−e^-2) + 0.5·(1−e^-1)
    const expected =
      Math.log(0.1) +
      Math.log(0.1 + 0.5 * Math.exp(-1)) -
      (0.2 + 0.5 * (1 - Math.exp(-2)) + 0.5 * (1 - Math.exp(-1)));
    expect(ll).toBeCloseTo(expected, 8);
  });
});

describe("simulateHawkes (Ogata thinning)", () => {
  test("with α=0 (no self-excitation) recovers a homogeneous Poisson process", () => {
    const rng = makeRng(42);
    const events = simulateHawkes({ mu: 5, alpha: 0, beta: 1 }, 100, rng);
    // Expected count ≈ μ·T = 500. Allow ±15% slack on a single seed.
    expect(events.length).toBeGreaterThan(425);
    expect(events.length).toBeLessThan(575);
    // Inter-arrivals should be roughly exponentially distributed (mean ≈ 1/μ = 0.2)
    const gaps = events.slice(1).map((t, i) => t - events[i]);
    const meanGap = gaps.reduce((a, b) => a + b, 0) / gaps.length;
    expect(meanGap).toBeGreaterThan(0.16);
    expect(meanGap).toBeLessThan(0.24);
  });

  test("self-exciting (α>0) produces clustering: more short gaps than Poisson would", () => {
    const rngHawkes = makeRng(7);
    const rngPoisson = makeRng(7);
    const T = 200;
    const baseline = simulateHawkes({ mu: 1, alpha: 0, beta: 1 }, T, rngPoisson);
    const clustered = simulateHawkes({ mu: 1, alpha: 0.7, beta: 1 }, T, rngHawkes);
    // Clustered process has higher overall rate (μ/(1−α/β) = 1/0.3 ≈ 3.33)
    expect(clustered.length).toBeGreaterThan(baseline.length * 1.8);
    // Fraction of inter-arrivals < 0.1 should be higher under clustering
    const shortFrac = (events: number[]) => {
      const gaps = events.slice(1).map((t, i) => t - events[i]);
      return gaps.filter((g) => g < 0.1).length / gaps.length;
    };
    expect(shortFrac(clustered)).toBeGreaterThan(shortFrac(baseline));
  });

  test("returns events strictly increasing within (0, T]", () => {
    const events = simulateHawkes({ mu: 2, alpha: 0.3, beta: 1.5 }, 50, makeRng(123));
    for (let i = 1; i < events.length; i++) {
      expect(events[i]).toBeGreaterThan(events[i - 1]);
    }
    if (events.length > 0) {
      expect(events[events.length - 1]).toBeLessThanOrEqual(50);
      expect(events[0]).toBeGreaterThan(0);
    }
  });
});

describe("fitHawkesMLE — recovers known parameters", () => {
  test("fits baseline-only process (α=0): μ̂ ≈ true μ", () => {
    const rng = makeRng(2024);
    const events = simulateHawkes({ mu: 2, alpha: 0, beta: 1 }, 500, rng);
    const fit = fitHawkesMLE(events, 500, { initialMu: 1, initialAlpha: 0.01, initialBeta: 1 });
    expect(fit.mu).toBeCloseTo(2, 0);
    expect(fit.alpha).toBeLessThan(0.5);
    expect(fit.isStable).toBe(true);
  });

  test("fit respects α/β < 1 stability constraint", () => {
    const rng = makeRng(99);
    const events = simulateHawkes({ mu: 1, alpha: 0.5, beta: 1 }, 1000, rng);
    const fit = fitHawkesMLE(events, 1000);
    expect(fit.alpha / fit.beta).toBeLessThan(1.0);
    expect(fit.isStable).toBe(true);
  });

  test("returns NaN-free finite parameters", () => {
    const events = simulateHawkes({ mu: 1, alpha: 0.4, beta: 1 }, 200, makeRng(5));
    const fit = fitHawkesMLE(events, 200);
    expect(Number.isFinite(fit.mu)).toBe(true);
    expect(Number.isFinite(fit.alpha)).toBe(true);
    expect(Number.isFinite(fit.beta)).toBe(true);
    expect(Number.isFinite(fit.logLikelihood)).toBe(true);
  });
});
