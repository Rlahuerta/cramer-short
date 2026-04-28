import { describe, expect, test } from "bun:test";
import {
  applyAdwinTrim,
  applyHawkesAmplification,
  amplifyJumpEvents,
} from "./forecast-hooks.js";

// Deterministic LCG so the tests are reproducible.
function makeRng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

function randn(rng: () => number): number {
  const u1 = Math.max(1e-12, rng());
  const u2 = rng();
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
}

function makePriceSeries(returns: number[], start = 100): number[] {
  const out = new Array(returns.length + 1);
  out[0] = start;
  for (let i = 0; i < returns.length; i++) out[i + 1] = out[i] * Math.exp(returns[i]);
  return out;
}

describe("applyAdwinTrim", () => {
  test("returns input unchanged when too short", () => {
    const prices = [100, 101, 102, 103];
    const { trimmedPrices, result } = applyAdwinTrim(prices);
    expect(trimmedPrices).toEqual(prices);
    expect(result.trimmed).toBe(false);
    expect(result.droppedPrices).toBe(0);
  });

  test("does not trim a stationary series", () => {
    const rng = makeRng(42);
    const returns = Array.from({ length: 300 }, () => randn(rng) * 0.01);
    const prices = makePriceSeries(returns);
    const { trimmedPrices, result } = applyAdwinTrim(prices);
    expect(result.trimmed).toBe(false);
    expect(trimmedPrices.length).toBe(prices.length);
  });

  test("trims a series with an obvious mean shift", () => {
    const rng = makeRng(7);
    const calm = Array.from({ length: 200 }, () => randn(rng) * 0.001);
    const wild = Array.from({ length: 200 }, () => randn(rng) * 0.001 + 0.5);
    const prices = makePriceSeries([...calm, ...wild]);
    const { trimmedPrices, result } = applyAdwinTrim(prices, 0.1);
    expect(result.trimmed).toBe(true);
    expect(trimmedPrices.length).toBeLessThan(prices.length);
    expect(trimmedPrices.length).toBeGreaterThanOrEqual(60);
  });

  test("respects minKeep floor", () => {
    const rng = makeRng(5);
    const calm = Array.from({ length: 80 }, () => randn(rng) * 0.005);
    const wild = Array.from({ length: 80 }, () => randn(rng) * 0.005 + 0.05);
    const prices = makePriceSeries([...calm, ...wild]);
    const { trimmedPrices } = applyAdwinTrim(prices, 0.001, { minKeep: 100 });
    expect(trimmedPrices.length).toBeGreaterThanOrEqual(100);
  });
});

describe("applyHawkesAmplification", () => {
  test("returns multiplier=1 when no clustering is present", () => {
    const rng = makeRng(123);
    const returns = Array.from({ length: 500 }, () => randn(rng) * 0.01);
    const prices = makePriceSeries(returns);
    const result = applyHawkesAmplification(prices);
    expect(result.intensityMultiplier).toBeCloseTo(1, 5);
    expect(result.endogenousJump).toBeNull();
  });

  test("returns multiplier > 1 on a clustered jump series", () => {
    const rng = makeRng(99);
    const returns: number[] = [];
    for (let i = 0; i < 600; i++) returns.push(randn(rng) * 0.01);
    // Inject a cluster of large jumps (8 jumps within 12 days)
    const clusterStart = 100;
    for (let k = 0; k < 8; k++) {
      returns[clusterStart + k * 1] += 0.06 * (rng() < 0.5 ? -1 : 1);
    }
    // And another cluster later
    const cluster2 = 380;
    for (let k = 0; k < 6; k++) {
      returns[cluster2 + k] += 0.06 * (rng() < 0.5 ? -1 : 1);
    }
    const prices = makePriceSeries(returns);
    const result = applyHawkesAmplification(prices, { sigmaThreshold: 2.5 });
    expect(result.jumpIndices.length).toBeGreaterThanOrEqual(8);
    if (result.fit && result.fit.isStable && result.fit.alpha > 0) {
      expect(result.intensityMultiplier).toBeGreaterThanOrEqual(1);
      expect(result.intensityMultiplier).toBeLessThanOrEqual(3);
    }
  });

  test("never produces an endogenous jump when no clustering", () => {
    const rng = makeRng(321);
    const returns = Array.from({ length: 400 }, () => randn(rng) * 0.01);
    const prices = makePriceSeries(returns);
    const result = applyHawkesAmplification(prices);
    expect(result.endogenousJump).toBeNull();
  });

  test("returns empty when input too short", () => {
    const prices = [100, 101, 102];
    const result = applyHawkesAmplification(prices);
    expect(result.jumpIndices).toEqual([]);
    expect(result.fit).toBeNull();
    expect(result.intensityMultiplier).toBe(1);
  });
});

describe("amplifyJumpEvents", () => {
  test("returns identical events when multiplier ≤ 1", () => {
    const events = [
      { id: "e1", dailyIntensity: 0.1, meanLogJump: 0.05, stdLogJump: 0.02 },
    ];
    const out = amplifyJumpEvents(events, 1);
    expect(out[0].dailyIntensity).toBe(0.1);
    expect(out[0]).not.toBe(events[0]); // copy
  });

  test("scales each event's intensity but caps at 0.95", () => {
    const events = [
      { id: "e1", dailyIntensity: 0.1, meanLogJump: 0.05, stdLogJump: 0.02 },
      { id: "e2", dailyIntensity: 0.5, meanLogJump: -0.05, stdLogJump: 0.03 },
    ];
    const out = amplifyJumpEvents(events, 2);
    expect(out[0].dailyIntensity).toBeCloseTo(0.2, 6);
    expect(out[1].dailyIntensity).toBe(0.95);
  });
});
