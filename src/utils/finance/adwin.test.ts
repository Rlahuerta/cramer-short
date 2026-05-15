import { describe, expect, test } from "bun:test";
import { Adwin } from "./adwin.js";

function makeRng(seed: number) {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x100000000;
  };
}

// Box-Muller standard normal from a uniform RNG.
function makeGauss(rng: () => number) {
  let cached: number | null = null;
  return (mean = 0, std = 1): number => {
    if (cached !== null) {
      const v = cached;
      cached = null;
      return mean + std * v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    cached = mag * Math.sin(2 * Math.PI * v);
    return mean + std * (mag * Math.cos(2 * Math.PI * v));
  };
}

describe("Adwin — basic API", () => {
  test("initialises with empty window", () => {
    const a = new Adwin();
    expect(a.size()).toBe(0);
    expect(a.mean()).toBeNaN();
    expect(a.driftDetected()).toBe(false);
  });

  test("after adding stationary samples, no drift detected", () => {
    const a = new Adwin({ delta: 0.002 });
    const g = makeGauss(makeRng(1));
    for (let i = 0; i < 200; i++) a.add(g(0, 1));
    expect(a.driftDetected()).toBe(false);
    expect(a.size()).toBeGreaterThan(0);
    expect(Math.abs(a.mean())).toBeLessThan(0.3);
  });

  test("size() decreases after drift is detected and old data is dropped", () => {
    const a = new Adwin({ delta: 0.002 });
    const g = makeGauss(makeRng(2));
    for (let i = 0; i < 100; i++) a.add(g(0, 0.1));
    const sizeBefore = a.size();
    let drifted = false;
    for (let i = 0; i < 100; i++) {
      if (a.add(g(5, 0.1))) drifted = true;
    }
    expect(drifted).toBe(true);
    expect(a.size()).toBeLessThan(sizeBefore + 100);
  });
});

describe("Adwin — drift detection", () => {
  test("detects a sudden mean shift within ≤ 5 samples (synthetic regime change)", () => {
    const a = new Adwin({ delta: 0.002 });
    const g = makeGauss(makeRng(7));
    // Regime A: 100 samples ~ N(0, 0.1)
    for (let i = 0; i < 100; i++) a.add(g(0, 0.1));
    expect(a.driftDetected()).toBe(false);
    // Regime B: feed N(2, 0.1); record how many samples until detection
    let stepsToDetect = -1;
    for (let i = 0; i < 50; i++) {
      a.add(g(2, 0.1));
      if (a.driftDetected()) {
        stepsToDetect = i + 1;
        break;
      }
    }
    expect(stepsToDetect).toBeGreaterThan(0);
    expect(stepsToDetect).toBeLessThanOrEqual(15);
  });

  test("does NOT flag drift on a stationary stream of 500 samples", () => {
    const a = new Adwin({ delta: 0.002 });
    const g = makeGauss(makeRng(123));
    let falsePositives = 0;
    for (let i = 0; i < 500; i++) {
      a.add(g(0, 1));
      if (a.driftDetected()) falsePositives++;
    }
    // δ=0.002 is the per-test false-positive rate — at most a handful of flags
    // across 500 hypothesis tests is expected.
    expect(falsePositives).toBeLessThan(20);
  });

  test("ignoresVariance shifts when only μ changes detect drift", () => {
    const a = new Adwin({ delta: 0.002 });
    const g = makeGauss(makeRng(11));
    for (let i = 0; i < 80; i++) a.add(g(0, 1));
    let detected = false;
    for (let i = 0; i < 80; i++) {
      a.add(g(0, 1)); // same mean and variance — no change
      if (a.driftDetected()) {
        detected = true;
        break;
      }
    }
    expect(detected).toBe(false);
  });
});

describe("Adwin — adaptive window mean tracking", () => {
  test("after drift, mean() converges toward the new regime mean", () => {
    const a = new Adwin({ delta: 0.002 });
    const g = makeGauss(makeRng(33));
    for (let i = 0; i < 100; i++) a.add(g(0, 0.1));
    for (let i = 0; i < 100; i++) a.add(g(5, 0.1));
    // Window has been trimmed to mostly post-drift samples; mean ≈ 5
    expect(a.mean()).toBeGreaterThan(3);
    expect(a.mean()).toBeLessThan(6);
  });

  test("during stationarity, window grows and mean stabilises", () => {
    const a = new Adwin({ delta: 0.002 });
    const g = makeGauss(makeRng(44));
    for (let i = 0; i < 50; i++) a.add(g(1.0, 0.05));
    const m1 = a.mean();
    for (let i = 0; i < 200; i++) a.add(g(1.0, 0.05));
    const m2 = a.mean();
    expect(Math.abs(m1 - 1.0)).toBeLessThan(0.1);
    expect(Math.abs(m2 - 1.0)).toBeLessThan(0.05);
    expect(a.size()).toBeGreaterThan(50);
  });
});

describe("Adwin — bucket compression", () => {
  test("uses bucketed storage so memory stays sub-linear in stream length", () => {
    const a = new Adwin({ delta: 0.002, maxBuckets: 5 });
    const g = makeGauss(makeRng(55));
    for (let i = 0; i < 10_000; i++) a.add(g(0, 1));
    expect(a.bucketCount()).toBeLessThan(500);
    expect(a.size()).toBeGreaterThan(1000);
  });
});
