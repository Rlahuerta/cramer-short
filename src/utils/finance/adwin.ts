/**
 * Mirrors `research/models/adwin.py`.
 *
 * ADWIN (ADaptive WINdowing) drift detector.
 *
 * Bifet & Gavaldà 2007 — "Learning from Time-Changing Data with Adaptive
 * Windowing". Maintains a variable-length window of recent observations,
 * compressed into exponentially-sized buckets, and shrinks the window
 * whenever a Hoeffding-bound test detects a change in the mean between
 * any prefix and the corresponding suffix.
 *
 * Use as a "train-only-when-required" trigger for adaptive estimators.
 *
 * Empirical reference: Yelleti 2025 (arXiv 2504.10229) — ADWIN won 4/5
 * datasets in the ROSFD streaming-fraud benchmark vs DDM/EDDM.
 */

interface AdwinOptions {
  /** Per-test false-positive rate. Smaller => fewer false alarms, slower detect. */
  delta?: number;
  /** Max buckets per row before compaction (M in the paper). */
  maxBuckets?: number;
  /** Min total window size before drift checks fire. */
  minWindow?: number;
}

interface Bucket {
  total: number; // sum of values in this bucket
  variance: number; // sum of (xi - bucketMean)^2 within this bucket
  size: number; // number of samples
}

interface Row {
  buckets: Bucket[]; // buckets at this exponential level
}

const DEFAULT_DELTA = 0.002;
const DEFAULT_MAX_BUCKETS = 5;
const DEFAULT_MIN_WINDOW = 10;

export class Adwin {
  private rows: Row[] = [];
  private total = 0;
  private variance = 0; // sum of (xi - mean)^2 over the window (Welford-style)
  private width = 0;
  private readonly delta: number;
  private readonly maxBuckets: number;
  private readonly minWindow: number;
  private lastDrift = false;

  constructor(opts: AdwinOptions = {}) {
    this.delta = opts.delta ?? DEFAULT_DELTA;
    this.maxBuckets = opts.maxBuckets ?? DEFAULT_MAX_BUCKETS;
    this.minWindow = opts.minWindow ?? DEFAULT_MIN_WINDOW;
  }

  size(): number {
    return this.width;
  }

  mean(): number {
    if (this.width === 0) return Number.NaN;
    return this.total / this.width;
  }

  driftDetected(): boolean {
    return this.lastDrift;
  }

  bucketCount(): number {
    let c = 0;
    for (const r of this.rows) c += r.buckets.length;
    return c;
  }

  /** Push a new observation; returns true if drift was detected on this step. */
  add(value: number): boolean {
    this.insert(value);
    this.compress();
    this.lastDrift = this.checkDrift();
    return this.lastDrift;
  }

  private insert(value: number): void {
    if (this.rows.length === 0) this.rows.push({ buckets: [] });
    const row0 = this.rows[0];
    row0.buckets.push({ total: value, variance: 0, size: 1 });
    if (this.width > 0) {
      const oldMean = this.total / this.width;
      const incr = ((value - oldMean) * (value - oldMean) * this.width) / (this.width + 1);
      this.variance += incr;
    }
    this.total += value;
    this.width += 1;
  }

  private compress(): void {
    let level = 0;
    while (level < this.rows.length) {
      const row = this.rows[level];
      if (row.buckets.length <= this.maxBuckets) break;
      // Merge the two oldest buckets (front of the array) and promote.
      const b1 = row.buckets.shift()!;
      const b2 = row.buckets.shift()!;
      const newSize = b1.size + b2.size;
      const newTotal = b1.total + b2.total;
      const newMean = newTotal / newSize;
      const m1 = b1.total / b1.size;
      const m2 = b2.total / b2.size;
      const newVar =
        b1.variance + b2.variance + (b1.size * b2.size * (m1 - m2) * (m1 - m2)) / newSize;
      const merged: Bucket = { total: newTotal, variance: newVar, size: newSize };
      if (level + 1 >= this.rows.length) this.rows.push({ buckets: [] });
      this.rows[level + 1].buckets.push(merged);
      level += 1;
    }
  }

  private checkDrift(): boolean {
    if (this.width < this.minWindow) return false;
    let detected = false;
    let dropAttempts = 0;
    const MAX_DROPS = 5;
    while (dropAttempts < MAX_DROPS) {
      let droppedThisPass = false;
      // Walk buckets from oldest to newest; compute prefix sum and test.
      let n0 = 0;
      let s0 = 0;
      let v0 = 0;
      const flatBuckets: { bucket: Bucket; row: number; idx: number }[] = [];
      for (let r = this.rows.length - 1; r >= 0; r--) {
        for (let i = 0; i < this.rows[r].buckets.length; i++) {
          flatBuckets.push({ bucket: this.rows[r].buckets[i], row: r, idx: i });
        }
      }
      for (let k = 0; k < flatBuckets.length - 1; k++) {
        const b = flatBuckets[k].bucket;
        n0 += b.size;
        s0 += b.total;
        v0 += b.variance;
        const n1 = this.width - n0;
        if (n0 < this.minWindow / 2 || n1 < this.minWindow / 2) continue;
        const mean0 = s0 / n0;
        const mean1 = (this.total - s0) / n1;
        // Hoeffding bound (Bifet & Gavaldà 2007 eq. for ε_cut)
        const m = 1 / (1 / n0 + 1 / n1);
        const dPrime = this.delta / Math.log(this.width);
        const varEst = this.variance / this.width;
        const epsilon =
          Math.sqrt((2 / m) * varEst * Math.log(2 / dPrime)) +
          (2 / 3) * Math.log(2 / dPrime) / m;
        if (Math.abs(mean0 - mean1) > epsilon) {
          // Drop the oldest bucket and recompute.
          const oldest = flatBuckets[0];
          const dropped = this.rows[oldest.row].buckets.shift()!;
          this.total -= dropped.total;
          this.width -= dropped.size;
          // Recompute variance from scratch (cheap; window is bounded).
          this.recomputeVariance();
          detected = true;
          droppedThisPass = true;
          break;
        }
      }
      if (!droppedThisPass) break;
      dropAttempts++;
    }
    return detected;
  }

  private recomputeVariance(): void {
    if (this.width === 0) {
      this.variance = 0;
      return;
    }
    const mean = this.total / this.width;
    let v = 0;
    for (const row of this.rows) {
      for (const b of row.buckets) {
        const bMean = b.total / b.size;
        v += b.variance + b.size * (bMean - mean) * (bMean - mean);
      }
    }
    this.variance = v;
  }
}
