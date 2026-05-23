"""ADWIN (ADaptive WINdowing) drift detector — mirror of TS adwin.ts.

Bifet & Gavaldà 2007.  Maintains a variable-length window of recent
observations, compressed into exponentially-sized buckets, and shrinks
the window whenever a Hoeffding-bound test detects a change in mean
between any prefix and corresponding suffix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List


@dataclass
class _Bucket:
    total: float
    variance: float
    size: int


@dataclass
class _Row:
    buckets: List[_Bucket] = field(default_factory=list)


class Adwin:
    def __init__(
        self,
        delta: float = 0.002,
        max_buckets: int = 5,
        min_window: int = 10,
    ) -> None:
        self.delta = delta
        self.max_buckets = max_buckets
        self.min_window = min_window
        self._rows: List[_Row] = []
        self._total = 0.0
        self._variance = 0.0
        self._width = 0
        self._last_drift = False

    def size(self) -> int:
        return self._width

    def mean(self) -> float:
        if self._width == 0:
            return float("nan")
        return self._total / self._width

    def drift_detected(self) -> bool:
        return self._last_drift

    def bucket_count(self) -> int:
        return sum(len(r.buckets) for r in self._rows)

    def add(self, value: float) -> bool:
        self._insert(value)
        self._compress()
        self._last_drift = self._check_drift()
        return self._last_drift

    def _insert(self, value: float) -> None:
        if not self._rows:
            self._rows.append(_Row())
        self._rows[0].buckets.append(_Bucket(total=value, variance=0.0, size=1))
        if self._width > 0:
            old_mean = self._total / self._width
            incr = (value - old_mean) ** 2 * self._width / (self._width + 1)
            self._variance += incr
        self._total += value
        self._width += 1

    def _compress(self) -> None:
        level = 0
        while level < len(self._rows):
            row = self._rows[level]
            if len(row.buckets) <= self.max_buckets:
                break
            b1 = row.buckets.pop(0)
            b2 = row.buckets.pop(0)
            new_size = b1.size + b2.size
            new_total = b1.total + b2.total
            m1 = b1.total / b1.size
            m2 = b2.total / b2.size
            new_var = (
                b1.variance + b2.variance + (b1.size * b2.size * (m1 - m2) ** 2) / new_size
            )
            merged = _Bucket(total=new_total, variance=new_var, size=new_size)
            if level + 1 >= len(self._rows):
                self._rows.append(_Row())
            self._rows[level + 1].buckets.append(merged)
            level += 1

    def _check_drift(self) -> bool:
        if self._width < self.min_window:
            return False
        detected = False
        drop_attempts = 0
        max_drops = 5
        while drop_attempts < max_drops:
            dropped_this_pass = False
            n0 = 0
            s0 = 0.0
            flat = []
            for r in range(len(self._rows) - 1, -1, -1):
                for i, b in enumerate(self._rows[r].buckets):
                    flat.append((b, r, i))
            for k in range(len(flat) - 1):
                b = flat[k][0]
                n0 += b.size
                s0 += b.total
                n1 = self._width - n0
                if n0 < self.min_window / 2 or n1 < self.min_window / 2:
                    continue
                mean0 = s0 / n0
                mean1 = (self._total - s0) / n1
                m = 1 / (1 / n0 + 1 / n1)
                d_prime = self.delta / math.log(self._width)
                var_est = self._variance / self._width
                epsilon = math.sqrt((2 / m) * var_est * math.log(2 / d_prime)) + (
                    2 / 3
                ) * math.log(2 / d_prime) / m
                if abs(mean0 - mean1) > epsilon:
                    oldest_row = flat[0][1]
                    dropped = self._rows[oldest_row].buckets.pop(0)
                    self._total -= dropped.total
                    self._width -= dropped.size
                    self._recompute_variance()
                    detected = True
                    dropped_this_pass = True
                    break
            if not dropped_this_pass:
                break
            drop_attempts += 1
        return detected

    def _recompute_variance(self) -> None:
        if self._width == 0:
            self._variance = 0.0
            return
        mean = self._total / self._width
        v = 0.0
        for row in self._rows:
            for b in row.buckets:
                b_mean = b.total / b.size
                v += b.variance + b.size * (b_mean - mean) ** 2
        self._variance = v
