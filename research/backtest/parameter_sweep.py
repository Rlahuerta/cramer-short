"""Parameter sweep for Markov backtest: find optimal decay_rate × threshold_multiplier.

Usage:
    python -m research.backtest.parameter_sweep --ticker BTC --csv prices.csv --horizon 7

Output: table sorted by native_direction_accuracy, showing best configs first.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.data.prices import fetch_historical_prices
from research.backtest.walk_forward import walk_forward

# ── CLI (mirrors markov_tool_backtest) ───────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Sweep Markov backtest parameters.")
    p.add_argument("--ticker", default="BTC")
    p.add_argument("--csv", help="Offline CSV with close column.")
    p.add_argument("--horizon", type=int, default=7, help="Forecast horizon in days.")
    p.add_argument("--days", type=int, default=365, help="Lookback days.")
    p.add_argument("--warmup", type=int, default=120)
    p.add_argument("--stride", type=int, default=10)
    p.add_argument("--top", type=int, default=10, help="Show top N results.")
    return p


def load_prices(args: argparse.Namespace) -> list[float]:
    import math
    import pandas as pd

    if args.csv:
        path = Path(args.csv).expanduser()
        frame = pd.read_csv(path)
        col = "close" if "close" in frame.columns else frame.columns[0]
        closes = pd.to_numeric(frame[col], errors="coerce").dropna()
        prices = [float(v) for v in closes if math.isfinite(float(v)) and v > 0]
        if args.days > 0:
            prices = prices[-args.days:]
        return prices

    frame = fetch_historical_prices(args.ticker, days=args.days, sources=None)
    closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
    return [float(v) for v in closes if math.isfinite(float(v)) and v > 0]


# ── Sweep ────────────────────────────────────────────────────────────────

DECAY_RATES = [0.95, 0.96, 0.97, 0.975, 0.98, 0.985, 0.99, 0.992, 0.995, 0.998]
THRESHOLD_MULTIPLIERS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]


def run_sweep(
    prices: list[float],
    horizon: int,
    warmup: int,
    stride: int,
    top_n: int,
) -> list[dict[str, Any]]:
    results = []

    for dr in DECAY_RATES:
        for tm in THRESHOLD_MULTIPLIERS:
            result = walk_forward(
                prices,
                horizon=horizon,
                warmup=warmup,
                stride=stride,
                decay_rate=dr,
                return_threshold_multiplier=tm,
            )
            if not result.steps:
                continue

            nat_correct = sum(1 for s in result.steps if s.direction_correct)
            nat_acc = nat_correct / len(result.steps)

            # TS-style directional accuracy
            from research.backtest.markov_tool_backtest import ts_directional_accuracy
            ts_acc = ts_directional_accuracy(result.steps, horizon)

            # Decisiveness: mean |p_up - 0.5|
            decisiveness = float(np.mean([
                abs(s.predicted_prob - 0.5) for s in result.steps
            ]))

            results.append({
                "decay_rate": dr,
                "threshold_multiplier": tm,
                "steps": len(result.steps),
                "nat_acc": nat_acc,
                "ts_acc": ts_acc,
                "decisiveness": decisiveness,
            })

    # Sort by native accuracy descending, then decisiveness descending
    results.sort(key=lambda r: (-r["nat_acc"], -r["decisiveness"]))
    return results[:top_n]


def format_table(results: list[dict[str, Any]]) -> str:
    lines = [
        "decay  thr_mult  steps  nat_acc  ts_acc   decisiveness",
        "-----  --------  -----  -------  -------  ------------",
    ]
    for r in results:
        lines.append(
            f"{r['decay_rate']:.3f}  "
            f"{r['threshold_multiplier']:>8.1f}  "
            f"{r['steps']:>5}  "
            f"{r['nat_acc']:>6.1%}  "
            f"{r['ts_acc']:>6.1%}  "
            f"{r['decisiveness']:>12.4f}"
        )
    return "\n".join(lines)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prices = load_prices(args)
    print(f"Sweeping {args.ticker} h={args.horizon} "
          f"({len(DECAY_RATES)}×{len(THRESHOLD_MULTIPLIERS)} = "
          f"{len(DECAY_RATES)*len(THRESHOLD_MULTIPLIERS)} combos)\n")
    top = run_sweep(prices, args.horizon, args.warmup, args.stride, args.top)
    print(format_table(top))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
