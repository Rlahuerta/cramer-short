"""Walk-forward reproduction harness for the TypeScript Markov forecast tool.

This module is intentionally a *script-shaped* research artifact: it can be
run from the repository root either as a module,

    python -m research.backtest.markov_tool_backtest --ticker BTC --csv prices.csv

or as a direct script,

    python research/backtest/markov_tool_backtest.py --ticker BTC --days 365

The goal is to reproduce the TypeScript Markov distribution tool with the
existing Python mirror functions rather than reimplementing its mathematics.
The script therefore delegates model work to ``research.models.markov`` and
``research.backtest.walk_forward`` and focuses on orchestration, reporting, and
documentation.

Theory of the pipeline
----------------------
1. Prices -> returns
   Closing prices are converted to one-period fractional returns. A Markov
   chain needs a discrete state sequence, so the continuous return series is
   reduced to a small set of regimes.

2. Returns -> regimes
   ``classify_regime_series`` maps each return to ``bull``, ``bear``, or
   ``sideways`` using an adaptive threshold based on median absolute return.
   This keeps the states comparable across assets with very different daily
   volatility.

3. Regimes -> transition matrix
   ``estimate_transition_matrix`` counts observed regime-to-regime moves. It
   applies a Dirichlet prior (smoothing) so unseen transitions do not become
   impossible, and exponential decay so recent regime moves carry more weight
   than stale history. The result is a row-stochastic matrix P where P[i, j] is
   the probability of moving from state i to state j in one step.

4. Transition matrix -> horizon probabilities
   ``compute_markov_forecast`` raises the transition matrix to a horizon power:
   P^h gives h-step transition probabilities. The current regime selects the
   relevant row, yielding forecast probabilities for bull/bear/sideways at the
   requested horizon.

5. Structural break detection
   ``detect_structural_break`` estimates separate transition matrices for the
   first and second halves of a window and computes their squared Frobenius
   divergence. A large divergence means the transition dynamics may have
   changed; the TypeScript tool responds by widening confidence intervals and,
   for BTC short horizons, optionally rerunning on a shorter recent window.

6. Trajectory and interval scoring
   The walk-forward harness converts regime probabilities and regime-specific
   return statistics into a price trajectory with an expected terminal price
   and confidence interval. Each historical step is scored by comparing that
   interval and directional probability to the realised future price.

7. Walk-forward evaluation metrics
   Brier score measures probability calibration for up/down outcomes; lower is
   better. Directional accuracy is the hit rate of the predicted direction.
   CI coverage is the fraction of realised prices inside the forecast interval.
   CRPS approximates full-distribution quality from the interval-implied normal
   distribution; lower is better. Murphy-Winkler decomposes interval quality
   into width and miss penalties.

Offline reproducibility
-----------------------
Use ``--csv`` with a ``close`` column to avoid network/API dependencies. The
``--seed`` argument fixes the Monte Carlo path sampling used by the mirrored
trajectory code, making repeated offline runs reproducible.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Sequence

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from research.data.prices import fetch_historical_prices
from research.models.markov import get_gold_short_horizon_live_policy
from research.backtest.metrics import (
    brier_score,
    calibration_table,
    ci_coverage,
    crps,
    mean_absolute_error,
    murphy_winkler_decomposition,
    scaled_crps,
)
from research.backtest.walk_forward import BacktestStep, WalkForwardResult, walk_forward

# ---------------------------------------------------------------------------
# TS-style directional accuracy (reproduces TypeScript metrics.ts)
# ---------------------------------------------------------------------------

TS_HOLD_THRESHOLD = 0.03
DEFAULT_HORIZONS = "1,7,14,30"
DEFAULT_SOURCES = "financial_datasets,binance,yahoo"

def _ts_action_thresholds(horizon: int) -> tuple[float, float]:
    """Return (buy_threshold, sell_threshold) matching TS computeActionSignal fallback."""
    if horizon <= 7:
        return 0.003, 0.002
    if horizon <= 30:
        return 0.005, 0.003
    return 0.008, 0.005


def ts_directional_accuracy(steps: list[BacktestStep], horizon: int) -> float:
    """Reproduce TypeScript directionalAccuracy with recommendation + HOLD zone.

    Mirrors ``metrics.ts::directionalAccuracy`` with the recommendation derivation
    from ``markov-distribution.ts::computeActionSignal`` fallback thresholds:
    - BUY   = predicted_return >  buy_threshold (horizon-dependent, 0.3 %–0.8 %)
    - SELL  = predicted_return < -sell_threshold (horizon-dependent, 0.2 %–0.5 %)
    - HOLD  = otherwise
    Scoring:
    - BUY  correct → actualReturn > 0
    - SELL correct → actualReturn < 0
    - HOLD correct → |actualReturn| < 0.03
    """
    if not steps:
        return 0.0
    buy_thr, sell_thr = _ts_action_thresholds(horizon)
    correct = 0
    for s in steps:
        if s.predicted_return > buy_thr:
            correct += 1 if s.realised_return > 0 else 0
        elif s.predicted_return < -sell_thr:
            correct += 1 if s.realised_return < 0 else 0
        else:  # HOLD
            correct += 1 if abs(s.realised_return) < TS_HOLD_THRESHOLD else 0
    return correct / len(steps)


def parse_horizons(raw: str) -> list[int]:
    """Parse CLI horizon text into positive integer forecast horizons."""
    horizons: list[int] = []
    for piece in raw.split(","):
        stripped = piece.strip()
        if not stripped:
            continue
        horizon = int(stripped)
        if horizon < 1:
            raise argparse.ArgumentTypeError("horizons must be positive integers")
        horizons.append(horizon)
    if not horizons:
        raise argparse.ArgumentTypeError("at least one horizon is required")
    return horizons


def parse_sources(raw: str | None) -> list[str] | None:
    """Parse an optional comma-separated live price source fallback list."""
    if raw is None:
        return None
    sources = [piece.strip() for piece in raw.split(",") if piece.strip()]
    return sources or None


def parse_garch_regime_ceiling(raw: str | None) -> tuple[float, float] | None:
    """Parse ``LOW,HIGH`` GARCH ceiling text into the tuple expected by walk_forward."""
    if raw is None:
        return None
    pieces = [piece.strip() for piece in raw.split(",") if piece.strip()]
    if len(pieces) != 2:
        raise argparse.ArgumentTypeError("--garch-regime-ceiling must be LOW,HIGH")
    low, high = (float(pieces[0]), float(pieces[1]))
    if low <= 0 or high <= 0:
        raise argparse.ArgumentTypeError("GARCH regime ceilings must be positive")
    return low, high


def positive_int(value: str) -> int:
    """Argparse helper for strictly positive integer options."""
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def non_negative_int(value: str) -> int:
    """Argparse helper for integer options where zero has a documented meaning."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def load_prices_from_csv(path: Path, close_column: str, days: int) -> list[float]:
    """Load oldest-first close prices from a CSV file for offline reproduction.

    The model only needs close prices. Date columns are intentionally ignored so
    researchers can use exports from different vendors, provided the rows are
    already ordered oldest-to-newest.
    """
    if not path.exists():
        raise FileNotFoundError(f"CSV file not found: {path}")

    frame = pd.read_csv(path)
    column = close_column
    if column not in frame.columns:
        matches = [name for name in frame.columns if str(name).lower() == close_column.lower()]
        if not matches:
            raise ValueError(
                f"CSV must contain a '{close_column}' column. "
                f"Available columns: {', '.join(map(str, frame.columns))}"
            )
        column = str(matches[0])

    closes = pd.to_numeric(frame[column], errors="coerce").dropna()
    prices = [float(value) for value in closes if math.isfinite(float(value)) and value > 0]
    if days > 0:
        prices = prices[-days:]
    if len(prices) < 10:
        raise ValueError(f"Need at least 10 positive close prices; found {len(prices)}")
    return prices


def load_prices(args: argparse.Namespace) -> tuple[list[float], str]:
    """Load prices from CSV when supplied, otherwise fetch live data via mirrored fallback."""
    if args.csv:
        csv_path = Path(args.csv).expanduser()
        return load_prices_from_csv(csv_path, args.close_column, args.days), f"csv:{csv_path}"

    if args.days < 1:
        raise ValueError("--days must be positive when fetching live prices")

    sources = parse_sources(args.sources)
    frame = fetch_historical_prices(args.ticker, days=args.days, sources=sources)
    if "close" not in frame.columns:
        raise ValueError("fetch_historical_prices returned no close column")
    closes = pd.to_numeric(frame["close"], errors="coerce").dropna()
    prices = [float(value) for value in closes if math.isfinite(float(value)) and value > 0]
    if len(prices) < 10:
        raise ValueError(f"Need at least 10 fetched prices; found {len(prices)}")
    source = str(frame.attrs.get("source") or ",".join(sources or DEFAULT_SOURCES.split(",")))
    return prices, f"live:{source}"


def effective_walk_forward_options(
    args: argparse.Namespace,
    horizon: int,
) -> dict[str, Any]:
    """Resolve CLI options into the exact argument bundle passed to walk_forward.

    BTC live policy is implemented inside ``walk_forward``. GLD short-horizon
    policy is mirrored here because the existing harness exposes the gold
    helper but not a dedicated ``use_live_gold_short_horizon_policy`` argument.
    """
    warmup = args.warmup
    break_threshold = args.break_divergence_threshold

    gold_policy = (
        get_gold_short_horizon_live_policy(args.ticker, horizon)
        if args.live_policy
        else None
    )
    if gold_policy is not None:
        warmup = gold_policy.history_days
        break_threshold = gold_policy.break_divergence_threshold

    return {
        "horizon": horizon,
        "warmup": warmup,
        "stride": args.stride,
        "ticker": args.ticker,
        "return_threshold_multiplier": args.return_threshold_multiplier,
        "decay_rate": args.decay_rate,
        "break_divergence_threshold": break_threshold,
        "btc_break_divergence_threshold": args.btc_break_divergence_threshold,
        "use_live_btc_short_horizon_policy": args.live_policy,
        "use_hmm": args.use_hmm,
        "asset_profile": args.asset_profile,
        "enable_garch_vol": args.enable_garch_vol,
        "garch_horizon_cap": args.garch_horizon_cap,
        "garch_regime_ceiling": args.garch_regime_ceiling,
        "enable_entropy_ci_modulation": args.enable_entropy_ci_modulation,
        "entropy_window_size": args.entropy_window_size,
        "entropy_kappa": args.entropy_kappa,
    }


def summarize_steps(steps: list[BacktestStep], horizon: int) -> dict[str, Any]:
    murphy_winkler = murphy_winkler_decomposition(steps)
    return {
        "steps": len(steps),
        "brier_score": brier_score(steps),
        "directional_accuracy": ts_directional_accuracy(steps, horizon),
        "ci_coverage": ci_coverage(steps),
        "mean_absolute_error": mean_absolute_error(steps),
        "crps": crps(steps),
        "scaled_crps": scaled_crps(steps),
        "murphy_winkler": dict(murphy_winkler),
        "structural_break_steps": sum(1 for step in steps if step.structural_break_detected),
        "structural_break_reruns": sum(
            1 for step in steps if step.structural_break_rerun_triggered
        ),
        "garch_applied_steps": sum(1 for step in steps if step.garch_vol_applied),
        "entropy_modulated_steps": sum(1 for step in steps if step.entropy_ci_modulation_applied),
        "calibration": calibration_table(steps, bins=5),
    }


def step_to_json(step: BacktestStep) -> dict[str, Any]:
    """Convert one walk-forward row into a compact, JSON-serialisable record."""
    return {
        "start_idx": step.start_idx,
        "predicted_prob": step.predicted_prob,
        "predicted_return": step.predicted_return,
        "ci_lower": step.ci_lower,
        "ci_upper": step.ci_upper,
        "realised_return": step.realised_return,
        "realised_price": step.realised_price,
        "direction_correct": step.direction_correct,
        "in_ci": step.in_ci,
        "structural_break_detected": step.structural_break_detected,
        "structural_break_rerun_triggered": step.structural_break_rerun_triggered,
        "transition_entropy_norm": step.transition_entropy_norm,
        "entropy_ci_scale": step.entropy_ci_scale,
        "garch_vol_applied": step.garch_vol_applied,
    }


def run_horizon(
    prices: list[float],
    args: argparse.Namespace,
    horizon: int,
) -> tuple[WalkForwardResult, dict[str, Any], dict[str, Any]]:
    """Run one horizon with a fixed seed so stochastic trajectory CIs are reproducible."""
    if args.seed is not None:
        np.random.seed(args.seed + horizon)
    options = effective_walk_forward_options(args, horizon)
    result = walk_forward(prices, **options)
    metrics = summarize_steps(result.steps, horizon)
    return result, metrics, options


def format_percent(value: float) -> str:
    """Render a fraction as a compact percentage for terminal tables."""
    return f"{100.0 * value:5.1f}%"


def format_float(value: float) -> str:
    """Render numeric metrics compactly while preserving enough precision."""
    return f"{value:8.4f}"


def format_report(
    ticker: str,
    price_source: str,
    price_count: int,
    rows: list[dict[str, Any]],
) -> str:
    """Create a dependency-free terminal table for all requested horizons."""
    lines = [
        f"Markov tool reproduction backtest for {ticker}",
        f"Prices: {price_count} closes from {price_source}",
        "",
        "horizon  steps  dir_acc  brier     coverage  mae       crps      s_crps   breaks  errors",
        "-------  -----  -------  --------  --------  --------  --------  -------  ------  ------",
    ]

    for row in rows:
        metrics = row["metrics"]
        lines.append(
            f"{row['horizon']:>7}  "
            f"{metrics['steps']:>5}  "
            f"{format_percent(metrics['directional_accuracy']):>7}  "
            f"{format_float(metrics['brier_score'])}  "
            f"{format_percent(metrics['ci_coverage']):>8}  "
            f"{format_float(metrics['mean_absolute_error'])}  "
            f"{format_float(metrics['crps'])}  "
            f"{format_float(metrics['scaled_crps'])}  "
            f"{metrics['structural_break_steps']:>6}  "
            f"{len(row['errors']):>6}"
        )

    return "\n".join(lines)


def write_json_payload(path: Path, payload: dict[str, Any]) -> None:
    """Write optional machine-readable results for notebooks or downstream analysis."""
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI parser with TS-like model toggles exposed as explicit flags."""
    parser = argparse.ArgumentParser(
        description="Backtest the Python mirror of the TypeScript Markov forecast tool.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ticker",
        default="BTC",
        help="Ticker label used for live fetch and policies.",
    )
    parser.add_argument(
        "--horizons",
        type=parse_horizons,
        default=parse_horizons(DEFAULT_HORIZONS),
        help="Comma-separated forecast horizons in days.",
    )
    parser.add_argument(
        "--days",
        type=non_negative_int,
        default=365,
        help="Live fetch lookback, or CSV tail length. Use 0 with --csv to keep all rows.",
    )
    parser.add_argument(
        "--warmup",
        type=positive_int,
        default=120,
        help="Training window length.",
    )
    parser.add_argument(
        "--stride",
        type=positive_int,
        default=10,
        help="Days between test windows.",
    )
    parser.add_argument("--csv", help="Offline CSV input with a close column.")
    parser.add_argument("--close-column", default="close", help="CSV close column name.")
    parser.add_argument(
        "--sources",
        default=DEFAULT_SOURCES,
        help="Comma-separated live source fallback list.",
    )
    parser.add_argument("--output-json", help="Optional path for JSON metrics and sample rows.")
    parser.add_argument(
        "--sample-rows",
        type=non_negative_int,
        default=5,
        help="Number of walk-forward rows to include per horizon in JSON.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Monte Carlo seed for reproducible trajectory sampling.",
    )
    parser.add_argument(
        "--return-threshold-multiplier",
        type=float,
        default=0.5,
        help="Adaptive regime threshold multiplier.",
    )
    parser.add_argument(
        "--decay-rate",
        type=float,
        default=0.97,
        help="Exponential transition-count decay rate.",
    )
    parser.add_argument(
        "--break-divergence-threshold",
        type=float,
        default=0.05,
        help="Structural break matrix-divergence threshold.",
    )
    parser.add_argument(
        "--btc-break-divergence-threshold",
        type=float,
        default=None,
        help="Optional BTC live-policy break threshold override.",
    )
    parser.add_argument(
        "--live-policy",
        action="store_true",
        help="Apply mirrored BTC/GLD short-horizon live-policy settings when available.",
    )
    parser.add_argument(
        "--use-hmm",
        action="store_true",
        help="Blend the optional HMM forecast path.",
    )
    parser.add_argument(
        "--asset-profile",
        choices=["crypto", "equity", "etf", "commodity"],
        default="crypto",
        help="HMM profile used when --use-hmm is enabled.",
    )
    parser.add_argument(
        "--enable-garch-vol",
        action="store_true",
        help="Apply GARCH volatility scaling in the trajectory interval.",
    )
    parser.add_argument(
        "--garch-horizon-cap",
        type=positive_int,
        default=None,
        help="Optional GARCH scale horizon cap.",
    )
    parser.add_argument(
        "--garch-regime-ceiling",
        type=parse_garch_regime_ceiling,
        default=None,
        help="Optional GARCH clamp ceiling as LOW,HIGH.",
    )
    parser.add_argument(
        "--enable-entropy-ci-modulation",
        action="store_true",
        help="Scale confidence intervals by transition-entropy z-score.",
    )
    parser.add_argument(
        "--entropy-window-size",
        type=positive_int,
        default=60,
        help="Rolling entropy window for CI modulation.",
    )
    parser.add_argument(
        "--entropy-kappa",
        type=float,
        default=0.15,
        help="Entropy-to-CI scale sensitivity.",
    )
    return parser


def build_payload(
    args: argparse.Namespace,
    price_source: str,
    price_count: int,
    horizon_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Assemble stable JSON output without serialising argparse internals directly."""
    return {
        "ticker": args.ticker,
        "price_source": price_source,
        "price_count": price_count,
        "options": {
            "days": args.days,
            "warmup": args.warmup,
            "stride": args.stride,
            "return_threshold_multiplier": args.return_threshold_multiplier,
            "decay_rate": args.decay_rate,
            "break_divergence_threshold": args.break_divergence_threshold,
            "live_policy": args.live_policy,
            "use_hmm": args.use_hmm,
            "asset_profile": args.asset_profile,
            "enable_garch_vol": args.enable_garch_vol,
            "enable_entropy_ci_modulation": args.enable_entropy_ci_modulation,
            "seed": args.seed,
        },
        "horizons": horizon_rows,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: load prices, run horizons, print a report, optionally write JSON."""
    parser = build_parser()
    args = parser.parse_args(argv)

    prices, price_source = load_prices(args)
    horizon_rows: list[dict[str, Any]] = []

    for horizon in args.horizons:
        result, metrics, options = run_horizon(prices, args, horizon)
        horizon_rows.append(
            {
                "horizon": horizon,
                "metrics": metrics,
                "errors": result.errors,
                "effective_options": options,
                "sample_rows": [
                    step_to_json(step) for step in result.steps[: args.sample_rows]
                ],
            }
        )

    print(format_report(args.ticker, price_source, len(prices), horizon_rows))

    all_errors = [error for row in horizon_rows for error in row["errors"]]
    if all_errors:
        print("\nWarnings/errors surfaced by walk_forward:", file=sys.stderr)
        for error in all_errors[:10]:
            print(f"- {error}", file=sys.stderr)
        if len(all_errors) > 10:
            print(f"- ... {len(all_errors) - 10} more", file=sys.stderr)

    if args.output_json:
        payload = build_payload(args, price_source, len(prices), horizon_rows)
        write_json_payload(Path(args.output_json), payload)
        print(f"\nJSON written to {args.output_json}")

    return 0 if any(row["metrics"]["steps"] > 0 for row in horizon_rows) else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
