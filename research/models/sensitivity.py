"""Sobol sensitivity analysis mirroring xiphos Saltelli/Sobol sampling.

Mirrors the xiphos Markov design-variable order and uses ``scipy.stats.qmc.Sobol``
to build Saltelli matrices for first-order (S1) and total-order (ST) indices.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
from scipy.stats import qmc

MARKOV_DESIGN_VARIABLES: tuple[str, ...] = (
    "warmup",
    "decay_rate",
    "return_threshold_multiplier",
    "break_divergence_threshold",
    "abstention_threshold",
    "student_t_nu",
    "break_ci_slope",
    "break_ci_max",
    "garch_horizon_cap",
    "garch_calm_ceiling",
    "garch_turbulent_ceiling",
    "entropy_window_size",
    "hmm_num_states",
    "meta_vol_window",
    "meta_high_vol_threshold",
    "volume_lookback",
    "volume_threshold_multiplier",
)

INTEGER_MARKOV_DESIGN_VARIABLES: frozenset[str] = frozenset(
    {
        "warmup",
        "student_t_nu",
        "garch_horizon_cap",
        "entropy_window_size",
        "hmm_num_states",
        "meta_vol_window",
        "volume_lookback",
    }
)

GARCH_CEILING_MIN_SPREAD = 0.1


def _coerce_bounds(
    bounds: Mapping[str, tuple[float, float]] | Sequence[tuple[float, float]] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    if isinstance(bounds, Mapping):
        names = list(bounds)
        arr = np.array([bounds[name] for name in names], dtype=float)
    else:
        names = None
        arr = np.asarray(bounds, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("bounds must be a mapping of name -> (lb, ub) or an array shaped (d, 2)")
    return arr[:, 0], arr[:, 1], names


def _integer_mask(names: list[str] | None, d: int, integer_variables: frozenset[str] | Sequence[bool]) -> np.ndarray:
    if names is not None:
        integer_names = set(integer_variables)
        return np.array([name in integer_names for name in names], dtype=bool)

    mask = np.asarray(integer_variables, dtype=bool)
    if mask.shape == (0,):
        return np.zeros(d, dtype=bool)
    if mask.shape != (d,):
        raise ValueError(f"integer_variables must have shape ({d},)")
    return mask


def _sobol_base(n_base: int, d: int, seed: int | None) -> tuple[np.ndarray, np.ndarray]:
    if n_base <= 0:
        empty = np.empty((0, d), dtype=float)
        return empty, empty

    sampler = qmc.Sobol(d=2 * d, scramble=True, seed=seed)
    if n_base & (n_base - 1) == 0:
        unit = sampler.random_base2(int(np.log2(n_base)))
    else:
        unit = sampler.random(n_base)
    return unit[:, :d], unit[:, d:]


def _unit_to_parameters(
    unit: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    mask: np.ndarray,
    names: list[str] | None,
) -> np.ndarray:
    samples = lb + unit * (ub - lb)

    if names is not None and "garch_calm_ceiling" in names and "garch_turbulent_ceiling" in names:
        calm_idx = names.index("garch_calm_ceiling")
        turbulent_idx = names.index("garch_turbulent_ceiling")
        calm = samples[:, calm_idx]
        turbulent_ub = ub[turbulent_idx]
        turbulent_lb = lb[turbulent_idx]
        spread_room = np.maximum(turbulent_ub - (calm + GARCH_CEILING_MIN_SPREAD), 0.0)
        samples[:, turbulent_idx] = calm + GARCH_CEILING_MIN_SPREAD + unit[:, turbulent_idx] * spread_room
        samples[:, turbulent_idx] = np.maximum(samples[:, turbulent_idx], turbulent_lb)
        samples[:, turbulent_idx] = np.minimum(samples[:, turbulent_idx], turbulent_ub)

    samples[:, mask] = np.round(samples[:, mask])
    return samples


def saltelli_samples(
    n_base: int,
    bounds: Mapping[str, tuple[float, float]] | Sequence[tuple[float, float]] | np.ndarray,
    integer_variables: frozenset[str] | Sequence[bool] = frozenset(),
    seed: int | None = None,
) -> np.ndarray:
    """Build Saltelli samples in ``A, B, A_B^(0), ..., A_B^(D-1)`` order."""
    lb, ub, names = _coerce_bounds(bounds)
    d = len(lb)
    mask = _integer_mask(names, d, integer_variables)
    a_unit, b_unit = _sobol_base(n_base, d, seed)
    blocks = [a_unit, b_unit]
    for i in range(d):
        ab_i = a_unit.copy()
        ab_i[:, i] = b_unit[:, i]
        blocks.append(ab_i)
    return _unit_to_parameters(np.vstack(blocks), lb, ub, mask, names)


def sobol_indices(y: np.ndarray, n_base: int, d: int) -> dict[str, np.ndarray]:
    """Estimate first-order S1 and total-order ST indices for one scalar output."""
    values = np.asarray(y, dtype=float)
    expected = n_base * (d + 2)
    if values.shape != (expected,):
        raise ValueError(f"y must have shape ({expected},)")
    if n_base < 2:
        return {"S1": np.zeros(d, dtype=float), "ST": np.zeros(d, dtype=float)}

    a = values[:n_base]
    b = values[n_base : 2 * n_base]
    variance = np.var(np.concatenate([a, b]), ddof=1)
    if variance <= 0.0 or not np.isfinite(variance):
        return {"S1": np.zeros(d, dtype=float), "ST": np.zeros(d, dtype=float)}

    s1 = np.empty(d, dtype=float)
    st = np.empty(d, dtype=float)
    for i in range(d):
        ab_i = values[(2 + i) * n_base : (3 + i) * n_base]
        s1[i] = np.mean(b * (ab_i - a)) / variance
        st[i] = np.mean((a - ab_i) ** 2) / (2.0 * variance)
    return {"S1": s1, "ST": st}


def sobol_sensitivity(
    objective: Callable[[np.ndarray], float],
    bounds: Mapping[str, tuple[float, float]] | Sequence[tuple[float, float]] | np.ndarray,
    n_base: int,
    seed: int | None = None,
    integer_variables: frozenset[str] | Sequence[bool] = frozenset(),
) -> dict[str, np.ndarray]:
    """Evaluate ``objective`` over Saltelli/Sobol samples and return S1/ST indices."""
    samples = saltelli_samples(n_base, bounds, integer_variables=integer_variables, seed=seed)
    d = samples.shape[1]
    outputs = np.array([float(objective(row)) for row in samples], dtype=float)
    return sobol_indices(outputs, n_base=n_base, d=d)
