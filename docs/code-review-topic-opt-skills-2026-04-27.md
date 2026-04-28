# Code Review: `topic/opt-skills` Branch

**Date:** 2026-04-27  
**Scope:** Deep review of committed branch changes plus current unstaged edits  
**Reviewed areas:** HMM regime-detection work, walk-forward backtest integration, and the current YES-bias correction changes

---

## Summary

This review covered:

1. **Committed changes** adding a Gaussian HMM implementation and integrating it into the walk-forward backtest flow
2. **Current unstaged changes** changing the YES-bias correction behavior in `research/models/ensemble.py` and its related tests

Files called out by the review:

- `research/models/hmm.py`
- `research/backtest/walk_forward.py`
- `research/models/trajectory.py`
- `research/models/ensemble.py`
- `research/utils/calibration.py`
- `research/tests/test_hmm_parity.py`
- `research/tests/test_backtest_parity.py`
- `research/tests/test_ensemble_parity.py`

The review found **2 high-severity issues**, **2 medium-severity issues**, and **2 low-severity issues**.

---

## High severity

### 1. Incorrect confidence-interval math for price forecasts

**Affected file:** `research/backtest/walk_forward.py:160-161`

The backtest computes confidence bounds with a linear approximation:

```python
ci_lower = current_price * (1 - 1.96 * sigma)
ci_upper = current_price * (1 + 1.96 * sigma)
```

That treats uncertainty as if price itself were normally distributed. For return-driven forecasts the price distribution is log-normal, so the interval should be computed in exponent space using `mu_n` and `sigma_n`.

**Why it matters:**  
This makes the reported confidence interval coverage (`in_ci`) statistically wrong, especially in high-volatility scenarios. It can materially understate downside risk and can even produce impossible negative lower bounds.

**Recommendation:**  
Use log-normal bounds instead:

```python
ci_lower = current_price * math.exp(mu_n - 1.96 * sigma_n)
ci_upper = current_price * math.exp(mu_n + 1.96 * sigma_n)
```

---

### 2. Two conflicting `adjust_yes_bias()` implementations exist

**Affected files:**

- `research/models/ensemble.py:77`
- `research/utils/calibration.py:12`

There are now two exported functions named `adjust_yes_bias()` with different semantics:

- one is multiplicative and applies to all probabilities
- the other is additive and applies only above `0.5`

Because both remain available, future callers can silently use different algorithms depending on import path.

**Why it matters:**  
This creates inconsistent behavior across the codebase, makes later maintenance error-prone, and leaves the intended calibration algorithm ambiguous.

**Recommendation:**  
Choose one canonical implementation, remove or deprecate the other, and update callers plus docstrings to match the intended algorithm change.

---

## Medium severity

### 3. HMM non-convergence falls back silently

**Affected file:** `research/backtest/walk_forward.py:141`

When `use_hmm=True` but the HMM fit does not converge, the code appears to continue with the non-HMM path without surfacing that fallback to the user.

**Why it matters:**  
A caller can request HMM-enhanced behavior and receive plain Markov behavior without any signal, which makes results hard to trust and harder to debug.

**Recommendation:**  
Emit a warning, attach a structured note to the backtest output, or fail explicitly depending on the intended UX.

---

### 4. `fit_2state_return_hmm()` rebuilds a model after fitting

**Affected file:** `research/models/hmm.py:328-335`

After fitting parameters, `fit_2state_return_hmm()` reconstructs a `GaussianHMM` just to call `predict_proba()` on the same data.

**Why it matters:**  
This repeats work, adds dependence on internal model reconstruction details, and increases the chance of subtle ordering/permutation mistakes after state reordering.

**Recommendation:**  
Prefer returning or caching posterior probabilities from the fitting step directly, or document clearly why rebuilding is required and safe here.

---

## Low severity

### 5. Backtest step construction does repeated scalar coercion

**Affected file:** `research/backtest/walk_forward.py:171-178`

`BacktestStep(...)` wraps many fields in `float(...)` and `bool(...)` at construction time.

**Why it matters:**  
This is low risk, but it suggests upstream numpy/Python scalar mixing that may be better handled at the boundary rather than throughout the code path.

**Recommendation:**  
Either normalize values once before constructing the dataclass or add a short comment explaining that the coercion is intentional for serialization/type stability.

---

### 6. Volatility HMM baseline is based on a simple midpoint

**Affected file:** `research/models/hmm.py:392-395`

The volatility scaling baseline is the midpoint of the two state means.

**Why it matters:**  
This may be noisier than necessary when the states are asymmetric. It is not obviously wrong, but it is a somewhat arbitrary calibration choice.

**Recommendation:**  
Consider a stationary-probability-weighted baseline instead, or document why midpoint is the intended behavior.

---

## Residual risk areas to spot-check

Even after the issues above are addressed, these areas are worth validating:

1. **State reordering logic** in `research/models/hmm.py` after sorting by volatility or mean
2. **Transition-matrix permutation correctness** when applying reordered state indices
3. **End-to-end forecast quality** to confirm the HMM blending improves outputs rather than only matching parity/unit expectations

---

## Overall assessment

The new HMM work appears well-tested and thoughtfully structured, but the current review found a small number of substantive correctness and maintainability problems. The most urgent fixes are the confidence-interval formula in the backtest and the duplicate YES-bias calibration implementations.
