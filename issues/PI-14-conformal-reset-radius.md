# PI-14: Conformal reset keeps the adapted radius instead of restoring the initial radius

- Severity: M
- Area: C1b
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/conformal.ts:195` — `if (opts.radius !== undefined) {`, `src/tools/finance/conformal.ts:148` — `this.q = Math.max(0, opts.initialRadius ?? 1.0);`, `research/models/conformal.py:148` — `def reset(self, radius: Optional[float] = None) -> None:`
- Repro kind: runtime

## Summary

`ConformalPID.reset()` is documented as resetting the controller when the wrapped forecaster restarts, but when called without a `radius` option it leaves the adapted interval radius unchanged. The Python mirror restores the constructor's `initialRadius` on `reset()` and has a parity test asserting that behaviour (`research/tests/test_conformal_parity.py:244`), so the TS class silently diverges: a restarted forecaster keeps the stale, drift-adapted interval width.

## Root cause

The TS constructor never stores `opts.initialRadius`; it only writes the clamped value into `this.q` (`conformal.ts:148`). `reset()` therefore cannot restore it: the only branch that touches `this.q` is guarded by `opts.radius !== undefined` (`conformal.ts:195-197`), so the default `reset()` clears `integral`, `prevBias`, `samples`, and `hits` but keeps the last adapted `q`. The Python mirror stores `_initial_radius` in `__init__` and assigns `self._q = self._initial_radius if radius is None else max(0.0, float(radius))` in `reset()`.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-14-conformal-reset-radius.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-14-conformal-reset-radius.ts
```

Expected: radiusAfterReset=1

Actual: radiusAfterReset=1.054

OBSERVED:
    initialRadius=1
    radiusAfterMiss=1.054
    radiusAfterReset=1.054

## Impact

Any caller that resets a conformal controller after a forecaster restart without passing an explicit radius keeps the pre-restart interval width, so post-restart coverage/sharpness is measured against stale state. The current walk-forward caller passes `{ radius: currentRadius() }` explicitly (`src/tools/finance/backtest/walk-forward.ts:370`), which masks the bug there; the divergence remains reachable through the public `reset()` API and through `AdaptiveConformalPID.reset()` (which inherits the behaviour). It is also a TS↔Python parity gap in a file marked as a mirror of `research/models/conformal.py`.

## Suggested fix direction (not applied)

Store the initial radius and use it as the default reset target in `src/tools/finance/conformal.ts`:

- `src/tools/finance/conformal.ts:138` — after `protected q: number;` add `protected readonly initialRadius: number;`
- `src/tools/finance/conformal.ts:148` — replace `this.q = Math.max(0, opts.initialRadius ?? 1.0);` with:
  ```ts
  this.initialRadius = Math.max(0, opts.initialRadius ?? 1.0);
  this.q = this.initialRadius;
  ```
- `src/tools/finance/conformal.ts:195-197` — replace the guarded assignment with:
  ```ts
  this.q = Math.max(0, opts.radius ?? this.initialRadius);
  ```

This matches `ConformalPID.reset()` in `research/models/conformal.py:148-155` (`radius is None ⇒ _initial_radius`) and keeps the explicit-radius call site (`walk-forward.ts:370`) unchanged.

## Evidence

- Script: `issues/repro/PI-14-conformal-reset-radius.ts`
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-3-C1b.txt`
- Python reference: `research/models/conformal.py:148-155`; assertion: `research/tests/test_conformal_parity.py:244-251`
