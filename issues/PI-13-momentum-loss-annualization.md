# PI-13: Momentum velocity annualizes losses as (1+|r|), yielding returns below -100%

- Severity: M
- Area: C1a
- Status: Resolved (docs only, by decision)
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/markov-distribution/regime.ts:94` — `const velocity = Math.sign(totalReturn) * (Math.pow(1 + Math.abs(totalReturn), 252 / lookback) - 1);`, `src/tools/finance/markov-distribution/regime.ts:100` — `const recentVel = Math.sign(recentRet) * (Math.pow(1 + Math.abs(recentRet), 252 / half) - 1);`
- Repro kind: runtime

## Resolution

Decision: keep the numbers, fix the description. `velocity` is used only for its sign and the clamped drift `adjustment` (never displayed), and the scale 0.25 / clamp 0.003 were tuned against this symmetric `sign(r) * ((1+|r|)^k - 1)` form. The geometric `(1+r)^(252/L)-1` removes the impossible values below -100% but makes the tilt strongly asymmetric (at the shipped lookback of 10, a -10% move gives -0.0009/day while +10% gives +0.003/day, clamped), which would shift every downtrend forecast. The `regime.ts` comment now describes it as a symmetric compounded magnitude, not an annualized return.

The repro still exits 42 because it asserts the geometric annualization. Alternatives if this is revisited: symmetric log annualization `252*ln(1+r)/L` with scale/clamp re-tuned by backtest.

## Summary

`computeMomentumSignal` documents "Annualized return over lookback window" but annualizes a negative total return `r` as `−((1+|r|)^(252/L) − 1)` instead of `(1+r)^(252/L) − 1`. For a −9.54% 20-day decline it reports −215.2% annualized instead of −71.7% — below −100%, which a long-only price series can never produce — and drives the daily drift adjustment to −0.00213 instead of −0.00072. The same mirrored formula is used for the recent/older half velocities (lines 100–101).

## Root cause

The growth factor uses `1 + Math.abs(totalReturn)` and then restores the sign with `Math.sign`. That is correct only for gains. For losses the correct geometric annualization is `Math.pow(1 + totalReturn, 252 / lookback) - 1` (which is already negative and cannot go below −100% because the base is positive). The mirrored form compounds the absolute loss as if it were a gain, so it overstates the magnitude by roughly 3x at typical daily returns and can exceed −100% annualized. The overstated velocity is divided by 252 and multiplied by `trendStrength × momentumAdjustmentScale`, so a persistent downtrend saturates the ±0.003/day clamp much earlier than a trend of the same magnitude in the opposite direction.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-13-momentum-loss-annualization.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-13-momentum-loss-annualization.ts
```

Expected: reported velocity equals the geometric annualization (1+r)^(252/lookback)-1 and stays above -100%

Actual: reported velocity=-2.1518397341856463 (-215.2% annualized) vs geometric -0.7172410580445934 (-71.7%)

OBSERVED:
    {
      "totalReturn": -0.09538951972538223,
      "lookback": 20,
      "reportedVelocity": -2.1518397341856463,
      "geometricVelocity": -0.7172410580445934,
      "reportedAdjustmentPerDay": -0.0021347616410571884,
      "trendStrength": 0.9999999999999999
    }

## Impact

`computeMomentumSignal` is called on every live forecast (`markov-distribution.ts:3657-3662`) and its `adjustment` (halved at line 3670 and combined with the ensemble and cross-asset terms) is added to every day's Markov drift. For downtrends the bearish tilt is about 3x too strong and reaches the clamp for ordinary declines, shifting expected return, P(up), scenario buckets and the BUY/HOLD/SELL signal. The asymmetry also biases the model: equally sized up-moves and down-moves produce different adjusted magnitudes.

## Suggested fix direction (not applied)

Replace the mirrored formulas with the geometric annualization:

- `src/tools/finance/markov-distribution/regime.ts:94`: `const velocity = Math.sign(totalReturn) * (Math.pow(1 + Math.abs(totalReturn), 252 / lookback) - 1);` → `const velocity = Math.pow(1 + totalReturn, 252 / lookback) - 1;`
- `src/tools/finance/markov-distribution/regime.ts:100`: `const recentVel = Math.sign(recentRet) * (Math.pow(1 + Math.abs(recentRet), 252 / half) - 1);` → `const recentVel = Math.pow(1 + recentRet, 252 / half) - 1;`
- `src/tools/finance/markov-distribution/regime.ts:101`: `const olderVel  = Math.sign(olderRet) * (Math.pow(1 + Math.abs(olderRet), 252 / half) - 1);` → `const olderVel  = Math.pow(1 + olderRet, 252 / half) - 1;`

Prices are validated positive by `assertPositiveFinitePrices`, so `1 + totalReturn > 0` is guaranteed on the shipped path.

## Evidence

Reproduction script: `issues/repro/PI-13-momentum-loss-annualization.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-2-C1a.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above; the reported −215% annualized return is arithmetically impossible for a price that fell 9.54%.
