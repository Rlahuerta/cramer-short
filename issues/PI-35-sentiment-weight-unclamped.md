# PI-35: Unclamped engagement weight makes the sentiment score NaN or out of range

- Severity: M
- Area: C6a
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/social-sentiment.ts:120` — `const w = 1 + (p.score / maxScore);`
- Repro kind: runtime

## Summary

`aggregateStats()` weights each post by its engagement: `w = 1 + score / maxScore`. Reddit's `ups` can be negative, so the weight can reach exactly 0 (making `weightSum` 0 → `0 / 0` → `avgScore = NaN`) or go negative (inverting the post's contribution and pushing `avgScore` outside −100..100). `formatSentimentReport()` then prints the raw score and `sentimentEmoji()` maps `NaN` — which fails every `>=` comparison — to "🐻 Very Bearish".

## Root cause

`maxScore = Math.max(1, ...posts.map(p => p.score))` normalises positive scores, but the weight formula is never clamped at 0. For a single post with `score = -1`, `w = 1 + (-1 / 1) = 0`, so `weightedSum = 0` and `weightSum = 0`; `Math.round((0 / 0) * 100)` is `NaN`. With `score < -maxScore` the weight is negative, so a bearish post with a heavily negative score subtracts a negative contribution and the resulting score can exceed 100.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-35-sentiment-weight-unclamped.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-35-sentiment-weight-unclamped.ts
```

Expected: avgScore finite and within -100..100 for both post sets

Actual: avgScore = NaN (single post) and 300 (mixed extremes)

OBSERVED:
    single downvoted bullish post: avgScore = NaN, emoji = 🐻 Very Bearish
    [+100 bullish, -200 bearish]: avgScore = 300, emoji = 🐂 Very Bullish

## Impact

The social-sentiment tool's headline score and qualitative verdict can be NaN or out of range whenever the fetched posts include downvoted items. The report then claims a strong directional verdict ("Very Bearish" / "Very Bullish") that the underlying posts do not support, and the same score feeds the ensemble's sentiment input.

## Suggested fix direction (not applied)

Clamp the per-post weight to a non-negative value so `weightSum` is at least the post count: `src/tools/finance/social-sentiment.ts:120` — `const w = 1 + (p.score / maxScore);` → `const w = 1 + Math.max(0, p.score) / maxScore;`. (Ignoring negative engagement is the minimal change; clamping the final `avgScore` to −100..100 is an equivalent alternative.) Verified green-flip: with this change the repro exits 0.

## Evidence

- Reproduction script: `issues/repro/PI-35-sentiment-weight-unclamped.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-11-C6a.txt`.
- Green-flip: applied the one-line weight clamp to a scratch copy of `src/`; the same script exits 0.
