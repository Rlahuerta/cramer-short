# Polymarket Snapshot History — What It Does and When It Helps

**Last Updated:** 2026-04-21
**Applies to:** `polymarket_forecast`

---

## Quick Answer

The snapshot-history feature automatically records Polymarket market probabilities fetched through the history-aware forecast path, then uses that history to detect price spikes and transitory moves in later runs. It runs silently in the background. You never invoke it directly.

**Use this when:**
- You run `polymarket_forecast` more than once for the same asset over hours or days
- You want the forecast to automatically discount whale-driven spikes and fleeting moves
- You are monitoring an unfolding thesis and re-checking Polymarket odds
- You want to track short-horizon BTC price markets over time and see when a sudden move looks thin, crowded, or already reversing

**Do NOT use this when:**
- You need deterministic reproduction of spike/transitory warnings from live agent prompts
- This is effectively your first history-bearing run for a market (there may be no usable prior-window history to check yet)
- You expect the snapshot log to replace Polymarket's own historical data (it only stores what this tool fetched, not the full market timeline)

---

## What the Feature Does

| Aspect | Detail |
|--------|--------|
| **Writes on** | Every `polymarket_forecast` call and other internal code paths that use `fetchPolymarketMarkets` |
| **Writes what** | One JSON line per market: `marketId`, `question`, `probability`, `volume24h`, `endDate`, `capturedAt` |
| **Storage path** | `.cramer-short/polymarket-snapshots.jsonl` (JSONL, append-only) |
| **Reads when** | Each `polymarket_forecast` run reads prior snapshots for each market it found |
| **Effect on output** | Flags detected spikes and transitory moves; discounts their quality weight in the ensemble |
| **Does `polymarket_search` use it?** | No. `polymarket_search` is still useful for manual market discovery, but it does not currently read or write snapshot history. |

---

## How Snapshot Writing Works

When `polymarket_forecast` or another internal history-aware fetch path pulls Polymarket markets, it creates a snapshot record for each market that has a `marketId` and appends it to the JSONL file. This happens automatically. There is no opt-in flag.

**Example record:**
```json
{"marketId":"0xabc123","question":"Will BTC be above $100K on June 30?","probability":0.72,"volume24h":150000,"endDate":"2026-06-30","capturedAt":"2026-04-21T14:30:00.000Z"}
```

The file grows over time. Old entries are never deleted or pruned by the system.

---

## How to Actually Use It for BTC Forecasts and Whale Detection

If you want the workflow described in the implementation plan — **Bitcoin price forecast first, whale/spike detection second** — the cleanest way to use the feature is to rerun the **same short-horizon BTC forecast prompt** over time.

### Recommended workflow

| Step | When | Prompt | Why it matters |
|------|------|--------|----------------|
| **1. Baseline run** | Now | `Give me a Polymarket-based forecast for BTC over the next 7 days. Current price is 68000.` | Seeds the snapshot file and gives you the initial BTC forecast. On the first run, expect cold-start warnings. |
| **2. Whale check** | 2–4 hours later | Run the same prompt again | Now the tool can compare the current BTC market probabilities against the 2–4h snapshot and detect rapid, low-volume moves. |
| **3. Persistence check** | 24–48 hours after the baseline | Run the same prompt again | Now the tool can tell whether the earlier move persisted or reversed. |
| **4. Optional market inspection** | Any time | `Search Polymarket for Bitcoin price markets and summarize the top results.` | Lets you manually inspect which BTC threshold markets are active before or between forecast runs. |

### What to expect at each step

1. **Baseline run**
   - Writes the current BTC market probabilities into `.cramer-short/polymarket-snapshots.jsonl`
   - Returns a normal BTC forecast
   - Usually shows cold-start warnings because there is no usable 2–4h or 24–48h history yet

2. **Whale check (2–4h later)**
   - Uses the earlier BTC snapshot as the comparison point
   - If a relevant BTC market moved by more than **8 percentage points** and current `volume24h` is below **$100K**, you get:
     - `priceSpikeDetected = true`
     - warning: `has a price spike (possible whale activity)`
     - **50% quality discount** on that market

3. **Persistence check (24–48h later)**
   - Becomes meaningful only if the tool also has a usable 2–4h snapshot from the earlier step
   - If the original move was larger than **10 percentage points** and has reversed by more than half, you get:
     - `transitoryMove = true`
     - warning: `shows a transitory 24-48h move`
     - **30% quality discount** on that market, unless the stronger whale discount already dominates

### Practical guidance

- **Use short BTC horizons first.** `7d` or `30d` is more useful for this feature than `180d`, because repeated checks are easier to interpret when the underlying BTC threshold markets are short-dated and active.
- **Keep the asset and horizon stable across reruns.** If you switch from BTC 7d to BTC 180d or to another asset, the relevant Polymarket contracts can change and the history becomes less comparable.
- **Pass a current price.** That keeps the forecast output in dollar terms instead of base-100 percentages.
- **Use the search prompt for inspection, not for the history check itself.** The history logic lives in the forecast path.

### Recommended monitoring cadence

You do not need high-frequency polling. The feature is built around two specific history windows, so a simple cadence works better than constant reruns.

| Goal | Example schedule | Why |
|------|------------------|-----|
| **Baseline + whale check** | Run at **09:00**, then again at **12:00** | The second run lands inside the 2–4h spike window |
| **Baseline + persistence check** | Run at **09:00 today**, then again at **09:00 tomorrow** | The next-day run starts giving you usable 24h history |
| **Full workflow** | Run at **09:00 today**, **12:00 today**, then **09:00 tomorrow** | Covers the baseline, the whale/spike check, and the next-day persistence read |

If you miss the exact window, just run the same BTC forecast again at the next convenient time. The feature is tolerant of normal human usage; it just needs snapshots that land inside the relevant windows.

### Additional prompt samples

Use these as copy-paste starting points when you want to work the feature manually.

| Intent | Prompt |
|--------|--------|
| **Baseline BTC forecast** | `Give me a Polymarket-based forecast for BTC over the next 7 days. Current price is 68000.` |
| **Whale-check rerun** | `Give me a Polymarket-based forecast for BTC over the next 7 days. Current price is 68000. Focus on whether any low-volume spike warnings appear.` |
| **Whale-aware BTC prediction** | `Give me a Polymarket-based forecast for BTC over the next 7 days. Current price is 68000. Pay special attention to the warning section and tell me whether any low-volume spike suggests possible whale activity or whether the history is still insufficient.` |
| **Persistence-check rerun** | `Give me a Polymarket-based forecast for BTC over the next 7 days. Current price is 68000. Tell me whether any earlier move now looks transitory.` |
| **Warning-focused rerun** | `Give me a Polymarket-based forecast for BTC over the next 7 days and focus on the warning section. Current price is 68000.` |
| **Short-horizon comparison** | `Give me a Polymarket-based forecast for BTC over the next 30 days. Current price is 68000.` |
| **Long-horizon contrast** | `Give me a Polymarket-based forecast for BTC over the next 180 days. Current price is 68000.` |
| **Manual market inspection** | `Search Polymarket for Bitcoin price markets and summarize the top results.` |
| **Threshold-market inspection** | `Search Polymarket for Bitcoin price markets and tell me which thresholds have the most volume.` |

**Tip:** For the history feature itself, the most important habit is not fancy wording. It is rerunning the **same BTC forecast prompt** on a stable cadence so the snapshot comparisons stay meaningful.

**Live-checked note:** The whale-aware BTC prompt above was live-checked with `ollama:glm-5:cloud`. On a cold-start run it still routed through `polymarket_forecast`, but the warning analysis correctly reported that the history was still insufficient rather than inventing whale activity.

---

## Time-Window Detection in `polymarket_forecast`

When `polymarket_forecast` runs, it reads past snapshots for each market and checks two time windows:

| Window | Timespan | What it detects | Key thresholds |
|--------|----------|-----------------|----------------|
| **Spike window** | 2–4 hours before now | Whale-driven price spike | \|delta\| > 8pp (0.08) **and** volume24h < $100K |
| **Persistence window** | 24–48 hours before now | Transitory (reversing) move | Original move > 10pp (0.10), moved toward baseline, reversal > 50% of original move |

These are **explanatory descriptions** of the code logic, not live-validated example outputs. Whether a specific run triggers a spike or transitory flag depends on which snapshots already exist in your local file and the live market state at query time. You cannot deterministically reproduce a spike/transitory warning without a seeded test harness.

### What the warnings look like in output

When prior data is **missing**, you see cold-start warnings in the Warnings section:

```
⚠ Spike detection unavailable: no prior snapshot found for market 0xabc123
⚠ Persistence test unavailable: no prior snapshot found for market 0xabc123 in 24-48h window
```

These are normal on your first runs for a market. They mean the history check was skipped, not that something is broken.

When a spike **is** detected, you see:

```
⚠ Market "…" has a price spike (possible whale activity) — quality discounted 50%
```

When a transitory move **is** detected, you see:

```
⚠ Market "…" shows a transitory 24-48h move — quality discounted 30%
```

### How quality discounts work

| Condition | Quality discount |
|-----------|-----------------|
| `priceSpikeDetected = true` | 50% reduction to market quality weight |
| `transitoryMove = true` (and no spike also flagged) | 30% reduction to market quality weight |
| Neither | No discount |

A lower quality weight means the market has less influence on the final ensemble forecast. This happens inside `computeMarketQualityWeight` in `ensemble.ts`.

---

## Cold-Start Behavior

The feature has a cold-start period for each market. The two history checks do **not** activate on the same schedule:

| History available before the current run | Spike detection | Transitory detection |
|------------------------------------------|----------------|----------------------|
| No prior snapshot | Unavailable (warning) | Unavailable (warning) |
| Only current-run snapshots | Unavailable (same-run writes do not count) | Unavailable |
| At least one snapshot in the **2–4h** window | **Active** | Still unavailable unless a **24–48h** snapshot also exists |
| At least one snapshot in the **24–48h** window only | Unavailable | Unavailable — transitory logic still needs a **2–4h** snapshot too |
| At least one snapshot in **both** windows | **Active** | **Active** |

In practice, spike warnings can disappear after a later run with usable 2–4h history. Transitory warnings need usable history in **both** the 2–4h and 24–48h windows.

---

## Live-Validated Example Prompts

These exact prompts were tested against live Polymarket data with `ollama:glm-5:cloud`. The user originally wrote `glmglm-5:cloud`, but the codebase and local Ollama runtime we verified support `ollama:glm-5:cloud` / `glm-5:cloud` instead.

### History-aware forecast prompts

| # | Prompt | What it returns |
|---|--------|-----------------|
| 1 | `Give me a Polymarket-based forecast for BTC over the next 7 days. Current price is 68000.` | Best starting prompt for the BTC/whale-detection workflow: short-horizon BTC forecast, dollar CI, and cold-start history warnings on the first run |
| 2 | `Give me a Polymarket-based forecast for BTC over the next 180 days. Current price is 68000.` | Broader long-horizon BTC forecast, including the >90-day warning and a wider confidence interval |

### Related discovery prompt (live-validated, but tool-level only)

| Prompt | What it returns |
|--------|-----------------|
| `Search Polymarket for Bitcoin price markets and summarize the top results.` | Active BTC threshold/barrier markets with crowd probabilities. Useful for manual discovery before rerunning the forecast workflow. At the **tool** level, `polymarket_search` does not read or write snapshot history; at the **full-agent** level, internal prompt injection may still fetch Polymarket context before tool selection. |

---

## Signal Quality by Horizon

| Horizon | Polymarket signal strength | Snapshot history value |
|---------|--------------------------|----------------------|
| 1–30 days | Strong | High — most liquid markets, frequent snapshots |
| 30–90 days | Moderate | Moderate — fewer markets, sparser history |
| 90–365 days | Weaker | Low — thin markets, long gaps between snapshots |

---

## What Not to Expect

| Not this | Why |
|----------|-----|
| Deterministic spike/transitory reproduction | Depends on when you last fetched that market; no seeded harness |
| Full Polymarket historical time series | Only writes what this tool fetches; does not backfill |
| Spike detection on first run | No 2–4h snapshot exists yet |
| Transitory detection within 24h of first use | No 24–48h snapshot exists yet |
| Alerts or notifications for detected spikes | The feature only sets quality weights and emits warnings in forecast output |
| Automatic pruning of old snapshots | The JSONL file grows indefinitely; manage it yourself if disk space matters |

---

## Storage Details

| Property | Value |
|----------|-------|
| Path | `.cramer-short/polymarket-snapshots.jsonl` |
| Format | Newline-delimited JSON (JSONL) |
| Write mode | Append-only (no overwrites) |
| Required fields per record | `marketId`, `question`, `probability`, `volume24h`, `endDate`, `capturedAt` |
| Malformed lines | Skipped with a warning to stderr; do not crash the tool |
| Market without `marketId` | Not recorded (the field is required) |

---

## Also See

- [`polymarket_forecast` description](../src/tools/finance/polymarket-forecast.ts) — full tool schema and signal-composition details
- [`markov-when-to-use.md`](markov-when-to-use.md) — coverage and confidence thresholds for Markov model forecasts
- [`watchlist.md`](watchlist.md) — `/watchlist` command reference with Markov confidence indicators
