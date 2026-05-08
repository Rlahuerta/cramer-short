# Replay Label Pipeline Guide

The replay label pipeline is the operational bridge between:

1. **capturing forecast-arbitrator replay bundles**, and
2. **having enough labeled evidence to benchmark and eventually gate fusion decisions**.

It gives the repo a safe, auditable workflow for taking stored replay bundles, labeling the ones that are mature enough, benchmarking the labeled output, promoting reviewed artifacts into a maintained labeled cache, and deciding whether the resulting evidence is strong enough to move past `hold`.

---

## What this implementation adds

This rollout adds four operator-facing surfaces:

| Surface | Purpose |
|---|---|
| `bun start replay-label run` | Labels stored replay bundles into a staged JSONL and writes machine-readable reports. |
| `bun start schedule run <job>` with `kind: "replay_label"` | Runs the same pipeline headlessly from `~/.cramer-short/schedules.json`. |
| `bun start replay-label promote` | Explicitly copies reviewed staged artifacts into the labeled cache target. |
| `bun start replay-label readiness` | Produces a read-only readiness decision (`eligible` or `hold`) from the benchmark artifact/report. |

This is intentionally **conservative**:

- no in-place rewrite of the raw replay bundle file,
- no automatic promotion,
- no automatic config mutation,
- no live network dependency inside the first-cut CLI loader,
- and no assumption that every replay bundle is label-ready.

---

## Mental model

Use the pipeline as a 4-step operational loop:

1. **Run** labeling on stored replay bundles into a staged output file.
2. **Inspect** the staged labeled JSONL plus the label and benchmark reports.
3. **Promote** those staged artifacts into the maintained labeled cache only when you are satisfied.
4. **Check readiness** to decide whether evidence is still too thin (`hold`) or has crossed the configured thresholds (`eligible`).

If you remember one rule, remember this:

> **Raw capture stays raw, staged labeling is reviewable, promotion is explicit, and readiness is advisory.**

---

## Implementation map

These are the key files behind the feature:

| File | What it owns |
|---|---|
| `src/index-routing.ts` | Top-level argv routing for `replay-label`, `schedule`, and `lab`. |
| `src/index.tsx` | Wires `replay-label` into the real CLI entrypoint. |
| `src/cli-replay-label.ts` | User-facing command parser and JSON output for `run`, `promote`, and `readiness`. |
| `src/cli-schedule.ts` | `replay_label` schedule job kind and path allow-list enforcement. |
| `src/tools/finance/backtest/replay-label-runner.ts` | Core label pass and staged labeled JSONL writing. |
| `src/tools/finance/backtest/replay-label-batch-runner.ts` | Per-ticker history-window batching and the label batch report. |
| `src/tools/finance/backtest/replay-label-benchmark-pipeline.ts` | Labeling + replay benchmark handoff and benchmark artifact writing. |
| `src/tools/finance/backtest/replay-label-promotion.ts` | Explicit staged-artifact promotion plus promotion receipt. |
| `src/tools/finance/backtest/replay-label-readiness.ts` | Horizon-aware readiness evaluation and report generation. |
| `src/tools/finance/arbiter-replay.ts` | Raw and labeled replay bundle path constants. |

---

## Artifact flow and default paths

### Raw replay capture

The default raw replay bundle source is:

```text
.cramer-short/cache/arbiter-replay/bundles.jsonl
```

This is exposed as:

- `DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH`

There is also a legacy raw path:

```text
.cramer-short/arbiter-replay-bundles.jsonl
```

Promotion code explicitly protects both raw paths from being used as labeled targets.

### Staged labeled output

The default staged labeled JSONL is:

```text
.cramer-short/arbiter-replay-bundles-labeled.jsonl
```

This is exposed as:

- `DEFAULT_ARBITER_REPLAY_LABELED_PATH`

The staged label batch report is derived from the output path:

```text
.cramer-short/arbiter-replay-bundles-labeled.report.json
```

The staged benchmark artifact/report is also derived from the output path:

```text
.cramer-short/arbiter-replay-bundles-labeled.benchmark.report.json
```

### Promoted labeled cache

The maintained labeled cache target is:

```text
.cramer-short/cache/arbiter-replay/labeled/bundles.jsonl
```

Its default receipt path is:

```text
.cramer-short/cache/arbiter-replay/labeled/bundles.promotion.receipt.json
```

Promotion also derives matching promoted label and benchmark report paths beside that labeled cache file.

---

## What each artifact contains

### 1. Staged labeled JSONL

This is the replay bundle stream after the pipeline has:

- preserved already-labeled rows unchanged,
- labeled newly ready bundles,
- left not-yet-ready bundles pending,
- and left missing-history bundles unlabeled.

The file still round-trips through `readArbiterReplayBundles(...)`.

### 2. Label batch report

Format:

```text
replay-label-batch-report.v1
```

It contains:

- input path,
- output path,
- `labeledAt`,
- summary counts,
- per-reason pending counts,
- per-ticker counts,
- and audit data for each history request window.

This is the fastest way to answer questions like:

- how many bundles were newly labeled,
- which tickers had missing history,
- and which bundles are still blocked because the horizon has not matured.

### 3. Benchmark artifact/report

Format:

```text
replay-label-benchmark-report.v1
```

It contains:

- the staged labeled output path,
- the benchmark report path,
- horizon counts,
- and the nested short-horizon replay benchmark report.

The nested benchmark report still uses:

```text
polymarket-short-horizon-benchmark.v1
```

The readiness command accepts this benchmark artifact/report directly.

### 4. Promotion receipt

Format:

```text
replay-label-promotion-receipt.v1
```

It records:

- source staged artifact paths,
- promoted target paths,
- promotion timestamp,
- receipt path,
- total promoted bundle count,
- labeled promoted bundle count.

This is your audit trail proving exactly what was promoted and where it went.

### 5. Readiness report

Format:

```text
replay-label-readiness-report.v1
```

It contains:

- `decision: "eligible" | "hold"`,
- top-level reasons,
- the thresholds used,
- per-horizon pass/hold decisions,
- and the metrics behind those decisions.

---

## How `replay-label run` works

Command:

```bash
bun start replay-label run
```

The command:

1. reads replay bundles from the input JSONL,
2. groups unlabeled bundles by ticker,
3. computes the minimum history window needed for each ticker,
4. loads history once per ticker,
5. runs the existing eligibility + labeling logic,
6. writes the staged labeled JSONL,
7. writes the label batch report,
8. runs the short-horizon replay benchmark handoff,
9. writes the benchmark artifact/report,
10. prints a machine-readable JSON summary to stdout.

### Default behavior

If you run it with no flags:

```bash
bun start replay-label run
```

it defaults to:

- `--input .cramer-short/cache/arbiter-replay/bundles.jsonl`
- `--output .cramer-short/arbiter-replay-bundles-labeled.jsonl`
- `--loader fixture`

and derives the report paths from `--output`.

### Supported flags

```bash
bun start replay-label run \
  --input PATH \
  --output PATH \
  --label-report PATH \
  --benchmark-report PATH \
  --loader fixture|local:/absolute/path/to/fixture.json
```

### Current loader modes

The first cut is intentionally **local-only**:

| Loader | Meaning |
|---|---|
| `fixture` | Uses the bundled deterministic backtest price fixture. |
| `local:<path>` | Uses a local JSON fixture file at the provided path. |

There is **no live API mode** in this command yet.

That is deliberate: the initial operational surface favors deterministic behavior and safe testing over implicit network fetches.

### Example

```bash
bun start replay-label run \
  --input .cramer-short/cache/arbiter-replay/bundles.jsonl \
  --output .cramer-short/replay/2026-05-08-labeled.jsonl \
  --loader fixture
```

### Output summary shape

The command prints JSON like:

```json
{
  "status": "ok",
  "inputPath": ".cramer-short/cache/arbiter-replay/bundles.jsonl",
  "outputPath": ".cramer-short/replay/2026-05-08-labeled.jsonl",
  "labelReportPath": ".cramer-short/replay/2026-05-08-labeled.report.json",
  "benchmarkReportPath": ".cramer-short/replay/2026-05-08-labeled.benchmark.report.json",
  "labelingSummary": {
    "total": 0,
    "alreadyLabeled": 0,
    "newlyLabeled": 0,
    "skippedByMissingHistory": 0,
    "pending": 0
  }
}
```

Treat this stdout JSON as the machine-readable handoff for wrappers or automation.

---

## How the batch history logic behaves

The pipeline does **not** fetch history once per bundle. It batches by ticker.

For each ticker with unlabeled bundles, it computes:

- `windowStartAt`: earliest relevant `capturedAt`,
- `windowEndAt`: latest required evaluation boundary from forecast horizon or market end date,
- `bundleCount`: how many bundles depend on that same history window.

That matters because it keeps the implementation efficient and auditable:

- one ticker history request can unlock multiple bundles,
- already-labeled bundles do not trigger new history loads,
- and missing history is reported explicitly instead of silently collapsing to empty labels.

---

## How `replay-label promote` works

Command:

```bash
bun start replay-label promote
```

This command is the **explicit promotion gate** between staged output and maintained labeled cache.

It verifies:

1. the staged labeled JSONL exists,
2. the staged label report exists,
3. the staged benchmark artifact/report exists,
4. the staged labeled JSONL parses cleanly with `readArbiterReplayBundles(...)`,
5. none of the staged, promoted, or receipt paths collide with each other,
6. none of those paths collide with either raw replay bundle path.

Only after those checks pass does it copy artifacts into the promoted labeled cache target and write the promotion receipt.

### Supported flags

```bash
bun start replay-label promote \
  --input PATH \
  --output PATH \
  --label-report PATH \
  --benchmark-report PATH \
  --receipt PATH
```

### Example

```bash
bun start replay-label promote \
  --input .cramer-short/replay/2026-05-08-labeled.jsonl \
  --output .cramer-short/cache/arbiter-replay/labeled/bundles.jsonl
```

### Important safety rule

Promotion is **not** the same thing as replacing the raw replay capture file.

That distinction is enforced by code. The command refuses to target:

- `.cramer-short/cache/arbiter-replay/bundles.jsonl`
- `.cramer-short/arbiter-replay-bundles.jsonl`

for staged or promoted labeled artifacts.

---

## How `replay-label readiness` works

Command:

```bash
bun start replay-label readiness
```

This command reads the benchmark artifact/report and decides whether the replay evidence is currently strong enough to be considered:

- `eligible`, or
- `hold`

It is **read-only**. It does not flip any environment variable or feature flag for you.

### Default thresholds

| Horizon | Min labeled bundles | Min traded rows | Max abstain | Max Brier |
|---|---:|---:|---:|---:|
| `1d` | 30 | 15 | 0.45 | 0.24 |
| `2d` | 25 | 12 | 0.50 | 0.25 |
| `3d` | 20 | 10 | 0.55 | 0.26 |

By default, readiness also requires each slice to already be marked `ready`.

### Supported flags

```bash
bun start replay-label readiness \
  --input PATH \
  --min-labeled 1d=30,2d=25,3d=20 \
  --min-traded 1d=15,2d=12,3d=10 \
  --max-abstain 1d=0.45,2d=0.5,3d=0.55 \
  --max-brier 1d=0.24,2d=0.25,3d=0.26 \
  --allow-not-ready
```

### Example

```bash
bun start replay-label readiness \
  --input .cramer-short/replay/2026-05-08-labeled.benchmark.report.json
```

### What usually causes `hold`

Common reasons include:

- missing `1d`, `2d`, or `3d` benchmark slices,
- slices that are not marked ready,
- too few labeled bundles,
- too few traded rows,
- abstain rate above threshold,
- missing Brier score,
- Brier score above threshold.

You should interpret `hold` as:

> **Keep collecting and labeling evidence; do not treat the replay set as rollout-ready yet.**

---

## How scheduled `replay_label` jobs work

The schedule runner supports a dedicated job kind:

```json
{
  "id": "nightly-replay-label",
  "kind": "replay_label",
  "description": "Nightly replay labeling",
  "outputPath": "~/.cramer-short/replay/{date}-labeled.jsonl",
  "loader": "fixture",
  "outputFile": "~/.cramer-short/reports/{date}-replay-label.md"
}
```

Then run:

```bash
bun start schedule list
bun start schedule run nightly-replay-label
```

### Supported schedule fields

| Field | Meaning |
|---|---|
| `id` | Unique job id. |
| `kind` | Must be `"replay_label"`. |
| `description` | Optional human-readable label. |
| `inputPath` | Optional replay input path. |
| `outputPath` | Optional staged labeled output path. |
| `labelReportPath` | Optional explicit label report path. |
| `benchmarkReportPath` | Optional explicit benchmark report path. |
| `loader` | `fixture` or `local:<path>`. |
| `outputFile` | Optional human-readable summary file. |

### Schedule path restrictions

Schedule-managed paths are intentionally restricted to:

- `~/.cramer-short/...`
- `<cwd>/.cramer-short/...`

That applies to both replay-label outputs **and** replay-label input paths.

The reason is simple: `schedules.json` must not become an arbitrary filesystem read/write escape hatch.

---

## Recommended operator workflow

For real usage, the safest sequence is:

1. **Run a staged labeling pass**

   ```bash
   bun start replay-label run --loader fixture
   ```

2. **Inspect the artifacts**
   - staged labeled JSONL,
   - label batch report,
   - benchmark artifact/report.

3. **If the staged artifacts look correct, promote them**

   ```bash
   bun start replay-label promote
   ```

4. **Evaluate readiness**

   ```bash
   bun start replay-label readiness
   ```

5. **Only treat the replay set as rollout-ready if readiness returns `eligible` for the thresholds you actually trust.**

That sequence makes it hard to accidentally overwrite raw replay data and easy to audit what happened.

---

## What this implementation does not do

It is important to be explicit about what is still out of scope:

- It does **not** fetch live history from an API in the CLI by default.
- It does **not** auto-promote staged artifacts.
- It does **not** mutate `POLYMARKET_CROSS_PLATFORM_FUSION_ENABLED`.
- It does **not** assume every replay row is mature enough to label.
- It does **not** replace human review of the artifacts.

This is a deliberately operational, review-first pipeline rather than an automatic rollout switch.

---

## Troubleshooting

### `replay-label: unknown --loader mode`

Use only:

- `fixture`
- `local:/absolute/path/to/file.json`

### `missing staged ...`

Promotion needs all three staged artifacts:

- labeled JSONL,
- label report,
- benchmark artifact/report.

Run `replay-label run` first, or point `promote` at the actual staged paths.

### `must differ from the raw replay capture bundle path`

You tried to promote into a raw replay bundle path or used a raw path as a staged artifact path. Use the labeled cache path instead.

### Schedule job rejected outside allowed roots

Move the configured replay-label paths under:

- `~/.cramer-short/...`, or
- `<repo>/.cramer-short/...`

### Readiness stays on `hold`

Open the readiness JSON and inspect:

- top-level `reasons`,
- `horizons["1d" | "2d" | "3d"].reasons`,
- `labeledBundleCount`,
- `tradedRowCount`,
- `abstainRate`,
- `brierScore`,
- `ready`.

The report is designed to tell you exactly which threshold is still failing.

---

## Quick command reference

```bash
# Stage labeled replay bundles + reports
bun start replay-label run

# Same, but with explicit paths
bun start replay-label run --input INPUT.jsonl --output OUTPUT.jsonl --loader fixture

# Promote reviewed staged artifacts into the labeled cache
bun start replay-label promote

# Produce a read-only rollout-readiness decision
bun start replay-label readiness

# List scheduled jobs
bun start schedule list

# Run one replay_label scheduled job
bun start schedule run nightly-replay-label
```

---

## Bottom line

This implementation turns replay labeling into a real operational workflow:

- **staged instead of destructive,**
- **explicit instead of magical,**
- **machine-readable instead of ad hoc,**
- and **auditable from capture through promotion and readiness.**

Use it when you want to convert stored replay evidence into something you can inspect, benchmark, promote, and eventually trust for rollout decisions.
