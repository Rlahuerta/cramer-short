# Forecast Lab Guide

Forecast Lab is bounded experiment governance for repo-native forecast tools. It is not a free-form self-editing loop, infinite autoresearch system, or general permission to mutate the repository. V1 runs fixed baseline and candidate gates, records artifacts under `.cramer-short/experiments/`, and makes an explicit keep/drop decision from measured gates.

## List available profiles

Run from the repository root:

```bash
bun start lab list
```

The command prints the configured profile IDs and target subsystems. V1 profiles are:

| Profile | Target subsystem |
|---|---|
| `btc-markov-short-horizon` | `markov-distribution` |
| `btc-arbiter-replay` | `forecast-arbiter` |
| `polymarket-selection-sanity` | `polymarket-selection` |

## Run a dry-run experiment

V1 does not support real code mutation. The lab runner requires either `--dry-run` or `--skip-mutation`.

```bash
bun start lab run <profile> --dry-run
```

Example:

```bash
bun start lab run btc-markov-short-horizon --dry-run
```

The command:

1. loads the selected profile from `src/experiments/forecast-lab/profiles.ts`,
2. creates a run ID like `forecast-lab-<profile>-<timestamp>`,
3. writes a manifest,
4. runs the profile baseline commands,
5. runs the same candidate commands without mutating source code,
6. evaluates the profile keep/drop rule,
7. appends one row to `.cramer-short/experiments/forecast-results.tsv`.

The CLI prints the decision and the run artifact directory:

```text
forecast-lab <keep|drop>: <reason>
artifacts: .cramer-short/experiments/runs/<run-id>
```

## Run artifacts

Each run writes files directly under:

```text
.cramer-short/experiments/runs/<run-id>/
```

V1 artifacts are:

| File | Meaning |
|---|---|
| `manifest.json` | Run metadata: `runId`, `startedAt`, `profileId`, `targetSubsystem`, candidate branch name, allowed globs, and `artifactsPath`. |
| `baseline.json` | Full baseline gate results, including command IDs, commands, exit codes, stdout, stderr, durations, and timeout status. |
| `candidate.json` | Full candidate gate results plus `mutation`, which is `dry-run: no code mutation attempted` or `skipped by --skip-mutation`. |
| `decision.json` | Final `keep`/`drop` decision, reason, and compared metrics. |

Although path helpers include an `artifacts` directory helper, the current V1 runner writes the files above directly in the run directory and sets `artifactsPath` to that run directory.

## Reading `forecast-results.tsv`

The append-only ledger lives at:

```text
.cramer-short/experiments/forecast-results.tsv
```

The header columns are:

```text
runId	startedAt	profileId	targetSubsystem	candidateBranch	allowedGlobs	baselineSummary	candidateSummary	decision	reason	artifactsPath
```

Rows are tab-separated, and every field is JSON-serialized. That means string fields appear quoted, arrays and objects are valid JSON, and nested summaries can be parsed with a JSON-aware TSV reader.

Important fields:

- `runId`: safe path segment used under `.cramer-short/experiments/runs/`.
- `startedAt`: ISO timestamp for the run.
- `profileId`: profile that selected the gates and keep/drop rule.
- `targetSubsystem`: forecast subsystem covered by the profile.
- `candidateBranch`: generated branch name for candidate bookkeeping.
- `allowedGlobs`: source files the profile would allow a bounded candidate to edit.
- `baselineSummary`: compact baseline gate result with command IDs, exit codes, durations, and timeout flags.
- `candidateSummary`: compact candidate gate result with the same shape.
- `decision`: `keep` or `drop`.
- `reason`: measured rule that caused the decision.
- `artifactsPath`: run directory containing the full JSON artifacts.

## Keep/drop decisions

Decisions come from each profile's `keepDropRule`, not prose confidence. Current profiles compare required metrics extracted from baseline and candidate gate summaries. In V1 those metrics are command exit-code gates because the targeted Bun tests do not export richer JSON metrics yet.

The runner:

- drops immediately when a configured `dropWhen.any` criterion matches,
- keeps only when every `keepWhen.all` criterion matches,
- otherwise uses the profile default decision, which is currently `drop`.

For dry runs, no source mutation occurs. A dry run may still report `keep` if the unchanged candidate pass satisfies the profile's gates; interpret that as "the orchestration and gates passed," not as a source change to retain.

`--skip-mutation` always drops in V1. The runner rewrites the measured decision to:

```text
mutation skipped by --skip-mutation; no candidate code change to keep
```

This keeps no-op non-dry-run jobs from being mistaken for accepted code improvements.

## Immutable harnesses and editable surfaces

Forecast Lab deliberately separates editable forecast surfaces from read-only harnesses.

Default editable forecast surfaces from the skill are narrow:

- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/polymarket-forecast.ts`
- `src/tools/finance/forecast-arbitrator.ts`

Profiles may allow a small set of related helper files through `allowedGlobs`. Harness files remain read-only acceptance infrastructure, especially:

- `src/tools/finance/backtest/walk-forward.ts`
- `src/tools/finance/backtest/arbiter-replay-runner.ts`
- profile-selected targeted unit, integration, or E2E tests

Do not edit harnesses to make a candidate pass. A candidate must pass the same fixed gates used by the baseline.

## Add a new profile safely

Profiles live in `src/experiments/forecast-lab/profiles.ts`. To add one safely:

1. Add the new profile ID to `ForecastLabProfileId`.
2. Add or reuse a `ForecastLabTargetSubsystem`.
3. Add a `PROFILES_BY_ID` entry with:
   - narrow `allowedGlobs`,
   - explicit `readOnlyHarnessFiles`,
   - fixed `baselineCommands`,
   - fixed `candidateCommands`,
   - required `minimumMetrics`,
   - a conservative `keepDropRule` with default `drop`.
4. Add the profile to `FORECAST_LAB_PROFILES` so `bun start lab list` can show it.
5. Prefer existing targeted `bun test ...` commands. The runner rejects unsafe shell commands containing shell metacharacters or mutating git commands such as `git add`, `git commit`, `git push`, `git reset`, `git checkout`, or `git clean`.
6. Keep baseline and candidate gates comparable. If a candidate command differs, document why the metric paths still compare the same measurement.
7. Add or update tests for the profile and run the relevant test/typecheck commands.

Do not add profiles that allow broad repository edits, harness edits, destructive git operations, or unbounded iteration.

## Schedule forecast-lab jobs

Schedule configuration is global at:

```text
~/.cramer-short/schedules.json
```

Experiment artifacts remain project-local under `.cramer-short/experiments/` in the repository where the schedule command runs.

Example dry-run job:

```json
[
  {
    "id": "nightly-btc-markov-lab",
    "kind": "forecast_lab",
    "description": "Nightly BTC Markov forecast-lab dry run",
    "profileId": "btc-markov-short-horizon",
    "dryRun": true,
    "outputFile": "~/.cramer-short/reports/{date}-btc-markov-lab.md"
  }
]
```

Run scheduled jobs with:

```bash
bun start schedule list
bun start schedule run
bun start schedule run nightly-btc-markov-lab
```

Schedule fields for `forecast_lab` jobs:

| Field | Required | Meaning |
|---|---:|---|
| `id` | Yes | Job identifier used by `schedule run <job-id>`. |
| `kind` | Yes | Must be `forecast_lab`. |
| `description` | No | Human-readable label. |
| `profileId` | Yes | Must match a configured forecast-lab profile. |
| `maxIterations` | No | Accepted for config compatibility, reserved for future bounded mutation loops, and not used by V1. |
| `outputFile` | No | Optional Markdown summary path. Supports `~` and `{date}`. Schedule outputs must stay under `~/.cramer-short` or the project `.cramer-short`. |
| `dryRun` | No | Defaults to `true` unless `skipMutation` is true. |
| `skipMutation` | No | Runs the existing no-mutation mode instead of dry-run; V1 records a drop because there is no candidate code change to keep. |

If a scheduled forecast-lab job sets `"dryRun": false`, it must also set `"skipMutation": true`; otherwise the schedule runner fails loudly because real mutation is not implemented in V1.
