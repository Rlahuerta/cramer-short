# Forecast Lab Guide

Forecast Lab is a **bounded experiment-governance workflow** for this repository's forecast tools. It helps you run repeatable forecast checks, record results, and decide whether a candidate change should be kept or dropped.

It is **not**:

- a free-form self-editing agent,
- an infinite autoresearch loop,
- a replacement for human review,
- or permission to change the evaluation harness to make a candidate pass.

The current implementation is intentionally conservative about **what** can mutate and **how** it mutates. Real structured mutation is now supported for the shipped Markov profiles, but it is still bounded, typed, and auditable rather than free-form.

---

## What the 6 implemented pieces are

Forecast Lab now consists of 6 concrete pieces that work together:

| Phase | What was added | Why it exists |
|---|---|---|
| 1 | `src/skills/forecast-lab/SKILL.md` | Defines the operating rules for bounded forecast experimentation. |
| 2 | Ledger + path helpers | Stores run history and artifacts under `.cramer-short/experiments/`. |
| 3 | Typed profiles | Defines which subsystem to test, which files are editable, which harnesses are read-only, and how keep/drop is decided. |
| 4 | Lab runner + CLI | Gives you `bun start lab ...` commands and the run orchestration. |
| 5 | Schedule integration | Lets you run `forecast_lab` jobs headlessly from `~/.cramer-short/schedules.json`. |
| 6 | This guide | Explains how to use all of the above effectively and safely. |

If you only remember one thing, remember this:

> **Profiles decide the rules, the runner executes them, the ledger records them, and the guide explains how to use them safely.**

---

## Recommended way to use Forecast Lab

For most people, the most efficient workflow is:

1. **List profiles** to see what already exists.
2. **Pick the narrowest profile** that matches the subsystem you care about.
3. **Run a dry run** first.
4. **If the profile is one of the shipped Markov profiles, run one explicit structured mutation** with `--mutation structured`.
5. **Inspect the run artifacts** in `.cramer-short/experiments/runs/<run-id>/`.
6. **Check the ledger** to see the historical keep/drop and lineage record.
7. **Use schedules** only after the manual workflow is clear.

That sequence is the safest way to learn the system and the fastest way to avoid confusion.

---

## A gentle mental model

It helps to think of Forecast Lab as a 4-part loop:

1. **Choose a profile**
   A profile defines the subsystem, commands, metrics, and keep/drop rules.

2. **Run baseline and candidate gates**
   Structured profiles can run a real candidate mutation in an isolated worktree, but only when you opt in with `--mutation structured`. `--dry-run` and `--skip-mutation` still preserve the no-mutation paths, and unknown or conflicting flags are rejected instead of falling through to a real mutation.

3. **Write artifacts and ledger entries**
   Every run leaves a paper trail under `.cramer-short/experiments/`.

4. **Interpret the outcome carefully**
   A `keep` during dry-run means "the orchestration passed the gates," not "new code was proven better."

This distinction matters a lot: the system can now run one real bounded mutation, but it still does not behave like an unbounded self-editing search loop.

---

## Quick start

From the repository root:

```bash
bun start lab list
```

Then start with a dry run:

```bash
bun start lab run multi-asset-markov-short-horizon --dry-run
```

If that profile is the one you actually want to optimize, follow it with one explicit structured mutation:

```bash
bun start lab run multi-asset-markov-short-horizon --mutation structured
```

You should see output like:

```text
forecast-lab <keep|drop>: <reason>
artifacts: .cramer-short/experiments/runs/<run-id>
```

After that, inspect:

1. `.cramer-short/experiments/runs/<run-id>/decision.json`
2. `.cramer-short/experiments/runs/<run-id>/candidate.json`
3. `.cramer-short/experiments/runs/<run-id>/manifest.json`
4. `.cramer-short/experiments/forecast-results.tsv`

If you do just those four checks, you already understand the main Forecast Lab workflow.

---

## Available lab commands

### List profiles

```bash
bun start lab list
```

This prints all configured Forecast Lab profiles and their target subsystems.

Current shipped profiles:

| Profile | Target subsystem |
|---|---|
| `multi-asset-markov-short-horizon` | `markov-distribution` |
| `btc-markov-ultra-short-horizon` | `markov-distribution` |
| `btc-arbiter-replay` | `forecast-arbiter` |
| `polymarket-selection-sanity` | `polymarket-selection` |

### Run a profile in dry-run mode

```bash
bun start lab run <profileId> --dry-run
```

Example:

```bash
bun start lab run btc-arbiter-replay --dry-run
```

### Run in skip-mutation mode

```bash
bun start lab run <profileId> --skip-mutation
```

This is supported, but it is mostly useful for plumbing and schedule compatibility right now. It always ends in a **drop** because no candidate code change exists to keep.

### Run a shipped structured mutation

```bash
bun start lab run multi-asset-markov-short-horizon --mutation structured
```

Today this is supported for the shipped Markov profiles. Forecast Lab will:

1. run the baseline gate in the repo root
2. reuse the last kept structured run as the parent seed when lineage metadata is available
3. create an isolated candidate worktree
4. replay the kept parent lineage into that worktree
5. apply one new structured mutation from the shipped catalog
6. run the candidate gate in that worktree
7. store `parentRunId`, `mutationId`, `mutationSummary`, and an immutable structured replay payload in the run manifest (the ledger keeps the lighter lineage summary)
8. keep or drop through the normal acceptance path, then clean up the candidate worktree unless you kept it for debugging

If there is no compatible kept structured parent yet, the run starts from the current repo baseline just like the first structured mutation run.

The old id `btc-markov-short-horizon` is still accepted as a compatibility alias, but the canonical shipped profile name is now `multi-asset-markov-short-horizon`. The old name was misleading because this gate is not BTC-only.

For real structured mutation runs, the CLI now prints the **selected mutator** and **patch summary** before the long baseline gate begins, and the final summary prints the concrete **previous parameter values** and **new mutated values** rather than only echoing the harness command.

For debugging, you can keep the workspace around:

```bash
bun start lab run multi-asset-markov-short-horizon --mutation structured --keep-worktree
```

You can also force one shipped mutation candidate:

```bash
bun start lab run multi-asset-markov-short-horizon --mutation structured --mutator markov-longer-stability-window
```

### How to use mutation effectively

If you want the mutation feature to be useful instead of confusing, use it in this order:

1. **Start with a dry run.**
   That tells you whether the profile and harness are the right fit before any candidate edit is attempted.

2. **Move to one explicit structured mutation.**
   Use:

   ```bash
   bun start lab run multi-asset-markov-short-horizon --mutation structured
   ```

   or:

   ```bash
   bun start lab run btc-markov-ultra-short-horizon --mutation structured
   ```

3. **Read the mutation evidence, not just the decision.**
   For structured runs, inspect:

   1. `decision.json`
   2. `candidate.json`
   3. `manifest.json`
   4. `forecast-results.tsv`

   `candidate.json` tells you what actually ran. `manifest.json` tells you the mutation contract, lineage, workspace, and replay payload that made the candidate reproducible. The terminal summary should also show a human-readable before/after parameter diff for the applied mutation.

4. **Use `--mutator` only when you are debugging or comparing a specific shipped mutation.**
   If you do not force a mutator, Forecast Lab picks the first shipped mutation that is both:
   - allowed by the profile contract, and
   - still applicable after replaying the last kept parent lineage.

5. **Use `--keep-worktree` only when you need to inspect the mutated checkout itself.**
   This is mainly for debugging why a structured run passed or failed.

6. **Expect lineage to matter after your first kept structured run.**
   The next structured run for the same profile can reuse that kept run as its parent, replay the parent mutation payload into a fresh worktree, and then apply one new shipped mutation on top.

This is the most important practical difference between:

- `--dry-run`: orchestration only, no candidate edit
- `--skip-mutation`: explicit no-op mutation path that always drops
- `--mutation structured`: one real shipped mutation in an isolated worktree

### Show lab usage

```bash
bun start lab
```

or:

```bash
bun start lab --help
```

---

## What each profile really means

Profiles are defined in `src/experiments/forecast-lab/profiles.ts`.

Each profile tells the runner:

- which subsystem is being evaluated,
- which files are narrow candidate edit surfaces,
- which mutation contract is allowed for that profile,
- which harness files are read-only,
- which baseline commands to run,
- which candidate commands to run,
- which metrics must exist,
- which rules force `keep` or `drop`.

### Current mutation contract

Profiles now carry a bounded mutation contract, and shipped Markov profiles can execute a real structured mutation in an isolated candidate worktree when you opt in with `--mutation structured`.

Profiles now carry a bounded `mutation` config in `src/experiments/forecast-lab/profiles.ts`:

- `mode`: `dry-run`, `structured`, or reserved `llm`
- `mutableFiles`: the files the selected mutation mode may target
- `allowedMutatorIds`: a bounded allowlist of structured mutators, present only when `mode` is `structured`
- `allowMultipleCandidateAttempts`: reserved for later multi-candidate runs

The shared contract lives in `src/experiments/forecast-lab/mutation.ts`. It keeps mutation behavior typed and narrow:

- unknown mutator ids fail loudly,
- non-structured modes do not accept fake `allowedMutatorIds`,
- mutation configs are deeply frozen before export,
- run artifacts persist the effective mutation contract snapshot (`mode`, `mutableFiles`, structured `allowedMutatorIds`, and `allowMultipleCandidateAttempts`) and can also carry optional mutation lineage metadata (`mutationMode`, `parentRunId`, `mutationId`, `mutationSummary`, `lineage`, `mutationSpecSummary`, `candidateWorkspace`). Structured manifests now also persist the exact replay payload for the kept mutation so future lineage replays do not depend on whatever catalog entry happens to ship later.

`allowedGlobs` and `mutation.mutableFiles` are also intentionally related but not identical concepts. `allowedGlobs` defines the profile-level editable surface, while `mutableFiles` defines the mode-specific mutation input. The current profiles keep them aligned for simplicity, but later phases may narrow or expand one without changing the other.

### `multi-asset-markov-short-horizon`

Use this when you want to work on the **multi-asset** short-horizon Markov forecast path.

Allowed candidate surfaces:

- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/conformal.ts`
- `src/tools/finance/regime-calibrator.ts`

Structured mutation contract:

- mode: `structured`
- allowed mutators: `search-replace`
- multiple candidate attempts: `false`

Read-only harness:

- `src/tools/finance/backtest/walk-forward.ts`

Gate command:

```bash
bun test src/tools/finance/backtest/walk-forward-short-horizon.test.ts --timeout 480000
```

This harness currently evaluates:

- `SPY`
- `QQQ`
- `GLD`
- `VOO`
- `NVDA`
- `MSFT`

across 7d, 14d, and 30d horizons.

### `btc-markov-ultra-short-horizon`

Use this when you want a **BTC-only ultra-short-horizon optimization loop** across **1d, 2d, and 3d** horizons. This is the lighter-weight profile to use when you care about very short BTC moves and want faster feedback than the broader 7d/14d/30d short-horizon profile.

Allowed candidate surfaces:

- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/conformal.ts`
- `src/tools/finance/regime-calibrator.ts`

Structured mutation contract:

- mode: `structured`
- allowed mutators: `search-replace`
- multiple candidate attempts: `false`

Read-only harness:

- `src/tools/finance/backtest/walk-forward.ts`

Gate command:

```bash
bun test src/tools/finance/backtest/walk-forward-btc-ultra-short-horizon.test.ts --timeout 360000
```

### `btc-arbiter-replay`

Use this when you want to work on the forecast arbitration path.

Allowed candidate surfaces:

- `src/tools/finance/forecast-arbitrator.ts`
- `src/tools/finance/forecast-hooks.ts`

Current mutation contract:

- mode: `dry-run`
- multiple candidate attempts: `false`

Read-only harness:

- `src/tools/finance/backtest/arbiter-replay-runner.ts`

Gate command:

```bash
bun test src/tools/finance/backtest/arbiter-replay-runner.test.ts
```

### `polymarket-selection-sanity`

Use this when you want to work on the Polymarket selection/sanity path.

Allowed candidate surfaces:

- `src/tools/finance/polymarket-forecast.ts`
- `src/tools/finance/polymarket.ts`

Current mutation contract:

- mode: `dry-run`
- multiple candidate attempts: `false`

Read-only harness:

- none in the profile itself, but the profile still assumes fixed sanity gates

Gate command:

```bash
bun test src/tools/finance/polymarket-forecast.test.ts
```

### Which profile should you pick?

Use this simple rule:

| If you want to evaluate... | Start with... |
|---|---|
| Multi-asset Markov forecast mechanics | `multi-asset-markov-short-horizon` |
| BTC-only 1d/2d/3d Markov optimization | `btc-markov-ultra-short-horizon` |
| Arbitrator behavior or replay behavior | `btc-arbiter-replay` |
| Polymarket forecast selection/sanity | `polymarket-selection-sanity` |

When in doubt, choose the **narrowest** profile that still matches your goal.

If your goal is specifically **BTC**, start with `btc-markov-ultra-short-horizon`, not `multi-asset-markov-short-horizon`.

---

## What happens during a run

Every run does the same base orchestration:

1. loads the profile,
2. creates a run ID like `forecast-lab-<profile>-<timestamp>`,
3. computes a candidate branch name,
4. creates the run directory under `.cramer-short/experiments/runs/<run-id>/`,
5. writes `manifest.json`,
6. runs the baseline commands,
7. writes `baseline.json`,
8. evaluates the candidate path,
9. writes `candidate.json`,
10. evaluates the keep/drop rule,
11. writes `decision.json`,
12. appends one ledger row to `.cramer-short/experiments/forecast-results.tsv`.

The important difference is **what the candidate path means**.

### Candidate path for `--dry-run`

When you run:

```bash
bun start lab run <profile> --dry-run
```

the candidate gate still runs, but it runs without creating a candidate worktree or applying any edit. This is the safest way to inspect the profile, harness, artifact shape, and ledger plumbing.

### Candidate path for `--skip-mutation`

When you run:

```bash
bun start lab run <profile> --skip-mutation
```

the runner still exercises the no-mutation candidate path, but it rewrites the final outcome to a drop because no candidate code change exists to keep.

### Candidate path for `--mutation structured`

When you run:

```bash
bun start lab run <markov-profile> --mutation structured
```

the runner:

1. resolves the effective mutation contract,
2. looks for the last kept structured parent run for the same profile,
3. creates a fresh candidate worktree,
4. replays the parent's immutable mutation payload when lineage exists,
5. selects either the requested shipped mutator or the first applicable unused shipped mutation,
6. applies that mutation inside the worktree,
7. runs the candidate gate in that worktree,
8. persists mutation metadata to artifacts and the ledger,
9. cleans up the worktree unless you passed `--keep-worktree`.

### Important clarification about the candidate branch

`--dry-run` and `--skip-mutation` still only record the candidate branch name as bookkeeping metadata. A real structured mutation run also creates an isolated candidate worktree and branch, records that workspace in the manifest/ledger, and then cleans it up unless you pass `--keep-worktree`. After you have one kept structured run, the next structured run can replay that kept lineage before applying the new mutation.

---

## Run artifacts: what each file is for

Each run writes directly under:

```text
.cramer-short/experiments/runs/<run-id>/
```

Current run files:

| File | What it tells you | What to check first |
|---|---|---|
| `manifest.json` | Which profile ran, baseline commit, mutation contract, lineage, and workspace metadata | Confirm you ran the profile and mutation mode you intended |
| `baseline.json` | Raw baseline command results | Check `exitCode`, command IDs, and timing |
| `candidate.json` | Raw candidate command results plus mutation metadata | Check `exitCode`, `mutationMode`, `mutationSummary`, `mutationSpecSummary`, and mutated files |
| `decision.json` | Final keep/drop decision and compared metrics | Read this first when diagnosing outcomes |

### Fastest inspection order

If you want to be efficient, inspect in this order:

1. `decision.json`
2. `candidate.json`
3. `baseline.json`
4. `manifest.json`

That gives you the answer first, then the evidence.

For structured mutation runs, `manifest.json` is especially important because it is where the immutable replay payload lives. The ledger keeps the lighter summary needed for triage; the manifest keeps the exact replay material needed for deterministic lineage replay later.

---

## Understanding `forecast-results.tsv`

The append-only ledger lives at:

```text
.cramer-short/experiments/forecast-results.tsv
```

Header:

```text
runId	startedAt	profileId	targetSubsystem	candidateBranch	allowedGlobs	effectiveMutationContract	mutationMode	parentRunId	mutationId	mutationSummary	lineage	mutationSpecSummary	candidateWorkspace	baselineSummary	candidateSummary	decision	reason	artifactsPath
```

### Important format detail

Rows are tab-separated, but **every field is JSON-serialized**. So:

- strings appear quoted,
- arrays are valid JSON arrays,
- summaries are valid JSON objects.

That means a plain TSV parser can read the row shape, but a JSON-aware parser is better if you want to inspect nested fields like `baselineSummary.commands`.

### What the important ledger fields mean

| Field | Meaning |
|---|---|
| `runId` | Safe path segment for the run directory |
| `startedAt` | ISO timestamp |
| `profileId` | Profile used for the run |
| `targetSubsystem` | Logical subsystem covered by the run |
| `candidateBranch` | Generated bookkeeping branch name |
| `allowedGlobs` | Candidate edit surface allowed by the profile |
| `effectiveMutationContract` | Persisted snapshot of the actual mutation contract for this run |
| `mutationMode` | Optional applied mutation mode metadata (`structured` for real shipped mutations) |
| `parentRunId` | Explicit kept-parent pointer for replaying the next structured run |
| `mutationId` | The shipped mutation candidate id that was actually applied |
| `mutationSummary` | Short human-auditable description of the applied mutation |
| `lineage` | Optional parent/root lineage for derived candidates |
| `mutationSpecSummary` | Optional compact summary of the concrete mutation request, including the selected mutator |
| `candidateWorkspace` | Optional workspace metadata for where the candidate was evaluated |
| `baselineSummary` | Compact baseline command summary |
| `candidateSummary` | Compact candidate command summary |
| `decision` | Final `keep` or `drop` |
| `reason` | Rule-based reason for that decision |
| `artifactsPath` | Directory containing the full JSON artifacts |

### Efficient way to use the ledger

Use the ledger for:

1. **history** — what was run and when,
2. **triage** — which runs dropped and why,
3. **comparison** — whether the same profile keeps failing for the same reason,
4. **traceability** — which artifact directory belongs to which result.

If you are scanning many runs, start with:

- `profileId`
- `decision`
- `reason`
- `artifactsPath`

Those four fields tell you most of what you need quickly.

For structured mutation history, also look at:

- `parentRunId`
- `mutationId`
- `mutationSummary`
- `candidateWorkspace`

---

## How keep/drop decisions work

Keep/drop decisions come from the profile's structured `keepDropRule`, not from natural-language impressions.

Current profiles use **command exit-code gates** as the measured metrics because the targeted Bun tests do not yet export richer JSON metrics.

The runner logic is:

1. evaluate required metrics,
2. if any `dropWhen.any` rule matches -> **drop**,
3. else if all `keepWhen.all` rules match -> **keep**,
4. else use the profile default decision -> currently **drop**.

### What `keep` means

For a dry run, `keep` does **not** mean:

- code was changed,
- the forecast got better,
- or a mutation has been approved for merge.

During `--dry-run`, `keep` only means:

> the orchestrated candidate pass satisfied the profile's declared gates.

For a real structured mutation run, `keep` means the mutated candidate satisfied the profile gates inside the isolated workspace. It still does **not** mean the change is auto-merged or exempt from review.

### What `drop` means

A drop usually means one of:

- a candidate command failed,
- a required metric was missing,
- the keep rules were not all satisfied,
- the default decision remained `drop`,
- or `--skip-mutation` forced a no-op drop.

### Why `--skip-mutation` always drops

When you run with:

```bash
bun start lab run <profile> --skip-mutation
```

the runner explicitly rewrites the result to:

```text
mutation skipped by --skip-mutation; no candidate code change to keep
```

This prevents no-op runs from looking like accepted improvements.

Use `--skip-mutation` only when you intentionally want:

- non-dry-run scheduling compatibility,
- runner plumbing checks,
- or explicit proof that no candidate change was evaluated.

For most people, **use `--dry-run` first**.

### Why `--dry-run` is still useful

`--dry-run` remains the safest way to inspect a profile without creating a candidate workspace.

Right now, the runner can:

- select a profile,
- run the fixed baseline gate,
- run the candidate gate,
- stream the optimization output,
- write artifacts,
- and record a keep/drop decision.

When you opt into `--mutation structured`, shipped Markov profiles can also create a bounded candidate workspace, apply one shipped mutator, and evaluate that candidate. `--dry-run` is still the honest default when you want orchestration and artifact checks without any candidate edit attempt.

So today:

- `--dry-run` = run the optimization harness safely with **no code mutation**
- `--skip-mutation` = explicitly run the no-mutation path and force a **drop**
- `--mutation structured` = run one shipped bounded mutation in an isolated candidate workspace

---

## Safety model: what is editable and what is not

Forecast Lab is built around **narrow editable surfaces** and **read-only harnesses**.

### Default narrow forecast surfaces from the skill

- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/polymarket-forecast.ts`
- `src/tools/finance/forecast-arbitrator.ts`

### Read-only harnesses

These are fixed acceptance infrastructure and should not be edited to make a run pass:

- `src/tools/finance/backtest/walk-forward.ts`
- `src/tools/finance/backtest/arbiter-replay-runner.ts`
- targeted profile tests used as acceptance gates

### Safe-command restrictions

The runner rejects unsafe profile commands before shell execution. It blocks:

- shell metacharacters such as `;`, `&`, `|`, `` ` ``, `<`, `>`, and `$()`
- mutating git commands such as:
  - `git add`
  - `git commit`
  - `git push`
  - `git reset`
  - `git checkout`
  - `git clean`

This is one of the most important safety guarantees in the system.

---

## The `forecast-lab` skill: what it is and how to think about it

The skill lives at:

```text
src/skills/forecast-lab/SKILL.md
```

It is **not** a separate shell command. Instead, it defines the bounded experimentation rules the system is built around.

Its job is to encode principles like:

- mutate only approved forecast surfaces,
- baseline first,
- compare candidate against fixed gates,
- record results under `.cramer-short/experiments/`,
- and drop or revert failed candidates.

Think of the skill as the **policy layer**, while:

- profiles are the **configuration layer**,
- the runner is the **execution layer**,
- and the ledger is the **history layer**.

---

## How to use the six pieces efficiently

If you want the most practical workflow, use the system like this:

### 1. Start from the CLI, not the code

Use:

```bash
bun start lab list
```

first. It is faster and safer than manually reading `profiles.ts` every time.

### 2. Use the narrowest profile

Do not start with a broad goal like "improve forecasting." Start with one subsystem:

- Markov,
- arbitrator,
- or Polymarket selection.

### 3. Start with `--dry-run`, then graduate to `--mutation structured`

`--dry-run` is the clearest mode to learn from because:

- the workflow runs fully,
- artifacts are written,
- no mutation is attempted,
- and the decision is still recorded.

Once that feels obvious, use `--mutation structured` on the shipped Markov profiles to run one real bounded candidate edit.

### 4. Use the ledger for history, not just the latest run

The run directory tells you what happened once.
The ledger tells you what keeps happening.

### 5. Schedule only after the manual flow feels obvious

If you cannot explain one manual run from `decision.json`, do not automate it yet.

### 6. Treat the current system as bounded optimization, not optimization magic

The current system is strongest at:

- enforcing boundaries,
- recording evidence,
- applying one bounded mutation safely,
- and preventing unsafe experimentation.

It is still **not** a free-form autonomous improvement engine.

---

## Scheduling forecast-lab jobs

Schedule configuration is global:

```text
~/.cramer-short/schedules.json
```

Experiment artifacts remain project-local:

```text
.cramer-short/experiments/
```

### Example `forecast_lab` dry-run job

```json
[
  {
    "id": "nightly-btc-markov-lab",
    "kind": "forecast_lab",
    "description": "Nightly BTC Markov forecast-lab dry run",
    "profileId": "multi-asset-markov-short-horizon",
    "dryRun": true,
    "outputFile": "~/.cramer-short/reports/{date}-btc-markov-lab.md"
  }
]
```

### Run scheduled jobs

```bash
bun start schedule list
bun start schedule run
bun start schedule run nightly-btc-markov-lab
```

When a scheduled run uses real structured mutation, the headless summary now includes the mutation mode, selected mutator, mutation operator, lineage, and candidate workspace so the report tells you that a real candidate edit happened.

### Fields for `forecast_lab` jobs

| Field | Required | Meaning |
|---|---:|---|
| `id` | Yes | Job id for `schedule run <job-id>` |
| `kind` | Yes | Must be `forecast_lab` |
| `description` | No | Friendly label |
| `profileId` | Yes | Must match a configured forecast-lab profile |
| `maxIterations` | No | Reserved for future bounded mutation loops; currently unused |
| `outputFile` | No | Optional summary path; supports `~` and `{date}` |
| `dryRun` | No | Defaults to `true` unless `skipMutation` is true |
| `skipMutation` | No | Runs no-mutation mode and records a drop |
| `mutationMode` | No | Explicit real-mutation mode. Set to `structured` to allow scheduled mutation runs |
| `keepWorktree` | No | Preserves the candidate worktree after a scheduled structured mutation |
| `mutator` | No | Optional shipped structured mutation id override for debugging |

### Important schedule rule

Scheduled mutation is **not** the default. If a scheduled forecast-lab job sets:

```json
"dryRun": false
```

it must choose exactly one explicit path:

```json
"mutationMode": "structured"
```

for a real structured mutation run, or:

```json
"skipMutation": true
```

Otherwise the schedule runner fails loudly instead of inferring a real mutation mode.

### Example scheduled structured mutation

```json
[
  {
    "id": "nightly-btc-markov-mutation",
    "kind": "forecast_lab",
    "description": "Nightly BTC Markov structured mutation",
    "profileId": "multi-asset-markov-short-horizon",
    "dryRun": false,
    "mutationMode": "structured",
    "mutator": "markov-longer-stability-window",
    "outputFile": "~/.cramer-short/reports/{date}-btc-markov-mutation.md"
  }
]
```

### Output path restrictions

Schedule output files must stay under either:

- `~/.cramer-short`
- or `<repo>/.cramer-short`

This is a safety boundary. The schedule runner rejects outputs outside those roots.

---

## How to add a new profile safely

Profiles live in:

```text
src/experiments/forecast-lab/profiles.ts
```

Use this checklist:

1. Add the new profile ID to `ForecastLabProfileId`.
2. Add or reuse a `ForecastLabTargetSubsystem`.
3. Add a `PROFILES_BY_ID` entry.
4. Use **narrow** `allowedGlobs`.
5. Declare explicit `readOnlyHarnessFiles`.
6. Use fixed baseline and candidate commands.
7. Declare required `minimumMetrics`.
8. Start with a conservative `keepDropRule` and default `drop`.
9. Add the profile to `FORECAST_LAB_PROFILES`.
10. Add or update tests.

### Good profile design

Good profiles are:

- narrow,
- measurable,
- conservative,
- and easy to explain.

### Bad profile design

Do **not** add profiles that:

- allow broad repository edits,
- allow harness edits,
- depend on destructive git behavior,
- use unsafe shell commands,
- or imply unbounded looping.

---

## Common situations and what to do

### "I got `keep` from a dry run. Should I trust that as an improvement?"

Not yet. Treat it as:

> "the gates passed under orchestration"

not:

> "the forecast logic was improved and should be merged"

### "I want a repeated nightly signal check, not code mutation."

Use a scheduled `forecast_lab` job with:

- `dryRun: true`
- an `outputFile`
- and the narrowest profile that matches your goal

### "I want the lab to actually try one shipped mutation each night."

Use a scheduled `forecast_lab` job with:

- `dryRun: false`
- `mutationMode: "structured"`
- optionally `mutator` if you want one exact shipped candidate
- optionally `keepWorktree: true` only when you are debugging a failure or unexpected keep

### "I want to understand why a run dropped."

Read in this order:

1. `decision.json`
2. `candidate.json`
3. `baseline.json`
4. `forecast-results.tsv`

### "I need the fastest way to inspect one run."

Open:

```text
.cramer-short/experiments/runs/<run-id>/decision.json
```

first.

### "Can I edit the harness if a candidate almost passes?"

No. The harness is the acceptance standard, not the thing being optimized.

---

## Troubleshooting

### Error: missing profile id

You probably ran:

```bash
bun start lab run
```

without a profile. Re-run with a profile id:

```bash
bun start lab run multi-asset-markov-short-horizon --dry-run
```

### Error: unknown forecast-lab profile id

Use:

```bash
bun start lab list
```

and copy a valid profile id exactly.

### Error: real mutation requires a shipped structured profile

Use a shipped Markov profile with:

```bash
--mutation structured
```

For the no-mutation paths, use:

```bash
--dry-run
```

or:

```bash
--skip-mutation
```

if you intentionally want the no-mutation paths.

### Error: real mutation requires an explicit mutation mode

You probably ran a real mutation shape without:

```bash
--mutation structured
```

Forecast Lab fails closed here on purpose. A bare:

```bash
bun start lab run multi-asset-markov-short-horizon
```

does **not** silently fall through to a real mutation run.

### Error: unknown forecast-lab flag

The CLI rejects typos like:

```bash
--dryrun
```

or conflicting combinations like:

```bash
--dry-run --skip-mutation
```

Use one explicit path only:

- `--dry-run`
- `--skip-mutation`
- or `--mutation structured`

### Error: schedule output outside allowed roots

Your `outputFile` escaped the permitted schedule roots. Keep it under:

- `~/.cramer-short/...`
- or `<repo>/.cramer-short/...`

---

## Final guidance

Forecast Lab is already useful, but its usefulness still comes from **discipline**, not raw automation.

Use it to:

- choose a narrow profile,
- run a reproducible experiment,
- inspect artifacts,
- record outcomes,
- and avoid unsafe or vague experimentation.

If you use it that way, the current six implemented pieces work well together and stay understandable as the system grows.
