---
name: forecast-lab
description: >
  Bounded forecast experimentation workflow for improving repo-native forecast
  tools. Use when the user wants to run, design, or assess a baseline-vs-candidate
  experiment for Markov, Polymarket, or forecast arbitrator logic with fixed
  gates and a keep-or-drop decision.
---

# Forecast Lab Skill

You are running the forecast-lab skill. Your job is to manage a bounded forecast
experiment, not to perform broad self-editing. Treat every run as a small
baseline-vs-candidate comparison with a fixed harness and an explicit keep/drop
decision.

## Non-negotiable safety rules

- No infinite loops. Use a fixed iteration budget from the user or stop after one
  baseline and one candidate pass.
- Do not enter broad self-edit mode. Only modify files explicitly allowed for the
  selected forecast experiment.
- Do not edit evaluation harnesses, test harnesses, package scripts, lockfiles,
  generated output, or unrelated application code.
- Run the baseline before any candidate changes.
- Compare the candidate against the same fixed gates used for the baseline.
- Keep only candidates with measurable improvement that pass all required gates.
- Drop or revert failed candidates instead of leaving speculative code in place.
- Record results under `.cramer-short/experiments/`; these are runtime artifacts,
  not source files to commit.

## Editable forecast surfaces

Editable files are profile-driven, not an open-ended per-run allowlist. In the
current shipped implementation, only the two Markov profiles advertise
the structured mutation mode, and their
shipped catalog is currently limited to `search-replace`. The arbiter and
Polymarket profiles remain `dry-run` until they have concrete structured
mutation catalogs.

- `multi-asset-markov-short-horizon`
  - `src/tools/finance/markov-distribution.ts`
  - `src/tools/finance/conformal.ts`
  - `src/tools/finance/regime-calibrator.ts`
- `btc-markov-ultra-short-horizon`
  - `src/tools/finance/markov-distribution.ts`
  - `src/tools/finance/conformal.ts`
  - `src/tools/finance/regime-calibrator.ts`
- `btc-arbiter-replay`
  - `src/tools/finance/forecast-arbitrator.ts`
  - `src/tools/finance/forecast-hooks.ts`
- `polymarket-selection-sanity`
  - `src/tools/finance/polymarket-forecast.ts`
  - `src/tools/finance/polymarket.ts`

Do not widen the editable surface beyond the selected profile. If the user asks to
edit a forecast file outside that profile contract, refuse that part and explain
that forecast-lab is intentionally bounded in the shipped implementation.

Read-only harnesses for acceptance include:

- `src/tools/finance/backtest/walk-forward.ts`
- `src/tools/finance/backtest/arbiter-replay-runner.ts`
- existing targeted unit, integration, or E2E tests chosen by the profile

## Routing intent and auto-route boundaries

Automatic Forecast Lab routing is a hint layer, not a mutation trigger.

- Agent-side routing hints are only injected when both
  `forecasting.enableForecastLabAutoRoute` and
  `forecasting.enableForecastLabSkillHint` are enabled.
- Improvement intent is separate from ordinary tool usage. Queries need explicit
  improvement cues such as "improve", "optimize", "tune", "calibrate", or
  "fix" plus enough profile keywords before Forecast Lab should be considered.
- Plain usage queries such as "use BTC forecast tool" or "What is the BTC
  forecast?" are ordinary forecast requests and must not auto-enter Forecast Lab
  improvement mode.
- A profile/topic match does not authorize immediate mutation. Even when a
  routed query points at `btc-markov-ultra-short-horizon` or
  `multi-asset-markov-short-horizon`, real candidate edits still require an
  explicit structured run.
- If the operator forces a mutator, require
  `--mutation structured --mutator <id>`. If they do not force one, leave
  selection to the runner. Ledger-based mutator ranking is only considered when
  the `forecasting.enableForecastLabMutatorRanking` setting is enabled, and it defaults to
  off.
- When a routed or manually requested run executes, preserve the routing context
  in the manifest and ledger and let the runner update
  `.cramer-short/forecast-lab-routing-stats.json`. This is telemetry for future
  evidence-driven evolution work, not a current multi-iteration self-evolution
  loop.

Examples:

- "improve BTC short-horizon forecast" → improvement intent; prefer
  `btc-markov-ultra-short-horizon`; start with `bun start lab run
  btc-markov-ultra-short-horizon --dry-run`.
- "use BTC forecast tool" → ordinary usage; do not switch into Forecast Lab.
- "force a specific mutator" → use `bun start lab run
  btc-markov-ultra-short-horizon --mutation structured --mutator
  markov-longer-stability-window`.

## Workflow

### 1. Define the experiment

State the target subsystem, the approved editable files, the fixed evaluation
commands, and the candidate hypothesis. If any of those are missing, make the
smallest reasonable assumption and call it out before proceeding.

### 2. Establish the baseline first

Before changing code, run or request the fixed baseline gates for the selected
profile. Capture the metrics that will decide the keep/drop outcome. Do not
change code until the baseline result is known.

### 3. Make one bounded candidate change

If code mutation is authorized, use the isolated candidate workspace and touch
only the approved files. Make the smallest candidate change that fits the
profile's shipped structured mutator catalog. Do not modify harnesses or
unrelated files. Avoid speculative abstractions and avoid changing public
behavior outside the forecast target.

### 4. Run the same gates for the candidate

Run the exact candidate gate set corresponding to the baseline. Compare only
measured outputs: pass/fail status, forecast accuracy metrics, replay metrics,
calibration metrics, runtime, and regression test results.

### 5. Decide keep or drop

Use fixed gates, not prose confidence:

- Keep: candidate passes all required gates and improves the selected metric
  beyond the profile threshold.
- Drop: candidate fails any required gate, has no measurable lift, increases
  unacceptable runtime, or changes files outside the allowlist.

For a drop decision, revert the candidate changes or instruct the operator to
discard the candidate branch. Do not leave failed experiments as source edits.

### 6. Write the experiment record

When actually executing a run, use `write_file` or `edit_file` only for runtime
artifacts under `.cramer-short/experiments/`, such as:

- `.cramer-short/experiments/forecast-results.tsv`
- `.cramer-short/experiments/runs/<run-id>/manifest.json`
- `.cramer-short/experiments/runs/<run-id>/baseline.json`
- `.cramer-short/experiments/runs/<run-id>/candidate.json`
- `.cramer-short/experiments/runs/<run-id>/decision.json`

Do not write planning notes into the repository.

## Output format

End every forecast-lab response with:

1. Experiment scope — target subsystem and allowed files
2. Baseline — commands and observed metrics, or what must be run first
3. Candidate — exact change attempted or proposed
4. Gates — pass/fail and metric comparison
5. Decision — keep or drop, with the measured reason
6. Artifacts — `.cramer-short/experiments/` paths written or expected
