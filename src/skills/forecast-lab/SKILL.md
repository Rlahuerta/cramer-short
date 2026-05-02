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

The default editable allowlist is narrow:

- `src/tools/finance/markov-distribution.ts`
- `src/tools/finance/polymarket-forecast.ts`
- `src/tools/finance/forecast-arbitrator.ts`

Additional files are read-only unless the user explicitly names another forecast
file path as editable for that run.

Read-only harnesses for acceptance include:

- `src/tools/finance/backtest/walk-forward.ts`
- `src/tools/finance/backtest/arbiter-replay-runner.ts`
- existing targeted unit, integration, or E2E tests chosen by the profile

If the user asks to edit outside the allowlist, refuse that part and explain that
forecast-lab is intentionally bounded.

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

If code mutation is authorized, inspect only the approved files with `read_file`
and make the smallest candidate change with `edit_file`. Do not modify harnesses
or unrelated files. Avoid speculative abstractions and avoid changing public
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
