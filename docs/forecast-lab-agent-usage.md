# Forecast Lab Agent Usage Guide

This guide explains how to use **Forecast Lab through the agent** with plain prompts.

Use it when you want to:

1. plan a bounded forecast experiment,
2. run a real improvement attempt,
3. compare the current best kept run against the shipped baseline,
4. approve and activate a kept result,
5. reset a misleading promoted baseline,
6. or understand why a mutator or lineage state cannot proceed.

For lower-level CLI details and artifact structure, see `docs/forecast-lab-guide.md`.

---

## The most important rule

> A kept structured run is **not live yet** until you explicitly approve promotion.

After approval, the promoted parameters become live:

1. in the current running process,
2. in the updated source defaults for future restarts.

That means ordinary forecast prompts start using the promoted baseline **only after promotion succeeds**.

---

## What Forecast Lab is for

Forecast Lab is for **bounded, auditable forecast improvement**, not ordinary forecasting.

Good uses:

- improve the BTC `1d/2d/3d` Markov workflow,
- compare the latest kept result with the shipped default baseline,
- force one shipped mutator if the lineage allows it,
- promote a kept result to make it live,
- reset back to shipped defaults or the last known-good activated baseline.

Not a Forecast Lab request:

```text
Give me a BTC forecast for the next 7 days.
```

That is a normal forecast query. It uses whatever baseline is currently live.

---

## Mental model

Forecast Lab is an **agent-first bounded loop**:

1. **Plan** the experiment
2. **Improve** the profile
3. **Compare** the best kept run vs the shipped default baseline
4. **Approve** promotion if you want that kept run to become live
5. **Use ordinary forecast prompts**
6. **Reset** if the promoted baseline turns out to be misleading

The main safety property is that the agent should stay inside the bounded Forecast Lab tool flow rather than wandering through local files.

In particular, questions like:

```text
Is the current best better than the shipped default baseline?
```

now route into a dedicated bounded comparison flow instead of falling back into generic filesystem probing.

---

## Quick reference

| Goal | Recommended prompt | What happens |
|---|---|---|
| Plan only | `Optimize the BTC 1d/2d/3d Markov forecast workflow. Do not edit files, run shell commands, or write artifacts; explain the exact experiment plan you would follow.` | Forecast Lab returns the bounded plan only |
| Run improvement | `Improve the BTC 1d/2d/3d Markov forecast workflow.` | Runs the bounded baseline vs candidate path |
| Force a mutator | `Improve the BTC 1d/2d/3d Markov forecast workflow using mutator markov-faster-decay-reaction.` | Runs the bounded improvement path with that requested mutator if allowed |
| List shipped mutator ids | `List the shipped mutator ids for btc-markov-ultra-short-horizon.` | Returns the bounded shipped catalog for that profile without repo or web probing |
| Implement a non-shipped mutator | `implement and run the markov-entropy-adaptive-anchor-weighting` | Returns bounded catalog-extension guidance instead of probing the repo or attempting edits in agent mode |
| Compare best vs shipped | `Is the current best better than the shipped default baseline?` | Returns the latest kept-vs-shipped comparison and tells you whether it is already live |
| Ask how to promote | `How do I promote these improvements?` | Guidance only; does **not** execute promotion |
| Actually promote | `Approve forecast-lab promotion for btc-markov-ultra-short-horizon run <run-id>.` | Verifies and activates the kept run |
| Contextual approval | `I approve forecast-lab promotion for these improvements.` | May execute promotion if the current approval target is unique in context |
| Reset to defaults | `Reset the forecast-lab baseline for btc-markov-ultra-short-horizon to shipped defaults.` | Rolls the live baseline back to repo defaults |
| Reset to last known-good | `Restore the forecast-lab baseline for btc-markov-ultra-short-horizon to the last known good activation.` | Restores the previous activated baseline |

---

## The safest workflow

If you want the most reliable operator flow, use these prompts in order.

### 1. Plan only

```text
Optimize the BTC 1d/2d/3d Markov forecast workflow. Do not edit files, run shell commands, or write artifacts; explain the exact experiment plan you would follow.
```

Expected behavior:

- routes into Forecast Lab,
- stays read-only,
- explains baseline, candidate, gates, and artifacts,
- does not mutate files.

### 2. Run the bounded improvement

```text
Improve the BTC 1d/2d/3d Markov forecast workflow.
```

Expected behavior:

- resolves `btc-markov-ultra-short-horizon`,
- runs the bounded baseline/candidate workflow,
- returns a `keep` or `drop` style result,
- if kept, it remains **approval-required**, not live.

### 3. Compare the latest kept result against shipped defaults

```text
Is the current best better than the shipped default baseline?
```

Expected behavior:

- uses the bounded comparison flow,
- compares the latest kept structured run against the shipped baseline,
- reports whether the result is already live,
- if not live, gives you the exact promotion command to run next.

This is usually the fastest operator check before promotion.

### 4. Promote the kept run explicitly

```text
Approve forecast-lab promotion for btc-markov-ultra-short-horizon run <kept-run-id>.
```

Example:

```text
Approve forecast-lab promotion for btc-markov-ultra-short-horizon run forecast-lab-btc-markov-ultra-short-horizon-2026-05-03T08-32-54-584Z.
```

Expected behavior:

- runs the bounded promotion path,
- re-verifies the kept run,
- activates the parameters,
- reports that the new baseline is live.

### 5. Use regular forecasts normally

After promotion, just ask the normal forecast question:

```text
Give me a BTC forecast for the next 7 days.
```

Or:

```text
What is the BTC 1d/2d/3d forecast now?
```

Expected behavior:

- the system uses the currently active promoted baseline,
- it does **not** rerun Forecast Lab automatically.

### 6. Reset if the promoted baseline misleads

To go back to canonical repo defaults:

```text
Reset the forecast-lab baseline for btc-markov-ultra-short-horizon to shipped defaults.
```

To restore the previous activated baseline:

```text
Restore the forecast-lab baseline for btc-markov-ultra-short-horizon to the last known good activation.
```

---

## Prompt cookbook

### Plan only

```text
Optimize the BTC 1d/2d/3d Markov forecast workflow. Do not edit files, run shell commands, or write artifacts; explain the exact experiment plan you would follow.
```

### Run a bounded improvement

```text
Improve the BTC 1d/2d/3d Markov forecast workflow.
```

### Force one shipped mutator

```text
Improve the BTC 1d/2d/3d Markov forecast workflow using mutator markov-faster-decay-reaction.
```

Important:

- the mutator name is now routed correctly in agent mode,
- but Forecast Lab can still reject it if it is already used or no longer applicable after replaying the current kept lineage.

This means:

> "Try this exact shipped mutator if the current lineage allows it."

It does **not** mean:

> "Ignore lineage safety and force the edit anyway."

### List the shipped mutator ids

Use this when you want the current shipped candidate ids for one profile:

```text
List the shipped mutator ids for btc-markov-ultra-short-horizon.
```

Expected behavior:

- the agent stays inside the bounded Forecast Lab tool flow,
- it returns the shipped candidate catalog ids for that profile,
- it also tells you the allowed structured operator ids for that profile,
- it does **not** search the web or probe the repo to discover the catalog.

If you omit the profile in a Forecast Lab context, the agent can fall back to the current profile or summarize every structured profile.

### Request a new non-shipped mutator implementation

Use this when the mutator does **not** exist in the shipped catalog yet:

```text
implement and run the markov-entropy-adaptive-anchor-weighting
```

Expected behavior:

- the agent stays inside the bounded Forecast Lab tool flow,
- it returns the **catalog-extension plan** for the target profile,
- it points you to the catalog files and validation files that must be updated first,
- it does **not** treat the request as permission to read arbitrary repo files, browse the web, or attempt source edits on its own.

Practical meaning:

> "This mutator is not shipped yet. Tell me the bounded code-change path first."

It does **not** mean:

> "The agent should improvise after an unknown-mutator error and start exploring or editing the repo."

Once the new mutator actually exists in the shipped catalog and profile contract, rerun the normal Forecast Lab improvement flow.

### Compare current best vs shipped baseline

Use this when you want the answer without manually inspecting artifacts:

```text
Is the current best better than the shipped default baseline?
```

You can also be more explicit:

```text
Compare the current best kept btc-markov-ultra-short-horizon run against the shipped default baseline.
```

Expected behavior:

- Forecast Lab compares the latest kept structured run for that profile,
- summarizes the shipped-default vs kept-run deltas,
- tells you whether the kept run is already live,
- gives you the promotion command if it is not live.

### Ask how to promote without executing promotion

If you want instructions only, ask a guidance question:

```text
How do I promote these improvements?
```

Or:

```text
What is the exact promotion command for the latest kept btc-markov-ultra-short-horizon run?
```

Expected behavior:

- the agent explains the promotion step,
- it does **not** execute promotion just because you used the word "promote".

### Actually promote

The safest wording is explicit:

```text
Approve forecast-lab promotion for btc-markov-ultra-short-horizon run <kept-run-id>.
```

Example:

```text
Approve forecast-lab promotion for btc-markov-ultra-short-horizon run forecast-lab-btc-markov-ultra-short-horizon-2026-05-03T08-32-54-584Z.
```

### Contextual shorthand approval

This can work when the recent conversation already contains one unique approval-required kept run:

```text
I approve forecast-lab promotion for these improvements.
```

Or:

```text
Yes, promote that kept run.
```

Use shorthand only when:

1. the conversation is already about one specific kept run,
2. there is no ambiguity about which run should be promoted.

If you want the most reliable behavior, prefer the explicit prompt with the full run id.

### Reset to shipped defaults

```text
Reset the forecast-lab baseline for btc-markov-ultra-short-horizon to shipped defaults.
```

### Reset to the last known-good activated baseline

```text
Restore the forecast-lab baseline for btc-markov-ultra-short-horizon to the last known good activation.
```

### Ask what to do when the lineage is exhausted

```text
Use the forecast-lab skill to explain what to do when no shipped structured mutator remains applicable after replaying the kept parent lineage for btc-markov-ultra-short-horizon.
```

That is the right prompt when you see errors like:

```text
No shipped structured mutator remains applicable after replaying the kept parent lineage...
```

Typical next actions are:

1. keep the current best candidate,
2. request a new shipped mutator,
3. reset the lineage or live baseline if you intentionally want to restart.

---

## Ask vs act

This distinction matters a lot in agent mode.

| Prompt shape | Meaning |
|---|---|
| `How do I promote these improvements?` | guidance only |
| `What is the exact promotion command?` | guidance only |
| `Approve forecast-lab promotion for ...` | execute promotion |
| `I approve forecast-lab promotion for these improvements.` | execute promotion if the target is uniquely implied by context |

Recommended rule:

> If you want the agent to **act**, start with `Approve`, `Reset`, `Restore`, or another direct imperative. If you want the agent to **explain**, ask `How`, `What`, or `Which`.

---

## How to read the results

### Improvement result

Typical outcomes:

- **keep / approval required**
- **drop**
- **mutator not applicable**
- **no shipped mutator remains applicable**

If you get **approval required**, the next step is the explicit approval prompt.

If you get **mutator not applicable**, the request was understood, but the requested shipped mutation cannot be applied safely in the current lineage.

If you get **no shipped mutator remains applicable**, treat that as an exhausted-lineage state and switch to guidance/reset instead of retrying blindly.

### Comparison result

A comparison answer should tell you:

1. which kept run was compared,
2. whether it beats the shipped baseline,
3. whether that kept run is already live,
4. the exact promotion command if it is not live.

This is usually the best answer to questions like:

```text
Is the current best better than the shipped default baseline?
```

### Promotion result

A successful promotion should tell you that:

- verification passed,
- activation succeeded,
- the promoted parameters are now live.

### Reset result

A successful reset should tell you:

- which reset mode was used,
- where the reset artifact was written,
- whether a promoted baseline is still active afterward.

---

## Promotion behavior you should expect now

### Promotion is explicit

Promotion does not happen automatically after a keep.

You must explicitly approve it.

### Older kept runs can still be promoted

Forecast Lab now repairs older kept source manifests during bounded promotion if they are missing newer persisted promotion metadata.

Practical meaning:

- if the kept run is still valid in the ledger,
- and its structured replay payload is intact,
- the promotion path can repair the missing promotion block and continue.

This matters for older approval-required runs that were created before the newer manifest metadata was always persisted.

### Comparison answers are bounded

Questions about whether the current best is better than the shipped baseline now use the bounded comparison flow.

Practical meaning:

- ask the question directly,
- do not manually inspect `.cramer-short/experiments` unless you actually want artifact-level evidence,
- the agent should stay inside the Forecast Lab tool flow.

### Non-shipped mutator implementation requests are also bounded

Requests such as:

```text
implement and run the markov-entropy-adaptive-anchor-weighting
```

now stay in the bounded catalog-extension path when the named mutator is not part of the shipped catalog yet.

Practical meaning:

- Forecast Lab returns the catalog-extension guidance,
- names the catalog and validation files to change,
- and stops there until the new mutator is implemented and allowed by the profile contract.

It should **not** fall through into generic `read_file`, web search, or edit attempts after the unknown-mutator result.

---

## Where the evidence lives

Forecast Lab artifacts are written under:

```text
.cramer-short/experiments/runs/<run-id>/
```

The active promoted baseline is tracked under:

```text
.cramer-short/experiments/active-promotions/<profile-id>.json
```

Most useful files:

1. `decision.json`
2. `candidate.json`
3. `baseline.json`
4. `manifest.json`

If you only inspect one file first, inspect `decision.json`.

If you are checking whether a promotion is currently live, inspect:

```text
.cramer-short/experiments/active-promotions/btc-markov-ultra-short-horizon.json
```

---

## Practical troubleshooting

### "No shipped structured mutator remains applicable"

Meaning:

- the lineage is exhausted for the currently shipped catalog.

What to do:

1. inspect the latest run artifacts if you need evidence,
2. decide whether the current best candidate should remain the best known result,
3. reset to defaults or last known-good if you want to restart from a cleaner live state,
4. add a new shipped mutator in code only if you actually want to extend the catalog.

### "Mutator is not applicable after replaying the kept parent lineage"

Meaning:

- the prompt was understood,
- but the requested mutator cannot be safely applied in the current lineage.

What to do:

1. try a different shipped mutator,
2. reset the lineage if you intentionally want a restart,
3. or stop and keep the current best candidate.

### "Unknown mutator" or "the mutator does not exist in the catalog"

Meaning:

- the requested mutator is not part of the currently shipped Forecast Lab catalog,
- so there is nothing safe to execute yet in the bounded mutation runner.

What to do:

1. ask for the bounded catalog-extension plan,
2. update the shipped mutator catalog and profile contract in code,
3. update the focused validation tests,
4. only then rerun the bounded improvement flow.

Current agent expectation:

- prompts like `implement and run the markov-entropy-adaptive-anchor-weighting` should return catalog-extension guidance,
- they should **not** trigger repo exploration or direct source-edit attempts just because the mutator name was recognized.

### "Promotion source was not found"

Meaning:

- the agent got an approval-style prompt,
- but there is no current approval-required kept run matching what you asked for.

What to do:

1. rerun the improvement flow or comparison flow,
2. copy the exact `runId` returned by Forecast Lab,
3. retry the explicit approval prompt.

### "Missing promotion metadata"

This used to block promotion for some older kept runs.

Current expectation:

- bounded promotion should repair that source manifest automatically if the kept run is still promotable,
- if promotion still fails, the more likely cause is missing replay payload or inconsistent run metadata rather than the old missing-promotion-block bug.

### Reset fails

Meaning:

- there is no active promoted baseline for that profile,
- or there is no previous activated baseline to restore for `last-known-good`.

What to do:

1. confirm whether the profile currently has an active promotion record,
2. use `shipped defaults` reset if you only need to go back to repo defaults,
3. use `last known good` only when a previous activated baseline actually exists.

---

## Recommended copy/paste workflow

```text
1. Optimize the BTC 1d/2d/3d Markov forecast workflow. Do not edit files, run shell commands, or write artifacts; explain the exact experiment plan you would follow.

2. Improve the BTC 1d/2d/3d Markov forecast workflow.

3. Is the current best better than the shipped default baseline?

4. Approve forecast-lab promotion for btc-markov-ultra-short-horizon run <kept-run-id>.

5. Give me a BTC forecast for the next 7 days.

6. If needed: Reset the forecast-lab baseline for btc-markov-ultra-short-horizon to shipped defaults.
```

If you want a specific mutator:

```text
Improve the BTC 1d/2d/3d Markov forecast workflow using mutator markov-faster-decay-reaction.
```

If you only want promotion instructions:

```text
How do I promote these improvements?
```

---

## One-line summary

> Use Forecast Lab through the agent for bounded forecast improvement, use comparison prompts to check kept results against shipped defaults, treat non-shipped mutator implementation requests as catalog-extension guidance first, approve kept runs explicitly to make them live, then use normal forecast prompts afterward, and reset explicitly when the active promoted baseline is no longer the one you want.
