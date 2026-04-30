# MCP readiness and future implementation guide — 2026-04-30

## Executive summary

**Short answer:** if capture starts now and the system produces **3-5 qualifying BTC replay bundles per day**, the first **honest MCP replay gate** is usually reachable in about:

| Scope | Approx. labeled data target | Practical time from zero |
| --- | ---: | --- |
| BTC-only **7d pilot** | 60-100 labeled bundles | **3-5 weeks** |
| BTC-only **30d gate** | 60-100 labeled bundles | **6-8 weeks** |
| Multi-horizon **7d/14d/30d** go/no-go | 150-250 labeled bundles | **2-3 months** |
| Robust **90d-inclusive** gate | 150-250 labeled bundles with long-horizon coverage | **3-5 months** |

That timing is driven by the replay code itself:

1. a bundle is only labelable after its **forecast horizon** has elapsed,
2. semantic market labels are only ready after the captured Polymarket markets reach their **own end times**,
3. and the MCP gate is only worth running once there are enough labeled rows to compare baseline vs candidate without fooling ourselves on tiny samples.

**Recommended operating assumption:** treat **75 labeled BTC 7d bundles** as the first serious MCP checkpoint, and **150+ labeled bundles** as the first credible keep/drop decision.

---

## What “MCP” means here

In this repo, **MCP** means **Market-Conditioned Prompting**, following the literature review based on Kim et al. [2602.21229].

The core idea is:

1. start from an explicit **prior** probability,
2. provide the model with frozen decision-time evidence,
3. ask it to **revise the prior**, not predict from scratch,
4. and optionally mix the posterior with the original prior using a conservative convex blend.

For this codebase, the natural future MCP variant is:

- **prior:** the Markov/arbiter baseline directional probability or equivalent prior summary
- **evidence:** frozen Polymarket semantics, whale/on-chain summary, and other structured context already present at decision time
- **posterior:** an LLM-produced revised probability plus textual rationale
- **mixture:** conservative blend such as `final = alpha * prior + (1 - alpha) * posterior`, with `alpha` initially biased toward the prior

This is **not** ready to promote live yet. It must first beat the existing arbiter in **offline replay** on real captured rows.

---

## Current repo status

### What is already ready

The replay substrate is now in place:

- `src/tools/finance/arbiter-replay.ts`
  - raw/source row contracts
  - normalized `ArbiterReplayBundle`
  - JSONL persistence helpers
- `src/tools/finance/arbiter-replay-labeler.ts`
  - forecast and semantic label logic
- `src/tools/finance/backtest/arbiter-replay-runner.ts`
  - baseline vs candidate replay
  - abstain-aware acceptance gate

Automatic local capture is now wired at the decision boundaries and writes under:

- `.cramer-short/cache/arbiter-replay/bundles.jsonl`
- `.cramer-short/cache/arbiter-replay/polymarket-raw.jsonl`
- `.cramer-short/cache/arbiter-replay/whale-raw.jsonl`

The current capture hooks are:

- `src/tools/finance/polymarket-forecast.ts`
- `src/tools/finance/onchain-crypto.ts`
- `src/tools/finance/forecast-arbitrator.ts`

### What is still missing before an honest MCP experiment

1. **A sufficiently large labeled replay corpus**
2. **A price-history labeling job or adapter** that feeds real history into the labeler on matured bundles
3. **An MCP candidate evaluator** running in replay/shadow mode only
4. **A chronological holdout evaluation**, not just ad hoc spot checks

---

## What counts as “enough data”

There is no single hard-coded threshold in the repo. The right answer is operational, not ceremonial: you need enough labeled rows that a baseline-vs-candidate comparison is informative in the slices where MCP is supposed to help.

### Recommended thresholds

| Readiness level | Labeled bundles | What it is good for | Not enough for |
| --- | ---: | --- | --- |
| Smoke check | 20-30 | prove the replay/label path works on real rows | any keep/drop decision |
| First pilot | 60-100 | detect obvious MCP failure or obvious promise | robust slice conclusions |
| First serious gate | 150-250 | baseline-vs-MCP keep/drop decision | broad multi-ticker generalization |
| Production-grade tuning | 250+ | alpha/prompt tuning across slices and horizons | n/a |

### Slice coverage matters more than total count

MCP is most likely to help when the baseline is uncertain or evidence is mixed. A dataset of 100 nearly identical rows is weaker than 75 varied rows. For the first serious gate, aim for at least:

- **20+ divergent rows** where evidence sources disagree
- **20+ rows with whale support**
- **30+ rows with rich Polymarket evidence**
- **20+ mid-confidence rows** where the baseline prior is not already extreme

If the corpus does not cover those slices, the replay result will be numerically real but strategically weak.

---

## Why the time-to-data is not just “wait for N rows”

The effective time from zero is approximately:

```text
time_to_ready ≈ labeling_lag + (target_labeled_rows / qualifying_rows_per_day) + slack
```

Where:

- `labeling_lag` is the larger of:
  - the forecast horizon (`horizonDays`)
  - the captured market end time needed for semantic labels
- `qualifying_rows_per_day` means **actual replay-worthy bundles**, not raw tool invocations
- `slack` covers unresolved markets, missing price history, bad captures, and slice imbalance

### Practical example

For a BTC 7d pilot:

- target: **75 labeled bundles**
- capture rate: **3-5 qualifying bundles/day**
- label lag: **~7 days**, sometimes a bit longer if a captured market resolves later

This yields:

- **best-case math:** about **22-32 days**
- **practical expectation:** about **3-5 weeks**

For a 30d gate, the exact same capture rate usually pushes the first honest replay window into **6-8 weeks**.

For a 90d-inclusive gate, even with healthy capture volume, the dominant cost is simply waiting for labels to mature.

---

## Recommended capture strategy

### Phase 1 — BTC-only fast loop

Start with:

- **ticker:** BTC
- **horizon:** 7d first
- **frequency:** 3-5 captures/day
- **goal:** reach 75 labeled rows quickly

Why BTC first:

- it already has the richest Polymarket support,
- whale evidence is more likely to be present,
- and the 7d horizon closes the feedback loop fast enough to learn whether MCP is worth building.

### Phase 2 — widen horizons, not assets

After the 7d pilot:

- add **14d**
- then **30d**

Only widen to more tickers after the BTC replay gate shows the MCP idea is promising. Otherwise the team risks spending months collecting broader data for an idea that should have been dropped early.

### Phase 3 — keep “bad” rows

Do **not** curate away awkward cases:

- low-confidence Polymarket bundles
- no-whale rows
- contradictory evidence
- rows where the arbiter abstains

Those are exactly the rows needed for an honest abstain-aware gate.

---

## What a future MCP candidate should look like

### Candidate contract

The future candidate should be implemented as a replay evaluator that consumes `ForecastArbiterInput` and returns the same `ForecastArbiterResult` shape as the baseline.

That keeps it compatible with:

- `runArbiterReplay(...)`
- `compareReplayEvaluators(...)`

### Minimal candidate design

The first MCP candidate should stay deliberately conservative:

1. **Build prior**
   - derive a scalar prior from the current baseline signal
   - keep the prior visible in metadata for auditability

2. **Build evidence packet**
   - frozen Polymarket selected-market summary
   - frozen semantic interpretation and extracted levels
   - whale direction/confidence/summary
   - structured on-chain context if available

3. **Prompt the LLM to revise**
   - instruct the model to explain why it is moving away from or staying close to the prior
   - forbid it from inventing new live data
   - require a bounded posterior probability

4. **Mix conservatively**
   - begin with a prior-heavy blend
   - do not let the first MCP version fully override the prior

5. **Map posterior into existing arbiter output**
   - verdict
   - preferred direction
   - confidence
   - shouldEnterNow
   - rationale / metadata

### What not to do

Do **not** make the first MCP version:

- a live-only UX feature
- a replacement for the replay gate
- a free-form narrative LLM that bypasses the typed arbiter result
- a prompt that sees fresh web/search data during replay

Replay must stay deterministic and use only the captured bundle.

---

## Future replay protocol once enough data exists

### Step 1 — freeze the corpus

Create a chronological snapshot of matured replay bundles from:

- `.cramer-short/cache/arbiter-replay/bundles.jsonl`

Only include bundles that are truly ready for labeling and replay.

### Step 2 — attach labels

For each matured bundle:

1. fetch or load the relevant historical price path,
2. run `labelReplayBundle(...)`,
3. persist the labeled bundle snapshot used for evaluation.

Important detail from the current code:

- forecast labels require `currentPrice` plus enough history to pass the bundle horizon
- semantic labels require every captured market to reach its own end time

If a bundle is not ready, keep it out of the gate.

### Step 3 — split train vs holdout chronologically

Recommended first split:

- **train/tune:** earliest 70-80%
- **holdout:** latest 20-30%

Use the train slice for:

- prompt wording
- alpha tuning
- threshold tuning

Use the holdout slice only for the final keep/drop decision.

### Step 4 — run baseline vs MCP on identical rows

Use:

- baseline = `arbitrateForecast`
- candidate = MCP evaluator

Both evaluators must receive the exact same `ForecastArbiterInput` rows produced from the labeled bundle snapshot.

### Step 5 — apply the acceptance gate

The current replay comparator already protects against fake wins from over-abstaining. By default it rejects a candidate when:

- directional accuracy does not improve enough
- Brier score regresses
- abstain rate rises by more than the allowed tolerance

Current default tolerances in `compareReplayEvaluators(...)`:

- `maxAbstainRateIncrease = 0.05`
- `maxBrierRegression = 0`
- `minDirectionalAccuracyLift = 0`

For the first MCP gate, a candidate should beat baseline on **at least Brier or directional accuracy** without buying that improvement via abstention.

---

## Recommended future acceptance criteria

For the first serious MCP go/no-go decision, use all of the following:

1. **Holdout only**
   - no cherry-picked examples

2. **No Brier regression**
   - calibration must not worsen

3. **No material abstain inflation**
   - the existing 5 percentage-point tolerance is a good default

4. **At least one meaningful improvement**
   - directional accuracy lift, or
   - clearly better Brier score, especially in uncertain rows

5. **Slice sanity**
   - if MCP only “wins” in trivial rows and loses in divergent or rich-evidence slices, reject it

If those conditions are not met, drop the MCP candidate and keep the baseline arbiter.

---

## Failure modes to watch for

### 1. Data looks large but is strategically thin

Example: 120 labeled rows, but almost all are BTC 7d with no whale support and weak Polymarket evidence.

Result: technically enough for replay, not enough for an MCP decision.

### 2. Semantic label lag is longer than forecast lag

A bundle can pass forecast horizon readiness but still fail semantic readiness because one or more captured Polymarket markets resolve later.

That makes the actual runway longer than the nominal `horizonDays`.

### 3. Replay contamination

If the candidate fetches live data during replay, the experiment is invalid. Replay must use the frozen bundle only.

### 4. Over-tuning alpha or prompt wording

If prompt/alpha choices are tuned on the same rows used for the final verdict, the reported lift will be optimistic.

### 5. Unsupported semantics

`path_dependent` and `unknown` semantics can be captured but are not currently fully scorable beyond `unsupported`. A corpus dominated by those rows weakens the gate.

---

## Recommended next operational milestone

The most efficient next milestone is:

1. run the automatic capture path for **BTC 7d** on a schedule,
2. accumulate toward **75 labeled bundles**,
3. keep the corpus chronological and unfiltered,
4. then implement the smallest possible MCP candidate and replay it offline.

If the capture rate stays around **3-5 good bundles/day**, that first honest BTC MCP decision should be reachable in roughly **3-5 weeks**.

If the system captures materially less than that, expect the schedule to stretch proportionally.

---

## Bottom line

**Enough data for the first real MCP gate is not months away if the team stays BTC-only and 7d-first.** Under normal usage or scheduled shadow capture, the first serious replay window is likely **within 3-5 weeks**.

**Enough data for a broader, decision-grade multi-horizon MCP evaluation is more like 2-3 months**, and a 90d-inclusive gate is closer to **3-5 months**.

Until at least the **75-labeled-bundle BTC 7d checkpoint** is reached, MCP should still be treated as **blocked for promotion** and limited to planning only.
