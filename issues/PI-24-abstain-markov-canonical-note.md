# PI-24: Canonical-Markov prompt guard fires for abstained payloads

- Severity: M
- Area: C4
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/agent/prompts.ts:563` — `const hasCanonicalMarkovOutput = /"_tool"\s*:\s*"markov_distribution"/.test(fullToolResults);`, `src/agent/prompts.ts:567` — `Use the reported scenario bucket probabilities and diagnostics verbatim.`, `src/agent/prompts.ts:573` — `if (hasCanonicalMarkovOutput && requestedBucketMatch) {`, `src/agent/prompts.ts:576` — `Preserve that ${requestedBucketMatch[1]}-part bucket granularity in the final answer when presenting the canonical Markov distribution.`
- Repro kind: runtime

## Summary

`buildIterationPrompt()` decides that a canonical Markov payload is present by matching only `"_tool":"markov_distribution"` in the tool-result text, without checking the payload's `status`. For an abstained run (`status: "abstain"`, `canonical.scenarios: null`, `distribution: null`) it still appends "Use the reported scenario bucket probabilities and diagnostics verbatim", and when the query names N parts/buckets it also appends "Preserve that N-part bucket granularity in the final answer when presenting the canonical Markov distribution". The same prompt simultaneously carries the abstain guard stating that no calibrated distribution is available and that synthesizing replacement scenario buckets is forbidden, so a financial answer path receives two contradictory authoritative instructions about a distribution that does not exist.

## Root cause

At `prompts.ts:563` the guard is `/_tool"\s*:\s*"markov_distribution"/` — the tool name alone. The sibling abstain guard is computed later at `prompts.ts:624-626` from the same text plus `"status":"abstain"`, but it is not used to suppress the distribution-specific sentences. Consequently:

- `prompts.ts:567` tells the model to use "the reported scenario bucket probabilities … verbatim" even though `canonical.scenarios` and `data.distribution` are both `null`.
- `prompts.ts:573-576` keys the bucket-granularity instruction off the same `hasCanonicalMarkovOutput` flag, so "preserve that 5-part bucket granularity … when presenting the canonical Markov distribution" is emitted for an abstained payload.

The answer formatter already has the correct behaviour: `ensureStructuredDensityTable()` returns no table when the newest matching Markov payload is not `status: "ok"` (`src/agent/answer-formatting/density-table.ts:101`), so the prompt and the formatter disagree about whether a distribution exists. The proxy/anchor paragraph in the same guard (`prompts.ts:569`) is valid for abstained payloads (diagnostics still exist) and is intentionally covered for that case by `src/agent/prompts.test.ts:212-215`; only the distribution-specific clauses are wrong.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-24-abstain-markov-canonical-note.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-24-abstain-markov-canonical-note.ts
```

Expected: abstained payload: scenario-verbatim = false; bucket-granularity = false; abstain guard = true; proxy guard = true ok payload: scenario-verbatim = true; bucket-granularity = true

Actual: abstained payload: scenario-verbatim = true; bucket-granularity = true (no distribution exists)

OBSERVED:
    abstained payload: scenario-verbatim instruction = true
    abstained payload: 5-part bucket-granularity instruction = true
    abstained payload: abstain guard = true
    abstained payload: proxy/anchor guard = true
    ok payload: scenario-verbatim instruction = true
    ok payload: 5-part bucket-granularity instruction = true

## Impact

The model is instructed both to reproduce a canonical bucket distribution verbatim and that no distribution exists and bucket synthesis is forbidden. On the weaker instruction-following models this is exactly the path that produces fabricated scenario percentages — the failure mode the abstain guard (`prompts.ts:645-651`) and `ensureStructuredDensityTable` were added to prevent. The bucket-granularity sentence is the sharper case: it explicitly demands that a bucket split be preserved in an answer that must not contain one.

## Suggested fix direction (not applied)

Compute the abstain flag before the canonical block and gate only the distribution-specific clauses on it.

- Move the `hasAbstainingMarkovOutput` definition (`prompts.ts:624-626`) above `prompts.ts:563`.
- `prompts.ts:563` — current: `const hasCanonicalMarkovOutput = /"_tool"\s*:\s*"markov_distribution"/.test(fullToolResults);` → proposed: keep the variable, and immediately after it add `const hasNonAbstainingMarkovOutput = hasCanonicalMarkovOutput && !hasAbstainingMarkovOutput;`
- `prompts.ts:567` — split the paragraph: emit "Treat the canonical Markov payload as authoritative." and the final "Do NOT recompute, sum, or infer scenario probabilities …" sentence for `hasCanonicalMarkovOutput`, but emit "Use the reported scenario bucket probabilities and diagnostics verbatim." only when `hasNonAbstainingMarkovOutput`.
- `prompts.ts:573` — current: `if (hasCanonicalMarkovOutput && requestedBucketMatch) {` → proposed: `if (hasNonAbstainingMarkovOutput && requestedBucketMatch) {`

Leave the proxy/anchor paragraph and the standalone abstain guard unchanged so the existing abstain diagnostics guidance still fires.

## Evidence

- Reproduction script: `issues/repro/PI-24-abstain-markov-canonical-note.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-9-C4.txt`.
