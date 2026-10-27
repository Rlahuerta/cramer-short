# PI-25: Negated or questioning "use the X skill" requests still trigger the intent

- Severity: M
- Area: C4
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/agent/query-router/classification.ts:142` — `const match = query.match(/\buse (?:the )?([a-z0-9_-]+) skill\b/i);`, `src/agent/query-router/classification.ts:54` — `if (/\buse the\s+probability_assessment\s+skill\b/i.test(query)) return false;`, `src/agent/agent.ts:364` — `const explicitlyRequestedSkill = detectExplicitSkillRequest(query);`, `src/agent/agent.ts:375` — `for await (const event of this.toolExecutor.executeTool(`, `src/agent/agent.ts:388` — `runState.sequentialThinkingUsed = true;`
- Repro kind: runtime

## Summary

Two intent matchers in `classification.ts` treat the literal phrase `use [the] <name> skill` as an explicit request without checking whether it is negated or asked as a question. `detectExplicitSkillRequest("Don't use the portfolio_risk skill for AAPL")` returns `portfolio_risk`, and `Agent.run()` then pre-executes that skill, injects its workflow into the prompt, and marks `sequentialThinkingUsed` so the mandatory planning gate is skipped — the exact opposite of the user's instruction. The sibling exclusion in `isCryptoForecastQuery()` has the same omission: a negated or questioning mention of the `probability_assessment` skill disables the entire crypto forced-forecast pipeline for an otherwise plain BTC forecast query.

## Root cause

`detectExplicitSkillRequest` (`classification.ts:141-148`) matches `/\buse (?:the )?([a-z0-9_-]+) skill\b/i` against the whole query and resolves the captured name against `discoverSkills()`. There is no negation or question cue check, so `don't use`, `do not use`, `never use`, and `Should I use …?` all match. `Agent.run()` consumes the result unconditionally at `agent.ts:364` and pre-executes the skill at `agent.ts:374-389`; after the skill runs it sets `runState.sequentialThinkingUsed = true` (`agent.ts:388`), so the first-tool-call planning enforcement is bypassed for the rest of the run.

`isCryptoForecastQuery` (`classification.ts:50-64`) likewise bails out for any query containing `use the probability_assessment skill` (`classification.ts:54`), again without a negation check. A query that explicitly declines the skill is therefore misclassified as a non-crypto-forecast query, so `isCryptoForecastPipelineQuery` (and every `shouldForceCryptoForecastTools` gate built on it) is false.

The repository already established the intended semantics for negated or questioning routing requests in `87dd643` (`fix(forecast-lab): ignore negated or questioning promotion/reset requests`), which added a `FORECAST_LAB_NEGATION_OR_QUESTION` guard. These two matchers were not covered.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-25-skill-intent-ignores-negation.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-25-skill-intent-ignores-negation.ts
```

Expected: negated/questioned skill mentions resolve no skill; crypto forecast routing is unchanged

Actual: detected skills: negated (don't)=portfolio_risk, negated (do not)=portfolio_risk, question=portfolio_risk disabled crypto routing: negated skill mention, questioned skill mention

OBSERVED:
    affirmative: detectExplicitSkillRequest(...) = "portfolio_risk" (expected a skill)
    negated (don't): detectExplicitSkillRequest(...) = "portfolio_risk" (expected null)
    negated (do not): detectExplicitSkillRequest(...) = "portfolio_risk" (expected null)
    question: detectExplicitSkillRequest(...) = "portfolio_risk" (expected null)
    plain forecast: isCryptoForecastQuery(...) = true (expected true)
    negated skill mention: isCryptoForecastQuery(...) = false (expected true)
    questioned skill mention: isCryptoForecastQuery(...) = false (expected true)

## Impact

A user who writes "don't use the portfolio_risk skill, just summarise the risk" gets the full portfolio-risk workflow force-loaded before the model's first turn, together with a disabled sequential-thinking gate; the answer is shaped by a skill the user explicitly declined. On the crypto path, "don't use the probability_assessment skill — BTC 7-day forecast" silently disables the forced market-data/Markov/Polymarket/arbiter pipeline, so the run depends entirely on the model volunteering the tools. Both are silent: no warning or error is surfaced, and the run still ends with a normal answer.

## Suggested fix direction (not applied)

Add one shared cue test in `src/agent/query-router/classification.ts`, mirroring `FORECAST_LAB_NEGATION_OR_QUESTION` in `src/experiments/forecast-lab/query-router.ts` (e.g. `const NEGATION_OR_QUESTION = /\b(?:no|not|never|don'?t|do not|avoid|skip)\b|n['’]t\b|\?/i;`).

- `classification.ts:142` — current: `const match = query.match(/\buse (?:the )?([a-z0-9_-]+) skill\b/i);` → proposed: `if (NEGATION_OR_QUESTION.test(query)) return null; const match = query.match(/\buse (?:the )?([a-z0-9_-]+) skill\b/i);`
- `classification.ts:54` — current: `if (/\buse the\s+probability_assessment\s+skill\b/i.test(query)) return false;` → proposed: `if (!NEGATION_OR_QUESTION.test(query) && /\buse the\s+probability_assessment\s+skill\b/i.test(query)) return false;`

Keep the affirmative path unchanged (an affirmative `use the portfolio_risk skill` must still resolve the skill). A narrower cue test scoped to the matched phrase is also acceptable, but the whole-query guard is the minimal change consistent with `87dd643`.

## Evidence

- Reproduction script: `issues/repro/PI-25-skill-intent-ignores-negation.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-9-C4.txt`.
