# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This repository uses [`AGENTS.md`](./AGENTS.md) as the canonical agent instruction file. Follow AGENTS.md for coding style, change discipline, and verification practices. This file supplements it with commands and architecture context.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:
```
1. [Step] → verify: [check]
2. [Step] → verify: [check]
3. [Step] → verify: [check]
```

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

## Commands

### TypeScript (primary runtime)

```bash
bun install                  # postinstall runs playwright install chromium
bun start                    # default TUI entry (src/index.tsx)
bun run dev                  # watch mode
bun run typecheck            # tsc --noEmit
bun run depcruise            # dependency boundary enforcement (.dependency-cruiser.cjs)
bun test                     # unit tests only (*.test.ts, excludes *.integration.test.ts and *.e2e.test.ts)
bun test ./src/foo.test.ts   # single test file
bun test -t "case name"      # filter by test name
bun run test:watch           # watch mode for unit tests
bun run test:integration     # integration tier (real APIs, no LLM; isolated process via scripts/run-isolated-bun-tests.ts)
bun run test:e2e             # e2e tier (full agent + Ollama; each file in its own isolated process)
bun run test:all             # all three tiers in sequence
bun run test:coverage        # coverage gate on targeted files + scripts/check-coverage.ts
bun run gateway              # WhatsApp gateway
bun run gateway:login        # WhatsApp link/QR
```

Integration and E2E tests are excluded from `bun test` by `bunfig.toml` path ignore patterns. They run through `scripts/run-isolated-bun-tests.ts` to avoid `mock.module()` contamination from unit tests.

### Python research mirror

```bash
conda env create -f environment-research.yml   # creates cramer-research env (Python 3.12)
conda activate cramer-research
cd research && pip install -e .                # editable install
pytest research/tests/ -v
```

Code style: `black research/` (line-length 100), `ruff check research/`, `mypy research/`.

### Release

```bash
bash scripts/release.sh [version]
```

Requires `gh` CLI, `node`, and a clean working tree. Interactive; defaults to calendar version `YYYY.M.D`. Creates version-bump commit, tag, push, and GitHub release.

## High-level architecture

### Dual-language codebase

The repository is **Bun-first TypeScript** for the production CLI/TUI, with a **Python research mirror** under `research/` for interactive analysis and parameter tuning. Sixteen TypeScript modules in `src/tools/finance/` and `src/utils/finance/` have Python counterparts in `research/models/` and `research/utils/`. `research/PARITY.md` is the authoritative manifest of which TS files mirror which Python files, their parity-test status, and known divergences. When adding a new mirrored TS module, add a `Mirrors research/models/X.py` docstring tag and update `PARITY.md`.

### Entry routing

`src/index.tsx` is the single entry point. It detects subcommands via `src/index-routing.ts` and dispatches:

- default / `cli` → `runCli()` (`src/cli.ts`) for the interactive TUI
- `schedule` → `src/cli-schedule.ts` (headless scheduled jobs; reads `~/.cramer-short/schedules.json`)
- `lab` → `src/cli-forecast-lab.ts` (Forecast Lab CLI)
- `replayLabel` → `src/cli-replay-label.ts`

All headless paths share the same browser-lifecycle cleanup on shutdown.

### Layer boundaries and dependency direction

Preferred dependency direction (enforced by `depcruise` in CI):

```
entrypoints → controllers/components → agent → tools/experiments → model/utils/memory
```

Key forbidden edges:
- `src/tools/` must not import from `src/(cli|components|controllers)/` (error)
- `src/agent/` must not import from `src/(cli|components|controllers)/` (error)
- `src/utils/` importing from `src/(agent|tools|cli|components|controllers)/` is a warn (existing session/export helpers still depend on app types)
- Circular dependencies are warned but accepted in some existing cases

New layer violations must be justified; 32 warnings currently exist as accepted design tradeoffs.

### Data flow through the agent

```
User input
  → src/index.tsx → runCli() / schedule runner
  → AgentRunnerController (src/controllers/)
  → Agent.run() (src/agent/agent.ts)
  → query routing (src/agent/query-router/) + memory/polymarket context injection
  → model call (src/model/)
  → tool calls through AgentToolExecutor (src/agent/tool-executor.ts)
  → RunContext scratchpad (src/agent/run-context.ts) + optional memory flush
  → answer-formatting guards (src/agent/answer-formatting/)
  → TUI / session store / export
```

### Runtime state locations

All runtime state lives under `.cramer-short/` (hardcoded in `src/utils/paths.ts`). Never use `.dexter/` for runtime state.

Key paths:
- `.cramer-short/settings.json` — user config (Zod-validated; unknown keys preserved, invalid known fields stripped with a warning)
- `.cramer-short/SOUL.md` — user-specific system prompt override (loaded before repo `SOUL.md`)
- `.cramer-short/skills/` — project skill overrides (shadow builtins by name)
- `.cramer-short/sessions/` — auto-saved conversation history
- `~/.cramer-short/schedules.json` — scheduled jobs (not project-local)

### Provider and tool registry

`src/providers.ts` is the canonical LLM provider registry. Default model is `gpt-5.4` via OpenAI. Provider routing is prefix-based (e.g., `claude-` → Anthropic, `gemini-` → Google).

`src/tools/registry.ts` is the canonical tool registry. Tool availability is env-gated:
- `web_search` registers when `EXASEARCH_API_KEY` | `PERPLEXITY_API_KEY` | `TAVILY_API_KEY` is set (in that priority order)
- `x_search` requires `X_BEARER_TOKEN`
- `skill` only when skills are discovered

All `z.string()` tool parameters must carry `.max()` validators. New tools must include input length limits.

### Forecast Lab and experiments

Forecast Lab logic lives in `src/experiments/forecast-lab/` (router, profiles, mutators, runner). The agent-facing query-router for Forecast Lab is in `src/experiments/forecast-lab/query-router.ts`. Keep profile/routing logic in that directory; compatibility re-exports from other locations are acceptable during incremental moves.

### Markov distribution decomposition

`src/tools/finance/markov-distribution.ts` was decomposed into `src/tools/finance/markov-distribution/` with 8 sub-modules (regime, transition, blending, CI, diagnostics, core, asset-profile, live-policies). Import from the specific sub-module rather than the top-level barrel unless you need everything.

## Important conventions

- **TypeScript is strict ESM**. Use explicit `.js` extensions in local imports. `@/*` resolves to `src/*` via `tsconfig.json`.
- **Calendar versioning**: `YYYY.M.D`, no zero padding (e.g., `2026.3.25`).
- **Config validation** (`src/utils/config.ts`) never crashes startup. It preserves unknown keys and strips invalid known fields with a warning to stderr.
- **Test tiers**: use `integrationIt` and `e2eIt` from `src/utils/test-guards.ts` instead of raw `it` for integration/E2E tests. Unit tests mock `../model/llm.js`, so E2E must run in isolated processes to avoid fake-LLM contamination.
- **SOUL.md load order**: `.cramer-short/SOUL.md` first, then repo `SOUL.md`. User overrides take priority.
- **Skill overrides**: `.cramer-short/skills/` shadow builtins by name via `src/skills/registry.ts`.
- **Schedule mode**: reads from `~/.cramer-short/schedules.json` (not project-local). Job `outputFile` supports `~` and `{date}`.
- **Postinstall**: `bun install` runs `playwright install chromium`. CI skips this via `--ignore-scripts`.
- **Deleted/relocated types**: `src/types.ts` was deleted in Phase 1 refactoring. `HistoryItem`, `WorkingState`, and `HistoryItemStatus` now live in `src/controllers/types.ts`.
