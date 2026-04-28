# Copilot Instructions for Cramer-Short

Cramer-Short is a Bun-first TypeScript CLI/TUI for financial research. When prose and code disagree, trust `package.json`, CI commands, and the source.

## Commands

```bash
bun install                  # installs deps; postinstall also patches pi-tui and installs Playwright Chromium
bun start                    # launch the interactive TUI
bun start schedule list      # list scheduled jobs from ~/.cramer-short/schedules.json
bun start schedule run       # run all scheduled jobs headlessly
bun start schedule run JOB   # run one scheduled job
bun run dev                  # watch mode
bun run typecheck            # repo-wide static check

bun test                     # unit tests
bun test src/path/to/file.test.ts
bun test -t "case name"
bun run test:integration     # integration tier
bun run test:e2e             # curated E2E list with explicit timeouts
bun run test:all
```

CI uses:

```bash
bun install --frozen-lockfile --ignore-scripts
bun run typecheck
bun test
```

## High-level architecture

- `src/index.tsx` is the entrypoint. It loads `.env`, then routes to the interactive CLI (`runCli`) or the headless schedule runner (`runScheduleCommand`) based on the first subcommand.
- `src/cli.ts` owns the terminal application: slash commands, watchlist/session screens, exports, model selection, and handing user queries off to the agent.
- `src/agent/agent.ts` is the core execution loop. `Agent.create()` builds the system prompt from the active model, `SOUL.md`, discovered tools, and persisted memory/session context. `Agent.run()` injects relevant memory and Polymarket context, enforces `sequential_thinking` as the first tool, executes tools, compacts old tool output into summaries when context gets large, periodically flushes findings to memory, and produces a best-effort answer when it hits the iteration cap.
- `src/tools/registry.ts` is the canonical tool registry. Core finance, filesystem, browser, and memory tools are always registered; `web_search` is enabled only when one of `EXASEARCH_API_KEY`, `PERPLEXITY_API_KEY`, or `TAVILY_API_KEY` is present (in that order); `x_search` is enabled only with `X_BEARER_TOKEN`; `skill` is enabled only if any skills are discovered.
- Persistence is split across two systems:
  - `src/controllers/session-controller.ts` and `src/utils/session-store.ts` persist `.cramer-short/sessions/` in a two-layer format: compact `llmMessages` for resume plus full `history` for terminal scrollback.
  - `src/memory/` manages long-term memory in both Markdown files under `.cramer-short/memory/` and the SQLite/FTS/vector-backed database used by memory search.
- Prompt and skill assembly are separate from the agent loop:
  - `src/agent/prompts.ts` loads `.cramer-short/SOUL.md` first, then falls back to the repo `SOUL.md`.
  - `src/skills/registry.ts` loads built-in skills and then project-local overrides from `<cwd>/.cramer-short/skills/`.
- `src/providers.ts` is the single provider registry. Model routing is prefix-based (`claude-`, `gemini-`, `ollama:`, `openrouter:`, etc.); models with no recognized prefix fall back to OpenAI.

## Key conventions

- Runtime state lives under `.cramer-short/`, never `.dexter/`. Most state is project-local under the current working directory, but scheduled jobs are configured separately at `~/.cramer-short/schedules.json`.
- TypeScript is strict ESM. Use explicit `.js` extensions in local imports. The `@/*` alias resolves to `src/*`.
- `src/utils/config.ts` preserves unknown keys in `.cramer-short/settings.json`, strips invalid known fields with a warning instead of crashing, and upgrades deprecated `modelId` values on load.
- Tests are tiered by filename and runtime guards:
  - `*.test.ts` = unit
  - `*.integration.test.ts` = integration
  - `*.e2e.test.ts` = end-to-end
- Integration tests run locally by default unless `SKIP_INTEGRATION=1`; E2E tests always require explicit `RUN_E2E=1` because they must avoid unit-test mock contamination.
- `bun run test:e2e` is not a glob. It is a manually curated list in `package.json`, so new E2E files must be added there or they will never run in the standard E2E command.
- Skill overrides are project-local. The current code looks for overrides in `<cwd>/.cramer-short/skills/`, not in a home-directory skills folder.
- `bun install` runs repository-specific postinstall behavior (`patches/pi-tui-terminal-fix.js` and `playwright install chromium`). CI skips that with `--ignore-scripts`, so do not assume browser binaries were installed there.
