# Architecture

Cramer-Short is a Bun-first TypeScript CLI/TUI for financial research. Runtime state is stored under `.cramer-short/`; scheduled jobs are configured at `~/.cramer-short/schedules.json`.

## Module map

```text
src/index.tsx
  ├─ cli.ts                         interactive TUI entry
  │  ├─ controllers/                session, agent runner, input/model/watchlist state
  │  ├─ controllers/slash-commands/ incremental slash command handlers
  │  └─ components/                 pi-tui view components
  ├─ cli-schedule.ts                headless scheduled jobs
  └─ cli-forecast-lab.ts            Forecast Lab CLI entry

src/agent/
  ├─ agent.ts                       core loop and tool orchestration
  ├─ query-router.ts                finance/tool routing heuristics
  ├─ answer-formatting/             final-answer guards, warnings, sources
  ├─ prompts.ts                     system/iteration prompt assembly
  ├─ run-context.ts                 per-run scratchpad and context state
  └─ tool-executor.ts               tool dispatch and circuit breaker integration

src/tools/
  ├─ registry.ts                    canonical tool registry and env gating
  ├─ finance/                       market data, forecasting, risk tools
  ├─ fetch/, filesystem/, memory/   non-finance tools
  └─ search/, osint/, browser/      optional external-data tools

src/experiments/forecast-lab/
  ├─ router.ts                      low-level profile scoring
  ├─ query-router.ts                agent-facing Forecast Lab intent/routing hints
  ├─ profiles.ts                    profile and mutator catalog
  └─ runner.ts                      command execution for lab workflows

src/memory/
  ├─ store.ts                       Markdown memory files
  ├─ database.ts                    SQLite/FTS/vector storage
  ├─ auto-store.ts, flush.ts        automatic persistence from agent runs
  └─ dream.ts                       periodic consolidation
```

## Data flow

```text
User input
  → src/index.tsx
  → runCli() / schedule runner
  → AgentRunnerController
  → Agent.run()
  → query routing + memory/polymarket context injection
  → model call
  → tool calls through AgentToolExecutor
  → RunContext scratchpad + optional memory flush
  → answer-formatting guards and source footer
  → TUI/session store/export
```

## Layering guidance

Preferred dependency direction:

```text
entrypoints → controllers/components → agent → tools/experiments → model/utils/memory
```

Guidelines:

- Keep TUI rendering in `components/` or `cli.ts`; put reusable command behavior in `controllers/`.
- Keep agent final-answer shaping in `agent/answer-formatting/`, not in the core loop.
- Keep Forecast Lab profile/routing logic in `experiments/forecast-lab/`; compatibility re-exports are acceptable during incremental moves.
- Keep finance math in focused `tools/finance/` submodules and re-export moved public helpers from the original module when needed.
- Avoid tools importing from controllers or agent internals. Prefer passing data as tool arguments.
- Use explicit `.js` extensions for local TypeScript imports.
