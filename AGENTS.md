# Cramer-Short

Bun-first TypeScript CLI agent for financial research. When prose conflicts with code, package.json / CI / source win.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

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

## 4. Goal-Driven Execution

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

## 5. Key Files

| File | What it owns |
|------|-------------|
| `src/index.tsx` | Entry point. Loads dotenv, routes default TUI vs `schedule` subcommand |
| `src/providers.ts` | Canonical provider registry + model-prefix routing. Default: `gpt-5.4` / `openai` |
| `src/tools/registry.ts` | Tool env-gating: `web_search` needs `EXASEARCH_API_KEY` or `PERPLEXITY_API_KEY` or `TAVILY_API_KEY`; `x_search` needs `X_BEARER_TOKEN` |
| `src/utils/paths.ts` | Runtime state lives under `.cramer-short/` (not `.dexter/`) |
| `src/utils/config.ts` | `settings.json` schema. Preserves unknown keys; invalid known fields warn+strip. Auto-migrates legacy model/provider settings |
| `src/agent/prompts.ts` | Loads `.cramer-short/SOUL.md` first, then repo `SOUL.md`; major agent policy/fallback rules live here |
| `src/skills/registry.ts` | Loads builtin skills + project overrides from `.cramer-short/skills/`; overrides win by name |
| `src/utils/test-guards.ts` | Defines unit/integration/e2e tiers and E2E isolation |
| `src/cli-schedule.ts` | Reads `~/.cramer-short/schedules.json` |
| `scripts/release.sh` | Requires `gh` + `node` + clean tree. Interactive; may create version-bump commit, tag, push tag, create GitHub release |

## 6. Commands

```bash
bun install                  # postinstall runs playwright install chromium
bun start                    # default TUI
bun start schedule list      # list configured scheduled jobs
bun start schedule run       # run all scheduled jobs
bun start schedule run JOB   # run one scheduled job
bun run dev                  # watch mode
bun run typecheck            # type check only
bun test                     # unit tests
bun test ./src/foo.test.ts   # single test file
bun test -t "case name"      # filter by test name
bun run test:unit            # unit tier
bun run test:integration     # integration tier (needs RUN_INTEGRATION=1)
bun run test:e2e             # e2e tier (needs RUN_E2E=1)
bun run test:all             # all tiers
bun run test:watch           # watch mode
bun run gateway              # WhatsApp gateway
bun run gateway:login        # WhatsApp link/QR
```

## 7. CI

```bash
bun install --frozen-lockfile --ignore-scripts
bun run typecheck
bun test                    # unit only; integration/e2e are local-only workflows
```

## 8. Gotchas

- Runtime state dir is `.cramer-short/`, never `.dexter/`. Hardcoded in `src/utils/paths.ts`.
- TypeScript is strict ESM. Use explicit `.js` extensions in local imports; `@/*` resolves to `src/*` via `tsconfig.json`.
- Tool availability depends on env keys. `web_search` only registers when `EXASEARCH_API_KEY`, `PERPLEXITY_API_KEY`, or `TAVILY_API_KEY` is set, in that priority order; `X_BEARER_TOKEN` gates `x_search`.
- Settings validation (`src/utils/config.ts`) preserves unknown keys, but invalid known fields are stripped with a warning to stderr instead of crashing startup.
- Default model is `gpt-5.4` via OpenAI. Provider routing is prefix-based in `src/providers.ts`.
- SOUL.md load order: `.cramer-short/SOUL.md` first, then repo `SOUL.md`. User overrides take priority.
- Skill overrides in `.cramer-short/skills/` shadow builtins by name.
- `bun install` runs `playwright install chromium` in postinstall. CI skips this via `--ignore-scripts`.
- `bun run test:e2e` is a manually maintained list of specific files in `package.json`, not a glob. Add new e2e tests there or the standard e2e command will miss them.
- Calendar versioning: `YYYY.M.D`, no zero padding.
- Schedule mode reads from `~/.cramer-short/schedules.json`, not a project-local path. Job `outputFile` supports `~` and `{date}`.
