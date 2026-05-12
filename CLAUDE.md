# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

## 5. Project Overview

Cramer-Short is an AI-powered financial research agent with a terminal UI (TUI). It uses LangChain to orchestrate LLM calls and provides tools for financial data retrieval, web search, and research synthesis.

## 6. Development Commands

```bash
# Run the TUI
bun start

# Development with hot reload
bun run dev

# Type checking
bun run typecheck

# Run unit tests only
bun test --ignore 'src/**/*.integration.test.ts' --ignore 'src/**/*.e2e.test.ts'

# Run integration tests
RUN_INTEGRATION=1 bun test --filter integration

# Run E2E tests
RUN_E2E=1 bun test --filter e2e

# Run all tests
RUN_INTEGRATION=1 RUN_E2E=1 bun test

# Run evaluations
bun run src/evals/run.ts --sample 10

# WhatsApp gateway
bun run gateway:login  # Link account
bun run gateway        # Start gateway
```

## 7. Architecture

### Core Flow
- `src/index.tsx` → TUI entry point
- `src/cli.ts` → Command parsing, agent orchestration, slash command handlers
- `src/agent/agent.ts` → Main agent loop (LangChain ReAct-style)
- `src/agent/tool-executor.ts` → Tool dispatch with circuit breaker pattern

### Tool Registry
`src/tools/registry.ts` defines all available tools. Tools are organized by domain:
- `finance/` — Market data, fundamentals, earnings, filings, crypto, fixed-income
- `memory/` — Store/recall financial insights
- `fetch/` — Web scraping, PDF parsing
- `search/` — Web search (Exa → Tavily fallback)
- `osint/` — Open-source intelligence, Polymarket
- `browser/` — Playwright for dynamic content

### Memory System (`src/memory/`)
- Persistent Markdown in `.cramer-short/memory/` (`MEMORY.md`, `FINANCE.md`, daily `YYYY-MM-DD.md`)
- Four-tier priority (P1 critical → P4 noise) for pruning
- Auto-injection: Ticker-based + semantic search passes prepend `📚 Prior Research:` to queries
- Namespaces: Scope insights per workflow (`namespace="dcf"`, `namespace="short-thesis"`)
- Dream consolidation: Background merge of daily notes, archive to `.cramer-short/memory/archive/`

### Skills (`src/skills/`)
Pre-built research workflows: `dcf`, `full-analysis`, `short-thesis`, `earnings-preview`, `peer-comparison`, `watchlist-briefing`, `probability-assessment`, `sector-overview`, `geopolitics-osint`, `x-research`. Skills are prompt templates with parameter substitution.

### Run Context
`src/agent/run-context.ts` tracks session state: iteration count, tool results, context window usage, scratchpad for facts that survive compaction.

### TUI (`@mariozechner/pi-tui`)
Terminal UI with components in `src/components/`. Key views: chat history, tool output, scratchpad, help panel.

## 8. Environment Variables

Required: **one** LLM provider key. Optional: financial data, web search, LangSmith.

```bash
# LLM (need at least one)
OPENAI_API_KEY=...
ANTHROPIC_API_KEY=...
GOOGLE_API_KEY=...
OLLAMA_BASE_URL=http://127.0.0.1:11434

# Financial data
FINANCIAL_DATASETS_API_KEY=...
FMP_API_KEY=...  # International ticker fallback

# Web search
EXASEARCH_API_KEY=...
TAVILY_API_KEY=...
```

## 9. Test Structure

Three tiers separated by file suffix:
- `*.test.ts` — Unit tests (default)
- `*.integration.test.ts` — Integration tests (requires `RUN_INTEGRATION=1`)
- `*.e2e.test.ts` — End-to-end tests (requires `RUN_E2E=1`)

Run a single test file: `bun test src/path/to/file.test.ts`

## 10. Key Files

| File | Purpose |
|------|---------|
| `src/cli.ts` | Main entry, slash commands, agent lifecycle |
| `src/agent/agent.ts` | Core agent loop, LangChain orchestration |
| `src/tools/registry.ts` | Tool definitions and registration |
| `src/memory/database.ts` | SQLite + sqlite-vec for memory storage |
| `src/memory/dream.ts` | Background consolidation logic |
| `src/skills/loader.ts` | Skill loading and parameter injection |

## 10. Circuit Breaker Pattern

`src/agent/tool-executor.ts` implements circuit breaker for external API calls. On repeated failures, tool is temporarily disabled. Check `src/utils/circuit-breaker.ts` for state management.

## API Routing Cache

Ticker→API routing persisted in `.cramer-short/api-routing.json`. After discovering a ticker works better with FMP vs Yahoo Finance, that preference is cached and reused.

## 11. Session Files

- `.cramer-short/settings.json` — Runtime config (`maxIterations`, `contextThreshold`, etc.)
- `.cramer-short/schedules.json` — Scheduled research jobs
- `.cramer-short/logs/errors.jsonl` — Structured error log
- `.cramer-short/memory/` — Persistent Markdown memory
- `.cramer-short/api-routing.json` — Ticker→API mappings (30-day TTL)

## 12. Documentation

- `docs/memory.md` — Memory system architecture
- `docs/features.md` — Feature documentation
- `docs/financial-analysis-tutorial.md` — Usage guide
- `docs/watchlist.md` — Portfolio briefing feature

# context-mode — MANDATORY routing rules

You have context-mode MCP tools available. These rules are NOT optional — they protect your context window from flooding. A single unrouted command can dump 56 KB into context and waste the entire session.

## BLOCKED commands — do NOT attempt these

### curl / wget — BLOCKED
Any Bash command containing `curl` or `wget` is intercepted and replaced with an error message. Do NOT retry.
Instead use:
- `ctx_fetch_and_index(url, source)` to fetch and index web pages
- `ctx_execute(language: "javascript", code: "const r = await fetch(...)")` to run HTTP calls in sandbox

### Inline HTTP — BLOCKED
Any Bash command containing `fetch('http`, `requests.get(`, `requests.post(`, `http.get(`, or `http.request(` is intercepted and replaced with an error message. Do NOT retry with Bash.
Instead use:
- `ctx_execute(language, code)` to run HTTP calls in sandbox — only stdout enters context

### WebFetch — BLOCKED
WebFetch calls are denied entirely. The URL is extracted and you are told to use `ctx_fetch_and_index` instead.
Instead use:
- `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` to query the indexed content

## REDIRECTED tools — use sandbox equivalents

### Bash (>20 lines output)
Bash is ONLY for: `git`, `mkdir`, `rm`, `mv`, `cd`, `ls`, `npm install`, `pip install`, and other short-output commands.
For everything else, use:
- `ctx_batch_execute(commands, queries)` — run multiple commands + search in ONE call
- `ctx_execute(language: "shell", code: "...")` — run in sandbox, only stdout enters context

### Read (for analysis)
If you are reading a file to **Edit** it → Read is correct (Edit needs content in context).
If you are reading to **analyze, explore, or summarize** → use `ctx_execute_file(path, language, code)` instead. Only your printed summary enters context. The raw file content stays in the sandbox.

### Grep (large results)
Grep results can flood context. Use `ctx_execute(language: "shell", code: "grep ...")` to run searches in sandbox. Only your printed summary enters context.

## Tool selection hierarchy

1. **GATHER**: `ctx_batch_execute(commands, queries)` — Primary tool. Runs all commands, auto-indexes output, returns search results. ONE call replaces 30+ individual calls.
2. **FOLLOW-UP**: `ctx_search(queries: ["q1", "q2", ...])` — Query indexed content. Pass ALL questions as array in ONE call.
3. **PROCESSING**: `ctx_execute(language, code)` | `ctx_execute_file(path, language, code)` — Sandbox execution. Only stdout enters context.
4. **WEB**: `ctx_fetch_and_index(url, source)` then `ctx_search(queries)` — Fetch, chunk, index, query. Raw HTML never enters context.
5. **INDEX**: `ctx_index(content, source)` — Store content in FTS5 knowledge base for later search.

## Subagent routing

When spawning subagents (Agent/Task tool), the routing block is automatically injected into their prompt. Bash-type subagents are upgraded to general-purpose so they have access to MCP tools. You do NOT need to manually instruct subagents about context-mode.

## Output constraints

- Keep responses under 500 words.
- Write artifacts (code, configs, PRDs) to FILES — never return them as inline text. Return only: file path + 1-line description.
- When indexing content, use descriptive source labels so others can `ctx_search(source: "label")` later.

## ctx commands

| Command | Action |
|---------|--------|
| `ctx stats` | Call the `ctx_stats` MCP tool and display the full output verbatim |
| `ctx doctor` | Call the `ctx_doctor` MCP tool, run the returned shell command, display as checklist |
| `ctx upgrade` | Call the `ctx_upgrade` MCP tool, run the returned shell command, display as checklist |
