# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Dexter is an AI-powered financial research agent with a terminal UI (TUI). It uses LangChain to orchestrate LLM calls and provides tools for financial data retrieval, web search, and research synthesis.

## Development Commands

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

## Architecture

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
- Persistent Markdown in `.dexter/memory/` (`MEMORY.md`, `FINANCE.md`, daily `YYYY-MM-DD.md`)
- Four-tier priority (P1 critical → P4 noise) for pruning
- Auto-injection: Ticker-based + semantic search passes prepend `📚 Prior Research:` to queries
- Namespaces: Scope insights per workflow (`namespace="dcf"`, `namespace="short-thesis"`)
- Dream consolidation: Background merge of daily notes, archive to `.dexter/memory/archive/`

### Skills (`src/skills/`)
Pre-built research workflows: `dcf`, `full-analysis`, `short-thesis`, `earnings-preview`, `peer-comparison`, `watchlist-briefing`, `probability-assessment`, `sector-overview`, `geopolitics-osint`, `x-research`. Skills are prompt templates with parameter substitution.

### Run Context
`src/agent/run-context.ts` tracks session state: iteration count, tool results, context window usage, scratchpad for facts that survive compaction.

### TUI (`@mariozechner/pi-tui`)
Terminal UI with components in `src/components/`. Key views: chat history, tool output, scratchpad, help panel.

## Environment Variables

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

## Test Structure

Three tiers separated by file suffix:
- `*.test.ts` — Unit tests (default)
- `*.integration.test.ts` — Integration tests (requires `RUN_INTEGRATION=1`)
- `*.e2e.test.ts` — End-to-end tests (requires `RUN_E2E=1`)

Run a single test file: `bun test src/path/to/file.test.ts`

## Key Files

| File | Purpose |
|------|---------|
| `src/cli.ts` | Main entry, slash commands, agent lifecycle |
| `src/agent/agent.ts` | Core agent loop, LangChain orchestration |
| `src/tools/registry.ts` | Tool definitions and registration |
| `src/memory/database.ts` | SQLite + sqlite-vec for memory storage |
| `src/memory/dream.ts` | Background consolidation logic |
| `src/skills/loader.ts` | Skill loading and parameter injection |

## Circuit Breaker Pattern

`src/agent/tool-executor.ts` implements circuit breaker for external API calls. On repeated failures, tool is temporarily disabled. Check `src/utils/circuit-breaker.ts` for state management.

## API Routing Cache

Ticker→API routing persisted in `.dexter/api-routing.json`. After discovering a ticker works better with FMP vs Yahoo Finance, that preference is cached and reused.

## Session Files

- `.dexter/settings.json` — Runtime config (`maxIterations`, `contextThreshold`, etc.)
- `.dexter/schedules.json` — Scheduled research jobs
- `.dexter/logs/errors.jsonl` — Structured error log
- `.dexter/memory/` — Persistent Markdown memory
- `.dexter/api-routing.json` — Ticker→API mappings (30-day TTL)

## Documentation

- `docs/memory.md` — Memory system architecture
- `docs/features.md` — Feature documentation
- `docs/financial-analysis-tutorial.md` — Usage guide
- `docs/watchlist.md` — Portfolio briefing feature