# Second-Wave Quality Improvements

**Status**: Corrected after verification — Phase 1 applied
**Date**: 2026-05-15
**Scope**: Post-phase-4 continuous improvement audit
**Source**: 4 parallel audit agents (type safety, performance, API design, security)

---

## Executive Summary

The original 4-phase refactoring plan eliminated all layer violations, `as any`/`: any` from production code, empty catch blocks, and god modules. This second-wave audit goes deeper — finding issues the first plan deliberately deferred or couldn't detect because they required a stable architecture baseline.

**Four parallel agents scanned 165K lines of TypeScript** and found issues across four domains:

| Domain | Critical | High | Medium | Low |
|--------|----------|------|--------|-----|
| Security & Dependencies | **1** | 3 | 4 | 4 |
| Type Safety | 0 | 5 | 3 | 2 |
| Performance | 0 | 3 | 4 | 3 |
| API Design | 0 | 3 | 4 | 3 |

**Most important findings:**
1. **Input length limits**: many tool schemas accepted unbounded `z.string()` inputs — allows trivial DoS via large content/query payloads
2. **Dependency/git hygiene**: before Phase 1, `@types/bun` was unpinned, `coverage/` and `*.log` were not gitignored, and `fmp-quota.json` was duplicated; Phase 1 now pins `@types/bun` to `1.3.3` and fixes the `.gitignore` entries
3. **Path/command safety audit**: forecast-lab has sandbox/root guards and git argv validation; remaining risk is profile command execution through `shell: true`
4. **Architecture warnings**: depcruiser found 32 violations (19 circular warnings, 11 utils-layer warnings, 2 info orphans, 0 errors) — moved to P2 architecture review

---

## 1. Security & Dependency Risks

### 1.1 Critical: Input Validation

| ID | Issue | Location | Risk |
|----|-------|----------|------|
| **S-1** | **Missing Zod `.max()` validators** | Tool schemas with user-controlled string inputs | Unbounded string inputs allow trivial DoS (multi-GB content via `write_file`, oversized search/query payloads) |

### 1.2 High: Command/Path Safety Gaps

| ID | Issue | Location | Risk |
|----|-------|----------|------|
| **S-2** | **Forecast-lab shell command execution** | `src/experiments/forecast-lab/runner.ts:700` — `spawn(command.command, { shell: true })` | Profile commands execute with shell interpolation; requires validation/allow-listing or explicit documentation of intended scope |
| **S-3** | **Python parity arbitrary execution** | `src/utils/finance/python-parity.ts:54` — `Bun.spawn([python, '-c', script])` | Accepts arbitrary Python code via `script` parameter; safe if internal-only but needs guard/docs if exposed |

### 1.3 Medium: Environment & Config

| ID | Issue | Location |
|----|-------|----------|
| **S-4** | **Direct `process.env` in 31 production files** | `tools/search/*.ts`, `tools/finance/{api,fmp,fundamentals,onchain-crypto,social-sentiment,kalshi-vol-signals}.ts`, `memory/embeddings.ts`, `model/llm.ts`, `cli.ts`, `tools/registry.ts`, `gateway/config.ts`, `gateway/sessions/store.ts` |
| **S-5** | **Unpinned `@types/bun` before Phase 1** | Fixed in Phase 1: `package.json:54` now pins `"@types/bun": "1.3.3"` |
| **S-6** | **All deps use `^` ranges** | `package.json:28-57` — every runtime and dev dependency has `^` prefix |

### 1.4 Low: Config & Git Hygiene

| ID | Issue | Location |
|----|-------|----------|
| **S-7** | **`coverage/` directory missing before Phase 1** | Fixed in Phase 1: `.gitignore` now includes `coverage/` and keeps `coverage.json` |
| **S-8** | **`*.log` not globally ignored before Phase 1** | Fixed in Phase 1: `.gitignore` now includes `*.log` |
| **S-9** | **Duplicate `fmp-quota.json` before Phase 1** | Fixed in Phase 1: `.gitignore` has a single `fmp-quota.json` entry |
| **S-10** | **Brittle manual `.env` parsing** | `src/utils/env.ts:30-43` — parses `.env` manually instead of using `dotenv.parse()` |

### 1.5 Dependency Freshness

| Package | Version | Concern |
|---------|---------|---------|
| `dotenv` | 17.2.3 | **Verified as official from motdotla/dotenv** — actually latest stable; prior "supply chain risk" claim was incorrect |
| `@whiskeysockets/baileys` | 7.0.0-rc.9 | Release candidate, not stable |
| `qrcode-terminal` | 0.12.0 | Unmaintained (~7 years since last release) |
| `zod` | 4.1.13 | Brand-new v4; monitor for breaking changes |
| `@types/bun` | 1.3.3 | Pinned in Phase 1 to match the lockfile version |

### 1.6 New Circular Dependencies (depcruiser)

Dependency-cruiser found **32 violations** across the restructured codebase:
- 19 circular import warnings (most are benign tool co-dependencies)
- 11 utils-layer warnings (utils/ importing from agent/ or controllers/)
- 2 info-level orphan warnings
- **0 error-level violations**

The "2 actual circular paths" claim in the original draft was incorrect. Key cycles are:

```
warn no-circular: src/tools/memory/index.ts →
  src/tools/memory/memory-update.ts →
  src/memory/index.ts →
  src/memory/dream.ts →
  src/model/llm.ts →
  src/agent/prompts.ts →
  src/tools/registry.ts →
  src/tools/memory/index.ts          ← 7-node cycle (expected pattern in tool/memory boundary)

warn no-circular: src/tools/finance/markov-distribution.ts →
  src/tools/finance/regime-calibrator.ts →
  src/tools/finance/markov-distribution.ts   ← direct cycle (shared parameter defaults)
```

Most violations are acceptable design tradeoffs. Breaking these would require invasive refactoring with limited safety benefit.

---

## 2. Type Safety — Deeper Than `as any`

The first refactoring eliminated all `as any` and `: any` from production code (201 remaining are all in tests). This audit found more subtle type-safety compromises.

### 2.1 `Record<string, unknown>` Overuse

The most pervasive pattern. **213+ occurrences** in production code (not the 150 originally estimated). Key hotspots:

| File | Occurrences | Issue |
|------|-------------|-------|
| `src/agent/query-router.ts` | 30+ | Tool call args/results parsed as `Record<string, unknown>` with no runtime shape validation |
| `src/tools/finance/arbiter-replay.ts` | 8+ | JSON line parsing casts entire objects to `Record<string, unknown>` |
| `src/experiments/forecast-lab/runner.ts` | 6+ | Experiment results and mutation data as loose records |
| `src/tools/finance/fundamentals.ts` | 4+ | Financial statement data as `Record<string, unknown>[]` |
| `src/utils/errors.ts` | 3+ | Error context objects as `Record<string, unknown>` |

**Why it matters**: `Record<string, unknown>` defeats the type system. Property access returns `unknown`, requiring cast chains. The original `as any` problem was replaced by `data.foo as string` on `Record<string, unknown>` — same unsafe behavior, different wrapper.

**Recommended fix**: Define typed interfaces for each tool's result shape. Replace `Record<string, unknown>` with `z.infer<typeof Schema>` from existing Zod schemas.

### 2.2 `as unknown as` Cast Without Guards

**~15 occurrences in production code** (not 5 as originally estimated). Examples:

| File:Line | Pattern |
|-----------|---------|
| `src/tools/finance/backtest/replay-price-history-adapter.ts:108` | `return parsed as unknown as ReplayFixturePriceStore` |
| `src/tools/finance/polymarket.ts:416` | `return [] as unknown as T` — unconstrained generic cast |
| `src/tools/forecast-lab-run.ts:1420` | `return payload as unknown as ForecastLabRunToolPayload` |
| `src/utils/in-memory-chat-history.ts:160` | `response as unknown as { message_ids: number[] }` |
| `src/controllers/cli-rendering.ts:189` | `tui as unknown as { ... }` — private API access |

Pattern: `as unknown as Target` is a type-punning escape hatch that suppresses the compiler's type checking entirely. It's equivalent to `as any` but passes linters.

### 2.3 Unsafe Type Assertions (Not `as any`)

| File | Casts Found |
|------|-------------|
| `src/gateway/channels/whatsapp/error.ts` | `as number`, `as string` on Baileys error properties |
| `src/tools/finance/social-sentiment.ts:161-167` | Mass `as string`, `as number` on Reddit JSON |
| `src/tools/finance/arbiter-replay.ts:412-418` | `parsed.capturedAt as string`, `parsed.ticker as string`, `parsed.horizonDays as number` |
| `src/tools/finance/fundamentals.ts:146-148` | `s.report_period as string`, `s.period as string`, `s.revenue as number` |
| `src/agent/query-router.ts:89` | `parseInt(...) as 1 \| 2 \| 3 \| 4` — narrows `number` to literal union without guard |

These assertions bypass the type checker. If the runtime data doesn't match (e.g., API changes shape), the assertion silently returns wrong types.

### 2.4 Excessive Optional Chaining

Only one notable production case (not test files):
- `src/gateway/channels/whatsapp/login.ts:20` — 4-level chain hiding missing error type:
  ```typescript
  (err as { error?: { output?: { statusCode?: number } } })?.error?.output?.statusCode ?? 500
  ```

Most optional chaining in the codebase is appropriate defensive coding.

### 2.5 Export Return Type Omissions

Most exported functions in the restructured codebase **do** have explicit return types. A few exceptions:

| File | Function | Missing Return |
|------|----------|----------------|
| `src/evals/run.ts:217` | `createEvaluationRunner` | No return type annotation |
| `src/experiments/forecast-lab/runner.ts:1627` | `snapshotPromotionSourceInvariants` | Returns `unknown` (same problem) |
| `src/utils/finance/number-format.ts:72` | `annotateFinancialNumbers` | Returns `unknown` |
| `src/experiments/forecast-lab/ledger.ts:225` | `stableValue` | Returns `unknown` |

Several functions return `unknown` explicitly — this is correct but defeats inference. Better to narrow to specific types.

### 2.6 `Partial<T>` Usage

Frequent use of `Partial<T>` in config objects (e.g., `Partial<WalkForwardConfig>`, `Partial<ConnectionState>`, `Partial<typeof FORECAST_LAB_MARKOV_PARAMETER_DEFAULTS>`). Most are reasonable for config merging, but some could use explicit optional interfaces for better documentation.

---

## 3. Performance & Concurrency

*Note: Original draft overstated some hot-path claims. Corrections incorporated below.*

### 3.1 Sequential Awaits

| File:Line | Pattern |
|-----------|---------|
| `src/agent/agent.ts:213-220` | Three independent memory operations (`listFiles`, `loadSessionContext`, etc.) run sequentially during `Agent.create()` |
| `src/memory/dream.ts:284-285` | Two independent file reads sequential |
| `src/memory/dream.ts:188-191` | Sequential reads in a loop |

**Fix**: Bundle independent async ops with `Promise.all([...])` to reduce cold-start and Dream consolidation latency.

### 3.2 Missing AbortSignal Propagation

Real gaps confirmed:
- `src/tools/browser/browser.ts` — Browser tool `func` does not accept or forward `AbortSignal`. Esc cancels the agent but the Playwright navigation keeps running.
- `src/gateway/heartbeat/runner.ts` — Deep async loops don't propagate signal.
- `src/memory/dream.ts:76` — `callLlm` with 300s timeout but no external signal.

### 3.3 Resource Leaks

**Playwright browser** (`src/tools/browser/browser.ts`):
- Module-level `browser` singleton launched lazily with `headless: false`
- If error is thrown mid-tool, `closeBrowser()` is never called — Chromium process leaks
- `open` action overwrites `page` variable without closing previous page

**Database** (`src/memory/database.ts`):
- `MemoryDatabase.close()` exists but not called on process exit
- `FSWatcher` in `memory/indexer.ts` — file watchers may not be cleaned on uncaught exception

**Gateway sessions** (`src/gateway/agent-runner.ts:20`):
- `sessions` Map grows boundlessly — never evicted or cleaned up

### 3.4 Synchronous Blocking in Async Contexts

**Verified real cases:**

| File | API | Context |
|------|-----|---------|
| `src/components/at-path-provider.ts:61` | `readdirSync`, `statSync` | Called on every `@` keystroke in TUI — blocks event loop |
| `src/utils/cache.ts` | `readFileSync`, `writeFileSync` | Invoked from async tool paths — serializes concurrent tool execution |
| `src/controllers/watchlist-controller.ts:101` | `readFileSync` | Called from async UI flow |
| `src/gateway/gateway.ts:30` | `appendFileSync` | Blocks on every debug log call (but logs are metadata/timestamps, not full bodies) |

**Note:** Original claim about "gateway logs full bodies" was false — actual logging is metadata/previews only.

### 3.5 Unbounded In-Memory Structures

| Location | Structure | Risk |
|----------|-----------|------|
| `src/agent/tool-executor.ts:53,64` | `requestCache` and `pendingRequests` Maps | Grow without eviction during a single agent run |
| `src/utils/in-memory-chat-history.ts:47` | `relevantMessagesByQuery` Map | Grows per distinct query, never cleaned |
| `src/gateway/agent-runner.ts:20` | `sessions` Map | Module-level, never evicted in long-running gateway |

### 3.6 Regex Compilation in Hot Paths

Lower priority than originally stated:
- `src/components/select-list.ts:196-198` — `new RegExp(q, 'i')` compiled on every keystroke (UI component, acceptable pattern)
- `src/experiments/forecast-lab/query-router.ts:132` — `new RegExp(...)` on every function call (experiment code, not production hot path)

### 3.7 JSON in Hot Paths

**Real but tolerable:**
- `src/agent/tool-executor.ts:247` — `JSON.stringify` for cache key on **every** tool invocation (acceptable for cache key generation)
- `src/utils/in-memory-chat-history.ts:149` — `JSON.stringify` on entire message history per query turn (not ideal but context is bounded)
- `src/controllers/watchlist-price-fetchers.ts:37,43` — `JSON.parse` on every fulfilled result (bounded by watchlist size)

**Note:** Original claim of "hot path" was overstated — these are normal-frequency operations, not tight loops.

### 3.8 Timer Leaks

- `src/cli.ts:232` — `watchlistRefreshTimer` setInterval not cleared on SIGINT
- `src/gateway/heartbeat/runner.ts:175-184` — `scheduleNext` can leak timer if config load throws
- `src/memory/indexer.ts:75` — `watchTimer` setTimeout not cleaned without explicit `stopWatching()`

---

## 4. API Design & Naming Inconsistencies

### 4.1 Tool Naming Inconsistencies

The tool registry mixes naming conventions:

| Registry Name | File | Export Symbol | Pattern |
|---------------|------|---------------|---------|
| `get_financials` | `get-financials.ts` | `createGetFinancials()` | snake_case / kebab / camelCase factory |
| `stock_screener` | `screen-stocks.ts` | `createScreenStocks()` | Mismatched name ("stock_screener" ≠ "screen_stocks") |
| `wacc_inputs` | `wacc-inputs.ts` | `waccInputsTool` | No `create` prefix |
| `bitmex_market` | `bitmex.ts` | `bitmexMarketTool` | Direct export, no create |
| `portfolio_risk` | `portfolio-risk.ts` | `createPortfolioRiskTool()` | `create` prefix + `Tool` suffix |
| `get_robinhood_quote` | `robinhood.ts` | `getRobinhoodQuote` | `get` prefix, no `Tool` suffix |

There is no unifying convention. Some tools use `get_` prefix, some use bare nouns, some use `create` factories, others export constants.

### 4.2 File Naming: Snake vs Kebab

- `insider_trades.ts` (snake_case) vs `earnings-transcripts.ts` (kebab-case)
- `get-financials.ts` (kebab-case) vs `onchain-crypto.ts` (kebab-case)
- Mix of conventions within the same directory

### 4.3 Error Handling Inconsistency

Within `tools/finance/`, two incompatible styles coexist:
- **Throw-heavy**: `api.ts`, `fmp.ts`, `yahoo-client.ts`, `fixed-income.ts`
- **Null-returning**: `fundamentals.ts`, `robinhood-client.ts`, `social-sentiment.ts`, `get-financials.ts`

Callers must know which tool uses which pattern.

### 4.4 Export Style by Layer

| Layer | Primary Style | Consistency |
|-------|---------------|-------------|
| `tools/finance/` | `export const tool = new DynamicStructuredTool(...)` | Mostly consistent |
| `utils/` | `export function`, `export const` | Consistent |
| `controllers/` | `export class` | Consistent |
| `memory/` | `export class`, `export function` | Mixed |

Within each directory there's good consistency, but across layers expectations differ.

### 4.5 Module Organization Suspicious Placements

| File | Location | Suggested |
|------|----------|-----------|
| `tools/forecast-lab-run.ts` | `src/tools/` root | Move to `experiments/forecast-lab/` — it imports from experiments/ and leaks that layer into tools |
| `utils/finance/ensemble.ts` (906 lines) | `utils/finance/` | Domain engine — belongs in `tools/finance/` or `experiments/` |
| `utils/finance/cross-validate.ts`, `adwin.ts`, `kswin.ts`, `garch.ts`, `vol-regime.ts` | `utils/finance/` | Finance-domain algorithms in utils/ instead of `tools/finance/` |
| `tools/finance/backtest/` (30+ files) | `tools/finance/` | Blurs line between tools and experiments — consider `experiments/backtest/` |

### 4.6 Import Path Inconsistency

- **Source code**: Almost exclusively relative (`../`, `./`)
- **Test files**: Primary users of `@/` alias (e.g., `@/utils/test-guards.js`)
- A few production files also use `@/` (e.g., `utils/env.ts`, `utils/model.ts`)

This bifurcation means tests follow a different import style than source.

### 4.7 Test File Naming

Multiple extra suffixes beyond the standard `.test.ts`:

| Suffix | Examples |
|--------|----------|
| `.coverage.test.ts` | `select-list-coverage.test.ts` |
| `.e2e.test.ts` | `agent.e2e.test.ts`, `llm.e2e.test.ts` |
| `.integration.test.ts` | `markov-distribution.integration.test.ts` |
| `.parity.test.ts` | `rnd-integration.parity.test.ts` |
| `.longshot.test.ts` | `rnd-integration.longshot.test.ts` |

No single convention. Some integration tests use `.test.ts` alone, others append the tier.

### 4.8 Interface vs Type Usage

- **592** `interface` declarations vs **274** `type` declarations
- Convention appears to be: `interface` for object shapes, `type` for unions/aliases/Zod inference
- Mostly followed, but some simple objects use `type` instead of `interface`

### 4.9 Zod Schema Location

- **Co-located**: Schemas live in tool files (e.g., `FinancialStatementsInputSchema` in `fundamentals.ts`)
- **Separated**: Config schemas in `utils/config.ts`, chat history validation in `utils/in-memory-chat-history.ts`
- No centralized `schemas/` directory — schemas are scattered inline

---

## 5. Prioritized Recommendations

### P1 — Critical Validation + Quick Hygiene (Week 1)

| ID | Task | Status | Effort |
|----|------|--------|--------|
| **1** | Add risk-based `z.string().max()` validators to production tool schemas (path: 4096, content: 100K, query: 10K) | **Done in Phase 1** | 2-3h |
| **2** | Pin `@types/bun` to lockfile version `1.3.3` | **Done in Phase 1** | 15m |
| **3** | Fix .gitignore by adding `coverage/` and `*.log`, and keeping one `fmp-quota.json` entry | **Done in Phase 1** | 15m |
| **4** | Guard and document forecast-lab runner `shell: true` usage and profile command constraints | **Done in Phase 1** | 30m |
| **5** | Document `python-parity.ts` as internal trusted-script execution | **Done in Phase 1** | 30m |

### P2 — High Priority (Week 2–3)

| ID | Task | Effort |
|----|------|--------|
| **6** | Review and document circular dependencies (19 circular warnings acceptable if by design) | 2h |
| **7** | Fix the 11 `utils-no-import-agent` layer violations if safely addressable | 3h |
| **8** | Add AbortSignal propagation to `browser.ts`, heartbeat runner, `dream.ts` | 3h |
| **9** | Wrap Playwright tool in try/finally with `closeBrowser()` guarantee | 2h |
| **10** | Add `Promise.all` batching to `agent.ts` startup and `dream.ts` reads | 2h |
| **11** | Centralize `process.env` access through `utils/env.ts` (except tool registry gating) | 3h |

### P3 — Medium Priority (Week 4–6)

| ID | Task | Effort |
|----|------|--------|
| **12** | Define typed result interfaces for top-10 most-used tools (replace `Record<string, unknown>`) | 6h |
| **13** | Standardize tool naming: pick `snake_case` registry names, `kebab-case` files, `camelCase` symbols | 3h |
| **14** | Establish single error-handling contract (throw vs return null) for `tools/finance/` | 4h |
| **15** | Add LRU eviction to unbounded Maps (`tool-executor.ts`, `agent-runner.ts`, `chat-history.ts`) | 4h |
| **16** | Move `utils/finance/` domain algorithms to `tools/finance/` or `experiments/` | 3h |
| **17** | Standardize test naming: always include tier suffix or never include it | 2h |
| **18** | Move `tools/forecast-lab-run.ts` into `experiments/forecast-lab/` | 2h |

### P4 — Low Priority: Polish (Ongoing)

| ID | Task | Effort |
|----|------|--------|
| **19** | Replace `as unknown as Target` patterns with proper type guards (~15 locations) | 3h |
| **20** | Replace `.catch(() => {})` in remaining test files (2 locations) | 15m |
| **21** | Add explicit return types to `createEvaluationRunner` and the 4 `unknown`-returning functions | 1h |
| **22** | Move inline Zod schemas from `utils/` to either co-locate or into `schemas/` | 2h |

---

## 6. Success Criteria

- [x] Review document corrected with verified facts (dotenv is official, git.ts uses argv not shell, forecast-lab has guards, python-parity is internal, counts updated)
- [x] All `z.string()` tool params have `.max()` validators appropriate to their use
- [x] `@types/bun` pinned to lockfile version 1.3.3
- [x] `.gitignore` updated: `coverage/` and `*.log` added, duplicate `fmp-quota.json` removed
- [x] Forecast-lab `shell: true` usage documented with profile command constraints
- [x] Python-parity marked internal-only or guarded if exposed
- [ ] depcruiser violations reviewed and documented (0 errors; warnings acceptable if by design)
- [ ] `process.env` access centralized through `utils/env.ts` (except in registry.ts which gates tool availability)
- [ ] AbortSignal flows through browser, heartbeat, and dream async paths
- [ ] Top 5 tools have typed result interfaces (replacing `Record<string, unknown>`)
- [ ] Tool naming standardized
- [x] `bun run typecheck` passes clean
- [x] `bun test` passes all unit tests

---

## Appendix A: Audit Methodology

Four independent background agents scanned the entire codebase in parallel:

1. **Type Safety Agent**: Pattern-matched optional chaining, `as` casts (excluding `as any`), `unknown` without guards, `Record<string, unknown>`, missing return types, `any[]`, unconstrained generics, `Partial<T>`, union narrowing gaps.
2. **Performance Agent**: Scanned for sequential awaits, missing AbortSignals, resource leaks, sync I/O, unbounded collections, regex compilation, JSON in hot paths, lazy init, deep cloning, timer leaks.
3. **API Design Agent**: Compared tool names across registry/files/symbols, checked parameter ordering, error handling patterns, export styles, module organization, import paths, test naming, interface/type usage, Zod locations.
4. **Security Agent**: Checked dependency pinning, env var usage, hardcoded secrets, shell injection, path traversal, input limits, dependency freshness, circular deps, .gitignore coverage, Bun-specific APIs.

Results were synthesized into this document for review and prioritization.
