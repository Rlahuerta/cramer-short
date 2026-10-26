# PI-34: get_financials router prompt and keyword fallback name an unregistered tool

- Severity: M
- Area: C6a
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/finance/get-financials.ts:113` — `return [{ name: 'get_financial_metrics_snapshot', args: { ticker }, id: 'kw-fb', type: 'tool_call' as const }];`, `src/tools/finance/get-financials.ts:116` — `return [{ name: 'get_financial_metrics_snapshot', args: { ticker }, id: 'kw-fb', type: 'tool_call' as const }];`, `src/tools/finance/get-financials.ts:163` — `- For latest financial metrics snapshot (P/E, margins, ROE, EPS, growth rates) → get_financial_metrics_snapshot`
- Repro kind: runtime

## Summary

`get_financial_metrics_snapshot` is not a registered finance tool. The registry exposes that capability as `get_key_ratios` (`FINANCE_TOOLS` at `get-financials.ts:120-137`), yet the router system prompt and both ratio/default branches of `keywordFallbackFinancials()` still emit the stale name. When the router LLM returns no tool calls twice — the documented reason the keyword fallback exists, e.g. local Ollama models — the fallback builds a call for a name that is absent from `FINANCE_TOOL_MAP`, and the tool returns `{_errors: [{error: "Tool 'get_financial_metrics_snapshot' not found"}]}` instead of data.

## Root cause

The tool was renamed from `get_financial_metrics_snapshot` to `get_key_ratios` (key-ratios.ts:19), but three references in `get-financials.ts` were not updated: the "p/e / ratio / valuation / metric" branch (line 113), the default branch (line 116), and the router prompt instruction (line 163). Because the map lookup at line 262 misses, every query reaching those branches fails deterministically.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-34-keyword-fallback-unregistered-tool.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-34-keyword-fallback-unregistered-tool.ts
```

Expected: fallback emits get_key_ratios_AAPL with no _errors

Actual: fallback emits get_financial_metrics_snapshot and reports "Tool 'get_financial_metrics_snapshot' not found"

OBSERVED:
    query = "AAPL valuation metrics"
    result.data keys = ["_errors"]
    result.data._errors = [{"tool":"get_financial_metrics_snapshot","args":{"ticker":"AAPL"},"error":"Tool 'get_financial_metrics_snapshot' not found"}]

## Impact

The deterministic routing fallback — which exists precisely so weak-tool-calling models still get financial data — is broken for every query that reaches the ratio or default branch. Queries such as "AAPL valuation metrics", "AAPL P/E", or a bare company question return a tool error instead of key ratios. A model that follows the prompt literally also selects the missing name and produces the same failure.

## Suggested fix direction (not applied)

Replace the stale identifier with the registered tool name in all three places: `src/tools/finance/get-financials.ts:113`, `:116`, and `:163` — `get_financial_metrics_snapshot` → `get_key_ratios`. Verified green-flip: the repro exits 0 for that change.

## Evidence

- Reproduction script: `issues/repro/PI-34-keyword-fallback-unregistered-tool.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-11-C6a.txt`.
- Green-flip: applied the three-token rename to a scratch copy of `src/`; the same script exits 0.
