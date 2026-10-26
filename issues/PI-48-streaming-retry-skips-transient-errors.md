# PI-48: Streaming LLM calls never retry transient provider errors

- Severity: M
- Area: C8b
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/model/llm.ts:497` — `const stream = await withRetry(`, `src/model/llm.ts:498` — `() => llm.stream(messages, signal ? { signal } : {}),`, `src/model/llm.ts:507` — `for await (const chunk of stream) {`
- Repro kind: runtime

## Summary

`streamCallLlm()` routes its provider call through `withRetry()`, but the retried call is `llm.stream(...)`, which is a lazy async generator: invoking it performs no network I/O and cannot fail. Every real provider failure surfaces one step later, on the first `for await` iteration at line 507, outside the retry loop. A transient 429/5xx therefore gets exactly one attempt on the streaming path while the identical error on the non-streaming `callLlm()` path is attempted three times.

## Root cause

`src/model/llm.ts:496-500`:

```ts
  const filter = new StreamingThinkFilter();
  const stream = await withRetry(
    () => llm.stream(messages, signal ? { signal } : {}),
    provider.displayName,
  );
```

`@langchain/core`'s `BaseChatModel.stream()` returns an `AsyncGenerator`; its body (and the HTTP request) only runs when `.next()` is called. `withRetry` (lines 43-63) awaits the factory once and receives a generator object, so its `catch` never runs for a provider error. The error is raised by the `for await` loop at line 507, which has no retry at all.

The non-streaming path (`callLlm`, lines 384-390) retries `invokeRunnable(...)`, which performs the request eagerly, so it applies the intended 3-attempt / 500-1000 ms policy.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-48-streaming-retry-skips-transient-errors.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-48-streaming-retry-skips-transient-errors.ts
```

Expected: streamCallLlm makes 3 attempts on a retryable 429 (same policy as callLlm)

Actual: streamCallLlm made 1 attempt(s) and surfaced "429 Too Many Requests: rate limit exceeded"

OBSERVED:
    streamCallLlm attempts on 429: 1
    streamCallLlm error: 429 Too Many Requests: rate limit exceeded
    callLlm attempts on 429 (control): 3
    callLlm error: [Ollama API] 429 Too Many Requests: rate limit exceeded

## Impact

`streamCallLlm` is the max-iterations synthesis path (`src/agent/agent.ts:1309`). When the model hits a transient rate limit or gateway hiccup there, the retry that the code appears to provide never fires; the agent's catch block degrades the answer to the "Research interrupted — synthesis failed" message plus raw tool-result text. The user gets an unfinished report where a retry would have produced the synthesized answer. The failure is silent apart from that fallback, and the dead retry also means the backoff schedule never runs on this path.

## Suggested fix direction (not applied)

Move the retry from the generator factory to the consumption loop. Extract the chunk-processing body at `src/model/llm.ts:507-538` into a private async generator (`consumeStream()`) that opens `llm.stream(messages, signal ? { signal } : {})` and forwards text, then drive it from a loop that restarts on a retryable failure with the same policy as `withRetry` (3 attempts, `500 * 2 ** attempt` ms backoff, `isNonRetryableError(message)` short-circuit). Restart only while no chunk has been forwarded to the caller; once a chunk is emitted, rethrow instead of duplicating output. Replace the `withRetry(() => llm.stream(...))` wrapper at lines 497-500 with that loop and delete the now-redundant wrapper.

## Evidence

- Reproduction script: `issues/repro/PI-48-streaming-retry-skips-transient-errors.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-16-C8b.txt`.
