# PI-58: Rendered-line cache collides on equal-length answers with a shared prefix

- Severity: L
- Area: C9
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/utils/ui/markdown-table.ts:365` — `const renderedLineCache = new Map<string, number>();`, `src/utils/ui/markdown-table.ts:368` — `content.slice(0, 120)`
- Repro kind: runtime

## Summary

`countRenderedTuiMarkdownLines` memoises rendered line counts in a module-global map keyed by `content.length`, the render width, and the first 120 characters of the content. Two different answers that have the same length and share their first 120 characters map to the same cache entry, so the second answer is assigned the first answer's line count.

## Root cause

The cache key truncates the content (`content.slice(0, 120)`) but also includes only the total length. Different suffixes of equal-length strings are indistinguishable. The consumer (`src/controllers/cli-rendering.ts:77` and `:231`) uses this count to decide whether to truncate the live answer to the viewport budget; a wrong (smaller) count skips truncation and lets the answer overflow into terminal scrollback, which is exactly the duplicate-rendering failure the truncation path exists to prevent.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-58-rendered-line-cache-collision.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-58-rendered-line-cache-collision.ts
```

Expected: countRenderedTuiMarkdownLines(secondAnswer) equals the renderer line count for secondAnswer.

Actual: countRenderedTuiMarkdownLines(secondAnswer) returned 4 (the first answer's 4).

OBSERVED:
    firstAnswer  true=4 counted=4
    secondAnswer true=13 counted=4
    cache key (truncated content) = 140:40:xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx...

## Impact

A colliding pair of answers makes the viewport-cap decision use the wrong line count: the second answer can render past the budget (overflowing into native scrollback and appearing twice) or be truncated too aggressively. Requires a length+prefix collision, so it is edge-only. Severity L.

## Suggested fix direction (not applied)

`src/utils/ui/markdown-table.ts:368` — key the cache on the full content and the width instead of a truncated prefix:

current:

```ts
const key = `${content.length}:${width}:${content.slice(0, 120)}`;
```

proposed:

```ts
const key = `${width}:${content}`;
```

## Evidence

- Repro: `issues/repro/PI-58-rendered-line-cache-collision.ts` (exit 42 at HEAD, `OBSERVED:` block above).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-17-C9.txt`.
