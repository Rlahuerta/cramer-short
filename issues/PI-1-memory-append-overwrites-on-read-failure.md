# PI-1: Memory append overwrites a file it failed to read

- Severity: C
- Area: C5
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/memory/store.ts:75` — `return '';`, `src/memory/store.ts:86` — `const previous = await this.readMemoryFile(path);`
- Repro kind: runtime

## Summary

`MemoryStore.appendMemoryFile` is a read-modify-write: it reads the current file, appends, then atomically replaces the file. Its reader, `readMemoryFile`, catches every error and returns `''`, so a file that exists but cannot be read right now (EACCES, EIO, EMFILE) is indistinguishable from an empty file. The append then writes only the new text over the original, permanently deleting durable memory (MEMORY.md, FINANCE.md, or a dated memory file).

## Root cause

`src/memory/store.ts:70-77`:

```ts
  async readMemoryFile(path: string): Promise<string> {
    const resolved = this.resolveMemoryPath(path);
    try {
      return await readFile(resolved, 'utf-8');
    } catch {
      return '';
    }
  }
```

The catch does not distinguish `ENOENT` (a genuinely new file) from a read error on an existing file. `appendMemoryFile` (line 84) trusts the returned `''` as the file's real content and calls `writeMemoryFile`, whose `atomicWriteFile` does a temp-file `rename` over the target — a rename needs only directory write permission, so it succeeds even when the original file is unreadable. The original bytes are gone.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-1-memory-append-overwrites-on-read-failure.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-1-memory-append-overwrites-on-read-failure.ts
```

Expected: appendMemoryFile rejects the read error and MEMORY.md still contains the original fact

Actual: appendMemoryFile resolved and MEMORY.md was replaced with "- new fact"

OBSERVED:
    {
      "appendRejected": false,
      "originalPreserved": false,
      "fileContent": "- new fact"
    }

## Impact

Any user whose memory file is temporarily unreadable (permission change, file-descriptor exhaustion, I/O error, a file replaced by a non-readable stub) loses that file's accumulated facts the next time any memory tool appends to it. This is silent: the append succeeds and reports success.

## Suggested fix direction (not applied)

Make the append path distinguish "missing" from "unreadable". At `src/memory/store.ts:86`, replace

```ts
      const previous = await this.readMemoryFile(path);
```

with a strict read:

```ts
      let previous: string;
      try {
        previous = await readFile(this.resolveMemoryPath(path), 'utf-8');
      } catch (err) {
        if ((err as NodeJS.ErrnoException).code !== 'ENOENT') throw err;
        previous = '';
      }
```

(`readFile` is already imported at the top of `store.ts`.) The append now fails loudly instead of replacing a file it could not read; the caller can retry.

## Evidence

Reproduction script: `issues/repro/PI-1-memory-append-overwrites-on-read-failure.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-10-C5.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above.
