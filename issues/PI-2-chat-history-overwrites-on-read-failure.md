# PI-2: Chat history file is replaced after a failed read

- Severity: C
- Area: C5
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/utils/long-term-chat-history.ts:58` — `this.messages = [];`, `src/utils/long-term-chat-history.ts:70` — `await atomicWriteFile(this.filePath, JSON.stringify(data, null, 2));`
- Repro kind: runtime

## Summary

`LongTermChatHistory.load()` treats every read or parse failure as "there is no history" and marks the store as loaded. The next `addUserMessage` or `updateAgentResponse` then saves the empty in-memory list over the existing file. A transient read error or a partially written JSON file therefore destroys the entire persisted input history instead of preserving it.

## Root cause

`src/utils/long-term-chat-history.ts:44-62`:

```ts
    try {
      if (existsSync(this.filePath)) {
        const content = await readFile(this.filePath, 'utf-8');
        const data: MessagesFile = JSON.parse(content);
        this.messages = data.messages || [];
      } else {
        this.messages = [];
        await this.save();
      }
    } catch {
      // If there's any error reading/parsing, start fresh
      this.messages = [];
    }

    this.loaded = true;
```

The catch collapses `ENOENT`, `EACCES`, `EIO`, and a JSON parse error into the same state as a brand-new history. `save()` (line 68) has no guard, because `loaded` is already `true`, so the next message writes `{ messages: [newEntry] }` over the file.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-2-chat-history-overwrites-on-read-failure.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-2-chat-history-overwrites-on-read-failure.ts
```

Expected: the unreadable chat history file is preserved instead of being replaced by the next save

Actual: chat_history.json was replaced by the next save, so the original history is gone

OBSERVED:
    {
      "loadRejected": false,
      "addRejected": false,
      "originalPreserved": false
    }

## Impact

A single failed read wipes the user's entire persistent query/answer history (used for terminal input-history navigation). The data is gone after the next keystroke is recorded; there is no backup.

## Suggested fix direction (not applied)

Never write back after a failed read.

- `src/utils/long-term-chat-history.ts:25` (fields): add `private readFailed = false;`
- `src/utils/long-term-chat-history.ts:58` (catch body): add `this.readFailed = true;` next to `this.messages = [];`
- `src/utils/long-term-chat-history.ts:69` (`save()`): add `if (this.readFailed) return;` as the first statement.

The failed file is left untouched for manual recovery, and new messages are skipped instead of destroying it. (A follow-up could move the unparseable file aside as `chat_history.json.corrupt-<ts>`, mirroring `config.ts`.)

## Evidence

Reproduction script: `issues/repro/PI-2-chat-history-overwrites-on-read-failure.ts`. Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-10-C5.txt`. The script exits 42 with the `OBSERVED:`/`EXPECTED:`/`ACTUAL:` block above.
