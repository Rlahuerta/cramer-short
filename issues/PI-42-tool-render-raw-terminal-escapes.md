# PI-42: Tool event and approval prompt render untrusted text without terminal-escape sanitization

- Severity: M
- Area: C7b
- Status: Open
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/components/tool-event.ts:114` — `truncateAtWord(error, 80)`, `src/components/tool-event.ts:95` — `this.activeDetail = new Text(`, `src/components/approval-prompt.ts:22` — `const path = (args.path as string)`, `src/components/approval-prompt.ts:39` — `${formatToolLabel(tool)} ${path}`
- Repro kind: runtime

## Summary

The answer path sanitizes untrusted text before rendering (`AnswerBoxComponent.setText` calls `formatResponseTui`, which strips ANSI/OSC/C1 control bytes). The tool and approval components do not: `ToolEventComponent` interpolates tool args, progress messages, result summaries, errors, and denied paths directly into pi-tui `Text`, and `ApprovalPromptComponent` does the same with the proposed path and content preview. pi-tui's `Text` preserves escape sequences, so model/tool-controlled content reaches the terminal raw. A newline in the proposed path also creates an extra line that looks like an approval option.

## Root cause

Components construct `Text` directly from unsanitized strings (`src/components/tool-event.ts:95`, `:103`, `:114`, `:123`, `:134`; `src/components/approval-prompt.ts:22`, `:39`, `:41`) instead of routing them through the existing `formatResponseTui` sanitizer. Tool arguments originate from the LLM (and therefore indirectly from fetched web content), and result summaries/errors originate from remote tool output, so they are not trustworthy terminal input.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-42-tool-render-raw-terminal-escapes.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-42-tool-render-raw-terminal-escapes.ts
```

Expected: tool/approval components strip terminal control sequences and collapse path newlines

Actual: raw OSC sequences reach the render buffer and a newline in the approval path forges an option line

OBSERVED:
    oscLines=3 forgedApprovalOptionLines=2 answerSanitized=true; component rendered a raw OSC sequence

  3 !== 0

## Impact

Remote or prompt-injected content can drive terminal control sequences from the TUI: OSC 52 writes the user's clipboard, screen-park/clear sequences corrupt the display, and carriage returns can overwrite the rendered line. In the approval prompt, a newline in a proposed path forges an extra option line directly above the real selector, which can mislead the user into approving a write or edit they did not intend. The answer path already defends against exactly this payload, so protection is inconsistent across the same screen.

## Suggested fix direction (not applied)

Sanitize before colouring, using the existing exported `formatResponseTui` (`src/utils/ui/markdown-table.ts:351`):

- `src/components/tool-event.ts`: import `formatResponseTui`; wrap `message` (`:95`), `summary` (`:103`), the `Error:` string (`:114`), the limit warning (`:123`), and the denied-path string (`:134`) with `formatResponseTui(...)` before passing them to `theme.*`.
- `src/components/approval-prompt.ts`: import `formatResponseTui`; build `path` as `formatResponseTui(String(args.path ?? args.action ?? '<unknown>')).replace(/[\r\n]+/g, ' ')` (`:22`), and wrap `previewText` (`:30`) before quoting it.

The newline collapse is required in addition to the escape strip: `formatResponseTui` removes control bytes but preserves `\n`.

## Evidence

- Reproduction script: `issues/repro/PI-42-tool-render-raw-terminal-escapes.ts`
- Producing evidence file: `.omo/evidence/bug-hunt-reproducible-issues/task-14-C7b.txt`
- Control: `AnswerBoxComponent` strips the same OSC payload (`answerSanitized=true`), and `src/components/chat-log.test.ts` asserts that behaviour for the answer path.
