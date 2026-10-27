# PI-3: edit_file fuzzy fallback silently rewrites unrelated lines

- Severity: C
- Area: C8a
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/tools/filesystem/utils/edit-diff.ts:69` — `contentForReplacement: fuzzyContent,`, `src/tools/filesystem/utils/edit-diff.ts:22` — `.map((line) => line.trimEnd())`, `src/tools/filesystem/edit-file.ts:89` — `const baseContent = matchResult.contentForReplacement;`
- Repro kind: runtime

## Summary

When `old_text` does not match byte-for-byte, `edit_file` falls back to
`fuzzyFindText`. That helper normalises the entire file — it trims the end of
every line and folds smart quotes, typographic dashes and exotic spaces to ASCII
— and then hands the normalised copy back as `contentForReplacement`. `edit_file`
builds the written file from that copy, so characters and whitespace outside the
edited region are silently changed. A file containing `const c = “x”;` and a
trailing-space line comes back as `const c = "x";` with the trailing spaces gone
after an edit to an unrelated line. The tool reports success and shows a diff
that only mentions the intended line.

## Root cause

`fuzzyFindText` returns `contentForReplacement: fuzzyContent` (edit-diff.ts:69),
where `fuzzyContent` is produced by `normalizeForFuzzyMatch`, whose per-line
`trimEnd()` (edit-diff.ts:22) and character folding make it a lossy copy of the
file. `editFileTool` then splices the replacement into that lossy copy
(edit-file.ts:89) and writes the result, instead of splicing into the original
content. `matchResult.usedFuzzyMatch` is returned by the helper but never
consulted, so nothing warns that the written file is not the original plus one
edit. Removing `trimEnd()` makes the normalisation length-preserving, after which
the fuzzy match offsets address the original content one-to-one.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-3-edit-file-fuzzy-normalizes-whole-file.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-3-edit-file-fuzzy-normalizes-whole-file.ts
```

Expected: only the edited line changes; every untouched line keeps its exact bytes

Actual: lines 2-3 rewritten: ["const b = 1;   ","const c = “x”;"] -> ["const b = 1;","const c = \"x\";"]

OBSERVED:
    {
      "before": "const a = “it’s”;\nconst b = 1;   \nconst c = “x”;\n",
      "after": "const a = \"REPLACED\";\nconst b = 1;\nconst c = \"x\";\n",
      "untouchedLinesRewritten": [
        {
          "before": "const b = 1;   ",
          "after": "const b = 1;"
        },
        {
          "before": "const c = “x”;",
          "after": "const c = \"x\";"
        }
      ],
      "targetLineApplied": true
    }

## Impact

Silent, unbounded data corruption of a user file: any fuzzy edit rewrites
typographic characters and trailing whitespace across the whole file. In source
code a smart quote inside a string literal changes the program's value; in
Markdown/JSON/YAML files the same folding can change content. The user is shown a
diff that omits the collateral changes, so the corruption is easy to commit.

## Suggested fix direction (not applied)

Make the fuzzy path operate on the original content:

- `src/tools/filesystem/utils/edit-diff.ts:22` — current `.map((line) => line.trimEnd())` → proposed `.map((line) => line)`
- `src/tools/filesystem/utils/edit-diff.ts:69` — current `contentForReplacement: fuzzyContent,` → proposed `contentForReplacement: content,`

With the per-line trimming removed, `normalizeForFuzzyMatch` is a character-for-character
fold, so `fuzzyIndex`/`matchLength` address the original text and the splice keeps
every untouched byte. If trailing-whitespace tolerance must be retained, map the
fuzzy offsets back to original offsets instead of returning a rewritten buffer.

## Evidence

Script: `issues/repro/PI-3-edit-file-fuzzy-normalizes-whole-file.ts`
Evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-15-C8a.txt` (rc=42).
