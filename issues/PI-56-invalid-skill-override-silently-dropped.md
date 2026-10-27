# PI-56: Invalid project skill overrides are silently discarded

- Severity: L
- Area: C8b
- Status: Fixed
- Found at: e3e2088b21acc4963b3c20661e5d15c739244409
- Location: `src/skills/registry.ts:44` — `const metadata = extractSkillMetadata(skillFilePath, source);`, `src/skills/registry.ts:46` — `} catch {`, `src/skills/registry.ts:47` — `// Skip invalid skill files silently`
- Repro kind: runtime

## Summary

`scanSkillDirectory()` swallows every error from `extractSkillMetadata()` and drops the skill with no diagnostic. A project override under `.cramer-short/skills/<name>/SKILL.md` that fails validation (for example a missing `description`, a YAML syntax error, or an unreadable file) therefore never shadows the builtin of the same name and never appears in the prompt or `/skills` panel. The user's customized workflow silently does not run while the builtin workflow silently does.

## Root cause

`src/skills/registry.ts:43-48`:

```ts
        try {
          const metadata = extractSkillMetadata(skillFilePath, source);
          skills.push(metadata);
        } catch {
          // Skip invalid skill files silently
        }
```

`extractSkillMetadata()` (`src/skills/loader.ts:63-80`) throws for a missing/invalid `name` or `description`, for a YAML parse failure, and for an unreadable file. The bare `catch` discards the error and the path, which are the only two facts that would tell the user why their override is not active. `discoverSkills()` then keeps the previously registered builtin entry of that name (lines 69-75), so the failure is invisible at every surface: no warning, no error, and the builtin description still appears in the system prompt.

## Reproduction

Run from the repo root through the canonical harness (it invokes `bun --no-install run issues/repro/PI-56-invalid-skill-override-silently-dropped.ts` under `timeout 30`, with `CS_REPRO_ROOT`/`HOME`/`TMPDIR` pointing at a throwaway root):

```bash
source .omo/evidence/bug-hunt-reproducible-issues/run_repro.sh
run_repro issues/repro/PI-56-invalid-skill-override-silently-dropped.ts
```

Expected: scanning the two invalid overrides emits at least one diagnostic naming the file and the missing field

Actual: no diagnostic emitted; dcf-valuation silently resolves to the builtin (builtin) and my-custom-skill is absent

OBSERVED:
    diagnostics mentioning the two invalid overrides: 0
    diagnostics total: 0
    dcf-valuation resolved source: builtin
    dcf-valuation resolved path: <REPO>src/skills/dcf/SKILL.md
    my-custom-skill listed: false

## Impact

A user who customizes a shipped skill (for example edits the `dcf-valuation` workflow) and introduces a frontmatter typo keeps getting the builtin workflow with no indication that the override was rejected; a brand-new project skill with a malformed header simply does not exist as far as the agent is concerned. The agent confidently follows instructions the user believes they replaced, and the only way to notice is to read the skill file and re-run discovery. The same silent swallow hides permission and I/O errors on the skills directory entry.

## Suggested fix direction (not applied)

Report the skipped file and the reason, matching the existing startup-warning style in `src/utils/config.ts`:

- `src/skills/registry.ts:46` — current: `} catch {` → proposed: `} catch (error) {`
- `src/skills/registry.ts:47` — current: `// Skip invalid skill files silently` → proposed:

```ts
          console.warn(
            `[cramer-short] skipping invalid skill file ${skillFilePath}: ${error instanceof Error ? error.message : String(error)}`,
          );
```

Discovery still skips the invalid entry (no behavior change for valid skills); the failure is now observable.

## Evidence

- Reproduction script: `issues/repro/PI-56-invalid-skill-override-silently-dropped.ts` (exit 42 with `OBSERVED:`/`EXPECTED:`/`ACTUAL:`).
- Producing evidence: `.omo/evidence/bug-hunt-reproducible-issues/task-16-C8b.txt`.
