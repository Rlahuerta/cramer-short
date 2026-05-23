# Extending Cramer-Short

## Add a tool

1. Create or update a tool module under `src/tools/`. Most tools use LangChain's `DynamicStructuredTool` plus a `zod` schema, and return data with `formatToolResult` from `src/tools/types.ts`.
2. Export a concise description constant for prompt injection.
3. Register the tool in `src/tools/registry.ts` by adding a `RegisteredTool` entry in `getToolRegistry()`. Gate it there if it needs an API key; `web_search`, `x_search`, and `skill` are current examples.
4. Use strict ESM imports: local TypeScript imports must include `.js`.
5. Add or update focused tests, then run `bun test path/to/test.ts` and `bun run typecheck`.

## Add a skill

Skills are Markdown instruction packs loaded from built-ins and project overrides:

- Built-ins: `src/skills/<skill>/SKILL.md`
- Project overrides: `.cramer-short/skills/<skill>/SKILL.md` from the current working directory; these override built-ins with the same `name`.

Each `SKILL.md` needs YAML frontmatter:

```yaml
---
name: example-skill
description: When this skill should be used.
parameters:
  optional_number:
    type: number
    default: 1
    min: 0
    max: 10
---
```

Put the workflow instructions in the Markdown body. `src/skills/registry.ts` discovers skills, and the `skill` tool is registered only when at least one skill is available.
