import { describe, expect, it } from 'bun:test';
import { readdirSync } from 'node:fs';
import { createRequire } from 'node:module';
import { join } from 'node:path';

const ROOT = join(import.meta.dir, '..');
const config = createRequire(import.meta.url)('../.dependency-cruiser.cjs') as {
  forbidden: Array<{ name: string; from?: { path?: string }; to?: { path?: string } }>;
};

function toPattern(ruleName: string): RegExp {
  const rule = config.forbidden.find((candidate) => candidate.name === ruleName);
  if (!rule?.to?.path) throw new Error(`rule ${ruleName} has no to.path`);
  return new RegExp(rule.to.path);
}

/** Every real file in the CLI layer: top-level src/cli*.ts entry files plus components/ and controllers/. */
function realCliLayerPaths(): string[] {
  const topLevel = readdirSync(join(ROOT, 'src')).filter((file) => /^cli[-.]/.test(file)).map((file) => `src/${file}`);
  const controller = readdirSync(join(ROOT, 'src/controllers'))[0];
  const component = readdirSync(join(ROOT, 'src/components'))[0];
  return [...topLevel, `src/controllers/${controller}`, `src/components/${component}`];
}

describe.each(['tools-no-import-cli', 'agent-no-import-cli'])('%s', (ruleName) => {
  const pattern = toPattern(ruleName);

  it('covers the top-level CLI entry files as well as components/ and controllers/', () => {
    const paths = realCliLayerPaths();
    expect(paths).toContain('src/cli.ts');
    expect(paths).toContain('src/cli-schedule.ts');

    expect(paths.filter((path) => !pattern.test(path))).toEqual([]);
  });

  it('does not flag non-CLI modules whose name merely starts with "cli"', () => {
    for (const path of ['src/clipboard.ts', 'src/client.ts', 'src/agent/agent.ts', 'src/tools/registry.ts']) {
      expect(pattern.test(path)).toBe(false);
    }
  });
});
