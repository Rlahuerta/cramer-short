/**
 * Tests for filesystem tools: read-file, edit-file, sandbox, path-utils.
 *
 * Uses a temp subdirectory inside process.cwd() so all paths remain within the
 * sandbox root without requiring a process.cwd() spy.
 */
import { afterEach, beforeEach, describe, expect, it } from 'bun:test';
import { randomUUID } from 'node:crypto';
import { mkdir, rm, symlink, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { join, resolve } from 'node:path';

import { editFileTool } from './edit-file.js';
import { isSecretPath, readFileTool } from './read-file.js';
import { assertSandboxPath, resolveSandboxPath } from './sandbox.js';
import { expandPath, resolveReadPath, resolveToCwd } from './utils/path-utils.js';

// ---------------------------------------------------------------------------
// Temp directory setup — kept inside cwd so sandbox checks pass without mocking
// ---------------------------------------------------------------------------

// Per-process name: concurrent `bun test` runs in one checkout would otherwise delete each other's files.
const REL_TMP = `.cramer-short-fs-test-tmp-${randomUUID()}`; // relative to cwd
const TMP_DIR = join(process.cwd(), REL_TMP);

beforeEach(async () => {
  await mkdir(TMP_DIR, { recursive: true });
});

afterEach(async () => {
  await rm(TMP_DIR, { recursive: true, force: true });
});

function tmpFile(name: string): string {
  return join(TMP_DIR, name);
}

function relPath(name: string): string {
  return `${REL_TMP}/${name}`;
}

function parseResult(str: string): Record<string, unknown> {
  return (JSON.parse(str) as { data: unknown }).data as Record<string, unknown>;
}

// ---------------------------------------------------------------------------
// expandPath (path-utils)
// ---------------------------------------------------------------------------

describe('expandPath', () => {
  it('strips @ prefix', () => {
    const result = expandPath('@/some/path');
    expect(result).toBe('/some/path');
  });

  it('expands ~ to home directory', () => {
    expect(expandPath('~')).toBe(homedir());
  });

  it('expands ~/ to home directory + rest', () => {
    expect(expandPath('~/documents')).toBe(homedir() + '/documents');
  });

  it('normalizes unicode non-breaking spaces to regular spaces', () => {
    // \u00A0 is non-breaking space
    const result = expandPath('/path\u00A0with\u00A0spaces');
    expect(result).toBe('/path with spaces');
  });

  it('normalizes other unicode space variants', () => {
    // \u2003 is EM SPACE (in \u2000-\u200A range)
    const result = expandPath('hello\u2003world');
    expect(result).toBe('hello world');
  });

  it('returns plain paths unchanged', () => {
    expect(expandPath('/absolute/path')).toBe('/absolute/path');
    expect(expandPath('relative/path')).toBe('relative/path');
  });

  it('@ prefix + ~ expansion applies in order (@ first)', () => {
    // @~ → strips @ → ~ → homedir
    expect(expandPath('@~')).toBe(homedir());
  });
});

// ---------------------------------------------------------------------------
// resolveToCwd (path-utils)
// ---------------------------------------------------------------------------

describe('resolveToCwd', () => {
  it('returns absolute paths unchanged', () => {
    expect(resolveToCwd('/absolute/path', '/some/cwd')).toBe('/absolute/path');
  });

  it('resolves relative paths against cwd', () => {
    expect(resolveToCwd('file.txt', '/project')).toBe('/project/file.txt');
  });

  it('resolves .. segments against cwd', () => {
    expect(resolveToCwd('../sibling', '/project/child')).toBe('/project/sibling');
  });

  it('expands ~ before resolving', () => {
    expect(resolveToCwd('~/foo', '/project')).toBe(homedir() + '/foo');
  });
});

// ---------------------------------------------------------------------------
// resolveReadPath (path-utils)
// ---------------------------------------------------------------------------

describe('resolveReadPath', () => {
  it('returns resolved path when file exists (fileExists returns true)', async () => {
    const file = tmpFile('exists.txt');
    await writeFile(file, 'content');

    const result = resolveReadPath(file, TMP_DIR);
    expect(result).toBe(file);
  });

  it('returns resolved path even when file does not exist', () => {
    const ghost = tmpFile('nonexistent.txt');
    const result = resolveReadPath(ghost, TMP_DIR);
    expect(result).toBe(ghost);
  });

  it('resolves relative path against cwd', async () => {
    const file = tmpFile('rel-test.txt');
    await writeFile(file, 'hi');

    const result = resolveReadPath('rel-test.txt', TMP_DIR);
    expect(result).toBe(file);
  });
});

// ---------------------------------------------------------------------------
// resolveSandboxPath (sandbox)
// ---------------------------------------------------------------------------

describe('resolveSandboxPath', () => {
  it('returns resolved absolute path and relative for a file inside root', () => {
    const result = resolveSandboxPath({ filePath: 'test.txt', cwd: TMP_DIR, root: TMP_DIR });
    expect(result.resolved).toBe(join(TMP_DIR, 'test.txt'));
    expect(result.relative).toBe('test.txt');
  });

  it('returns relative="" when filePath resolves to exactly the root', () => {
    // Passing the root dir itself as filePath → relative becomes ''
    const result = resolveSandboxPath({ filePath: TMP_DIR, cwd: TMP_DIR, root: TMP_DIR });
    expect(result.relative).toBe('');
    expect(result.resolved).toBe(resolve(TMP_DIR));
  });

  it('throws when path escapes the sandbox root', () => {
    expect(() =>
      resolveSandboxPath({ filePath: '../../etc/passwd', cwd: TMP_DIR, root: TMP_DIR }),
    ).toThrow(/sandbox/i);
  });

  it('throws for absolute path outside root', () => {
    expect(() =>
      resolveSandboxPath({ filePath: '/etc/passwd', cwd: TMP_DIR, root: TMP_DIR }),
    ).toThrow(/sandbox/i);
  });
});

// ---------------------------------------------------------------------------
// assertSandboxPath (sandbox) — covers lines 14, 37 (empty relative path)
// ---------------------------------------------------------------------------

describe('assertSandboxPath', () => {
  it('resolves a valid relative path inside cwd', async () => {
    const result = await assertSandboxPath({ filePath: 'test.txt', cwd: TMP_DIR });
    expect(result.resolved).toBe(join(TMP_DIR, 'test.txt'));
    expect(result.relative).toBe('test.txt');
  });

  it('returns relative="" when filePath is the root — covers sandbox.ts lines 14 and 37', async () => {
    // This hits: resolveSandboxPath line 14 (rel === '') + assertNoSymlink line 37 (!relativePath)
    const result = await assertSandboxPath({ filePath: TMP_DIR, cwd: TMP_DIR, root: TMP_DIR });
    expect(result.relative).toBe('');
  });

  it('throws when a symlink is detected inside sandbox', async () => {
    const real = tmpFile('real.txt');
    const link = tmpFile('link.txt');
    await writeFile(real, 'real content');
    await symlink(real, link);

    await expect(assertSandboxPath({ filePath: 'link.txt', cwd: TMP_DIR })).rejects.toThrow(
      /Symlink/i,
    );
  });

  it('throws when path escapes sandbox root', async () => {
    await expect(
      assertSandboxPath({ filePath: '../../etc/passwd', cwd: TMP_DIR }),
    ).rejects.toThrow(/sandbox/i);
  });
});

// ---------------------------------------------------------------------------
// read_file tool
// ---------------------------------------------------------------------------

describe('read_file tool', () => {
  it('reads full file content', async () => {
    await writeFile(tmpFile('hello.txt'), 'line1\nline2\nline3');

    const result = parseResult(await readFileTool.invoke({ path: relPath('hello.txt') }));
    expect(result.content).toContain('line1');
    expect(result.content).toContain('line3');
    expect(result.totalLines).toBe(3);
    expect(result.truncated).toBe(false);
  });

  it('respects offset (1-indexed)', async () => {
    await writeFile(tmpFile('offset.txt'), 'a\nb\nc\nd\ne');

    const result = parseResult(await readFileTool.invoke({ path: relPath('offset.txt'), offset: 3 }));
    expect(result.content).toContain('c');
    expect((result.content as string)).not.toContain('a\n');
  });

  it('respects limit and appends continuation hint when more lines remain', async () => {
    await writeFile(tmpFile('limited.txt'), 'a\nb\nc\nd\ne');

    const result = parseResult(
      await readFileTool.invoke({ path: relPath('limited.txt'), offset: 1, limit: 2 }),
    );
    expect(result.content).toContain('a');
    expect(result.content as string).toContain('more lines');
    expect(result.content as string).toContain('offset=3');
  });

  it('throws when offset is beyond end of file', async () => {
    await writeFile(tmpFile('short.txt'), 'one\ntwo');

    await expect(
      readFileTool.invoke({ path: relPath('short.txt'), offset: 99 }),
    ).rejects.toThrow(/beyond end of file/i);
  });

  it('throws when file does not exist', async () => {
    await expect(readFileTool.invoke({ path: relPath('ghost.txt') })).rejects.toThrow();
  });

  it('throws when path escapes sandbox', async () => {
    await expect(readFileTool.invoke({ path: '../../etc/passwd' })).rejects.toThrow(/sandbox/i);
  });

  it('refuses to read secret-bearing files inside the sandbox', async () => {
    for (const name of ['.env', '.env.local', 'server.pem', 'id.KEY']) {
      await writeFile(tmpFile(name), 'SECRET=1');
      await expect(readFileTool.invoke({ path: relPath(name) })).rejects.toThrow(/secrets/i);
    }
  });

  it('isSecretPath flags cramer-short settings and credentials', () => {
    expect(isSecretPath('.cramer-short/settings.json')).toBe(true);
    expect(isSecretPath('.cramer-short/credentials/token.json')).toBe(true);
    expect(isSecretPath('.ENV')).toBe(true);
    expect(isSecretPath('sub\\dir\\.env.production')).toBe(true);
    expect(isSecretPath('.cramer-short/memory/notes.md')).toBe(false);
    expect(isSecretPath('src/environment.ts')).toBe(false);
  });

  it('appends truncation hint when content exceeds line limit (2000 lines)', async () => {
    // Create a file that triggers line-based truncation (> 2000 lines default)
    const lines = Array.from({ length: 2001 }, (_, i) => `line ${i + 1}`).join('\n');
    await writeFile(tmpFile('big.txt'), lines);

    const result = parseResult(await readFileTool.invoke({ path: relPath('big.txt') }));
    expect(result.truncated).toBe(true);
    expect(result.content as string).toContain('offset=');
  });

  it('returns size message when first line exceeds byte limit', async () => {
    // A single line > 500 KB triggers firstLineExceedsLimit
    const longLine = 'X'.repeat(513_000);
    await writeFile(tmpFile('bigline.txt'), longLine);

    const result = parseResult(await readFileTool.invoke({ path: relPath('bigline.txt') }));
    // The content describes the oversized line rather than returning raw content
    expect(result.content as string).toMatch(/exceeds.*limit/i);
    expect(result.content as string).toContain('Use offset=');
  });

  it('reads with both offset and limit (no continuation hint when at EOF)', async () => {
    await writeFile(tmpFile('exact.txt'), 'a\nb\nc');

    // offset=2 limit=2 → reads lines 2-3, which are the last lines, no hint
    const result = parseResult(
      await readFileTool.invoke({ path: relPath('exact.txt'), offset: 2, limit: 2 }),
    );
    expect(result.content).toContain('b');
    expect(result.content).toContain('c');
    expect(result.content as string).not.toContain('more lines');
  });
});

// ---------------------------------------------------------------------------
// edit_file tool
// ---------------------------------------------------------------------------

describe('edit_file tool', () => {
  it('replaces text in a file', async () => {
    await writeFile(tmpFile('edit.txt'), 'Hello world\nGoodbye world\n');

    const result = parseResult(
      await editFileTool.invoke({
        path: relPath('edit.txt'),
        old_text: 'Hello world',
        new_text: 'Hello Cramer-Short',
      }),
    );

    expect(result.message).toContain('Successfully replaced');
    expect(result.path).toBe(relPath('edit.txt'));
  });

  it('writes the replacement back to disk', async () => {
    const file = tmpFile('write-back.txt');
    await writeFile(file, 'original content\nother line\n');

    await editFileTool.invoke({
      path: relPath('write-back.txt'),
      old_text: 'original content',
      new_text: 'updated content',
    });

    const { readFile } = await import('node:fs/promises');
    const updated = await readFile(file, 'utf-8');
    expect(updated).toContain('updated content');
    expect(updated).not.toContain('original content');
  });

  it('includes a diff in the result', async () => {
    await writeFile(tmpFile('diff.txt'), 'foo bar baz\n');

    const result = parseResult(
      await editFileTool.invoke({
        path: relPath('diff.txt'),
        old_text: 'foo',
        new_text: 'qux',
      }),
    );

    expect(result.diff).toBeDefined();
  });

  it('throws when file does not exist', async () => {
    await expect(
      editFileTool.invoke({
        path: relPath('ghost.txt'),
        old_text: 'x',
        new_text: 'y',
      }),
    ).rejects.toThrow(/not found|not writable/i);
  });

  it('throws when old_text is not found in file', async () => {
    await writeFile(tmpFile('no-match.txt'), 'some content here\n');

    await expect(
      editFileTool.invoke({
        path: relPath('no-match.txt'),
        old_text: 'nonexistent text',
        new_text: 'replacement',
      }),
    ).rejects.toThrow(/could not find/i);
  });

  it('throws when old_text has multiple occurrences', async () => {
    await writeFile(tmpFile('multi.txt'), 'foo\nfoo\nfoo\n');

    await expect(
      editFileTool.invoke({
        path: relPath('multi.txt'),
        old_text: 'foo',
        new_text: 'bar',
      }),
    ).rejects.toThrow(/occurrences/i);
  });

  it('throws when replacement produces no changes', async () => {
    await writeFile(tmpFile('noop.txt'), 'hello world\n');

    await expect(
      editFileTool.invoke({
        path: relPath('noop.txt'),
        old_text: 'hello world',
        new_text: 'hello world',
      }),
    ).rejects.toThrow(/No changes made/i);
  });

  it('throws when path escapes sandbox', async () => {
    await expect(
      editFileTool.invoke({
        path: '../../etc/hosts',
        old_text: 'localhost',
        new_text: 'hacked',
      }),
    ).rejects.toThrow(/sandbox/i);
  });

  it('preserves other lines when making targeted replacement', async () => {
    const file = tmpFile('preserve.txt');
    await writeFile(file, 'line 1\nchange me\nline 3\n');

    await editFileTool.invoke({
      path: relPath('preserve.txt'),
      old_text: 'change me',
      new_text: 'changed!',
    });

    const { readFile } = await import('node:fs/promises');
    const content = await readFile(file, 'utf-8');
    expect(content).toContain('line 1');
    expect(content).toContain('changed!');
    expect(content).toContain('line 3');
  });

  it('fuzzy match edits only the matched region and keeps every other byte (PI-3)', async () => {
    const file = tmpFile('fuzzy-bytes.ts');
    await writeFile(file, 'const a = \u201cit\u2019s\u201d;\nconst b = 1;   \nconst c = \u201Cx\u201D;\n');

    // Straight quotes in old_text force the fuzzy path (the file has curly quotes).
    await editFileTool.invoke({
      path: relPath('fuzzy-bytes.ts'),
      old_text: `const a = "it's";`,
      new_text: 'const a = "REPLACED";',
    });

    const { readFile } = await import('node:fs/promises');
    expect(await readFile(file, 'utf-8')).toBe('const a = "REPLACED";\nconst b = 1;   \nconst c = \u201Cx\u201D;\n');
  });

  it('fuzzy match spanning lines with trailing whitespace keeps the surrounding bytes (PI-3)', async () => {
    const file = tmpFile('fuzzy-span.txt');
    await writeFile(file, 'keep   \nfoo   \nbar   \nend\u00A0 \n');

    await editFileTool.invoke({
      path: relPath('fuzzy-span.txt'),
      old_text: 'foo\nbar',
      new_text: 'X',
    });

    const { readFile } = await import('node:fs/promises');
    expect(await readFile(file, 'utf-8')).toBe('keep   \nX   \nend\u00A0 \n');
  });

  it('fuzzy match on a CRLF file keeps CRLF and every untouched byte (PI-3)', async () => {
    const file = tmpFile('fuzzy-crlf.ts');
    await writeFile(file, 'const a = \u201cit\u2019s\u201d;\r\nconst b = 1;   \r\nconst c = \u201Cx\u201D;\r\n');

    await editFileTool.invoke({
      path: relPath('fuzzy-crlf.ts'),
      old_text: `const a = "it's";`,
      new_text: 'const a = "REPLACED";',
    });

    const { readFile } = await import('node:fs/promises');
    expect(await readFile(file, 'utf-8')).toBe(
      'const a = "REPLACED";\r\nconst b = 1;   \r\nconst c = \u201Cx\u201D;\r\n',
    );
  });
});
