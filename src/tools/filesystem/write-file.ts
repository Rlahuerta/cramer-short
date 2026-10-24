import { DynamicStructuredTool } from '@langchain/core/tools';
import { mkdir, writeFile } from 'node:fs/promises';
import { homedir } from 'node:os';
import { dirname, relative } from 'node:path';
import { z } from 'zod';
import { formatToolResult } from '../types.js';
import { assertSandboxPath } from './sandbox.js';

export const WRITE_FILE_DESCRIPTION = `
Create or overwrite files in the local workspace or home directory.

## When to Use

- Creating a new file with full contents
- Replacing an existing file entirely
- Writing generated output to disk (e.g. ~/reports/analysis.md)

## When NOT to Use

- Small surgical updates in an existing file (use \`edit_file\`)
- Reading file contents (use \`read_file\`)

## Usage Notes

- The system will prompt the user for confirmation automatically; just call the tool directly
- Accepts \`path\` and full \`content\`
- Creates parent directories when they do not exist
- Overwrites existing file content completely
- Supports ~ home-directory expansion (e.g. ~/reports/thesis.md)
`.trim();

const writeFileSchema = z.object({
  path: z.string().max(4096).describe('Path to the file to write (relative, absolute, or ~/... home-relative).'),
  content: z.string().max(100_000).describe('Content to write to the file.'),
});

/** Writes UTF-8 text content to disk. */
export const writeFileTool = new DynamicStructuredTool({
  name: 'write_file',
  description:
    'Create or overwrite a file inside the workspace or home directory. Automatically creates parent directories when needed.',
  schema: writeFileSchema,
  func: async (input) => {
    const cwd = process.cwd();
    // Allow ~/... and absolute paths by sandboxing to homedir instead of cwd.
    const isHomePath = input.path.startsWith('~/') || input.path === '~';
    const root = isHomePath ? homedir() : cwd;
    const { resolved } = await assertSandboxPath({
      filePath: input.path,
      cwd,
      root,
    });

    // Narrow ~/ writes to allowed subdirectories only: ~/reports/ and ~/.cramer-short/.
    // Prevents overwriting dotfiles (~/.bashrc, ~/.ssh/authorized_keys) even if
    // the approval gate is bypassed. CWD-relative writes are unaffected.
    if (isHomePath) {
      const ALLOWED_HOME_SUBDIRS = ['reports', '.cramer-short'];
      const rel = relative(homedir(), resolved);
      const firstSegment = rel.split(/[\\/]/).filter(Boolean)[0] ?? '';
      if (!ALLOWED_HOME_SUBDIRS.includes(firstSegment)) {
        throw new Error(
          `Path outside allowed home directories: ~/reports/ and ~/.cramer-short/ are the only writable home paths. Got: ${input.path}`,
        );
      }
    }

    const dir = dirname(resolved);
    await mkdir(dir, { recursive: true });
    await writeFile(resolved, input.content, 'utf-8');

    return formatToolResult({
      path: input.path,
      bytesWritten: Buffer.byteLength(input.content, 'utf-8'),
      message: `Successfully wrote ${input.content.length} characters to ${input.path}`,
    });
  },
});
