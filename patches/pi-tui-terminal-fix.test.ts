import { afterEach, describe, expect, it } from 'bun:test';
import { copyFileSync, mkdirSync, mkdtempSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

// The patch resolves terminal.js relative to its own location
// (<root>/patches/../node_modules/@mariozechner/pi-tui/dist/terminal.js), so each
// test copies the script into a throwaway root with a fixture terminal.js and
// never touches the real node_modules.
const PATCH_SCRIPT = join(import.meta.dir, 'pi-tui-terminal-fix.cjs');
const MARKER = 'Restore raw mode state BEFORE pausing stdin';

// pi-tui 0.52.12 stop() layout before patching: setRawMode runs after the
// handler cleanup and just before pause().
const ORIGINAL_STOP = `class ProcessTerminal {
    stop() {
        // Disable bracketed paste mode
        process.stdout.write("\\x1b[?2004l");
        // Clean up StdinBuffer
        if (this.stdinBuffer) {
            this.stdinBuffer.destroy();
            this.stdinBuffer = undefined;
        }
        // Remove event handlers
        if (this.stdinDataHandler) {
            process.stdin.removeListener("data", this.stdinDataHandler);
            this.stdinDataHandler = undefined;
        }
        if (process.stdin.setRawMode) {
            process.stdin.setRawMode(this.wasRaw);
        }
        process.stdin.pause();
    }
}
`;

const roots: string[] = [];

afterEach(() => {
  for (const root of roots.splice(0)) rmSync(root, { recursive: true, force: true });
});

function runPatchAgainst(terminalJs: string): { status: number; stdout: string; stderr: string; after: string } {
  const root = mkdtempSync(join(tmpdir(), 'pi-tui-patch-'));
  roots.push(root);
  mkdirSync(join(root, 'patches'));
  copyFileSync(PATCH_SCRIPT, join(root, 'patches', 'pi-tui-terminal-fix.cjs'));
  const distDir = join(root, 'node_modules', '@mariozechner', 'pi-tui', 'dist');
  mkdirSync(distDir, { recursive: true });
  const terminalPath = join(distDir, 'terminal.js');
  writeFileSync(terminalPath, terminalJs, 'utf8');

  const result = Bun.spawnSync([process.execPath, join(root, 'patches', 'pi-tui-terminal-fix.cjs')], {
    cwd: root,
  });
  return {
    status: result.exitCode,
    stdout: result.stdout.toString(),
    stderr: result.stderr.toString(),
    after: readFileSync(terminalPath, 'utf8'),
  };
}

describe('pi-tui-terminal-fix patch', () => {
  it('moves setRawMode before pause() when the original layout matches', () => {
    const { status, after } = runPatchAgainst(ORIGINAL_STOP);

    expect(status).toBe(0);
    expect(after).toContain(MARKER);
    expect(after.indexOf('setRawMode(this.wasRaw)')).toBeLessThan(after.indexOf('process.stdin.pause()'));
    expect(after.match(/setRawMode\(this\.wasRaw\)/g)).toHaveLength(1);
  });

  it('refuses to patch, and keeps the original setRawMode call, when the insertion anchor is missing', () => {
    // Anchor comment renamed, original setRawMode call still present.
    const drifted = ORIGINAL_STOP.replace('// Clean up StdinBuffer', '// Tear down the stdin buffer');

    const { status, stdout, after } = runPatchAgainst(drifted);

    expect(status).not.toBe(0);
    expect(stdout).not.toContain('terminal.js patched');
    expect(after).toBe(drifted);
    expect(after).toContain('process.stdin.setRawMode(this.wasRaw);');
  });

  it('fails instead of reporting success when nothing matches', () => {
    const unrelated = 'class ProcessTerminal { stop() { process.stdin.pause(); } }\n';

    const { status, stdout, after } = runPatchAgainst(unrelated);

    expect(status).not.toBe(0);
    expect(stdout).not.toContain('terminal.js patched');
    expect(after).toBe(unrelated);
  });

  it('is a no-op that exits 0 once the marker is present', () => {
    const patched = `// ${MARKER}\n${ORIGINAL_STOP}`;

    const { status, after } = runPatchAgainst(patched);

    expect(status).toBe(0);
    expect(after).toBe(patched);
  });
});
