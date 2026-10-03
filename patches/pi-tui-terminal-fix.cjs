#!/usr/bin/env node
/**
 * Postinstall patch for @mariozechner/pi-tui terminal.js
 *
 * Fixes: Ctrl+C exit leaves terminal in broken state (no echo, Kitty protocol
 * escape sequences leak as text). Root cause: terminal.stop() calls
 * process.stdin.pause() BEFORE setRawMode(), causing setRawMode to silently
 * fail on some platforms/SSH sessions.
 *
 * This patch reorders stop() to call setRawMode(wasRaw) BEFORE pause().
 * It also adds try-catch around setRawMode since stdin may already be closed.
 *
 * Run automatically via postinstall hook.
 */

const fs = require('node:fs');
const path = require('node:path');

const terminalPath = path.join(
  __dirname,
  '..',
  'node_modules',
  '@mariozechner',
  'pi-tui',
  'dist',
  'terminal.js',
);

if (!fs.existsSync(terminalPath)) {
  console.log('⏭  pi-tui terminal.js not found, skipping patch');
  process.exit(0);
}

let content = fs.readFileSync(terminalPath, 'utf8');

// Check if patch is already applied (look for our comment marker)
if (content.includes('Restore raw mode state BEFORE pausing stdin')) {
  console.log('✅ pi-tui terminal.js patch already applied');
  process.exit(0);
}

// The original stop() method in pi-tui v0.52.12 calls setRawMode AFTER
// removing handlers and pausing stdin. We need to move setRawMode BEFORE
// the StdinBuffer cleanup and pause() call.
//
// Original pattern (simplified):
//   stop() {
//     ...disable bracketed paste & kitty...
//     // Clean up StdinBuffer
//     if (this.stdinBuffer) { ... }
//     // Remove event handlers
//     ...
//     if (process.stdin.setRawMode) {
//       process.stdin.setRawMode(this.wasRaw);
//     }
//     process.stdin.pause();
//   }
//
// Patched pattern:
//   stop() {
//     ...disable bracketed paste & kitty...
//     // Restore raw mode state BEFORE pausing stdin.
//     if (process.stdin.setRawMode) {
//       try {
//         process.stdin.setRawMode(this.wasRaw);
//       } catch { /* best-effort */ }
//     }
//     // Clean up StdinBuffer
//     if (this.stdinBuffer) { ... }
//     // Remove event handlers
//     ...
//     process.stdin.pause();
//   }

const originalBlock =
  /\/\/ Clean up StdinBuffer\n\s+if \(this\.stdinBuffer\) \{/;

const replacementBlock = `// Restore raw mode state BEFORE pausing stdin.
    // Calling setRawMode after pause() can silently fail on some
    // platforms/SSH sessions, leaving the terminal in raw mode (no echo,
    // no signal generation) which breaks the parent shell.
    if (process.stdin.setRawMode) {
        try {
            process.stdin.setRawMode(this.wasRaw);
        } catch {
            // setRawMode may throw if stdin is already closed; best-effort.
        }
    }
    // Clean up StdinBuffer
    if (this.stdinBuffer) {`;

content = content.replace(originalBlock, replacementBlock);

// Now remove the old setRawMode call that was after the handler cleanup.
// It's in a different location in the original file — after the event handler
// removal section and before pause().
const oldSetRawMode = /\n\s+if \(process\.stdin\.setRawMode\) \{\n\s+process\.stdin\.setRawMode\(this\.wasRaw\);\n\s+\}/;

if (oldSetRawMode.test(content)) {
  content = content.replace(oldSetRawMode, '');
}

fs.writeFileSync(terminalPath, content, 'utf8');
console.log('✅ pi-tui terminal.js patched: setRawMode() now runs before pause()');