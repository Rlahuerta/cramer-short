import { describe, expect, it } from 'bun:test';
import { mkdirSync, mkdtempSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

describe('headless commands', () => {
  it('exit once their work is done, even when something left a timer open', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'headless-exit-'));
    // Stands in for whatever an agent run leaves behind (keep-alive sockets, library timers).
    const linger = join(dir, 'linger.ts');
    writeFileSync(linger, 'setInterval(() => {}, 1_000);\n');
    mkdirSync(join(dir, '.cramer-short'));
    writeFileSync(join(dir, '.cramer-short', 'schedules.json'), '[]');

    const started = Date.now();
    const child = Bun.spawn(['bun', '--preload', linger, join(import.meta.dir, 'index.tsx'), 'schedule', 'list'], {
      env: { ...process.env, HOME: dir },
      stdout: 'ignore',
      stderr: 'ignore',
    });
    const timer = setTimeout(() => child.kill(), 20_000);
    const code = await child.exited;
    clearTimeout(timer);

    expect(code).toBe(0);
    expect(Date.now() - started).toBeLessThan(20_000);
  }, 30_000);
});
