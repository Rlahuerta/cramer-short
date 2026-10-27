import { afterAll, beforeAll, describe, expect, it } from 'bun:test';
import chalk from 'chalk';
import { theme } from './theme.js';

// chalk disables colour off a TTY; force truecolor so the escape codes are observable,
// and restore the shared singleton afterwards (the unit tests share one process).
let previousLevel: typeof chalk.level;
beforeAll(() => {
  previousLevel = chalk.level;
  chalk.level = 3;
});
afterAll(() => {
  chalk.level = previousLevel;
});

describe('theme semantic colors', () => {
  it('renders the named-colour entries as their ANSI colours, not black', () => {
    expect(theme.success('X')).toBe(chalk.green('X'));
    expect(theme.error('X')).toBe(chalk.red('X'));
    expect(theme.warning('X')).toBe(chalk.yellow('X'));
    expect(theme.accent('X')).toBe(chalk.cyan('X'));
    for (const key of ['success', 'error', 'warning', 'accent'] as const) {
      expect(theme[key]('X')).not.toContain('38;2;0;0;0');
    }
  });

  it('still renders hex entries through chalk.hex', () => {
    expect(theme.primary('X')).toBe(chalk.hex('#258bff')('X'));
    expect(theme.muted('X')).toBe(chalk.hex('#a6a6a6')('X'));
  });
});
