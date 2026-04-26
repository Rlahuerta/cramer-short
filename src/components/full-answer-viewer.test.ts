import { describe, expect, it, mock } from 'bun:test';
import type { TUI } from '@mariozechner/pi-tui';
import { FullAnswerViewerComponent } from './full-answer-viewer.js';

const fakeTui = { requestRender: mock(() => {}) } as unknown as TUI;

describe('FullAnswerViewerComponent', () => {
  it('renders markdown content and keeps tables as renderable markdown output', () => {
    const viewer = new FullAnswerViewerComponent(
      fakeTui,
      '| A | B |\n|---|---|\n| 1 | 2 |',
      () => {},
      { viewportRows: 12 },
    );

    const lines = viewer.render(40).join('\n');
    expect(lines).toContain('A');
    expect(lines).toContain('B');
    expect(lines).toContain('1');
    expect(lines).toContain('2');
  });

  it('closes on escape', () => {
    const onClose = mock(() => {});
    const viewer = new FullAnswerViewerComponent(fakeTui, 'hello', onClose, { viewportRows: 12 });
    viewer.handleInput('\u001b');
    expect(onClose).toHaveBeenCalled();
  });

  it('scrolls down and up with vim-style keys', () => {
    const viewer = new FullAnswerViewerComponent(
      fakeTui,
      Array.from({ length: 40 }, (_, i) => `line ${i + 1}`).join('\n'),
      () => {},
      { viewportRows: 10 },
    );

    const before = viewer.render(40).join('\n');
    viewer.handleInput('j');
    const afterDown = viewer.render(40).join('\n');
    viewer.handleInput('k');
    const afterUp = viewer.render(40).join('\n');

    expect(afterDown).not.toBe(before);
    expect(afterUp).toBe(before);
  });

  it('sanitizes terminal control sequences before rendering', () => {
    const viewer = new FullAnswerViewerComponent(
      fakeTui,
      'safe \u001b[31mred\u001b[0m \u001b]8;;https://example.com\u0007link\u001b]8;;\u0007',
      () => {},
      { viewportRows: 12 },
    );

    const lines = viewer.render(80).join('\n');

    expect(lines).toContain('safe red link');
    expect(lines).not.toContain('\u001b');
  });

  it('can scroll all the way to the bottom', () => {
    const viewer = new FullAnswerViewerComponent(
      fakeTui,
      Array.from({ length: 20 }, (_, i) => `line ${i + 1}`).join('\n'),
      () => {},
      { viewportRows: 10 },
    );

    viewer.handleInput('G');
    const lines = viewer.render(40).join('\n');

    expect(lines).toContain('line 20');
    expect(lines).not.toContain('↓ 2 lines below');
  });
});
