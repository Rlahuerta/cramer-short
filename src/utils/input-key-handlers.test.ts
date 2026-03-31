import { describe, expect, it } from 'bun:test';
import { cursorHandlers, type CursorContext } from './input-key-handlers.js';

function ctx(text: string, cursorPosition: number): CursorContext {
  return { text, cursorPosition };
}

// ---------------------------------------------------------------------------
// moveLeft
// ---------------------------------------------------------------------------

describe('cursorHandlers.moveLeft', () => {
  it('moves cursor one position left', () => {
    expect(cursorHandlers.moveLeft(ctx('hello', 3))).toBe(2);
  });

  it('clamps at 0 when already at the start', () => {
    expect(cursorHandlers.moveLeft(ctx('hello', 0))).toBe(0);
  });

  it('moves from the end of a string', () => {
    expect(cursorHandlers.moveLeft(ctx('hi', 2))).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// moveRight
// ---------------------------------------------------------------------------

describe('cursorHandlers.moveRight', () => {
  it('moves cursor one position right', () => {
    expect(cursorHandlers.moveRight(ctx('hello', 2))).toBe(3);
  });

  it('clamps at text length when already at the end', () => {
    expect(cursorHandlers.moveRight(ctx('hello', 5))).toBe(5);
  });

  it('moves from the start of a string', () => {
    expect(cursorHandlers.moveRight(ctx('hi', 0))).toBe(1);
  });
});

// ---------------------------------------------------------------------------
// moveToLineStart
// ---------------------------------------------------------------------------

describe('cursorHandlers.moveToLineStart', () => {
  it('moves to 0 when on the first line', () => {
    expect(cursorHandlers.moveToLineStart(ctx('hello world', 5))).toBe(0);
  });

  it('moves to the start of a subsequent line', () => {
    // "line1\nline2" — cursor at position 8 (in "line2")
    const text = 'line1\nline2';
    expect(cursorHandlers.moveToLineStart(ctx(text, 8))).toBe(6);
  });

  it('stays at start when cursor is already at line start', () => {
    expect(cursorHandlers.moveToLineStart(ctx('hello', 0))).toBe(0);
  });
});

// ---------------------------------------------------------------------------
// moveToLineEnd
// ---------------------------------------------------------------------------

describe('cursorHandlers.moveToLineEnd', () => {
  it('moves to the end of the only line', () => {
    expect(cursorHandlers.moveToLineEnd(ctx('hello', 0))).toBe(5);
  });

  it('moves to the end of the first line in multi-line text', () => {
    const text = 'line1\nline2';
    expect(cursorHandlers.moveToLineEnd(ctx(text, 0))).toBe(5);
  });

  it('stays at end when cursor is already at line end', () => {
    expect(cursorHandlers.moveToLineEnd(ctx('hello', 5))).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// moveUp
// ---------------------------------------------------------------------------

describe('cursorHandlers.moveUp', () => {
  it('returns null when on the first (only) line', () => {
    expect(cursorHandlers.moveUp(ctx('hello', 3))).toBeNull();
  });

  it('returns null when on the first line of multi-line text', () => {
    const text = 'line1\nline2';
    expect(cursorHandlers.moveUp(ctx(text, 2))).toBeNull();
  });

  it('moves up one line maintaining column position', () => {
    // "abc\ndef" — cursor at position 5 (column 1 of "def")
    const text = 'abc\ndef';
    const pos = cursorHandlers.moveUp(ctx(text, 5));
    expect(pos).not.toBeNull();
    // column 1 of line 0 ("abc") → position 1
    expect(pos).toBe(1);
  });

  it('clamps column when the upper line is shorter', () => {
    const text = 'ab\nlong line here';
    // cursor at column 6 of line 1 → "lo|ng ..."
    const pos = cursorHandlers.moveUp(ctx(text, 3 + 6)); // 3=offset for 'ab\n', 6=col
    expect(pos).not.toBeNull();
    // "ab" has 2 chars; column clamped to 2
    expect(pos).toBeLessThanOrEqual(2);
  });
});

// ---------------------------------------------------------------------------
// moveDown
// ---------------------------------------------------------------------------

describe('cursorHandlers.moveDown', () => {
  it('returns null when on the last (only) line', () => {
    expect(cursorHandlers.moveDown(ctx('hello', 2))).toBeNull();
  });

  it('returns null when on the last line of multi-line text', () => {
    const text = 'line1\nline2';
    expect(cursorHandlers.moveDown(ctx(text, 8))).toBeNull();
  });

  it('moves down one line maintaining column position', () => {
    // "abc\ndef" — cursor at position 1 (column 1 of "abc")
    const text = 'abc\ndef';
    const pos = cursorHandlers.moveDown(ctx(text, 1));
    expect(pos).not.toBeNull();
    // column 1 of line 1 ("def") → position 5
    expect(pos).toBe(5);
  });
});

// ---------------------------------------------------------------------------
// moveWordBackward
// ---------------------------------------------------------------------------

describe('cursorHandlers.moveWordBackward', () => {
  it('moves to start of current word when in the middle', () => {
    // "hello world" cursor at 8 (inside "world")
    const pos = cursorHandlers.moveWordBackward(ctx('hello world', 8));
    expect(pos).toBeLessThan(8);
  });

  it('stays at 0 when already at the start', () => {
    expect(cursorHandlers.moveWordBackward(ctx('hello', 0))).toBe(0);
  });

  it('moves backward across a word boundary', () => {
    // Cursor right after "hello " (position 6) should land at 0 (start of "hello")
    const pos = cursorHandlers.moveWordBackward(ctx('hello world', 6));
    expect(pos).toBeLessThan(6);
  });
});

// ---------------------------------------------------------------------------
// moveWordForward
// ---------------------------------------------------------------------------

describe('cursorHandlers.moveWordForward', () => {
  it('moves to end of current/next word', () => {
    // "hello world" cursor at 0 → should move past "hello"
    const pos = cursorHandlers.moveWordForward(ctx('hello world', 0));
    expect(pos).toBeGreaterThan(0);
  });

  it('stays at text length when already at the end', () => {
    const text = 'hello';
    expect(cursorHandlers.moveWordForward(ctx(text, text.length))).toBe(text.length);
  });

  it('advances forward across a word boundary', () => {
    const text = 'hello world';
    const pos = cursorHandlers.moveWordForward(ctx(text, 5)); // after "hello"
    expect(pos).toBeGreaterThan(5);
  });
});
