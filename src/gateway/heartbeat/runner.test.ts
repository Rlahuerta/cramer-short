import { afterEach, describe, expect, it, setSystemTime } from 'bun:test';
import { isWithinActiveHours } from './runner.js';

const NY = 'America/New_York';

afterEach(() => {
  setSystemTime();
});

describe('isWithinActiveHours', () => {
  // 2026-10-26 is a Monday; New York is on EDT (UTC-4).
  const at = (utc: string) => setSystemTime(new Date(`2026-10-26T${utc}Z`));

  it('accepts an unpadded start hour once the clock passes 10:00', () => {
    at('14:05:00'); // 10:05 ET
    expect(isWithinActiveHours({ start: '9:30', end: '16:00', timezone: NY })).toBe(true);
  });

  it('accepts an unpadded end hour', () => {
    at('12:45:00'); // 08:45 ET
    expect(isWithinActiveHours({ start: '8:00', end: '9:30', timezone: NY })).toBe(true);
  });

  it('still rejects times before the start and after the end', () => {
    at('13:00:00'); // 09:00 ET
    expect(isWithinActiveHours({ start: '9:30', end: '16:00', timezone: NY })).toBe(false);
    at('20:30:00'); // 16:30 ET
    expect(isWithinActiveHours({ start: '09:30', end: '16:00', timezone: NY })).toBe(false);
  });

  it('keeps both bounds inclusive', () => {
    at('13:30:00'); // 09:30 ET
    expect(isWithinActiveHours({ start: '9:30', end: '16:00', timezone: NY })).toBe(true);
    at('20:00:00'); // 16:00 ET
    expect(isWithinActiveHours({ start: '9:30', end: '16:00', timezone: NY })).toBe(true);
  });
});
