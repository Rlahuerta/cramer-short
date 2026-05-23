export const FIXED_TEST_NOW_ISO = '2026-01-15T12:00:00.000Z';
export const FIXED_TEST_NOW_MS = Date.parse(FIXED_TEST_NOW_ISO);
export const FIXED_TEST_DATE = new Date(FIXED_TEST_NOW_MS);
export const FIXED_TEST_TODAY = FIXED_TEST_NOW_ISO.slice(0, 10);

let testIdCounter = 0;
let randomState = 0x12345678;

export function nextTestId(prefix = 'test'): string {
  testIdCounter += 1;
  return `${prefix}-${testIdCounter}`;
}

export function deterministicRandom(): number {
  randomState ^= randomState << 13;
  randomState ^= randomState >>> 17;
  randomState ^= randomState << 5;
  return (randomState >>> 0) / 0x1_0000_0000;
}
