import { join } from 'node:path';

const CRAMER_SHORT_DIR = '.cramer-short';

export function getCramerShortDir(): string {
  return CRAMER_SHORT_DIR;
}

export function cramerShortPath(...segments: string[]): string {
  return join(getCramerShortDir(), ...segments);
}
