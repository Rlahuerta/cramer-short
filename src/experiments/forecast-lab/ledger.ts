import { appendFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import type { ForecastLabLedgerEntry, ForecastLabRunManifest } from './types.js';

export const LEDGER_COLUMNS = [
  'runId',
  'startedAt',
  'profileId',
  'targetSubsystem',
  'candidateBranch',
  'allowedGlobs',
  'baselineSummary',
  'candidateSummary',
  'decision',
  'reason',
  'artifactsPath',
] as const satisfies readonly (keyof ForecastLabLedgerEntry)[];

export const LEDGER_HEADER = LEDGER_COLUMNS.join('\t');

type LedgerColumn = (typeof LEDGER_COLUMNS)[number];
const SAFE_RUN_ID_PATTERN = /^[A-Za-z0-9][A-Za-z0-9_-]*(\.[A-Za-z0-9_-]+)*$/;

export class ForecastLabLedgerError extends Error {
  override name = 'ForecastLabLedgerError';
}

function assertJsonSerializable(field: string, value: unknown): void {
  if (value === undefined) {
    throw new ForecastLabLedgerError(`${field} must not be undefined`);
  }

  if (typeof value === 'function' || typeof value === 'symbol' || typeof value === 'bigint') {
    throw new ForecastLabLedgerError(`${field} is not JSON serializable`);
  }

  if (typeof value === 'number' && !Number.isFinite(value)) {
    throw new ForecastLabLedgerError(`${field} must be a finite number`);
  }

  if (Array.isArray(value)) {
    value.forEach((item, index) => assertJsonSerializable(`${field}[${index}]`, item));
    return;
  }

  if (value && typeof value === 'object') {
    const prototype = Object.getPrototypeOf(value);
    if (prototype !== Object.prototype && prototype !== null) {
      throw new ForecastLabLedgerError(`${field} must be a plain JSON object`);
    }

    for (const [key, item] of Object.entries(value)) {
      assertJsonSerializable(`${field}.${key}`, item);
    }
  }
}

function stableValue(value: unknown): unknown {
  if (Array.isArray(value)) {
    return value.map(stableValue);
  }

  if (value && typeof value === 'object') {
    return Object.fromEntries(
      Object.entries(value)
        .sort(([left], [right]) => left.localeCompare(right))
        .map(([key, item]) => [key, stableValue(item)]),
    );
  }

  return value;
}

export function stableJsonStringify(value: unknown): string {
  assertJsonSerializable('value', value);
  const serialized = JSON.stringify(stableValue(value));

  if (serialized === undefined) {
    throw new ForecastLabLedgerError('value is not JSON serializable');
  }

  return serialized;
}

function requireString(entry: Record<string, unknown>, field: string): string {
  const value = entry[field];

  if (typeof value !== 'string') {
    throw new ForecastLabLedgerError(`${field} must be a string`);
  }

  if (value.trim() === '') {
    throw new ForecastLabLedgerError(`${field} must not be empty`);
  }

  return value;
}

function assertSafeRunId(runId: string): void {
  if (!SAFE_RUN_ID_PATTERN.test(runId)) {
    throw new ForecastLabLedgerError('runId must be a safe path segment');
  }
}

export function validateLedgerEntry(entry: unknown): asserts entry is ForecastLabLedgerEntry {
  if (!entry || typeof entry !== 'object') {
    throw new ForecastLabLedgerError('ledger entry must be an object');
  }

  const record = entry as Record<string, unknown>;
  const runId = requireString(record, 'runId');
  assertSafeRunId(runId);

  for (const field of ['startedAt', 'profileId', 'targetSubsystem', 'candidateBranch', 'reason', 'artifactsPath'] as const) {
    requireString(record, field);
  }

  if (!Array.isArray(record.allowedGlobs) || record.allowedGlobs.some((glob) => typeof glob !== 'string')) {
    throw new ForecastLabLedgerError('allowedGlobs must be an array of strings');
  }

  if (record.decision !== 'keep' && record.decision !== 'drop') {
    throw new ForecastLabLedgerError('decision must be keep or drop');
  }

  assertJsonSerializable('baselineSummary', record.baselineSummary);
  assertJsonSerializable('candidateSummary', record.candidateSummary);
}

export function validateRunManifest(manifest: unknown): asserts manifest is ForecastLabRunManifest {
  if (!manifest || typeof manifest !== 'object') {
    throw new ForecastLabLedgerError('run manifest must be an object');
  }

  const record = manifest as Record<string, unknown>;
  const runId = requireString(record, 'runId');
  assertSafeRunId(runId);

  for (const field of ['startedAt', 'profileId', 'targetSubsystem', 'candidateBranch', 'artifactsPath'] as const) {
    requireString(record, field);
  }

  if (!Array.isArray(record.allowedGlobs) || record.allowedGlobs.some((glob) => typeof glob !== 'string')) {
    throw new ForecastLabLedgerError('allowedGlobs must be an array of strings');
  }

  assertJsonSerializable('manifest', record);
}

export function serializeLedgerRow(entry: ForecastLabLedgerEntry): string {
  validateLedgerEntry(entry);
  return LEDGER_COLUMNS.map((column) => stableJsonStringify(entry[column])).join('\t');
}

export function parseLedgerRow(row: string): ForecastLabLedgerEntry {
  const fields = row.split('\t');

  if (fields.length !== LEDGER_COLUMNS.length) {
    throw new ForecastLabLedgerError(`Malformed ledger row: expected ${LEDGER_COLUMNS.length} fields, got ${fields.length}`);
  }

  const parsed: Partial<Record<LedgerColumn, unknown>> = {};

  for (let index = 0; index < LEDGER_COLUMNS.length; index += 1) {
    const column = LEDGER_COLUMNS[index]!;

    try {
      parsed[column] = JSON.parse(fields[index]!);
    } catch (error) {
      throw new ForecastLabLedgerError(`Invalid JSON in ${column}: ${error instanceof Error ? error.message : String(error)}`);
    }
  }

  validateLedgerEntry(parsed);
  return parsed;
}

function readLedgerText(path: string): string {
  if (!existsSync(path)) {
    return '';
  }

  return readFileSync(path, 'utf8');
}

function assertLedgerHeader(text: string): void {
  const firstLine = text.split(/\r?\n/, 1)[0] ?? '';

  if (firstLine !== LEDGER_HEADER) {
    throw new ForecastLabLedgerError('ledger header does not match expected schema');
  }
}

export function appendLedgerEntry(path: string, entry: ForecastLabLedgerEntry): void {
  const row = serializeLedgerRow(entry);

  mkdirSync(dirname(path), { recursive: true });

  if (!existsSync(path)) {
    try {
      appendFileSync(path, `${LEDGER_HEADER}\n${row}\n`, { encoding: 'utf8', flag: 'wx' });
      return;
    } catch (error) {
      if (!(error && typeof error === 'object' && 'code' in error && error.code === 'EEXIST')) {
        throw error;
      }
    }
  }

  const existing = readLedgerText(path);
  assertLedgerHeader(existing);
  appendFileSync(path, `${existing.endsWith('\n') ? '' : '\n'}${row}\n`, 'utf8');
}

export function readLedgerEntries(path: string): ForecastLabLedgerEntry[] {
  const text = readLedgerText(path);

  if (text.length === 0) {
    return [];
  }

  assertLedgerHeader(text);

  const lines = text.split(/\r?\n/).slice(1);
  if (lines.at(-1) === '') {
    lines.pop();
  }

  return lines.map(parseLedgerRow);
}

export function serializeManifest(manifest: ForecastLabRunManifest): string {
  validateRunManifest(manifest);
  return `${JSON.stringify(stableValue(manifest), null, 2)}\n`;
}

export function writeRunManifest(path: string, manifest: ForecastLabRunManifest): void {
  mkdirSync(dirname(path), { recursive: true });
  writeFileSync(path, serializeManifest(manifest), 'utf8');
}

export function readRunManifest(path: string): ForecastLabRunManifest {
  const manifest: unknown = JSON.parse(readFileSync(path, 'utf8'));
  validateRunManifest(manifest);
  return manifest;
}
