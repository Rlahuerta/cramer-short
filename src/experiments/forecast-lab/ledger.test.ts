import { afterEach, describe, expect, it } from 'bun:test';
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import {
  getExperimentLedgerPath,
  getExperimentRunArtifactsDir,
  getExperimentRunDir,
  getExperimentRunManifestPath,
  getExperimentsDir,
} from '../../utils/paths.js';
import type { ForecastLabLedgerEntry, ForecastLabRunManifest } from './types.js';
import {
  LEDGER_COLUMNS,
  LEDGER_HEADER,
  appendLedgerEntry,
  parseLedgerRow,
  readLedgerEntries,
  serializeLedgerRow,
  serializeManifest,
  stableJsonStringify,
  validateLedgerEntry,
  validateRunManifest,
  writeRunManifest,
  readRunManifest,
} from './ledger.js';

const TEST_ROOT = join(process.cwd(), '.cramer-short', 'experiments', '__ledger_test__');
const PATH_TEST_RUN_ID = 'ledger-path-test';

function makeEntry(overrides: Partial<ForecastLabLedgerEntry> = {}): ForecastLabLedgerEntry {
  return {
    runId: 'run-1',
    startedAt: '2026-05-02T00:00:00.000Z',
    profileId: 'btc-markov-short-horizon',
    targetSubsystem: 'markov-distribution',
    candidateBranch: 'topic/forecast-lab-run-1',
    allowedGlobs: ['src/tools/finance/markov-distribution.ts', 'src/tools/finance/polymarket-forecast.ts'],
    baselineSummary: { z: 2, rankIC: 0.12 },
    candidateSummary: { rankIC: 0.13, lift: 0.01 },
    decision: 'keep',
    reason: 'measurable lift',
    artifactsPath: '.cramer-short/experiments/runs/run-1',
    ...overrides,
  };
}

afterEach(() => {
  rmSync(TEST_ROOT, { recursive: true, force: true });
  rmSync(getExperimentRunDir(PATH_TEST_RUN_ID), { recursive: true, force: true });
});

describe('forecast-lab ledger serialization', () => {
  it('has a stable TSV header', () => {
    expect(LEDGER_COLUMNS).toEqual([
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
    ]);
    expect(LEDGER_HEADER).toBe(
      'runId\tstartedAt\tprofileId\ttargetSubsystem\tcandidateBranch\tallowedGlobs\tbaselineSummary\tcandidateSummary\tdecision\treason\tartifactsPath',
    );
  });

  it('serializes rows as deterministic reversible JSON fields', () => {
    expect(serializeLedgerRow(makeEntry())).toBe(
      '"run-1"\t"2026-05-02T00:00:00.000Z"\t"btc-markov-short-horizon"\t"markov-distribution"\t"topic/forecast-lab-run-1"\t["src/tools/finance/markov-distribution.ts","src/tools/finance/polymarket-forecast.ts"]\t{"rankIC":0.12,"z":2}\t{"lift":0.01,"rankIC":0.13}\t"keep"\t"measurable lift"\t".cramer-short/experiments/runs/run-1"',
    );
  });

  it('keeps object keys stable without sorting arrays', () => {
    const left = stableJsonStringify({ b: 2, a: { z: 3, y: 4 }, c: ['b', 'a'] });
    const right = stableJsonStringify({ c: ['b', 'a'], a: { y: 4, z: 3 }, b: 2 });

    expect(left).toBe('{"a":{"y":4,"z":3},"b":2,"c":["b","a"]}');
    expect(right).toBe(left);
  });

  it('rejects non-plain objects before they can serialize as empty objects', () => {
    expect(() => stableJsonStringify({ at: new Date('2026-05-02T00:00:00.000Z') })).toThrow(/plain JSON object/);
  });

  it('parses serialized rows back into entries', () => {
    const entry = makeEntry({ reason: 'contains tab\tand newline\nsafely' });
    const parsed = parseLedgerRow(serializeLedgerRow(entry));

    expect(parsed).toEqual(entry);
  });
});

describe('forecast-lab ledger validation', () => {
  it('rejects missing and empty run ids', () => {
    expect(() => validateLedgerEntry({ ...makeEntry(), runId: '' })).toThrow(/runId/);
    expect(() => validateLedgerEntry({ ...makeEntry(), runId: undefined })).toThrow(/runId/);
    expect(() => validateLedgerEntry({ ...makeEntry(), runId: '../escape' })).toThrow(/runId/);
    expect(() => validateLedgerEntry({ ...makeEntry(), runId: 'run..id' })).toThrow(/runId/);
    expect(() => validateLedgerEntry({ ...makeEntry(), runId: 'run-id.' })).toThrow(/runId/);
  });

  it('rejects empty required string fields', () => {
    expect(() => validateLedgerEntry({ ...makeEntry(), profileId: '' })).toThrow(/profileId/);
  });

  it('rejects malformed rows with the wrong field count', () => {
    expect(() => parseLedgerRow('"run-1"\t"only-two-fields"')).toThrow(/expected 11 fields/);
  });

  it('rejects malformed rows with invalid JSON fields', () => {
    const fields = LEDGER_COLUMNS.map(() => '"x"');
    fields[0] = '{bad-json';

    expect(() => parseLedgerRow(fields.join('\t'))).toThrow(/Invalid JSON in runId/);
  });

  it('rejects malformed rows with invalid decisions', () => {
    const fields = serializeLedgerRow(makeEntry()).split('\t');
    fields[8] = '"maybe"';

    expect(() => parseLedgerRow(fields.join('\t'))).toThrow(/decision/);
  });
});

describe('forecast-lab ledger file helpers', () => {
  it('appends entries without rewriting the header', () => {
    const ledgerPath = join(TEST_ROOT, 'forecast-results.tsv');
    const first = makeEntry();
    const second = makeEntry({ runId: 'run-2', decision: 'drop', reason: 'no measurable lift' });

    appendLedgerEntry(ledgerPath, first);
    appendLedgerEntry(ledgerPath, second);

    expect(readFileSync(ledgerPath, 'utf8')).toBe(
      `${LEDGER_HEADER}\n${serializeLedgerRow(first)}\n${serializeLedgerRow(second)}\n`,
    );
    expect(readLedgerEntries(ledgerPath)).toEqual([first, second]);
  });

  it('rejects an existing empty ledger as an invalid schema', () => {
    const ledgerPath = join(TEST_ROOT, 'forecast-results.tsv');
    const first = makeEntry();
    mkdirSync(TEST_ROOT, { recursive: true });
    writeFileSync(ledgerPath, '', 'utf8');

    expect(() => appendLedgerEntry(ledgerPath, first)).toThrow(/ledger header does not match expected schema/);
    expect(readFileSync(ledgerPath, 'utf8')).toBe('');
  });

  it('returns no rows when the ledger file does not exist', () => {
    expect(readLedgerEntries(join(TEST_ROOT, 'missing.tsv'))).toEqual([]);
  });
});

describe('forecast-lab manifest helpers', () => {
  it('writes deterministic JSON manifests', () => {
    const manifestPath = join(TEST_ROOT, 'runs', 'run-1', 'manifest.json');
    const manifest: ForecastLabRunManifest = {
      runId: 'run-1',
      startedAt: '2026-05-02T00:00:00.000Z',
      profileId: 'btc-markov-short-horizon',
      targetSubsystem: 'markov-distribution',
      candidateBranch: 'topic/forecast-lab-run-1',
      allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
      artifactsPath: '.cramer-short/experiments/runs/run-1',
    };

    writeRunManifest(manifestPath, manifest);

    expect(readFileSync(manifestPath, 'utf8')).toBe(serializeManifest(manifest));
    expect(readRunManifest(manifestPath)).toEqual(manifest);
    expect(serializeManifest(manifest)).toBe(
      '{\n  "allowedGlobs": [\n    "src/tools/finance/markov-distribution.ts"\n  ],\n  "artifactsPath": ".cramer-short/experiments/runs/run-1",\n  "candidateBranch": "topic/forecast-lab-run-1",\n  "profileId": "btc-markov-short-horizon",\n  "runId": "run-1",\n  "startedAt": "2026-05-02T00:00:00.000Z",\n  "targetSubsystem": "markov-distribution"\n}\n',
    );
  });

  it('rejects malformed manifests when validating or reading', () => {
    expect(() => validateRunManifest({ runId: '', allowedGlobs: [] })).toThrow(/runId/);
    expect(() => validateRunManifest({ ...makeEntry(), runId: '../escape' })).toThrow(/runId/);

    const manifestPath = join(TEST_ROOT, 'runs', 'bad-run', 'manifest.json');
    mkdirSync(join(TEST_ROOT, 'runs', 'bad-run'), { recursive: true });
    writeFileSync(
      manifestPath,
      JSON.stringify({
        runId: 'bad-run',
        startedAt: '2026-05-02T00:00:00.000Z',
        profileId: 123,
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/bad-run',
        allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
        artifactsPath: '.cramer-short/experiments/runs/bad-run',
      }),
      'utf8',
    );

    expect(() => readRunManifest(manifestPath)).toThrow(/profileId/);
  });
});

describe('experiment path helpers', () => {
  it('returns deterministic experiment paths', () => {
    expect(getExperimentsDir()).toBe(join('.cramer-short', 'experiments'));
    expect(getExperimentLedgerPath()).toBe(join('.cramer-short', 'experiments', 'forecast-results.tsv'));
    expect(getExperimentRunDir('run-1')).toBe(join('.cramer-short', 'experiments', 'runs', 'run-1'));
    expect(getExperimentRunArtifactsDir('run-1')).toBe(join('.cramer-short', 'experiments', 'runs', 'run-1', 'artifacts'));
    expect(getExperimentRunManifestPath('run-1')).toBe(join('.cramer-short', 'experiments', 'runs', 'run-1', 'manifest.json'));
  });

  it('creates experiment directories only when requested', () => {
    const ledgerPath = getExperimentLedgerPath({ create: true });
    const artifactsDir = getExperimentRunArtifactsDir(PATH_TEST_RUN_ID, { create: true });
    const manifestPath = getExperimentRunManifestPath(PATH_TEST_RUN_ID, { create: true });

    expect(ledgerPath).toBe(join('.cramer-short', 'experiments', 'forecast-results.tsv'));
    expect(existsSync(getExperimentsDir())).toBe(true);
    expect(existsSync(artifactsDir)).toBe(true);
    expect(manifestPath).toBe(join('.cramer-short', 'experiments', 'runs', PATH_TEST_RUN_ID, 'manifest.json'));
    expect(existsSync(getExperimentRunDir(PATH_TEST_RUN_ID))).toBe(true);
  });

  it('rejects unsafe run ids before building per-run paths', () => {
    expect(() => getExperimentRunDir('')).toThrow(/runId/);
    expect(() => getExperimentRunArtifactsDir('../escape')).toThrow(/runId/);
    expect(() => getExperimentRunManifestPath('nested/run')).toThrow(/runId/);
    expect(() => getExperimentRunDir('run..id')).toThrow(/runId/);
    expect(() => getExperimentRunDir('run-id.')).toThrow(/runId/);
  });
});
