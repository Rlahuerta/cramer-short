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
import { listForecastLabStructuredMutations } from './profiles.js';

const SHORTER_REACTIVE_WINDOW = listForecastLabStructuredMutations('multi-asset-markov-short-horizon')
  .find((candidate) => candidate.id === 'markov-shorter-reactive-window');

if (!SHORTER_REACTIVE_WINDOW) {
  throw new Error('Missing shorter-reactive-window test fixture');
}

function makeMutationReplayPayload() {
  return {
    kind: 'markov-parameter-candidate' as const,
    id: SHORTER_REACTIVE_WINDOW.id,
    profileId: SHORTER_REACTIVE_WINDOW.profileId,
    mutatorId: SHORTER_REACTIVE_WINDOW.mutatorId,
    specSummary: SHORTER_REACTIVE_WINDOW.specSummary,
    patchSummary: [...SHORTER_REACTIVE_WINDOW.patchSummary],
    edits: SHORTER_REACTIVE_WINDOW.edits.map((edit) => ({ ...edit })),
  };
}

function serializeLegacyLedgerField(value: unknown): string {
  return stableJsonStringify(value === undefined ? null : value);
}

const TEST_ROOT = join(process.cwd(), '.cramer-short', 'experiments', '__ledger_test__');
const PATH_TEST_RUN_ID = 'ledger-path-test';
const LEGACY_LEDGER_HEADER = [
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
].join('\t');
const PRE_CONTRACT_LEDGER_HEADER = [
  'runId',
  'startedAt',
  'profileId',
  'targetSubsystem',
  'candidateBranch',
  'allowedGlobs',
  'mutationMode',
  'lineage',
  'mutationSpecSummary',
  'candidateWorkspace',
  'baselineSummary',
  'candidateSummary',
  'decision',
  'reason',
  'artifactsPath',
].join('\t');
const PRE_ROUTING_LEDGER_HEADER = [
  'runId',
  'startedAt',
  'profileId',
  'targetSubsystem',
  'candidateBranch',
  'allowedGlobs',
  'effectiveMutationContract',
  'mutationMode',
  'parentRunId',
  'mutationId',
  'mutationSummary',
  'lineage',
  'mutationSpecSummary',
  'candidateWorkspace',
  'baselineSummary',
  'candidateSummary',
  'decision',
  'reason',
  'artifactsPath',
].join('\t');
const PRE_PROMOTION_LEDGER_HEADER = [
  'runId',
  'startedAt',
  'profileId',
  'targetSubsystem',
  'candidateBranch',
  'allowedGlobs',
  'routingContext',
  'effectiveMutationContract',
  'mutationMode',
  'parentRunId',
  'mutationId',
  'mutationSummary',
  'lineage',
  'mutationSpecSummary',
  'candidateWorkspace',
  'baselineSummary',
  'candidateSummary',
  'decision',
  'reason',
  'artifactsPath',
].join('\t');

function makeEntry(overrides: Partial<ForecastLabLedgerEntry> = {}): ForecastLabLedgerEntry {
  return {
    runId: 'run-1',
    startedAt: '2026-05-02T00:00:00.000Z',
    profileId: 'multi-asset-markov-short-horizon',
    targetSubsystem: 'markov-distribution',
    candidateBranch: 'topic/forecast-lab-run-1',
    allowedGlobs: ['src/tools/finance/markov-distribution.ts', 'src/tools/finance/polymarket-forecast.ts'],
    effectiveMutationContract: {
      mode: 'structured',
      mutableFiles: ['src/tools/finance/markov-distribution.ts', 'src/tools/finance/polymarket-forecast.ts'],
      allowedMutatorIds: ['replace-range', 'search-replace'],
      allowMultipleCandidateAttempts: false,
    },
    baselineSummary: { z: 2, rankIC: 0.12 },
    candidateSummary: { rankIC: 0.13, lift: 0.01 },
    decision: 'keep',
    reason: 'measurable lift',
    artifactsPath: '.cramer-short/experiments/runs/run-1',
    ...overrides,
  };
}

function makeMutationMetadata(): Pick<
  ForecastLabLedgerEntry,
  'mutationMode' | 'parentRunId' | 'mutationId' | 'mutationSummary' | 'lineage' | 'mutationSpecSummary' | 'candidateWorkspace'
> {
  return {
    mutationMode: 'structured',
    parentRunId: 'run-0',
    mutationId: SHORTER_REACTIVE_WINDOW.id,
    mutationSummary: SHORTER_REACTIVE_WINDOW.specSummary.summary,
    lineage: {
      rootRunId: 'run-0',
      parentRunId: 'run-0',
      generation: 1,
    },
    mutationSpecSummary: {
      mutatorId: SHORTER_REACTIVE_WINDOW.specSummary.mutatorId,
      targetFiles: [...SHORTER_REACTIVE_WINDOW.specSummary.targetFiles],
      summary: SHORTER_REACTIVE_WINDOW.specSummary.summary,
    },
    candidateWorkspace: {
      kind: 'current-worktree',
      rootDir: '/repo',
      branch: 'topic/forecast-lab-run-1',
    },
  };
}

function makeRoutingContext() {
  return {
    originatingQuery: 'Improve the short-horizon Markov calibration.',
    selectedProfileId: 'multi-asset-markov-short-horizon',
    routerReason: 'Matched improvement intent and Markov short-horizon routing keywords.',
    invocationSource: 'auto-routed' as const,
  };
}

function makePendingPromotion(runId = 'run-1') {
  return {
    status: 'approval-required' as const,
    source: {
      runId,
      manifestPath: getExperimentRunManifestPath(runId),
    },
    requestedAt: '2026-05-02T00:30:00.000Z',
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
      'routingContext',
      'effectiveMutationContract',
      'mutationMode',
      'parentRunId',
      'mutationId',
      'mutationSummary',
      'lineage',
      'mutationSpecSummary',
      'candidateWorkspace',
      'promotion',
      'baselineSummary',
      'candidateSummary',
      'decision',
      'reason',
      'artifactsPath',
    ]);
    expect(LEDGER_HEADER).toBe(
      'runId\tstartedAt\tprofileId\ttargetSubsystem\tcandidateBranch\tallowedGlobs\troutingContext\teffectiveMutationContract\tmutationMode\tparentRunId\tmutationId\tmutationSummary\tlineage\tmutationSpecSummary\tcandidateWorkspace\tpromotion\tbaselineSummary\tcandidateSummary\tdecision\treason\tartifactsPath',
    );
  });

  it('serializes rows as deterministic reversible JSON fields', () => {
    expect(serializeLedgerRow(makeEntry())).toBe(
      '"run-1"\t"2026-05-02T00:00:00.000Z"\t"multi-asset-markov-short-horizon"\t"markov-distribution"\t"topic/forecast-lab-run-1"\t["src/tools/finance/markov-distribution.ts","src/tools/finance/polymarket-forecast.ts"]\tnull\t{"allowedMutatorIds":["replace-range","search-replace"],"allowMultipleCandidateAttempts":false,"mode":"structured","mutableFiles":["src/tools/finance/markov-distribution.ts","src/tools/finance/polymarket-forecast.ts"]}\tnull\tnull\tnull\tnull\tnull\tnull\tnull\tnull\t{"rankIC":0.12,"z":2}\t{"lift":0.01,"rankIC":0.13}\t"keep"\t"measurable lift"\t".cramer-short/experiments/runs/run-1"',
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

  it('round-trips optional mutation and routing metadata through ledger rows and manifests', () => {
    const entry = makeEntry({
      ...makeMutationMetadata(),
      routingContext: makeRoutingContext(),
      promotion: makePendingPromotion(),
    });
    const manifest: ForecastLabRunManifest = {
      runId: entry.runId,
      startedAt: entry.startedAt,
      profileId: entry.profileId,
      targetSubsystem: entry.targetSubsystem,
      baselineCommit: '0123456789abcdef0123456789abcdef01234567',
      candidateBranch: entry.candidateBranch,
      allowedGlobs: entry.allowedGlobs,
      routingContext: entry.routingContext,
      effectiveMutationContract: entry.effectiveMutationContract,
      mutationMode: entry.mutationMode,
      parentRunId: entry.parentRunId,
      mutationId: entry.mutationId,
      mutationSummary: entry.mutationSummary,
      lineage: entry.lineage,
      mutationSpecSummary: entry.mutationSpecSummary,
      mutationReplayPayload: makeMutationReplayPayload(),
      candidateWorkspace: entry.candidateWorkspace,
      promotion: entry.promotion,
      artifactsPath: entry.artifactsPath,
    };
    const manifestPath = join(TEST_ROOT, 'runs', entry.runId, 'manifest.json');

    expect(parseLedgerRow(serializeLedgerRow(entry))).toEqual(entry);

    writeRunManifest(manifestPath, manifest);
    expect(readRunManifest(manifestPath)).toEqual(manifest);
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
    expect(() => parseLedgerRow('"run-1"\t"only-two-fields"')).toThrow(/expected 21 fields/);
  });

  it('rejects malformed rows with invalid JSON fields', () => {
    const fields = LEDGER_COLUMNS.map(() => '"x"');
    fields[0] = '{bad-json';

    expect(() => parseLedgerRow(fields.join('\t'))).toThrow(/Invalid JSON in runId/);
  });

  it('rejects malformed rows with invalid decisions', () => {
    const fields = serializeLedgerRow(makeEntry()).split('\t');
    fields[18] = '"maybe"';

    expect(() => parseLedgerRow(fields.join('\t'))).toThrow(/decision/);
  });

  it('rejects invalid optional mutation metadata', () => {
    expect(() =>
      validateLedgerEntry({
        ...makeEntry(),
        mutationMode: 'manual-edit',
      })).toThrow(/Unknown forecast-lab mutation mode/);

    expect(() =>
      validateLedgerEntry({
        ...makeEntry(),
        effectiveMutationContract: {
          mode: 'llm',
          mutableFiles: ['src/tools/finance/markov-distribution.ts'],
          allowedMutatorIds: ['replace-range'],
          allowMultipleCandidateAttempts: false,
        } as never,
      })).toThrow(/allowedMutatorIds is only supported for structured mutation mode/);

    expect(() =>
      validateLedgerEntry({
        ...makeEntry(),
        ...makeMutationMetadata(),
        mutationSpecSummary: {
          mutatorId: 'unknown-mutator',
          targetFiles: ['src/tools/finance/markov-distribution.ts'],
          summary: 'Bad mutator id',
        },
      })).toThrow(/Unknown forecast-lab mutator id/);

    expect(() =>
      validateLedgerEntry({
        ...makeEntry(),
        ...makeMutationMetadata(),
        mutationMode: undefined,
      })).toThrow(/structured mutation lineage metadata requires mutationMode="structured"/);

    expect(() =>
      validateLedgerEntry({
        ...makeEntry(),
        promotion: makePendingPromotion(),
      })).toThrow(/promotion metadata requires mutationMode="structured"/);

    expect(() =>
      validateLedgerEntry({
        ...makeEntry({
          ...makeMutationMetadata(),
          decision: 'drop',
        }),
        promotion: makePendingPromotion(),
      })).toThrow(/promotion metadata is only supported for kept runs/);

    expect(() =>
      validateLedgerEntry({
        ...makeEntry({
          ...makeMutationMetadata(),
          promotion: {
            ...makePendingPromotion(),
            source: {
              runId: 'run-2',
              manifestPath: getExperimentRunManifestPath('run-2'),
            },
          },
        }),
      })).toThrow(/promotion\.source\.runId must match runId/);

    expect(() =>
      validateRunManifest({
        ...makeEntry(makeMutationMetadata()),
        baselineCommit: '0123456789abcdef0123456789abcdef01234567',
        promotion: {
          ...makePendingPromotion(),
          status: 'approved',
        },
      })).toThrow(/promotion\.approvedAt must be a non-empty string/);

    expect(() =>
      validateLedgerEntry({
        ...makeEntry(),
        ...makeMutationMetadata(),
        parentRunId: 'run-2',
      })).toThrow(/parentRunId must match lineage.parentRunId/);

    expect(() =>
      validateLedgerEntry({
        ...makeEntry(),
        ...makeMutationMetadata(),
        mutationSummary: 'another summary',
      })).toThrow(/mutationSummary must match mutationSpecSummary.summary/);

    expect(() =>
      validateRunManifest({
        runId: 'run-1',
        startedAt: '2026-05-02T00:00:00.000Z',
        profileId: 'multi-asset-markov-short-horizon',
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/forecast-lab-run-1',
        allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
        mutationMode: 'structured',
        parentRunId: makeMutationMetadata().parentRunId,
        mutationId: makeMutationMetadata().mutationId,
        mutationSummary: makeMutationMetadata().mutationSummary,
        mutationSpecSummary: makeMutationMetadata().mutationSpecSummary,
        candidateWorkspace: makeMutationMetadata().candidateWorkspace,
        artifactsPath: '.cramer-short/experiments/runs/run-1',
      })).toThrow(/lineage is required when mutationMode="structured"/);

    expect(() =>
      validateRunManifest({
        runId: 'run-1',
        startedAt: '2026-05-02T00:00:00.000Z',
        profileId: 'multi-asset-markov-short-horizon',
        targetSubsystem: 'markov-distribution',
        baselineCommit: 'not-a-sha',
        candidateBranch: 'topic/forecast-lab-run-1',
        allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
        artifactsPath: '.cramer-short/experiments/runs/run-1',
      })).toThrow(/baselineCommit must be a full git commit sha/);

    expect(() =>
      validateRunManifest({
        runId: 'run-1',
        startedAt: '2026-05-02T00:00:00.000Z',
        profileId: 'multi-asset-markov-short-horizon',
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/forecast-lab-run-1',
        allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
        effectiveMutationContract: {
          mode: 'structured',
          mutableFiles: ['src/tools/finance/markov-distribution.ts'],
          allowedMutatorIds: ['replace-range'],
          allowMultipleCandidateAttempts: 'nope',
        } as never,
        artifactsPath: '.cramer-short/experiments/runs/run-1',
      })).toThrow(/allowMultipleCandidateAttempts/);

    expect(() =>
      validateRunManifest({
        runId: 'run-1',
        startedAt: '2026-05-02T00:00:00.000Z',
        profileId: 'multi-asset-markov-short-horizon',
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/forecast-lab-run-1',
        allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
        candidateWorkspace: {
          kind: 'detached-worktree',
          rootDir: '/repo',
          branch: 'topic/forecast-lab-run-1',
        },
        artifactsPath: '.cramer-short/experiments/runs/run-1',
      })).toThrow(/candidateWorkspace\.kind/);

      expect(() => validateRunManifest({
        ...makeEntry(makeMutationMetadata()),
        baselineCommit: '0123456789abcdef0123456789abcdef01234567',
        mutationReplayPayload: {
          ...makeMutationReplayPayload(),
          id: 'different-id',
        },
      })).toThrow(/mutationReplayPayload\.id must match mutationId/);
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

  it('reads legacy 11-column ledgers into the current entry shape', () => {
    const ledgerPath = join(TEST_ROOT, 'forecast-results.tsv');
    const legacyEntry = makeEntry();
    const legacyRow = [
      legacyEntry.runId,
      legacyEntry.startedAt,
      legacyEntry.profileId,
      legacyEntry.targetSubsystem,
      legacyEntry.candidateBranch,
      legacyEntry.allowedGlobs,
      legacyEntry.baselineSummary,
      legacyEntry.candidateSummary,
      legacyEntry.decision,
      legacyEntry.reason,
      legacyEntry.artifactsPath,
    ].map((value) => serializeLegacyLedgerField(value)).join('\t');

    mkdirSync(TEST_ROOT, { recursive: true });
    writeFileSync(ledgerPath, `${LEGACY_LEDGER_HEADER}\n${legacyRow}\n`, 'utf8');

    expect(readLedgerEntries(ledgerPath)).toEqual([{ ...legacyEntry, effectiveMutationContract: undefined }]);
  });

  it('reads pre-contract 15-column ledgers into the current entry shape', () => {
    const ledgerPath = join(TEST_ROOT, 'forecast-results.tsv');
    const legacyEntry = makeEntry(makeMutationMetadata());
    const legacyRow = [
      legacyEntry.runId,
      legacyEntry.startedAt,
      legacyEntry.profileId,
      legacyEntry.targetSubsystem,
      legacyEntry.candidateBranch,
      legacyEntry.allowedGlobs,
      legacyEntry.mutationMode,
      legacyEntry.lineage,
      legacyEntry.mutationSpecSummary,
      legacyEntry.candidateWorkspace,
      legacyEntry.baselineSummary,
      legacyEntry.candidateSummary,
      legacyEntry.decision,
      legacyEntry.reason,
      legacyEntry.artifactsPath,
    ].map((value) => serializeLegacyLedgerField(value)).join('\t');

    mkdirSync(TEST_ROOT, { recursive: true });
    writeFileSync(ledgerPath, `${PRE_CONTRACT_LEDGER_HEADER}\n${legacyRow}\n`, 'utf8');

    expect(readLedgerEntries(ledgerPath)).toEqual([{
      ...legacyEntry,
      effectiveMutationContract: undefined,
      parentRunId: undefined,
      mutationId: undefined,
      mutationSummary: undefined,
    }]);
  });

  it('reads pre-routing ledgers into the current entry shape', () => {
    const ledgerPath = join(TEST_ROOT, 'forecast-results.tsv');
    const legacyEntry = makeEntry();
    const legacyRow = [
      legacyEntry.runId,
      legacyEntry.startedAt,
      legacyEntry.profileId,
      legacyEntry.targetSubsystem,
      legacyEntry.candidateBranch,
      legacyEntry.allowedGlobs,
      legacyEntry.effectiveMutationContract,
      legacyEntry.mutationMode,
      legacyEntry.parentRunId,
      legacyEntry.mutationId,
      legacyEntry.mutationSummary,
      legacyEntry.lineage,
      legacyEntry.mutationSpecSummary,
      legacyEntry.candidateWorkspace,
      legacyEntry.baselineSummary,
      legacyEntry.candidateSummary,
      legacyEntry.decision,
      legacyEntry.reason,
      legacyEntry.artifactsPath,
    ].map((value) => serializeLegacyLedgerField(value)).join('\t');

    mkdirSync(TEST_ROOT, { recursive: true });
    writeFileSync(ledgerPath, `${PRE_ROUTING_LEDGER_HEADER}\n${legacyRow}\n`, 'utf8');

    expect(readLedgerEntries(ledgerPath)).toEqual([{ ...legacyEntry, routingContext: undefined }]);
  });

  it('reads pre-promotion ledgers into the current entry shape', () => {
    const ledgerPath = join(TEST_ROOT, 'forecast-results.tsv');
    const legacyEntry = makeEntry({
      ...makeMutationMetadata(),
      routingContext: makeRoutingContext(),
    });
    const legacyRow = [
      legacyEntry.runId,
      legacyEntry.startedAt,
      legacyEntry.profileId,
      legacyEntry.targetSubsystem,
      legacyEntry.candidateBranch,
      legacyEntry.allowedGlobs,
      legacyEntry.routingContext,
      legacyEntry.effectiveMutationContract,
      legacyEntry.mutationMode,
      legacyEntry.parentRunId,
      legacyEntry.mutationId,
      legacyEntry.mutationSummary,
      legacyEntry.lineage,
      legacyEntry.mutationSpecSummary,
      legacyEntry.candidateWorkspace,
      legacyEntry.baselineSummary,
      legacyEntry.candidateSummary,
      legacyEntry.decision,
      legacyEntry.reason,
      legacyEntry.artifactsPath,
    ].map((value) => serializeLegacyLedgerField(value)).join('\t');

    mkdirSync(TEST_ROOT, { recursive: true });
    writeFileSync(ledgerPath, `${PRE_PROMOTION_LEDGER_HEADER}\n${legacyRow}\n`, 'utf8');

    expect(readLedgerEntries(ledgerPath)).toEqual([{ ...legacyEntry, promotion: undefined }]);
  });

  it('migrates legacy ledgers to the current header before appending new rows', () => {
    const ledgerPath = join(TEST_ROOT, 'forecast-results.tsv');
    const legacyEntry = makeEntry({ effectiveMutationContract: undefined });
    const newEntry = makeEntry({
      runId: 'run-2',
      decision: 'drop',
      reason: 'no measurable lift',
      ...makeMutationMetadata(),
    });
    const legacyRow = [
      legacyEntry.runId,
      legacyEntry.startedAt,
      legacyEntry.profileId,
      legacyEntry.targetSubsystem,
      legacyEntry.candidateBranch,
      legacyEntry.allowedGlobs,
      legacyEntry.baselineSummary,
      legacyEntry.candidateSummary,
      legacyEntry.decision,
      legacyEntry.reason,
      legacyEntry.artifactsPath,
    ].map((value) => serializeLegacyLedgerField(value)).join('\t');

    mkdirSync(TEST_ROOT, { recursive: true });
    writeFileSync(ledgerPath, `${LEGACY_LEDGER_HEADER}\n${legacyRow}\n`, 'utf8');

    appendLedgerEntry(ledgerPath, newEntry);

    expect(readFileSync(ledgerPath, 'utf8')).toBe(
      `${LEDGER_HEADER}\n${serializeLedgerRow({
        ...legacyEntry,
        parentRunId: undefined,
        mutationId: undefined,
        mutationSummary: undefined,
      })}\n${serializeLedgerRow(newEntry)}\n`,
    );
    expect(readLedgerEntries(ledgerPath)).toEqual([{
      ...legacyEntry,
      parentRunId: undefined,
      mutationId: undefined,
      mutationSummary: undefined,
    }, newEntry]);
  });

  it('migrates pre-contract ledgers to the current header before appending new rows', () => {
    const ledgerPath = join(TEST_ROOT, 'forecast-results.tsv');
    const legacyEntry = makeEntry({
      effectiveMutationContract: undefined,
      ...makeMutationMetadata(),
    });
    const newEntry = makeEntry({ runId: 'run-2', decision: 'drop', reason: 'no measurable lift' });
    const legacyRow = [
      legacyEntry.runId,
      legacyEntry.startedAt,
      legacyEntry.profileId,
      legacyEntry.targetSubsystem,
      legacyEntry.candidateBranch,
      legacyEntry.allowedGlobs,
      legacyEntry.mutationMode,
      legacyEntry.lineage,
      legacyEntry.mutationSpecSummary,
      legacyEntry.candidateWorkspace,
      legacyEntry.baselineSummary,
      legacyEntry.candidateSummary,
      legacyEntry.decision,
      legacyEntry.reason,
      legacyEntry.artifactsPath,
    ].map((value) => serializeLegacyLedgerField(value)).join('\t');

    mkdirSync(TEST_ROOT, { recursive: true });
    writeFileSync(ledgerPath, `${PRE_CONTRACT_LEDGER_HEADER}\n${legacyRow}\n`, 'utf8');

    appendLedgerEntry(ledgerPath, newEntry);

    expect(readFileSync(ledgerPath, 'utf8')).toBe(
      `${LEDGER_HEADER}\n${serializeLedgerRow({
        ...legacyEntry,
        parentRunId: undefined,
        mutationId: undefined,
        mutationSummary: undefined,
      })}\n${serializeLedgerRow(newEntry)}\n`,
    );
    expect(readLedgerEntries(ledgerPath)).toEqual([{
      ...legacyEntry,
      parentRunId: undefined,
      mutationId: undefined,
      mutationSummary: undefined,
    }, newEntry]);
  });

  it('migrates pre-routing ledgers to the current header before appending new rows', () => {
    const ledgerPath = join(TEST_ROOT, 'forecast-results.tsv');
    const legacyEntry = makeEntry();
    const newEntry = makeEntry({
      runId: 'run-2',
      decision: 'drop',
      reason: 'no measurable lift',
      routingContext: makeRoutingContext(),
    });
    const legacyRow = [
      legacyEntry.runId,
      legacyEntry.startedAt,
      legacyEntry.profileId,
      legacyEntry.targetSubsystem,
      legacyEntry.candidateBranch,
      legacyEntry.allowedGlobs,
      legacyEntry.effectiveMutationContract,
      legacyEntry.mutationMode,
      legacyEntry.parentRunId,
      legacyEntry.mutationId,
      legacyEntry.mutationSummary,
      legacyEntry.lineage,
      legacyEntry.mutationSpecSummary,
      legacyEntry.candidateWorkspace,
      legacyEntry.baselineSummary,
      legacyEntry.candidateSummary,
      legacyEntry.decision,
      legacyEntry.reason,
      legacyEntry.artifactsPath,
    ].map((value) => serializeLegacyLedgerField(value)).join('\t');

    mkdirSync(TEST_ROOT, { recursive: true });
    writeFileSync(ledgerPath, `${PRE_ROUTING_LEDGER_HEADER}\n${legacyRow}\n`, 'utf8');

    appendLedgerEntry(ledgerPath, newEntry);

    expect(readFileSync(ledgerPath, 'utf8')).toBe(
      `${LEDGER_HEADER}\n${serializeLedgerRow({
        ...legacyEntry,
        routingContext: undefined,
      })}\n${serializeLedgerRow(newEntry)}\n`,
    );
    expect(readLedgerEntries(ledgerPath)).toEqual([
      { ...legacyEntry, routingContext: undefined },
      newEntry,
    ]);
  });

  it('migrates pre-promotion ledgers to the current header before appending new rows', () => {
    const ledgerPath = join(TEST_ROOT, 'forecast-results.tsv');
    const legacyEntry = makeEntry({
      ...makeMutationMetadata(),
      routingContext: makeRoutingContext(),
    });
    const newEntry = makeEntry({
      runId: 'run-2',
      decision: 'keep',
      reason: 'promotion pending',
      ...makeMutationMetadata(),
      promotion: makePendingPromotion('run-2'),
    });
    const legacyRow = [
      legacyEntry.runId,
      legacyEntry.startedAt,
      legacyEntry.profileId,
      legacyEntry.targetSubsystem,
      legacyEntry.candidateBranch,
      legacyEntry.allowedGlobs,
      legacyEntry.routingContext,
      legacyEntry.effectiveMutationContract,
      legacyEntry.mutationMode,
      legacyEntry.parentRunId,
      legacyEntry.mutationId,
      legacyEntry.mutationSummary,
      legacyEntry.lineage,
      legacyEntry.mutationSpecSummary,
      legacyEntry.candidateWorkspace,
      legacyEntry.baselineSummary,
      legacyEntry.candidateSummary,
      legacyEntry.decision,
      legacyEntry.reason,
      legacyEntry.artifactsPath,
    ].map((value) => serializeLegacyLedgerField(value)).join('\t');

    mkdirSync(TEST_ROOT, { recursive: true });
    writeFileSync(ledgerPath, `${PRE_PROMOTION_LEDGER_HEADER}\n${legacyRow}\n`, 'utf8');

    appendLedgerEntry(ledgerPath, newEntry);

    expect(readFileSync(ledgerPath, 'utf8')).toBe(
      `${LEDGER_HEADER}\n${serializeLedgerRow({
        ...legacyEntry,
        promotion: undefined,
      })}\n${serializeLedgerRow(newEntry)}\n`,
    );
    expect(readLedgerEntries(ledgerPath)).toEqual([
      { ...legacyEntry, promotion: undefined },
      newEntry,
    ]);
  });
});

describe('forecast-lab manifest helpers', () => {
  it('writes deterministic JSON manifests', () => {
    const manifestPath = join(TEST_ROOT, 'runs', 'run-1', 'manifest.json');
    const manifest: ForecastLabRunManifest = {
      runId: 'run-1',
      startedAt: '2026-05-02T00:00:00.000Z',
      profileId: 'multi-asset-markov-short-horizon',
      targetSubsystem: 'markov-distribution',
      baselineCommit: '0123456789abcdef0123456789abcdef01234567',
      candidateBranch: 'topic/forecast-lab-run-1',
      allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
      candidateWorkspace: {
        kind: 'candidate-worktree',
        rootDir: '/repo/.cramer-short/experiments/worktrees/run-1',
        branch: 'topic/forecast-lab-run-1',
      },
      effectiveMutationContract: {
        mode: 'structured',
        mutableFiles: ['src/tools/finance/markov-distribution.ts'],
        allowedMutatorIds: ['replace-range'],
        allowMultipleCandidateAttempts: false,
      },
      artifactsPath: '.cramer-short/experiments/runs/run-1',
    };

    writeRunManifest(manifestPath, manifest);

    expect(readFileSync(manifestPath, 'utf8')).toBe(serializeManifest(manifest));
    expect(readRunManifest(manifestPath)).toEqual(manifest);
    expect(serializeManifest(manifest)).toBe(
      '{\n  "allowedGlobs": [\n    "src/tools/finance/markov-distribution.ts"\n  ],\n  "artifactsPath": ".cramer-short/experiments/runs/run-1",\n  "baselineCommit": "0123456789abcdef0123456789abcdef01234567",\n  "candidateBranch": "topic/forecast-lab-run-1",\n  "candidateWorkspace": {\n    "branch": "topic/forecast-lab-run-1",\n    "kind": "candidate-worktree",\n    "rootDir": "/repo/.cramer-short/experiments/worktrees/run-1"\n  },\n  "effectiveMutationContract": {\n    "allowedMutatorIds": [\n      "replace-range"\n    ],\n    "allowMultipleCandidateAttempts": false,\n    "mode": "structured",\n    "mutableFiles": [\n      "src/tools/finance/markov-distribution.ts"\n    ]\n  },\n  "profileId": "multi-asset-markov-short-horizon",\n  "runId": "run-1",\n  "startedAt": "2026-05-02T00:00:00.000Z",\n  "targetSubsystem": "markov-distribution"\n}\n',
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
