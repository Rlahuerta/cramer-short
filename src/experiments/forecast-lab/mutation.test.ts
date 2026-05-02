import { describe, expect, it } from 'bun:test';
import type { ForecastLabMutatorId } from './mutation.js';
import {
  FORECAST_LAB_MUTATOR_IDS,
  assertForecastLabMutatorId,
  defineForecastLabProfileMutationConfig,
  isForecastLabMutatorId,
  listForecastLabMutatorIds,
} from './mutation.js';

describe('forecast-lab mutation contract', () => {
  it('lists the bounded Phase 1 mutator ids', () => {
    expect(listForecastLabMutatorIds()).toBe(FORECAST_LAB_MUTATOR_IDS);
    expect([...FORECAST_LAB_MUTATOR_IDS]).toEqual([
      'replace-range',
      'search-replace',
      'insert-block',
    ]);

    let mutatorId: string = 'search-replace';
    assertForecastLabMutatorId(mutatorId);
    const narrowed: ForecastLabMutatorId = mutatorId;

    expect(narrowed).toBe('search-replace');
    expect(isForecastLabMutatorId('insert-block')).toBe(true);
    expect(isForecastLabMutatorId('unknown-mutator')).toBe(false);
  });

  it('rejects unknown mutator ids loudly', () => {
    expect(() => assertForecastLabMutatorId('unknown-mutator')).toThrow(/Unknown forecast-lab mutator id: unknown-mutator/);
    expect(() =>
      defineForecastLabProfileMutationConfig({
        mode: 'structured',
        mutableFiles: ['src/tools/finance/markov-distribution.ts'],
        allowedMutatorIds: ['replace-range', 'unknown-mutator'],
      })).toThrow(/Unknown forecast-lab mutator id: unknown-mutator/);
  });

  it('returns immutable mutation configs for profile reuse', () => {
    const config = defineForecastLabProfileMutationConfig({
      mode: 'structured',
      mutableFiles: ['src/tools/finance/markov-distribution.ts'],
      allowedMutatorIds: ['replace-range', 'search-replace'],
    });

    expect(config).toEqual({
      mode: 'structured',
      mutableFiles: ['src/tools/finance/markov-distribution.ts'],
      allowedMutatorIds: ['replace-range', 'search-replace'],
      allowMultipleCandidateAttempts: false,
    });
    expect(Object.isFrozen(config)).toBe(true);
    expect(Object.isFrozen(config.mutableFiles)).toBe(true);
    expect(Object.isFrozen(config.allowedMutatorIds)).toBe(true);

    expect(() => {
      (config.mutableFiles as unknown as string[]).push('src/tools/finance/conformal.ts');
    }).toThrow();
    expect(() => {
      (config.allowedMutatorIds as unknown as string[]).push('insert-block');
    }).toThrow();
  });

  it('omits structured mutator ids for non-structured modes', () => {
    const dryRunConfig = defineForecastLabProfileMutationConfig({
      mode: 'dry-run',
      mutableFiles: ['src/tools/finance/markov-distribution.ts'],
    });
    const llmConfig = defineForecastLabProfileMutationConfig({
      mode: 'llm',
      mutableFiles: ['src/tools/finance/markov-distribution.ts'],
      allowMultipleCandidateAttempts: true,
    });

    expect(dryRunConfig).toEqual({
      mode: 'dry-run',
      mutableFiles: ['src/tools/finance/markov-distribution.ts'],
      allowMultipleCandidateAttempts: false,
    });
    expect(llmConfig).toEqual({
      mode: 'llm',
      mutableFiles: ['src/tools/finance/markov-distribution.ts'],
      allowMultipleCandidateAttempts: true,
    });
    expect('allowedMutatorIds' in dryRunConfig).toBe(false);
    expect('allowedMutatorIds' in llmConfig).toBe(false);
    expect(Object.isFrozen(dryRunConfig)).toBe(true);
    expect(Object.isFrozen(dryRunConfig.mutableFiles)).toBe(true);
    expect(Object.isFrozen(llmConfig)).toBe(true);
    expect(Object.isFrozen(llmConfig.mutableFiles)).toBe(true);
  });

  it('rejects structured mutator ids for non-structured modes', () => {
    expect(() =>
      defineForecastLabProfileMutationConfig({
        mode: 'dry-run',
        mutableFiles: ['src/tools/finance/markov-distribution.ts'],
        allowedMutatorIds: ['replace-range'],
      } as {
        readonly mode: 'dry-run';
        readonly mutableFiles: readonly string[];
        readonly allowedMutatorIds: readonly string[];
      })).toThrow(/allowedMutatorIds is only supported for structured mutation mode: dry-run/);
  });
});
