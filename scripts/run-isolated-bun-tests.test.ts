import { describe, expect, it } from 'bun:test';
import { getIntegrationSpecs } from './run-isolated-bun-tests.js';

const integrationFiles = async () => (await getIntegrationSpecs()).map((spec) => spec.file);

describe('integration tier test discovery', () => {
  it('runs integrationIt cases that live in ordinary *.test.ts files, not only *.integration.test.ts', async () => {
    const files = await integrationFiles();
    expect(files).toContain('src/tools/finance/markov-distribution.integration.test.ts');
    for (const file of [
      'src/tools/finance/backtest/walk-forward-btc-ultra-short-horizon.test.ts',
      'src/tools/finance/backtest/phase9-warmup90-promotion.test.ts',
      'src/tools/finance/markov-distribution.tool-output.test.ts',
    ]) {
      expect(files).toContain(file);
    }
  });

  it('skips a file that only imports integrationIt without calling it', async () => {
    expect(await integrationFiles()).not.toContain('src/tools/finance/markov-distribution.test.ts');
  });

  it('never pulls in e2e files', async () => {
    expect((await integrationFiles()).some((file) => file.endsWith('.e2e.test.ts'))).toBe(false);
  });
});
