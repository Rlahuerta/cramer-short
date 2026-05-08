import { describe, expect, it, mock } from 'bun:test';
import { createForecastLabRunTool, parseForecastLabRunToolPayload } from './forecast-lab-run.js';

function makeRunResult(overrides: Record<string, unknown> = {}) {
  return {
    runId: 'btc-markov-ultra-short-horizon.keep-1',
    manifest: {
      runId: 'btc-markov-ultra-short-horizon.keep-1',
      profileId: 'btc-markov-ultra-short-horizon',
      artifactsPath: '.cramer-short/experiments/runs/btc-markov-ultra-short-horizon.keep-1',
    },
    baseline: {},
    candidate: {},
    decision: {
      decision: 'keep' as const,
      reason: 'candidate passed the fixed gates',
    },
    ledgerEntry: {
      promotion: {
        status: 'approval-required' as const,
        source: {
          runId: 'btc-markov-ultra-short-horizon.keep-1',
          manifestPath: '.cramer-short/experiments/runs/btc-markov-ultra-short-horizon.keep-1/manifest.json',
        },
        requestedAt: '2026-05-03T00:00:00.000Z',
      },
    },
    ...overrides,
  };
}

function makePromotionResult(overrides: Record<string, unknown> = {}) {
  return {
    runId: 'forecast-lab-promo-1',
    sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
    manifest: {
      runId: 'forecast-lab-promo-1',
      profileId: 'btc-markov-ultra-short-horizon',
      artifactsPath: '.cramer-short/experiments/runs/forecast-lab-promo-1',
    },
    sourceManifest: {
      runId: 'btc-markov-ultra-short-horizon.keep-1',
      profileId: 'btc-markov-ultra-short-horizon',
      artifactsPath: '.cramer-short/experiments/runs/btc-markov-ultra-short-horizon.keep-1',
    },
    baseline: {},
    candidate: {},
    decision: {
      decision: 'keep' as const,
      reason: 'promotion verification passed',
    },
    activation: {
      artifactsPath: '.cramer-short/experiments/runs/forecast-lab-promo-1',
    },
    activeStatePath: '.cramer-short/experiments/active-promotions/btc-markov-ultra-short-horizon.json',
    ...overrides,
  };
}

function makeResetResult(overrides: Record<string, unknown> = {}) {
  return {
    runId: 'forecast-lab-reset-1',
    profileId: 'btc-markov-ultra-short-horizon',
    mode: 'defaults' as const,
    artifactsPath: '.cramer-short/experiments/runs/forecast-lab-reset-1',
    resetArtifactPath: '.cramer-short/experiments/runs/forecast-lab-reset-1/reset.json',
    ...overrides,
  };
}

describe('forecast_lab_run tool', () => {
  it('returns the bounded plan without executing when execute=false', async () => {
    const runForecastLabFn = mock(async () => makeRunResult());
    const tool = createForecastLabRunTool({ runForecastLabFn: runForecastLabFn as any });

    const result = await tool.invoke({
      action: 'guided-improve',
      profileId: 'btc-markov-ultra-short-horizon',
      execute: false,
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(runForecastLabFn).not.toHaveBeenCalled();
    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'guided-improve',
      status: 'ok',
      execute: false,
      profileId: 'btc-markov-ultra-short-horizon',
      promotionReady: false,
    });
    expect(payload?.answer).toContain('bounded plan');
    expect(payload?.answer).toContain('.cramer-short/experiments');
  });

  it('runs structured guided improvement and exposes approval-required promotion state', async () => {
    const runForecastLabFn = mock(async () => makeRunResult());
    const tool = createForecastLabRunTool({ runForecastLabFn: runForecastLabFn as any });

    const result = await tool.invoke({
      action: 'guided-improve',
      query: 'Improve the BTC 1d/2d/3d Markov forecast workflow.',
      profileId: 'btc-markov-ultra-short-horizon',
      routingSource: 'auto-routed',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(runForecastLabFn).toHaveBeenCalledWith({
      profileId: 'btc-markov-ultra-short-horizon',
      mutationMode: 'structured',
      routingContext: {
        originatingQuery: 'Improve the BTC 1d/2d/3d Markov forecast workflow.',
        selectedProfileId: 'btc-markov-ultra-short-horizon',
        routerReason: 'Profile supplied explicitly.',
        invocationSource: 'auto-routed',
      },
      progress: undefined,
    });
    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'guided-improve',
      status: 'ok',
      execute: true,
      profileId: 'btc-markov-ultra-short-horizon',
      runId: 'btc-markov-ultra-short-horizon.keep-1',
      decision: 'keep',
      promotionReady: true,
      promotionStatus: 'approval-required',
      sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
    });
    expect(payload?.answer).toContain('Approval required before promotion');
  });

  it('returns a bounded catalog-extension plan without reading experiment artifacts', async () => {
    const runForecastLabFn = mock(async () => makeRunResult());
    const readLedgerEntriesFn = mock(() => []);
    const readRunManifestFn = mock(() => {
      throw new Error('should not read manifest');
    });
    const readTextFileFn = mock(() => {
      throw new Error('should not read artifact');
    });
    const tool = createForecastLabRunTool({
      runForecastLabFn: runForecastLabFn as any,
      readLedgerEntriesFn,
      readRunManifestFn: readRunManifestFn as any,
      readTextFileFn,
    });

    const result = await tool.invoke({
      action: 'catalog-extension-plan',
      profileId: 'btc-markov-ultra-short-horizon',
      query: 'design a new shipped mutator outside the existing catalog and re-run the lineage',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(runForecastLabFn).not.toHaveBeenCalled();
    expect(readLedgerEntriesFn).not.toHaveBeenCalled();
    expect(readRunManifestFn).not.toHaveBeenCalled();
    expect(readTextFileFn).not.toHaveBeenCalled();
    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'catalog-extension-plan',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      mutationMode: 'structured',
      allowedMutatorIds: ['search-replace'],
      catalogFiles: [
        'src/experiments/forecast-lab/mutators/markov-parameters.ts',
        'src/experiments/forecast-lab/profiles.ts',
      ],
      validationFiles: [
        'src/experiments/forecast-lab/mutators/markov-parameters.test.ts',
        'src/experiments/forecast-lab/profiles.test.ts',
        'src/tools/finance/markov-distribution.test.ts',
        'src/tools/finance/backtest/walk-forward-r5.test.ts',
      ],
      operatorMutatorIds: ['replace-range', 'search-replace', 'insert-block'],
    });
    expect(payload?.answer).toContain('bounded code-change');
    expect(payload?.answer).toContain('did not inspect experiment artifacts');
    expect(payload?.answer).toContain('forecast_lab_run(action="guided-improve", profileId="btc-markov-ultra-short-horizon")');
    expect(payload?.answer).toContain('src/experiments/forecast-lab/mutators/markov-parameters.ts');
  });

  it('echoes requested mutator ids and parameter deltas for detailed catalog-extension briefs', async () => {
    const tool = createForecastLabRunTool();

    const result = await tool.invoke({
      action: 'catalog-extension-plan',
      query: [
        'Target anchor trust weighting.',
        'Add a new shipped structured mutator for btc-markov-ultra-short-horizon that makes the Markov/anchor blend more adaptive under high posterior entropy using the existing soft-regime weighting controls in src/tools/finance/markov-distribution.ts.',
        'Suggested starting values:',
        '- softRegimeConfidenceFloor: 0.65 -> 0.55',
        '- softRegimeConfidenceEntropyWeight: 0.35 -> 0.50',
        'Name it something like:',
        'markov-entropy-adaptive-anchor-weighting',
        'Keep it bounded, add the shipped mutator to the catalog, and validate it with the existing BTC ultra-short-horizon walk-forward gate.',
      ].join('\n'),
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'catalog-extension-plan',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      requestedMutatorId: 'markov-entropy-adaptive-anchor-weighting',
      requestedParameterChanges: [
        'softRegimeConfidenceFloor: 0.65 -> 0.55',
        'softRegimeConfidenceEntropyWeight: 0.35 -> 0.50',
      ],
    });
    expect(payload?.answer).toContain('Requested mutator id: markov-entropy-adaptive-anchor-weighting.');
    expect(payload?.answer).toContain('- softRegimeConfidenceFloor: 0.65 -> 0.55');
  });

  it('lists the shipped mutator ids for an explicit profile without reading experiment artifacts', async () => {
    const readLedgerEntriesFn = mock(() => []);
    const readRunManifestFn = mock(() => {
      throw new Error('should not read manifest');
    });
    const readTextFileFn = mock(() => {
      throw new Error('should not read artifact');
    });
    const tool = createForecastLabRunTool({
      readLedgerEntriesFn,
      readRunManifestFn: readRunManifestFn as any,
      readTextFileFn,
    });

    const result = await tool.invoke({
      action: 'list-mutators',
      profileId: 'btc-markov-ultra-short-horizon',
      query: 'List the shipped mutator ids for btc-markov-ultra-short-horizon.',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(readLedgerEntriesFn).not.toHaveBeenCalled();
    expect(readRunManifestFn).not.toHaveBeenCalled();
    expect(readTextFileFn).not.toHaveBeenCalled();
    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'list-mutators',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      profiles: [
        {
          profileId: 'btc-markov-ultra-short-horizon',
          targetSubsystem: 'markov-distribution',
          mutationMode: 'structured',
          allowedOperatorIds: ['search-replace'],
        },
      ],
      frameworkOperatorIds: ['replace-range', 'search-replace', 'insert-block'],
    });
    expect(payload?.answer).toContain('Forecast-lab shipped mutator ids for btc-markov-ultra-short-horizon.');
    expect(payload?.answer).toContain('| Field | Value |');
    expect(payload?.answer).toContain('| # | Mutator id |');
    expect(payload?.answer).toContain('markov-shorter-reactive-window');
  });

  it('lists every structured profile when a mutator-list query is ambiguous', async () => {
    const tool = createForecastLabRunTool();

    const result = await tool.invoke({
      action: 'list-mutators',
      query: 'List the mutate availible',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'list-mutators',
      status: 'ok',
      dryRunProfiles: ['btc-arbiter-replay', 'polymarket-selection-sanity'],
    });
    if (!payload || payload.action !== 'list-mutators' || payload.status !== 'ok') {
      throw new Error('Expected an ok list-mutators payload');
    }
    expect(payload.profiles).toEqual(expect.arrayContaining([
      expect.objectContaining({
        profileId: 'multi-asset-markov-short-horizon',
        mutationMode: 'structured',
      }),
      expect.objectContaining({
        profileId: 'btc-markov-ultra-short-horizon',
        mutationMode: 'structured',
      }),
      expect.objectContaining({
        profileId: 'gold-markov-short-horizon',
        mutationMode: 'structured',
      }),
    ]));
    expect(payload?.answer).toContain('Forecast-lab shipped mutator catalog summary.');
    expect(payload?.answer).toContain('| Profile id | Target subsystem | Mutation mode | Shipped ids | Allowed operators |');
    expect(payload?.answer).toContain('Shipped candidate catalog ids for btc-markov-ultra-short-horizon:');
    expect(payload?.answer).toContain('Shipped candidate catalog ids for gold-markov-short-horizon:');
    expect(payload?.answer).toContain('btc-markov-ultra-short-horizon');
  });

  it('compares the latest kept structured run against the shipped baseline and returns the promotion command', async () => {
    const tool = createForecastLabRunTool({
      findLatestKeptLedgerEntryFn: () => ({
        runId: 'btc-markov-ultra-short-horizon.keep-1',
        profileId: 'btc-markov-ultra-short-horizon',
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/forecast-lab-btc-markov-ultra-short-horizon.keep-1',
        allowedGlobs: [
          'src/tools/finance/markov-distribution.ts',
          'src/tools/finance/conformal.ts',
          'src/tools/finance/regime-calibrator.ts',
        ],
        mutationMode: 'structured',
        mutationId: 'markov-lower-confidence-trend-penalty',
        mutationSummary: 'Lower the confidence gate while keeping the trend-only break penalty path active.',
        baselineSummary: { exitCode: 0 },
        candidateSummary: { exitCode: 0 },
        decision: 'keep',
        reason: 'candidate passed the fixed gates',
        artifactsPath: '.cramer-short/experiments/runs/btc-markov-ultra-short-horizon.keep-1',
      } as any),
      readRunManifestFn: () => ({
        runId: 'btc-markov-ultra-short-horizon.keep-1',
        startedAt: '2026-05-03T00:00:00.000Z',
        profileId: 'btc-markov-ultra-short-horizon',
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/forecast-lab-btc-markov-ultra-short-horizon.keep-1',
        allowedGlobs: [],
        mutationMode: 'structured',
        mutationId: 'markov-lower-confidence-trend-penalty',
        mutationSummary: 'Lower the confidence gate while keeping the trend-only break penalty path active.',
        lineage: { generation: 2, rootRunId: 'root-run' },
        mutationSpecSummary: { mutatorId: 'search-replace', summary: 'Lower the confidence gate while keeping the trend-only break penalty path active.', targetFiles: ['src/tools/finance/markov-distribution.ts'] },
        mutationReplayPayload: { kind: 'markov-parameter-candidate', id: 'markov-lower-confidence-trend-penalty', profileId: 'btc-markov-ultra-short-horizon', mutatorId: 'search-replace', edits: [], patchSummary: [], specSummary: { mutatorId: 'search-replace', summary: 'Lower the confidence gate while keeping the trend-only break penalty path active.', targetFiles: ['src/tools/finance/markov-distribution.ts'] } },
        candidateWorkspace: { kind: 'candidate-worktree', rootDir: '/tmp/worktree', branch: 'topic/forecast-lab-btc-markov-ultra-short-horizon.keep-1' },
        artifactsPath: '.cramer-short/experiments/runs/btc-markov-ultra-short-horizon.keep-1',
      } as any),
      readTextFileFn: (path: string) => {
        if (path.endsWith('/baseline.json')) {
          return JSON.stringify({
            commands: [{
              id: 'walk-forward-btc-ultra-short-horizon',
              stdout: [
                'BTC-USD horizon 1d',
                'baseline warmup=120 stride=3    │   204 │      0 │   53.4% │ 0.256 │  97.1% │      +0.0pp',
                'BTC-USD horizon 2d',
                'baseline warmup=120 stride=3    │   203 │      0 │   49.8% │ 0.256 │  98.5% │      +0.0pp',
                'BTC-USD horizon 3d',
                'baseline warmup=120 stride=3    │   203 │      0 │   48.8% │ 0.259 │  99.5% │      +0.0pp',
              ].join('\n'),
            }],
          });
        }
        if (path.endsWith('/candidate.json')) {
          return JSON.stringify({
            commands: [{
              id: 'walk-forward-btc-ultra-short-horizon',
              stdout: [
                'BTC-USD horizon 1d',
                'baseline warmup=120 stride=3    │   204 │      0 │   54.4% │ 0.256 │  97.1% │      +0.0pp',
                'BTC-USD horizon 2d',
                'baseline warmup=120 stride=3    │   203 │      0 │   50.7% │ 0.255 │  98.5% │      +0.0pp',
                'BTC-USD horizon 3d',
                'baseline warmup=120 stride=3    │   203 │      0 │   48.8% │ 0.258 │  99.5% │      +0.0pp',
              ].join('\n'),
            }],
          });
        }
        throw new Error(`unexpected path ${path}`);
      },
      existsSyncFn: () => false,
    });

    const result = await tool.invoke({
      action: 'compare-best-vs-shipped',
      profileId: 'btc-markov-ultra-short-horizon',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'compare-best-vs-shipped',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
      liveStatus: 'ready-to-promote',
      promotionCommand: 'Approve forecast-lab promotion for btc-markov-ultra-short-horizon run btc-markov-ultra-short-horizon.keep-1.',
    });
    expect(payload?.answer).toContain('53.4% -> 54.4%');
    expect(payload?.answer).toContain('Reply "Approve forecast-lab promotion for btc-markov-ultra-short-horizon run btc-markov-ultra-short-horizon.keep-1."');
  });

  it('returns an operator-friendly compare answer when no kept structured run exists yet', async () => {
    const tool = createForecastLabRunTool({
      findLatestKeptLedgerEntryFn: () => undefined,
    });

    const result = await tool.invoke({
      action: 'compare-best-vs-shipped',
      profileId: 'btc-markov-ultra-short-horizon',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'compare-best-vs-shipped',
      status: 'error',
      error: 'No kept structured run is recorded yet for profile "btc-markov-ultra-short-horizon".',
    });
    expect(payload?.answer).toContain('Current best: no kept structured run exists yet');
    expect(payload?.answer).toContain('shipped baseline');
    expect(payload?.answer).toContain('not live yet');
  });

  it('resolves comparison profile from routed result queries when profileId is omitted', async () => {
    const tool = createForecastLabRunTool({
      findLatestKeptLedgerEntryFn: () => ({
        runId: 'btc-markov-ultra-short-horizon.keep-1',
        profileId: 'btc-markov-ultra-short-horizon',
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/forecast-lab-btc-markov-ultra-short-horizon.keep-1',
        allowedGlobs: [],
        mutationMode: 'structured',
        baselineSummary: { exitCode: 0 },
        candidateSummary: { exitCode: 0 },
        decision: 'keep',
        reason: 'candidate passed the fixed gates',
        artifactsPath: '.cramer-short/experiments/runs/btc-markov-ultra-short-horizon.keep-1',
      } as any),
      readRunManifestFn: () => ({
        runId: 'btc-markov-ultra-short-horizon.keep-1',
        startedAt: '2026-05-03T00:00:00.000Z',
        profileId: 'btc-markov-ultra-short-horizon',
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/forecast-lab-btc-markov-ultra-short-horizon.keep-1',
        allowedGlobs: [],
        mutationMode: 'structured',
        lineage: { generation: 2, rootRunId: 'root-run' },
        mutationSpecSummary: { mutatorId: 'search-replace', summary: 'summary', targetFiles: ['src/tools/finance/markov-distribution.ts'] },
        mutationReplayPayload: { kind: 'markov-parameter-candidate', id: 'id', profileId: 'btc-markov-ultra-short-horizon', mutatorId: 'search-replace', edits: [], patchSummary: [], specSummary: { mutatorId: 'search-replace', summary: 'summary', targetFiles: ['src/tools/finance/markov-distribution.ts'] } },
        candidateWorkspace: { kind: 'candidate-worktree', rootDir: '/tmp/worktree', branch: 'topic/branch' },
        artifactsPath: '.cramer-short/experiments/runs/btc-markov-ultra-short-horizon.keep-1',
      } as any),
      readTextFileFn: () => JSON.stringify({ commands: [] }),
      existsSyncFn: () => false,
    });

    const result = await tool.invoke({
      action: 'compare-best-vs-shipped',
      query: 'provide the results of the Optimize the BTC 1d/2d/3d Markov forecast workflow',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'compare-best-vs-shipped',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
    });
  });

  it('compares a named kept mutator against the active promoted run', async () => {
    const readLedgerEntriesFn = mock(() => [
      {
        runId: 'btc-active.keep-1',
        profileId: 'btc-markov-ultra-short-horizon',
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/active',
        allowedGlobs: [],
        mutationMode: 'structured',
        mutationId: 'markov-lower-confidence-trend-penalty',
        mutationSummary: 'Lower the confidence gate while keeping the trend-only break penalty path active.',
        baselineSummary: { exitCode: 0 },
        candidateSummary: { exitCode: 0 },
        decision: 'keep',
        reason: 'active run passed',
        artifactsPath: '.cramer-short/experiments/runs/btc-active.keep-1',
      },
      {
        runId: 'btc-requested.keep-1',
        profileId: 'btc-markov-ultra-short-horizon',
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/requested',
        allowedGlobs: [],
        mutationMode: 'structured',
        mutationId: 'markov-entropy-adaptive-anchor-weighting',
        mutationSummary: 'Make the anchor blend more adaptive under entropy.',
        baselineSummary: { exitCode: 0 },
        candidateSummary: { exitCode: 0 },
        decision: 'keep',
        reason: 'requested run passed',
        artifactsPath: '.cramer-short/experiments/runs/btc-requested.keep-1',
      },
    ] as any);
    const tool = createForecastLabRunTool({
      readLedgerEntriesFn,
      getLedgerPathFn: () => '.cramer-short/experiments/forecast-results.tsv',
      readTextFileFn: (path: string) => {
        if (path.endsWith('active-promotions/btc-markov-ultra-short-horizon.json')) {
          return JSON.stringify({
            profileId: 'btc-markov-ultra-short-horizon',
            sourceRunId: 'btc-active.keep-1',
          });
        }
        if (path.endsWith('btc-active.keep-1/candidate.json')) {
          return JSON.stringify({
            commands: [{
              id: 'walk-forward-btc-ultra-short-horizon',
              stdout: [
                'BTC-USD horizon 1d',
                'baseline warmup=120 stride=3    │   204 │      0 │   54.4% │ 0.256 │  97.1% │      +0.0pp',
                'BTC-USD horizon 2d',
                'baseline warmup=120 stride=3    │   203 │      0 │   50.7% │ 0.255 │  98.5% │      +0.0pp',
                'BTC-USD horizon 3d',
                'baseline warmup=120 stride=3    │   203 │      0 │   48.8% │ 0.258 │  99.5% │      +0.0pp',
              ].join('\n'),
            }],
          });
        }
        if (path.endsWith('btc-requested.keep-1/candidate.json')) {
          return JSON.stringify({
            commands: [{
              id: 'walk-forward-btc-ultra-short-horizon',
              stdout: [
                'BTC-USD horizon 1d',
                'baseline warmup=120 stride=3    │   204 │      0 │   55.1% │ 0.254 │  96.9% │      +0.0pp',
                'BTC-USD horizon 2d',
                'baseline warmup=120 stride=3    │   203 │      0 │   51.0% │ 0.254 │  98.3% │      +0.0pp',
                'BTC-USD horizon 3d',
                'baseline warmup=120 stride=3    │   203 │      0 │   49.0% │ 0.257 │  99.3% │      +0.0pp',
              ].join('\n'),
            }],
          });
        }
        throw new Error(`unexpected path ${path}`);
      },
      existsSyncFn: (path: string) => path.endsWith('active-promotions/btc-markov-ultra-short-horizon.json'),
    });

    const result = await tool.invoke({
      action: 'compare-best-vs-shipped',
      profileId: 'btc-markov-ultra-short-horizon',
      mutationId: 'markov-entropy-adaptive-anchor-weighting',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'compare-best-vs-shipped',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      comparisonTarget: 'active-live',
      sourceRunId: 'btc-requested.keep-1',
      activeSourceRunId: 'btc-active.keep-1',
      mutationId: 'markov-entropy-adaptive-anchor-weighting',
      activeMutationId: 'markov-lower-confidence-trend-penalty',
    });
    expect(payload?.answer).toContain('Active Dir Acc 54.4% vs Requested Dir Acc 55.1% (+0.7pp)');
  });

  it('promotes an explicitly approved kept run without reimplementing promotion logic', async () => {
    const promoteForecastLabFn = mock(async () => makePromotionResult());
    const tool = createForecastLabRunTool({ promoteForecastLabFn: promoteForecastLabFn as any });

    const result = await tool.invoke({
      action: 'promote-approved',
      profileId: 'btc-markov-ultra-short-horizon',
      sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(promoteForecastLabFn).toHaveBeenCalledWith({
      profileId: 'btc-markov-ultra-short-horizon',
      sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
      progress: undefined,
    });
    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'promote-approved',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      runId: 'forecast-lab-promo-1',
      sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
      activeStatePath: '.cramer-short/experiments/active-promotions/btc-markov-ultra-short-horizon.json',
    });
    expect(payload?.answer).toContain('promoted parameters are now live');
  });

  it('infers a unique promotable source when approval omits profile details', async () => {
    const promoteForecastLabFn = mock(async () => makePromotionResult());
    const readLedgerEntriesFn = mock(() => [
      {
        runId: 'btc-markov-ultra-short-horizon.keep-1',
        profileId: 'btc-markov-ultra-short-horizon',
        decision: 'keep',
        mutationMode: 'structured',
        promotion: {
          status: 'approval-required' as const,
          source: {
            runId: 'btc-markov-ultra-short-horizon.keep-1',
            manifestPath: '.cramer-short/experiments/runs/btc-markov-ultra-short-horizon.keep-1/manifest.json',
          },
          requestedAt: '2026-05-03T00:00:00.000Z',
        },
      },
    ] as any);
    const tool = createForecastLabRunTool({
      promoteForecastLabFn: promoteForecastLabFn as any,
      readLedgerEntriesFn,
      getLedgerPathFn: () => '.cramer-short/experiments/forecast-results.tsv',
    });

    await tool.invoke({
      action: 'promote-approved',
    });

    expect(readLedgerEntriesFn).toHaveBeenCalledWith('.cramer-short/experiments/forecast-results.tsv');
    expect(promoteForecastLabFn).toHaveBeenCalledWith({
      profileId: 'btc-markov-ultra-short-horizon',
      sourceRunId: 'btc-markov-ultra-short-horizon.keep-1',
      progress: undefined,
    });
  });

  it('returns a structured error when promotion approval is ambiguous', async () => {
    const promoteForecastLabFn = mock(async () => makePromotionResult());
    const tool = createForecastLabRunTool({
      promoteForecastLabFn: promoteForecastLabFn as any,
      readLedgerEntriesFn: () => [
        {
          runId: 'btc-markov-ultra-short-horizon.keep-1',
          profileId: 'btc-markov-ultra-short-horizon',
          decision: 'keep',
          mutationMode: 'structured',
          promotion: {
            status: 'approval-required' as const,
            source: {
              runId: 'btc-markov-ultra-short-horizon.keep-1',
              manifestPath: '.cramer-short/experiments/runs/btc-markov-ultra-short-horizon.keep-1/manifest.json',
            },
            requestedAt: '2026-05-03T00:00:00.000Z',
          },
        },
        {
          runId: 'multi-asset-markov-short-horizon.keep-2',
          profileId: 'multi-asset-markov-short-horizon',
          decision: 'keep',
          mutationMode: 'structured',
          promotion: {
            status: 'approval-required' as const,
            source: {
              runId: 'multi-asset-markov-short-horizon.keep-2',
              manifestPath: '.cramer-short/experiments/runs/multi-asset-markov-short-horizon.keep-2/manifest.json',
            },
            requestedAt: '2026-05-03T00:00:00.000Z',
          },
        },
      ] as any,
    });

    const result = await tool.invoke({
      action: 'promote-approved',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(promoteForecastLabFn).not.toHaveBeenCalled();
    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'promote-approved',
      status: 'error',
    });
    expect(payload?.answer).toContain('Specify profileId or sourceRunId');
  });

  it('runs a bounded reset back to shipped defaults', async () => {
    const resetForecastLabFn = mock(async () => makeResetResult());
    const tool = createForecastLabRunTool({ resetForecastLabFn: resetForecastLabFn as any });

    const result = await tool.invoke({
      action: 'reset-live',
      profileId: 'btc-markov-ultra-short-horizon',
      resetMode: 'defaults',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(resetForecastLabFn).toHaveBeenCalledWith({
      profileId: 'btc-markov-ultra-short-horizon',
      mode: 'defaults',
      progress: undefined,
    });
    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'reset-live',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      resetMode: 'defaults',
      runId: 'forecast-lab-reset-1',
    });
    expect(payload?.answer).toContain('shipped defaults');
  });

  it('returns a structured error when reset-live omits resetMode', async () => {
    const resetForecastLabFn = mock(async () => makeResetResult());
    const tool = createForecastLabRunTool({ resetForecastLabFn: resetForecastLabFn as any });

    const result = await tool.invoke({
      action: 'reset-live',
      profileId: 'btc-markov-ultra-short-horizon',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(resetForecastLabFn).not.toHaveBeenCalled();
    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'reset-live',
      status: 'error',
    });
    expect(payload?.answer).toContain('resetMode');
  });

  it('formats structured mutator lineage exhaustion errors as readable sections', async () => {
    const runForecastLabFn = mock(async () => {
      throw new Error(
        'Forecast-lab mutator "markov-faster-decay-reaction" is not applicable after replaying the kept parent lineage for profile "btc-markov-ultra-short-horizon". '
        + 'No shipped structured mutator remains applicable after replaying the kept parent lineage for profile "btc-markov-ultra-short-horizon". '
        + 'Current kept lineage already applied: markov-shorter-reactive-window, markov-faster-decay-reaction, markov-lower-confidence-trend-penalty. '
        + 'Remaining shipped mutators checked and found inapplicable: markov-longer-stability-window, markov-slower-decay-persistence, markov-higher-confidence-divergence-weighted, markov-calibrator-higher-sample-floor, markov-calibrator-lower-sample-floor. '
        + 'Next actions: keep the current best candidate, add a new shipped structured mutator, or intentionally reset the forecast-lab lineage outside the CLI.',
      );
    });
    const tool = createForecastLabRunTool({ runForecastLabFn: runForecastLabFn as any });

    const result = await tool.invoke({
      action: 'guided-improve',
      profileId: 'btc-markov-ultra-short-horizon',
      mutator: 'markov-faster-decay-reaction',
      query: 'Improve the BTC 1d/2d/3d Markov forecast workflow using mutator markov-faster-decay-reaction',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'guided-improve',
      status: 'error',
      error: expect.stringContaining('markov-faster-decay-reaction'),
    });
    expect(payload?.answer).toContain('Forecast-lab guided-improve could not continue.');
    expect(payload?.answer).toContain('| Field | Value |');
    expect(payload?.answer).toContain('| Requested mutator | markov-faster-decay-reaction |');
    expect(payload?.answer).toContain('Already applied in the kept lineage:');
    expect(payload?.answer).toContain('| # | Mutator id |');
    expect(payload?.answer).toContain('Remaining shipped mutators checked and found inapplicable:');
    expect(payload?.answer).toContain('1. Keep the current best candidate');
    expect(payload?.answer).toContain('2. Add a new shipped structured mutator');
  });

  it('includes the requested mutator id in successful guided-improve answers', async () => {
    const runForecastLabFn = mock(async () => makeRunResult());
    const tool = createForecastLabRunTool({ runForecastLabFn: runForecastLabFn as any });

    const result = await tool.invoke({
      action: 'guided-improve',
      profileId: 'btc-markov-ultra-short-horizon',
      mutator: 'markov-faster-decay-reaction',
      query: 'Improve the BTC 1d/2d/3d Markov forecast workflow using mutator markov-faster-decay-reaction',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'guided-improve',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
    });
    expect(payload?.answer).toContain('Requested mutator: markov-faster-decay-reaction.');
  });

  it('returns mutator scorecard with explicit profile', async () => {
    const readLedgerEntriesFn = mock(() => [
      {
        runId: 'run-1',
        profileId: 'btc-markov-ultra-short-horizon',
        startedAt: '2026-05-03T00:00:00.000Z',
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/run-1',
        allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
        mutationMode: 'structured',
        mutationId: 'markov-shorter-reactive-window',
        decision: 'keep',
        reason: 'passed',
        artifactsPath: '.cramer-short/experiments/runs/run-1',
        baselineSummary: { exitCode: 0 },
        candidateSummary: { exitCode: 0 },
      } as any,
      {
        runId: 'run-2',
        profileId: 'btc-markov-ultra-short-horizon',
        startedAt: '2026-05-04T00:00:00.000Z',
        targetSubsystem: 'markov-distribution',
        candidateBranch: 'topic/run-2',
        allowedGlobs: ['src/tools/finance/markov-distribution.ts'],
        mutationMode: 'structured',
        mutationId: 'markov-lower-confidence-trend-penalty',
        decision: 'drop',
        reason: 'regressed',
        artifactsPath: '.cramer-short/experiments/runs/run-2',
        baselineSummary: { exitCode: 0 },
        candidateSummary: { exitCode: 1 },
      } as any,
    ]);
    const readRunManifestFn = mock(() => {
      throw new Error('should not read manifest');
    });
    const readTextFileFn = mock(() => {
      throw new Error('should not read artifact');
    });
    const tool = createForecastLabRunTool({
      readLedgerEntriesFn: readLedgerEntriesFn as any,
      readRunManifestFn: readRunManifestFn as any,
      readTextFileFn,
    });

    const result = await tool.invoke({
      action: 'mutator-scorecard',
      profileId: 'btc-markov-ultra-short-horizon',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(readLedgerEntriesFn).toHaveBeenCalled();
    expect(readRunManifestFn).not.toHaveBeenCalled();
    expect(readTextFileFn).not.toHaveBeenCalled();
    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'mutator-scorecard',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      totalStructuredRuns: 2,
    });
    if (payload && 'rankedMutators' in payload) {
      expect(payload.rankedMutators.length).toBeGreaterThan(0);
    }
    expect(payload?.answer).toContain('Forecast-lab mutator scorecard for btc-markov-ultra-short-horizon.');
    expect(payload?.answer).toContain('Total structured runs: 2');
    expect(payload?.answer).toContain('| Mutator ID | Status | Behavior | Attempts | Kept | Regressed | Health | Score |');
  });

  it('returns error when mutator-scorecard is called with ambiguous profile resolution', async () => {
    const tool = createForecastLabRunTool();

    const result = await tool.invoke({
      action: 'mutator-scorecard',
      query: 'show me mutator health',
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'mutator-scorecard',
      status: 'error',
    });
    if (payload && 'error' in payload) {
      expect(payload.error).toContain('Multiple structured forecast-lab profiles exist');
      expect(payload.error).toContain('Please specify profileId');
    }
  });

  it('runs batch-replay-mutators and returns a comparative matrix', async () => {
    const runForecastLabFn = mock(async (options: any) => {
      const mutatorId = options.mutator;
      return makeRunResult({
        manifest: {
          runId: `replay-${mutatorId}`,
          profileId: 'btc-markov-ultra-short-horizon',
          artifactsPath: `.cramer-short/experiments/runs/replay-${mutatorId}`,
          mutationId: mutatorId,
          mutationSummary: `Test mutator ${mutatorId}`,
        },
        decision: {
          decision: mutatorId === 'markov-warmup-120' ? 'keep' : 'drop',
          reason: mutatorId === 'markov-warmup-120' ? 'passed gates' : 'failed gates',
          metrics: [],
        },
      });
    });

    const readTextFileFn = mock((path: string) => {
      if (path.endsWith('baseline.json')) {
        return JSON.stringify({
          commands: [{
            id: 'walk-forward-btc-ultra-short-horizon',
            stdout: 'BTC-USD horizon 1d\nbaseline warmup=120 stride=3 │ 100 │ 50 │ 60.0% │ 0.200 │ 95.0%\n',
          }],
        });
      }
      if (path.endsWith('candidate.json')) {
        return JSON.stringify({
          commands: [{
            id: 'walk-forward-btc-ultra-short-horizon',
            stdout: 'BTC-USD horizon 1d\nbaseline warmup=120 stride=3 │ 100 │ 50 │ 62.0% │ 0.190 │ 96.0%\n',
          }],
        });
      }
      return '{}';
    });

    const tool = createForecastLabRunTool({
      runForecastLabFn: runForecastLabFn as any,
      readTextFileFn: readTextFileFn as any,
    });

    const result = await tool.invoke({
      action: 'batch-replay-mutators',
      profileId: 'btc-markov-ultra-short-horizon',
      limit: 2,
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(runForecastLabFn).toHaveBeenCalledTimes(2);
    expect(runForecastLabFn).toHaveBeenCalledWith(
      expect.objectContaining({
        profileId: 'btc-markov-ultra-short-horizon',
        mutationMode: 'structured',
        forceNoParent: true,
        diagnosticOnly: true,
      }),
    );

    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'batch-replay-mutators',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      baselineDescription: 'shipped defaults (fresh runs with no parent lineage)',
      replayedCount: 2,
    });
    expect(Array.isArray((payload as any)?.results)).toBe(true);
    expect(payload?.answer).toContain('batch replay');
    expect(payload?.answer).toContain('shipped defaults');
  });

  it('rejects batch-replay-mutators when profileId is missing', async () => {
    const runForecastLabFn = mock(async () => makeRunResult());
    const tool = createForecastLabRunTool({ runForecastLabFn: runForecastLabFn as any });

    const result = await tool.invoke({
      action: 'batch-replay-mutators',
    } as any);
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(runForecastLabFn).not.toHaveBeenCalled();
    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'batch-replay-mutators',
      status: 'error',
    });
    expect(payload?.answer).toContain('batch-replay-mutators requires profileId');
  });

  it('handles batch-replay-mutators errors gracefully', async () => {
    const runForecastLabFn = mock(async (options: any) => {
      const mutatorId = options.mutator;
      if (mutatorId === 'markov-shorter-reactive-window') {
        throw new Error('Replay failed for markov-shorter-reactive-window');
      }
      return makeRunResult({
        manifest: {
          runId: `replay-${mutatorId}`,
          profileId: 'btc-markov-ultra-short-horizon',
          artifactsPath: `.cramer-short/experiments/runs/replay-${mutatorId}`,
        },
      });
    });

    const readTextFileFn = mock(() => JSON.stringify({ commands: [] }));
    const tool = createForecastLabRunTool({
      runForecastLabFn: runForecastLabFn as any,
      readTextFileFn: readTextFileFn as any,
    });

    const result = await tool.invoke({
      action: 'batch-replay-mutators',
      profileId: 'btc-markov-ultra-short-horizon',
      limit: 2,
    });
    const payload = parseForecastLabRunToolPayload(result as string);

    expect(payload).toMatchObject({
      _tool: 'forecast_lab_run',
      action: 'batch-replay-mutators',
      status: 'ok',
      profileId: 'btc-markov-ultra-short-horizon',
      replayedCount: 2,
    });

    const errorResult = (payload as any)?.results?.find((r: any) => r.mutatorId === 'markov-shorter-reactive-window');
    expect(errorResult).toMatchObject({
      mutatorId: 'markov-shorter-reactive-window',
      decision: 'drop',
      reason: expect.stringContaining('error:'),
    });
  });

  it('rejects batch-replay-mutators with invalid limit values', async () => {
    const runForecastLabFn = mock(async () => makeRunResult());
    const tool = createForecastLabRunTool({ runForecastLabFn: runForecastLabFn as any });

    // Test negative limit
    await expect(
      tool.invoke({
        action: 'batch-replay-mutators',
        profileId: 'btc-markov-ultra-short-horizon',
        limit: -1,
      } as any),
    ).rejects.toThrow();

    // Test zero limit
    await expect(
      tool.invoke({
        action: 'batch-replay-mutators',
        profileId: 'btc-markov-ultra-short-horizon',
        limit: 0,
      } as any),
    ).rejects.toThrow();

    // Test exceeding max limit
    await expect(
      tool.invoke({
        action: 'batch-replay-mutators',
        profileId: 'btc-markov-ultra-short-horizon',
        limit: 51,
      } as any),
    ).rejects.toThrow();

    // Test non-integer limit
    await expect(
      tool.invoke({
        action: 'batch-replay-mutators',
        profileId: 'btc-markov-ultra-short-horizon',
        limit: 3.5,
      } as any),
    ).rejects.toThrow();

    expect(runForecastLabFn).not.toHaveBeenCalled();
  });
});
