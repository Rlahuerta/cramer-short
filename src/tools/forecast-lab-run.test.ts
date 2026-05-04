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
      profiles: [
        {
          profileId: 'multi-asset-markov-short-horizon',
          mutationMode: 'structured',
        },
        {
          profileId: 'btc-markov-ultra-short-horizon',
          mutationMode: 'structured',
        },
      ],
      dryRunProfiles: ['btc-arbiter-replay', 'polymarket-selection-sanity'],
    });
    expect(payload?.answer).toContain('Forecast-lab shipped mutator catalog summary.');
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
        mutationSummary: 'Lower the confidence gate and enable the trend-only break penalty ablation.',
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
        mutationSummary: 'Lower the confidence gate and enable the trend-only break penalty ablation.',
        lineage: { generation: 2, rootRunId: 'root-run' },
        mutationSpecSummary: { mutatorId: 'search-replace', summary: 'Lower the confidence gate and enable the trend-only break penalty ablation.', targetFiles: ['src/tools/finance/markov-distribution.ts'] },
        mutationReplayPayload: { kind: 'markov-parameter-candidate', id: 'markov-lower-confidence-trend-penalty', profileId: 'btc-markov-ultra-short-horizon', mutatorId: 'search-replace', edits: [], patchSummary: [], specSummary: { mutatorId: 'search-replace', summary: 'Lower the confidence gate and enable the trend-only break penalty ablation.', targetFiles: ['src/tools/finance/markov-distribution.ts'] } },
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
        mutationSummary: 'Lower the confidence gate and enable the trend-only break penalty ablation.',
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
});
