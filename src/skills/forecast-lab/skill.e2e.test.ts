/**
 * E2E tests — forecast-lab skill with real Ollama model.
 *
 * Run with:  bun run test:e2e
 * Skipped in normal `bun test` / CI runs.
 *
 * This dry-run prompt proves the skill can be invoked without allowing the
 * agent to mutate source files during the E2E check.
 */
import { describe, expect, beforeAll } from 'bun:test';
import { e2eIt, RUN_E2E } from '@/utils/test-guards.js';
import { runAgentE2EWithTimeoutRetry, E2E_TIMEOUT_MS } from '@/utils/e2e-helpers.js';
import type { E2EResult, E2ESeedMessage } from '@/utils/e2e-helpers.js';
import type { ToolStartEvent } from '@/agent/types.js';

const FORECAST_LAB_OPTIMIZATION_QUERY =
  'Optimize the BTC 1d/2d/3d Markov forecast workflow in a bounded baseline-first way. Do not edit files, run shell commands, or write artifacts; explain the exact experiment plan you would follow.';

const ORDINARY_BTC_FORECAST_QUERY =
  'Give me a BTC forecast for the next 7 days and explain the main drivers.';
const FORECAST_LAB_LIFECYCLE_QUERY =
  'Use the forecast-lab skill to explain the full BTC ultra-short-horizon lifecycle after a kept candidate: approval-required promotion, activation for ordinary forecasts, and how to reset to shipped defaults or the last known-good baseline if the promoted parameters mislead. Do not edit files, run shell commands, or write artifacts.';
const FORECAST_LAB_EXHAUSTION_QUERY =
  'Given this forecast-lab failure: "No shipped structured mutator remains applicable after replaying the kept parent lineage for btc-markov-ultra-short-horizon." Explain the supported next actions only. Do not edit files, run shell commands, or write artifacts.';
const FORECAST_LAB_MUTATOR_GUIDANCE_QUERY =
  'Use the forecast-lab skill to explain how to force a specific shipped mutator for btc-markov-ultra-short-horizon. Do not edit files, run shell commands, or write artifacts.';
const FORECAST_LAB_MUTATOR_EXECUTION_QUERY =
  'Improve the BTC 1d/2d/3d Markov forecast workflow using mutator markov-faster-decay-reaction.';
const FORECAST_LAB_APPROVAL_COMMAND_QUERY =
  'Approve forecast-lab promotion for that kept run.';
const FORECAST_LAB_COMPARISON_QUERY =
  'Is the current best better than the shipped default baseline?';
const FORECAST_LAB_RESULTS_QUERY =
  'provide the results of the Optimize the BTC 1d/2d/3d Markov forecast workflow';
const FORECAST_LAB_KEEP_CURRENT_BEST_QUERY =
  'keep the current best candidate';
const FORECAST_LAB_CATALOG_EXTENSION_QUERY =
  'design a new shipped mutator outside the existing catalog and re-run the lineage';
const FORECAST_LAB_CATALOG_EXTENSION_IMPLEMENTATION_QUERY = [
  'Target anchor trust weighting.',
  'Add a new shipped structured mutator for btc-markov-ultra-short-horizon that makes the Markov/anchor blend more adaptive under high posterior entropy using the existing soft-regime weighting controls in src/tools/finance/markov-distribution.ts.',
  'Mutator goal:',
  '- reduce retained HMM weight when the regime posterior is ambiguous,',
  '- widen CI slightly more under entropy,',
  '- lower confidence more under entropy,',
  '- keep the change bounded to the existing soft-regime weighting parameters.',
  'Suggested starting values:',
  '- softRegimeConfidenceFloor: 0.65 -> 0.55',
  '- softRegimeConfidenceEntropyWeight: 0.35 -> 0.50',
  '- softRegimeCiEntropyWeight: 0.35 -> 0.50',
  '- softRegimeHmmWeightFloor: 0.50 -> 0.35',
  '- softRegimeHmmWeightEntropyWeight: 0.40 -> 0.60',
  'Name it something like:',
  'markov-entropy-adaptive-anchor-weighting',
  'Keep it bounded, add the shipped mutator to the catalog, and validate it with the existing BTC ultra-short-horizon walk-forward gate.',
].join('\n');
const FORECAST_LAB_MUTATOR_VS_ACTIVE_QUERY =
  'I need to compare the markov-entropy-adaptive-anchor-weighting with the active one, I need to see the accurace numbers';
const FORECAST_LAB_HISTORY_MUTATOR_VS_ACTIVE_QUERY =
  'I need to compare the live one with the new mutate one that I created and it is not promoted';
const FORECAST_LAB_IMPLEMENT_NEW_MUTATOR_QUERY =
  'implement and run the markov-entropy-adaptive-anchor-weighting';
const FORECAST_LAB_LIST_MUTATORS_QUERY =
  'List the mutate availible';

const FORECAST_LAB_APPROVAL_HISTORY: readonly E2ESeedMessage[] = [
  {
    query: 'Improve the BTC 1d/2d/3d Markov forecast workflow.',
    answer: 'Forecast-lab guided improvement finished. Approval required before promotion. Reply "approve forecast-lab promotion for btc-markov-ultra-short-horizon run btc-markov-ultra-short-horizon.keep-1" to continue.',
    summary: null,
  },
];

const FORECAST_LAB_EXHAUSTED_LINEAGE_HISTORY: readonly E2ESeedMessage[] = [
  {
    query: 'Use the forecast-lab skill to explain what to do when no shipped structured mutator remains applicable after replaying the kept parent lineage for btc-markov-ultra-short-horizon.',
    answer: 'No shipped structured mutator remains applicable after replaying the kept parent lineage for profile "btc-markov-ultra-short-horizon". Next actions: keep the current best candidate, add a new shipped structured mutator, or intentionally reset the forecast-lab lineage outside the CLI.',
    summary: null,
  },
];

let optimizationResult: E2EResult;
let optimizationTools: string[];
let optimizationAnswer: string;
let ordinaryForecastResult: E2EResult;
let lifecycleResult: E2EResult;
let lifecycleTools: string[];
let lifecycleAnswer: string;
let exhaustionResult: E2EResult;
let exhaustionTools: string[];
let exhaustionAnswer: string;
let mutatorGuidanceResult: E2EResult;
let mutatorGuidanceTools: string[];
let mutatorGuidanceAnswer: string;
let mutatorExecutionResult: E2EResult;
let mutatorExecutionTools: string[];
let mutatorExecutionAnswer: string;
let approvalCommandResult: E2EResult;
let approvalCommandTools: string[];
let approvalCommandAnswer: string;
let comparisonResult: E2EResult;
let comparisonTools: string[];
let comparisonAnswer: string;
let resultsQueryResult: E2EResult;
let resultsQueryTools: string[];
let resultsQueryAnswer: string;
let keepCurrentBestResult: E2EResult;
let keepCurrentBestTools: string[];
let keepCurrentBestAnswer: string;
let catalogExtensionResult: E2EResult;
let catalogExtensionTools: string[];
let catalogExtensionAnswer: string;
let catalogExtensionImplementationResult: E2EResult;
let catalogExtensionImplementationTools: string[];
let catalogExtensionImplementationAnswer: string;
let mutatorVsActiveResult: E2EResult;
let mutatorVsActiveTools: string[];
let mutatorVsActiveAnswer: string;
let historyMutatorVsActiveResult: E2EResult;
let historyMutatorVsActiveTools: string[];
let historyMutatorVsActiveAnswer: string;
let implementNewMutatorResult: E2EResult;
let implementNewMutatorTools: string[];
let implementNewMutatorAnswer: string;
let listMutatorsResult: E2EResult;
let listMutatorsTools: string[];
let listMutatorsAnswer: string;

function findSkillCall(result: E2EResult, skillName: string): ToolStartEvent | undefined {
  return result.events.find((event): event is ToolStartEvent => {
    if (!event || typeof event !== 'object') return false;
    if (event.type !== 'tool_start' || event.tool !== 'skill') return false;
    return event.args?.skill === skillName;
  });
}

function findToolCall(result: E2EResult, toolName: string): ToolStartEvent | undefined {
  return result.events.find((event): event is ToolStartEvent => {
    if (!event || typeof event !== 'object') return false;
    if (event.type !== 'tool_start' || event.tool !== toolName) return false;
    return true;
  });
}

describe('forecast-lab skill E2E', () => {
  beforeAll(async () => {
    if (!RUN_E2E) return;
    optimizationResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_OPTIMIZATION_QUERY, {
      maxIterations: 6,
    });
    optimizationTools = optimizationResult.toolsCalled;
    optimizationAnswer = optimizationResult.answer;
    ordinaryForecastResult = await runAgentE2EWithTimeoutRetry(ORDINARY_BTC_FORECAST_QUERY, {
      maxIterations: 6,
    });
    lifecycleResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_LIFECYCLE_QUERY, {
      maxIterations: 6,
    });
    lifecycleTools = lifecycleResult.toolsCalled;
    lifecycleAnswer = lifecycleResult.answer;
    exhaustionResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_EXHAUSTION_QUERY, {
      maxIterations: 6,
      historySeed: FORECAST_LAB_EXHAUSTED_LINEAGE_HISTORY,
    });
    exhaustionTools = exhaustionResult.toolsCalled;
    exhaustionAnswer = exhaustionResult.answer;
    mutatorGuidanceResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_MUTATOR_GUIDANCE_QUERY, {
      maxIterations: 6,
    });
    mutatorGuidanceTools = mutatorGuidanceResult.toolsCalled;
    mutatorGuidanceAnswer = mutatorGuidanceResult.answer;
    mutatorExecutionResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_MUTATOR_EXECUTION_QUERY, {
      maxIterations: 6,
    });
    mutatorExecutionTools = mutatorExecutionResult.toolsCalled;
    mutatorExecutionAnswer = mutatorExecutionResult.answer;
    approvalCommandResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_APPROVAL_COMMAND_QUERY, {
      maxIterations: 4,
      historySeed: FORECAST_LAB_APPROVAL_HISTORY,
    });
    approvalCommandTools = approvalCommandResult.toolsCalled;
    approvalCommandAnswer = approvalCommandResult.answer;
    comparisonResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_COMPARISON_QUERY, {
      maxIterations: 4,
      historySeed: FORECAST_LAB_APPROVAL_HISTORY,
    });
    comparisonTools = comparisonResult.toolsCalled;
    comparisonAnswer = comparisonResult.answer;
    resultsQueryResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_RESULTS_QUERY, {
      maxIterations: 4,
    });
    resultsQueryTools = resultsQueryResult.toolsCalled;
    resultsQueryAnswer = resultsQueryResult.answer;
    keepCurrentBestResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_KEEP_CURRENT_BEST_QUERY, {
      maxIterations: 4,
      historySeed: FORECAST_LAB_APPROVAL_HISTORY,
    });
    keepCurrentBestTools = keepCurrentBestResult.toolsCalled;
    keepCurrentBestAnswer = keepCurrentBestResult.answer;
    catalogExtensionResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_CATALOG_EXTENSION_QUERY, {
      maxIterations: 4,
      historySeed: FORECAST_LAB_EXHAUSTED_LINEAGE_HISTORY,
    });
    catalogExtensionTools = catalogExtensionResult.toolsCalled;
    catalogExtensionAnswer = catalogExtensionResult.answer;
    catalogExtensionImplementationResult = await runAgentE2EWithTimeoutRetry(
      FORECAST_LAB_CATALOG_EXTENSION_IMPLEMENTATION_QUERY,
      {
        maxIterations: 4,
      },
    );
    catalogExtensionImplementationTools = catalogExtensionImplementationResult.toolsCalled;
    catalogExtensionImplementationAnswer = catalogExtensionImplementationResult.answer;
    mutatorVsActiveResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_MUTATOR_VS_ACTIVE_QUERY, {
      maxIterations: 4,
      historySeed: [
        {
          query: 'Target anchor trust weighting. Add a new shipped structured mutator for btc-markov-ultra-short-horizon.',
          answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. Requested mutator id: markov-entropy-adaptive-anchor-weighting.',
          summary: null,
        },
      ],
    });
    mutatorVsActiveTools = mutatorVsActiveResult.toolsCalled;
    mutatorVsActiveAnswer = mutatorVsActiveResult.answer;
    historyMutatorVsActiveResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_HISTORY_MUTATOR_VS_ACTIVE_QUERY, {
      maxIterations: 4,
      historySeed: [
        {
          query: 'Target anchor trust weighting. Add a new shipped structured mutator for btc-markov-ultra-short-horizon.',
          answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. Requested mutator id: markov-entropy-adaptive-anchor-weighting.',
          summary: null,
        },
      ],
    });
    historyMutatorVsActiveTools = historyMutatorVsActiveResult.toolsCalled;
    historyMutatorVsActiveAnswer = historyMutatorVsActiveResult.answer;
    implementNewMutatorResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_IMPLEMENT_NEW_MUTATOR_QUERY, {
      maxIterations: 4,
      historySeed: [
        {
          query: 'Target anchor trust weighting. Add a new shipped structured mutator for btc-markov-ultra-short-horizon.',
          answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. Requested mutator id: markov-entropy-adaptive-anchor-weighting.',
          summary: null,
        },
      ],
    });
    implementNewMutatorTools = implementNewMutatorResult.toolsCalled;
    implementNewMutatorAnswer = implementNewMutatorResult.answer;
    listMutatorsResult = await runAgentE2EWithTimeoutRetry(FORECAST_LAB_LIST_MUTATORS_QUERY, {
      maxIterations: 4,
      historySeed: FORECAST_LAB_APPROVAL_HISTORY,
    });
    listMutatorsTools = listMutatorsResult.toolsCalled;
    listMutatorsAnswer = listMutatorsResult.answer;
  }, E2E_TIMEOUT_MS);

  e2eIt('invokes the forecast-lab skill for optimization queries', () => {
    expect(
      Boolean(findSkillCall(optimizationResult, 'forecast-lab')),
      `skill(forecast-lab) must be called for optimization routing. Tools: [${optimizationTools.join(', ')}]`,
    ).toBe(true);
  });

  e2eIt('describes the baseline-first candidate comparison', () => {
    const lower = optimizationAnswer.toLowerCase();
    expect(lower, 'answer must mention baseline').toMatch(/baseline/);
    expect(lower, 'answer must mention candidate comparison').toMatch(/candidate/);
    expect(lower, 'answer must mention fixed gates').toMatch(/gate|harness|metric/);
  });

  e2eIt('includes bounded mutation and drop/revert rules', () => {
    const lower = optimizationAnswer.toLowerCase();
    expect(lower, 'answer must mention allowlisted forecast files').toMatch(/allowlist|approved|editable/);
    expect(lower, 'answer must mention dropping or reverting failed candidates').toMatch(/drop|revert|discard/);
  });

  e2eIt('points experiment records at .cramer-short/experiments', () => {
    expect(optimizationAnswer).toMatch(/\.cramer-short\/experiments/);
  });

  e2eIt('does not mutate files in dry-run mode', () => {
    expect(
      optimizationTools.some((t) => t === 'write_file' || t === 'edit_file'),
      `dry-run skill invocation must not write or edit files. Tools: [${optimizationTools.join(', ')}]`,
    ).toBe(false);
  });

  e2eIt('does not auto-enter forecast-lab for ordinary BTC forecast queries', () => {
    expect(findSkillCall(ordinaryForecastResult, 'forecast-lab')).toBeUndefined();
  });

  e2eIt('explains approval-required promotion, live activation, and reset options', () => {
    const lower = lifecycleAnswer.toLowerCase();
    expect(lower, 'answer must mention explicit approval before promotion').toMatch(/approval|required|approve/);
    expect(lower, 'answer must mention the parameters becoming live for ordinary forecasts').toMatch(/live|ordinary forecast|normal forecast/);
    expect(lower, 'answer must mention reset to shipped defaults').toMatch(/shipped defaults|defaults/);
    expect(lower, 'answer must mention reset to last known-good baseline').toMatch(/last known good|known-good|previous activated/);
  });

  e2eIt('does not mutate files while explaining the lifecycle', () => {
    expect(
      lifecycleTools.some((t) => t === 'write_file' || t === 'edit_file'),
      `lifecycle explanation must not write or edit files. Tools: [${lifecycleTools.join(', ')}]`,
    ).toBe(false);
  });

  e2eIt('explains exhausted-lineage next actions without mutating files', () => {
    const lower = exhaustionAnswer.toLowerCase();
    expect(lower, 'answer must mention the exhausted mutator state').toMatch(/no shipped|exhausted.*shipped.*mutator|lineage|applicable|can be applied/);
    expect(lower, 'answer must mention one of the supported next actions').toMatch(
      /keep the current best candidate|add a new shipped structured mutator|reset|catalog be extended|catalog update|extend the catalog|catalog extension|different profile|request a new mutator|human review|bounded plan|stop/,
    );
    expect(
      exhaustionTools.some((t) => t === 'write_file' || t === 'edit_file'),
      `exhaustion guidance must not write or edit files. Tools: [${exhaustionTools.join(', ')}]`,
    ).toBe(false);
  });

  e2eIt('explains how to force a specific shipped mutator', () => {
    const lower = mutatorGuidanceAnswer.toLowerCase();
    expect(Boolean(findSkillCall(mutatorGuidanceResult, 'forecast-lab'))).toBe(true);
    expect(mutatorGuidanceAnswer).toMatch(/--mutation structured[\s\\]+--mutator/);
    expect(lower, 'answer must mention the btc ultra-short-horizon profile').toMatch(/btc-markov-ultra-short-horizon/);
    expect(
      mutatorGuidanceTools.some((t) => t === 'write_file' || t === 'edit_file'),
      `mutator guidance must not write or edit files. Tools: [${mutatorGuidanceTools.join(', ')}]`,
    ).toBe(false);
  });

  e2eIt('passes explicit mutator ids through routed agent-mode improvement prompts', () => {
    const runCall = findToolCall(mutatorExecutionResult, 'forecast_lab_run');
    expect(Boolean(findSkillCall(mutatorExecutionResult, 'forecast-lab'))).toBe(true);
    expect(mutatorExecutionTools).toContain('forecast_lab_run');
    expect(runCall?.args?.mutator).toBe('markov-faster-decay-reaction');
    expect(mutatorExecutionAnswer.toLowerCase()).toContain('markov-faster-decay-reaction');
  });

  e2eIt('routes explicit approval commands through forecast_lab_run using seeded history', () => {
    const lower = approvalCommandAnswer.toLowerCase();
    expect(approvalCommandTools).toContain('forecast_lab_run');
    expect(findSkillCall(approvalCommandResult, 'forecast-lab')).toBeUndefined();
    expect(lower, 'answer must describe the missing approval source instead of mutating files').toMatch(
      /promotion source|approval-required forecast-lab promotion source|was not found/,
    );
    expect(
      approvalCommandTools.some((t) => t === 'write_file' || t === 'edit_file'),
      `approval routing must not write or edit files. Tools: [${approvalCommandTools.join(', ')}]`,
    ).toBe(false);
  });

  e2eIt('routes current-best vs shipped-baseline questions through forecast_lab_run without filesystem probing', () => {
    const lower = comparisonAnswer.toLowerCase();
    expect(comparisonTools).toContain('forecast_lab_run');
    expect(findSkillCall(comparisonResult, 'forecast-lab')).toBeUndefined();
    expect(comparisonTools).not.toContain('read_file');
    expect(comparisonTools).not.toContain('web_fetch');
    expect(comparisonTools).not.toContain('browser');
    expect(lower).toMatch(/shipped baseline|current best|not live yet|already live/);
  });

  e2eIt('routes forecast-lab results prompts through forecast_lab_run instead of guided improvement', () => {
    const lower = resultsQueryAnswer.toLowerCase();
    expect(resultsQueryTools).toContain('forecast_lab_run');
    expect(resultsQueryTools).not.toContain('read_file');
    expect(resultsQueryTools).not.toContain('web_fetch');
    expect(resultsQueryTools).not.toContain('browser');
    expect(lower).toMatch(/shipped baseline|current best|approve forecast-lab promotion|already live|not live yet/);
  });

  e2eIt('routes keep-the-current-best follow-ups through forecast_lab_run without edit attempts', () => {
    const lower = keepCurrentBestAnswer.toLowerCase();
    expect(keepCurrentBestTools).toContain('forecast_lab_run');
    expect(keepCurrentBestTools).not.toContain('edit_file');
    expect(keepCurrentBestTools).not.toContain('write_file');
    expect(keepCurrentBestTools).not.toContain('read_file');
    expect(lower).toMatch(/current best|approve forecast-lab promotion|already live|not live yet/);
  });

  e2eIt('routes catalog-extension prompts through forecast_lab_run without filesystem or browser probing', () => {
    const lower = catalogExtensionAnswer.toLowerCase();
    expect(catalogExtensionTools).toContain('forecast_lab_run');
    expect(catalogExtensionTools).not.toContain('skill');
    expect(catalogExtensionTools).not.toContain('sequential_thinking');
    expect(catalogExtensionTools).not.toContain('read_file');
    expect(catalogExtensionTools).not.toContain('web_fetch');
    expect(catalogExtensionTools).not.toContain('browser');
    expect(lower).toMatch(/bounded code-change plan|did not inspect experiment artifacts|catalog-extension plan|rerun the lineage/);
  });

  e2eIt('routes detailed mutator implementation briefs through forecast_lab_run without repo or web probing', () => {
    const lower = catalogExtensionImplementationAnswer.toLowerCase();
    expect(catalogExtensionImplementationTools).toContain('forecast_lab_run');
    expect(catalogExtensionImplementationTools).not.toContain('skill');
    expect(catalogExtensionImplementationTools).not.toContain('sequential_thinking');
    expect(catalogExtensionImplementationTools).not.toContain('read_file');
    expect(catalogExtensionImplementationTools).not.toContain('web_fetch');
    expect(catalogExtensionImplementationTools).not.toContain('browser');
    expect(lower).toMatch(/markov-entropy-adaptive-anchor-weighting|catalog files to open directly next|requested parameter deltas/);
  });

  e2eIt('routes mutator-vs-active comparison prompts through forecast_lab_run without filesystem probing', () => {
    const lower = mutatorVsActiveAnswer.toLowerCase();
    expect(mutatorVsActiveTools).toContain('forecast_lab_run');
    expect(mutatorVsActiveTools).not.toContain('read_file');
    expect(mutatorVsActiveTools).not.toContain('web_fetch');
    expect(mutatorVsActiveTools).not.toContain('browser');
    expect(mutatorVsActiveTools).not.toContain('skill');
    expect(mutatorVsActiveTools).not.toContain('sequential_thinking');
    expect(lower).toMatch(/active|mutation|not found|accuracy|dir acc|compare/);
  });

  e2eIt('routes history-based live-vs-new-mutator prompts through forecast_lab_run without filesystem probing', () => {
    const lower = historyMutatorVsActiveAnswer.toLowerCase();
    expect(historyMutatorVsActiveTools).toContain('forecast_lab_run');
    expect(historyMutatorVsActiveTools).not.toContain('read_file');
    expect(historyMutatorVsActiveTools).not.toContain('web_fetch');
    expect(historyMutatorVsActiveTools).not.toContain('browser');
    expect(historyMutatorVsActiveTools).not.toContain('skill');
    expect(historyMutatorVsActiveTools).not.toContain('sequential_thinking');
    expect(lower).toMatch(/active|mutation|not found|accuracy|dir acc|compare/);
  });

  e2eIt('routes implement-and-run requests for non-shipped mutators through catalog-extension guidance without repo probing', () => {
    const lower = implementNewMutatorAnswer.toLowerCase();
    expect(implementNewMutatorTools).toContain('forecast_lab_run');
    expect(implementNewMutatorTools).not.toContain('skill');
    expect(implementNewMutatorTools).not.toContain('sequential_thinking');
    expect(implementNewMutatorTools).not.toContain('read_file');
    expect(implementNewMutatorTools).not.toContain('web_fetch');
    expect(implementNewMutatorTools).not.toContain('browser');
    expect(implementNewMutatorTools).not.toContain('edit_file');
    expect(implementNewMutatorTools).not.toContain('write_file');
    expect(lower).toMatch(/bounded code-change plan|requested mutator id|did not inspect experiment artifacts|catalog-extension plan/);
  });

  e2eIt('routes mutator-list prompts through forecast_lab_run without filesystem or web probing', () => {
    const lower = listMutatorsAnswer.toLowerCase();
    expect(listMutatorsTools).toContain('forecast_lab_run');
    expect(listMutatorsTools).not.toContain('skill');
    expect(listMutatorsTools).not.toContain('sequential_thinking');
    expect(listMutatorsTools).not.toContain('read_file');
    expect(listMutatorsTools).not.toContain('web_fetch');
    expect(listMutatorsTools).not.toContain('browser');
    expect(lower).toMatch(/shipped mutator ids|mutator catalog summary|markov-shorter-reactive-window|markov-faster-decay-reaction/);
  });
});
