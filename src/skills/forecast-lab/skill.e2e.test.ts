/**
 * E2E tests — forecast-lab skill with real Ollama model.
 *
 * Run with:  bun run test:e2e
 * Skipped in normal `bun test` / CI runs.
 *
 * This dry-run prompt proves the skill can be invoked without allowing the
 * agent to mutate source files during the E2E check.
 */
import { describe, expect } from 'bun:test';
import { e2eIt } from '@/utils/test-guards.js';
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

interface ForecastLabE2EFixture {
  result: E2EResult;
  tools: string[];
  answer: string;
}

type ForecastLabE2EOptions = Parameters<typeof runAgentE2EWithTimeoutRetry>[1];

function createForecastLabE2EFixture(query: string, opts: ForecastLabE2EOptions = {}) {
  let fixturePromise: Promise<ForecastLabE2EFixture> | null = null;

  return async () => {
    if (!fixturePromise) {
      fixturePromise = runAgentE2EWithTimeoutRetry(query, opts).then((result) => ({
        result,
        tools: result.toolsCalled,
        answer: result.answer,
      }));
    }

    return fixturePromise;
  };
}

const getOptimizationFixture = createForecastLabE2EFixture(FORECAST_LAB_OPTIMIZATION_QUERY, {
  maxIterations: 6,
});
const getOrdinaryForecastFixture = createForecastLabE2EFixture(ORDINARY_BTC_FORECAST_QUERY, {
  maxIterations: 6,
});
const getLifecycleFixture = createForecastLabE2EFixture(FORECAST_LAB_LIFECYCLE_QUERY, {
  maxIterations: 6,
});
const getExhaustionFixture = createForecastLabE2EFixture(FORECAST_LAB_EXHAUSTION_QUERY, {
  maxIterations: 6,
  historySeed: FORECAST_LAB_EXHAUSTED_LINEAGE_HISTORY,
});
const getMutatorGuidanceFixture = createForecastLabE2EFixture(FORECAST_LAB_MUTATOR_GUIDANCE_QUERY, {
  maxIterations: 6,
});
const getMutatorExecutionFixture = createForecastLabE2EFixture(FORECAST_LAB_MUTATOR_EXECUTION_QUERY, {
  maxIterations: 6,
});
const getApprovalCommandFixture = createForecastLabE2EFixture(FORECAST_LAB_APPROVAL_COMMAND_QUERY, {
  maxIterations: 4,
  historySeed: FORECAST_LAB_APPROVAL_HISTORY,
});
const getComparisonFixture = createForecastLabE2EFixture(FORECAST_LAB_COMPARISON_QUERY, {
  maxIterations: 4,
  historySeed: FORECAST_LAB_APPROVAL_HISTORY,
});
const getResultsQueryFixture = createForecastLabE2EFixture(FORECAST_LAB_RESULTS_QUERY, {
  maxIterations: 4,
});
const getKeepCurrentBestFixture = createForecastLabE2EFixture(FORECAST_LAB_KEEP_CURRENT_BEST_QUERY, {
  maxIterations: 4,
  historySeed: FORECAST_LAB_APPROVAL_HISTORY,
});
const getCatalogExtensionFixture = createForecastLabE2EFixture(FORECAST_LAB_CATALOG_EXTENSION_QUERY, {
  maxIterations: 4,
  historySeed: FORECAST_LAB_EXHAUSTED_LINEAGE_HISTORY,
});
const getCatalogExtensionImplementationFixture = createForecastLabE2EFixture(
  FORECAST_LAB_CATALOG_EXTENSION_IMPLEMENTATION_QUERY,
  {
    maxIterations: 4,
  },
);
const getMutatorVsActiveFixture = createForecastLabE2EFixture(FORECAST_LAB_MUTATOR_VS_ACTIVE_QUERY, {
  maxIterations: 4,
  historySeed: [
    {
      query: 'Target anchor trust weighting. Add a new shipped structured mutator for btc-markov-ultra-short-horizon.',
      answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. Requested mutator id: markov-entropy-adaptive-anchor-weighting.',
      summary: null,
    },
  ],
});
const getHistoryMutatorVsActiveFixture = createForecastLabE2EFixture(FORECAST_LAB_HISTORY_MUTATOR_VS_ACTIVE_QUERY, {
  maxIterations: 4,
  historySeed: [
    {
      query: 'Target anchor trust weighting. Add a new shipped structured mutator for btc-markov-ultra-short-horizon.',
      answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. Requested mutator id: markov-entropy-adaptive-anchor-weighting.',
      summary: null,
    },
  ],
});
const getImplementNewMutatorFixture = createForecastLabE2EFixture(FORECAST_LAB_IMPLEMENT_NEW_MUTATOR_QUERY, {
  maxIterations: 4,
  historySeed: [
    {
      query: 'Target anchor trust weighting. Add a new shipped structured mutator for btc-markov-ultra-short-horizon.',
      answer: 'Forecast-lab catalog-extension plan for btc-markov-ultra-short-horizon. Requested mutator id: markov-entropy-adaptive-anchor-weighting.',
      summary: null,
    },
  ],
});
const getListMutatorsFixture = createForecastLabE2EFixture(FORECAST_LAB_LIST_MUTATORS_QUERY, {
  maxIterations: 4,
  historySeed: FORECAST_LAB_APPROVAL_HISTORY,
});

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
  e2eIt('invokes the forecast-lab skill for optimization queries', async () => {
    const { result, tools } = await getOptimizationFixture();
    expect(
      Boolean(findSkillCall(result, 'forecast-lab')),
      `skill(forecast-lab) must be called for optimization routing. Tools: [${tools.join(', ')}]`,
    ).toBe(true);
  }, E2E_TIMEOUT_MS);

  e2eIt('describes the baseline-first candidate comparison', async () => {
    const { answer } = await getOptimizationFixture();
    const lower = answer.toLowerCase();
    expect(lower, 'answer must mention baseline').toMatch(/baseline/);
    expect(lower, 'answer must mention candidate comparison').toMatch(/candidate/);
    expect(lower, 'answer must mention fixed gates').toMatch(/gate|harness|metric/);
  }, E2E_TIMEOUT_MS);

  e2eIt('includes bounded mutation and drop/revert rules', async () => {
    const { answer } = await getOptimizationFixture();
    const lower = answer.toLowerCase();
    expect(lower, 'answer must mention allowlisted forecast files').toMatch(/allowlist|approved|editable/);
    expect(lower, 'answer must mention dropping or reverting failed candidates').toMatch(/drop|revert|discard/);
  }, E2E_TIMEOUT_MS);

  e2eIt('points experiment records at .cramer-short/experiments', async () => {
    const { answer } = await getOptimizationFixture();
    expect(answer).toMatch(/\.cramer-short\/experiments/);
  }, E2E_TIMEOUT_MS);

  e2eIt('does not mutate files in dry-run mode', async () => {
    const { tools } = await getOptimizationFixture();
    expect(
      tools.some((t) => t === 'write_file' || t === 'edit_file'),
      `dry-run skill invocation must not write or edit files. Tools: [${tools.join(', ')}]`,
    ).toBe(false);
  }, E2E_TIMEOUT_MS);

  e2eIt('does not auto-enter forecast-lab for ordinary BTC forecast queries', async () => {
    const { result } = await getOrdinaryForecastFixture();
    expect(findSkillCall(result, 'forecast-lab')).toBeUndefined();
  }, E2E_TIMEOUT_MS);

  e2eIt('explains approval-required promotion, live activation, and reset options', async () => {
    const { answer } = await getLifecycleFixture();
    const lower = answer.toLowerCase();
    expect(lower, 'answer must mention explicit approval before promotion').toMatch(/approval|required|approve/);
    expect(lower, 'answer must mention the parameters becoming live for ordinary forecasts').toMatch(/live|ordinary forecast|normal forecast/);
    expect(lower, 'answer must mention reset to shipped defaults').toMatch(/shipped defaults|defaults/);
    expect(lower, 'answer must mention reset to last known-good baseline').toMatch(/last known good|known-good|previous activated/);
  }, E2E_TIMEOUT_MS);

  e2eIt('does not mutate files while explaining the lifecycle', async () => {
    const { tools } = await getLifecycleFixture();
    expect(
      tools.some((t) => t === 'write_file' || t === 'edit_file'),
      `lifecycle explanation must not write or edit files. Tools: [${tools.join(', ')}]`,
    ).toBe(false);
  }, E2E_TIMEOUT_MS);

  e2eIt('explains exhausted-lineage next actions without mutating files', async () => {
    const { answer, tools } = await getExhaustionFixture();
    const lower = answer.toLowerCase();
    expect(lower, 'answer must mention the exhausted mutator state').toMatch(/no shipped|exhausted.*shipped.*mutator|lineage|applicable|can be applied/);
    expect(lower, 'answer must mention one of the supported next actions').toMatch(
      /keep the current best candidate|add a new shipped structured mutator|reset|catalog be extended|catalog update|extend the catalog|catalog extension|different profile|request a new mutator|human review|bounded plan|stop/,
    );
    expect(
      tools.some((t) => t === 'write_file' || t === 'edit_file'),
      `exhaustion guidance must not write or edit files. Tools: [${tools.join(', ')}]`,
    ).toBe(false);
  }, E2E_TIMEOUT_MS);

  e2eIt('explains how to force a specific shipped mutator', async () => {
    const { answer, result, tools } = await getMutatorGuidanceFixture();
    const lower = answer.toLowerCase();
    expect(Boolean(findSkillCall(result, 'forecast-lab'))).toBe(true);
    expect(answer).toMatch(/--mutation structured[\s\\]+--mutator/);
    expect(lower, 'answer must mention the btc ultra-short-horizon profile').toMatch(/btc-markov-ultra-short-horizon/);
    expect(
      tools.some((t) => t === 'write_file' || t === 'edit_file'),
      `mutator guidance must not write or edit files. Tools: [${tools.join(', ')}]`,
    ).toBe(false);
  }, E2E_TIMEOUT_MS);

  e2eIt('passes explicit mutator ids through routed agent-mode improvement prompts', async () => {
    const { answer, result, tools } = await getMutatorExecutionFixture();
    const runCall = findToolCall(result, 'forecast_lab_run');
    expect(Boolean(findSkillCall(result, 'forecast-lab'))).toBe(true);
    expect(tools).toContain('forecast_lab_run');
    expect(runCall?.args?.mutator).toBe('markov-faster-decay-reaction');
    expect(answer.toLowerCase()).toContain('markov-faster-decay-reaction');
  }, E2E_TIMEOUT_MS);

  e2eIt('routes explicit approval commands through forecast_lab_run using seeded history', async () => {
    const { answer, result, tools } = await getApprovalCommandFixture();
    const lower = answer.toLowerCase();
    expect(tools).toContain('forecast_lab_run');
    expect(findSkillCall(result, 'forecast-lab')).toBeUndefined();
    expect(lower, 'answer must describe the missing approval source instead of mutating files').toMatch(
      /promotion source|approval-required forecast-lab promotion source|was not found/,
    );
    expect(
      tools.some((t) => t === 'write_file' || t === 'edit_file'),
      `approval routing must not write or edit files. Tools: [${tools.join(', ')}]`,
    ).toBe(false);
  }, E2E_TIMEOUT_MS);

  e2eIt('routes current-best vs shipped-baseline questions through forecast_lab_run without filesystem probing', async () => {
    const { answer, result, tools } = await getComparisonFixture();
    const lower = answer.toLowerCase();
    expect(tools).toContain('forecast_lab_run');
    expect(findSkillCall(result, 'forecast-lab')).toBeUndefined();
    expect(tools).not.toContain('read_file');
    expect(tools).not.toContain('web_fetch');
    expect(tools).not.toContain('browser');
    expect(lower).toMatch(/shipped baseline|current best|not live yet|already live/);
  }, E2E_TIMEOUT_MS);

  e2eIt('routes forecast-lab results prompts through forecast_lab_run instead of guided improvement', async () => {
    const { answer, tools } = await getResultsQueryFixture();
    const lower = answer.toLowerCase();
    expect(tools).toContain('forecast_lab_run');
    expect(tools).not.toContain('read_file');
    expect(tools).not.toContain('web_fetch');
    expect(tools).not.toContain('browser');
    expect(lower).toMatch(/shipped baseline|current best|approve forecast-lab promotion|already live|not live yet/);
  }, E2E_TIMEOUT_MS);

  e2eIt('routes keep-the-current-best follow-ups through forecast_lab_run without edit attempts', async () => {
    const { answer, tools } = await getKeepCurrentBestFixture();
    const lower = answer.toLowerCase();
    expect(tools).toContain('forecast_lab_run');
    expect(tools).not.toContain('edit_file');
    expect(tools).not.toContain('write_file');
    expect(tools).not.toContain('read_file');
    expect(lower).toMatch(/current best|approve forecast-lab promotion|already live|not live yet/);
  }, E2E_TIMEOUT_MS);

  e2eIt('routes catalog-extension prompts through forecast_lab_run without filesystem or browser probing', async () => {
    const { answer, tools } = await getCatalogExtensionFixture();
    const lower = answer.toLowerCase();
    expect(tools).toContain('forecast_lab_run');
    expect(tools).not.toContain('skill');
    expect(tools).not.toContain('sequential_thinking');
    expect(tools).not.toContain('read_file');
    expect(tools).not.toContain('web_fetch');
    expect(tools).not.toContain('browser');
    expect(lower).toMatch(/bounded code-change plan|did not inspect experiment artifacts|catalog-extension plan|rerun the lineage/);
  }, E2E_TIMEOUT_MS);

  e2eIt('routes detailed mutator implementation briefs through forecast_lab_run without repo or web probing', async () => {
    const { answer, tools } = await getCatalogExtensionImplementationFixture();
    const lower = answer.toLowerCase();
    expect(tools).toContain('forecast_lab_run');
    expect(tools).not.toContain('skill');
    expect(tools).not.toContain('sequential_thinking');
    expect(tools).not.toContain('read_file');
    expect(tools).not.toContain('web_fetch');
    expect(tools).not.toContain('browser');
    expect(lower).toMatch(/markov-entropy-adaptive-anchor-weighting|catalog files to open directly next|requested parameter deltas/);
  }, E2E_TIMEOUT_MS);

  e2eIt('routes mutator-vs-active comparison prompts through forecast_lab_run without filesystem probing', async () => {
    const { answer, tools } = await getMutatorVsActiveFixture();
    const lower = answer.toLowerCase();
    expect(tools).toContain('forecast_lab_run');
    expect(tools).not.toContain('read_file');
    expect(tools).not.toContain('web_fetch');
    expect(tools).not.toContain('browser');
    expect(tools).not.toContain('skill');
    expect(tools).not.toContain('sequential_thinking');
    expect(lower).toMatch(/active|mutation|not found|accuracy|dir acc|compare/);
  }, E2E_TIMEOUT_MS);

  e2eIt('routes history-based live-vs-new-mutator prompts through forecast_lab_run without filesystem probing', async () => {
    const { answer, tools } = await getHistoryMutatorVsActiveFixture();
    const lower = answer.toLowerCase();
    expect(tools).toContain('forecast_lab_run');
    expect(tools).not.toContain('read_file');
    expect(tools).not.toContain('web_fetch');
    expect(tools).not.toContain('browser');
    expect(tools).not.toContain('skill');
    expect(tools).not.toContain('sequential_thinking');
    expect(lower).toMatch(/active|mutation|not found|accuracy|dir acc|compare/);
  }, E2E_TIMEOUT_MS);

  e2eIt('routes implement-and-run requests for non-shipped mutators through catalog-extension guidance without repo probing', async () => {
    const { answer, tools } = await getImplementNewMutatorFixture();
    const lower = answer.toLowerCase();
    expect(tools).toContain('forecast_lab_run');
    expect(tools).not.toContain('skill');
    expect(tools).not.toContain('sequential_thinking');
    expect(tools).not.toContain('read_file');
    expect(tools).not.toContain('web_fetch');
    expect(tools).not.toContain('browser');
    expect(tools).not.toContain('edit_file');
    expect(tools).not.toContain('write_file');
    expect(lower).toMatch(/bounded code-change plan|requested mutator id|did not inspect experiment artifacts|catalog-extension plan/);
  }, E2E_TIMEOUT_MS);

  e2eIt('routes mutator-list prompts through forecast_lab_run without filesystem or web probing', async () => {
    const { answer, tools } = await getListMutatorsFixture();
    const lower = answer.toLowerCase();
    expect(tools).toContain('forecast_lab_run');
    expect(tools).not.toContain('skill');
    expect(tools).not.toContain('sequential_thinking');
    expect(tools).not.toContain('read_file');
    expect(tools).not.toContain('web_fetch');
    expect(tools).not.toContain('browser');
    expect(lower).toMatch(/shipped mutator ids|mutator catalog summary|markov-shorter-reactive-window|markov-faster-decay-reaction/);
  }, E2E_TIMEOUT_MS);
});
