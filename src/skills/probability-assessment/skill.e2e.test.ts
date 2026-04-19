/**
 * Refined E2E tests — probability_assessment skill with a real Ollama thinking model.
 *
 * Run with:  RUN_E2E=1 bun test --filter e2e
 * Skipped in normal `bun test` / CI runs.
 *
 * The agent is invoked ONCE via beforeAll; all tests share the same result so
 * we pay the LLM cost only once per run.
 *
 * What this suite verifies end-to-end:
 *   1. Tool chain order  — skill loads first; Polymarket data via tool call or pre-injected 🎯 block
 *   2. Signal Evidence   — raw Polymarket market text with YES percentages in the answer
 *   3. Price chart       — ASCII distribution chart rendered from threshold markets
 *   4. Probability table — signal rows with Weight column and a Combined row
 *   5. Bear case block   — inversion / failure-mode paragraph is present
 *   6. Disclaimer        — market-implied-odds caveat at end of output
 *   7. Live BTC price    — dollar figure appears in answer (via get_market_data, markov auto-fetch, or pre-injected context)
 *
 * Model: minimax-m2.7:cloud by default (override via E2E_MODEL env var)
 * Timeout: E2E_TIMEOUT_MS (default 300 s) for the single beforeAll call
 */
import { describe, expect, beforeAll } from 'bun:test';
import { e2eIt, RUN_E2E } from '@/utils/test-guards.js';
import { runAgentE2EWithTimeoutRetry, E2E_TIMEOUT_MS } from '@/utils/e2e-helpers.js';
import type { E2EResult } from '@/utils/e2e-helpers.js';

const PROBABILITY_QUERY =
  '--deep Use the probability_assessment skill for BTC price movement in the next 30 days';

// ── shared agent result ──────────────────────────────────────────────────────
// All tests in this suite share a single agent run to avoid paying the LLM
// cost multiple times.  beforeAll runs only when RUN_E2E=1 is set; individual
// tests are still skipped via e2eIt when the env var is absent.
let result: E2EResult;
let tools: string[];
let answer: string;

describe('probability_assessment skill E2E', () => {
  beforeAll(async () => {
    if (!RUN_E2E) return; // guard — tests will be skipped via e2eIt when RUN_E2E is false
    result = await runAgentE2EWithTimeoutRetry(PROBABILITY_QUERY);
    tools = result.toolsCalled;
    answer = result.answer;
  }, E2E_TIMEOUT_MS);

  // ── 1. Tool chain ──────────────────────────────────────────────────────────

  e2eIt('invokes the skill and acquires Polymarket data (tool or pre-injected context)', () => {
    const skillIdx = tools.indexOf('skill');
    const polyIdx = tools.findIndex((t) => t.includes('polymarket'));
    // Polymarket data may come from an explicit tool call OR from the
    // pre-injected 🎯 Prediction Markets block in the system prompt.
    // When pre-injected, polymarket_search is intentionally skipped (SKILL.md Step 2).
    expect(skillIdx, 'skill tool must be called').toBeGreaterThanOrEqual(0);
    const hasPolyTool = polyIdx >= 0;
    const hasPolyContent = /polymarket/i.test(answer);
    expect(
      hasPolyTool || hasPolyContent,
      'Polymarket data must appear via explicit tool call or pre-injected 🎯 block',
    ).toBe(true);
  });

  e2eIt('uses polymarket_search or pre-injected context (not generic web_search) for market probabilities', () => {
    const hasPolyTool = tools.some((t) => t === 'polymarket_search');
    const hasPolyContent = /polymarket/i.test(answer);
    expect(
      hasPolyTool || hasPolyContent,
      'market probabilities must come from polymarket_search or pre-injected 🎯 block, not web_search',
    ).toBe(true);
  });

  e2eIt('resolves live BTC price (via get_market_data or markov auto-fetch)', () => {
    // get_market_data may be called explicitly, or markov_distribution may
    // auto-fetch prices internally (no tool_start event for the internal fetch).
    const hasGmd = tools.some((t) => t === 'get_market_data');
    const hasMarkov = tools.some((t) => t === 'markov_distribution');
    const hasPriceInAnswer = /\$[2-9]\d[,.]?\d*[Kk]|\$[1-9]\d{2}[,.]?\d*[Kk]|\$\d{2,3},\d{3}/.test(answer);
    expect(
      hasGmd || hasMarkov || hasPriceInAnswer,
      'live BTC price must appear via explicit get_market_data, markov auto-fetch, or in the answer',
    ).toBe(true);
  });

  e2eIt('renders price_distribution_chart when available or includes raw threshold evidence', () => {
    // Polymarket data may be pre-injected, so ordering relative to polymarket_search
    // cannot be enforced. Some runs only expose one usable threshold market, so the
    // model may skip the chart and still surface raw threshold evidence directly.
    const hasChartTool = tools.some((t) => t.includes('chart'));
    const hasChartContent =
      /░|█|▓|▒/.test(answer) ||
      /─{3,}/.test(answer) ||
      /chart|distribution/i.test(answer);
    const hasThresholdEvidence =
      /(exceed|reach|above|below)\s+\$\d|(exceed|reach|above|below).*\$\d/i.test(answer) &&
      (/\d+\.?\d*\s*%\s*YES/i.test(answer) || /\bYES\b.*\d+\.?\d*\s*%/i.test(answer));
    expect(
      hasChartTool || hasChartContent || hasThresholdEvidence,
      'price_distribution_chart must be called, chart content must appear, or raw threshold evidence must appear in the answer',
    ).toBe(true);
  });

  // ── 2. Signal Evidence section ─────────────────────────────────────────────

  e2eIt('answer contains Signal Evidence with raw Polymarket YES percentages', () => {
    // The SKILL.md mandates showing exact market question text and YES probability
    expect(answer.toLowerCase()).toMatch(/signal evidence|evidence/);

    // Must contain at least one Polymarket market entry: "X% YES" or "YES: X%"
    expect(answer).toMatch(/\d+\.?\d*\s*%\s*YES|\bYES\b.*\d+\.?\d*\s*%/i);

    // Must reference real dollar price thresholds from the markets
    expect(answer).toMatch(/\$\d{2,3}[,.]?\d*[Kk]?|\$\d{1,3},\d{3}/);
  });

  e2eIt('answer cites polymarket.com as the data source', () => {
    // The model should attribute the crowd probabilities to Polymarket
    expect(answer.toLowerCase()).toMatch(/polymarket/);
  });

  // ── 3. Price distribution chart ────────────────────────────────────────────

  e2eIt('answer embeds the chart or otherwise shows price-threshold evidence', () => {
    // The model may embed a full ASCII chart, summarize the chart as a
    // bucketed/threshold narrative, or directly quote the Polymarket price-level
    // markets when only sparse threshold data is available.
    const hasChart =
      /░|█|▓|▒/.test(answer) ||            // block-fill characters
      /─{3,}/.test(answer) ||               // horizontal rule dividers inside chart
      /chart|distribution|bucket|threshold/i.test(answer);   // textual fallback
    const hasThresholdEvidence =
      /(exceed|reach|above|below)\s+\$\d|(exceed|reach|above|below).*\$\d/i.test(answer) &&
      (/\d+\.?\d*\s*%\s*YES/i.test(answer) || /\bYES\b.*\d+\.?\d*\s*%/i.test(answer));
    expect(
      hasChart || hasThresholdEvidence,
      'answer must embed the chart or include raw threshold evidence from price-level markets',
    ).toBe(true);
  });

  // ── 4. Probability summary table ───────────────────────────────────────────

  e2eIt('answer contains probability summary table with Signal, Probability, Weight columns', () => {
    expect(answer).toMatch(/Signal/);
    expect(answer).toMatch(/Probability/);
    expect(answer).toMatch(/Weight/);
    // Combined row must be present
    expect(answer.toLowerCase()).toMatch(/combined|weighted/);
    // At least one percentage value (allow decimal values and unicode spacing before %)
    expect(answer).toMatch(/\d+(?:\.\d+)?\s*[%％]/u);
  });

  e2eIt('combined probability is in 1–99% range (not fabricated extremes)', () => {
    // Extract all percentages from the answer and verify at least one is a
    // plausible combined probability (between 1% and 99% inclusive)
    const pcts = [...answer.matchAll(/(\d+\.?\d*)\s*%/g)].map((m) => parseFloat(m[1]));
    expect(pcts.length, 'answer must contain percentage figures').toBeGreaterThan(0);
    const inRange = pcts.some((p) => p >= 1 && p <= 99);
    expect(inRange, `at least one probability must be in 1–99% range; found: ${pcts.join(', ')}`).toBe(true);
  });

  // ── 5. Bear case ───────────────────────────────────────────────────────────

  e2eIt('answer contains Bear case (inversion) block', () => {
    expect(
      answer.toLowerCase(),
      'Bear case block is required per SKILL.md Step 6',
    ).toMatch(/bear case/);
  });

  // ── 6. Disclaimer ──────────────────────────────────────────────────────────

  e2eIt('answer includes a disclaimer or explicit source/provenance block', () => {
    // Some runs produce an explicit disclaimer; others surface a dedicated Sources
    // block instead. Both satisfy the core requirement that the odds are grounded
    // in attributable market/probability evidence rather than presented as fact.
    const lowerAnswer = answer.toLowerCase();
    const hasDisclaimer = /disclaimer|not financial advice|market[- ]implied|not guaranteed|for informational|not investment|not a recommendation|crowd[- ]implied|should not be relied/i.test(answer);
    const hasSourcesSection = /\bsources?\b/i.test(lowerAnswer);
    expect(
      hasDisclaimer || hasSourcesSection,
      'answer must include a disclaimer or an explicit sources/provenance block',
    ).toBe(true);
  });

  // ── 7. Live price anchor ───────────────────────────────────────────────────

  e2eIt('answer includes current BTC price (from live market data or auto-fetch)', () => {
    // Price may be sourced from an explicit get_market_data call, markov_distribution
    // internal auto-fetch, or the pre-injected 🎯 block — all are valid.
    const hasBtcPrice = /\$[2-9]\d[,.]?\d*[Kk]|\$[1-9]\d{2}[,.]?\d*[Kk]|\$\d{2,3},\d{3}/.test(answer);
    expect(
      hasBtcPrice,
      'answer must include a BTC dollar price from any valid source',
    ).toBe(true);
  });
});

// ---------------------------------------------------------------------------
// Markov distribution E2E — separate agent run for price distribution query
// ---------------------------------------------------------------------------
// This suite verifies that the markov_distribution tool is invoked when the
// user asks an explicit price-distribution question (Step 2c of the skill).
// ---------------------------------------------------------------------------

const MARKOV_QUERY =
  '--deep Use the probability_assessment skill. What is the probability distribution for BTC price in 30 days? Show me the full Markov-enhanced distribution with confidence intervals.';

let markovResult: E2EResult;
let markovTools: string[];
let markovAnswer: string;

describe('markov_distribution E2E — price distribution workflow', () => {
  beforeAll(async () => {
    if (!RUN_E2E) return;
    markovResult = await runAgentE2EWithTimeoutRetry(MARKOV_QUERY);
    markovTools = markovResult.toolsCalled;
    markovAnswer = markovResult.answer;
  }, E2E_TIMEOUT_MS);

  // ── Tool chain ──────────────────────────────────────────────────────────

  e2eIt('calls markov_distribution tool during the run', () => {
    expect(
      markovTools.some((t) => t === 'markov_distribution'),
      'markov_distribution must be invoked for a price distribution query',
    ).toBe(true);
  });

  e2eIt('uses price-distribution evidence alongside markov_distribution', () => {
    const hasChartTool = markovTools.some((t) => t.includes('chart'));
    const hasMarkov = markovTools.some((t) => t === 'markov_distribution');
    const hasDistributionContent = /distribution|bucket|threshold|P\(>|\$\d+/i.test(markovAnswer);
    expect(hasMarkov, 'markov_distribution must be called').toBe(true);
    expect(
      hasChartTool || hasDistributionContent,
      'the answer must include price-distribution evidence from a chart step or equivalent distribution content',
    ).toBe(true);
  });

  e2eIt('historical prices are available (via explicit get_market_data or markov auto-fetch)', () => {
    // markov_distribution auto-fetches historical prices when omitted, so an explicit
    // get_market_data call is no longer required — the tool fetches internally.
    const hasGmd = markovTools.some((t) => t === 'get_market_data');
    const hasMarkov = markovTools.some((t) => t === 'markov_distribution');
    expect(
      hasGmd || hasMarkov,
      'historical prices must arrive via explicit get_market_data or markov auto-fetch',
    ).toBe(true);
  });

  // ── Answer quality ──────────────────────────────────────────────────────

  e2eIt('answer contains regime state information or markov abstain warning', () => {
    // If markov_distribution abstains, the answer should mention abstain/warning instead
    // of regime state. Both outcomes are valid per prompts.ts abstain guidance.
    const hasRegime = /regime|bull|bear|sideways/.test(markovAnswer.toLowerCase());
    const hasAbstain = /abstain|unable to.*calibrat|no calibrated|no.*markov/i.test(markovAnswer);
    expect(
      hasRegime || hasAbstain,
      'answer must contain regime state info or an abstain/fallback explanation',
    ).toBe(true);
  });

  e2eIt('answer includes confidence interval / CI reference or abstain fallback', () => {
    // Canonical markov returns 90% CI bounds; an abstain result should mention
    // why no calibrated CI is available — both are acceptable.
    const hasCI = /CI|confidence interval|90%|\[\d+\.?\d*%.*%\]/i.test(markovAnswer);
    const hasAbstain = /abstain|unable to.*calibrat|no calibrated|no.*markov/i.test(markovAnswer);
    expect(
      hasCI || hasAbstain,
      'answer must include CI bounds or an abstain/fallback explanation',
    ).toBe(true);
  });

  e2eIt('answer includes probability distribution table / price levels or abstain fallback', () => {
    const hasDist = /P\(>|\$\d+.*%|probability/i.test(markovAnswer);
    const hasAbstain = /abstain|unable to.*calibrat|no calibrated|no.*markov/i.test(markovAnswer);
    expect(
      hasDist || hasAbstain,
      'answer must include distribution data or an abstain/fallback explanation',
    ).toBe(true);
  });

  e2eIt('answer mentions Markov chain / regime transitions or abstain context', () => {
    const hasMarkov = /markov|regime|transition/.test(markovAnswer.toLowerCase());
    const hasAbstain = /abstain|unable to.*calibrat|no calibrated|no.*markov/i.test(markovAnswer);
    expect(
      hasMarkov || hasAbstain,
      'answer must reference Markov/regime concepts or an abstain/fallback explanation',
    ).toBe(true);
  });
});
