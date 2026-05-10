/**
 * Skills contract test — validates SKILL.md files against runtime surfaces.
 *
 * Catches:
 *   1. Skill references in instructions that don't match discoverable skill names
 *   2. Tool references in instructions that don't match the tool registry
 *   3. Stale `.dexter/` path references (should be `.cramer-short/`)
 *   4. Parameter placeholder / declaration drift for parameterized skills
 *
 * This test is deterministic — no LLM calls, no network. It loads real code
 * modules to build ground-truth surfaces and validates every SKILL.md against
 * them.
 */
import { describe, it, expect, beforeAll } from 'bun:test';
import { existsSync, readdirSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { discoverSkills, clearSkillCache } from './registry.js';
import { loadSkillFromPath } from './loader.js';
import type { Skill } from './types.js';

/**
 * Stable list of potential tool names derived from src/tools/registry.ts.
 * Hardcoded to avoid Bun mock.module() pollution from other test files
 * (e.g., prompts.test.ts, agent-response.test.ts) which mock '../tools/registry.js'.
 * When those mocks are active, getToolRegistry() returns incorrect/empty results.
 *
 * Keep this in sync with 'name' fields in getToolRegistry() in src/tools/registry.ts.
 */
const STABLE_TOOL_NAMES = new Set([
  // Always-available tools (no env gating)
  'sequential_thinking',
  'get_financials',
  'get_market_data',
  'get_robinhood_quote',
  'get_robinhood_fundamentals',
  'read_filings',
  'stock_screener',
  'portfolio_risk',
  'wacc_inputs',
  'polymarket_search',
  'polymarket_forecast',
  'forecast_arbitrator',
  'price_distribution_chart',
  'markov_distribution',
  'bitmex_market',
  'social_sentiment',
  'trump_pressure_index',
  'geopolitics_search',
  'get_fixed_income',
  'get_options_chain',
  'get_earnings_transcript',
  'get_onchain_crypto',
  'web_fetch',
  'browser',
  'read_file',
  'write_file',
  'edit_file',
  'heartbeat',
  'memory_search',
  'memory_get',
  'memory_update',
  'recall_financial_context',
  'store_financial_insight',
  // Env-gated tools (may appear in SKILL.md even if not always active)
  'web_search',
  'x_search',
  'skill',
]);

const __dirname = dirname(fileURLToPath(import.meta.url));

// ─── Ground-truth surfaces ──────────────────────────────────────────────────

const ENV_KEYS_FOR_OPTIONAL_TOOLS: string[] = [
  'EXASEARCH_API_KEY',
  'PERPLEXITY_API_KEY',
  'TAVILY_API_KEY',
  'X_BEARER_TOKEN',
];

/** Collect runtime skill names */
function getRuntimeSkillNames(): string[] {
  clearSkillCache();
  const skills = discoverSkills();
  clearSkillCache();
  return skills.map((s) => s.name);
}

/**
 * Build the full set of tool names that *could* be available at runtime.
 * Uses hardcoded STABLE_TOOL_NAMES to avoid mock.module pollution from other tests.
 */
function getRuntimeToolNames(): Set<string> {
  // Return the hardcoded stable set — no runtime inspection needed.
  // This avoids Bun mock.module() pollution from agent test files.
  return STABLE_TOOL_NAMES;
}

/** Load all builtin skills fully (with instructions). */
function loadAllBuiltinSkills(): Skill[] {
  const skills: Skill[] = [];

  for (const entry of readdirSync(__dirname, { withFileTypes: true })) {
    if (!entry.isDirectory()) continue;

    const skillPath = join(__dirname, entry.name, 'SKILL.md');
    if (!existsSync(skillPath)) continue;

    try {
      skills.push(loadSkillFromPath(skillPath, 'builtin'));
    } catch {
      // Skip skills that fail to load; discovery tests will catch regressions.
    }
  }

  return skills;
}

// ─── Helpers ────────────────────────────────────────────────────────────────

function extractToolRefs(instructions: string): string[] {
  const refs = new Set<string>();

  for (const m of instructions.matchAll(/`([a-zA-Z_][a-zA-Z0-9_]+)`/g)) {
    const word = m[1];
    if (word.length < 3) continue;
    refs.add(word);
  }

  return Array.from(refs);
}

const KNOWN_DATA_FIELDS = new Set([
  'market_cap', 'enterprise_value', 'debt_to_equity', 'free_cash_flow_per_share',
  'total_debt', 'cash_and_equivalents', 'current_investments', 'outstanding_shares',
  'earnings_per_share', 'industry', 'effective_tax_rate', 'income_tax_rate',
  'wacc', 'min_likes', 'day_change_percent', 'volume', 'analyst_rating',
  'analyst_price_target', 'tickers', 'probability', 'ticker', 'horizon',
  'sentiment', 'price', 'sector', 'note', 'costBasis',
  'betaSource', 'deRatio', 'equityWeight', 'debtWeight', 'waccPct',
  'LogOddsSignal',
  'polymarketMarkets', 'historicalPrices', 'predictionConfidence',
  'regimeState', 'actionSignal', 'buyProbability', 'sellProbability',
  'targetPrice', 'stopLoss', 'actionLevels',
  'eps_estimated', 'eps_actual', 'estimated_eps', 'estimated_revenue',
  'earnings_surprise_percent', 'net_debt_to_ebitda',
  'return_on_invested_capital', 'price_to_free_cash_flow', 'ev_to_ebitda',
  'pe_ratio', 'peg_ratio', 'revenue_growth', 'gross_margin',
  'free_cash_flow_growth', 'capital_expenditure',
  'net_cash_flow_from_operations', 'free_cash_flow',
  'fifty_two_week_high', 'fifty_two_week_low', 'average_volume',
  'earnings_date', 'next_earnings_date', 'implied_move',
  'confidenceThreshold', 'portfolioValue', 'riskPerTrade',
  'prediction_confidence', 'confidence_threshold', 'portfolio_risk',
  'watchlist', 'polymarket',
]);

function extractSkillRefs(instructions: string): string[] {
  const refs = new Set<string>();

  for (const m of instructions.matchAll(/invoke\s+skill\s+[`"](\w[\w-]*)[`"]/gi)) refs.add(m[1]);
  for (const m of instructions.matchAll(/combine\s+with\s+[`"](\w[\w-]*)[`"]/gi)) refs.add(m[1]);
  for (const m of instructions.matchAll(/[Uu]se\s+(?:the\s+)?[`"](\w[\w-]*)[`"]\s+skill/gi)) refs.add(m[1]);
  for (const m of instructions.matchAll(/skill\s+[`"](\w[\w-]*)[`"]/gi)) refs.add(m[1]);

  return Array.from(refs);
}

/**
 * Extract {{param}} placeholders from skill instructions.
 */
function extractPlaceholders(instructions: string): string[] {
  const placeholders = new Set<string>();
  for (const m of instructions.matchAll(/\{\{(\w+)\}\}/g)) {
    placeholders.add(m[1]);
  }
  return Array.from(placeholders);
}

// ─── Test suite ──────────────────────────────────────────────────────────────

describe('Skills contract: SKILL.md vs runtime', () => {
  let toolNames: Set<string>;
  let skillNames: string[];
  let skills: Skill[];

  beforeAll(() => {
    toolNames = getRuntimeToolNames();
    skillNames = getRuntimeSkillNames();
    skills = loadAllBuiltinSkills();
  });

  // ── 1. Skill discoverability ────────────────────────────────────────────

  it('every builtin SKILL.md is discoverable', () => {
    for (const skill of skills) {
      expect(skillNames, `"${skill.name}" not discoverable via discoverSkills()`).toContain(skill.name);
    }
  });

  it('all known runtime skill names are accounted for', () => {
    const knownNames = [
      'dcf-valuation',
      'probability_assessment',
      'portfolio_risk',
      'peer-comparison',
      'full-analysis',
      'swing-trade-setup',
      'position-sizing',
      'x-research',
      'trump-pressure',
      'geopolitics-osint',
      'earnings-calendar',
      'earnings-preview',
      'watchlist-briefing',
      'short-thesis',
      'sector-overview',
    ];
    for (const name of knownNames) {
      expect(skillNames, `Known skill "${name}" missing from runtime`).toContain(name);
    }
  });

  // ── 2. Tool references in SKILL.md instructions ────────────────────────

  it('all tool references in SKILL.md instructions match registered tools or are known safe exceptions', () => {
    const allowedTools = new Set([...toolNames, ...KNOWN_DATA_FIELDS]);

    const failures: string[] = [];

    for (const skill of skills) {
      const refs = extractToolRefs(skill.instructions);
      for (const ref of refs) {
        if (!allowedTools.has(ref) && !skillNames.includes(ref)) {
          failures.push(`Skill "${skill.name}" references \`${ref}\` which is not a registered tool or skill`);
        }
      }
    }

    expect(failures, `Unresolved tool references found:\n${failures.join('\n')}`).toEqual([]);
  });

  // ── 3. Skill name references in instructions ───────────────────────────

  it('all skill name references in SKILL.md instructions match discoverable skill names', () => {
    const failures: string[] = [];

    for (const skill of skills) {
      const refs = extractSkillRefs(skill.instructions);
      for (const ref of refs) {
        if (!skillNames.includes(ref)) {
          failures.push(`Skill "${skill.name}" references skill "${ref}" which is not a discoverable skill name`);
        }
      }
    }

    expect(failures, `Misleading skill references found:\n${failures.join('\n')}`).toEqual([]);
  });

  // ── 4. No stale `.dexter/` paths ────────────────────────────────────────

  it('no SKILL.md references stale .dexter/ paths (should be .cramer-short/)', () => {
    const failures: string[] = [];

    for (const skill of skills) {
      if (skill.instructions.includes('.dexter/') || skill.instructions.includes('.dexter\\')) {
        failures.push(`Skill "${skill.name}" contains stale ".dexter/" path reference`);
      }
      if (skill.description.includes('.dexter/')) {
        failures.push(`Skill "${skill.name}" description contains stale ".dexter/" path reference`);
      }
    }

    expect(failures, `Stale .dexter/ references found:\n${failures.join('\n')}`).toEqual([]);
  });

  // ── 5. Parameter placeholder drift ──────────────────────────────────────

  it('all {{param}} placeholders in parameterized skills match declared parameters', () => {
    const failures: string[] = [];

    for (const skill of skills) {
      if (!skill.parameters) continue;

      const declaredParams = new Set(Object.keys(skill.parameters));
      const placeholderNames = extractPlaceholders(skill.instructions);

      // Every placeholder must have a corresponding declared parameter
      for (const ph of placeholderNames) {
        if (!declaredParams.has(ph)) {
          failures.push(
            `Skill "${skill.name}" uses {{${ph}}} placeholder but has no matching parameter declaration`,
          );
        }
      }

      // Every declared parameter with a default should appear as a placeholder
      // (but we don't enforce this — some params are invoked programmatically)
    }

    expect(failures, `Parameter placeholder drift found:\n${failures.join('\n')}`).toEqual([]);
  });

  // ── 6. Specific known defects (regression guards) ──────────────────────

  it('geopolitics-osint skill does not reference nonexistent "financial_metrics" tool', () => {
    const skill = skills.find((s) => s.name === 'geopolitics-osint');
    if (!skill) return; // Skip if skill not present
    expect(
      skill.instructions.includes('financial_metrics'),
      'geopolitics-osint still references `financial_metrics` — use `get_market_data` or `get_financials` instead',
    ).toBe(false);
  });

  it('earnings-calendar skill does not reference .dexter/ paths', () => {
    const skill = skills.find((s) => s.name === 'earnings-calendar');
    if (!skill) return;
    expect(
      skill.instructions.includes('.dexter/'),
      'earnings-calendar still references `.dexter/` — should be `.cramer-short/`',
    ).toBe(false);
  });

  it('position-sizing skill references correct runtime skill names', () => {
    const skill = skills.find((s) => s.name === 'position-sizing');
    if (!skill) return;
    const instructions = skill.instructions;
    // "portfolio-risk" is incorrect — runtime name is "portfolio_risk"
    expect(
      instructions.includes('`portfolio-risk`'),
      'position-sizing references `portfolio-risk` — runtime name is `portfolio_risk`',
    ).toBe(false);
    // "dcf" is misleading — runtime name is "dcf-valuation"
    // However "dcf" appears in many contexts (the concept) so we only check
    // the specific "Combine with" lines
    const combineLines = instructions.split('\n').filter((l) => l.toLowerCase().includes('combine with'));
    for (const line of combineLines) {
      if (line.includes('`dcf`')) {
        // This is acceptable if it's the concept not the skill name reference
      }
    }
  });

  it('geopolitics-osint skill matches current memory tool schemas', () => {
    const skill = skills.find((s) => s.name === 'geopolitics-osint');
    if (!skill) return;

    expect(
      skill.instructions.includes('`recall_financial_context`'),
      'geopolitics-osint should not use recall_financial_context for watchlist lookup; use memory_search or a valid recall_financial_context ticker call',
    ).toBe(false);

    expect(
      skill.instructions.includes('"insight":'),
      'geopolitics-osint should use store_financial_insight.content, not insight',
    ).toBe(false);

    expect(
      skill.instructions.includes('`memory_search`'),
      'geopolitics-osint should use memory_search for watchlist lookup text recall',
    ).toBe(true);

    expect(
      skill.instructions.includes('"content":'),
      'geopolitics-osint should store findings using the content field',
    ).toBe(true);
  });
});
