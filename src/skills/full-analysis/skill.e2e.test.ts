/**
 * E2E tests — full-analysis skill with a real model.
 *
 * Run with:  bun run test:e2e
 * Skipped in normal `bun test` / CI runs.
 *
 * full-analysis is only an orchestrator: its value comes from running the four sub-skills it names.
 * A run that answers from the orchestrator alone invents its own probability weights instead of
 * using the probability_assessment workflow, so these tests check that each sub-skill is invoked.
 */
import { describe, expect, beforeAll } from 'bun:test';
import { e2eIt, RUN_E2E } from '@/utils/test-guards.js';
import { markE2ESkippedFromError, runAgentE2EWithTimeoutRetry, E2E_TIMEOUT_MS } from '@/utils/e2e-helpers.js';
import type { E2EResult } from '@/utils/e2e-helpers.js';

const FULL_ANALYSIS_QUERY = '--deep Full analysis of MSFT: is it worth buying?';
const FULL_ANALYSIS_TIMEOUT_MS = Math.max(E2E_TIMEOUT_MS, 900_000);
const SUB_SKILLS = ['dcf-valuation', 'peer-comparison', 'short-thesis', 'probability_assessment'];

let result: E2EResult;
let skillsInvoked: string[];

describe('full-analysis skill E2E', () => {
  beforeAll(async () => {
    if (!RUN_E2E) return;
    try {
      result = await runAgentE2EWithTimeoutRetry(FULL_ANALYSIS_QUERY, { timeoutMs: FULL_ANALYSIS_TIMEOUT_MS });
      skillsInvoked = result.events
        .filter((e) => e.type === 'tool_start' && e.tool === 'skill')
        .map((e) => String((e as { args: Record<string, unknown> }).args.skill));
    } catch (error) {
      if (markE2ESkippedFromError(error)) return;
      throw error;
    }
  }, FULL_ANALYSIS_TIMEOUT_MS);

  e2eIt('invokes the full-analysis skill', () => {
    expect(skillsInvoked, `skills invoked: [${skillsInvoked.join(', ')}]`).toContain('full-analysis');
  });

  for (const subSkill of SUB_SKILLS) {
    e2eIt(`invokes the ${subSkill} sub-skill`, () => {
      expect(skillsInvoked, `skills invoked: [${skillsInvoked.join(', ')}]`).toContain(subSkill);
    });
  }

  e2eIt('answers about MSFT with a weighted probability', () => {
    expect(result.answer.toLowerCase()).toMatch(/msft|microsoft/);
    expect(result.answer).toMatch(/\d+(?:\.\d+)?\s*%/);
  });
});
