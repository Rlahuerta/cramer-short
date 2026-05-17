/**
 * Shared test-tier guards for Cramer-Short's three-tier test convention:
 *
 *   *.test.ts             — unit (pure logic, always run)
 *   *.integration.test.ts — real public APIs, no LLM
 *   *.e2e.test.ts         — full agent + Ollama LLM
 *
 * Local usage:
 *   bun test                  — unit tests only (fast)
 *   bun run test:integration  — integration tests (real APIs, no LLM)
 *   bun run test:e2e          — E2E tests isolated (real Ollama, sets RUN_E2E=1)
 *   bun run test:all          — all three in sequence (no contamination)
 *
 * Integration and E2E tests both require explicit opt-in so `bun test`
 * stays unit-only and live suites run only through the isolated scripts.
 *
 * CI opt-in:
 *   RUN_INTEGRATION=1  — enable integration tests in CI
 *   RUN_E2E=1          — enable E2E tests in CI (run via test:e2e for isolation)
 *
 * Local opt-out:
 *   SKIP_INTEGRATION=1 — disable integration tests locally
 */
import { it } from 'bun:test';
import { getE2EDynamicSkipReason, getE2EPreflightStatus, markE2ESkippedFromError } from './e2e-helpers.js';
import { getBooleanEnv } from './env.js';

export const RUN_INTEGRATION =
  getBooleanEnv('SKIP_INTEGRATION') ? false : getBooleanEnv('RUN_INTEGRATION');

// E2E tests always require explicit opt-in to prevent worker contamination:
// unit test files mock ../model/llm.js which would make E2E use a fake LLM.
// Use `bun run test:e2e` which sets RUN_E2E=1 and runs only e2e files.
export const RUN_E2E = getBooleanEnv('RUN_E2E');

const initialE2EPreflight = RUN_E2E ? await getE2EPreflightStatus() : null;
if (initialE2EPreflight && !initialE2EPreflight.available) {
  console.warn(
    `Skipping live E2E for ${initialE2EPreflight.model}: ${initialE2EPreflight.reason ?? 'preflight failed'}`,
  );
}

/** Use instead of `it` for tests that hit real external APIs (no LLM). */
export const integrationIt: typeof it = RUN_INTEGRATION ? it : it.skip;

const guardedE2EIt: typeof it = ((label: string, fn: () => void | Promise<unknown>, options?: Parameters<typeof it>[2]) => {
  it(label, async () => {
    const skipReason = getE2EDynamicSkipReason();
    if (skipReason) {
      console.warn(skipReason);
      return;
    }

    try {
      await fn();
    } catch (error) {
      if (markE2ESkippedFromError(error)) {
        return;
      }
      throw error;
    }
  }, options);
}) as unknown as typeof it;

/** Use instead of `it` for tests that run the full Cramer-Short agent against Ollama. */
export const e2eIt: typeof it =
  RUN_E2E && (initialE2EPreflight?.available ?? false) ? guardedE2EIt : it.skip;
