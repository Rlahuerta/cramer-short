/**
 * Shared test-tier guards for Dexter's three-tier test convention:
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
 * Integration tests run automatically when not in CI.
 * E2E tests always require explicit RUN_E2E=1 (set by test:e2e script) because
 * they must run in an isolated process to avoid unit-test mock contamination.
 *
 * CI opt-in:
 *   RUN_INTEGRATION=1  — enable integration tests in CI
 *   RUN_E2E=1          — enable E2E tests in CI (run via test:e2e for isolation)
 *
 * Local opt-out:
 *   SKIP_INTEGRATION=1 — disable integration tests locally
 */
import { it } from 'bun:test';

const IS_CI = process.env.CI === 'true' || process.env.CI === '1';

export const RUN_INTEGRATION =
  process.env.SKIP_INTEGRATION === '1' ? false
  : process.env.RUN_INTEGRATION === '1' ? true
  : !IS_CI;

// E2E tests always require explicit opt-in to prevent worker contamination:
// unit test files mock ../model/llm.js which would make E2E use a fake LLM.
// Use `bun run test:e2e` which sets RUN_E2E=1 and runs only e2e files.
export const RUN_E2E = process.env.RUN_E2E === '1';

/** Use instead of `it` for tests that hit real external APIs (no LLM). */
export const integrationIt: typeof it = RUN_INTEGRATION ? it : it.skip;

/** Use instead of `it` for tests that run the full Dexter agent against Ollama. */
export const e2eIt: typeof it = RUN_E2E ? it : it.skip;
