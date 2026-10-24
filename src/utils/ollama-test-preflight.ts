/**
 * Shared preflight for live-Ollama integration tests.
 *
 * Mirrors the guarded-skip pattern in `e2e-helpers.ts`: probe the configured
 * daemon once, then let each test skip with an explicit reason when the
 * capability it needs is unavailable. Keeps `bun run test:integration` green
 * without a daemon (skips instead of failing) while keeping every assertion
 * strict when the daemon IS reachable.
 */

import { getEnvOrDefault } from './env.js';
import { getOllamaModels } from './ollama.js';

/** Short enough to keep a no-daemon run fast, long enough for a local server. */
const OLLAMA_PREFLIGHT_TIMEOUT_MS = 2_000;

export interface OllamaTestPreflight {
  /** The configured daemon answered /api/tags with a 2xx response. */
  reachable: boolean;
  /** The live model list (empty when unreachable). */
  models: string[];
  /** Why the daemon is unusable, or null when reachable. */
  reason: string | null;
}

let preflightPromise: Promise<OllamaTestPreflight> | null = null;

async function probeReachability(baseUrl: string): Promise<string | null> {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), OLLAMA_PREFLIGHT_TIMEOUT_MS);
  try {
    const response = await fetch(`${baseUrl}/api/tags`, { signal: controller.signal });
    return response.ok
      ? null
      : `Ollama health check at ${baseUrl}/api/tags returned HTTP ${response.status}`;
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    return `Ollama is unreachable at ${baseUrl}: ${message}`;
  } finally {
    clearTimeout(timer);
  }
}

/** Memoized so each test file pays the probe once, in its `beforeAll`. */
export function getOllamaTestPreflight(): Promise<OllamaTestPreflight> {
  if (!preflightPromise) {
    preflightPromise = (async () => {
      const baseUrl = getEnvOrDefault('OLLAMA_BASE_URL', 'http://127.0.0.1:11434');
      const reason = await probeReachability(baseUrl);
      return { reachable: reason === null, models: await getOllamaModels(), reason };
    })();
  }
  return preflightPromise;
}

/**
 * Prints the explicit skip reason and returns true when the configured Ollama
 * daemon is unreachable, so the caller should skip its assertions.
 */
export function skipWhenOllamaUnreachable(preflight: OllamaTestPreflight, label: string): boolean {
  if (preflight.reachable) {
    return false;
  }
  console.warn(`Skipping "${label}": ${preflight.reason ?? 'Ollama preflight failed'}`);
  return true;
}

/**
 * Prints the explicit skip reason and returns true when the live model list
 * carries no ':cloud' model, so cloud-only assertions must be skipped. Only a
 * signed-in Ollama account can serve ':cloud' tags.
 */
export function skipWhenNoCloudModel(preflight: OllamaTestPreflight, label: string): boolean {
  if (preflight.models.some((model) => model.endsWith(':cloud'))) {
    return false;
  }
  console.warn(
    `Skipping "${label}": no ':cloud' model available (requires a signed-in Ollama account)`,
  );
  return true;
}
