/**
 * Ollama API utilities
 */

import { getEnvOrDefault } from './env.js';

interface OllamaModel {
  name: string;
  modified_at: string;
  size: number;
}

interface OllamaTagsResponse {
  models: OllamaModel[];
}

/**
 * Fetches locally downloaded models from the Ollama API
 */
export async function getOllamaModels(): Promise<string[]> {
  const baseUrl = getEnvOrDefault('OLLAMA_BASE_URL', 'http://127.0.0.1:11434');
  
  try {
    const response = await fetch(`${baseUrl}/api/tags`, {
      signal: AbortSignal.timeout(5_000),
    });

    if (!response.ok) {
      return [];
    }

    const data = (await response.json()) as OllamaTagsResponse;
    return (data?.models ?? [])
      .map((m) => m?.name)
      .filter((n): n is string => typeof n === 'string');
  } catch {
    // Ollama not running or unreachable
    return [];
  }
}
