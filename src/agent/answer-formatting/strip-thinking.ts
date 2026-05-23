/**
 * Remove <think>...</think> blocks that Ollama thinking models sometimes embed
 * directly in response text rather than separating into reasoning_content.
 * Also handles orphan </think> tags (e.g. the model output was: <think>…</think>\nAnswer).
 */
export function stripThinkingTags(text: string): string {
  return text
    .replace(/<think>[\s\S]*?<\/think>/gi, '') // full <think>…</think> blocks
    .replace(/^[\s\S]*?<\/think>\s*/i, '')      // orphan </think>: strip everything up to and including it
    .trim();
}
