import { callLlm } from '../model/llm.js';
import { MemoryManager } from '../memory/index.js';

/**
 * Writes a brief LLM summary of the current session to today's daily memory file.
 * This ensures Dream has meaningful input to consolidate even when the context-
 * overflow threshold is never reached (the usual case for shorter sessions).
 * Skips silently if there are no exchanges, the model call fails, or the output
 * is trivially short.
 */
export async function writeSessionDailySummary(
  history: { query: string; answer: string }[],
  model: string,
): Promise<void> {
  const exchanges = history.filter((h) => h.query && h.answer);
  if (exchanges.length === 0) return;

  const transcript = exchanges
    .map((h, i) => `[${i + 1}] User: ${h.query.slice(0, 400)}\nAssistant: ${h.answer.slice(0, 600)}`)
    .join('\n\n');

  const prompt = `You are a financial research assistant. A user just finished a Cramer-Short session.
Summarize the key facts, decisions, and insights from this session in concise markdown bullet points.

Rules:
- Only include durable, reusable facts (investment theses, risk flags, model assumptions, ticker notes)
- Date-stamp any financial figures (e.g. "AMD P/E ~45x as of 2026-03")
- Skip trivial exchanges (greetings, command help, failed lookups)
- Output 3–10 bullet points max. If nothing worth remembering occurred, output only: NOTHING_TO_STORE

Session transcript:
${transcript}`;

  try {
    // thinkOverride:false — session summaries need concise plain-text output, not thinking tokens
    // Use a 3-minute timeout — session summary is non-critical but can be slow on cloud models
    const result = await callLlm(prompt, { model, thinkOverride: false, timeoutMs: 180_000 });
    const text = typeof result.response === 'string' ? result.response.trim() : '';
    if (!text || text === 'NOTHING_TO_STORE' || text.length < 40) return;
    const today = new Date().toISOString().slice(0, 10);
    const manager = await MemoryManager.get();
    await manager.appendDailyMemory(`## Session summary — ${today}\n${text}`);
  } catch {
    // Non-fatal — never block exit on memory errors.
  }
}
