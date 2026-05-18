import { z } from 'zod';
import { MS_PER_DAY } from '../utils/time.js';

export const ConfigSchema = z.object({
  provider: z.string().optional(),
  modelId: z.string().optional(),
  model: z.string().optional(), // legacy
  memory: z.object({
    enabled: z.boolean().optional(),
    embeddingProvider: z.enum(['openai', 'gemini', 'ollama', 'auto']).optional(),
    embeddingModel: z.string().optional(),
    maxSessionContextTokens: z.number().optional(),
  }).passthrough().optional(),
  maxIterations: z.number().min(5).max(100).optional(),
  contextThreshold: z.number().min(10000).max(500000).optional(),
  keepToolUses: z.number().min(2).max(20).optional(),
  cacheTtlMs: z.number().min(60000).max(MS_PER_DAY).optional(),
  parallelToolLimit: z.number().min(0).max(10).optional(),
  llmCallTimeoutMs: z.number().min(30000).max(600000).optional(),
  /**
   * Forecasting pipeline settings (markov_distribution tool).
   * All fields are optional; defaults are applied by the respective modules.
   */
  forecasting: z.object({
    /** Enable Merton jump-diffusion step in Monte Carlo trajectory. Default: false. */
    enableJumpDiffusion: z.boolean().optional(),
    /** Cap on the Market Price of Risk (Sharpe Ratio) used in Q→P transformation. Default: 1.5. */
    qToPMprCap: z.number().min(0.1).max(10).optional(),
    /** Enable Markov-Switching Multifractal volatility model. Default: false. */
    enableMSM: z.boolean().optional(),
    /** Enable forecast-lab query auto-routing hints in the agent. Default: true. */
    enableForecastLabAutoRoute: z.boolean().optional(),
    /** Inject forecast-lab skill hints when auto-routing matches. Default: true. */
    enableForecastLabSkillHint: z.boolean().optional(),
    /** Rank structured forecast-lab mutators from ledger evidence. Default: false. */
    enableForecastLabMutatorRanking: z.boolean().optional(),
  }).passthrough().optional(),
}).passthrough(); // allow unknown keys without throwing

export type Config = z.infer<typeof ConfigSchema> & Record<string, unknown>;
