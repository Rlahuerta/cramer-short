import { z } from 'zod';

/**
 * Schema for LLM to select relevant messages.
 */
export const SelectedMessagesSchema = z.object({
  message_ids: z.array(z.number()).describe('List of relevant message IDs (0-indexed)'),
});

export type SelectedMessages = z.infer<typeof SelectedMessagesSchema>;
