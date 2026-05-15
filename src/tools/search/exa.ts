import { DynamicStructuredTool } from '@langchain/core/tools';
import { ExaSearchResults } from '@langchain/exa';
import Exa from 'exa-js';
import { z } from 'zod';
import { formatToolResult, parseSearchResults } from '../types.js';
import { logger } from '../../utils/logger.js';

// Lazily initialized to avoid errors when API key is not set
let exaTool: { invoke: (query: string) => Promise<unknown> } | null = null;
type ExaSearchResultsOptions = ConstructorParameters<typeof ExaSearchResults>[0];

function getExaTool(): { invoke: (query: string) => Promise<unknown> } {
  if (!exaTool) {
    const client = new Exa(process.env.EXASEARCH_API_KEY);
    // exa-js@2.x (root) vs exa-js@1.x (inside @langchain/exa) have
    // incompatible private fields but are compatible at runtime.
    exaTool = new ExaSearchResults({
      client: client as unknown as ExaSearchResultsOptions['client'],
      searchArgs: { numResults: 5, highlights: true },
    });
  }
  return exaTool!;
}

/**
 * LangChain web-search tool backed by Exa when EXASEARCH_API_KEY is configured.
 */
export const exaSearch = new DynamicStructuredTool({
  name: 'web_search',
  description:
    'Search the web for current information on any topic. Returns relevant search results with URLs and content snippets.',
  schema: z.object({
    query: z.string().describe('The search query to look up on the web'),
  }),
  func: async (input) => {
    try {
      const result = await getExaTool().invoke(input.query);
      const { parsed, urls } = parseSearchResults(result);
      return formatToolResult(parsed, urls);
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      logger.error(`[Exa API] error: ${message}`);
      throw new Error(`[Exa API] ${message}`);
    }
  },
});
