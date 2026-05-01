/**
 * Inspect response shapes for Ollama cloud models
 */
import { callLlm } from './src/model/llm.js';
import { getOllamaModels } from './src/utils/ollama.js';

async function main() {
  console.log('Fetching Ollama cloud models...');
  const allModels = await getOllamaModels();
  console.log(`Total models: ${allModels.length}`);
  console.log(`Models: ${allModels.join(', ')}\n`);

  // Filter for cloud models from the list
  const cloudModelPatterns = [
    'deepseek-v4-pro:cloud',
    'glm-5.1:cloud',
    'glm-5:cloud',
    'minimax-m2.7:cloud',
    'qwen3.5:cloud',
    'qwen3-next:80b-cloud',
  ];

  const availableCloudModels = allModels.filter((m) =>
    cloudModelPatterns.some((p) => m.includes(p))
  );

  console.log(`Available cloud models: ${availableCloudModels.length}`);
  console.log(`Models: ${availableCloudModels.join(', ')}\n`);

  if (availableCloudModels.length === 0) {
    console.log('No cloud models found.');
    return;
  }

  // Test up to 4 models
  const modelsToTest = availableCloudModels.slice(0, 4);

  for (const modelName of modelsToTest) {
    console.log(`\n${'='.repeat(60)}`);
    console.log(`Testing model: ${modelName}`);
    console.log('='.repeat(60));

    const startTime = Date.now();
    try {
      const result = await callLlm(
        'Reply with exactly the word PONG and nothing else.',
        {
          model: `ollama:${modelName}`,
          systemPrompt: 'You are a minimal test assistant. Follow instructions exactly.',
          timeoutMs: 45000,
          thinkOverride: false,
        }
      );

      const elapsed = Date.now() - startTime;
      console.log(`✓ Success (${elapsed}ms)`);
      console.log(`Response type: ${typeof result.response}`);

      if (typeof result.response === 'object' && result.response !== null) {
        const keys = Object.keys(result.response);
        console.log(`Content block keys: ${keys.join(', ')}`);
        for (const key of keys) {
          const val = (result.response as Record<string, unknown>)[key];
          console.log(`  - ${key}: ${typeof val}`);
        }
      } else if (typeof result.response === 'string') {
        console.log(`Response content: "${result.response}"`);
      }

      if (result.usage) {
        console.log(`Usage: ${JSON.stringify(result.usage)}`);
      }
    } catch (error) {
      const elapsed = Date.now() - startTime;
      console.log(`✗ Failed (${elapsed}ms)`);
      console.log(`Error: ${error instanceof Error ? error.message : String(error)}`);
    }
  }
}

main().catch(console.error);
