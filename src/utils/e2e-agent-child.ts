import { setEnv } from './env.js';

setEnv('CRAMER_E2E_CHILD', '1');
export {};

const {
  CHILD_RESULT_MARKER,
  readE2EChildPayloadFromArgv,
  runAgentE2EInProcess,
} = await import('./e2e-helpers.js');

async function main() {
  const { query, opts, resultFilePath } = readE2EChildPayloadFromArgv();
  const result = await runAgentE2EInProcess(query, opts);
  const json = JSON.stringify(result);

  if (resultFilePath) {
    await Bun.write(resultFilePath, json);
    console.log(`${CHILD_RESULT_MARKER}:file:${resultFilePath}`);
    return;
  }

  const encoded = Buffer.from(json, 'utf8').toString('base64');
  console.log(`${CHILD_RESULT_MARKER}:${encoded}`);
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error);
  console.error(message);
  process.exit(1);
});
