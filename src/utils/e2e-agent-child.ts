process.env.CRAMER_E2E_CHILD = '1';
export {};

const {
  CHILD_RESULT_MARKER,
  readE2EChildPayloadFromArgv,
  runAgentE2EInProcess,
} = await import('./e2e-helpers.js');

async function main() {
  const { query, opts } = readE2EChildPayloadFromArgv();
  const result = await runAgentE2EInProcess(query, opts);
  const encoded = Buffer.from(JSON.stringify(result), 'utf8').toString('base64');
  console.log(`${CHILD_RESULT_MARKER}:${encoded}`);
}

main().catch((error) => {
  const message = error instanceof Error ? error.stack ?? error.message : String(error);
  console.error(message);
  process.exit(1);
});
