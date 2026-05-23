type IsolatedTestSpec = {
  file: string;
  timeoutMs?: number;
  env?: Record<string, string>;
};

const E2E_SPECS: IsolatedTestSpec[] = [
  { file: 'src/agent/agent.e2e.test.ts', timeoutMs: 360_000 },
  { file: 'src/agent/bitmex-trade-prompt.e2e.test.ts', timeoutMs: 600_000, env: { E2E_TIMEOUT_MS: '600000' } },
  { file: 'src/skills/dcf/skill.e2e.test.ts', timeoutMs: 600_000, env: { E2E_TIMEOUT_MS: '600000' } },
  { file: 'src/skills/probability-assessment/skill.e2e.test.ts', timeoutMs: 600_000, env: { E2E_TIMEOUT_MS: '600000' } },
  { file: 'src/skills/peer-comparison/skill.e2e.test.ts', timeoutMs: 360_000 },
  { file: 'src/skills/portfolio-risk/skill.e2e.test.ts', timeoutMs: 360_000 },
  { file: 'src/skills/forecast-lab/skill.e2e.test.ts', timeoutMs: 360_000 },
  { file: 'src/skills/forecast-lab/asset-scope.e2e.test.ts', timeoutMs: 600_000, env: { E2E_TIMEOUT_MS: '600000' } },
  { file: 'src/model/llm.e2e.test.ts', timeoutMs: 360_000 },
  { file: 'src/model/llm-streaming.e2e.test.ts', timeoutMs: 360_000 },
  { file: 'src/memory/embeddings.e2e.test.ts', timeoutMs: 60_000 },
  { file: 'src/tools/finance/polymarket-history-docs.e2e.test.ts', timeoutMs: 360_000 },
];

async function getIntegrationSpecs(): Promise<IsolatedTestSpec[]> {
  const files: string[] = [];
  for await (const file of new Bun.Glob('src/**/*.integration.test.ts').scan('.')) {
    files.push(file);
  }
  files.sort((a, b) => a.localeCompare(b));
  return files.map((file) => ({ file }));
}

async function resolveSpecs(mode: string): Promise<{ specs: IsolatedTestSpec[]; baseEnv: Record<string, string> }> {
  if (mode === 'e2e') {
    return { specs: E2E_SPECS, baseEnv: { RUN_E2E: '1' } };
  }
  if (mode === 'integration') {
    return { specs: await getIntegrationSpecs(), baseEnv: { RUN_INTEGRATION: '1' } };
  }
  throw new Error(`Unknown isolated test mode: ${mode}`);
}

const mode = process.argv[2];
if (!mode) {
  throw new Error('Usage: bun run scripts/run-isolated-bun-tests.ts <integration|e2e>');
}

const { specs, baseEnv } = await resolveSpecs(mode);
if (specs.length === 0) {
  console.log(`No ${mode} test files found.`);
  process.exit(0);
}

for (const spec of specs) {
  const cmd = [process.execPath, 'test', spec.file];
  if (spec.timeoutMs) {
    cmd.push('--timeout', String(spec.timeoutMs));
  }

  console.log(`\n=== ${spec.file} ===`);
  const proc = Bun.spawn({
    cmd,
    cwd: process.cwd(),
    env: {
      ...process.env,
      ...baseEnv,
      ...spec.env,
    },
    stdout: 'inherit',
    stderr: 'inherit',
  });

  const exitCode = await proc.exited;
  if (exitCode !== 0) {
    process.exit(exitCode);
  }
}
