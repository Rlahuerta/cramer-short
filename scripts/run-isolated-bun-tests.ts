type IsolatedTestSpec = {
  file: string;
  timeoutMs?: number;
  env?: Record<string, string>;
};

export {};

const E2E_SPEC_OVERRIDES: Record<string, Omit<IsolatedTestSpec, 'file'>> = {
  'src/agent/agent.e2e.test.ts': { timeoutMs: 360_000 },
  'src/agent/bitmex-trade-prompt.e2e.test.ts': { timeoutMs: 600_000, env: { E2E_TIMEOUT_MS: '600000' } },
  'src/skills/dcf/skill.e2e.test.ts': { timeoutMs: 600_000, env: { E2E_TIMEOUT_MS: '600000' } },
  'src/skills/probability-assessment/skill.e2e.test.ts': { timeoutMs: 600_000, env: { E2E_TIMEOUT_MS: '600000' } },
  'src/skills/peer-comparison/skill.e2e.test.ts': { timeoutMs: 360_000 },
  'src/skills/portfolio-risk/skill.e2e.test.ts': { timeoutMs: 360_000 },
  'src/skills/forecast-lab/skill.e2e.test.ts': { timeoutMs: 360_000 },
  'src/skills/forecast-lab/asset-scope.e2e.test.ts': { timeoutMs: 600_000, env: { E2E_TIMEOUT_MS: '600000' } },
  'src/model/llm.e2e.test.ts': { timeoutMs: 360_000 },
  'src/model/llm-streaming.e2e.test.ts': { timeoutMs: 360_000 },
  'src/memory/embeddings.e2e.test.ts': { timeoutMs: 60_000 },
  'src/tools/finance/polymarket-history-docs.e2e.test.ts': { timeoutMs: 360_000 },
};

async function findSpecFiles(pattern: string): Promise<string[]> {
  const files: string[] = [];
  for await (const file of new Bun.Glob(pattern).scan('.')) {
    files.push(file);
  }
  files.sort((a, b) => a.localeCompare(b));
  return files;
}

async function getIntegrationSpecs(): Promise<IsolatedTestSpec[]> {
  return (await findSpecFiles('src/**/*.integration.test.ts')).map((file) => ({ file }));
}

async function getE2ESpecs(): Promise<IsolatedTestSpec[]> {
  return (await findSpecFiles('src/**/*.e2e.test.ts')).map((file) => ({
    file,
    ...E2E_SPEC_OVERRIDES[file],
  }));
}

async function resolveSpecs(mode: string): Promise<{ specs: IsolatedTestSpec[]; baseEnv: Record<string, string> }> {
  if (mode === 'e2e') {
    return { specs: await getE2ESpecs(), baseEnv: { RUN_E2E: '1' } };
  }
  if (mode === 'integration') {
    // Explicit tier opt-in beats an ambient SKIP_INTEGRATION: getBooleanEnv('') is false.
    return { specs: await getIntegrationSpecs(), baseEnv: { RUN_INTEGRATION: '1', SKIP_INTEGRATION: '' } };
  }
  throw new Error(`Unknown isolated test mode: ${mode}`);
}

const mode = process.argv[2];
if (!mode) {
  throw new Error('Usage: bun run scripts/run-isolated-bun-tests.ts <integration|e2e>');
}

const { specs, baseEnv } = await resolveSpecs(mode);
if (specs.length === 0) {
  console.error(`No ${mode} test files found.`);
  process.exit(1);
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
