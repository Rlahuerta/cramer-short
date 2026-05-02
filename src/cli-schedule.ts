/**
 * Cramer-Short schedule command: headless job runner for scheduled research tasks.
 *
 * Config file: ~/.cramer-short/schedules.json
 * Usage:
 *   cramer-short schedule list          — list configured jobs
 *   cramer-short schedule run            — run all jobs
 *   cramer-short schedule run <job-id>   — run a specific job
 */
import { readFile, writeFile, mkdir } from 'fs/promises';
import { existsSync } from 'fs';
import { homedir } from 'os';
import { dirname, join, resolve, sep } from 'path';
import { Agent } from './agent/agent.js';
import type { AgentEvent } from './agent/types.js';
import { assertForecastLabProfileId } from './experiments/forecast-lab/profiles.js';
import { runForecastLab } from './experiments/forecast-lab/runner.js';
import type { ForecastLabRunOptions, ForecastLabRunResult } from './experiments/forecast-lab/runner.js';

export interface AgentScheduleJob {
  /** Unique identifier used in `cramer-short schedule run <id>` */
  id: string;
  /** Job kind. Omitted means "agent" for backward compatibility. */
  kind?: 'agent';
  /** Human-readable description shown in `schedule list` */
  description: string;
  /** Natural-language research query / skill invocation sent to the agent */
  query: string;
  /**
   * Output file path. Supports `{date}` which is replaced with today's ISO date
   * (YYYY-MM-DD). Paths starting with `~` are expanded to the home directory.
   */
  outputFile: string;
}

export interface ForecastLabScheduleJob {
  /** Unique identifier used in `cramer-short schedule run <id>` */
  id: string;
  kind: 'forecast_lab';
  /** Human-readable description shown in `schedule list` */
  description?: string;
  /** Forecast-lab profile id to run. */
  profileId?: string;
  /** Reserved for future bounded mutation loops; accepted for config compatibility. */
  maxIterations?: number;
  /**
   * Optional output summary path. Supports `{date}` and `~` like agent jobs.
   */
  outputFile?: string;
  /** Defaults to true unless skipMutation is explicitly set. */
  dryRun?: boolean;
  /** Run the existing no-mutation mode instead of dry-run. */
  skipMutation?: boolean;
}

export type ScheduleJob = AgentScheduleJob | ForecastLabScheduleJob;

type AgentLike = {
  run(query: string): AsyncIterable<AgentEvent>;
};

export interface ScheduleCommandOptions {
  schedulesPath?: string;
  homeDir?: string;
  cwd?: string;
  log?: (message: string) => void;
  error?: (message: string) => void;
  write?: (message: string) => void;
  exit?: (code: number) => void;
  now?: () => Date;
  createAgent?: () => Promise<AgentLike>;
  runLab?: (options: ForecastLabRunOptions) => Promise<ForecastLabRunResult>;
}

const EXAMPLE_CONFIG: ScheduleJob[] = [
  {
    kind: 'agent',
    id: 'morning-briefing',
    description: 'Daily watchlist briefing',
    query: 'Run the watchlist-briefing skill for my portfolio',
    outputFile: '~/.cramer-short/reports/{date}-briefing.md',
  },
];

function scheduleKind(job: ScheduleJob): 'agent' | 'forecast_lab' {
  return job.kind ?? 'agent';
}

function getPaths(options: ScheduleCommandOptions): {
  schedulesPath: string;
  homeDir: string;
  cwd: string;
  allowedOutputRoots: string[];
} {
  const homeDir = options.homeDir ?? homedir();
  const cwd = options.cwd ?? process.cwd();
  const cramerShortHome = join(homeDir, '.cramer-short');

  return {
    schedulesPath: options.schedulesPath ?? join(cramerShortHome, 'schedules.json'),
    homeDir,
    cwd,
    allowedOutputRoots: [
      resolve(cramerShortHome),
      resolve(cwd, '.cramer-short'),
    ],
  };
}

async function loadJobs(options: ScheduleCommandOptions): Promise<ScheduleJob[]> {
  const { schedulesPath } = getPaths(options);
  const error = options.error ?? console.error;
  const exit = options.exit ?? ((code: number) => process.exit(code));

  if (!existsSync(schedulesPath)) {
    error(
      `No schedules config found at ${schedulesPath}\n` +
        `Create it with:\n${JSON.stringify(EXAMPLE_CONFIG, null, 2)}`,
    );
    exit(1);
    return [];
  }
  const raw = await readFile(schedulesPath, 'utf-8');
  return JSON.parse(raw) as ScheduleJob[];
}

function resolveOutputPath(template: string, options: ScheduleCommandOptions): string {
  const { homeDir, cwd, allowedOutputRoots } = getPaths(options);
  const now = options.now ?? (() => new Date());
  const date = now().toISOString().slice(0, 10);
  const expanded = template.replace('{date}', date).replace(/^~(?=$|\/)/, homeDir);
  // Resolve relative paths against cwd, then normalise to eliminate `..` segments.
  const resolved = resolve(cwd, expanded);

  // Defence-in-depth: schedule outputs must stay within an allow-listed root
  // (~/.cramer-short or <cwd>/.cramer-short) so a malicious schedules.json
  // cannot drop files into arbitrary locations such as ~/.ssh or /etc.
  const isAllowed = allowedOutputRoots.some(
    (root) => resolved === root || resolved.startsWith(root + sep),
  );
  if (!isAllowed) {
    throw new Error(
      `Refusing to write schedule output outside allowed roots.\n` +
        `  resolved: ${resolved}\n` +
        `  allowed:  ${allowedOutputRoots.join(', ')}`,
    );
  }
  return resolved;
}

async function runAgentJob(job: AgentScheduleJob, options: ScheduleCommandOptions): Promise<void> {
  const log = options.log ?? console.log;
  const error = options.error ?? console.error;
  const write = options.write ?? ((message: string) => process.stdout.write(message));
  const now = options.now ?? (() => new Date());
  const createAgent = options.createAgent ?? (() => Agent.create());

  log(`\n▶ Running job "${job.id}": ${job.description}`);
  const outPath = resolveOutputPath(job.outputFile, options);
  const outDir = dirname(outPath);
  if (outDir) await mkdir(outDir, { recursive: true });

  const agent = await createAgent();
  let answer = '';

  for await (const event of agent.run(job.query)) {
    if (event.type === 'tool_start') {
      log(`  ⚙ ${event.tool}(${JSON.stringify(event.args ?? {}).slice(0, 80)})`);
    } else if (event.type === 'answer_chunk') {
      write(event.chunk);
      answer += event.chunk;
    } else if (event.type === 'done') {
      if (!answer) answer = event.answer ?? '';
      log(`\n✓ Job complete (${event.iterations} iterations, ${event.totalTime}ms)`);
    }
  }

  if (!answer) {
    error(`✗ Job "${job.id}" produced no answer.`);
    return;
  }

  const header = `# ${job.description}\n_Generated by Cramer-Short — ${now().toISOString()}_\n\n`;
  await writeFile(outPath, header + answer, 'utf-8');
  log(`\n💾 Output saved to ${outPath}`);
}

function assertForecastLabJob(job: ForecastLabScheduleJob): asserts job is ForecastLabScheduleJob & { profileId: string } {
  if (!job.profileId) {
    throw new Error(`Forecast-lab schedule job "${job.id}" is missing profileId.`);
  }
  assertForecastLabProfileId(job.profileId);
  if (job.dryRun === false && job.skipMutation !== true) {
    throw new Error(
      `Forecast-lab schedule job "${job.id}" sets dryRun: false, but forecast-lab V1 does not support real mutation. ` +
        `Set skipMutation: true or leave dryRun unset/true.`,
    );
  }
}

function formatForecastLabOutput(job: ForecastLabScheduleJob, result: ForecastLabRunResult, now: Date): string {
  const metrics = result.decision.metrics.length > 0
    ? result.decision.metrics
        .map((metric) => (
          `| ${metric.name} | ${metric.baseline} | ${metric.candidate} | ${metric.delta} |`
        ))
        .join('\n')
    : '| _none_ | | | |';

  return [
    `# ${job.description ?? job.id}`,
    `_Generated by Cramer-Short forecast-lab — ${now.toISOString()}_`,
    '',
    `- Profile: ${result.manifest.profileId}`,
    `- Run ID: ${result.runId}`,
    `- Decision: ${result.decision.decision}`,
    `- Reason: ${result.decision.reason}`,
    `- Artifacts: ${result.manifest.artifactsPath}`,
    '',
    '| Metric | Baseline | Candidate | Delta |',
    '|---|---:|---:|---:|',
    metrics,
    '',
  ].join('\n');
}

async function runForecastLabJob(job: ForecastLabScheduleJob, options: ScheduleCommandOptions): Promise<void> {
  const log = options.log ?? console.log;
  const now = options.now ?? (() => new Date());
  const runLab = options.runLab ?? runForecastLab;

  assertForecastLabJob(job);

  log(`\n▶ Running forecast-lab job "${job.id}": ${job.description ?? job.profileId}`);
  const result = await runLab({
    profileId: job.profileId,
    dryRun: job.dryRun ?? (job.skipMutation !== true),
    skipMutation: job.skipMutation === true,
  });

  log(`forecast-lab ${result.decision.decision}: ${result.decision.reason}`);
  log(`artifacts: ${result.manifest.artifactsPath}`);

  if (job.outputFile) {
    const outPath = resolveOutputPath(job.outputFile, options);
    const outDir = dirname(outPath);
    if (outDir) await mkdir(outDir, { recursive: true });
    await writeFile(outPath, formatForecastLabOutput(job, result, now()), 'utf-8');
    log(`\n💾 Output saved to ${outPath}`);
  }
}

async function runJob(job: ScheduleJob, options: ScheduleCommandOptions): Promise<void> {
  const kind = scheduleKind(job);

  if (kind === 'agent') {
    await runAgentJob(job as AgentScheduleJob, options);
    return;
  }

  if (kind === 'forecast_lab') {
    await runForecastLabJob(job as ForecastLabScheduleJob, options);
    return;
  }

  throw new Error(`Unknown schedule job kind "${String((job as { kind?: unknown }).kind)}" for job "${job.id}".`);
}

function printUsage(options: ScheduleCommandOptions): void {
  const { schedulesPath } = getPaths(options);
  const log = options.log ?? console.log;

  log(
    [
      'Usage:',
      '  cramer-short schedule list          — list configured jobs',
      '  cramer-short schedule run            — run all jobs',
      '  cramer-short schedule run <job-id>  — run a specific job',
      '',
      `Config: ${schedulesPath}`,
    ].join('\n'),
  );
}

export async function runScheduleCommand(argv: string[], options: ScheduleCommandOptions = {}): Promise<void> {
  const [subCmd, jobId] = argv; // e.g. ['run', 'morning-briefing'] or ['list']
  const { schedulesPath } = getPaths(options);
  const log = options.log ?? console.log;
  const error = options.error ?? console.error;
  const exit = options.exit ?? ((code: number) => process.exit(code));

  if (!subCmd || subCmd === 'help' || subCmd === '--help') {
    printUsage(options);
    return;
  }

  if (subCmd === 'list') {
    const jobs = await loadJobs(options);
    log(`\nConfigured jobs (${schedulesPath}):\n`);
    for (const job of jobs) {
      if (scheduleKind(job) === 'forecast_lab') {
        const forecastJob = job as ForecastLabScheduleJob;
        log(`  ${job.id.padEnd(24)} ${forecastJob.description ?? forecastJob.profileId ?? ''}`);
        log(`  ${''.padEnd(24)} kind: forecast_lab`);
        log(`  ${''.padEnd(24)} profile: ${forecastJob.profileId ?? '(missing)'}`);
      } else {
        log(`  ${job.id.padEnd(24)} ${job.description}`);
      }
      if (job.outputFile) {
        log(`  ${''.padEnd(24)} output: ${job.outputFile}\n`);
      }
    }
    return;
  }

  if (subCmd === 'run') {
    const jobs = await loadJobs(options);
    const targets = jobId ? jobs.filter((j) => j.id === jobId) : jobs;

    if (jobId && targets.length === 0) {
      error(`No job found with id "${jobId}". Run \`cramer-short schedule list\` to see available jobs.`);
      exit(1);
      return;
    }

    try {
      for (const job of targets) {
        await runJob(job, options);
      }
    } catch (caught) {
      error(caught instanceof Error ? caught.message : String(caught));
      exit(1);
    }
    return;
  }

  error(`Unknown subcommand: "${subCmd}"`);
  printUsage(options);
  exit(1);
}
