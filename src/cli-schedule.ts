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
import { createReplayHistoryLoader } from './cli-replay-label.js';
import { assertForecastLabProfileId } from './experiments/forecast-lab/profiles.js';
import { runForecastLab } from './experiments/forecast-lab/runner.js';
import type { ForecastLabRunOptions, ForecastLabRunResult } from './experiments/forecast-lab/runner.js';
import {
  runReplayLabelBenchmarkPipelineFromFile,
  toReplayLabelBenchmarkReportPath,
  type ReplayLabelBenchmarkPipelineResult,
} from './tools/finance/backtest/replay-label-benchmark-pipeline.js';
import {
  toReplayLabelBatchReportPath,
  type ReplayTickerHistoryLoader,
} from './tools/finance/backtest/replay-label-batch-runner.js';
import { DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH } from './tools/finance/arbiter-replay.js';
import { DEFAULT_ARBITER_REPLAY_LABELED_PATH } from './tools/finance/backtest/replay-label-runner.js';

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
  /** Explicit real-mutation mode. Required for scheduled mutation runs. */
  mutationMode?: 'structured';
  /** Preserve the candidate worktree after a real mutation run for debugging. */
  keepWorktree?: boolean;
  /** Optional shipped structured mutator id override for debugging. */
  mutator?: string;
}

export interface ReplayLabelScheduleJob {
  /** Unique identifier used in `cramer-short schedule run <id>` */
  id: string;
  kind: 'replay_label';
  /** Human-readable description shown in `schedule list` */
  description?: string;
  /** Input replay bundles path. Defaults to the replay-label CLI cache path. */
  inputPath?: string;
  /** Labeled output path. Supports `{date}` and `~` like agent jobs. */
  outputPath?: string;
  /** Optional explicit label batch report path. Defaults to a path derived from outputPath. */
  labelReportPath?: string;
  /** Optional explicit benchmark report path. Defaults to a path derived from outputPath. */
  benchmarkReportPath?: string;
  /** Replay history loader mode. Supports fixture and local:<path>. */
  loader?: string;
  /** Optional human-readable summary path. Supports `{date}` and `~` like agent jobs. */
  outputFile?: string;
}

export type ScheduleJob = AgentScheduleJob | ForecastLabScheduleJob | ReplayLabelScheduleJob;

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
  runReplayLabelPipeline?: (params: {
    inputPath?: string;
    outputPath?: string;
    labelReportPath?: string;
    benchmarkReportPath?: string;
    loadHistory: ReplayTickerHistoryLoader;
  }) => Promise<ReplayLabelBenchmarkPipelineResult>;
  replayLabelLoaderFactory?: (mode: string) => ReplayTickerHistoryLoader;
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

function scheduleKind(job: ScheduleJob): 'agent' | 'forecast_lab' | 'replay_label' {
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

  assertPathWithinAllowedRoots(resolved, allowedOutputRoots);
  return resolved;
}

function assertPathWithinAllowedRoots(resolved: string, allowedOutputRoots: string[]): void {
  // Defence-in-depth: schedule-managed paths must stay within an allow-listed root
  // (~/.cramer-short or <cwd>/.cramer-short) so a malicious schedules.json
  // cannot read from or write to arbitrary locations such as ~/.ssh or /etc.
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
}

function assertReplayLabelInputPathAllowed(inputPath: string, options: ScheduleCommandOptions): void {
  const { homeDir, cwd, allowedOutputRoots } = getPaths(options);
  const expanded = inputPath.replace(/^~(?=$|\/)/, homeDir);
  const resolved = resolve(cwd, expanded);

  assertPathWithinAllowedRoots(resolved, allowedOutputRoots);
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

function formatReplayLabelOutput(params: {
  job: ReplayLabelScheduleJob;
  now: Date;
  inputPath: string;
  outputPath: string;
  labelReportPath: string;
  benchmarkReportPath: string;
  loaderMode: string;
  result: ReplayLabelBenchmarkPipelineResult;
}): string {
  return [
    `# ${params.job.description ?? params.job.id}`,
    `_Generated by Cramer-Short replay-label — ${params.now.toISOString()}_`,
    '',
    `- Input: ${params.inputPath}`,
    `- Output: ${params.outputPath}`,
    `- Label report: ${params.labelReportPath}`,
    `- Benchmark report: ${params.benchmarkReportPath}`,
    `- Loader: ${params.loaderMode}`,
    '',
    `- Total bundles: ${params.result.labeling.summary.total}`,
    `- Already labeled: ${params.result.labeling.summary.alreadyLabeled}`,
    `- Newly labeled: ${params.result.labeling.summary.newlyLabeled}`,
    `- Missing history: ${params.result.labeling.summary.skippedByMissingHistory}`,
    `- Pending: ${params.result.labeling.summary.pending}`,
    '',
  ].join('\n');
}

async function runReplayLabelJob(job: ReplayLabelScheduleJob, options: ScheduleCommandOptions): Promise<void> {
  const log = options.log ?? console.log;
  const now = options.now ?? (() => new Date());
  const loaderMode = job.loader ?? 'fixture';
  const inputPath = job.inputPath ?? DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH;
  assertReplayLabelInputPathAllowed(inputPath, options);
  const outputPath = resolveOutputPath(job.outputPath ?? DEFAULT_ARBITER_REPLAY_LABELED_PATH, options);
  const labelReportPath = resolveOutputPath(
    job.labelReportPath ?? toReplayLabelBatchReportPath(outputPath),
    options,
  );
  const benchmarkReportPath = resolveOutputPath(
    job.benchmarkReportPath ?? toReplayLabelBenchmarkReportPath(outputPath),
    options,
  );
  const summaryOutputPath = job.outputFile ? resolveOutputPath(job.outputFile, options) : undefined;
  const loaderFactory = options.replayLabelLoaderFactory ?? createReplayHistoryLoader;
  const loadHistory = loaderFactory(loaderMode);
  const runReplayLabelPipeline = options.runReplayLabelPipeline ?? runReplayLabelBenchmarkPipelineFromFile;

  log(`\n▶ Running replay-label job "${job.id}": ${job.description ?? job.id}`);
  const result = await runReplayLabelPipeline({
    inputPath,
    outputPath,
    labelReportPath,
    benchmarkReportPath,
    loadHistory,
  });

  log(
    `replay-label labeled ${result.labeling.summary.newlyLabeled}/${result.labeling.summary.total} bundles ` +
      `(missing history: ${result.labeling.summary.skippedByMissingHistory}, pending: ${result.labeling.summary.pending})`,
  );
  log(`artifacts: ${outputPath}`);
  log(`label report: ${labelReportPath}`);
  log(`benchmark report: ${benchmarkReportPath}`);

  if (!summaryOutputPath) {
    return;
  }

  const outDir = dirname(summaryOutputPath);
  if (outDir) await mkdir(outDir, { recursive: true });
  await writeFile(
    summaryOutputPath,
    formatReplayLabelOutput({
      job,
      now: now(),
      inputPath,
      outputPath,
      labelReportPath,
      benchmarkReportPath,
      loaderMode,
      result,
    }),
    'utf-8',
  );
  log(`\n💾 Output saved to ${summaryOutputPath}`);
}

function assertForecastLabJob(job: ForecastLabScheduleJob): asserts job is ForecastLabScheduleJob & { profileId: string } {
  if (!job.profileId) {
    throw new Error(`Forecast-lab schedule job "${job.id}" is missing profileId.`);
  }
  assertForecastLabProfileId(job.profileId);

  if (job.mutationMode !== undefined && job.mutationMode !== 'structured') {
    throw new Error(
      `Forecast-lab schedule job "${job.id}" has unsupported mutationMode "${job.mutationMode}". Expected "structured".`,
    );
  }

  if (job.dryRun === true && job.skipMutation === true) {
    throw new Error(
      `Forecast-lab schedule job "${job.id}" cannot set both dryRun: true and skipMutation: true.`,
    );
  }

  if (job.dryRun !== false && job.mutationMode !== undefined) {
    throw new Error(
      `Forecast-lab schedule job "${job.id}" sets mutationMode, but scheduled mutation runs must also set dryRun: false.`,
    );
  }

  if (job.skipMutation === true && job.mutationMode !== undefined) {
    throw new Error(
      `Forecast-lab schedule job "${job.id}" cannot combine skipMutation: true with mutationMode.`,
    );
  }

  if ((job.keepWorktree === true || job.mutator !== undefined) && job.mutationMode !== 'structured') {
    throw new Error(
      `Forecast-lab schedule job "${job.id}" uses keepWorktree/mutator debugging controls, but they require mutationMode: "structured".`,
    );
  }

  if (job.dryRun === false && job.skipMutation !== true && job.mutationMode === undefined) {
    throw new Error(
      `Forecast-lab schedule job "${job.id}" sets dryRun: false without an explicit mutationMode. ` +
        `Set mutationMode: "structured" for real mutation or set skipMutation: true for the no-mutation path.`,
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
  const selectedMutator = job.mutator;
  const mutationOperator = result.manifest.mutationSpecSummary?.mutatorId;
  const mutationLines = result.manifest.mutationMode
    ? [
        `- Mutation mode: ${result.manifest.mutationMode}`,
        ...(selectedMutator ? [`- Selected mutator: ${selectedMutator}`] : []),
        ...(mutationOperator ? [`- Mutation operator: ${mutationOperator}`] : []),
        ...(result.manifest.lineage
          ? [
              `- Lineage: root=${result.manifest.lineage.rootRunId}, parent=${result.manifest.lineage.parentRunId ?? 'none'}, generation=${result.manifest.lineage.generation}`,
            ]
          : []),
        ...(result.manifest.candidateWorkspace
          ? [
              `- Candidate workspace: ${result.manifest.candidateWorkspace.kind} @ ${result.manifest.candidateWorkspace.rootDir}`,
              `- Candidate branch: ${result.manifest.candidateWorkspace.branch}`,
            ]
          : []),
        '',
      ]
    : [];

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
    ...mutationLines,
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
  const write = options.write ?? ((message: string) => process.stdout.write(message));

  assertForecastLabJob(job);

  log(`\n▶ Running forecast-lab job "${job.id}": ${job.description ?? job.profileId}`);
  const result = await runLab({
    profileId: job.profileId,
    dryRun: job.dryRun ?? (job.skipMutation !== true),
    skipMutation: job.skipMutation === true,
    mutationMode: job.mutationMode,
    keepWorktree: job.keepWorktree === true,
    mutator: job.mutator,
    progress: log,
    output: write,
  });

  log(`forecast-lab ${result.decision.decision}: ${result.decision.reason}`);
  log(`artifacts: ${result.manifest.artifactsPath}`);
  if (result.manifest.mutationMode) {
    log(`mutation: ${result.manifest.mutationMode}`);
    if (job.mutator) {
      log(`selected mutator: ${job.mutator}`);
    }
    if (result.manifest.mutationSpecSummary?.mutatorId) {
      log(`mutation operator: ${result.manifest.mutationSpecSummary.mutatorId}`);
    }
    if (result.manifest.lineage) {
      log(
        `lineage: root=${result.manifest.lineage.rootRunId}, parent=${result.manifest.lineage.parentRunId ?? 'none'}, generation=${result.manifest.lineage.generation}`,
      );
    }
    if (result.manifest.candidateWorkspace) {
      log(
        `candidate workspace: ${result.manifest.candidateWorkspace.kind} ${result.manifest.candidateWorkspace.rootDir} (${result.manifest.candidateWorkspace.branch})`,
      );
    }
  }

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

  if (kind === 'replay_label') {
    await runReplayLabelJob(job as ReplayLabelScheduleJob, options);
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
        if (forecastJob.mutationMode) {
          log(`  ${''.padEnd(24)} mutation: ${forecastJob.mutationMode}`);
        }
      } else if (scheduleKind(job) === 'replay_label') {
        const replayJob = job as ReplayLabelScheduleJob;
        log(`  ${job.id.padEnd(24)} ${replayJob.description ?? replayJob.id}`);
        log(`  ${''.padEnd(24)} kind: replay_label`);
        log(`  ${''.padEnd(24)} input: ${replayJob.inputPath ?? DEFAULT_ARBITER_REPLAY_CACHE_BUNDLES_PATH}`);
        log(`  ${''.padEnd(24)} artifacts: ${replayJob.outputPath ?? DEFAULT_ARBITER_REPLAY_LABELED_PATH}`);
        log(`  ${''.padEnd(24)} loader: ${replayJob.loader ?? 'fixture'}`);
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
