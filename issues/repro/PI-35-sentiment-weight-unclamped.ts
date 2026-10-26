/**
 * PI-35: engagement weighting makes the social sentiment score NaN / out of range
 * Found at: e3e2088b21acc4963b3c20661e5d15c739244409
 * Repro kind: runtime
 *
 * aggregateStats() weights each post by `1 + score / maxScore` where `score` is
 * the raw Reddit upvote count. Reddit scores can be negative, so the weight can
 * reach 0 (division by zero → NaN) or go negative (sentiment contribution
 * inverted, |score| > 100). formatSentimentReport then prints e.g. "score
 * +NaN/100" and sentimentEmoji(NaN) falls through to "Very Bearish".
 */

import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { homedir, tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const REPO_ROOT = fileURLToPath(new URL('../../', import.meta.url));
const EVIDENCE = join(REPO_ROOT, '.omo/evidence/bug-hunt-reproducible-issues');

class HarnessFailure extends Error {}

function harnessFailure(reason: string): never {
  console.log(`HARNESS-FAILURE: ${reason}`);
  throw new HarnessFailure(reason);
}

function normalize(text: string, root: string): string {
  return text
    .replaceAll(root, '<TMP>')
    .replace(/\u001b\[[0-9;]*m/g, '')
    .replace(/\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z/g, '<TS>');
}

interface Post {
  source: 'reddit';
  id: string;
  text: string;
  author: string;
  score: number;
  url: string;
  created_at: string;
  sentiment: 'bullish' | 'bearish' | 'neutral';
  sentimentScore: number;
}

const post = (id: string, score: number, sentiment: Post['sentiment'], sentimentScore: number): Post => ({
  source: 'reddit',
  id,
  text: 'sample',
  author: 'u/test',
  score,
  url: 'https://reddit.com/r/test',
  created_at: '2024-01-01T00:00:00.000Z',
  sentiment,
  sentimentScore,
});

async function main(): Promise<void> {
  const root = process.env.CS_REPRO_ROOT;
  if (!root) harnessFailure('CS_REPRO_ROOT is unset');
  if (homedir() !== root) harnessFailure(`homedir() ${homedir()} escaped ${root}`);
  if (tmpdir() !== root) harnessFailure(`tmpdir() ${tmpdir()} escaped ${root}`);

  const savedCwd = process.cwd();
  const savedEnv: Record<string, string | undefined> = { ...process.env };
  const savedFetch = globalThis.fetch;
  const keys = readFileSync(join(EVIDENCE, 'pre-work.envkeys'), 'utf-8')
    .split(/\s+/)
    .filter(Boolean);
  const extra = [
    'OLLAMA_BASE_URL', 'DEXTER_GATEWAY_CONFIG', 'DEXTER_PAIRING_PATH',
    'CRAMER_SHORT_SESSIONS_DIR', 'KALSHI_API_KEY', 'OPENROUTER_API_KEY',
    'E2E_MODEL', 'FORECAST_LAB_OUTPUT_METRICS', 'COVERAGE_LINES_MIN', 'TEST_API_KEY',
  ];
  const secretPattern = /(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|COOKIE)/;

  try {
    process.chdir(root);
    for (const key of [...keys, ...extra]) {
      if (key !== 'HOME' && key !== 'USERPROFILE') delete process.env[key];
    }
    for (const key of Object.keys(process.env)) {
      if (secretPattern.test(key)) delete process.env[key];
    }
    for (const key of Object.keys(process.env)) {
      if (secretPattern.test(key)) harnessFailure(`secret survived: ${key}`);
    }
    globalThis.fetch = (async (input: RequestInfo | URL): Promise<Response> => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : input.url;
      throw new Error(`network disabled: ${url}`);
    }) as typeof fetch;

    const mod = await import(
      pathToFileURL(join(REPO_ROOT, 'src/tools/finance/social-sentiment.ts')).href
    );

    // One downvoted post: weight = 1 + (-1 / 1) = 0 → 0 / 0.
    const single = mod.aggregateStats([post('a', -1, 'bullish', 1)]);
    // Extreme scores: weight = 1 + (-200 / 100) = -1 → contribution flips sign.
    const extreme = mod.aggregateStats([
      post('b', 100, 'bullish', 1),
      post('c', -200, 'bearish', -1),
    ]);

    const singleValid = Number.isFinite(single.avgScore) && single.avgScore >= -100 && single.avgScore <= 100;
    const extremeValid = Number.isFinite(extreme.avgScore) && extreme.avgScore >= -100 && extreme.avgScore <= 100;

    const emit = (line: string): void => { console.log(normalize(line, root)); };
    const observed = [
      `  single downvoted bullish post: avgScore = ${single.avgScore}, emoji = ${mod.sentimentEmoji(single.avgScore)}`,
      `  [+100 bullish, -200 bearish]: avgScore = ${extreme.avgScore}, emoji = ${mod.sentimentEmoji(extreme.avgScore)}`,
    ];

    let failure: string | null = null;
    try {
      assert.ok(singleValid, `single-post avgScore ${single.avgScore} is not a finite -100..100 value`);
      assert.ok(extremeValid, `mixed avgScore ${extreme.avgScore} is not a finite -100..100 value`);
    } catch (error) {
      if (!(error instanceof assert.AssertionError)) throw error;
      failure = error.message;
    }

    if (failure) {
      emit('OBSERVED:');
      for (const line of observed) emit(line);
      emit('EXPECTED:');
      emit('  avgScore finite and within -100..100 for both post sets');
      emit('ACTUAL:');
      emit(`  avgScore = ${single.avgScore} (single post) and ${extreme.avgScore} (mixed extremes)`);
      process.exitCode = 42;
      return;
    }
    console.log('NO BUG');
  } finally {
    globalThis.fetch = savedFetch;
    process.chdir(savedCwd);
    for (const key of Object.keys(process.env)) delete process.env[key];
    for (const [key, value] of Object.entries(savedEnv)) {
      if (value !== undefined) process.env[key] = value;
    }
  }
}

try {
  await main();
} catch (error) {
  if (error instanceof HarnessFailure) {
    process.exitCode = 43;
  } else {
    throw error;
  }
}
