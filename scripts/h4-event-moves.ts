/**
 * H4 of the Polymarket <-> Markov review (docs/reviews/polymarket-markov-connection-review-2026-11-26.md): day-over-day
 * moves in Polymarket probabilities for scheduled macro and crypto events, mapped to assets with a learned sign and
 * size, predict the next trading day's |return| (and possibly direction) beyond trailing volatility. A measurement, no
 * production change.
 *
 *   bun run scripts/h4-event-moves.ts build [out]   Gamma + CLOB -> fixtures/polymarket-event-history.json (~15 min)
 *   bun run scripts/h4-event-moves.ts score         the pre-registered test below, on that fixture and
 *                                                   fixtures/backtest-prices-2021-2025.json
 *
 * Pre-registered design (committed before the first score run, 2026-11-26):
 * - Events: the fixed title rules in `classifyEventTitle` (backtest/event-moves.ts): FOMC per-meeting decisions, US CPI
 *   prints, US payroll and unemployment reports, crypto ETF approvals. Candidates come from Gamma text search (QUERIES)
 *   and tag scans (TAGS) of closed events ending 2022-12-01..2026-03-31. Order-book markets only (the CLOB has no
 *   history for the older AMM markets). No volume filter: lifetime volume would select on the future.
 * - History: hourly CLOB prices (fidelity=60, 14-day chunks, the CLOB's range limit) from each market's createdAt to its
 *   closedTime; the YES price at 19:00Z each UTC day, from its first trade (so a market counts from creation) to the day
 *   after its last trade before close. Group and orientation come from titles and questions, never from outcomes.
 * - Rows: 14 tickers of the 2021-2025 price fixture, feature dates 2023-01-01 on. For feature date D_i: event moves
 *   between the 19:00Z cutoffs of D_{i-1} and D_i (the asset's previous trading date), HAR trailing volatility
 *   [ln|r_i|, ln rms 5d, ln rms 22d] through the D_i close; target r_{i+1} = ln(P_{i+1}/P_i). The cutoff precedes every
 *   asset's D_i close, so feature and target windows do not overlap.
 * - Features per group (fed, cpi, jobs, crypto_etf): |return| uses the largest |Δp| among the group's markets; direction
 *   uses Σ orientation × Δp (orientation +1 = YES means a higher rate, hotter inflation, a stronger labour market, an
 *   ETF approval).
 * - Models, coefficients per asset class (index: SPY QQQ VOO DIA VTI; stock: AAPL TSLA MSFT NVDA GOOGL AMZN; gold: GLD
 *   IAU; crypto: BTC-USD): |return| r ~ Laplace(0, s), ln s linear; direction logistic. Baseline = intercept + HAR terms;
 *   full = baseline + the four group features.
 * - Split: train = rows whose target date is in 2023-2024; test = rows with feature date in 2025.
 * - Tests, per target: (a) out-of-sample LR test, 1 df: the train-fitted event part of the full model, frozen as one
 *   index z, added to a baseline refit on 2025 (base vs base + γ·z, both fit on 2025); (b) frozen out-of-sample gain:
 *   mean per-row log-likelihood of full minus baseline, both fit on train, over 2025 rows, with a 95% CI from a circular
 *   block bootstrap over dates (blocks of 10 dates, 2000 reps, seed 20261126; all rows of a date move together).
 * - Kill rule: H4 survives for a target only if (a) p < 0.025 (0.05 split over the two targets) and (b) the 95% CI lies
 *   above zero. H4 is killed if neither |return| nor direction survives.
 * - Also reported, not part of the rule: the in-sample LR test on train, the learned mapping, per-asset gains with CIs,
 *   event-day counts, and the gain on event-days only.
 */
import { join } from 'node:path';
import {
  EVENT_GROUPS,
  blockBootstrapMeanCi,
  buildRows,
  chiSquareSf,
  classifyEventTitle,
  fitGlm,
  orientationWeights,
  sampleAtCutoff,
  type EventMarketHistory,
  type GlmFamily,
  type H4Row,
} from '../src/tools/finance/backtest/event-moves.js';
import { fetchWithRetry } from '../src/tools/finance/polymarket-gamma-client.js';

const GAMMA = 'https://gamma-api.polymarket.com';
const CLOB = 'https://clob.polymarket.com';
const FIXTURES = join(import.meta.dir, '..', 'src/tools/finance/fixtures');
const HISTORY = join(FIXTURES, 'polymarket-event-history.json');
const PRICES = join(FIXTURES, 'backtest-prices-2021-2025.json');

const END_MIN = '2022-12-01';
const END_MAX = '2026-03-31';
const QUERIES = ['Fed Interest Rates', 'Fed decision', 'inflation', 'jobs added', 'Unemployment Rate', 'ETF approved', 'ETF begins trading', 'Bitcoin ETF', 'Ethereum ETF', 'Solana ETF', 'SEC delay ETF'];
const TAGS = ['fed', 'fed-rates', 'interest-rates', 'inflation', 'cpi', 'economy', 'macro-indicators', 'economic-policy', 'jobs-report', 'unemployment', 'etf', 'crypto-policy'];
const CHUNK_SEC = 14 * 86_400;

const ROWS_FROM = '2023-01-01';
const TRAIN_TARGET_MAX = '2024-12-31';
const TEST_FROM = '2025-01-01';
const CLASSES: Record<string, string[]> = {
  index: ['SPY', 'QQQ', 'VOO', 'DIA', 'VTI'],
  stock: ['AAPL', 'TSLA', 'MSFT', 'NVDA', 'GOOGL', 'AMZN'],
  gold: ['GLD', 'IAU'],
  crypto: ['BTC-USD'],
};
const BLOCK = 10;
const REPS = 2000;
const SEED = 20261126;
const ALPHA = 0.025;
const EVENT_DAY_MOVE = 0.05;

interface GammaMarket {
  question?: string;
  createdAt?: string;
  endDate?: string;
  closedTime?: string;
  outcomes?: string;
  clobTokenIds?: string;
  enableOrderBook?: boolean;
}
interface GammaEvent { id: string; title: string; endDate?: string; markets?: GammaMarket[] }

/** Retries 429s, 5xx and dropped connections with backoff, and throws on anything else: a market is never skipped silently. */
async function getJson<T>(url: string): Promise<T> {
  return fetchWithRetry(async () => {
    const res = await fetch(url, { signal: AbortSignal.timeout(30_000) });
    if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`);
    return res.json() as Promise<T>;
  }, 6, [2_000, 4_000, 8_000, 16_000, 32_000, 64_000]);
}

async function candidateEvents(): Promise<GammaEvent[]> {
  const byId = new Map<string, GammaEvent>();
  for (const q of QUERIES) {
    for (let page = 1; ; page++) {
      const res = await getJson<{ events?: GammaEvent[]; pagination?: { hasMore?: boolean } }>(
        `${GAMMA}/public-search?q=${encodeURIComponent(q)}&events_status=closed&limit_per_type=50&page=${page}`);
      for (const e of res.events ?? []) byId.set(e.id, e);
      if (!res.pagination?.hasMore) break;
    }
  }
  for (const tag of TAGS) {
    for (let offset = 0; ; offset += 100) {
      const events = await getJson<GammaEvent[]>(`${GAMMA}/events?closed=true&tag_slug=${tag}&end_date_min=${END_MIN}&end_date_max=${END_MAX}&limit=100&offset=${offset}`);
      for (const e of events) byId.set(e.id, e);
      if (events.length < 100) break;
    }
  }
  return [...byId.values()].filter(e => (e.endDate ?? '') >= END_MIN && (e.endDate ?? '') <= `${END_MAX}T23:59:59Z`);
}

const isoTime = (s: string | undefined) => {
  const iso = s?.replace(' ', 'T').replace(/\+00(?::?00)?$/, 'Z');
  return iso && Number.isFinite(Date.parse(iso)) ? new Date(Date.parse(iso)).toISOString() : undefined;
};

function yesToken(m: GammaMarket): string | undefined {
  const tokens = JSON.parse(m.clobTokenIds ?? '[]') as string[];
  const yes = (JSON.parse(m.outcomes ?? '["Yes","No"]') as string[]).findIndex(o => o.toLowerCase() === 'yes');
  return tokens[yes >= 0 ? yes : 0];
}

async function hourlyHistory(token: string, createdAt: string, closedAt: string): Promise<Array<{ t: number; p: number }>> {
  const points: Array<{ t: number; p: number }> = [];
  const end = Math.floor(Date.parse(closedAt) / 1000);
  for (let start = Math.floor(Date.parse(createdAt) / 1000); start < end; start += CHUNK_SEC) {
    const res = await getJson<{ history?: Array<{ t: number; p: number }> }>(
      `${CLOB}/prices-history?market=${token}&startTs=${start}&endTs=${Math.min(start + CHUNK_SEC, end)}&fidelity=60`);
    points.push(...(res.history ?? []));
  }
  return points;
}

async function build(out: string): Promise<void> {
  const events = (await candidateEvents())
    .map(e => ({ e, group: classifyEventTitle(e.title) }))
    .filter((x): x is { e: GammaEvent; group: NonNullable<typeof x.group> } => x.group !== undefined);
  console.log(`${events.length} events match the rules`);
  const markets: EventMarketHistory[] = [];
  let done = 0;
  for (const { e, group } of events) {
    const all = e.markets ?? [];
    const weights = orientationWeights(group, all.map(m => m.question ?? ''));
    for (const [k, m] of all.entries()) {
      const token = yesToken(m);
      const createdAt = isoTime(m.createdAt);
      const closedAt = isoTime(m.closedTime) ?? isoTime(m.endDate);
      if (!m.enableOrderBook || !token || !createdAt || !closedAt || !m.question) continue;
      const daily = sampleAtCutoff(await hourlyHistory(token, createdAt, closedAt), createdAt, closedAt);
      if (Object.keys(daily).length > 0) markets.push({ group, event: e.title, question: m.question, weight: weights[k], createdAt, closedAt, daily });
    }
    if (++done % 20 === 0) console.log(`${done}/${events.length} events, ${markets.length} markets`);
  }
  markets.sort((a, b) => a.closedAt.localeCompare(b.closedAt) || a.question.localeCompare(b.question));
  await Bun.write(out, JSON.stringify({
    source: 'Polymarket Gamma closed events (search + tags, title rules in backtest/event-moves.ts) + CLOB prices-history fidelity=60, YES token, last price at or before 19:00Z each UTC day',
    builtAt: new Date().toISOString(),
    markets,
  }));
  console.log(`${markets.length} markets from ${events.length} events -> ${out}`);
}

// ---------------------------------------------------------------- scoring

const classOf = (ticker: string) => Object.keys(CLASSES).findIndex(c => CLASSES[c].includes(ticker));
const NC = Object.keys(CLASSES).length;

/** Puts `v` in its asset class's block, so one fit gives each class its own coefficients. */
function interact(v: readonly number[], cls: number): number[] {
  const x = new Array<number>(NC * v.length).fill(0);
  v.forEach((value, k) => { x[cls * v.length + k] = value; });
  return x;
}

interface TargetSpec { name: string; family: GlmFamily; y: (r: H4Row) => number; events: (r: H4Row) => number[] }
const TARGETS: TargetSpec[] = [
  { name: '|return|', family: 'laplace', y: r => r.absReturn, events: r => r.moves },
  { name: 'direction', family: 'logistic', y: r => r.up, events: r => r.signed },
];

const fmt = (x: number, d = 4) => (x >= 0 ? '+' : '') + x.toFixed(d);
const sum = (a: readonly number[]) => a.reduce((s, v) => s + v, 0);

function scoreTarget(spec: TargetSpec, train: H4Row[], test: H4Row[]) {
  const nb = train[0].base.length;
  const baseX = (r: H4Row) => interact(r.base, classOf(r.ticker));
  const fullX = (r: H4Row) => interact([...r.base, ...spec.events(r)], classOf(r.ticker));
  const yTrain = train.map(spec.y);
  const yTest = test.map(spec.y);

  const base = fitGlm(train.map(baseX), yTrain, spec.family);
  const full = fitGlm(train.map(fullX), yTrain, spec.family);
  const trainLr = 2 * (sum(train.map((r, i) => full.logLik(fullX(r), yTrain[i]))) - sum(train.map((r, i) => base.logLik(baseX(r), yTrain[i]))));
  const trainDf = Object.keys(CLASSES).flatMap((_, c) => EVENT_GROUPS.map((_, g) => train.some(r => classOf(r.ticker) === c && spec.events(r)[g] !== 0))).filter(Boolean).length;

  // (b) frozen out-of-sample gain
  const gains = test.map((r, i) => ({ date: r.targetDate, ticker: r.ticker, eventDay: Math.max(...r.moves) >= EVENT_DAY_MOVE, value: full.logLik(fullX(r), yTest[i]) - base.logLik(baseX(r), yTest[i]) }));
  const gain = sum(gains.map(g => g.value)) / gains.length;
  const gainCi = blockBootstrapMeanCi(gains, BLOCK, REPS, SEED);

  // (a) out-of-sample LR, 1 df: the train-fitted event part of the full model as one frozen index
  const eventIndex = (r: H4Row) => {
    const cls = classOf(r.ticker);
    return sum(spec.events(r).map((v, g) => v * full.beta[cls * (nb + EVENT_GROUPS.length) + nb + g]));
  };
  const test0 = fitGlm(test.map(baseX), yTest, spec.family);
  const test1 = fitGlm(test.map(r => [...baseX(r), eventIndex(r)]), yTest, spec.family);
  const oosLr = 2 * (sum(test.map((r, i) => test1.logLik([...baseX(r), eventIndex(r)], yTest[i]))) - sum(test.map((r, i) => test0.logLik(baseX(r), yTest[i]))));
  const p = chiSquareSf(oosLr, 1);
  const survives = p < ALPHA && gainCi[0] > 0;

  const eventDayGains = gains.filter(g => g.eventDay);
  const lines = [
    `### ${spec.name} (${spec.family})`,
    `- In-sample (train) LR: ${trainLr.toFixed(2)} on ${trainDf} df, p = ${chiSquareSf(trainLr, trainDf).toPrecision(3)}`,
    `- Out-of-sample LR (2025, frozen event index, 1 df): ${oosLr.toFixed(2)}, p = ${p.toPrecision(3)}, γ = ${fmt(test1.beta[test1.beta.length - 1], 3)} (1 = the train mapping holds, 0 = none)`,
    `- Out-of-sample gain, full - baseline (nats per asset-day, ${gains.length} rows): ${fmt(gain, 5)} [${fmt(gainCi[0], 5)}, ${fmt(gainCi[1], 5)}]`,
    `- Same, event-days only (max |Δp| >= ${EVENT_DAY_MOVE}, ${eventDayGains.length} rows): ${eventDayGains.length ? `${fmt(sum(eventDayGains.map(g => g.value)) / eventDayGains.length, 5)} [${blockBootstrapMeanCi(eventDayGains, BLOCK, REPS, SEED).map(x => fmt(x, 5)).join(', ')}]` : 'n/a'}`,
    `- Verdict: ${survives ? 'SURVIVES' : 'fails'} (needs p < ${ALPHA} and CI above 0)`,
    '',
    `Learned mapping (train, coefficient on each group feature, ${spec.family === 'laplace' ? 'ln scale per unit max |Δp|' : 'log-odds per unit oriented Δp'}):`,
    '',
    `| class | ${EVENT_GROUPS.join(' | ')} |`,
    `|---|${EVENT_GROUPS.map(() => '---').join('|')}|`,
    ...Object.keys(CLASSES).map((c, ci) => `| ${c} | ${EVENT_GROUPS.map((_, g) => fmt(full.beta[ci * (nb + EVENT_GROUPS.length) + nb + g], 3)).join(' | ')} |`),
    '',
    'Per asset (out-of-sample gain, nats per day, 95% block-bootstrap CI):',
    '',
    '| ticker | rows | event-days | gain | 95% CI |',
    '|---|---|---|---|---|',
    ...Object.values(CLASSES).flat().map(t => {
      const g = gains.filter(x => x.ticker === t);
      const ci = blockBootstrapMeanCi(g, BLOCK, REPS, SEED);
      return `| ${t} | ${g.length} | ${g.filter(x => x.eventDay).length} | ${fmt(sum(g.map(x => x.value)) / g.length, 5)} | [${fmt(ci[0], 5)}, ${fmt(ci[1], 5)}] |`;
    }),
    '',
  ];
  return { survives, lines };
}

async function score(): Promise<void> {
  const { markets } = await Bun.file(HISTORY).json() as { markets: EventMarketHistory[] };
  const { tickers } = await Bun.file(PRICES).json() as { tickers: Record<string, { dates: string[]; closes: number[] }> };
  const rows = Object.values(CLASSES).flat().flatMap(t => buildRows(t, tickers[t].dates, tickers[t].closes, markets, ROWS_FROM));
  const train = rows.filter(r => r.targetDate <= TRAIN_TARGET_MAX);
  const test = rows.filter(r => r.date >= TEST_FROM);

  const out = ['## H4: event-probability moves vs next-day returns', ''];
  out.push('| group | markets (events) closing 2023-24 | markets (events) closing 2025+ | train dates with a priced market | test dates |', '|---|---|---|---|---|');
  for (const group of EVENT_GROUPS) {
    const ms = markets.filter(m => m.group === group);
    const count = (pred: (m: EventMarketHistory) => boolean) => `${ms.filter(pred).length} (${new Set(ms.filter(pred).map(m => m.event)).size})`;
    const days = (rs: H4Row[]) => new Set(rs.filter(r => r.ticker === 'BTC-USD' && ms.some(m => m.daily[r.date] !== undefined)).map(r => r.date)).size;
    out.push(`| ${group} | ${count(m => m.closedAt < TEST_FROM)} | ${count(m => m.closedAt >= TEST_FROM)} | ${days(train)} | ${days(test)} |`);
  }
  const eventDays = (rs: H4Row[]) => {
    const ev = rs.filter(r => Math.max(...r.moves) >= EVENT_DAY_MOVE);
    return `${ev.length} asset-days on ${new Set(ev.map(r => r.date)).size} dates`;
  };
  out.push('', `Rows: train ${train.length} (${ROWS_FROM} .. targets ${TRAIN_TARGET_MAX}), test ${test.length} (${TEST_FROM} .. 2025-12-31).`);
  out.push(`Event-days (some group's max |Δp| >= ${EVENT_DAY_MOVE}): train ${eventDays(train)}, test ${eventDays(test)}.`, '');

  const verdicts = TARGETS.map(spec => {
    const { survives, lines } = scoreTarget(spec, train, test);
    out.push(...lines);
    return survives;
  });
  out.push(`**H4 ${verdicts.some(Boolean) ? 'SURVIVES' : 'is KILLED'}** (|return| ${verdicts[0] ? 'passes' : 'fails'}, direction ${verdicts[1] ? 'passes' : 'fails'}).`);
  console.log(out.join('\n'));
}

if (import.meta.main) {
  const [cmd, out = HISTORY] = process.argv.slice(2);
  if (cmd === 'build') await build(out);
  else if (cmd === 'score') await score();
  else console.log('usage: bun run scripts/h4-event-moves.ts build [out] | score');
}
