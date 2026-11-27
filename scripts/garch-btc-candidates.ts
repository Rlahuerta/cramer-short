/**
 * GARCH on live BTC: paired walk-forward measurements behind two decisions in the price-prediction plan
 * (docs/reviews/price-prediction-improvements-plan-2026-11-22.md, untracked): "GARCH filtered variance" and
 * "GARCH candidates after the filtered-variance merge". Each rule was written in the plan before its runs.
 *
 * BTC-USD model-only, btcLiveCell(h) (now GARCH off, as live), h = 1/2/3/7/14, origins 2021-09..2026-11, stride 1,
 * seed 42. Overrides are merged into the cell config, e.g. '{"enableGarchVol":true}' reproduces the GARCH-on arm.
 * (Candidate C1, a symmetric clamp, used a switch that was removed after it failed.)
 *
 *   bun run scripts/garch-btc-candidates.ts run <treeRoot> <outDir> <h> [configOverridesJson]
 *   bun run scripts/garch-btc-candidates.ts score <currentDir> <candidateDir>
 */
import { join } from 'node:path';

interface Closes { dates: string[]; closes: number[] }
interface Row { date: string; p: number; y: number; realized: number; lo: number[]; hi: number[] }
interface Pair { h: number; date: string; b: Row; a: Row }
const LEVELS = [0.9, 0.95, 0.99];
const HORIZONS = [1, 2, 3, 7, 14];
const SEAM = '2026-01-01';

async function run(root: string, out: string, h: number, overrides: Record<string, unknown>) {
  const fx = async (n: string) =>
    ((await Bun.file(join(root, 'src/tools/finance/fixtures', n)).json()) as { tickers: Record<string, Closes> }).tickers['BTC-USD']!;
  const early = await fx('backtest-prices-2021-2025.json');
  const late = await fx('crypto-prices-2025-2026.json');
  const pairs = [
    ...early.dates.flatMap((d, i) => (d < SEAM ? [[d, early.closes[i]!] as const] : [])),
    ...late.dates.flatMap((d, i) => (d >= SEAM ? [[d, late.closes[i]!] as const] : [])),
  ];
  const dates = pairs.map(([d]) => d);
  const closes = pairs.map(([, c]) => c);
  const { btcLiveCell } = await import(join(root, 'src/tools/finance/backtest/live-config-benchmark.ts'));
  const { walkForward } = await import(join(root, 'src/tools/finance/backtest/walk-forward.ts'));
  const cell = btcLiveCell(h);
  const res = await walkForward({
    ...cell.config, ...overrides, ticker: 'BTC-USD', prices: closes, horizon: h, warmup: cell.maxHistoryPrices, stride: 1,
    maxHistoryPrices: cell.maxHistoryPrices, randomState: 42,
  });
  if (res.errors.length) throw new Error(`${h}d: ${res.errors.length} errors, first ${res.errors[0].error}`);
  const rows: Row[] = res.steps.map((s: any) => ({
    date: dates[s.t]!, p: s.predictedProb, y: s.actualBinary, realized: s.realizedPrice,
    lo: [s.central90CiLower, s.central95CiLower, s.central99CiLower], hi: [s.central90CiUpper, s.central95CiUpper, s.central99CiUpper],
  }));
  await Bun.write(join(out, `BTC-${h}.json`), JSON.stringify(rows));
  console.log(`${h}d ${rows.length} origins ${rows[0]?.date}..${rows.at(-1)?.date} overrides=${JSON.stringify(overrides)}`);
}

const iscore = (r: Row) => LEVELS.reduce((s, lv, i) => {
  const a = 1 - lv, l = Math.log(r.lo[i]!), u = Math.log(r.hi[i]!), y = Math.log(r.realized);
  return s + (u - l) + (2 / a) * Math.max(0, l - y) + (2 / a) * Math.max(0, y - u);
}, 0) / LEVELS.length;
const inside = (r: Row, i: number) => (r.realized >= r.lo[i]! && r.realized <= r.hi[i]! ? 1 : 0);
const mean = (v: number[]) => v.reduce((x, y) => x + y, 0) / v.length;

/** 95% CI of stat over circular 28-day blocks of dates; every row of a date moves with it. */
function blockCi(set: Pair[], stat: (s: Pair[]) => number, reps = 1000, block = 28): [number, number] {
  const byDate = new Map<string, Pair[]>();
  for (const p of set) (byDate.get(p.date) ?? byDate.set(p.date, []).get(p.date)!).push(p);
  const days = [...byDate.keys()].sort().map(d => byDate.get(d)!);
  let seed = 7;
  const rand = () => (seed = (Math.imul(seed, 1664525) + 1013904223) >>> 0) / 4294967296;
  const draws: number[] = [];
  for (let k = 0; k < reps; k++) {
    const sample: Pair[] = [];
    while (sample.length < set.length) {
      const start = Math.floor(rand() * days.length);
      for (let j = 0; j < block && sample.length < set.length; j++) sample.push(...days[(start + j) % days.length]!);
    }
    draws.push(stat(sample));
  }
  draws.sort((x, y) => x - y);
  return [draws[Math.floor(0.025 * reps)]!, draws[Math.floor(0.975 * reps)]!];
}

async function score(current: string, candidate: string) {
  const all: Pair[] = [];
  for (const h of HORIZONS) {
    const b: Row[] = await Bun.file(join(current, `BTC-${h}.json`)).json();
    const a: Row[] = await Bun.file(join(candidate, `BTC-${h}.json`)).json();
    const am = new Map(a.map(r => [r.date, r]));
    for (const r of b) { const x = am.get(r.date); if (x) all.push({ h, date: r.date, b: r, a: x }); }
  }
  const f = (x: number, d = 4) => (x >= 0 ? '+' : '') + x.toFixed(d);
  const dIs = (s: Pair[]) => mean(s.map(p => iscore(p.a) - iscore(p.b)));
  const dBr = (s: Pair[]) => mean(s.map(p => (p.a.p - p.a.y) ** 2 - (p.b.p - p.b.y) ** 2));
  const cov = (s: Pair[], side: 'a' | 'b', i: number) => mean(s.map(p => inside(p[side], i)));
  const dMiss = (i: number) => (s: Pair[]) => Math.abs(cov(s, 'a', i) - LEVELS[i]!) - Math.abs(cov(s, 'b', i) - LEVELS[i]!);

  const isCi = blockCi(all, dIs), brCi = blockCi(all, dBr);
  console.log(`pooled n=${all.length}: IS ${mean(all.map(p => iscore(p.b))).toFixed(4)} -> ${mean(all.map(p => iscore(p.a))).toFixed(4)}, ` +
    `d ${f(dIs(all))} [${f(isCi[0])}, ${f(isCi[1])}]  Brier d ${f(dBr(all), 5)} [${f(brCi[0], 5)}, ${f(brCi[1], 5)}]`);
  let guardCov = true, guardMiss = true;
  LEVELS.forEach((lv, i) => {
    const cCi = blockCi(all, s => cov(s, 'a', i)), mCi = blockCi(all, dMiss(i));
    const covOk = cCi[0] <= lv && lv <= cCi[1], missOk = mCi[0] <= 0;
    guardCov &&= covOk; guardMiss &&= missOk;
    console.log(`  ${lv * 100}%: coverage ${(100 * cov(all, 'b', i)).toFixed(1)} -> ${(100 * cov(all, 'a', i)).toFixed(1)} ` +
      `[${(100 * cCi[0]).toFixed(1)}, ${(100 * cCi[1]).toFixed(1)}] ${covOk ? 'contains nominal' : 'EXCLUDES nominal'}; ` +
      `|miss| change ${f(100 * dMiss(i)(all), 2)}pp [${f(100 * mCi[0], 2)}, ${f(100 * mCi[1], 2)}]${missOk ? '' : ' ABOVE 0'}`);
  });
  for (const h of HORIZONS) {
    const s = all.filter(p => p.h === h), ci = blockCi(s, dIs);
    console.log(`  ${h}d: IS d ${f(dIs(s))} [${f(ci[0])}, ${f(ci[1])}]  Brier d ${f(dBr(s), 5)}  cov90 ${(100 * cov(s, 'b', 0)).toFixed(1)} -> ${(100 * cov(s, 'a', 0)).toFixed(1)}`);
  }
  let yearsNoWorse = 0;
  for (const y of ['2021', '2022', '2023', '2024', '2025', '2026']) {
    const s = all.filter(p => p.date.startsWith(y)), d = dIs(s);
    if (y >= '2023' && d <= 0) yearsNoWorse++;
    console.log(`  ${y}: IS d ${f(d)}  cov ${LEVELS.map((_, i) => `${(100 * cov(s, 'b', i)).toFixed(1)}->${(100 * cov(s, 'a', i)).toFixed(1)}`).join(' / ')}`);
  }
  const primary = isCi[1] < 0, brierOk = brCi[1] <= 0.0005, yearsOk = yearsNoWorse >= 3;
  console.log(`RULE: primary ${primary ? 'pass' : 'FAIL'}; coverage guard ${guardCov ? 'pass' : 'FAIL'}; |miss| guard ${guardMiss ? 'pass' : 'FAIL'}; ` +
    `Brier guard ${brierOk ? 'pass' : 'FAIL'}; years no worse 2023-26: ${yearsNoWorse}/4 ${yearsOk ? 'pass' : 'FAIL'} => ` +
    `${primary && guardCov && guardMiss && brierOk && yearsOk ? 'PASSES' : 'fails'}`);
}

const [cmd, a1, a2, a3, a4] = process.argv.slice(2);
if (cmd === 'run') await run(a1!, a2!, Number(a3), a4 ? JSON.parse(a4) : {});
else if (cmd === 'score') await score(a1!, a2!);
else throw new Error('usage: run <root> <out> <h> [overridesJson] | score <current> <candidate>');
