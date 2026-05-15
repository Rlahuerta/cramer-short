import { readFileSync } from 'fs';

const lcovPath = process.argv[2] ?? 'coverage/lcov.info';
const minimumLineCoverage = Number(process.env.COVERAGE_LINES_MIN ?? '22');

function readMetric(prefix: string, content: string): number {
  const matches = content.matchAll(new RegExp(`^${prefix}:(\\d+)`, 'gm'));
  let total = 0;
  for (const match of matches) {
    total += Number(match[1]);
  }
  return total;
}

const lcov = readFileSync(lcovPath, 'utf-8');
const linesFound = readMetric('LF', lcov);
const linesHit = readMetric('LH', lcov);

if (linesFound === 0) {
  throw new Error(`Coverage file ${lcovPath} did not contain line totals`);
}

const lineCoverage = (linesHit / linesFound) * 100;
console.log(`Line coverage: ${lineCoverage.toFixed(2)}% (${linesHit}/${linesFound})`);

if (lineCoverage < minimumLineCoverage) {
  throw new Error(
    `Line coverage ${lineCoverage.toFixed(2)}% is below required ${minimumLineCoverage.toFixed(2)}%`,
  );
}
