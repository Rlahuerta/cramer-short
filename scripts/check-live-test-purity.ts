const LIVE_GLOBS = [
  'src/**/*.integration.test.ts',
  'src/**/*.e2e.test.ts',
];

function stripComments(source: string): string {
  return source
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1');
}

const files = new Set<string>();
for (const pattern of LIVE_GLOBS) {
  for await (const file of new Bun.Glob(pattern).scan('.')) {
    files.add(file);
  }
}

const offenders: string[] = [];
for (const file of [...files].sort((a, b) => a.localeCompare(b))) {
  const text = stripComments(await Bun.file(file).text());
  if (/\bmock\.module\s*\(/.test(text)) {
    offenders.push(file);
  }
}

if (offenders.length > 0) {
  console.error('mock.module() is not allowed in live integration/E2E tests:');
  for (const offender of offenders) {
    console.error(`- ${offender}`);
  }
  process.exit(1);
}

console.log('Live test purity check passed.');
