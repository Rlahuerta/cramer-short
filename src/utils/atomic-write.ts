import { randomBytes } from 'node:crypto';
import { mkdir, rename, unlink, writeFile } from 'node:fs/promises';
import { dirname } from 'node:path';

export interface AtomicWriteFileOptions {
  encoding?: BufferEncoding;
  mode?: number;
  ensureDir?: boolean;
}

export async function atomicWriteFile(
  filePath: string,
  data: string | Uint8Array,
  options: AtomicWriteFileOptions = {},
): Promise<void> {
  const { encoding = 'utf-8', mode, ensureDir = true } = options;
  if (ensureDir) {
    await mkdir(dirname(filePath), { recursive: true });
  }

  const tmpPath = `${filePath}.${randomBytes(3).toString('hex')}.tmp`;
  try {
    const writeOptions = typeof data === 'string' ? { encoding, mode } : { mode };
    await writeFile(tmpPath, data, writeOptions);
    await rename(tmpPath, filePath);
  } catch (error) {
    try {
      await unlink(tmpPath);
    } catch {
      // Ignore cleanup failures; preserve the original write/rename error.
    }
    throw error;
  }
}
