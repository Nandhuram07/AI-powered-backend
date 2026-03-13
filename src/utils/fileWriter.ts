// src/utils/fileWriter.ts
import fs from 'fs-extra';
import path from 'path';

export async function writeFile(
  outputDir: string,
  relativePath: string,
  content: string
): Promise<string> {
  const fullPath = path.join(outputDir, relativePath);
  await fs.ensureDir(path.dirname(fullPath));
  await fs.writeFile(fullPath, content.trim() + '\n', 'utf8');
  return fullPath;
}

export async function writeJSON(
  outputDir: string,
  relativePath: string,
  obj: unknown
): Promise<string> {
  const fullPath = path.join(outputDir, relativePath);
  await fs.ensureDir(path.dirname(fullPath));
  await fs.writeFile(fullPath, JSON.stringify(obj, null, 2) + '\n', 'utf8');
  return fullPath;
}