// src/runner/selfHealer.ts
// The agentic loop — run → detect error → fix → retry

import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import ora from 'ora';
import { spawn } from 'child_process';
import { startServer, syntaxCheck } from './processRunner.js';
import { parseError, type ParsedError } from './errorParser.js';
import { callAI } from '../utils/aiProvider.js';
import { stripFences, fixImports } from '../codeSanitizer.js';

export interface HealResult {
  success:  boolean;
  attempts: number;
  message:  string;
}

const MAX_ATTEMPTS = 5;

async function runNpmInstall(absDir: string): Promise<void> {
  const spin = ora('Installing dependencies...').start();
  await new Promise<void>((resolve) => {
    const child = spawn('npm', ['install'], { cwd: absDir, shell: true, stdio: 'pipe' });
    child.on('exit', () => { spin.succeed('Dependencies installed'); resolve(); });
    child.on('error', () => { spin.warn('npm install failed — continuing anyway'); resolve(); });
  });
}

export async function selfHeal(outputDir: string): Promise<HealResult> {
  const absDir = path.resolve(outputDir);

  console.log(chalk.bold.cyan('\n🔁 Self-Healing Loop Started'));
  console.log(chalk.gray(`   Max attempts: ${MAX_ATTEMPTS}`));
  console.log(chalk.gray(`   Target: ${absDir}\n`));

  // ── Run npm install first ─────────────────────────────────
  await runNpmInstall(absDir);

  for (let attempt = 1; attempt <= MAX_ATTEMPTS; attempt++) {
    console.log(chalk.bold(`\n── Attempt ${attempt}/${MAX_ATTEMPTS} ─────────────────────`));

    // ── Step 1: Syntax check first (fast) ──────────────────
    const spin = ora('Running syntax check...').start();
    const { ok: syntaxOk, output: syntaxOut } = await syntaxCheck(absDir);

    if (!syntaxOk) {
      spin.warn(chalk.yellow('Syntax errors found'));
      const parsed = parseError(syntaxOut, absDir);
      console.log(chalk.red(`  File   : ${parsed.file}`));
      console.log(chalk.red(`  Error  : ${parsed.message}`));

      // Skip static infrastructure files — AI fixing them rarely helps
    const staticFiles = ['authMiddleware', 'errorMiddleware', 'loggerMiddleware', 'db.ts', 'authRoutes'];
    const isStatic = staticFiles.some(f => parsed.file.includes(f));
    if (isStatic) {
      console.log(chalk.yellow(`  ⚠  Static file has TS errors — skipping AI fix, continuing...`));
      console.log(chalk.gray(`     Tip: Run 'npx tsc --noEmit' in output dir to see full errors`));
      break;
    }

    const fixed = await healFile(absDir, parsed, attempt);
    if (!fixed) {
      return { success: false, attempts: attempt, message: `Could not fix: ${parsed.message}` };
    }
    continue;
    }

    spin.succeed('Syntax OK');

    // ── Step 2: Actually start the server ──────────────────
    const runSpin = ora('Starting server...').start();
    const result  = await startServer(absDir, 4000);

    if (result.healthy) {
      runSpin.succeed(chalk.green('✅ Server started successfully!'));
      console.log(chalk.gray('\n  Server output:'));
      result.output.split('\n').filter(l => l.trim()).forEach(l =>
        console.log(chalk.gray('  ' + l))
      );
      return { success: true, attempts: attempt, message: 'Server started healthy' };
    }

    // ── Step 3: Server crashed — parse and fix ──────────────
    runSpin.fail(chalk.red('Server crashed'));
    const parsed = parseError(result.output, absDir);

    console.log(chalk.red(`  File   : ${parsed.file}`));
    console.log(chalk.red(`  Error  : ${parsed.message}`));
    console.log(chalk.red(`  Type   : ${parsed.errorType}`));

    // Skip fix attempt on DB errors — not a code problem
    if (parsed.errorType === 'db_error') {
      console.log(chalk.yellow('\n  ⚠  DB connection error — this is a config issue, not a code error.'));
      console.log(chalk.yellow('     Check DB_HOST, DB_USER, DB_PASSWORD, DB_NAME in .env'));
      return { success: false, attempts: attempt, message: 'DB connection failed — check .env' };
    }

    if (attempt < MAX_ATTEMPTS) {
      const fixed = await healFile(absDir, parsed, attempt);
      if (!fixed) {
        return { success: false, attempts: attempt, message: `AI could not fix: ${parsed.message}` };
      }
    }
  }

  return {
    success: false,
    attempts: MAX_ATTEMPTS,
    message: `Exceeded ${MAX_ATTEMPTS} attempts — manual fix required`,
  };
}

// ── Fix a single file using AI ────────────────────────────────
async function healFile(absDir: string, parsed: ParsedError, attempt: number): Promise<boolean> {
  const filePath = path.join(absDir, parsed.file);

  // Read the broken file
  let fileContent: string;
  try {
    fileContent = (await fs.readFile(filePath)) as unknown as string;
  } catch {
    console.log(chalk.red(`  Cannot read file: ${parsed.file}`));
    return false;
  }

  const spin = ora(chalk.magenta(`  [AI] Fixing ${parsed.file} (attempt ${attempt})...`)).start();

  const systemPrompt =
    'You are a Node.js TypeScript bug fixer.\n' +
    'You will receive a broken TypeScript file and an error message.\n' +
    'Output ONLY the fixed file content — no explanation, no markdown fences.\n' +
    'Rules:\n' +
    '- Keep all existing logic intact — only fix the reported error\n' +
    '- ESM imports only (import/export)\n' +
    '- Named imports for pool and AppError\n' +
    '- All relative imports must end with .js\n' +
    '- No TypeScript strict violations';

  const userPrompt =
    `File: ${parsed.file}\n\n` +
    `Error: ${parsed.message}\n` +
    (parsed.line ? `Line: ${parsed.line}\n` : '') +
    `\nFull error output:\n${parsed.rawOutput.slice(0, 800)}\n\n` +
    `Current file content:\n\`\`\`typescript\n${fileContent}\n\`\`\`\n\n` +
    `Output ONLY the fixed TypeScript file. No markdown. No explanation.`;

  try {
    let fixed = await callAI(systemPrompt, userPrompt);
    fixed = stripFences(fixed);

    // Extract entity name from file path for fixImports
    const entityMatch = parsed.file.match(/\/(\w+)(Model|Service|Controller|Routes|Validation)/);
    if (entityMatch) {
      const entity = entityMatch[1].toLowerCase();
      const E      = entity.charAt(0).toUpperCase() + entity.slice(1);
      fixed = fixImports(fixed, entity, E);
    }

    await fs.writeFile(filePath, Buffer.from(fixed.trim() + '\n', 'utf8'));
    spin.succeed(chalk.green(`  ✔ Fixed ${parsed.file}`));
    return true;
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    spin.fail(chalk.red(`  AI fix failed: ${msg}`));
    return false;
  }
}