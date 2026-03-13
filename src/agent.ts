// src/agent.ts
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import ora, { type Ora } from 'ora';
import { callAI } from './utils/aiProvider.js';
import { writeFile } from './utils/fileWriter.js';
import { stripFences, fixImports } from './codeSanitizer.js';
import * as T from './templates/staticTemplates.js';
import * as P from './templates/prompts.js';
import type { AgentInput, AgentResult, FileResult } from './types/index.js';
import { selfHeal } from './runner/selfHealer.js';

export async function runAgent(input: AgentInput): Promise<AgentResult> {
  const { entity, fields, features, db, outputDir } = input;
  const E   = entity.charAt(0).toUpperCase() + entity.slice(1);
  const out = path.resolve(outputDir);

  const results: FileResult[] = [];
  let step = 0;
  const TOTAL = features.includes('auth') ? 16 : 14;

  const spin: Ora = ora({ color: 'cyan' });

  // ── Write static file ───────────────────────────────────────
  async function writeStatic(label: string, content: string, relPath: string): Promise<void> {
    step++;
    spin.start(chalk.cyan(`[${step}/${TOTAL}] Writing ${label}...`));
    try {
      await writeFile(out, relPath, content);
      spin.succeed(chalk.green(`✔  ${label}`) + chalk.gray(` → ${relPath}`) + chalk.blue(' [static]'));
      results.push({ label, ok: true });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      spin.fail(chalk.red(`✖  ${label}: ${msg}`));
      results.push({ label, ok: false, error: msg });
    }
  }

  // ── Write AI-generated file ─────────────────────────────────
  async function writeAI(
    label: string,
    promptObj: { system: string; user: string },
    relPath: string
  ): Promise<void> {
    step++;
    spin.start(chalk.cyan(`[${step}/${TOTAL}] Generating ${label}...`));
    try {
      let code = await callAI(promptObj.system, promptObj.user);
      code = stripFences(code);
      code = fixImports(code, entity, E);
      await writeFile(out, relPath, code);
      spin.succeed(chalk.green(`✔  ${label}`) + chalk.gray(` → ${relPath}`) + chalk.magenta(' [ai]'));
      results.push({ label, ok: true });
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      spin.fail(chalk.red(`✖  ${label}: ${msg}`));
      results.push({ label, ok: false, error: msg });
    }
  }

  console.log(chalk.bold.cyan(`\n🤖 AI Backend Generator — "${E}" on ${db}\n`));
  console.log(chalk.gray('  [static] = infrastructure (db, middleware, server)'));
  console.log(chalk.gray('  [ai]     = business logic (validation, auth routes)\n'));

  // ── Static — infrastructure ─────────────────────────────────
  await writeStatic('AppError utility',  T.appErrorTemplate(),                    'src/utils/AppError.ts');
  await writeStatic('DB config',         T.dbTemplate(db),                        'src/config/db.ts');
  await writeStatic('Auth middleware',   T.authMiddlewareTemplate(),              'src/middleware/authMiddleware.ts');
  await writeStatic('Logger middleware', T.loggerMiddlewareTemplate(),            'src/middleware/loggerMiddleware.ts');
  await writeStatic('Error middleware',  T.errorMiddlewareTemplate(),             'src/middleware/errorMiddleware.ts');
  await writeStatic('Migrate',          T.migrateTemplate(entity, fields, db),   'src/migrate.ts');
  await writeStatic('Server entry',     T.serverTemplate(entity, features),      'src/server.ts');
  await writeStatic('package.json',     T.packageJsonTemplate(entity, db),       'package.json');
  await writeStatic('tsconfig.json',    T.tsconfigTemplate(),                    'tsconfig.json');

  // ── Static — business logic ─────────────────────────────────
  await writeStatic('Model',      T.modelTemplate(entity, fields),      `src/models/${entity}Model.ts`);
  await writeStatic('Service',    T.serviceTemplate(entity),            `src/services/${entity}Service.ts`);
  await writeStatic('Controller', T.controllerTemplate(entity),         `src/controllers/${entity}Controller.ts`);
  await writeStatic('Routes',     T.routesTemplate(entity, features),   `src/routes/${entity}Routes.ts`);

  // ── AI — only validation (field-specific Zod rules) ─────────
  await writeStatic('Validation (Zod)', T.validationTemplate(entity, fields), `src/validations/${entity}Validation.ts`);

  if (features.includes('auth')) {
    await writeStatic('User model', T.userModelTemplate(), 'src/models/userModel.ts');
    await writeAI('Auth routes', P.authRoutePrompt(entity), 'src/routes/authRoutes.ts');
  }

  // ── .env files ──────────────────────────────────────────────
  await createEnvFiles(out, db, spin);

  // ── README ──────────────────────────────────────────────────
  await writeReadme(out, entity, features);

  // ── Phase 3: Self-healing loop ──────────────────────────────
  if (results.every((r) => r.ok)) {
    const healResult = await selfHeal(out);
    if (healResult.success) {
      console.log(chalk.green(`\n✅ Self-heal passed in ${healResult.attempts} attempt(s)`));
    } else {
      console.log(chalk.yellow(`\n⚠  Self-heal: ${healResult.message}`));
    }
  }

  const ok   = results.filter((r) => r.ok).length;
  const fail = results.filter((r) => !r.ok).length;
  return { ok, fail, results };
}

// ── Helpers ───────────────────────────────────────────────────
async function createEnvFiles(out: string, db: string, spin: Ora): Promise<void> {
  const port = db === 'mssql' ? '1433' : db === 'oracledb' ? '1521' : '3306';
  const content = [
    '# Fill in DB_PASSWORD and DB_NAME before running',
    'NODE_ENV=development',
    'PORT=3000',
    'DB_HOST=localhost',
    `DB_PORT=${port}`,
    'DB_USER=',
    'DB_PASSWORD=',
    'DB_NAME=',
    'JWT_SECRET=change_this_to_a_long_random_secret',
    'JWT_EXPIRES_IN=7d',
    'BCRYPT_ROUNDS=10',
  ].join('\n');

  await fs.ensureDir(out);
  await fs.writeFile(path.join(out, '.env.example'), content, 'utf8');
  const envPath = path.join(out, '.env');
  const exists  = await fs.pathExists(envPath);
  if (!exists) {
    await fs.writeFile(envPath, content, 'utf8');
    spin.succeed(chalk.green('✔  .env + .env.example') + chalk.yellow('  ← set DB_PASSWORD and DB_NAME'));
  } else {
    spin.succeed(chalk.green('✔  .env.example updated') + chalk.gray(' (.env exists — not overwritten)'));
  }
}

async function writeReadme(out: string, entity: string, features: string[]): Promise<void> {
  const E    = entity.charAt(0).toUpperCase() + entity.slice(1);
  const auth = features.includes('auth');
  const lines = [
    `# ${E} Backend`,
    '',
    '> Generated by AI Backend Generator v3 (TypeScript)',
    '',
    '## Stack',
    '- Node.js + Express.js (TypeScript)',
    '- Validation: Zod',
    '- Auth: JWT + bcryptjs',
    '',
    '## Quick Start',
    '```bash',
    'npm install',
    '# Edit .env — set DB_PASSWORD and DB_NAME',
    'npm run dev',
    '```',
    '',
    '## Endpoints',
    '| Method | Path | Auth |',
    '|--------|------|------|',
    `| GET    | /api/${entity}s       | ${auth ? 'Bearer token' : 'None'} |`,
    `| GET    | /api/${entity}s/:id   | ${auth ? 'Bearer token' : 'None'} |`,
    `| POST   | /api/${entity}s       | ${auth ? 'Bearer token' : 'None'} |`,
    `| PUT    | /api/${entity}s/:id   | ${auth ? 'Admin token' : 'None'} |`,
    `| DELETE | /api/${entity}s/:id   | ${auth ? 'Admin token' : 'None'} |`,
    ...(auth ? [
      '| POST   | /api/auth/register | None |',
      '| POST   | /api/auth/login    | None |',
      '| POST   | /api/auth/logout   | None |',
    ] : []),
    '| GET    | /health            | None |',
  ];
  await writeFile(out, 'README.md', lines.join('\n'));
}