// src/systemAgent.ts — Phase 2 multi-entity orchestrator
import path from 'path';
import fs from 'fs-extra';
import chalk from 'chalk';
import ora, { type Ora } from 'ora';
import { callAI } from './utils/aiProvider.js';
import { writeFile } from './utils/fileWriter.js';
import { stripFences, fixImports } from './codeSanitizer.js';
import * as T from './templates/staticTemplates.js';
import type { SystemInput, AgentResult, FileResult } from './types/index.js';
import { selfHeal } from './runner/selfHealer.js';

export async function runSystemAgent(input: SystemInput): Promise<AgentResult> {
  const { systemName, entities, relations, features, db, outputDir } = input;
  const out     = path.resolve(outputDir);
  const hasAuth = features.includes('auth');

  const results: FileResult[] = [];
  const spin: Ora = ora({ color: 'cyan' });

  // Per-entity files: model + service + controller + routes + validation = 5 each
  // Shared files: AppError, db, authMiddleware, logger, error, migrate, server,
  //               package.json, tsconfig, userModel, authRoutes = 11
  const perEntity   = entities.length * 5;
  const sharedFiles = hasAuth ? 11 : 9;
  let step = 0;
  const TOTAL = perEntity + sharedFiles;

  async function writeStatic(label: string, content: string, relPath: string): Promise<void> {
    step++;
    spin.start(chalk.cyan(`[${step}/${TOTAL}] ${label}...`));
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

  async function writeAI(
    label: string,
    promptObj: { system: string; user: string },
    relPath: string,
    entity: string
  ): Promise<void> {
    step++;
    spin.start(chalk.cyan(`[${step}/${TOTAL}] Generating ${label}...`));
    try {
      const E   = entity.charAt(0).toUpperCase() + entity.slice(1);
      let code  = await callAI(promptObj.system, promptObj.user);
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

  // ── Banner ──────────────────────────────────────────────────
  console.log(chalk.bold.cyan(`\n🤖 AI Backend Generator — "${systemName}" system on ${db}`));
  console.log(chalk.gray(`   Entities : ${entities.map(e => e.entity).join(', ')}`));
  console.log(chalk.gray(`   Relations: ${relations.length > 0 ? relations.map(r => `${r.fromEntity} → ${r.toEntity}`).join(', ') : 'none'}`));
  console.log(chalk.gray('   [static] = infrastructure   [ai] = validation\n'));

  // ── Shared infrastructure ───────────────────────────────────
  await writeStatic('AppError utility',  T.appErrorTemplate(),                          'src/utils/AppError.ts');
  await writeStatic('DB config',         T.dbTemplate(db),                              'src/config/db.ts');
  await writeStatic('Auth middleware',   T.authMiddlewareTemplate(),                    'src/middleware/authMiddleware.ts');
  await writeStatic('Logger middleware', T.loggerMiddlewareTemplate(),                  'src/middleware/loggerMiddleware.ts');
  await writeStatic('Error middleware',  T.errorMiddlewareTemplate(),                   'src/middleware/errorMiddleware.ts');
  await writeStatic('Migrate (all)',     T.multiMigrateTemplate(entities, relations, db), 'src/migrate.ts');
  await writeStatic('Server (all)',      T.multiServerTemplate(entities.map(e => e.entity), features), 'src/server.ts');
  await writeStatic('package.json',      T.packageJsonTemplate(systemName.toLowerCase().replace(/\s+/g, '-'), db), 'package.json');
  await writeStatic('tsconfig.json',     T.tsconfigTemplate(),                          'tsconfig.json');

  if (hasAuth) {
    await writeStatic('User model', T.userModelTemplate(), 'src/models/userModel.ts');
    await writeStatic('Auth routes', T.authRoutesTemplate(), 'src/routes/authRoutes.ts');
  }

  // ── Per-entity files ────────────────────────────────────────
  for (const { entity, fields } of entities) {
    const E = entity.charAt(0).toUpperCase() + entity.slice(1);
    console.log(chalk.bold.gray(`\n  ── ${E} ──`));

    await writeStatic(`${E} Model`,      T.modelTemplate(entity, fields),    `src/models/${entity}Model.ts`);
    await writeStatic(`${E} Service`,    T.serviceTemplate(entity),          `src/services/${entity}Service.ts`);
    await writeStatic(`${E} Controller`, T.controllerTemplate(entity),       `src/controllers/${entity}Controller.ts`);
    await writeStatic(`${E} Routes`,     T.routesTemplate(entity, features), `src/routes/${entity}Routes.ts`);
    await writeStatic(`${E} Validation`, T.validationTemplate(entity, fields), `src/validations/${entity}Validation.ts`);
  }

  // ── .env ────────────────────────────────────────────────────
  await createEnvFiles(out, db, spin);

  // ── README ──────────────────────────────────────────────────
  await writeSystemReadme(out, systemName, entities.map(e => e.entity), relations, hasAuth);

  // ── Phase 3: Self-healing loop ──────────────────────────────
  if (results.every((r) => r.ok)) {
    const healResult = await selfHeal(out);
    if (healResult.success) {
      console.log(chalk.green(`\n✅ Self-heal passed in ${healResult.attempts} attempt(s)`));
    } else {
      console.log(chalk.yellow(`\n⚠  Self-heal: ${healResult.message}`));
    }
  }

  const ok   = results.filter(r => r.ok).length;
  const fail = results.filter(r => !r.ok).length;
  return { ok, fail, results };
}

// ── Helpers ───────────────────────────────────────────────────
async function createEnvFiles(out: string, db: string, spin: Ora): Promise<void> {
  const port = db === 'mssql' ? '1433' : '3306';
  const content = [
    '# Fill in DB_PASSWORD and DB_NAME before running',
    'NODE_ENV=development',
    'PORT=3000',
    'DB_HOST=localhost',
    `DB_PORT=${port}`,
    'DB_USER=root',
    'DB_PASSWORD=',
    'DB_NAME=myapp',
    'JWT_SECRET=change_this_to_a_long_random_secret',
    'JWT_EXPIRES_IN=7d',
    'BCRYPT_ROUNDS=10',
  ].join('\n');

  await fs.ensureDir(out);
  await fs.writeFile(path.join(out, '.env.example'), content, 'utf8');
  const envPath = path.join(out, '.env');
  if (!(await fs.pathExists(envPath))) {
    await fs.writeFile(envPath, content, 'utf8');
    spin.succeed(chalk.green('✔  .env + .env.example') + chalk.yellow('  ← set DB_PASSWORD and DB_NAME'));
  } else {
    spin.succeed(chalk.green('✔  .env.example') + chalk.gray(' (.env exists — not overwritten)'));
  }
}

async function writeSystemReadme(
  out: string,
  systemName: string,
  entities: string[],
  relations: import('./types/index.js').Relation[],
  hasAuth: boolean
): Promise<void> {
  const lines = [
    `# ${systemName} Backend`,
    '',
    '> Generated by AI Backend Generator v3 — Phase 2 (Multi-entity)',
    '',
    '## Stack',
    '- Node.js + Express.js (TypeScript)',
    '- MySQL2 · Zod · JWT + bcryptjs',
    '',
    '## Quick Start',
    '```bash',
    'npm install',
    '# Edit .env — set DB_PASSWORD and DB_NAME',
    'npm run dev',
    '```',
    '',
    '## Entities',
    ...entities.map(e => `- **${e}s** → \`/api/${e}s\``),
    '',
    ...(relations.length > 0 ? [
      '## Relationships',
      ...relations.map(r => `- \`${r.fromEntity}.${r.foreignKey}\` → \`${r.toEntity}.id\``),
      '',
    ] : []),
    '## Auth Endpoints',
    ...(hasAuth ? [
      '| POST | /api/auth/register | public |',
      '| POST | /api/auth/login    | public |',
      '| POST | /api/auth/logout   | public |',
    ] : ['Auth disabled']),
    '',
    '| GET    | /health | public |',
  ];
  await writeFile(out, 'README.md', lines.join('\n'));
}