// src/index.ts — CLI entry point (Phase 1 + Phase 2)
import dotenv from 'dotenv';
dotenv.config();

import inquirer from 'inquirer';
import chalk from 'chalk';
import path from 'path';
import { runAgent } from './agent.js';
import type { Field, Feature, DbType, AIProvider, EntityDef, Relation } from './types/index.js';
import { startChat } from './conversation/chatAgent.js';
import { runSystemAgent } from './systemagent.js';

// ── Banner ────────────────────────────────────────────────────
console.log(chalk.bold.cyan(`
╔══════════════════════════════════════════╗
║       🤖  AI BACKEND GENERATOR  v3       ║
║   Single entity · Multi-entity systems  ║
║   TypeScript · Zod · JWT · MySQL        ║
╚══════════════════════════════════════════╝
`));

// ── Validate AI key ───────────────────────────────────────────
const provider = (process.env.AI_PROVIDER ?? 'groq').toLowerCase() as AIProvider;
const keyChecks: Record<AIProvider, { key: string; hint: string }> = {
  groq:      { key: 'GROQ_API_KEY',      hint: 'Get free key at https://console.groq.com' },
  gemini:    { key: 'GEMINI_API_KEY',    hint: 'Get key at https://aistudio.google.com' },
  anthropic: { key: 'ANTHROPIC_API_KEY', hint: 'Get key at https://console.anthropic.com' },
  openai:    { key: 'OPENAI_API_KEY',    hint: 'Get key at https://platform.openai.com' },
};
const check = keyChecks[provider];
if (check && !process.env[check.key]) {
  console.error(chalk.red(`❌  ${check.key} is not set in .env`));
  console.log(chalk.yellow(`\n${check.hint}\n`));
  process.exit(1);
}

const VALID_TYPES = new Set(['string', 'number', 'boolean', 'date']);

function parseFields(raw: string): Field[] {
  return raw.split(',')
    .map((pair: string) => {
      const [name, type = 'string'] = pair.trim().split(':');
      const fieldType = type.trim().toLowerCase();
      if (!VALID_TYPES.has(fieldType)) {
        console.warn(chalk.yellow(`  ⚠  Unknown type "${fieldType}" for "${name}" — defaulting to string`));
      }
      return {
        name: name.trim().toLowerCase(),
        type: (VALID_TYPES.has(fieldType) ? fieldType : 'string') as Field['type'],
      };
    })
    .filter((f: Field) => f.name.length > 0);
}

// ── Mode selection ────────────────────────────────────────────
const { mode } = await inquirer.prompt([{
  type: 'list',
  name: 'mode',
  message: '🔷 What do you want to generate?',
  choices: [
    { name: 'Single entity   (e.g. job, product, student)', value: 'single' },
    { name: 'Multi-entity system  (e.g. school, hospital, ecommerce)', value: 'system' },
    { name: 'Chat  — describe what you need in plain English', value: 'chat' },
  ],
}]);

// ── Shared questions ──────────────────────────────────────────
const sharedAnswers = await inquirer.prompt([
  {
    type: 'checkbox',
    name: 'features',
    message: '🔷 Features to include:',
    choices: [
      { name: 'CRUD endpoints',          value: 'crud',       checked: true },
      { name: 'JWT Authentication',       value: 'auth',       checked: true },
      { name: 'Zod Validation',           value: 'validation', checked: true },
      { name: 'Request Logging',          value: 'logging',    checked: true },
      { name: 'Role-based Authorization', value: 'rbac' },
    ],
  },
  {
    type: 'list',
    name: 'db',
    message: '🔷 Database:',
    choices: [
      { name: 'MySQL / MariaDB  (mysql2)', value: 'mysql' },
      { name: 'Microsoft SQL Server  (mssql)', value: 'mssql' },
    ],
  },
  {
    type: 'input',
    name: 'outputDir',
    message: '🔷 Output directory:',
    default: './output',
  },
]);

// ════════════════════════════════════════
// PHASE 1 — Single entity
// ════════════════════════════════════════
if (mode === 'single') {
  const answers = await inquirer.prompt([
    {
      type: 'input',
      name: 'entity',
      message: '🔷 Entity name (e.g. product, student, job):',
      validate: (v: string) => /^[a-zA-Z][a-zA-Z0-9]*$/.test(v.trim()) || 'Letters and numbers only — no underscores',
      filter: (v: string) => v.trim().toLowerCase(),
    },
    {
      type: 'input',
      name: 'rawFields',
      message: '🔷 Fields — name:type  (string | number | boolean | date)\n  Example: title:string, salary:number, active:boolean\n  >',
      validate: (v: string) => v.trim().length > 0 || 'At least one field required',
    },
  ]);

  const fields = parseFields(answers.rawFields);

  console.log(chalk.gray(`\nℹ  Entity  : ${chalk.white(answers.entity)}`));
  console.log(chalk.gray(`ℹ  Fields  : ${chalk.white(fields.map((f: Field) => `${f.name}:${f.type}`).join(', '))}`));
  console.log(chalk.gray(`ℹ  Output  : ${chalk.white(path.resolve(sharedAnswers.outputDir))}\n`));

  const { confirm } = await inquirer.prompt([{ type: 'confirm', name: 'confirm', message: '🚀 Generate?', default: true }]);
  if (!confirm) { console.log(chalk.yellow('Aborted.')); process.exit(0); }

  try {
    const { ok, fail } = await runAgent({
      entity: answers.entity, fields,
      features: sharedAnswers.features as Feature[],
      db: sharedAnswers.db as DbType,
      outputDir: sharedAnswers.outputDir,
    });
    printSummary(ok, fail, sharedAnswers.outputDir);
  } catch (err: unknown) {
    console.error(chalk.red(`\n❌ ${err instanceof Error ? err.message : String(err)}`));
    process.exit(1);
  }
}

// ════════════════════════════════════════
// PHASE 2 — Multi-entity system
// ════════════════════════════════════════
if (mode === 'system') {
  const { systemName, rawEntities } = await inquirer.prompt([
    {
      type: 'input',
      name: 'systemName',
      message: '🔷 System name (e.g. School Management, Hospital, E-Commerce):',
      validate: (v: string) => v.trim().length > 0 || 'Required',
    },
    {
      type: 'input',
      name: 'rawEntities',
      message: '🔷 Entity names — comma separated\n  Example: student, staff, billing\n  >',
      validate: (v: string) => v.trim().length > 0 || 'At least one entity required',
    },
  ]);

  const entityNames: string[] = rawEntities
    .split(',')
    .map((e: string) => e.trim().toLowerCase())
    .filter((e: string) => /^[a-zA-Z][a-zA-Z0-9]*$/.test(e));

  // ── Define fields for each entity ────────────────────────────
  const entityDefs: EntityDef[] = [];
  for (const entityName of entityNames) {
    const { rawFields } = await inquirer.prompt([{
      type: 'input',
      name: 'rawFields',
      message: `🔷 Fields for "${entityName}" (name:type, ...)\n  >`,
      validate: (v: string) => v.trim().length > 0 || 'At least one field required',
    }]);
    entityDefs.push({ entity: entityName, fields: parseFields(rawFields) });
  }

  // ── Define relationships ──────────────────────────────────────
  const relations: Relation[] = [];
  const { addRelations } = await inquirer.prompt([{
    type: 'confirm',
    name: 'addRelations',
    message: '🔷 Add relationships between entities? (e.g. billing belongs to student)',
    default: entityNames.length > 1,
  }]);

  if (addRelations) {
    let addMore = true;
    while (addMore) {
      const rel = await inquirer.prompt([
        {
          type: 'list',
          name: 'fromEntity',
          message: '  Which entity holds the foreign key?',
          choices: entityNames,
        },
        {
          type: 'list',
          name: 'toEntity',
          message: '  Which entity does it reference?',
          choices: entityNames,
        },
        {
          type: 'input',
          name: 'foreignKey',
          message: '  Foreign key column name (e.g. studentId):',
          validate: (v: string) => v.trim().length > 0 || 'Required',
        },
      ]);

      relations.push({
        type: 'belongsTo',
        fromEntity: rel.fromEntity,
        toEntity: rel.toEntity,
        foreignKey: rel.foreignKey.trim(),
      });

      const { more } = await inquirer.prompt([{
        type: 'confirm', name: 'more', message: '  Add another relationship?', default: false,
      }]);
      addMore = more;
    }
  }

  // ── Summary ──────────────────────────────────────────────────
  console.log(chalk.gray(`\nℹ  System   : ${chalk.white(systemName)}`));
  console.log(chalk.gray(`ℹ  Entities : ${chalk.white(entityNames.join(', '))}`));
  if (relations.length > 0) {
    console.log(chalk.gray(`ℹ  Relations: ${chalk.white(relations.map(r => `${r.fromEntity}.${r.foreignKey} → ${r.toEntity}`).join(', '))}`));
  }
  console.log(chalk.gray(`ℹ  Output   : ${chalk.white(path.resolve(sharedAnswers.outputDir))}\n`));

  const { confirm } = await inquirer.prompt([{ type: 'confirm', name: 'confirm', message: '🚀 Generate system?', default: true }]);
  if (!confirm) { console.log(chalk.yellow('Aborted.')); process.exit(0); }

  try {
    const { ok, fail } = await runSystemAgent({
      systemName,
      entities: entityDefs,
      relations,
      features: sharedAnswers.features as Feature[],
      db: sharedAnswers.db as DbType,
      outputDir: sharedAnswers.outputDir,
    });
    printSummary(ok, fail, sharedAnswers.outputDir);
  } catch (err: unknown) {
    console.error(chalk.red(`\n❌ ${err instanceof Error ? err.message : String(err)}`));
    process.exit(1);
  }
}

// ════════════════════════════════════════
// PHASE 4 — Chat mode
// ════════════════════════════════════════
if (mode === 'chat') {
  try {
    const systemInput = await startChat(
      sharedAnswers.features as Feature[],
      sharedAnswers.db as DbType,
      sharedAnswers.outputDir,
    );

    console.log(chalk.gray(`\nℹ  System   : ${chalk.white(systemInput.systemName)}`));
    console.log(chalk.gray(`ℹ  Entities : ${chalk.white(systemInput.entities.map(e => e.entity).join(', '))}`));
    console.log(chalk.gray(`ℹ  Output   : ${chalk.white(path.resolve(sharedAnswers.outputDir))}\n`));

    const { ok, fail } = await runSystemAgent(systemInput);
    printSummary(ok, fail, sharedAnswers.outputDir);
  } catch (err: unknown) {
    console.error(chalk.red(`\n❌ ${err instanceof Error ? err.message : String(err)}`));
    process.exit(1);
  }
}

// ── Shared summary printer ────────────────────────────────────
function printSummary(ok: number, fail: number, outputDir: string): void {
  console.log(chalk.bold('\n╔════════════════════════════════════╗'));
  console.log(chalk.bold('║       Generation Complete  v3      ║'));
  console.log(chalk.bold('╚════════════════════════════════════╝'));
  console.log(chalk.green(`  ✔  ${ok} files generated successfully`));
  if (fail > 0) console.log(chalk.red(`  ✖  ${fail} files failed`));
  console.log(chalk.bold.cyan('\n📂  Backend at: ' + path.resolve(outputDir)));
  console.log(chalk.bold.cyan('\n▶  Next steps:'));
  console.log(`   cd ${outputDir}`);
  console.log('   edit .env  (set DB_PASSWORD + DB_NAME)');
  console.log('   npm install');
  console.log('   npm run dev\n');
}