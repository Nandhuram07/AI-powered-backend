// src/conversation/chatAgent.ts
import inquirer from 'inquirer';
import chalk from 'chalk';
import ora from 'ora';
import { callAI } from '../utils/aiProvider.js';
import type { SystemInput, EntityDef, Relation, Feature, DbType, Field } from '../types/index.js';

interface Message {
  role:    'user' | 'assistant';
  content: string;
}

interface ExtractedSchema {
  systemName: string;
  entities:   EntityDef[];
  relations:  Relation[];
  ready:      boolean;
}

const SYSTEM_PROMPT = `You are a backend requirements analyst. Have a SHORT conversation to understand what backend system the user needs, then extract a precise schema.

Goals:
1. Understand the system name and entities
2. Infer sensible fields for each entity automatically (use your knowledge — don't ask about every field)
3. Identify relationships (foreign keys) between entities
4. Confirm with user before generating

Rules:
- Keep responses SHORT — 2-4 lines max
- Infer obvious fields (e.g. student → name, email, grade)
- Ask only ONE clarifying question per turn if needed
- Once you have enough info, show a summary and ask "Ready to generate? (yes/no)"
- When user says yes/ready/go/ok, respond with ONLY this JSON block and nothing else:

\`\`\`json
{
  "systemName": "School Management",
  "entities": [
    { "entity": "student", "fields": [{"name":"name","type":"string"},{"name":"grade","type":"string"},{"name":"email","type":"string"}] },
    { "entity": "billing", "fields": [{"name":"studentId","type":"number"},{"name":"fees","type":"number"},{"name":"status","type":"string"}] }
  ],
  "relations": [
    { "type": "belongsTo", "fromEntity": "billing", "toEntity": "student", "foreignKey": "studentId" }
  ],
  "ready": true
}
\`\`\`

Valid field types: string | number | boolean | date
Only use "belongsTo" for relations.
Never include "id" or "created_at" in fields.`;

export async function startChat(
  features: Feature[],
  db: DbType,
  outputDir: string
): Promise<SystemInput> {
  const history: Message[] = [];

  console.log(chalk.bold.cyan('\n💬 Chat Mode — describe what you need\n'));
  console.log(chalk.gray('   Describe your system in plain English.'));
  console.log(chalk.gray('   Type "exit" to cancel.\n'));
  console.log(chalk.gray('─'.repeat(50) + '\n'));

  let extracted: ExtractedSchema | null = null;

  while (!extracted) {
    // ── Get user input via inquirer ───────────────────────────
    const { userInput } = await inquirer.prompt([{
      type:    'input',
      name:    'userInput',
      message: chalk.green('You:'),
      validate: (v: string) => v.trim().length > 0 || 'Please type something',
    }]);

    const input = (userInput as string).trim();

    if (input.toLowerCase() === 'exit') {
      console.log(chalk.yellow('Cancelled.'));
      process.exit(0);
    }

    history.push({ role: 'user', content: input });

    // ── Call AI ───────────────────────────────────────────────
    const spin = ora({ text: chalk.cyan('Thinking...'), color: 'cyan' }).start();

    const conversation = history
      .map(m => `${m.role === 'user' ? 'User' : 'Assistant'}: ${m.content}`)
      .join('\n\n');

    let aiResponse: string;
    try {
      aiResponse = await callAI(SYSTEM_PROMPT, conversation);
    } catch (err: unknown) {
      spin.fail('AI call failed');
      console.log(chalk.red(err instanceof Error ? err.message : String(err)));
      history.pop();
      continue;
    }

    spin.stop();

    // ── Try to parse JSON schema ──────────────────────────────
    const jsonMatch = aiResponse.match(/```json\s*([\s\S]+?)\s*```/);
    if (jsonMatch) {
      try {
        const parsed = JSON.parse(jsonMatch[1]) as ExtractedSchema;
        if (parsed.ready && parsed.entities?.length > 0) {
          extracted = parsed;
          console.log('\n' + chalk.bold.cyan('Assistant:') + formatSummary(parsed));
          break;
        }
      } catch {
        // not valid JSON — fall through to normal response
      }
    }

    // ── Normal conversational reply ───────────────────────────
    const clean = aiResponse.replace(/```json[\s\S]*?```/g, '').trim();
    console.log('\n' + chalk.bold.cyan('Assistant:') + ' ' + clean + '\n');
    history.push({ role: 'assistant', content: clean });
  }

  // ── Build SystemInput ─────────────────────────────────────
  return {
    systemName: extracted!.systemName,
    entities:   extracted!.entities.map(e => ({
      entity: e.entity.toLowerCase().replace(/\s+/g, '_'),
      fields: validateFields(e.fields),
    })),
    relations:  extracted!.relations ?? [],
    features,
    db,
    outputDir,
  };
}

// ── Helpers ───────────────────────────────────────────────────
function validateFields(fields: Field[]): Field[] {
  const VALID = new Set(['string', 'number', 'boolean', 'date']);
  return fields
    .filter(f => f.name && !['id', 'created_at'].includes(f.name))
    .map(f => ({
      name: f.name.trim().toLowerCase(),
      type: (VALID.has(f.type) ? f.type : 'string') as Field['type'],
    }));
}

function formatSummary(schema: ExtractedSchema): string {
  const lines = [`\n  ✅ Ready to generate: ${chalk.white(schema.systemName)}\n`];
  for (const e of schema.entities) {
    const fields = e.fields.map(f => `${f.name}:${f.type}`).join(', ');
    lines.push(`  📦 ${chalk.white(e.entity)} — ${chalk.gray(fields)}`);
  }
  if (schema.relations.length > 0) {
    lines.push(`\n  🔗 ${chalk.white('Relationships:')}`);
    for (const r of schema.relations) {
      lines.push(`     ${r.fromEntity}.${r.foreignKey} → ${r.toEntity}`);
    }
  }
  lines.push('\n  Generating now...\n');
  return lines.join('\n');
}