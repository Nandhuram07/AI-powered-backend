// src/types/index.ts

// ── Field ─────────────────────────────────────────────────────
export type FieldType = 'string' | 'number' | 'boolean' | 'date';

export interface Field {
  name: string;
  type: FieldType;
}

// ── Relation ──────────────────────────────────────────────────
// e.g. billing belongsTo student via studentId
export interface Relation {
  type:       'belongsTo';   // Phase 2 supports belongsTo only
  fromEntity: string;        // entity that holds the FK  (e.g. billing)
  toEntity:   string;        // entity being referenced   (e.g. student)
  foreignKey: string;        // column name               (e.g. studentId)
}

// ── Single entity definition ──────────────────────────────────
export interface EntityDef {
  entity:   string;
  fields:   Field[];
}

// ── Full system input (Phase 2) ───────────────────────────────
export interface SystemInput {
  systemName: string;
  entities:   EntityDef[];
  relations:  Relation[];
  features:   Feature[];
  db:         DbType;
  outputDir:  string;
}

// ── Legacy single-entity input (Phase 1 compat) ───────────────
export interface AgentInput {
  entity:    string;
  fields:    Field[];
  features:  Feature[];
  db:        DbType;
  outputDir: string;
}

// ── Results ───────────────────────────────────────────────────
export interface FileResult {
  label: string;
  ok:    boolean;
  error?: string;
}

export interface AgentResult {
  ok:      number;
  fail:    number;
  results: FileResult[];
}

// ── Prompt ────────────────────────────────────────────────────
export interface PromptObject {
  system: string;
  user:   string;
}

// ── Enums ─────────────────────────────────────────────────────
export type Feature    = 'crud' | 'auth' | 'validation' | 'logging' | 'rbac';
export type DbType     = 'mysql' | 'mssql' | 'oracledb';
export type AIProvider = 'groq' | 'gemini' | 'anthropic' | 'openai';