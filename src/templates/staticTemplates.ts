// src/templates/staticTemplates.ts
import type { Field, DbType, Feature } from '../types/index.js';

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function sqlType(type: string, fieldName: string = ''): string {
  if (type === 'boolean') return 'TINYINT(1) DEFAULT 0';
  if (type === 'date')    return 'DATE';
  if (type === 'number') {
    // phone/mobile/zip/postal fields need BIGINT to avoid overflow
    const big = ['phone', 'mobile', 'zip', 'postal', 'pincode', 'zipcode'];
    if (big.some((k) => fieldName.toLowerCase().includes(k))) return 'BIGINT';
    return 'INT';
  }
  return 'VARCHAR(255)';
}

// ── AppError ──────────────────────────────────────────────────
export function appErrorTemplate(): string {
  return [
    "export class AppError extends Error {",
    "  constructor(message: string, public statusCode: number) {",
    "    super(message);",
    "    this.statusCode = statusCode;",
    "    Object.defineProperty(this, 'isOperational', { value: true });",
    "    Error.captureStackTrace(this, this.constructor);",
    "  }",
    "}",
    "",
    "export const notFound     = (msg?: string) => new AppError(msg ?? 'Not found', 404);",
    "export const badRequest   = (msg?: string) => new AppError(msg ?? 'Bad request', 400);",
    "export const unauthorized = (msg?: string) => new AppError(msg ?? 'Unauthorized', 401);",
    "export const forbidden    = (msg?: string) => new AppError(msg ?? 'Forbidden', 403);",
  ].join('\n');
}

// ── DB Config ─────────────────────────────────────────────────
export function dbTemplate(db: DbType): string {
  if (db === 'mysql') {
    return [
      "import 'dotenv/config';",
      "import mysql from 'mysql2/promise';",
      "",
      "process.setMaxListeners(20);",
      "",
      "export const pool = mysql.createPool({",
      "  host:             process.env.DB_HOST || 'localhost',",
      "  port:             Number(process.env.DB_PORT) || 3306,",
      "  user:             process.env.DB_USER || 'root',",
      "  password:         process.env.DB_PASSWORD || '',",
      "  database:         process.env.DB_NAME || 'myapp',",
      "  waitForConnections: true,",
      "  connectionLimit:  10,",
      "});",
      "",
      "export async function connectDB(): Promise<void> {",
      "  try {",
      "    const conn = await pool.getConnection();",
      "    console.log('MySQL connected to ' + process.env.DB_NAME);",
      "    conn.release();",
      "  } catch (err: any) {",
      "    console.error('MySQL connection failed: ' + err.message);",
      "    process.exit(1);",
      "  }",
      "}",
      "",
      "process.on('SIGINT',  async () => { await pool.end(); process.exit(0); });",
      "process.on('SIGTERM', async () => { await pool.end(); process.exit(0); });",
    ].join('\n');
  }
  // mssql fallback
  return [
    "import 'dotenv/config';",
    "import sql from 'mssql';",
    "",
    "const config = {",
    "  server:   process.env.DB_HOST || 'localhost',",
    "  port:     Number(process.env.DB_PORT) || 1433,",
    "  user:     process.env.DB_USER,",
    "  password: process.env.DB_PASSWORD,",
    "  database: process.env.DB_NAME,",
    "  options:  { encrypt: false, trustServerCertificate: true },",
    "  pool:     { max: 10, min: 0, idleTimeoutMillis: 30000 },",
    "};",
    "",
    "export let pool: sql.ConnectionPool;",
    "",
    "export async function connectDB(): Promise<void> {",
    "  try {",
    "    pool = await sql.connect(config);",
    "    console.log('MSSQL connected to ' + process.env.DB_NAME);",
    "  } catch (err: any) {",
    "    console.error('MSSQL connection failed: ' + err.message);",
    "    process.exit(1);",
    "  }",
    "}",
  ].join('\n');
}

// ── Auth Middleware ───────────────────────────────────────────
export function authMiddlewareTemplate(): string {
  return [
    "import jwt, { type JwtPayload } from 'jsonwebtoken';",
    "import type { Request, Response, NextFunction } from 'express';",
    "",
    "const SECRET  = process.env.JWT_SECRET ?? 'changeme';",
    "const EXPIRES = process.env.JWT_EXPIRES_IN ?? '7d';",
    "",
    "export interface AuthRequest extends Request {",
    "  user?: JwtPayload & { id: number; email: string; role: string };",
    "}",
    "",
    "export function generateToken(payload: object): string {",
    "  return jwt.sign(payload, SECRET, { expiresIn: EXPIRES } as jwt.SignOptions);",
    "}",
    "",
    "export function authenticate(req: AuthRequest, res: Response, next: NextFunction): void {",
    "  const header = req.headers['authorization'] ?? '';",
    "  if (!header.startsWith('Bearer ')) {",
    "    res.status(401).json({ success: false, message: 'No token provided' }); return;",
    "  }",
    "  try {",
    "    req.user = jwt.verify(header.split(' ')[1], SECRET) as AuthRequest['user'];",
    "    next();",
    "  } catch {",
    "    res.status(401).json({ success: false, message: 'Invalid or expired token' });",
    "  }",
    "}",
    "",
    "export function authorize(...roles: string[]) {",
    "  return function(req: AuthRequest, res: Response, next: NextFunction): void {",
    "    if (!req.user) { res.status(401).json({ success: false, message: 'Not authenticated' }); return; }",
    "    if (!roles.includes(req.user.role)) { res.status(403).json({ success: false, message: 'Forbidden' }); return; }",
    "    next();",
    "  };",
    "}",
  ].join('\n');
}

// ── Logger Middleware ─────────────────────────────────────────
export function loggerMiddlewareTemplate(): string {
  return [
    "export function requestLogger(req: any, res: any, next: any): void {",
    "  const start = Date.now();",
    "  res.on('finish', () => {",
    "    const ms    = Date.now() - start;",
    "    const code  = res.statusCode;",
    "    const color = code >= 500 ? '\\x1b[31m' : code >= 400 ? '\\x1b[33m' : '\\x1b[32m';",
    "    const reset = '\\x1b[0m';",
    "    const date  = new Date().toISOString().replace('T', ' ').slice(0, 19);",
    "    console.log(`[${date}] ${req.method} ${req.url} ${color}${code}${reset} ${ms}ms`);",
    "  });",
    "  next();",
    "}",
  ].join('\n');
}

// ── Error Middleware ──────────────────────────────────────────
export function errorMiddlewareTemplate(): string {
  return [
    "import { AppError } from '../utils/AppError.js';",
    "",
    "export default function errorHandler(err: any, _req: any, res: any, _next: any): void {",
    "  const isDev = process.env.NODE_ENV === 'development';",
    "  if (err.name === 'ZodError') {",
    "    res.status(400).json({ success: false, message: 'Validation failed', errors: err.errors }); return;",
    "  }",
    "  if (err.name === 'JsonWebTokenError' || err.name === 'TokenExpiredError') {",
    "    res.status(401).json({ success: false, message: 'Invalid or expired token' }); return;",
    "  }",
    "  if (err instanceof AppError) {",
    "    res.status(err.statusCode).json({ success: false, message: err.message }); return;",
    "  }",
    "  console.error('Unexpected error:', err);",
    "  res.status(500).json({ success: false, message: 'Internal server error', ...(isDev && { stack: err.stack }) });",
    "}",
  ].join('\n');
}

// ── Entity Model ──────────────────────────────────────────────
export function modelTemplate(entity: string, fields: Field[]): string {
  const E        = cap(entity);
  const cols     = fields.map((f) => f.name);
  const colList  = cols.join(', ');
  const holders  = cols.map(() => '?').join(', ');
  const valList  = cols.map((c) => {
    const f = fields.find((x) => x.name === c)!;
    return f.type === 'number'
      ? `data.${c} !== undefined ? data.${c} : null`
      : `data.${c} || null`;
  }).join(', ');

  const strFields = fields.filter((f) => f.type === 'string').map((f) => f.name);
  const searchLikes = strFields.map((f) => `LOWER(${f}) LIKE LOWER(?)`).join(' OR ');
  const searchPushes = strFields.map(() =>
    "    params.push('%' + filters.search + '%');"
  ).join('\n');
  const filterConditions = cols.map((c) =>
    `  if (filters.${c} !== undefined) { conditions.push('${c} = ?'); params.push(filters.${c}); }`
  ).join('\n');

  return [
    "import { pool } from '../config/db.js';",
    "",
    `export class ${E}Model {`,
    "",
    "  static async create(data: Record<string, any>) {",
    `    const [result]: any = await pool.execute(`,
    `      'INSERT INTO ${entity}s (${colList}) VALUES (${holders})',`,
    `      [${valList}]`,
    "    );",
    `    return ${E}Model.findById(result.insertId);`,
    "  }",
    "",
    "  static async findById(id: number) {",
    `    const [rows]: any = await pool.execute('SELECT * FROM ${entity}s WHERE id = ?', [id]);`,
    "    return rows[0] ?? null;",
    "  }",
    "",
    "  static async findAll(filters: Record<string, any>, page: number, limit: number) {",
    "    const pageNum   = Number(page)  || 1;",
    "    const limitNum  = Number(limit) || 10;",
    "    const offsetNum = (pageNum - 1) * limitNum;",
    "    const conditions: string[] = [];",
    "    const params: any[] = [];",
    "",
    `    if (filters.search) {`,
    strFields.length > 0 ? `      conditions.push('(${searchLikes})');` : '      // no string fields to search',
    searchPushes,
    "    }",
    "",
    filterConditions,
    "",
    "    const where = conditions.length ? ' WHERE ' + conditions.join(' AND ') : '';",
    `    const [[{ total }]]: any = await pool.query('SELECT COUNT(*) as total FROM ${entity}s' + where, params);`,
    `    const [rows]: any = await pool.query('SELECT * FROM ${entity}s' + where + ' LIMIT ' + limitNum + ' OFFSET ' + offsetNum, params);`,
    "    return { rows, total: Number(total) };",
    "  }",
    "",
    "  static async update(id: number, data: Record<string, any>) {",
    "    const keys = Object.keys(data);",
    `    if (keys.length === 0) return ${E}Model.findById(id);`,
    "    const set = keys.map((k) => k + ' = ?').join(', ');",
    `    await pool.execute('UPDATE ${entity}s SET ' + set + ' WHERE id = ?', [...Object.values(data), id]);`,
    `    return ${E}Model.findById(id);`,
    "  }",
    "",
    "  static async deleteById(id: number) {",
    `    await pool.execute('DELETE FROM ${entity}s WHERE id = ?', [id]);`,
    "    return true;",
    "  }",
    "}",
  ].join('\n');
}

// ── User Model (for auth) ─────────────────────────────────────
export function userModelTemplate(): string {
  return [
    "import { pool } from '../config/db.js';",
    "",
    "export class UserModel {",
    "",
    "  static async create(data: Record<string, any>) {",
    "    const { name, email, password, role = 'user' } = data;",
    "    const [result]: any = await pool.execute(",
    "      'INSERT INTO users (name, email, password, role) VALUES (?, ?, ?, ?)',",
    "      [name, email, password, role]",
    "    );",
    "    return UserModel.findById(result.insertId);",
    "  }",
    "",
    "  static async findById(id: number) {",
    "    const [rows]: any = await pool.execute('SELECT * FROM users WHERE id = ?', [id]);",
    "    return rows[0] ?? null;",
    "  }",
    "",
    "  static async findByEmail(email: string) {",
    "    const [rows]: any = await pool.execute('SELECT * FROM users WHERE email = ?', [email]);",
    "    return rows[0] ?? null;",
    "  }",
    "",
    "  static async findAll() {",
    "    const [rows]: any = await pool.execute('SELECT id, name, email, role, created_at FROM users');",
    "    return rows;",
    "  }",
    "",
    "  static async update(id: number, data: Record<string, any>) {",
    "    const keys = Object.keys(data);",
    "    if (keys.length === 0) return UserModel.findById(id);",
    "    const set = keys.map((k) => k + ' = ?').join(', ');",
    "    await pool.execute('UPDATE users SET ' + set + ' WHERE id = ?', [...Object.values(data), id]);",
    "    return UserModel.findById(id);",
    "  }",
    "",
    "  static async deleteById(id: number) {",
    "    await pool.execute('DELETE FROM users WHERE id = ?', [id]);",
    "    return true;",
    "  }",
    "}",
  ].join('\n');
}

// ── Service ───────────────────────────────────────────────────
export function serviceTemplate(entity: string): string {
  const E = cap(entity);
  return [
    `import { ${E}Model } from '../models/${entity}Model.js';`,
    "import { notFound } from '../utils/AppError.js';",
    "",
    `export class ${E}Service {`,
    "",
    "  async getAll(query: Record<string, any>) {",
    "    const page    = Math.max(1, Number(query.page)  || 1);",
    "    const limit   = Math.min(100, Number(query.limit) || 10);",
    "    const filters = { ...query };",
    "    delete filters.page;",
    "    delete filters.limit;",
    `    const { rows, total } = await ${E}Model.findAll(filters, Number(page), Number(limit));`,
    "    return { data: rows, pagination: { total, page, limit, totalPages: Math.ceil(total / limit) } };",
    "  }",
    "",
    "  async getById(id: number) {",
    `    const item = await ${E}Model.findById(id);`,
    `    if (!item) throw notFound('${E} not found');`,
    "    return item;",
    "  }",
    "",
    "  async create(data: Record<string, any>) {",
    `    return ${E}Model.create(data);`,
    "  }",
    "",
    "  async update(id: number, data: Record<string, any>) {",
    "    await this.getById(id);",
    `    return ${E}Model.update(id, data);`,
    "  }",
    "",
    "  async remove(id: number) {",
    "    await this.getById(id);",
    `    return ${E}Model.deleteById(id);`,
    "  }",
    "}",
  ].join('\n');
}

// ── Controller ────────────────────────────────────────────────
export function controllerTemplate(entity: string): string {
  const E = cap(entity);
  return [
    `import { ${E}Service } from '../services/${entity}Service.js';`,
    `import { create${E}Schema, update${E}Schema } from '../validations/${entity}Validation.js';`,
    "",
    `const service = new ${E}Service();`,
    "",
    `export async function getAll${E}s(req: any, res: any, next: any) {`,
    "  try {",
    "    const result = await service.getAll(req.query);",
    "    res.status(200).json({ success: true, data: result.data, pagination: result.pagination });",
    "  } catch (err) { next(err); }",
    "}",
    "",
    `export async function get${E}ById(req: any, res: any, next: any) {`,
    "  try {",
    "    const data = await service.getById(Number(req.params.id));",
    "    res.status(200).json({ success: true, data });",
    "  } catch (err) { next(err); }",
    "}",
    "",
    `export async function create${E}(req: any, res: any, next: any) {`,
    "  try {",
    `    const result = create${E}Schema.safeParse(req.body);`,
    "    if (!result.success) return res.status(400).json({ success: false, message: 'Validation failed', errors: result.error.errors });",
    "    const data = await service.create(result.data);",
    `    res.status(201).json({ success: true, data, message: '${E} created' });`,
    "  } catch (err) { next(err); }",
    "}",
    "",
    `export async function update${E}(req: any, res: any, next: any) {`,
    "  try {",
    `    const result = update${E}Schema.safeParse(req.body);`,
    "    if (!result.success) return res.status(400).json({ success: false, message: 'Validation failed', errors: result.error.errors });",
    "    const data = await service.update(Number(req.params.id), result.data);",
    `    res.status(200).json({ success: true, data, message: '${E} updated' });`,
    "  } catch (err) { next(err); }",
    "}",
    "",
    `export async function delete${E}(req: any, res: any, next: any) {`,
    "  try {",
    "    await service.remove(Number(req.params.id));",
    `    res.status(200).json({ success: true, message: '${E} deleted' });`,
    "  } catch (err) { next(err); }",
    "}",
  ].join('\n');
}

// ── Routes ────────────────────────────────────────────────────
export function routesTemplate(entity: string, features: Feature[]): string {
  const E       = cap(entity);
  const hasAuth = features.includes('auth');
  const lines = [
    "import { Router } from 'express';",
    `import { getAll${E}s, get${E}ById, create${E}, update${E}, delete${E} } from '../controllers/${entity}Controller.js';`,
    hasAuth ? "import { authenticate, authorize } from '../middleware/authMiddleware.js';" : null,
    "",
    "const router = Router();",
    "",
    `router.get('/',      ${hasAuth ? 'authenticate, ' : ''}getAll${E}s);`,
    `router.get('/:id',   ${hasAuth ? 'authenticate, ' : ''}get${E}ById);`,
    `router.post('/',     ${hasAuth ? 'authenticate, ' : ''}create${E});`,
    `router.put('/:id',   ${hasAuth ? "authenticate, authorize('admin'), " : ''}update${E});`,
    `router.delete('/:id',${hasAuth ? "authenticate, authorize('admin'), " : ''}delete${E});`,
    "",
    "export default router;",
  ];
  return lines.filter((l): l is string => l !== null).join('\n');
}

// ── Auto-migrate ──────────────────────────────────────────────
export function migrateTemplate(entity: string, fields: Field[], db: DbType): string {
  const reserved = ['id', 'created_at'];
  const entityColLines = fields
    .filter((f) => !reserved.includes(f.name))
    .map((f) => `      '  ${f.name} ${sqlType(f.type, f.name)},',`);

  if (db === 'mysql') {
    return [
      "import { pool } from './config/db.js';",
      "",
      "export async function migrate(): Promise<void> {",
      "  try {",
      "    await pool.execute([",
      `      'CREATE TABLE IF NOT EXISTS ${entity}s (',`,
      "      '  id INT AUTO_INCREMENT PRIMARY KEY,',",
      ...entityColLines,
      "      '  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP',",
      "      ')'",
      "    ].join(' '));",
      `    console.log('  Table \`${entity}s\` ready');`,
      "",
      "    await pool.execute([",
      "      'CREATE TABLE IF NOT EXISTS users (',",
      "      '  id         INT AUTO_INCREMENT PRIMARY KEY,',",
      "      '  name       VARCHAR(100) NOT NULL,',",
      "      '  email      VARCHAR(150) NOT NULL UNIQUE,',",
      "      '  password   VARCHAR(255),',",
      "      \"  role       VARCHAR(20) DEFAULT 'user',\",",
      "      '  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP',",
      "      ')'",
      "    ].join(' '));",
      "    console.log('  Table `users` ready');",
      "  } catch (err: any) {",
      "    console.error('Migration failed: ' + err.message);",
      "    process.exit(1);",
      "  }",
      "}",
    ].join('\n');
  }
  return "export async function migrate(): Promise<void> { console.log('Migration not implemented for this DB'); }";
}

// ── Server Entry ──────────────────────────────────────────────
export function serverTemplate(entity: string, features: Feature[]): string {
  const hasAuth = features.includes('auth');
  const lines = [
    "import 'dotenv/config';",
    "import express from 'express';",
    "import cors from 'cors';",
    "import { connectDB } from './config/db.js';",
    "import { migrate } from './migrate.js';",
    "import { requestLogger } from './middleware/loggerMiddleware.js';",
    "import errorHandler from './middleware/errorMiddleware.js';",
    `import ${entity}Router from './routes/${entity}Routes.js';`,
    hasAuth ? "import authRouter from './routes/authRoutes.js';" : null,
    "",
    "const app  = express();",
    "const PORT = Number(process.env.PORT) || 3000;",
    "",
    "app.use(cors());",
    "app.use(express.json());",
    "app.use(express.urlencoded({ extended: true }));",
    "app.use(requestLogger);",
    "",
    `app.use('/api/${entity}s', ${entity}Router);`,
    hasAuth ? "app.use('/api/auth', authRouter);" : null,
    "",
    "app.get('/health', (_req, res) => res.json({ success: true, message: 'Server is healthy' }));",
    "",
    "app.use((_req, res) => res.status(404).json({ success: false, message: 'Route not found' }));",
    "app.use(errorHandler);",
    "",
    "async function start(): Promise<void> {",
    "  try {",
    "    await connectDB();",
    "    await migrate();",
    "    app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));",
    "  } catch (err: any) {",
    "    console.error('Failed to start: ' + err.message);",
    "    process.exit(1);",
    "  }",
    "}",
    "",
    "process.on('SIGTERM', () => process.exit(0));",
    "start();",
    "export default app;",
  ];
  return lines.filter((l): l is string => l !== null).join('\n');
}

// ── tsconfig.json ────────────────────────────────────────────
export function tsconfigTemplate(): string {
  return JSON.stringify({
    compilerOptions: {
      target: 'ES2022',
      module: 'NodeNext',
      moduleResolution: 'NodeNext',
      strict: true,
      esModuleInterop: true,
      skipLibCheck: true,
      resolveJsonModule: true,
      outDir: 'dist',
      rootDir: 'src',
    },
    include: ['src/**/*'],
    exclude: ['node_modules', 'dist'],
  }, null, 2);
}

// ── package.json ──────────────────────────────────────────────
export function packageJsonTemplate(entity: string, db: DbType): string {
  const pkgMap: Record<DbType, string> = { mysql: 'mysql2', mssql: 'mssql', oracledb: 'oracledb' };
  const verMap: Record<string, string> = { mysql2: '^3.6.5', mssql: '^10.0.1', oracledb: '^6.4.0' };
  const pkg = pkgMap[db];
  return JSON.stringify({
    name: `${entity}-backend`,
    version: '1.0.0',
    type: 'module',
    scripts: { start: 'node --import tsx/esm src/server.ts', dev: 'tsx watch src/server.ts' },
    dependencies: {
      express: '^4.18.2',
      cors: '^2.8.5',
      dotenv: '^16.4.5',
      zod: '^3.22.4',
      jsonwebtoken: '^9.0.2',
      bcryptjs: '^2.4.3',
      [pkg]: verMap[pkg],
    },
    devDependencies: {
      tsx: '^4.7.0',
      typescript: '^5.4.0',
      '@types/node': '^20.0.0',
      '@types/express': '^4.17.21',
      '@types/cors': '^2.8.17',
      '@types/jsonwebtoken': '^9.0.5',
      '@types/bcryptjs': '^2.4.6',
    },
  }, null, 2);
}

// ── Multi-entity migrate ──────────────────────────────────────
export function multiMigrateTemplate(
  entities: Array<{ entity: string; fields: import('../types/index.js').Field[] }>,
  relations: import('../types/index.js').Relation[],
  db: import('../types/index.js').DbType
): string {
  if (db !== 'mysql') {
    return "export async function migrate(): Promise<void> { console.log('Migration not implemented for this DB'); }";
  }

  const reserved = ['id', 'created_at'];

  // Build FK set for quick lookup: "billing.studentId" -> "students"
  const fkSet = new Map<string, string>();
  for (const rel of relations) {
    fkSet.set(`${rel.fromEntity}.${rel.foreignKey}`, rel.toEntity);
  }

  // Sort entities so referenced tables are created first
  const sorted = topologicalSort(entities.map(e => e.entity), relations);

  const tableBlocks = sorted.map((entityName) => {
    const def = entities.find(e => e.entity === entityName)!;
    const cols = def.fields.filter(f => !reserved.includes(f.name)).map(f => {
      const fkTarget = fkSet.get(`${entityName}.${f.name}`);
      const colType  = fkTarget ? 'INT' : sqlTypeInline(f.type, f.name);
      const fkClause = fkTarget
        ? `,\\n  FOREIGN KEY (${f.name}) REFERENCES ${fkTarget}s(id) ON DELETE CASCADE`
        : '';
      return `  ${f.name} ${colType} NOT NULL${fkClause}`;
    });

    const colLines = [
      '  id INT AUTO_INCREMENT PRIMARY KEY',
      ...cols,
      '  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP',
    ].join(',\\n');

    return [
      `    await pool.execute(\`CREATE TABLE IF NOT EXISTS ${entityName}s (\\n${colLines}\\n)\`);`,
      `    console.log('  Table \\\`${entityName}s\\\` ready');`,
    ].join('\n');
  });

  return [
    "import { pool } from './config/db.js';",
    "",
    "export async function migrate(): Promise<void> {",
    "  try {",
    "    // users table — always created for auth",
    "    await pool.execute(`CREATE TABLE IF NOT EXISTS users (\\n" +
      "  id         INT AUTO_INCREMENT PRIMARY KEY,\\n" +
      "  name       VARCHAR(100) NOT NULL,\\n" +
      "  email      VARCHAR(150) NOT NULL UNIQUE,\\n" +
      "  password   VARCHAR(255),\\n" +
      "  role       VARCHAR(20) DEFAULT 'user',\\n" +
      "  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP\\n" +
    ")`);",
    "    console.log('  Table `users` ready');",
    "",
    tableBlocks.join('\n\n'),
    "  } catch (err: any) {",
    "    console.error('Migration failed: ' + err.message);",
    "    process.exit(1);",
    "  }",
    "}",
  ].join('\n');
}

// ── Multi-entity server ───────────────────────────────────────
export function multiServerTemplate(
  entities: string[],
  features: import('../types/index.js').Feature[]
): string {
  const hasAuth = features.includes('auth');
  const imports = entities.map(e =>
    `import ${e}Router from './routes/${e}Routes.js';`
  );
  const mounts = entities.map(e =>
    `app.use('/api/${e}s', ${e}Router);`
  );

  return [
    "import 'dotenv/config';",
    "import express from 'express';",
    "import cors from 'cors';",
    "import { connectDB } from './config/db.js';",
    "import { migrate } from './migrate.js';",
    "import { requestLogger } from './middleware/loggerMiddleware.js';",
    "import errorHandler from './middleware/errorMiddleware.js';",
    ...imports,
    hasAuth ? "import authRouter from './routes/authRoutes.js';" : null,
    "",
    "const app  = express();",
    "const PORT = Number(process.env.PORT) || 3000;",
    "",
    "app.use(cors());",
    "app.use(express.json());",
    "app.use(express.urlencoded({ extended: true }));",
    "app.use(requestLogger);",
    "",
    ...mounts,
    hasAuth ? "app.use('/api/auth', authRouter);" : null,
    "",
    "app.get('/health', (_req, res) => res.json({ success: true, message: 'Server is healthy' }));",
    "app.use((_req, res) => res.status(404).json({ success: false, message: 'Route not found' }));",
    "app.use(errorHandler);",
    "",
    "async function start(): Promise<void> {",
    "  try {",
    "    await connectDB();",
    "    await migrate();",
    "    app.listen(PORT, () => console.log(`Server running on http://localhost:${PORT}`));",
    "  } catch (err: any) {",
    "    console.error('Failed to start: ' + err.message);",
    "    process.exit(1);",
    "  }",
    "}",
    "",
    "process.on('SIGTERM', () => process.exit(0));",
    "start();",
    "export default app;",
  ].filter((l): l is string => l !== null).join('\n');
}

// ── Helpers (module-level) ────────────────────────────────────
function sqlTypeInline(type: string, fieldName: string = ''): string {
  if (type === 'boolean') return 'TINYINT(1) DEFAULT 0';
  if (type === 'date')    return 'DATE';
  if (type === 'number') {
    const big = ['phone', 'mobile', 'zip', 'postal', 'pincode', 'zipcode'];
    if (big.some(k => fieldName.toLowerCase().includes(k))) return 'BIGINT';
    return 'INT';
  }
  return 'VARCHAR(255)';
}

// Topological sort — ensures referenced tables are created before FKs
function topologicalSort(entities: string[], relations: import('../types/index.js').Relation[]): string[] {
  const deps = new Map<string, Set<string>>();
  for (const e of entities) deps.set(e, new Set());
  for (const r of relations) {
    deps.get(r.fromEntity)?.add(r.toEntity);
  }

  const sorted: string[] = [];
  const visited = new Set<string>();

  function visit(e: string) {
    if (visited.has(e)) return;
    visited.add(e);
    for (const dep of deps.get(e) ?? []) visit(dep);
    sorted.push(e);
  }

  for (const e of entities) visit(e);
  return sorted;
}

// ── Auth Routes (static) ──────────────────────────────────────
export function authRoutesTemplate(): string {
  return [
    "import { Router } from 'express';",
    "import bcrypt from 'bcryptjs';",
    "import { z } from 'zod';",
    "import { UserModel } from '../models/userModel.js';",
    "import { generateToken } from '../middleware/authMiddleware.js';",
    "",
    "const router = Router();",
    "",
    "const registerSchema = z.object({",
    "  name:     z.string().min(1),",
    "  email:    z.string().email(),",
    "  password: z.string().min(6),",
    "  role:     z.enum(['user', 'admin']).optional().default('user'),",
    "});",
    "",
    "const loginSchema = z.object({",
    "  email:    z.string().email(),",
    "  password: z.string().min(1),",
    "});",
    "",
    "router.post('/register', async (req, res, next) => {",
    "  try {",
    "    const parsed = registerSchema.safeParse(req.body);",
    "    if (!parsed.success) return res.status(400).json({ success: false, errors: parsed.error.errors });",
    "    const { name, email, password, role } = parsed.data;",
    "    const existing = await UserModel.findByEmail(email);",
    "    if (existing) return res.status(400).json({ success: false, message: 'Email already registered' });",
    "    const hashed = await bcrypt.hash(password, Number(process.env.BCRYPT_ROUNDS) || 10);",
    "    const user = await UserModel.create({ name, email, password: hashed, role: role ?? 'user' });",
    "    const token = generateToken({ id: user.id, email: user.email, role: user.role });",
    "    return res.status(201).json({ success: true, message: 'Registered successfully', token, user: { id: user.id, name: user.name, email: user.email, role: user.role } });",
    "  } catch (err) { next(err); }",
    "});",
    "",
    "router.post('/login', async (req, res, next) => {",
    "  try {",
    "    const parsed = loginSchema.safeParse(req.body);",
    "    if (!parsed.success) return res.status(400).json({ success: false, errors: parsed.error.errors });",
    "    const { email, password } = parsed.data;",
    "    const user = await UserModel.findByEmail(email);",
    "    if (!user) return res.status(401).json({ success: false, message: 'Invalid email or password' });",
    "    const stored = user.password ? user.password.toString() : '';",
    "    const match  = await bcrypt.compare(password, stored);",
    "    if (!match) return res.status(401).json({ success: false, message: 'Invalid email or password' });",
    "    const token = generateToken({ id: user.id, email: user.email, role: user.role });",
    "    return res.status(200).json({ success: true, message: 'Login successful', token, user: { id: user.id, name: user.name, email: user.email, role: user.role } });",
    "  } catch (err) { next(err); }",
    "});",
    "",
    "router.post('/logout', (_req, res) => {",
    "  return res.status(200).json({ success: true, message: 'Logged out successfully' });",
    "});",
    "",
    "export default router;",
  ].join('\n');
}

// ── Validation (static — no AI, field names preserved exactly) ─
export function validationTemplate(entity: string, fields: import('../types/index.js').Field[]): string {
  const E = entity.charAt(0).toUpperCase() + entity.slice(1);

  const schemaLines = fields.map((f) => {
    if (f.name === 'email')    return `  email:    z.string().email(),`;
    if (f.name === 'password') return `  password: z.string().min(6),`;
    if (f.type === 'number')   return `  ${f.name}: z.number().optional(),`;
    if (f.type === 'boolean')  return `  ${f.name}: z.boolean().optional(),`;
    if (f.type === 'date')     return `  ${f.name}: z.string().optional(),`;
    return `  ${f.name}: z.string().min(1),`;
  }).join('\n');

  return [
    "import { z } from 'zod';",
    "",
    `export const create${E}Schema = z.object({`,
    schemaLines,
    "});",
    "",
    `export const update${E}Schema = create${E}Schema.partial();`,
  ].join('\n');
}