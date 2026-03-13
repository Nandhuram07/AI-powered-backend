// src/templates/prompts.ts

import { Field, PromptObject } from "../types";

const SYSTEM =
  'You are a Node.js backend code generator. Your ONLY job is to output working JavaScript code.\n\n' +
  'NON-NEGOTIABLE RULES:\n' +
  '1. PLAIN JAVASCRIPT ONLY — zero TypeScript, zero type annotations\n' +
  '2. NO MARKDOWN — output raw code only, never wrap in ``` fences\n' +
  '3. ESM ONLY — import/export syntax, never require() or module.exports\n' +
  '4. MYSQL NOT MONGOOSE — this app uses mysql2, not MongoDB\n' +
  '   BANNED: .save() .find() .findOne() new Model()\n' +
  '5. NAMED IMPORTS — always destructure, never default import pool or AppError\n' +
  '   BAD:  import pool from "../config/db.js"\n' +
  '   GOOD: import { pool } from "../config/db.js"\n\n' +
  'Output ONLY the code file. No explanation. No comments outside the code.';

function cap(s: string): string {
  return s.charAt(0).toUpperCase() + s.slice(1);
}

// ── Validation (Zod) ──────────────────────────────────────────
export function validationPrompt(entity: string, fields: Field[]): PromptObject {
  const E = cap(entity);

  const schemaLines = fields.map((f) => {
    if (f.name === 'email')    return `  email:    z.string().email(),`;
    if (f.name === 'password') return `  password: z.string().min(6),`;
    if (f.type === 'number')   return `  ${f.name}: z.number().optional(),`;
    if (f.type === 'boolean')  return `  ${f.name}: z.boolean().optional(),`;
    if (f.type === 'date')     return `  ${f.name}: z.string().optional(),`;
    return `  ${f.name}: z.string().min(1),`;
  }).join('\n');

  return {
    system: SYSTEM,
    user:
      `Generate src/validations/${entity}Validation.js\n\n` +
      `COPY THIS EXACT STRUCTURE:\n\n` +
      `import { z } from 'zod';\n\n` +
      `export const create${E}Schema = z.object({\n` +
      `${schemaLines}\n` +
      `});\n\n` +
      `export const update${E}Schema = create${E}Schema.partial();\n\n` +
      `Output only this file. NO TypeScript. NO backticks. NO markdown.`,
  };
}

// ── Auth Routes ───────────────────────────────────────────────
export function authRoutePrompt(_entity: string): PromptObject {
  return {
    system: SYSTEM,
    user:
      'Generate src/routes/authRoutes.js\n\n' +
      'COPY THIS EXACT STRUCTURE — do not deviate:\n\n' +
      "import { Router } from 'express';\n" +
      "import bcrypt from 'bcryptjs';\n" +
      "import { z } from 'zod';\n" +
      "import { UserModel } from '../models/userModel.js';\n" +
      "import { generateToken } from '../middleware/authMiddleware.js';\n\n" +
      'const router = Router();\n\n' +
      'const registerSchema = z.object({\n' +
      '  name:     z.string().min(1),\n' +
      '  email:    z.string().email(),\n' +
      '  password: z.string().min(6),\n' +
      "  role:     z.enum(['user', 'admin']).optional().default('user'),\n" +
      '});\n\n' +
      'const loginSchema = z.object({\n' +
      '  email:    z.string().email(),\n' +
      '  password: z.string().min(1),\n' +
      '});\n\n' +
      "router.post('/register', async function(req, res, next) {\n" +
      '  try {\n' +
      '    const parsed = registerSchema.safeParse(req.body);\n' +
      '    if (!parsed.success) return res.status(400).json({ success: false, errors: parsed.error.errors });\n' +
      '    const { name, email, password, role } = parsed.data;\n' +
      '    const existing = await UserModel.findByEmail(email);\n' +
      '    if (existing) return res.status(400).json({ success: false, message: "Email already registered" });\n' +
      '    const hashed = await bcrypt.hash(password, Number(process.env.BCRYPT_ROUNDS) || 10);\n' +
      "    const user = await UserModel.create({ name, email, password: hashed, role: role || 'user' });\n" +
      '    const token = generateToken({ id: user.id, email: user.email, role: user.role });\n' +
      '    return res.status(201).json({ success: true, message: "Registered successfully", token, user: { id: user.id, name: user.name, email: user.email, role: user.role } });\n' +
      '  } catch (err) { next(err); }\n' +
      '});\n\n' +
      "router.post('/login', async function(req, res, next) {\n" +
      '  try {\n' +
      '    const parsed = loginSchema.safeParse(req.body);\n' +
      '    if (!parsed.success) return res.status(400).json({ success: false, errors: parsed.error.errors });\n' +
      '    const { email, password } = parsed.data;\n' +
      '    const user = await UserModel.findByEmail(email);\n' +
      '    if (!user) return res.status(401).json({ success: false, message: "Invalid email or password" });\n' +
      "    const storedPassword = user.password ? user.password.toString() : '';\n" +
      '    const match = await bcrypt.compare(password, storedPassword);\n' +
      '    if (!match) return res.status(401).json({ success: false, message: "Invalid email or password" });\n' +
      '    const token = generateToken({ id: user.id, email: user.email, role: user.role });\n' +
      '    return res.status(200).json({ success: true, message: "Login successful", token, user: { id: user.id, name: user.name, email: user.email, role: user.role } });\n' +
      '  } catch (err) { next(err); }\n' +
      '});\n\n' +
      "router.post('/logout', function(_req, res) {\n" +
      '  return res.status(200).json({ success: true, message: "Logged out successfully" });\n' +
      '});\n\n' +
      'export default router;\n\n' +
      'RULES:\n' +
      '- Use UserModel.findByEmail() and UserModel.create() — NEVER .save()\n' +
      '- storedPassword = user.password.toString() before bcrypt.compare()\n' +
      'NO TypeScript. NO backticks. NO markdown. NO Mongoose.',
  };
}