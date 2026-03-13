// src/utils/codeSanitizer.ts


// ── Strip markdown fences ─────────────────────────────────────
export function stripFences(code: string): string {
  return code
    .replace(/^```[a-zA-Z]*\r?\n?/m, '')
    .replace(/\r?\n?```\s*$/m, '')
    .replace(/[\u2018\u2019]/g, "'")   // smart single quotes
    .replace(/[\u201C\u201D]/g, '"')   // smart double quotes
    .replace(/[\u0060\u02CB\u02BB]/g, '`') // unicode backticks
    .trim();
}

// ── Fix known AI import mistakes ──────────────────────────────
export function fixImports(code: string, entity: string, E: string): string {
  // Fix wrong casing: { jobModel } -> { JobModel }
  code = code.replace(new RegExp('[{]\\s*' + entity + 'Model\\s*[}]', 'g'), '{ ' + E + 'Model }');
  code = code.replace(new RegExp('[{]\\s*' + entity + 'Service\\s*[}]', 'g'), '{ ' + E + 'Service }');

  // Fix default import of pool -> named import
  code = code.replace(
    /import\s+pool\s+from\s+(['"])([^'"]*db\.js)\1/g,
    "import { pool } from '$2'"
  );

  // Fix default import of AppError -> named import
  code = code.replace(
    /import\s+AppError\s+from\s+(['"])([^'"]*AppError\.js)\1/g,
    "import { AppError, notFound, badRequest, unauthorized, forbidden } from '$2'"
  );

  // Fix default import of Model/Service -> named import
  code = code.replace(
    /import\s+(\w+Model)\s+from\s+(['"])([^'"]+)\2/g,
    (_m, name, _q, p) => `import { ${name} } from '${p}'`
  );
  code = code.replace(
    /import\s+(\w+Service)\s+from\s+(['"])([^'"]+)\2/g,
    (_m, name, _q, p) => `import { ${name} } from '${p}'`
  );

  // Fix missing .js extensions on relative imports
  code = code.replace(/from\s+(['"])(\.{1,2}\/[^'"]+)\1/g, (_m, _q, p) => {
    if (/\.[a-z]+$/.test(p)) return `from '${p}'`;
    return `from '${p}.js'`;
  });

  return code;
}

// ── Check if service has raw SQL (safety net) ─────────────────
export function containsSQL(code: string): boolean {
  return /pool\.execute|pool\.query|SELECT\s+\*|INSERT\s+INTO|UPDATE\s+\w+\s+SET|DELETE\s+FROM/i.test(code);
}