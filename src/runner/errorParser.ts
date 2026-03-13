// src/runner/errorParser.ts
// Extracts structured error info from Node.js / TypeScript crash output

export interface ParsedError {
    file:       string;   // relative path e.g. src/models/studentModel.ts
    line?:      number;
    column?:    number;
    message:    string;   // clean error message
    rawOutput:  string;   // full original output for AI context
    errorType:  ErrorType;
  }
  
  export type ErrorType =
    | 'import_not_found'
    | 'syntax_error'
    | 'type_error'
    | 'runtime_error'
    | 'db_error'
    | 'unknown';
  
  /**
   * Parse raw crash output into structured error info
   */
  export function parseError(output: string, outputDir: string): ParsedError {
    const raw = output;
  
    // ── Pattern 1: Node ESM import error ─────────────────────
    // Cannot find module 'X' imported from '/path/to/file.ts'
    const importMatch = output.match(
      /Cannot find module '([^']+)' imported from '([^']+)'/
    );
    if (importMatch) {
      const file = toRelative(importMatch[2], outputDir);
      return {
        file,
        message: `Cannot find module '${importMatch[1]}'`,
        rawOutput: raw,
        errorType: 'import_not_found',
      };
    }
  
    // ── Pattern 2: TypeScript error ──────────────────────────
    // src/models/studentModel.ts(6,38): error TS2345: ...
    const tsMatch = output.match(/([^\s(]+\.ts)\((\d+),(\d+)\):\s*error\s+TS\d+:\s*(.+)/);
    if (tsMatch) {
      return {
        file:      tsMatch[1],
        line:      Number(tsMatch[2]),
        column:    Number(tsMatch[3]),
        message:   tsMatch[4].trim(),
        rawOutput: raw,
        errorType: 'type_error',
      };
    }
  
    // ── Pattern 3: SyntaxError with file path ────────────────
    // SyntaxError: ... at file:///C:/path/to/file.ts:10:5
    const syntaxMatch = output.match(/SyntaxError:\s*(.+)\n.*at\s+file:\/\/\/([^\s:]+):(\d+)/);
    if (syntaxMatch) {
      return {
        file:      toRelative(syntaxMatch[2], outputDir),
        line:      Number(syntaxMatch[3]),
        message:   `SyntaxError: ${syntaxMatch[1]}`,
        rawOutput: raw,
        errorType: 'syntax_error',
      };
    }
  
    // ── Pattern 4: Runtime error with stack trace ─────────────
    // Error: something at ClassName.method (file:///path/file.ts:line:col)
    const runtimeMatch = output.match(
      /(?:Error|TypeError|ReferenceError):\s*(.+)\n[\s\S]*?at\s+\S+\s+\(file:\/\/\/([^\s:)]+):(\d+)/
    );
    if (runtimeMatch) {
      return {
        file:      toRelative(runtimeMatch[2], outputDir),
        line:      Number(runtimeMatch[3]),
        message:   runtimeMatch[1].trim(),
        rawOutput: raw,
        errorType: 'runtime_error',
      };
    }
  
    // ── Pattern 5: DB connection error ───────────────────────
    if (output.includes('ER_') || output.includes('ECONNREFUSED') || output.includes('MySQL')) {
      return {
        file:      'src/config/db.ts',
        message:   extractFirstLine(output),
        rawOutput: raw,
        errorType: 'db_error',
      };
    }
  
    // ── Fallback ──────────────────────────────────────────────
    // Try to find any .ts file mention
    const anyTsFile = output.match(/src\/[^\s:)]+\.ts/);
    return {
      file:      anyTsFile ? anyTsFile[0] : 'src/server.ts',
      message:   extractFirstLine(output),
      rawOutput: raw,
      errorType: 'unknown',
    };
  }
  
  // ── Helpers ───────────────────────────────────────────────────
  function toRelative(absPath: string, outputDir: string): string {
    const normalized = absPath.replace(/\\/g, '/');
    const base       = outputDir.replace(/\\/g, '/');
    const idx        = normalized.indexOf(base);
    if (idx !== -1) return normalized.slice(idx + base.length).replace(/^\//, '');
    // fallback — extract src/... portion
    const srcMatch = normalized.match(/(src\/[^)]+)/);
    return srcMatch ? srcMatch[1] : normalized;
  }
  
  function extractFirstLine(output: string): string {
    return output.split('\n').find(l => l.trim().length > 0)?.trim() ?? 'Unknown error';
  }