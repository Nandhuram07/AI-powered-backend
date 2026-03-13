// src/runner/processRunner.ts
// Spawns the generated backend, captures stdout/stderr, detects crash vs healthy

import { spawn, type ChildProcess } from 'child_process';
import path from 'path';

export interface RunResult {
  healthy: boolean;   // true = server started and is still running
  output:  string;    // combined stdout + stderr
  pid?:    number;
}

/**
 * Starts the generated server with tsx.
 * Waits `waitMs` milliseconds — if the process is still alive, it's healthy.
 * If it crashes before waitMs, healthy = false with full output.
 */
export function startServer(outputDir: string, waitMs = 4000): Promise<RunResult> {
  return new Promise((resolve) => {
    const absDir = path.resolve(outputDir);
    const output: string[] = [];

    const child: ChildProcess = spawn('npx', ['tsx', 'src/server.ts'], {
      cwd: absDir,
      shell: true,
      env: { ...process.env },
    });

    child.stdout?.on('data', (d: Buffer) => output.push(d.toString()));
    child.stderr?.on('data', (d: Buffer) => output.push(d.toString()));

    let crashed = false;

    child.on('exit', (code) => {
      if (!crashed) {
        crashed = true;
        resolve({
          healthy: false,
          output:  output.join(''),
          pid:     child.pid,
        });
      }
    });

    // If still alive after waitMs — server started successfully
    setTimeout(() => {
      if (!crashed) {
        // Kill it — we only needed to verify it starts
        void child.kill('SIGTERM');
        resolve({
          healthy: true,
          output:  output.join(''),
          pid:     child.pid,
        });
      }
    }, waitMs);
  });
}

/**
 * Quick syntax check using tsx --check before actually running.
 * Returns { ok, errors } where errors is array of { file, message }
 */
export function syntaxCheck(outputDir: string): Promise<{ ok: boolean; output: string }> {
  return new Promise((resolve) => {
    const absDir = path.resolve(outputDir);
    const output: string[] = [];

    const child = spawn('npx', ['tsc', '--noEmit', '--skipLibCheck'], {
      cwd: absDir,
      shell: true,
      env: { ...process.env },
    });

    child.stdout?.on('data', (d: Buffer) => output.push(d.toString()));
    child.stderr?.on('data', (d: Buffer) => output.push(d.toString()));

    child.on('exit', (code) => {
      resolve({ ok: code === 0, output: output.join('') });
    });
  });
}