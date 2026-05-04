// src/utils/aiProvider.ts
import dotenv from 'dotenv';
import { AIProvider } from '../types';

dotenv.config();

const provider = (process.env.AI_PROVIDER ?? 'groq').toLowerCase() as AIProvider;

// ── Public entry point for streaming ────────────────────────────────────────
export async function* streamAI(
  systemPrompt: string,
  userPrompt: string,
  retries: number = 4
): AsyncGenerator<string, void, unknown> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      yield* dispatchStream(systemPrompt, userPrompt);
      return;
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : String(err);
      const is429 =
        msg.includes('429') ||
        msg.toLowerCase().includes('rate limit') ||
        msg.toLowerCase().includes('quota');

      if (is429 && attempt < retries) {
        const match = msg.match(/retry in ([\d.]+)s/i);
        const waitSec = match ? Math.ceil(parseFloat(match[1])) + 2 : 10 * attempt;
        console.log(`\n⏳  Rate limited — waiting ${waitSec}s (attempt ${attempt}/${retries - 1})...`);
        await sleep(waitSec * 1000);
      } else {
        throw err;
      }
    }
  }
  throw new Error('streamAI: exhausted all retries');
}

// ── Keep old API for backwards compatibility ────────────────────────────────
export async function callAI(
  systemPrompt: string,
  userPrompt: string,
  retries: number = 4
): Promise<string> {
  let result = '';
  for await (const chunk of streamAI(systemPrompt, userPrompt, retries)) {
    result += chunk;
  }
  return result;
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function dispatchStream(systemPrompt: string, userPrompt: string): AsyncGenerator<string, void, unknown> {
  switch (provider) {
    case 'groq':      return streamGroq(systemPrompt, userPrompt);
    case 'gemini':    return streamGemini(systemPrompt, userPrompt);
    case 'anthropic': return streamAnthropic(systemPrompt, userPrompt);
    case 'openai':    return streamOpenAI(systemPrompt, userPrompt);
    default:
      throw new Error(`Unknown AI_PROVIDER: "${provider}". Use groq | gemini | anthropic | openai`);
  }
}

// ── Groq ──────────────────────────────────────────────────────
async function* streamGroq(system: string, user: string): AsyncGenerator<string, void, unknown> {
  const apiKey = process.env.GROQ_API_KEY;
  if (!apiKey) throw new Error('GROQ_API_KEY is not set in .env');

  const model = process.env.GROQ_MODEL ?? 'llama-3.3-70b-versatile';
  const res = await fetch('https://api.groq.com/openai/v1/chat/completions', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${apiKey}` },
    body: JSON.stringify({
      model,
      max_tokens: 8192,
      temperature: 0.2,
      stream: true,
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({})) as { error?: { message?: string } };
    throw new Error(`Groq API error (${res.status}): ${err?.error?.message ?? res.statusText}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body from Groq');

  const decoder = new TextDecoder();
  for await (const chunk of readStream(reader)) {
    const text = decoder.decode(chunk, { stream: true });
    for (const line of text.split('\n')) {
      if (line.startsWith('data: ')) {
        const data = line.slice(6).trim();
        if (data === '[DONE]') break;
        try {
          const parsed = JSON.parse(data) as { choices?: Array<{ delta?: { content?: string } }> };
          const content = parsed.choices?.[0]?.delta?.content;
          if (content) yield content;
        } catch {
          // skip invalid json lines
        }
      }
    }
  }
}

// ── Gemini ────────────────────────────────────────────────────
async function* streamGemini(system: string, user: string): AsyncGenerator<string, void, unknown> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY is not set in .env');

  const model = process.env.GEMINI_MODEL ?? 'gemini-2.0-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:streamGenerateContent?key=${apiKey}`;

  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      system_instruction: { parts: [{ text: system }] },
      contents: [{ role: 'user', parts: [{ text: user }] }],
      generationConfig: { maxOutputTokens: 8192, temperature: 0.2 },
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({})) as { error?: { message?: string } };
    throw new Error(`Gemini API error (${res.status}): ${err?.error?.message ?? res.statusText}`);
  }

  const reader = res.body?.getReader();
  if (!reader) throw new Error('No response body from Gemini');

  const decoder = new TextDecoder();
  for await (const chunk of readStream(reader)) {
    const text = decoder.decode(chunk, { stream: true });
    for (const line of text.split('\n')) {
      if (line.trim()) {
        try {
          const parsed = JSON.parse(line) as { candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }> };
          const content = parsed.candidates?.[0]?.content?.parts?.[0]?.text;
          if (content) yield content;
        } catch {
          // skip invalid json lines
        }
      }
    }
  }
}

// ── Anthropic ─────────────────────────────────────────────────
async function* streamAnthropic(system: string, user: string): AsyncGenerator<string, void, unknown> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error('ANTHROPIC_API_KEY is not set in .env');

  const { default: Anthropic } = await import('@anthropic-ai/sdk');
  const client = new Anthropic({ apiKey });
  const model = process.env.ANTHROPIC_MODEL ?? 'claude-sonnet-4-20250514';

  const stream = await client.messages.stream({
    model,
    max_tokens: 4096,
    system,
    messages: [{ role: 'user', content: user }],
  });

  for await (const chunk of stream) {
    if (chunk.type === 'content_block_delta' && chunk.delta.type === 'text_delta') {
      yield chunk.delta.text;
    }
  }
}

// ── OpenAI ────────────────────────────────────────────────────
async function* streamOpenAI(system: string, user: string): AsyncGenerator<string, void, unknown> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY is not set in .env');

  const { default: OpenAI } = await import('openai');
  const client = new OpenAI({ apiKey });
  const model = process.env.OPENAI_MODEL ?? 'gpt-4o';

  const stream = await client.chat.completions.create({
    model,
    max_tokens: 4096,
    stream: true,
    messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
  });

  for await (const chunk of stream) {
    const content = chunk.choices[0]?.delta?.content;
    if (content) yield content;
  }
}

// ── Helper to read from ReadableStream ─────────────────────────
async function* readStream(reader: ReadableStreamDefaultReader<Uint8Array>): AsyncGenerator<Uint8Array, void, unknown> {
  try {
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      yield value;
    }
  } finally {
    reader.releaseLock();
  }
}
