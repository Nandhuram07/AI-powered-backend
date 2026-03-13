// src/utils/aiProvider.ts
import dotenv from 'dotenv';
import { AIProvider } from '../types';

dotenv.config();

const provider = (process.env.AI_PROVIDER ?? 'groq').toLowerCase() as AIProvider;

// ── Public entry point ────────────────────────────────────────
export async function callAI(
  systemPrompt: string,
  userPrompt: string,
  retries: number = 4
): Promise<string> {
  for (let attempt = 1; attempt <= retries; attempt++) {
    try {
      return await dispatch(systemPrompt, userPrompt);
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
  throw new Error('callAI: exhausted all retries');
}

function sleep(ms: number): Promise<void> {
  return new Promise((r) => setTimeout(r, ms));
}

function dispatch(systemPrompt: string, userPrompt: string): Promise<string> {
  switch (provider) {
    case 'groq':      return callGroq(systemPrompt, userPrompt);
    case 'gemini':    return callGemini(systemPrompt, userPrompt);
    case 'anthropic': return callAnthropic(systemPrompt, userPrompt);
    case 'openai':    return callOpenAI(systemPrompt, userPrompt);
    default:
      throw new Error(`Unknown AI_PROVIDER: "${provider}". Use groq | gemini | anthropic | openai`);
  }
}

// ── Groq ──────────────────────────────────────────────────────
async function callGroq(system: string, user: string): Promise<string> {
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
      messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({})) as { error?: { message?: string } };
    throw new Error(`Groq API error (${res.status}): ${err?.error?.message ?? res.statusText}`);
  }

  const data = await res.json() as { choices?: Array<{ message?: { content?: string } }> };
  const text = data?.choices?.[0]?.message?.content;
  if (!text) throw new Error('Groq returned an empty response.');
  return text;
}

// ── Gemini ────────────────────────────────────────────────────
async function callGemini(system: string, user: string): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) throw new Error('GEMINI_API_KEY is not set in .env');

  const model = process.env.GEMINI_MODEL ?? 'gemini-2.0-flash';
  const url = `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`;

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

  const data = await res.json() as { candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }> };
  const text = data?.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) throw new Error('Gemini returned an empty response.');
  return text;
}

// ── Anthropic ─────────────────────────────────────────────────
async function callAnthropic(system: string, user: string): Promise<string> {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) throw new Error('ANTHROPIC_API_KEY is not set in .env');

  const { default: Anthropic } = await import('@anthropic-ai/sdk');
  const client = new Anthropic({ apiKey });
  const model = process.env.ANTHROPIC_MODEL ?? 'claude-sonnet-4-20250514';

  const response = await client.messages.create({
    model,
    max_tokens: 4096,
    system,
    messages: [{ role: 'user', content: user }],
  });

  const block = response.content[0];
  if (block.type !== 'text') throw new Error('Anthropic returned non-text response');
  return block.text;
}

// ── OpenAI ────────────────────────────────────────────────────
async function callOpenAI(system: string, user: string): Promise<string> {
  const apiKey = process.env.OPENAI_API_KEY;
  if (!apiKey) throw new Error('OPENAI_API_KEY is not set in .env');

  const { default: OpenAI } = await import('openai');
  const client = new OpenAI({ apiKey });
  const model = process.env.OPENAI_MODEL ?? 'gpt-4o';

  const response = await client.chat.completions.create({
    model,
    max_tokens: 4096,
    messages: [{ role: 'system', content: system }, { role: 'user', content: user }],
  });

  const text = response.choices[0]?.message?.content;
  if (!text) throw new Error('OpenAI returned an empty response.');
  return text;
}