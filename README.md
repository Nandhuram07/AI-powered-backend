# 🤖 AI Backend Generator v3

AI-powered backend scaffolder that generates production-ready **TypeScript Express.js** backends from simple user input or plain English descriptions.

[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?style=for-the-badge&logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Express.js](https://img.shields.io/badge/Express.js-000000?style=for-the-badge&logo=express&logoColor=white)](https://expressjs.com/)
[![Zod](https://img.shields.io/badge/Zod-3E67B1?style=for-the-badge&logo=zod&logoColor=white)](https://zod.dev/)
[![MySQL](https://img.shields.io/badge/MySQL-4479A1?style=for-the-badge&logo=mysql&logoColor=white)](https://www.mysql.com/)

---

## 🚀 Overview

The **AI Backend Generator** is designed to eliminate the boilerplate of starting new projects. It doesn't just generate "hello world" examples; it builds a full-fledged architecture with services, controllers, models, and robust validation.

### Key Capabilities
- **Single Entity**: Quickly scaffold one entity with fields (e.g., `Job`, `Product`).
- **Multi-Entity Systems**: Define complex systems with multiple entities and relationships (e.g., `School Management`, `E-Commerce`).
- **Chat Mode**: Simply describe your requirements in plain English, and the AI extracts the schema for you.
- **Self-Healing Loop**: The generator includes an "Agentic Loop" that automatically installs dependencies, runs the generated code, detects syntax/runtime errors, and uses AI to fix them.

---

## ✨ Features

- **TypeScript First**: All generated code is strictly typed TypeScript.
- **Security**: Built-in JWT Authentication and Role-Based Access Control (RBAC).
- **Validation**: Automatic Zod schema generation for every field.
- **Database Support**: Choose between **MySQL/MariaDB** and **Microsoft SQL Server (MSSQL)**.
- **Architecture**: Follows a clean **Controller-Service-Model** pattern.
- **Error Handling**: Standardized error responses with custom middleware.
- **Logging**: Request logging middleware included.
- **Auto-Config**: Generates `.env`, `package.json`, `tsconfig.json`, and even a migration script.

---

## 🛠️ Installation & Setup

### 1. Prerequisites
- **Node.js**: v20 or higher recommended.
- **NPM**: v9 or higher.

### 2. Clone and Install
```bash
git clone <repository-url>
cd AI-powered-backend
npm install
```

### 3. Configure AI Provider
Create a `.env` file in the root directory (use `.env.example` as a template). You'll need an API key from one of the supported providers:

| Variable | Description | Hint |
| :--- | :--- | :--- |
| `AI_PROVIDER` | `groq`, `gemini`, `anthropic`, or `openai` | Pick your preferred model host |
| `GROQ_API_KEY` | API Key for Groq | [console.groq.com](https://console.groq.com) |
| `GEMINI_API_KEY` | API Key for Google Gemini | [aistudio.google.com](https://aistudio.google.com) |
| `ANTHROPIC_API_KEY` | API Key for Anthropic Claude | [console.anthropic.com](https://console.anthropic.com) |
| `OPENAI_API_KEY` | API Key for OpenAI | [platform.openai.com](https://platform.openai.com) |
| `OUTPUT_DIR` | (Optional) Default path for generated projects | Default: `./output` |

### 4. Optional Model Selection
You can also specify which model to use for each provider by adding these to `.env`:
- `GROQ_MODEL` (e.g., `llama-3.3-70b-versatile`)
- `GEMINI_MODEL` (e.g., `gemini-2.0-flash`)
- `ANTHROPIC_MODEL` (e.g., `claude-sonnet-4-20250514`)
- `OPENAI_MODEL` (e.g., `gpt-4o`)

**Set your preferred provider in `.env`:**
```env
AI_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

---

## 📖 Usage

Run the generator in development mode:

```bash
npm run dev
```

### Modes of Generation

1.  **Single Entity**: 
    - Input: `student`
    - Fields: `name:string, age:number, enrolled:boolean`
    - Result: Full CRUD for Students.
2.  **Multi-Entity System**:
    - Input: `School` (Entities: `student`, `staff`, `grade`)
    - Define relationships (e.g., `grade` belongs to `student`).
    - Result: A connected system with foreign key support.
3.  **Chat Mode**:
    - Input: *"I need a hospital management system with patients, doctors, and appointments. Patients can book multiple appointments."*
    - Result: The AI parses the entities and fields automatically.

---

## 📂 Generated Project Structure

The resulting backend follows this organized structure:

```text
output/
├── src/
│   ├── config/          # DB connection
│   ├── controllers/     # Route handlers
│   ├── middleware/      # Auth, Error, Logger
│   ├── models/          # Data access layer
│   ├── routes/          # API endpoints
│   ├── services/        # Business logic
│   ├── utils/           # Shared helpers
│   ├── validations/     # Zod schemas
│   ├── migrate.ts       # DB migration script
│   └── server.ts        # Entry point
├── .env                 # Environment config
├── package.json         # Scripts & deps
└── tsconfig.json        # TS config
```

---

## 🔄 The Self-Healing Loop

One of the most powerful features of this generator is the **Self-Healing Loop**. After generating the files:
1.  It automatically runs `npm install` in the generated folder.
2.  It performs a **TypeSript syntax check**.
3.  It attempts to **start the server**.
4.  If a crash occurs (e.g., AI hallucinated a bad import or missed a type), the **Agent** parses the error output, reads the problematic file, and **uses AI to fix the code** in a loop (up to 5 attempts).

This ensures that the output is not just "code" but a **runnable application**.

---

## 📜 Commands

- `npm run dev`: Start the interactive CLI.
- `npm run build`: Compile the generator to `dist/`.
- `npm run start`: Run the compiled generator.
- `npm run typecheck`: Validate the generator's source code.

---

