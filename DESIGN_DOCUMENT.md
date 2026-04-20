# Design Document: Requirements Transformer — React + Multi-Provider LLM

**Version:** 1.1  
**Date:** April 20, 2026  
**Status:** Draft

---

## Table of Contents

1. [Overview](#1-overview)
2. [Goals and Non-Goals](#2-goals-and-non-goals)
3. [System Architecture](#3-system-architecture)
4. [Tech Stack](#4-tech-stack)
5. [Application Features](#5-application-features)
6. [UI/UX Design](#6-uiux-design)
7. [Data Flow](#7-data-flow)
8. [LLM Provider Integration](#8-llm-provider-integration)
9. [Prompt Engineering](#9-prompt-engineering)
10. [Output Template — Internal System Requirements](#10-output-template--internal-system-requirements)
11. [File Structure](#11-file-structure)
12. [Component Design](#12-component-design)
13. [State Management](#13-state-management)
14. [Error Handling](#14-error-handling)
15. [Security Considerations](#15-security-considerations)
16. [Testing Strategy](#16-testing-strategy)
17. [Future Enhancements](#17-future-enhancements)

---

## 1. Overview

**Requirements Transformer** is a browser-based React application that allows users to upload Markdown-based requirements documents and automatically generate structured **Internal System Requirements (ISR)** documents using a configurable LLM backend — supporting **Anthropic Claude**, **Google Gemini**, and **Ollama** (local, free, no API key required).

The application bridges the gap between high-level stakeholder requirements (e.g., Product Requirement Documents, Business Requirement Documents) and actionable internal engineering specifications.

### Problem Statement

Engineering teams frequently receive requirements documents that are unstructured, incomplete, or written for non-technical audiences. Translating these into precise internal system requirements is time-consuming and error-prone. This tool automates that transformation using LLM-assisted analysis.

### Solution Summary

- User uploads one or more `.md` requirement files via a drag-and-drop interface.
- User selects a provider (Claude, Gemini, or Ollama) and optionally provides a custom system prompt.
- The app sends file content + prompt to the selected LLM provider.
- The LLM returns a structured Internal System Requirements document.
- The user can preview, edit, and download the generated `.md` ISR document.

---

## 2. Goals and Non-Goals

### Goals

- Accept `.md` files as input via file upload or paste.
- Allow users to customize the transformation prompt.
- Support multiple LLM providers: **Anthropic Claude**, **Google Gemini**, **Ollama** (local).
- Allow runtime provider + model selection without code changes.
- Produce a consistently formatted ISR `.md` document.
- Allow inline editing of the generated output.
- Enable download of the output as a `.md` file.
- Support multiple input files merged into a single ISR.

### Non-Goals

- No user authentication or multi-tenancy in v1.
- No persistent backend storage of documents.
- No support for non-Markdown input formats (PDF, DOCX) in v1.
- No real-time collaboration features.

---

## 3. System Architecture

```
┌───────────────────────────────────────────────────────────┐
│                        Browser                            │
│                                                           │
│  ┌─────────────────────────────────────────────────────┐  │
│  │              React Application (SPA)                │  │
│  │                                                     │  │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────┐  │  │
│  │  │  File Upload │  │ Prompt Editor│  │  Output  │  │  │
│  │  │  Component   │  │  Component   │  │  Viewer  │  │  │
│  │  └──────┬───────┘  └──────┬───────┘  └────┬─────┘  │  │
│  │         │                 │               ▲         │  │
│  │         └────────┬────────┘               │         │  │
│  │                  ▼                        │         │  │
│  │          ┌───────────────┐                │         │  │
│  │          │  LLM Service  │────────────────┘         │  │
│  │          │ (llmService)  │                          │  │
│  │          └───────┬───────┘                          │  │
│  └──────────────────┼──────────────────────────────────┘  │
│                     │ HTTPS / localhost                   │
└─────────────────────┼───────────────────────────────────┘
                      │
          ┌───────────┼───────────┐
          ▼           ▼           ▼
   ┌─────────────┐ ┌──────────┐ ┌──────────────┐
   │  Anthropic  │ │  Google  │ │    Ollama    │
   │ Claude API  │ │  Gemini  │ │  (localhost) │
   │  (External) │ │   API    │ │   — free —   │
   └─────────────┘ └──────────┘ └──────────────┘
```

> **Note:** API keys for Claude and Gemini are stored in `sessionStorage` only. Ollama requires no key — it runs entirely on `localhost:11434`. For production, a backend proxy is recommended to keep external API keys server-side (see [Security Considerations](#15-security-considerations)).

---

## 4. Tech Stack

| Layer              | Technology                          | Reason                                        |
|--------------------|-------------------------------------|-----------------------------------------------|
| Frontend Framework | React 18 + TypeScript               | Component model, type safety                  |
| Build Tool         | Vite                                | Fast HMR, modern ESM builds                   |
| Styling            | Tailwind CSS                        | Utility-first, rapid layout                   |
| Markdown Rendering | `react-markdown` + `remark-gfm`     | Safe, GFM-flavored Markdown preview           |
| Markdown Editing   | `@uiw/react-md-editor`              | In-browser Markdown editing of output         |
| HTTP Client        | Native `fetch` API                  | No extra dependency needed                    |
| LLM Provider       | **Anthropic Claude** (`claude-3-7-sonnet-20250219`) / **Google Gemini** (`gemini-2.0-flash`) / **Ollama** (`llama3.2`, local) | Multi-provider; Ollama is free & offline |
| State Management   | React Context + `useReducer`        | Lightweight; no Redux needed for v1           |
| File Download      | Native Blob + anchor trick          | No library dependency                         |
| Linting/Formatting | ESLint + Prettier                   | Code quality                                  |

---

## 5. Application Features

### 5.1 Document Upload

- Drag-and-drop zone accepting `.md` files.
- Click-to-browse fallback.
- Support for **multiple files** (merged in upload order).
- Live preview of uploaded file content before processing.
- File list management: add, remove, reorder files.

### 5.2 Prompt Configuration

- Default system prompt pre-filled (see [Section 9](#9-prompt-engineering)).
- Editable prompt textarea for customization.
- Preset dropdown: `Basic ISR`, `Detailed ISR`, `Agile Story Decomposition`, `Security-Focused ISR`.
- "Reset to default" button.

### 5.3 LLM Provider Settings Panel

- **Provider selector:** `Claude` | `Gemini` | `Ollama (local)`.
- API Key input (masked, stored in `sessionStorage` only) — hidden when Ollama is selected.
- Model selector — options populated based on the chosen provider:
  - Claude: `claude-3-7-sonnet-20250219`, `claude-3-5-haiku-20241022`, `claude-opus-4-5`
  - Gemini: `gemini-2.0-flash`, `gemini-2.0-pro`
  - Ollama: free-text or dropdown of locally available models (fetched from `GET /api/tags`)
- Max tokens slider (256–8000).
- Temperature slider (0.0–1.0, default 0.3).

### 5.4 Transform Action

- "Generate ISR" button triggers the API call.
- Loading spinner with status message during processing.
- Word/token count estimate shown before sending.

### 5.5 Output Viewer & Editor

- Side-by-side or tabbed view: **Rendered Preview** | **Raw Markdown**.
- Inline editing of generated Markdown.
- Syntax highlighting for code blocks within the output.
- "Copy to Clipboard" button.
- "Download as .md" button with configurable filename.
- "Regenerate" button to re-run with the same or modified prompt.

### 5.6 Diff View (Optional v1 Enhancement)

- Side-by-side comparison of input requirements vs. generated ISR.

---

## 6. UI/UX Design

### 6.1 Layout — Three-Panel Design

```
┌────────────────────────────────────────────────────────────────┐
│  HEADER: Requirements Transformer  |  Settings ⚙              │
├───────────────────┬────────────────┬───────────────────────────┤
│                   │                │                           │
│  PANEL 1          │  PANEL 2       │  PANEL 3                  │
│  Input Documents  │  Prompt Config │  Generated ISR Output     │
│                   │                │                           │
│  [Drop .md here]  │  System Prompt │  [Rendered Markdown]      │
│                   │  textarea      │                           │
│  File 1.md  ✕     │                │  [Edit] [Copy] [Download] │
│  File 2.md  ✕     │  Presets ▼     │                           │
│                   │                │                           │
│  [Add Files]      │  [Generate ISR]│                           │
│                   │                │                           │
└───────────────────┴────────────────┴───────────────────────────┘
```

### 6.2 Responsive Behavior

- **Desktop (≥1200px):** Three-panel horizontal layout.
- **Tablet (768–1199px):** Two-column layout (Input + Prompt stacked left; Output right).
- **Mobile (<768px):** Single-column stacked layout with tabs.

### 6.3 Theme

- Light mode default with dark mode toggle (via Tailwind `dark:` classes).
- Color palette: neutral grays + Anthropic-inspired accent (`#c96442` / `#d97706`).

---

## 7. Data Flow

```
1. User uploads .md file(s)
        │
        ▼
2. FileReader API reads file content as text
        │
        ▼
3. Files concatenated with separators into a single context string
        │
        ▼
4. User reviews/edits prompt in Prompt Config panel
        │
        ▼
5. User clicks "Generate ISR"
        │
        ▼
6. llmService.transform(inputText, systemPrompt, settings)
        │
        ├── Builds messages array:
        │     [{ role: "user", content: "<prompt>\n\n<file contents>" }]
        │
        ▼
7. Routes to selected provider:
   Claude  → POST https://api.anthropic.com/v1/messages
   Gemini  → POST https://generativelanguage.googleapis.com/v1beta/...
   Ollama  → POST http://localhost:11434/api/chat
        │
        ▼
8. Response normalized by LLMService → .text extracted
        │
        ▼
9. Output state updated → Output Viewer renders Markdown
        │
        ▼
10. User edits / downloads output .md file
```

---

## 8. LLM Provider Integration

All three providers are abstracted behind a single `llmService` module. The UI never calls provider APIs directly.

### 8.1 Provider Endpoints

| Provider | Endpoint                                                          | Auth                  |
|----------|-------------------------------------------------------------------|-----------------------|
| Claude   | `POST https://api.anthropic.com/v1/messages`                     | `x-api-key` header    |
| Gemini   | `POST https://generativelanguage.googleapis.com/v1beta/models/...`| `?key=` query param   |
| Ollama   | `POST http://localhost:11434/api/chat`                            | None (local only)     |

### 8.2 Normalized TypeScript Service Interface

```typescript
// src/services/llmService.ts

export type Provider = "claude" | "gemini" | "ollama";

export interface LLMSettings {
  provider: Provider;
  apiKey?: string;        // not required for Ollama
  model: string;
  maxTokens: number;
  temperature: number;
}

export interface TransformResult {
  content: string;
  inputTokens: number;
  outputTokens: number;
}

export async function transformRequirements(
  inputDocuments: string,
  systemPrompt: string,
  settings: LLMSettings
): Promise<TransformResult> {
  switch (settings.provider) {
    case "claude":  return callClaude(inputDocuments, systemPrompt, settings);
    case "gemini":  return callGemini(inputDocuments, systemPrompt, settings);
    case "ollama":  return callOllama(inputDocuments, systemPrompt, settings);
    default: throw new Error(`Unknown provider: ${settings.provider}`);
  }
}
```

### 8.3 Claude Request Body

```json
{
  "model": "claude-3-7-sonnet-20250219",
  "max_tokens": 4096,
  "temperature": 0.3,
  "system": "<SYSTEM_PROMPT>",
  "messages": [{ "role": "user", "content": "<DOCUMENTS>" }]
}
```

### 8.4 Gemini Request Body

```json
{
  "system_instruction": { "parts": [{ "text": "<SYSTEM_PROMPT>" }] },
  "contents": [{ "role": "user", "parts": [{ "text": "<DOCUMENTS>" }] }],
  "generationConfig": { "maxOutputTokens": 4096, "temperature": 0.3 }
}
```

### 8.5 Ollama Request Body

```json
{
  "model": "llama3.2",
  "stream": false,
  "messages": [
    { "role": "system", "content": "<SYSTEM_PROMPT>" },
    { "role": "user",   "content": "<DOCUMENTS>" }
  ],
  "options": { "temperature": 0.3, "num_predict": 4096 }
}
```

### 8.6 Python Reference Implementation

See [`llm_client_example.py`](llm_client_example.py) for a working Python version of the same multi-provider abstraction using the `LLMClient` class.

---

## 9. Prompt Engineering

### 9.1 Default System Prompt

```
You are a senior software architect and technical writer specializing in 
converting high-level requirement documents into structured Internal System 
Requirements (ISR) documents.

When given one or more requirement documents, you must:
1. Analyze and extract all functional and non-functional requirements.
2. Identify ambiguities and resolve them with reasonable technical assumptions 
   (note assumptions explicitly).
3. Map each requirement to a unique ISR ID using the format ISR-XXX.
4. Categorize requirements: Functional, Non-Functional, Interface, Security, 
   Performance, Constraint.
5. Assign a priority to each requirement: Critical, High, Medium, Low.
6. Produce the output strictly following the ISR Markdown template provided.
7. Do not omit any requirement present in the source document.
8. Do not add requirements not implied by the source document.

Output ONLY the Markdown document. Do not wrap it in code fences. 
Do not add commentary outside the document structure.
```

### 9.2 User Message Template

```
## Source Requirement Documents

The following document(s) have been uploaded for transformation:

---
{FILE_SEPARATOR: "=== Document: {filename} ==="}
{CONTENT_OF_EACH_FILE}
---

Please transform the above into an Internal System Requirements document 
following the ISR template exactly.
```

### 9.3 Preset Prompts

| Preset Name               | Focus                                               |
|---------------------------|-----------------------------------------------------|
| Basic ISR                 | Minimal structured ISR with IDs and categories      |
| Detailed ISR              | Full ISR with acceptance criteria per requirement   |
| Agile Story Decomposition | Breaks requirements into user stories + sub-tasks   |
| Security-Focused ISR      | Emphasizes threat modeling and security requirements|

---

## 10. Output Template — Internal System Requirements

The following template will be strictly followed in the Claude output:

```markdown
# Internal System Requirements Document

**Project Name:** {Derived from source document or "Unnamed Project"}  
**ISR Version:** 1.0  
**Date:** {Today's date}  
**Prepared By:** Requirements Transformer (AI-Assisted)  
**Source Documents:** {List of uploaded filenames}  
**Status:** Draft

---

## 1. Introduction

### 1.1 Purpose
{Brief description of the system being specified}

### 1.2 Scope
{What is in/out of scope based on source requirements}

### 1.3 Definitions and Acronyms

| Term / Acronym | Definition |
|----------------|------------|
| {term}         | {definition} |

### 1.4 Assumptions and Constraints
- {Assumption 1}
- {Constraint 1}

---

## 2. Stakeholders

| Role              | Responsibility                     |
|-------------------|------------------------------------|
| {Stakeholder Role}| {What they need from this system}  |

---

## 3. Functional Requirements

### FR-001 — {Requirement Title}

| Field            | Value                              |
|------------------|------------------------------------|
| **ISR ID**       | ISR-001                            |
| **Source Ref**   | {Section/line in source doc}       |
| **Category**     | Functional                         |
| **Priority**     | Critical / High / Medium / Low     |
| **Description**  | {Full requirement description}     |
| **Rationale**    | {Why this is needed}               |
| **Acceptance Criteria** | {How to verify this is met} |
| **Dependencies** | ISR-XXX, ISR-XXX                   |

---

## 4. Non-Functional Requirements

### NFR-001 — {Requirement Title}

| Field            | Value                              |
|------------------|------------------------------------|
| **ISR ID**       | ISR-0XX                            |
| **Category**     | Performance / Security / Usability / Reliability / Scalability |
| **Priority**     | {Priority}                         |
| **Description**  | {Full requirement description}     |
| **Metric**       | {Measurable target, e.g., "< 200ms response time"} |
| **Acceptance Criteria** | {Verification method}      |

---

## 5. Interface Requirements

### IR-001 — {Interface Name}

| Field            | Value                              |
|------------------|------------------------------------|
| **ISR ID**       | ISR-0XX                            |
| **Type**         | UI / API / Database / External Service |
| **Priority**     | {Priority}                         |
| **Description**  | {Interface description}            |
| **Protocol / Format** | {e.g., REST/JSON, GraphQL}   |

---

## 6. Security Requirements

| ISR ID   | Requirement                          | Priority |
|----------|--------------------------------------|----------|
| ISR-0XX  | {Security requirement description}   | {P}      |

---

## 7. Constraints

| ISR ID   | Constraint                           | Type               |
|----------|--------------------------------------|--------------------|
| ISR-0XX  | {Constraint description}             | Technical / Business / Regulatory |

---

## 8. Requirements Traceability Matrix

| ISR ID   | Source Requirement Ref | Category        | Priority | Status  |
|----------|------------------------|-----------------|----------|---------|
| ISR-001  | {Source Ref}           | Functional      | High     | Draft   |

---

## 9. Open Issues and TBDs

| # | Issue Description               | Owner   | Due Date  |
|---|---------------------------------|---------|-----------|
| 1 | {Ambiguity or missing detail}   | TBD     | TBD       |

---

## 10. Revision History

| Version | Date       | Author                        | Changes       |
|---------|------------|-------------------------------|---------------|
| 1.0     | {Date}     | Requirements Transformer (AI) | Initial Draft |
```

---

## 11. File Structure

```
requirements-transformer/
├── public/
│   └── favicon.svg
├── src/
│   ├── assets/
│   │   └── logo.svg
│   ├── components/
│   │   ├── layout/
│   │   │   ├── Header.tsx
│   │   │   └── ThreeColumnLayout.tsx
│   │   ├── upload/
│   │   │   ├── FileDropZone.tsx
│   │   │   ├── FileList.tsx
│   │   │   └── FilePreview.tsx
│   │   ├── prompt/
│   │   │   ├── PromptEditor.tsx
│   │   │   ├── PresetSelector.tsx
│   │   │   └── ApiSettingsPanel.tsx
│   │   ├── output/
│   │   │   ├── OutputViewer.tsx
│   │   │   ├── MarkdownEditor.tsx
│   │   │   └── DownloadButton.tsx
│   │   └── shared/
│   │       ├── LoadingSpinner.tsx
│   │       ├── ErrorBanner.tsx
│   │       └── TokenCounter.tsx
│   ├── context/
│   │   └── AppContext.tsx
│   ├── hooks/
│   │   ├── useFileUpload.ts
│   │   ├── useLLMTransform.ts
│   │   └── useMarkdownDownload.ts
│   ├── services/
│   │   └── llmService.ts
│   ├── prompts/
│   │   ├── systemPrompt.ts
│   │   └── presets.ts
│   ├── types/
│   │   └── index.ts
│   ├── utils/
│   │   ├── fileReader.ts
│   │   └── markdownFormatter.ts
│   ├── App.tsx
│   ├── main.tsx
│   └── index.css
├── .env.example
├── .eslintrc.cjs
├── .prettierrc
├── index.html
├── package.json
├── tailwind.config.ts
├── tsconfig.json
└── vite.config.ts
```

---

## 12. Component Design

### 12.1 `AppContext` — Global State

```typescript
interface AppState {
  // Uploaded files
  uploadedFiles: UploadedFile[];

  // Prompt configuration
  systemPrompt: string;
  selectedPreset: PresetKey;

  // LLM provider settings
  provider: "claude" | "gemini" | "ollama";
  apiKey: string;          // empty string when provider is "ollama"
  model: string;
  maxTokens: number;
  temperature: number;

  // Processing state
  isLoading: boolean;
  error: string | null;

  // Output
  generatedISR: string | null;
  tokenUsage: { input: number; output: number } | null;
}

interface UploadedFile {
  id: string;
  name: string;
  content: string;
  size: number;
}
```

### 12.2 `FileDropZone` Component

- Listens for `dragover`, `dragleave`, `drop` events.
- Validates file type (`.md` only) and size (< 1MB per file).
- Reads content via `FileReader.readAsText()`.
- Dispatches `ADD_FILES` action to context.

### 12.3 `PromptEditor` Component

- Controlled `<textarea>` bound to `systemPrompt` state.
- `PresetSelector` dropdown updates `systemPrompt` on selection.
- Character and estimated token count displayed below.

### 12.4 `useLLMTransform` Hook

```typescript
function useLLMTransform() {
  const { state, dispatch } = useAppContext();

  async function transform() {
    dispatch({ type: "SET_LOADING", payload: true });
    dispatch({ type: "CLEAR_ERROR" });

    try {
      const inputDocuments = buildInputString(state.uploadedFiles);
      const result = await transformRequirements(
        inputDocuments,
        state.systemPrompt,
        {
          provider: state.provider,
          apiKey: state.apiKey,
          model: state.model,
          maxTokens: state.maxTokens,
          temperature: state.temperature,
        }
      );
      dispatch({ type: "SET_OUTPUT", payload: result });
    } catch (err) {
      dispatch({ type: "SET_ERROR", payload: (err as Error).message });
    } finally {
      dispatch({ type: "SET_LOADING", payload: false });
    }
  }

  return { transform };
}
```

### 12.5 `OutputViewer` Component

- Tabs: **Preview** (rendered via `react-markdown`) | **Source** (editable via `@uiw/react-md-editor`).
- Changes to the Source tab update `generatedISR` in state.
- Token usage footer shows `↑ {inputTokens} / ↓ {outputTokens}`.

### 12.6 `DownloadButton` Component

```typescript
function downloadMarkdown(content: string, filename: string) {
  const blob = new Blob([content], { type: "text/markdown;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
```

---

## 13. State Management

State transitions via `useReducer` in `AppContext`:

| Action Type       | Payload                   | Effect                                  |
|-------------------|---------------------------|-----------------------------------------|
| `ADD_FILES`       | `UploadedFile[]`          | Merges new files into `uploadedFiles`   |
| `REMOVE_FILE`     | `string` (file id)        | Removes file from list                  |
| `REORDER_FILES`   | `string[]` (ordered ids)  | Reorders file list                      |
| `SET_PROMPT`      | `string`                  | Updates `systemPrompt`                  |
| `SET_PRESET`      | `PresetKey`               | Updates prompt from preset map          |
| `SET_PROVIDER`    | `Provider`                | Switches LLM provider; resets model     |
| `SET_API_KEY`     | `string`                  | Updates API key                         |
| `SET_MODEL`       | `string`                  | Updates selected model for active provider |
| `SET_LOADING`     | `boolean`                 | Toggles loading state                   |
| `SET_OUTPUT`      | `TransformResult`         | Stores generated ISR + token usage      |
| `SET_ERROR`       | `string`                  | Sets error message                      |
| `CLEAR_ERROR`     | —                         | Clears error message                    |
| `UPDATE_ISR`      | `string`                  | Updates inline-edited ISR content       |

---

## 14. Error Handling

| Scenario                             | Handling                                                    |
|--------------------------------------|-------------------------------------------------------------|
| No files uploaded                    | Disable "Generate ISR" button; show tooltip                 |
| Invalid file type uploaded           | Display inline warning on drop; reject file                 |
| File exceeds size limit              | Display inline warning; reject file                         |
| API key missing                      | Inline validation in settings modal; block submission       |
| LLM API 401 Unauthorized             | Error banner: "Invalid API key. Check your settings."       |
| LLM API 429 Rate Limit               | Error banner with retry suggestion and backoff hint         |
| LLM API 500 / network error          | Error banner: "Service unavailable. Please try again."      |
| Ollama not running                   | Error banner: "Cannot reach Ollama at localhost:11434. Is it running?" |
| Output empty or malformed            | Warning banner: "Output may be incomplete. Try regenerating."|
| Token limit exceeded                 | Pre-check estimated tokens; warn before sending             |

---

## 15. Security Considerations

### 15.1 API Key Handling

- API key is **never** stored in `localStorage` (persists after session).
- API key is stored in `sessionStorage` only — cleared on tab close.
- API key field masked with `type="password"`.
- For production deployments, a **backend proxy** (e.g., Express or Next.js API route) should relay requests to Claude/Gemini, keeping the API key server-side only.
- When the provider is **Ollama**, no API key is required; requests go to `localhost` and never leave the machine.

### 15.2 File Content Security

- Uploaded files are read client-side only; no server upload occurs in v1.
- File content is sanitized before rendering in Markdown preview (via `react-markdown`'s XSS-safe rendering — no `dangerouslySetInnerHTML`).
- Maximum file size enforced (1MB per file, 5MB total) to prevent memory exhaustion.

### 15.3 Content Security Policy

Recommended CSP headers for production deployment:

```
Content-Security-Policy:
  default-src 'self';
  connect-src 'self'
              https://api.anthropic.com
              https://generativelanguage.googleapis.com
              http://localhost:11434;
  script-src 'self';
  style-src 'self' 'unsafe-inline';
```

### 15.4 Input Validation

- Only `.md` MIME type / extension accepted.
- Maximum 10 files per batch.
- Prompt length capped at 10,000 characters in the UI.

---

## 16. Testing Strategy

### 16.1 Unit Tests (`Vitest`)

| Test Target             | Test Cases                                                        |
|-------------------------|-------------------------------------------------------------------|
| `llmService.ts`         | Correct request built per provider; error thrown on 4xx/5xx      |
| `llmService.ts`         | Routes to correct provider based on `settings.provider`          |
| `fileReader.ts`         | Reads `.md` content; rejects non-md files                        |
| `AppContext` reducer     | Each action produces correct state transition (incl. SET_PROVIDER)|
| `markdownFormatter.ts`  | Concatenation of multiple files with separators                  |

### 16.2 Component Tests (`React Testing Library`)

| Component          | Test Cases                                                |
|--------------------|-----------------------------------------------------------|
| `FileDropZone`     | Accepts `.md` drop; rejects `.pdf` drop; shows file list  |
| `PromptEditor`     | Edits update state; preset changes update prompt          |
| `OutputViewer`     | Renders markdown; switching tabs works                    |
| `DownloadButton`   | Creates blob URL and triggers download                    |
| `ErrorBanner`      | Displays on error state; dismissable                      |

### 16.3 Integration Tests

- Mock all three provider APIs with `msw` (Mock Service Worker).
- Full flow per provider: upload file → select provider → edit prompt → generate → download.
- Provider switch mid-session: verify model list and API key field update correctly.
- Error scenarios: 401 (Claude/Gemini), 429 (Claude/Gemini), 500, Ollama unreachable (network error on localhost).

### 16.4 End-to-End Tests (`Playwright`)

- Happy path: upload `sample-requirements.md` → generate → verify output headings.
- Download verification: output file has `.md` extension and non-empty content.

---

## 17. Future Enhancements

| Priority | Feature                                               |
|----------|-------------------------------------------------------|
| High     | Backend proxy (Next.js) to secure Claude/Gemini API keys server-side |
| High     | Support for PDF / DOCX input via document parsing                    |
| Medium   | Streaming response support (SSE/chunked) for all three providers      |
| Medium   | History panel: save/restore previous generations      |
| Medium   | Version diff: compare v1 ISR vs. re-generated v2      |
| Low      | Export to DOCX / PDF via Puppeteer or print CSS       |
| Low      | Multi-user collaboration via shared session link      |
| Low      | GitHub integration: open PRs with generated ISR       |
| Low      | Custom ISR template editor (user-defined sections)    |

---

## Appendix A — Environment Variables

```env
# .env.example

# Default provider on startup: "claude" | "gemini" | "ollama"
VITE_DEFAULT_PROVIDER=ollama

VITE_CLAUDE_DEFAULT_MODEL=claude-3-7-sonnet-20250219
VITE_GEMINI_DEFAULT_MODEL=gemini-2.0-flash
VITE_OLLAMA_DEFAULT_MODEL=llama3.2
VITE_OLLAMA_BASE_URL=http://localhost:11434

VITE_MAX_FILE_SIZE_MB=1
VITE_MAX_FILES=10
# Do NOT add real API keys here — enter them in the app UI at runtime
```

---

## Appendix B — Key Dependencies

```json
{
  "dependencies": {
    "react": "^18.3.0",
    "react-dom": "^18.3.0",
    "react-markdown": "^9.0.0",
    "remark-gfm": "^4.0.0",
    "@uiw/react-md-editor": "^4.0.0"
  },
  "devDependencies": {
    "@types/react": "^18.3.0",
    "@vitejs/plugin-react": "^4.3.0",
    "autoprefixer": "^10.4.0",
    "msw": "^2.0.0",
    "postcss": "^8.4.0",
    "tailwindcss": "^3.4.0",
    "typescript": "^5.5.0",
    "vite": "^5.4.0",
    "vitest": "^2.0.0",
    "@playwright/test": "^1.46.0",
    "@testing-library/react": "^16.0.0"
  }
}
```

---

## Appendix C — Python Reference Script

[`llm_client_example.py`](llm_client_example.py) is a standalone Python script that implements the same multi-provider abstraction as the React app. It is useful for:

- Testing API connectivity before building the UI.
- Running ISR generation in CI/CD pipelines (no browser required).
- Validating prompt outputs against all three providers.

Switch providers by changing the single `PROVIDER` constant at the top of the file:

```python
PROVIDER = "ollama"   # "claude" | "gemini" | "ollama"
```

Install dependencies:

```bash
pip install anthropic google-genai ollama python-dotenv
```

---

---

## Revision History

| Version | Date             | Changes                                                                 |
|---------|------------------|-------------------------------------------------------------------------|
| 1.0     | April 17, 2026   | Initial draft — Claude API only                                         |
| 1.1     | April 20, 2026   | Added multi-provider support (Claude, Gemini, Ollama); renamed service layer to `llmService`; added Appendix C for Python reference script |

*Document end. Next step: scaffold the project with `npm create vite@latest requirements-transformer -- --template react-ts`.*
