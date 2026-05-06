# AI Clone — Design Spec
**Date:** 2026-05-06
**Stack:** Python OOP, pytest, FastAPI, ChromaDB, OpenRouter, Whisper, ElevenLabs

---

## Overview

A personal AI clone that learns your personality, writing style, and knowledge from your chat history and documents. It starts with a baseline built from existing chat exports, then gets sharper over time through two learning mechanisms: RAG (immediate recall from stored conversations) and periodic fine-tuning (deeper style absorption into model weights).

**Interfaces:** CLI (`ai-clone` command) + Web App (FastAPI + vanilla JS)
**Architecture:** Layered monolith — one Python package with clean submodules

---

## Architecture

```
ai_clone/
├── ingestion/          # Chat parsers (WhatsApp, Telegram, iMessage, Slack)
│   └── parsers/        # One class per platform, common ParsedMessage output
├── personality/        # Profile builder + fine-tune scheduler
│   ├── builder.py      # Extracts style/tone/traits from parsed messages
│   └── trainer.py      # Schedules fine-tuning jobs via Together AI
├── memory/             # Semantic memory (ChromaDB vector store)
│   ├── store.py        # Embed + upsert conversations
│   └── retriever.py    # Similarity search for relevant context
├── agent/              # Conversational engine
│   ├── clone.py        # Main AIClone class — assembles context + calls OpenRouter
│   └── context.py      # Builds prompt from personality + memory + history
├── voice/              # Speech I/O
│   ├── listener.py     # Whisper STT
│   └── speaker.py      # ElevenLabs TTS with voice cloning
├── api/                # FastAPI web backend
│   └── routes/         # /chat, /voice, /ingest, /profile
└── cli/                # Click-based CLI (shares core with API)
```

### Data Flow

```
Chat exports → Ingestion → Personality Builder → ChromaDB
                                                      ↓
User input → Agent (context = personality + memory retrieval) → OpenRouter → Response
                                                                                  ↓
                                                              ElevenLabs TTS (voice mode)
```

### State
- **ChromaDB** — vector store for embeddings (persisted to `./data/chroma`)
- **SQLite** — conversation history + profile metadata (persisted to `./data/memory.db`)
- **`profile.json`** — extracted personality profile (saved to `./data/profile.json`)
- **`config.yml`** — all API keys and settings (project root)

---

## Component Design

### 1. Ingestion & Parsers

All parsers output a common `ParsedMessage(sender, text, timestamp, platform)` dataclass.

| Parser | Input Format |
|---|---|
| `WhatsAppParser` | `.txt` export |
| `TelegramParser` | `.json` export |
| `iMessageParser` | `.db` (SQLite) or `.txt` backup |
| `SlackParser` | `.json` zip export |

`IngestionPipeline` orchestrates: detect format → select parser → parse → filter user's messages → pass to `PersonalityBuilder` and `ConversationStore`.

---

### 2. Personality Profile Builder

Runs once on initial import, then updates incrementally after each new conversation.

**`PersonalityBuilder`** filters only your messages and extracts:
- **Writing style** — avg sentence length, punctuation patterns, emoji usage, formality score
- **Vocabulary fingerprint** — top N-grams, filler words, signature phrases
- **Tone traits** — humor, directness, empathy (lightweight classification via OpenRouter)
- **Topic interests** — top topics you discuss (clustered via embeddings)

Output saved as `./data/profile.json` and embedded into ChromaDB as the clone's identity context.

**`ProfileUpdater`** — after each new conversation, appends to the store and recalculates style metrics on a rolling window (last 500 messages).

**`FineTuneTrainer`** — batches your messages into `(prompt, response)` pairs and submits to **Together AI** on a weekly schedule. The fine-tuned model ID is saved to `config.yml` and used by the agent automatically. Falls back to OpenRouter if no fine-tuned model exists yet.

---

### 3. Memory Layer

**`ConversationStore`** — embeds every message (user + clone) via a local sentence-transformer and upserts into ChromaDB with metadata: `timestamp`, `session_id`, `speaker`.

**`MemoryRetriever`** — at query time, embeds the current user message and returns top-K semantically similar past exchanges for context injection.

**Two memory tiers:**
- **Short-term** — last N messages in the current session (in-memory deque)
- **Long-term** — ChromaDB vector store, searched on every turn

**Memory decay** — time-weighted relevance score:
`final_score = similarity_score * recency_weight`
Recent context naturally dominates older memories.

**`MemoryConsolidator`** — runs nightly, summarizes old conversation clusters into compact memory summaries via OpenRouter to keep ChromaDB lean as history grows.

---

### 4. Conversational Agent

**`AIClone`** — main entry point, orchestrates all components per turn:
```
user_input
    → MemoryRetriever (top-K relevant past exchanges)
    → ContextBuilder (assembles system prompt)
    → OpenRouter API call (streaming)
    → response stored in ConversationStore
    → returned to CLI / API
```

**`ContextBuilder`** — builds the system prompt in 4 layers:
1. **Identity layer** — personality traits, writing style, vocabulary fingerprint from `profile.json`
2. **Memory layer** — top-K retrieved past exchanges
3. **Session layer** — current conversation short-term memory (deque)
4. **Instruction layer** — "You are Vishwa. Respond exactly as he would. Match his tone, humor, and phrasing."

**Model routing:**
1. Check `config.yml` for a fine-tuned model ID → use via Together AI
2. Fallback: OpenRouter (default model: `tencent/hy3-preview:free`)

**`ConversationSession`** — tracks active session state (messages deque, session ID, start time). Fresh per CLI run or per API connection.

**Streaming** — responses stream token-by-token via SSE (web) or stdout (CLI).

---

### 5. Voice Interface

**`VoiceListener`** (faster-whisper, local):
- Records from mic until silence detected via `webrtcvad` (VAD)
- Transcribes audio → text → passed to `AIClone.chat()`
- Runs on CPU, ~1–2s latency for short utterances

**`VoiceSpeaker`** (ElevenLabs API):
- `VoiceCloner.setup()` — one-time setup: upload 1-min voice sample → creates ElevenLabs voice ID → saved to `config.yml`
- Every response synthesized in your cloned voice, played via `sounddevice`
- Falls back to system TTS if ElevenLabs API is unavailable

**`VoiceSession`** — continuous loop:
```
listen → transcribe → AIClone.chat() → synthesize → play → listen...
```

**`config.yml` voice settings:**
```yaml
elevenlabs_voice_id: ""
whisper_model: "base"       # tiny/base/small: speed vs accuracy
vad_aggressiveness: 2       # 0-3
```

---

### 6. CLI + Web App

**CLI** (Click, entry point `ai-clone`):
```
ai-clone chat                # text chat session
ai-clone chat --voice        # voice session
ai-clone ingest ./exports    # parse chat exports
ai-clone profile build       # rebuild personality profile
ai-clone profile show        # print profile summary
ai-clone memory search "q"   # search memory store
```

**Web App** (FastAPI + vanilla HTML/CSS/JS, no build step):

| Route | Purpose |
|---|---|
| `GET /` | Single-page chat UI |
| `POST /chat` | Text message, SSE streaming response |
| `POST /voice` | Audio blob in, synthesized audio out |
| `POST /ingest` | Upload export files, trigger parsers |
| `GET /profile` | Return profile JSON for sidebar display |
| `WS /ws/chat` | Real-time streaming chat |

Frontend: chat bubble UI, mic button for voice, profile sidebar showing personality traits.

---

### 7. Configuration

`config.yml` at project root:
```yaml
openrouter_api_key: ""
elevenlabs_api_key: ""
together_api_key: ""         # for fine-tuning
db_path: "./data/memory.db"
chroma_path: "./data/chroma"
profile_path: "./data/profile.json"
fine_tuned_model_id: ""      # populated after first fine-tuning run
```

---

## Testing Strategy

- **Unit tests** (pytest) — each parser, builder, retriever, and context builder tested in isolation with fixtures
- **Integration tests** — full ingest → profile → chat round-trip with a small sample dataset
- **Voice tests** — mock Whisper + ElevenLabs APIs; test session loop logic independently
- **No external API calls in tests** — all providers mocked via `pytest-mock`

---

## Build Order (Sub-projects)

Each sub-project gets its own implementation plan:

1. **Personality Profile Builder** — parsers + profile extraction + `profile.json`
2. **Memory Layer** — ChromaDB store + retriever + consolidator
3. **Conversational Agent** — `AIClone` + `ContextBuilder` + OpenRouter integration
4. **Voice Interface** — Whisper listener + ElevenLabs speaker + voice session
5. **CLI + Web App** — Click CLI + FastAPI routes + frontend
6. **Fine-tuning Pipeline** — Together AI integration + weekly scheduler

---

## Dependencies

```
fastapi uvicorn click chromadb sentence-transformers
faster-whisper webrtcvad sounddevice elevenlabs
openai httpx pyyaml pytest pytest-mock
```
